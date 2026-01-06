import os
import re
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class DiffusionSequenceDataset(Dataset):
    '''
    扩散模型专用序列数据集
    - 支持单侧或双侧数据加载
    - 合并传感器数据和力矩数据
    - 返回运动类型的类别标签
    '''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: Union[str, List[str]],
                 diffusion_sequence_length: int,
                 action_patterns: List[str],
                 participant_masses: Dict[str, float] = {},
                 device: torch.device = torch.device("cpu"),
                 mode: str = "train",
                 file_suffix: Dict[str, str] = None,
                 remove_nan: bool = True,
                 enable_action_filter: bool = False,
                 activity_flag: bool = False,
                 min_sequence_length: int = -1):
        """
        初始化扩散模型序列数据集

        参数:
            data_dir: 数据根目录路径
            input_names: 输入特征列名列表(传感器数据)
            label_names: 力矩特征列名列表
            side: 身体侧别 ('l', 'r' 或 ['l', 'r'])
            diffusion_sequence_length: 序列长度（子序列的时间步数）
            action_patterns: 运动类型筛选的正则表达式列表
            participant_masses: 参与者体重字典
            device: 计算设备
            mode: 数据集模式，'train' 或 'test'
            file_suffix: 文件后缀映射字典
            remove_nan: 是否自动检测并移除包含NaN的行
            activity_flag: 是否启用activity_flag掩码功能(默认False)
            min_sequence_length: 最小序列长度,-1表示不限制(默认-1)
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side if isinstance(side, list) else [side]
        self.diffusion_sequence_length = diffusion_sequence_length
        self.action_patterns = action_patterns
        self.participant_masses = participant_masses
        self.device = device
        self.mode = mode.lower()
        self.remove_nan = remove_nan
        self.activity_flag = activity_flag
        self.min_sequence_length = min_sequence_length

        # 设置文件后缀映射
        if file_suffix is None:
            self.file_suffix = {
                "input": "_exo.csv",
                "label": "_moment_filt.csv",
                "flag": "_activity_flag.csv"
            }
        else:
            self.file_suffix = file_suffix

        # 获取试验名称列表
        self.trial_names = self._get_trial_names()

        # 统计信息
        self.nan_removal_stats = {
            'trials_with_all_nan_data': 0  # 数据全为NaN的试验数
        }

        ## 测试,正式训练时改行需要注释
        self.trial_names = self.trial_names[:10]

        # 序列长度过滤统计信息
        self.length_filter_stats = {
            'trials_before_filter': 0,
            'trials_after_filter': 0,
            'trials_filtered_out': 0,
            'min_length_before': 0,
            'max_length_before': 0,
            'min_length_after': 0,
            'max_length_after': 0
        }

        print(f"开始加载 {self.mode} 数据集 (用于扩散模型训练)...")
        print(f"找到 {len(self.trial_names)} 个试验")
        print(f"使用侧别: {', '.join(self.side)}")
        print(f"运动类别数: {len(self.action_patterns)}")

        # 预加载所有数据到内存
        self.all_data = []  # 存储所有试验的合并数据(传感器+力矩)
        self.all_labels = []  # 存储所有试验的类别标签
        self.trial_lengths = []  # 存储每个试验的原始长度
        self._preload_all_data()

        # 根据min_sequence_length过滤序列
        if self.min_sequence_length > 0:
            self._filter_by_sequence_length()

        # 检测并移除数据全为NaN的序列
        self._remove_invalid_sequences()

        # 生成序列索引
        print(f"生成序列索引...")
        self.sequences = self._generate_sequences()

        print(f"数据集初始化完成 - 模式: {self.mode}, "
              f"试验数量: {len(self.trial_names)}, "
              f"序列数量: {len(self.sequences)}")

        # 打印序列长度过滤统计
        if self.min_sequence_length > 0:
            self.print_length_filter_summary()

        if self.remove_nan and self.nan_removal_stats['trials_with_all_nan_data'] > 0:
            self.print_nan_removal_summary()

    def __len__(self):
        '''返回数据集中序列的总数'''
        return len(self.sequences)

    def __getitem__(self, idx: int):
        '''
        获取单个序列样本

        返回: (data_seq, class_label)
            - data_seq: [num_features, sequence_length] 包含传感器数据和力矩数据
            - class_label: 标量，运动类型的类别索引
        '''
        # 获取序列索引信息
        trial_idx, start_idx = self.sequences[idx]

        # 提取数据序列
        # 尺寸为[num_features, sequence_length]
        data_seq = self.all_data[trial_idx][:, start_idx:start_idx + self.diffusion_sequence_length]

        # 获取类别标签
        class_label = self.all_labels[trial_idx]

        # 转换为tensor
        data_seq = torch.from_numpy(data_seq).float()
        class_label = torch.tensor(class_label, dtype=torch.long)

        return data_seq, class_label

    def _get_class_label(self, trial_name: str) -> int:
        '''根据试验名称获取类别标签'''
        # 从trial_name中提取action_type
        # 格式: participant/action_type
        parts = trial_name.split(os.sep)
        if len(parts) >= 2:
            action_type = parts[1]  # action_type部分

            # 遍历每个类别（每个类别可能包含多个patterns）
            for class_idx, patterns in enumerate(self.action_patterns):
                # 如果patterns是字符串，转换为列表
                if isinstance(patterns, str):
                    patterns = [patterns]

                # 检查action_type是否匹配该类别的任何pattern
                for pattern in patterns:
                    if re.match(pattern, action_type):
                        return class_idx

        # 如果没有匹配到任何pattern，返回-1表示未知类别
        print(f"警告: 试验 {trial_name} 无法匹配任何运动类型pattern")
        return -1

    def _is_action_matched(self, action_type: str) -> bool:
        '''检查动作类型是否匹配任何筛选模式'''
        for patterns in self.action_patterns:
            # 如果patterns是字符串，转换为列表
            patterns = [patterns] if isinstance(patterns, str) else patterns

            # 检查是否匹配该类别的任何pattern
            if any(re.match(pattern, action_type) for pattern in patterns):
                return True

        return False

    def _get_trial_names(self) -> List[str]:
        '''获取所有试验的名称列表'''
        # data/train
        mode_dir = os.path.join(self.data_dir, self.mode)

        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"模式目录不存在: {mode_dir}")

        trial_names = []

        for participant in os.listdir(mode_dir):
            # data/train/BT01
            participant_dir = os.path.join(mode_dir, participant)

            if os.path.isdir(participant_dir):

                for action_type in os.listdir(participant_dir):
                    # data/train/BT01/walk
                    action_dir = os.path.join(participant_dir, action_type)

                    if not os.path.isdir(action_dir):
                        continue

                    # 检查动作类型是否匹配筛选模式
                    if not self._is_action_matched(action_type):
                        continue

                    # 构建试验的相对路径, 例如: BT01/walk
                    trial_name = os.path.join(participant, action_type)
                    trial_names.append(trial_name)

        return sorted(trial_names)

    def _preload_all_data(self):
        '''预加载所有试验的数据到内存'''
        for trial_name in self.trial_names:
            # 加载数据
            data, class_label = self._load_trial_data(trial_name)

            # 存储数据
            self.all_data.append(data.numpy())
            self.all_labels.append(class_label)
            self.trial_lengths.append(data.shape[1])

    def _load_trial_data(self, trial_name: str) -> Tuple[torch.Tensor, int]:
        '''
        加载单个试验的数据

        返回:
            data: [num_features, time_steps] 合并的数据(传感器+力矩)
            class_label: 类别标签
        '''
        # data/train
        mode_dir = os.path.join(self.data_dir, self.mode)
        # data/train/BT01/walk
        trial_dir = os.path.join(mode_dir, trial_name)

        # 获取类别标签
        class_label = self._get_class_label(trial_name)

        # 为每个侧别加载数据并合并
        all_side_data = []

        for s in self.side:
            # 替换特征名中的通配符
            # ["foot_imu_r_gyro_x", "foot_imu_r_gyro_y",...]
            input_cols = [name.replace("*", s) for name in self.input_names]
            # ["hip_flexion_r_moment", "knee_angle_r_moment"]
            label_cols = [name.replace("*", s) for name in self.label_names]

            # 构建文件路径
            participant = trial_name.split(os.sep)[0]
            action_type = trial_name.split(os.sep)[1]
            base_filename = f"{participant}_{action_type}"

            # data/train/BT01/walk/BT01_walk_exo.csv
            input_file = os.path.join(trial_dir, base_filename + self.file_suffix["input"])
            # data/train/BT01/walk/BT01_walk_moment_filt.csv
            label_file = os.path.join(trial_dir, base_filename + self.file_suffix["label"])

            # 加载传感器数据,尺寸为[C,N]
            input_data = self._load_input_data(input_file, input_cols)

            # 加载力矩数据,尺寸为[2,N]
            label_data = self._load_label_data(label_file, label_cols)

            # 确保长度一致
            assert input_data.shape[1] == label_data.shape[1]

            # 合并传感器数据和力矩数据,尺寸为[C+2,N]
            side_data = torch.cat([input_data, label_data], dim=0)
            all_side_data.append(side_data)

        # 如果有多个侧别，沿特征维度拼接
        if len(all_side_data) > 1:
            data = torch.cat(all_side_data, dim=0)
        else:
            data = all_side_data[0]

        return data, class_label

    def _load_input_data(self, file_path: str, column_names: List[str]) -> torch.Tensor:
        '''加载输入数据（传感器数据）'''
        df = pd.read_csv(file_path)

        # 检查是否有缺失的必需列
        missing_cols = [col for col in column_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺失必需的列: {missing_cols}")

        # 提取数据并转换为tensor
        extracted_data = df[column_names].values
        input_data = torch.tensor(extracted_data, dtype=torch.float32).transpose(0, 1)

        return input_data

    def _load_label_data(self, file_path: str, column_names: List[str]) -> torch.Tensor:
        '''加载标签数据（力矩数据）'''
        df = pd.read_csv(file_path)

        # 检查是否有缺失的必需列
        missing_cols = [col for col in column_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"标签数据缺失必需的列: {missing_cols}")

        # 提取数据并转换为tensor
        extracted_data = df[column_names].values
        label_data = torch.tensor(extracted_data, dtype=torch.float32).transpose(0, 1)

        return label_data

    def _generate_sequences(self) -> List[Tuple[int, int]]:
        '''
        生成所有可用的序列索引
        返回: [(trial_idx, start_idx), ...]
        '''
        sequences = []

        for trial_idx in range(len(self.trial_names)):
            # 获取该试验的数据长度
            data_len = self.all_data[trial_idx].shape[1]

            # 检查数据长度是否足够
            if data_len < self.diffusion_sequence_length:
                print(f"警告: 试验 {self.trial_names[trial_idx]} 数据长度不足 "
                      f"(需要{self.diffusion_sequence_length}, 实际{data_len})，跳过")
                continue

            # 生成所有有效的起始索引
            max_start_idx = data_len - self.diffusion_sequence_length

            for start_idx in range(max_start_idx + 1):
                sequences.append((trial_idx, start_idx))

        return sequences

    def _filter_by_sequence_length(self):
        """根据最小序列长度过滤试验"""
        self.length_filter_stats['trials_before_filter'] = len(self.trial_names)

        if len(self.trial_lengths) > 0:
            self.length_filter_stats['min_length_before'] = min(self.trial_lengths)
            self.length_filter_stats['max_length_before'] = max(self.trial_lengths)

        # 找出需要保留的试验索引
        valid_indices = []
        for i, length in enumerate(self.trial_lengths):
            if length >= self.min_sequence_length:
                valid_indices.append(i)

        # 过滤数据
        self.all_data = [self.all_data[i] for i in valid_indices]
        self.all_labels = [self.all_labels[i] for i in valid_indices]
        self.trial_lengths = [self.trial_lengths[i] for i in valid_indices]
        self.trial_names = [self.trial_names[i] for i in valid_indices]

        self.length_filter_stats['trials_after_filter'] = len(self.trial_names)
        self.length_filter_stats['trials_filtered_out'] = (
                self.length_filter_stats['trials_before_filter'] -
                self.length_filter_stats['trials_after_filter']
        )

        if len(self.trial_lengths) > 0:
            self.length_filter_stats['min_length_after'] = min(self.trial_lengths)
            self.length_filter_stats['max_length_after'] = max(self.trial_lengths)

    def _remove_invalid_sequences(self):
        """检测并移除数据全为NaN的序列"""
        if not self.remove_nan:
            return

        valid_indices = []
        for i in range(len(self.all_data)):
            data = self.all_data[i]
            # 检查是否全为NaN
            if not np.all(np.isnan(data)):
                valid_indices.append(i)
            else:
                print(f"移除数据全为NaN的试验: {self.trial_names[i]}")
                self.nan_removal_stats['trials_with_all_nan_data'] += 1

        # 过滤数据
        self.all_data = [self.all_data[i] for i in valid_indices]
        self.all_labels = [self.all_labels[i] for i in valid_indices]
        self.trial_lengths = [self.trial_lengths[i] for i in valid_indices]
        self.trial_names = [self.trial_names[i] for i in valid_indices]

    def print_length_filter_summary(self):
        """打印序列长度过滤统计摘要"""
        stats = self.length_filter_stats
        print(f"\n{'=' * 60}")
        print(f"序列长度过滤统计摘要 - {self.mode.upper()} 数据集")
        print(f"{'=' * 60}")
        print(f"最小序列长度阈值: {self.min_sequence_length}")
        print(f"过滤前试验数量: {stats['trials_before_filter']}")
        print(f"过滤后试验数量: {stats['trials_after_filter']}")
        print(f"被过滤掉的试验数: {stats['trials_filtered_out']}")

        if stats['trials_before_filter'] > 0:
            filter_percentage = 100 * stats['trials_filtered_out'] / stats['trials_before_filter']
            print(f"过滤比例: {filter_percentage:.2f}%")

        if stats['min_length_before'] > 0:
            print(f"过滤前序列长度范围: [{stats['min_length_before']}, {stats['max_length_before']}]")

        if stats['trials_after_filter'] > 0 and stats['min_length_after'] > 0:
            print(f"过滤后序列长度范围: [{stats['min_length_after']}, {stats['max_length_after']}]")

        print(f"{'=' * 60}\n")

    def print_nan_removal_summary(self):
        """打印NaN移除统计摘要"""
        stats = self.nan_removal_stats
        print(f"\n{'=' * 60}")
        print(f"NaN移除统计摘要 - {self.mode.upper()} 数据集")
        print(f"{'=' * 60}")
        print(f"数据全为NaN的试验数: {stats['trials_with_all_nan_data']}")
        print(f"{'=' * 60}\n")

    def get_num_classes(self) -> int:
        """返回类别数量"""
        return len(self.action_patterns)


def main():
    import importlib
    import sys
    sys.path.insert(0, '.')
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description="快速测试 DiffusionSequenceDataset 数据加载流程"
    )
    parser.add_argument("--config", type=str, default="diffusion_config",
                        help="配置文件模块名")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="选择加载训练集或测试集")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备，如 cpu 或 cuda:0")

    args = parser.parse_args()

    def load_config(config_path: str):
        '''Load config file as module.'''
        config_path = config_path.replace("/", ".").replace("\\", ".")
        if config_path.endswith(".py"):
            config_path = config_path[:-3]
        print(f"Loading config file from {config_path}.")
        return importlib.import_module(config_path)

    # 导入配置
    config = importlib.import_module(args.config)

    device = torch.device(args.device)

    # 创建数据集
    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=config.participant_masses,
        device=device,
        mode=args.mode,
        remove_nan=True,
        activity_flag=config.activity_flag,
        min_sequence_length=getattr(config, 'min_sequence_length', -1)
    )

    dataset_size = len(dataset)
    num_classes = dataset.get_num_classes()

    print("\n" + "=" * 70)
    print("扩散模型数据集测试概览")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"试验数量: {len(dataset.trial_names)}")
    print(f"序列数量: {dataset_size}")
    print(f"类别数量: {num_classes}")
    print(f"序列长度: {config.diffusion_sequence_length}")

    # 计算总特征数
    total_features = 0
    for s in (config.side if isinstance(config.side, list) else [config.side]):
        total_features += len(config.input_names) + len(config.label_names)
    print(f"总特征数: {total_features}")

    if dataset.trial_lengths:
        print(f"序列长度统计 (帧) -> 平均 {np.mean(dataset.trial_lengths):.1f}, "
              f"中位 {np.median(dataset.trial_lengths):.1f}, "
              f"范围 [{np.min(dataset.trial_lengths)}, {np.max(dataset.trial_lengths)}]")
    print("=" * 70 + "\n")

    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("示例批次：")
    for batch_idx, (data, labels) in enumerate(data_loader):
        print(f"\n批次 {batch_idx}:")
        print(f"  数据形状: {data.shape}")  # [B, num_features, sequence_length]
        print(f"  标签形状: {labels.shape}")  # [B]
        print(f"  标签值: {labels.tolist()}")
        breakpoint()

        if batch_idx >= 2:
            break


if __name__ == "__main__":
    main()