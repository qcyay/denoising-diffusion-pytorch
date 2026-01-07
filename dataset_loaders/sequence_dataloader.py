import os
import re
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class DiffusionSequenceDataset(Dataset):
    '''
    扩散模型专用序列数据集
    - 支持单侧或双侧数据加载
    - 合并传感器数据和力矩数据
    - 支持activity_flag掩码
    - 支持参与者体重特征
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
                 mode: Union[str, List[str]] = "train",
                 file_suffix: Dict[str, str] = None,
                 remove_nan: bool = False,
                 remove_any_nan: bool = True,
                 activity_flag: bool = False,
                 use_participant_mass: bool = False,
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
            mode: 数据集模式，'train'、'test' 或 ['train', 'test']
            file_suffix: 文件后缀映射字典
            remove_nan: 是否移除数据全为NaN的序列
            remove_any_nan: 是否移除包含任何NaN的序列
            activity_flag: 是否启用activity_flag掩码功能(默认False)
            use_participant_mass: 是否使用参与者体重作为特征(默认False)
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
        self.mode = [mode] if isinstance(mode, str) else mode
        self.remove_nan = remove_nan
        self.remove_any_nan = remove_any_nan
        self.activity_flag = activity_flag
        self.use_participant_mass = use_participant_mass
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

        # 维护特征拼接列表
        self.feature_sources = []

        # 获取试验名称列表
        self.trial_names = self._get_trial_names()

        # 统计信息
        self.nan_removal_stats = {
            'trials_with_all_nan_data': 0,  # 数据全为NaN的试验数
            'trials_with_any_nan_data': 0  # 包含任何NaN的试验数
        }

        # 测试,正式训练时改行需要注释
        self.trial_names = self.trial_names[:100]

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

        print(f"开始加载 {'/'.join(self.mode)} 数据集 (用于扩散模型训练)...")
        print(f"找到 {len(self.trial_names)} 个试验")
        print(f"使用侧别: {', '.join(self.side)}")
        print(f"运动类别数: {len(self.action_patterns)}")

        # 预加载所有数据到内存
        self.all_data = []  # 存储所有试验的合并数据(传感器+力矩+其他特征)
        self.all_labels = []  # 存储所有试验的类别标签
        self.trial_lengths = []  # 存储每个试验的原始长度
        self._preload_all_data()

        # 根据min_sequence_length过滤序列
        if self.min_sequence_length > 0:
            self._filter_by_sequence_length()

        # 检测并移除数据全为NaN的序列
        if self.remove_nan:
            self._remove_all_nan_sequences()

        # 检测并移除包含任何NaN的序列
        if self.remove_any_nan:
            self._remove_sequences_with_nan()

        # 生成序列索引
        print(f"生成序列索引...")
        self.sequences = self._generate_sequences()

        print(f"数据集初始化完成 - 模式: {'/'.join(self.mode)}, "
              f"试验数量: {len(self.trial_names)}, "
              f"序列数量: {len(self.sequences)}")

        # 打印特征拼接信息
        self._print_feature_sources()

        # 打印序列长度过滤统计
        if self.min_sequence_length > 0:
            self.print_length_filter_summary()

        # 打印NaN移除统计
        if self.remove_nan or self.remove_any_nan:
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
        # 格式: participant/action_type 或 mode/participant/action_type
        parts = trial_name.split(os.sep)

        # 找到action_type (最后一个部分)
        action_type = parts[-1]

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
        trial_names = []

        # 遍历所有模式（train和/或test）
        for mode in self.mode:
            mode_dir = os.path.join(self.data_dir, mode)

            if not os.path.exists(mode_dir):
                print(f"警告: 模式目录不存在: {mode_dir}")
                continue

            for participant in os.listdir(mode_dir):
                participant_dir = os.path.join(mode_dir, participant)

                if not os.path.isdir(participant_dir):
                    continue

                for action_type in os.listdir(participant_dir):
                    action_dir = os.path.join(participant_dir, action_type)

                    if not os.path.isdir(action_dir):
                        continue

                    # 检查动作类型是否匹配筛选模式
                    if not self._is_action_matched(action_type):
                        continue

                    # 构建试验的相对路径, 例如: train/BT01/walk 或 BT01/walk
                    # 包含mode以区分来自不同数据集的同名试验
                    if len(self.mode) > 1:
                        trial_name = os.path.join(mode, participant, action_type)
                    else:
                        trial_name = os.path.join(participant, action_type)
                    trial_names.append(trial_name)

        return sorted(trial_names)

    def _preload_all_data(self):
        '''预加载所有试验的数据到内存'''
        for trial_idx, trial_name in enumerate(self.trial_names):
            if (trial_idx + 1) % 10 == 0 or (trial_idx + 1) == len(self.trial_names):
                print(f"  加载进度: {trial_idx + 1}/{len(self.trial_names)}")

            # 解析trial_name
            parts = trial_name.split(os.sep)
            if len(self.mode) > 1:
                # 格式: mode/participant/action_type
                mode, participant, action_type = parts[0], parts[1], parts[2]
            else:
                # 格式: participant/action_type
                mode = self.mode[0]
                participant, action_type = parts[0], parts[1]

            # 构建文件路径前缀
            file_prefix = os.path.join(self.data_dir, mode, participant,
                                       action_type, f"{participant}_{action_type}")

            # 输入文件路径
            input_file = file_prefix + self.file_suffix["input"]
            label_file = file_prefix + self.file_suffix["label"]

            # 初始化数据列表（用于拼接多个侧别的数据）
            all_sides_data = []

            # 遍历每个侧别
            for s in self.side:
                # 替换列名中的 * 为具体的侧别
                input_cols = [name.replace("*", s) for name in self.input_names]
                label_cols = [name.replace("*", s) for name in self.label_names]

                # 加载输入数据
                input_data = self._load_input_data(input_file, input_cols)

                # 加载标签数据（力矩数据）
                label_data = self._load_label_data(label_file, label_cols)

                # 合并输入和标签数据
                # 形状: [num_features, seq_len]
                merged_data = torch.cat([input_data, label_data], dim=0)

                # 加载activity_flag数据（如果启用）
                if self.activity_flag:
                    flag_file = file_prefix + self.file_suffix["flag"]
                    flag_data = self._load_activity_flag_data(flag_file, s)
                    merged_data = torch.cat([merged_data, flag_data], dim=0)

                all_sides_data.append(merged_data)

            # 如果有多个侧别，沿特征维度拼接
            if len(all_sides_data) > 1:
                combined_data = torch.cat(all_sides_data, dim=0)
            else:
                combined_data = all_sides_data[0]

            # 添加参与者体重特征（如果启用）
            # 体重对于某个固定序列是一定的，因此只添加一次
            if self.use_participant_mass:
                mass_data = self._create_mass_feature(participant, combined_data.shape[1])
                combined_data = torch.cat([combined_data, mass_data], dim=0)

            # 转换为numpy并存储
            self.all_data.append(combined_data.numpy())

            # 存储类别标签
            class_label = self._get_class_label(trial_name)
            self.all_labels.append(class_label)

            # 存储序列长度
            self.trial_lengths.append(combined_data.shape[1])

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

    def _load_activity_flag_data(self, file_path: str, side: str) -> torch.Tensor:
        '''
        加载activity_flag数据

        参数:
            file_path: activity_flag文件路径
            side: 'l' 或 'r'

        返回:
            形状为 [1, seq_len] 的tensor
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Activity flag文件不存在: {file_path}")

        df = pd.read_csv(file_path)

        # 根据side选择对应的列
        column_name = 'left' if side == 'l' else 'right'

        if column_name not in df.columns:
            raise ValueError(f"Activity flag文件缺失必需的列: {column_name}")

        # 提取数据并转换为tensor
        # 形状: [1, seq_len]
        flag_data = torch.tensor(df[column_name].values, dtype=torch.float32).unsqueeze(0)

        return flag_data

    def _create_mass_feature(self, participant: str, seq_len: int) -> torch.Tensor:
        '''
        创建参与者体重特征

        参数:
            participant: 参与者ID
            seq_len: 序列长度

        返回:
            形状为 [1, seq_len] 的tensor，所有时间步的值都是该参与者的体重
        '''
        if participant not in self.participant_masses:
            raise ValueError(f"参与者 {participant} 的体重信息未找到")

        mass = self.participant_masses[participant]
        # 创建一个重复的体重特征
        mass_feature = torch.full((1, seq_len), mass, dtype=torch.float32)

        return mass_feature

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

    def _remove_all_nan_sequences(self):
        """检测并移除数据全为NaN的序列"""
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

    def _remove_sequences_with_nan(self):
        """检测并移除包含任何NaN的序列"""
        valid_indices = []
        for i in range(len(self.all_data)):
            data = self.all_data[i]
            # 检查是否包含NaN
            if not np.any(np.isnan(data)):
                valid_indices.append(i)
            else:
                print(f"移除包含NaN的试验: {self.trial_names[i]}")
                self.nan_removal_stats['trials_with_any_nan_data'] += 1

        # 过滤数据
        self.all_data = [self.all_data[i] for i in valid_indices]
        self.all_labels = [self.all_labels[i] for i in valid_indices]
        self.trial_lengths = [self.trial_lengths[i] for i in valid_indices]
        self.trial_names = [self.trial_names[i] for i in valid_indices]

    def _print_feature_sources(self):
        """打印特征拼接信息"""
        # 构建特征源列表
        self.feature_sources = []

        for s in self.side:
            # 输入特征
            self.feature_sources.append(f"传感器数据 (side={s}): {len(self.input_names)} features")

            # 力矩特征
            self.feature_sources.append(f"力矩数据 (side={s}): {len(self.label_names)} features")

            # Activity flag
            if self.activity_flag:
                self.feature_sources.append(f"Activity flag (side={s}): 1 feature")

        # 参与者体重（只添加一次，不依赖于side）
        if self.use_participant_mass:
            self.feature_sources.append(f"参与者体重: 1 feature")

        # 打印信息
        print(f"\n{'=' * 60}")
        print(f"特征拼接信息")
        print(f"{'=' * 60}")
        for i, source in enumerate(self.feature_sources, 1):
            print(f"{i}. {source}")

        # 计算总特征数
        total_features = 0
        for s in self.side:
            total_features += len(self.input_names) + len(self.label_names)
            if self.activity_flag:
                total_features += 1
        # 参与者体重只计算一次
        if self.use_participant_mass:
            total_features += 1

        print(f"\n总特征数: {total_features}")
        print(f"{'=' * 60}\n")

    def print_length_filter_summary(self):
        """打印序列长度过滤统计摘要"""
        stats = self.length_filter_stats
        print(f"\n{'=' * 60}")
        print(f"序列长度过滤统计摘要 - {'/'.join(self.mode).upper()} 数据集")
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
        print(f"NaN移除统计摘要 - {'/'.join(self.mode).upper()} 数据集")
        print(f"{'=' * 60}")
        if self.remove_nan:
            print(f"数据全为NaN的试验数: {stats['trials_with_all_nan_data']}")
        if self.remove_any_nan:
            print(f"包含NaN的试验数: {stats['trials_with_any_nan_data']}")
        print(f"{'=' * 60}\n")

    def get_num_classes(self) -> int:
        """返回类别数量"""
        return len(self.action_patterns)

    def _get_feature_names(self) -> List[str]:
        """
        获取所有特征的名称列表（与数据维度对应）

        返回:
            特征名称列表，顺序与数据的维度顺序一致
        """
        feature_names = []

        for s in self.side:
            # 输入特征名称
            for name in self.input_names:
                feature_names.append(name.replace("*", s))

            # 力矩特征名称
            for name in self.label_names:
                feature_names.append(name.replace("*", s))

        # 参与者体重（只添加一次，不依赖于side）
        if self.use_participant_mass:
            feature_names.append("participant_mass")

        return feature_names

    def compute_and_save_statistics(self, output_dir: str = "statistics"):
        """
        计算并保存每个运动类别的特征统计信息（最大值和最小值）

        参数:
            output_dir: 统计信息保存目录

        保存格式:
        {
            "class_0": {
                "feature_name_1": {"max": xxx, "min": xxx},
                "feature_name_2": {"max": xxx, "min": xxx},
                ...
            },
            ...
            "participant_mass": {"max": xxx, "min": xxx}
        }
        """
        import json

        print(f"\n{'=' * 60}")
        print(f"开始计算特征统计信息...")
        print(f"{'=' * 60}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取特征名称列表
        feature_names = self._get_feature_names()
        num_features = len(feature_names)

        # 初始化统计字典：每个类别存储每个特征的最大值和最小值
        # {class_idx: {feature_name: {"max": [], "min": []}}}
        class_stats = {}
        for class_idx in range(len(self.action_patterns)):
            class_stats[class_idx] = {}
            for feat_name in feature_names:
                class_stats[class_idx][feat_name] = {
                    "max": float('-inf'),
                    "min": float('inf')
                }

        # 参与者体重统计（如果启用）
        mass_stats = None
        if self.use_participant_mass:
            mass_stats = {
                "max": float('-inf'),
                "min": float('inf')
            }

        # 遍历所有试验，按类别收集统计信息
        print(f"正在处理 {len(self.all_data)} 个试验...")

        for trial_idx in tqdm(range(len(self.all_data)),
                              desc="计算特征统计",
                              unit="trial",
                              ncols=80):
            data = self.all_data[trial_idx]  # shape: [num_features, seq_len]
            class_label = self.all_labels[trial_idx]

            if class_label == -1:
                # 跳过未知类别
                continue

            # 对每个特征维度计算统计信息
            for feat_idx in range(num_features):
                feat_data = data[feat_idx, :]  # 该特征的所有时间步数据
                feat_name = feature_names[feat_idx]

                # 更新该类别该特征的最大值和最小值
                feat_max = np.max(feat_data)
                feat_min = np.min(feat_data)

                breakpoint()

                class_stats[class_label][feat_name]["max"] = max(
                    class_stats[class_label][feat_name]["max"],
                    feat_max
                )
                class_stats[class_label][feat_name]["min"] = min(
                    class_stats[class_label][feat_name]["min"],
                    feat_min
                )

                # 如果是体重特征，也更新全局体重统计
                if self.use_participant_mass and "participant_mass" in feat_name:
                    mass_stats["max"] = max(mass_stats["max"], feat_max)
                    mass_stats["min"] = min(mass_stats["min"], feat_min)

        # 构建最终的统计字典
        final_stats = {}

        # 添加每个类别的统计信息
        for class_idx in range(len(self.action_patterns)):
            class_key = f"class_{class_idx}"

            # 只保存有数据的类别
            has_data = any(
                class_stats[class_idx][feat_name]["max"] != float('-inf')
                for feat_name in feature_names
            )

            if has_data:
                final_stats[class_key] = {}
                for feat_name in feature_names:
                    # 只保存有有效值的特征
                    if class_stats[class_idx][feat_name]["max"] != float('-inf'):
                        final_stats[class_key][feat_name] = {
                            "max": float(class_stats[class_idx][feat_name]["max"]),
                            "min": float(class_stats[class_idx][feat_name]["min"])
                        }

        # 添加参与者体重的全局统计（如果启用）
        if self.use_participant_mass and mass_stats["max"] != float('-inf'):
            final_stats["participant_mass"] = {
                "max": float(mass_stats["max"]),
                "min": float(mass_stats["min"])
            }

        # 保存到文件
        output_file = os.path.join(output_dir, "feature_statistics.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)

        print(f"统计信息已保存到: {output_file}")
        print(f"包含 {len([k for k in final_stats.keys() if k.startswith('class_')])} 个类别的统计信息")
        print(f"每个类别包含 {num_features} 个特征")

        # 打印示例统计信息
        if len(final_stats) > 0:
            print(f"\n示例统计信息 (class_0 的前3个特征):")
            if "class_0" in final_stats:
                count = 0
                for feat_name, stats in final_stats["class_0"].items():
                    print(f"  {feat_name}: max={stats['max']:.4f}, min={stats['min']:.4f}")
                    count += 1
                    if count >= 3:
                        break

        print(f"{'=' * 60}\n")

        return final_stats


def main():
    import importlib
    import sys
    sys.path.insert(0, '.')
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description="快速测试 DiffusionSequenceDataset 数据加载流程"
    )
    parser.add_argument("--config", type=str, default="default_config",
                        help="配置文件模块名")
    parser.add_argument("--mode", type=str, default="train",
                        help="选择加载的模式，如 'train', 'test', 或 'train,test'")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备，如 cpu 或 cuda:0")
    parser.add_argument("--compute_stats", action="store_true",
                        help="是否计算并保存特征统计信息")
    parser.add_argument("--stats_dir", type=str, default="statistics",
                        help="统计信息保存目录")

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

    # 解析mode参数
    mode = args.mode.split(',') if ',' in args.mode else args.mode

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
        mode=mode,
        remove_nan=True,
        remove_any_nan=True,
        activity_flag=config.activity_flag,
        use_participant_mass=getattr(config, 'use_participant_mass', False),
        min_sequence_length=getattr(config, 'min_sequence_length', -1)
    )

    dataset_size = len(dataset)
    num_classes = dataset.get_num_classes()

    print("\n" + "=" * 70)
    print("扩散模型数据集测试概览")
    print("=" * 70)
    print(f"模式: {mode if isinstance(mode, str) else '/'.join(mode)}")
    print(f"试验数量: {len(dataset.trial_names)}")
    print(f"序列数量: {dataset_size}")
    print(f"类别数量: {num_classes}")
    print(f"序列长度: {config.diffusion_sequence_length}")

    if dataset.trial_lengths:
        print(f"序列长度统计 (帧) -> 平均 {np.mean(dataset.trial_lengths):.1f}, "
              f"中位 {np.median(dataset.trial_lengths):.1f}, "
              f"范围 [{np.min(dataset.trial_lengths)}, {np.max(dataset.trial_lengths)}]")
    print("=" * 70 + "\n")

    # 计算并保存统计信息（如果指定）
    if args.compute_stats:
        dataset.compute_and_save_statistics(output_dir=args.stats_dir)

    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("示例批次：")
    for batch_idx, (data, labels) in enumerate(data_loader):
        print(f"\n批次 {batch_idx}:")
        print(f"  数据形状: {data.shape}")  # [B, num_features, sequence_length]
        print(f"  标签形状: {labels.shape}")  # [B]
        print(f"  标签值: {labels.tolist()}")
        print(f"  数据统计: min={data.min():.4f}, max={data.max():.4f}, "
              f"mean={data.mean():.4f}, std={data.std():.4f}")
        breakpoint()

        if batch_idx >= 2:
            break


if __name__ == "__main__":
    main()