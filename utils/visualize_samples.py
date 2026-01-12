"""
扩散模型采样结果可视化工具

功能：
1. 读取指定milestone的采样结果
2. 从配置文件读取特征名称
3. 列出每个类别的所有样本序号（可选）
4. 可视化指定类别、特征、样本的序列
5. 支持批量可视化和随机选择
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import argparse
import importlib
import torch
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import json

rcParams['font.sans-serif'] = ['SimHei']   # 黑体（Windows 基本都有）
rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

class SampleVisualizer:
    """扩散模型采样结果可视化器"""

    def __init__(self,
                 results_folder: str,
                 milestone: int,
                 config_name: Optional[str] = None):
        """
        初始化可视化器

        Args:
            results_folder: 结果文件夹路径
            milestone: 要加载的milestone编号
            config_name: 配置文件模块名（例如：'default_config'）
                        如果提供，将从配置文件读取特征名称
                        如果不提供，将使用默认的特征编号
        """
        self.results_folder = Path(results_folder)
        self.milestone = milestone
        self.sample_file = self.results_folder / f'sample-{milestone}.pt'

        # 检查文件是否存在
        if not self.sample_file.exists():
            raise FileNotFoundError(f"采样文件不存在: {self.sample_file}")

        # 加载采样数据
        print(f"正在加载采样文件: {self.sample_file}")
        self.data = torch.load(self.sample_file, map_location='cpu')

        # 提取样本和标签
        self.samples = self.data['samples']  # [N, C, seq_length]
        self.labels = self.data['labels']    # [N]

        # 获取数据维度
        self.num_samples, self.num_features, self.seq_length = self.samples.shape
        self.num_classes = int(self.labels.max().item()) + 1

        print(f"✓ 数据加载完成")
        print(f"  - 样本数量: {self.num_samples}")
        print(f"  - 特征维度: {self.num_features}")
        print(f"  - 序列长度: {self.seq_length}")
        print(f"  - 类别数量: {self.num_classes}")

        # 加载配置文件并生成特征名称
        self.config = None
        self.feature_names = None
        if config_name is not None:
            self._load_config_and_generate_feature_names(config_name)
        else:
            print(f"\n⚠ 未提供配置文件，将使用默认的特征编号")
            self.feature_names = [f"特征_{i}" for i in range(self.num_features)]

        # 创建输出目录
        self.output_dir = self.results_folder / f'visualizations_milestone_{milestone}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config_and_generate_feature_names(self, config_name: str):
        """
        加载配置文件并生成特征名称

        Args:
            config_name: 配置文件模块名
        """
        print(f"\n正在加载配置文件: {config_name}")

        try:
            # 加载配置模块
            if '/' in config_name or '\\' in config_name or config_name.endswith('.py'):
                # 文件路径形式：configs/default_config.py
                filepath = config_name.replace('\\', '/').rstrip('.py') + '.py'
                spec = importlib.util.spec_from_file_location("config", filepath)
                self.config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.config)
            else:
                # 模块名形式：configs.default_config
                self.config = importlib.import_module(config_name)
            print(f"✓ 配置文件加载成功")

            # 从配置文件提取必要参数
            input_names = self.config.input_names
            label_names = self.config.label_names
            side = self.config.side
            activity_flag = getattr(self.config, 'activity_flag', False)
            use_participant_mass = getattr(self.config, 'use_participant_mass', False)

            # 转换side为列表格式
            if isinstance(side, str):
                side = [side]

            # 生成特征名称（与DiffusionSequenceDataset._get_feature_names()逻辑一致）
            feature_names = []

            for s in side:
                # 输入特征名称
                for name in input_names:
                    feature_names.append(name.replace("*", s))

                # 力矩特征名称
                for name in label_names:
                    feature_names.append(name.replace("*", s))

                # activity_flag
                if activity_flag:
                    feature_names.append(f"activity_flag_{s}")

            # 参与者体重（只添加一次，不依赖于side）
            if use_participant_mass:
                feature_names.append("participant_mass")

            self.feature_names = feature_names

            # 验证特征名称数量是否与数据维度一致
            if len(self.feature_names) != self.num_features:
                print(f"\n⚠ 警告：特征名称数量 ({len(self.feature_names)}) "
                      f"与数据维度 ({self.num_features}) 不一致！")
                print(f"配置信息:")
                print(f"  - input_names: {len(input_names)} × {len(side)} sides = {len(input_names) * len(side)}")
                print(f"  - label_names: {len(label_names)} × {len(side)} sides = {len(label_names) * len(side)}")
                print(f"  - activity_flag: {activity_flag} × {len(side)} sides = {len(side) if activity_flag else 0}")
                print(f"  - participant_mass: {1 if use_participant_mass else 0}")
                print(f"  - 总计: {len(self.feature_names)}")
                print(f"\n将使用默认的特征编号")
                self.feature_names = [f"特征_{i}" for i in range(self.num_features)]
            else:
                print(f"✓ 特征名称生成成功，共 {len(self.feature_names)} 个特征")
                print(f"  - 前5个特征: {self.feature_names[:5]}")
                if len(self.feature_names) > 5:
                    print(f"  - 后5个特征: {self.feature_names[-5:]}")

        except Exception as e:
            print(f"✗ 加载配置文件失败: {e}")
            print(f"将使用默认的特征编号")
            self.feature_names = [f"特征_{i}" for i in range(self.num_features)]

    def get_feature_name(self, feature_id: int) -> str:
        """
        获取特征名称

        Args:
            feature_id: 特征ID

        Returns:
            特征名称字符串
        """
        if self.feature_names is not None and 0 <= feature_id < len(self.feature_names):
            return f"{self.feature_names[feature_id]} (通道{feature_id})"
        else:
            return f"特征_{feature_id}"

    def list_samples_by_class(self, save_to_file: bool = True) -> Dict[int, List[int]]:
        """
        列出每个类别对应的所有样本序号

        Args:
            save_to_file: 是否保存到txt文件

        Returns:
            字典，键为类别ID，值为样本序号列表
        """
        print("\n" + "=" * 70)
        print("统计每个类别的样本分布...")
        print("=" * 70)

        # 创建类别索引目录
        class_indices = {}
        for class_id in range(self.num_classes):
            # 找到所有属于该类别的样本索引
            indices = torch.where(self.labels == class_id)[0].tolist()
            class_indices[class_id] = indices

            print(f"类别 {class_id:2d}: {len(indices):3d} 个样本")

        if save_to_file:
            # 保存到文件
            index_dir = self.output_dir / 'class_indices'
            index_dir.mkdir(parents=True, exist_ok=True)

            # 保存所有类别的索引到一个indices.txt文件
            indices_file = index_dir / 'indices.txt'
            with open(indices_file, 'w', encoding='utf-8') as f:
                f.write(f"采样结果类别索引 (Milestone {self.milestone})\n")
                f.write("=" * 70 + "\n\n")

                for class_id, indices in class_indices.items():
                    f.write(f"类别 {class_id:02d} (共 {len(indices)} 个样本):\n")
                    f.write("-" * 70 + "\n")

                    # 将索引按每行10个进行格式化输出
                    for i in range(0, len(indices), 10):
                        batch = indices[i:i+10]
                        f.write("  " + ", ".join(f"{idx:4d}" for idx in batch) + "\n")

                    f.write("\n")

            # 保存汇总信息
            summary_file = index_dir / 'summary.txt'
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"采样结果统计 (Milestone {self.milestone})\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"总样本数: {self.num_samples}\n")
                f.write(f"特征维度: {self.num_features}\n")
                f.write(f"序列长度: {self.seq_length}\n")
                f.write(f"类别数量: {self.num_classes}\n\n")
                f.write("各类别样本分布:\n")
                f.write("-" * 70 + "\n")
                for class_id in range(self.num_classes):
                    count = len(class_indices[class_id])
                    percentage = count / self.num_samples * 100
                    f.write(f"类别 {class_id:2d}: {count:4d} 个样本 ({percentage:5.2f}%)\n")

            print(f"\n✓ 所有类别索引已保存到: {indices_file}")
            print(f"✓ 汇总信息已保存到: {summary_file}")

        return class_indices

    def visualize_sequence(self,
                          class_id: int,
                          feature_ids: List[int],
                          sample_idx: Optional[int] = None,
                          show_plot: bool = False,
                          save_plot: bool = True,
                          figsize: Tuple[int, int] = (15, 10),
                          dpi: int = 100) -> None:
        """
        可视化指定类别、特征的序列

        Args:
            class_id: 类别ID
            feature_ids: 要可视化的特征ID列表
            sample_idx: 样本在所有样本中的全局索引（如果为None，则从该类别中随机选择）
            show_plot: 是否显示图形
            save_plot: 是否保存图形
            figsize: 图形大小
            dpi: 图形分辨率
        """
        # 验证类别ID
        if class_id < 0 or class_id >= self.num_classes:
            raise ValueError(f"无效的类别ID: {class_id}，有效范围: [0, {self.num_classes-1}]")

        # 获取该类别的所有样本索引
        class_mask = self.labels == class_id
        class_sample_indices = torch.where(class_mask)[0].tolist()

        if len(class_sample_indices) == 0:
            print(f"警告: 类别 {class_id} 没有样本")
            return

        # 确定要可视化的样本索引
        if sample_idx is None:
            # 从该类别中随机选择一个样本
            sample_idx = np.random.choice(class_sample_indices)
            print(f"随机选择类别 {class_id} 的样本索引: {sample_idx}")
        else:
            # 验证指定的样本索引
            if sample_idx < 0 or sample_idx >= self.num_samples:
                raise ValueError(f"无效的样本索引: {sample_idx}，有效范围: [0, {self.num_samples-1}]")
            if sample_idx not in class_sample_indices:
                raise ValueError(f"样本 {sample_idx} 不属于类别 {class_id}")

        # 验证特征ID
        for feat_id in feature_ids:
            if feat_id < 0 or feat_id >= self.num_features:
                raise ValueError(f"无效的特征ID: {feat_id}，有效范围: [0, {self.num_features-1}]")

        # 提取数据
        sample_data = self.samples[sample_idx]  # [C, seq_length]

        # 创建图形
        num_features = len(feature_ids)
        fig, axes = plt.subplots(num_features, 1, figsize=figsize, dpi=dpi)

        # 如果只有一个特征，将axes转换为列表
        if num_features == 1:
            axes = [axes]

        # 时间轴
        time_steps = np.arange(self.seq_length)

        # 绘制每个特征
        for idx, (ax, feat_id) in enumerate(zip(axes, feature_ids)):
            # 获取该特征的数据
            feature_data = sample_data[feat_id].numpy()

            # 绘制曲线
            ax.plot(time_steps, feature_data, linewidth=1.5, color='steelblue', alpha=0.8)
            ax.fill_between(time_steps, feature_data, alpha=0.3, color='steelblue')

            # 设置标题和标签 - 使用特征名称
            feature_name = self.get_feature_name(feat_id)
            ax.set_title(feature_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=10)
            ax.set_ylabel('数值', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')

            # 添加统计信息
            mean_val = feature_data.mean()
            std_val = feature_data.std()
            min_val = feature_data.min()
            max_val = feature_data.max()

            stats_text = f'均值: {mean_val:.3f} | 标准差: {std_val:.3f} | 范围: [{min_val:.3f}, {max_val:.3f}]'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 设置整体标题
        fig.suptitle(f'样本可视化 - Milestone {self.milestone} | 类别 {class_id} | 样本索引 {sample_idx}',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()

        # 保存图形
        if save_plot:
            # 创建保存目录
            plot_dir = self.output_dir / f'class_{class_id:02d}'
            plot_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            feature_str = '_'.join([f'{fid:02d}' for fid in feature_ids])
            plot_file = plot_dir / f'sample_{sample_idx:04d}_features_{feature_str}.png'

            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            print(f"✓ 图形已保存到: {plot_file}")

        # 显示图形
        if show_plot:
            plt.show()
        else:
            plt.close()

    def visualize_multiple_samples(self,
                                   class_id: int,
                                   feature_id: int,
                                   num_samples: int = 5,
                                   sample_indices: Optional[List[int]] = None,
                                   show_plot: bool = False,
                                   save_plot: bool = True,
                                   figsize: Tuple[int, int] = (15, 10),
                                   dpi: int = 100) -> None:
        """
        在同一张图上可视化多个样本的同一特征

        Args:
            class_id: 类别ID
            feature_id: 特征ID
            num_samples: 要可视化的样本数量（如果sample_indices为None）
            sample_indices: 指定的样本索引列表（如果为None，则随机选择）
            show_plot: 是否显示图形
            save_plot: 是否保存图形
            figsize: 图形大小
            dpi: 图形分辨率
        """
        # 验证类别ID和特征ID
        if class_id < 0 or class_id >= self.num_classes:
            raise ValueError(f"无效的类别ID: {class_id}")
        if feature_id < 0 or feature_id >= self.num_features:
            raise ValueError(f"无效的特征ID: {feature_id}")

        # 获取该类别的所有样本索引
        class_mask = self.labels == class_id
        class_sample_indices = torch.where(class_mask)[0].tolist()

        if len(class_sample_indices) == 0:
            print(f"警告: 类别 {class_id} 没有样本")
            return

        # 确定要可视化的样本索引
        if sample_indices is None:
            # 随机选择样本
            num_available = len(class_sample_indices)
            num_to_select = min(num_samples, num_available)
            sample_indices = np.random.choice(class_sample_indices, num_to_select, replace=False).tolist()

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # 时间轴
        time_steps = np.arange(self.seq_length)

        # 使用不同颜色绘制每个样本
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_indices)))

        for idx, sample_idx in enumerate(sample_indices):
            # 提取数据
            feature_data = self.samples[sample_idx, feature_id].numpy()

            # 绘制曲线
            ax.plot(time_steps, feature_data, linewidth=1.5,
                   color=colors[idx], alpha=0.7, label=f'样本 {sample_idx}')

        # 设置标题和标签 - 使用特征名称
        feature_name = self.get_feature_name(feature_id)
        ax.set_title(f'多样本对比 - 类别 {class_id} | {feature_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.8)

        plt.tight_layout()

        # 保存图形
        if save_plot:
            plot_dir = self.output_dir / f'class_{class_id:02d}'
            plot_dir.mkdir(parents=True, exist_ok=True)

            sample_str = '_'.join([f'{idx:04d}' for idx in sample_indices[:5]])
            if len(sample_indices) > 5:
                sample_str += f'_and_{len(sample_indices)-5}_more'

            plot_file = plot_dir / f'comparison_feature_{feature_id:02d}_samples_{sample_str}.png'

            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            print(f"✓ 图形已保存到: {plot_file}")

        # 显示图形
        if show_plot:
            plt.show()
        else:
            plt.close()

    def visualize_all_features_for_sample(self,
                                         class_id: int,
                                         sample_idx: Optional[int] = None,
                                         show_plot: bool = False,
                                         save_plot: bool = True,
                                         max_features_per_fig: int = 8,
                                         figsize: Tuple[int, int] = (15, 12),
                                         dpi: int = 100) -> None:
        """
        可视化一个样本的所有特征

        Args:
            class_id: 类别ID
            sample_idx: 样本索引（如果为None，则随机选择）
            show_plot: 是否显示图形
            save_plot: 是否保存图形
            max_features_per_fig: 每张图最多显示的特征数量
            figsize: 图形大小
            dpi: 图形分辨率
        """
        # 验证类别ID
        if class_id < 0 or class_id >= self.num_classes:
            raise ValueError(f"无效的类别ID: {class_id}")

        # 获取该类别的所有样本索引
        class_mask = self.labels == class_id
        class_sample_indices = torch.where(class_mask)[0].tolist()

        if len(class_sample_indices) == 0:
            print(f"警告: 类别 {class_id} 没有样本")
            return

        # 确定要可视化的样本索引
        if sample_idx is None:
            sample_idx = np.random.choice(class_sample_indices)
            print(f"随机选择类别 {class_id} 的样本索引: {sample_idx}")
        else:
            if sample_idx not in class_sample_indices:
                raise ValueError(f"样本 {sample_idx} 不属于类别 {class_id}")

        # 提取数据
        sample_data = self.samples[sample_idx]  # [C, seq_length]

        # 计算需要多少张图
        num_figs = (self.num_features + max_features_per_fig - 1) // max_features_per_fig

        # 时间轴
        time_steps = np.arange(self.seq_length)

        for fig_idx in range(num_figs):
            # 确定这张图要显示的特征范围
            start_feat = fig_idx * max_features_per_fig
            end_feat = min((fig_idx + 1) * max_features_per_fig, self.num_features)
            num_feats_in_fig = end_feat - start_feat

            # 创建子图
            fig, axes = plt.subplots(num_feats_in_fig, 1, figsize=figsize, dpi=dpi)

            if num_feats_in_fig == 1:
                axes = [axes]

            # 绘制每个特征
            for idx, feat_id in enumerate(range(start_feat, end_feat)):
                ax = axes[idx]
                feature_data = sample_data[feat_id].numpy()

                # 绘制曲线
                ax.plot(time_steps, feature_data, linewidth=1.2, color='steelblue')
                # 使用特征名称
                feature_name = self.get_feature_name(feat_id)
                ax.set_title(feature_name, fontsize=10)
                ax.set_ylabel('数值', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')

                # 最后一个子图添加x轴标签
                if idx == num_feats_in_fig - 1:
                    ax.set_xlabel('时间步', fontsize=10)

            # 设置整体标题
            fig.suptitle(f'所有特征可视化 ({fig_idx+1}/{num_figs}) - Milestone {self.milestone} | '
                        f'类别 {class_id} | 样本 {sample_idx}',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()

            # 保存图形
            if save_plot:
                plot_dir = self.output_dir / f'class_{class_id:02d}'
                plot_dir.mkdir(parents=True, exist_ok=True)

                plot_file = plot_dir / f'sample_{sample_idx:04d}_all_features_part_{fig_idx+1}.png'

                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                print(f"✓ 图形已保存到: {plot_file}")

            # 显示图形
            if show_plot:
                plt.show()
            else:
                plt.close()


def main():
    parser = argparse.ArgumentParser(description='扩散模型采样结果可视化工具')

    # 必需参数
    parser.add_argument('--results_folder', type=str, required=True,
                       help='结果文件夹路径')
    parser.add_argument('--milestone', type=int, required=True,
                       help='要可视化的milestone编号')

    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件模块名（例如：default_config），用于获取特征名称')

    # 可选参数
    parser.add_argument('--list_classes', action='store_true',
                       help='列出每个类别的所有样本序号并保存到文件')
    parser.add_argument('--class_id', type=int, default=None,
                       help='要可视化的类别ID')
    parser.add_argument('--feature_ids', type=int, nargs='+', default=None,
                       help='要可视化的特征ID列表（用空格分隔）')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='样本索引（如果不指定，则随机选择）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='多样本对比时的样本数量')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multiple', 'all_features'],
                       help='可视化模式: single=单样本多特征, multiple=多样本单特征, all_features=单样本所有特征')
    parser.add_argument('--show_plot', action='store_true',
                       help='是否显示图形（默认不显示）')
    parser.add_argument('--no_save', action='store_true',
                       help='不保存图形（默认会保存）')
    parser.add_argument('--dpi', type=int, default=100,
                       help='图形分辨率（默认100）')

    args = parser.parse_args()

    # 创建可视化器
    print("=" * 70)
    print("扩散模型采样结果可视化工具")
    print("=" * 70)

    visualizer = SampleVisualizer(
        results_folder=args.results_folder,
        milestone=args.milestone,
        config_name=args.config
    )

    # 列出类别索引
    if args.list_classes:
        visualizer.list_samples_by_class(save_to_file=True)

    # 执行可视化
    if args.class_id is not None:
        print("\n" + "=" * 70)
        print("开始可视化...")
        print("=" * 70)

        if args.mode == 'single':
            # 单样本多特征模式
            if args.feature_ids is None:
                # 默认可视化前5个特征
                args.feature_ids = list(range(min(5, visualizer.num_features)))
                print(f"未指定特征，默认可视化特征: {args.feature_ids}")

            visualizer.visualize_sequence(
                class_id=args.class_id,
                feature_ids=args.feature_ids,
                sample_idx=args.sample_idx,
                show_plot=args.show_plot,
                save_plot=not args.no_save,
                dpi=args.dpi
            )

        elif args.mode == 'multiple':
            # 多样本单特征模式
            if args.feature_ids is None:
                feature_id = 0
                print(f"未指定特征，默认可视化特征 {feature_id}")
            else:
                feature_id = args.feature_ids[0]

            visualizer.visualize_multiple_samples(
                class_id=args.class_id,
                feature_id=feature_id,
                num_samples=args.num_samples,
                show_plot=args.show_plot,
                save_plot=not args.no_save,
                dpi=args.dpi
            )

        elif args.mode == 'all_features':
            # 单样本所有特征模式
            visualizer.visualize_all_features_for_sample(
                class_id=args.class_id,
                sample_idx=args.sample_idx,
                show_plot=args.show_plot,
                save_plot=not args.no_save,
                dpi=args.dpi
            )

        print("\n" + "=" * 70)
        print("可视化完成！")
        print("=" * 70)

    elif not args.list_classes:
        print("\n提示: 请使用 --list_classes 列出类别索引，或使用 --class_id 进行可视化")
        print("使用 --help 查看完整帮助信息")


if __name__ == '__main__':
    main()