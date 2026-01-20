"""
虚拟数据可视化和分析工具

功能：
1. 根据体重搜索最接近的真实参与者
2. 计算虚拟数据与真实数据的L1范数并排序
3. 可视化虚拟数据（单独或与真实数据对比）

使用示例：
    # 1. 查找最接近的参与者
    python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 --find_participant

    # 2. 计算范数并排序（结果保存在 results/class_00_synthetic_0000/ 目录下）
    python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 \
        --participant BT01 --compute_norm --feature_indices 0,1,2,3,4,5 --output_dir results

    # 3. 可视化虚拟数据（图片保存在 results/class_00_synthetic_0000/ 目录下）
    python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 \
        --visualize --feature_indices 0,1,2,3,4,5 --output_dir results

    # 4. 对比可视化（虚拟数据 vs 真实数据，图片保存在 results/class_00_synthetic_0000/ 目录下）
    python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 \
        --visualize --feature_indices 0,1,2,3,4,5 --real_data_path train/BT01/normal_walk_shuffle_0-6_01 \
        --real_start_idx 100 --output_dir results
"""

import os
import sys
import re
import argparse
import importlib
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无需显示器的后端
from matplotlib import rcParams
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import matplotlib as mpl
# rcParams['font.sans-serif'] = ['SimHei']   # 黑体（Windows 基本都有）
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
rcParams['axes.unicode_minus'] = False     # 解决负号显示问题


def get_feature_indices_str(feature_indices, max_display=10):
    """
    生成特征索引的字符串表示，用于文件命名

    Args:
        feature_indices: 特征索引列表，如果为None表示所有特征
        max_display: 最多显示的索引数量，超过则显示范围

    Returns:
        str: 特征索引字符串，如 "all" 或 "0_1_2_3_4_5" 或 "0-50"
    """
    if feature_indices is None:
        return "all"

    if len(feature_indices) == 0:
        return "none"

    # 如果特征数量较少，直接列出
    if len(feature_indices) <= max_display:
        return "_".join(map(str, feature_indices))

    # 如果特征数量较多，显示范围
    return f"{min(feature_indices)}-{max(feature_indices)}"


def get_optimal_subplot_layout(num_plots):
    """
    根据子图数量计算最佳的行列布局

    Args:
        num_plots: 子图数量

    Returns:
        tuple: (n_rows, n_cols) 最佳的行列数
    """
    if num_plots <= 0:
        return (1, 1)

    # 特殊情况：1-4个图使用固定布局
    if num_plots == 1:
        return (1, 1)
    elif num_plots == 2:
        return (1, 2)
    elif num_plots == 3:
        return (1, 3)
    elif num_plots == 4:
        return (2, 2)

    # 5个及以上：优先使用3列布局或正方形布局
    import math
    sqrt_n = int(math.sqrt(num_plots))

    # 首先检查是否是完美正方形
    if sqrt_n * sqrt_n == num_plots:
        return (sqrt_n, sqrt_n)

    # 尝试3列布局
    n_cols = 3
    n_rows = (num_plots + n_cols - 1) // n_cols
    empty_plots = n_rows * n_cols - num_plots

    # 如果3列布局的空白格子少于3个，就使用3列
    if empty_plots < 3:
        return (n_rows, n_cols)

    # 否则，寻找最接近正方形的其他布局
    best_layout = None
    min_diff = float('inf')

    # 在sqrt附近搜索因子
    for n_cols in range(max(2, sqrt_n - 1), min(sqrt_n + 2, num_plots + 1)):
        if n_cols == 3:  # 3列已经在上面检查过了
            continue

        n_rows = (num_plots + n_cols - 1) // n_cols
        empty_plots = n_rows * n_cols - num_plots

        # 避免太多空白（空白不超过一行）
        if empty_plots >= n_cols:
            continue

        # 计算行列差异（越接近正方形越好）
        diff = abs(n_rows - n_cols)

        # 如果是完美因子（无空白），优先选择
        if num_plots % n_cols == 0:
            if diff < min_diff:
                min_diff = diff
                best_layout = (n_rows, n_cols)
        # 否则，选择差异最小的
        elif best_layout is None or diff < min_diff:
            min_diff = diff
            best_layout = (n_rows, n_cols)

    if best_layout:
        return best_layout

    # 最后的默认值：使用3列布局
    n_cols = 3
    n_rows = (num_plots + n_cols - 1) // n_cols
    return (n_rows, n_cols)


def load_config(config_name):
    """
    加载配置文件

    Args:
        config_name: 配置文件名或路径
    """
    if '/' in config_name or '\\' in config_name or config_name.endswith('.py'):
        filepath = config_name.replace('\\', '/').rstrip('.py') + '.py'
        spec = importlib.util.spec_from_file_location("config", filepath)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        config = importlib.import_module(config_name)

    return config


def load_synthetic_data(synthetic_dir, class_id, synthetic_id):
    """
    加载虚拟数据

    Args:
        synthetic_dir: 虚拟数据根目录
        class_id: 类别ID (0-27)
        synthetic_id: 虚拟数据序号 (0-9999)

    Returns:
        pd.DataFrame: 虚拟数据
    """
    csv_path = Path(synthetic_dir) / f"class_{class_id:02d}" / f"synthetic_{synthetic_id:04d}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"虚拟数据文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 虚拟数据加载成功: {csv_path}")
    print(f"  - 形状: {df.shape}")
    print(f"  - 特征数: {len(df.columns)}")

    return df


def find_closest_participant(synthetic_df, config):
    """
    根据体重查找最接近的真实参与者

    Args:
        synthetic_df: 虚拟数据DataFrame
        config: 配置对象

    Returns:
        tuple: (participant_name, actual_mass, synthetic_mass_mean)
    """
    if 'participant_mass' not in synthetic_df.columns:
        raise ValueError("虚拟数据中没有 participant_mass 列")

    # 计算虚拟数据中体重的均值
    synthetic_mass_mean = synthetic_df['participant_mass'].mean()

    # 从config中获取参与者体重字典
    participant_masses = config.participant_masses

    # 找到最接近的参与者
    min_diff = float('inf')
    closest_participant = None
    closest_mass = None

    for participant, mass in participant_masses.items():
        diff = abs(mass - synthetic_mass_mean)
        if diff < min_diff:
            min_diff = diff
            closest_participant = participant
            closest_mass = mass

    print("\n" + "=" * 70)
    print("体重匹配结果")
    print("=" * 70)
    print(f"虚拟数据体重均值: {synthetic_mass_mean:.2f} kg")
    print(f"最接近的参与者:   {closest_participant}")
    print(f"参与者实际体重:   {closest_mass:.2f} kg")
    print(f"体重差异:         {abs(closest_mass - synthetic_mass_mean):.2f} kg")
    print("=" * 70)

    return closest_participant, closest_mass, synthetic_mass_mean


def get_matching_files(data_dir, participant, class_pattern):
    """
    获取匹配类别正则表达式的文件列表

    Args:
        data_dir: 数据目录
        participant: 参与者名称
        class_pattern: 类别正则表达式（可以是单个pattern或列表）

    Returns:
        list: 匹配的文件夹路径列表
    """
    # 如果pattern是列表，合并为单个正则表达式
    if isinstance(class_pattern, list):
        combined_pattern = '|'.join([f'({p})' for p in class_pattern])
    else:
        combined_pattern = class_pattern

    matching_folders = []

    # 遍历participant目录
    participant_path = Path(data_dir) / participant
    if not participant_path.exists():
        print(f"警告: 参与者目录不存在: {participant_path}")
        return matching_folders

    for folder in participant_path.iterdir():
        if folder.is_dir():
            # 检查文件夹名是否匹配正则表达式
            if re.match(combined_pattern, folder.name):
                matching_folders.append(folder)

    return matching_folders


def read_real_data_features(folder_path, participant, feature_names, config):
    """
    从真实数据文件夹读取指定特征

    Args:
        folder_path: 数据文件夹路径
        participant: 参与者名称
        feature_names: 要读取的特征名称列表
        config: 配置对象

    Returns:
        pd.DataFrame: 包含所有特征的DataFrame
    """
    folder_name = folder_path.name
    combined_df = pd.DataFrame()

    # 准备side替换的input_names和label_names
    sides = config.side if isinstance(config.side, list) else [config.side]

    input_features = []
    label_features = []
    activity_features = []

    for side in sides:
        # 替换input_names中的*
        for name in config.input_names:
            input_features.append(name.replace('*', side))
        # 替换label_names中的*
        for name in config.label_names:
            label_features.append(name.replace('*', side))

    # 添加activity_flag特征
    for side in sides:
        activity_features.append(f'activity_flag_{side}')

    # 读取每个特征
    for feature_name in feature_names:
        if feature_name == 'participant_mass':
            # 体重特征：使用config中的值
            if participant in config.participant_masses:
                mass = config.participant_masses[participant]
                # 暂时不添加，等确定数据长度后再添加
                continue
            else:
                raise ValueError(f"参与者 {participant} 的体重未在config中定义")

        elif feature_name in input_features:
            # 从exo.csv读取
            csv_path = folder_path / f"{participant}_{folder_name}_exo.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"文件不存在: {csv_path}")
            df = pd.read_csv(csv_path)
            if feature_name not in df.columns:
                raise ValueError(f"特征 {feature_name} 不在文件 {csv_path} 中")
            combined_df[feature_name] = df[feature_name]

        elif feature_name in label_features:
            # 从moment_filt.csv读取
            csv_path = folder_path / f"{participant}_{folder_name}_moment_filt.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"文件不存在: {csv_path}")
            df = pd.read_csv(csv_path)
            if feature_name not in df.columns:
                raise ValueError(f"特征 {feature_name} 不在文件 {csv_path} 中")
            combined_df[feature_name] = df[feature_name]

        elif feature_name in activity_features:
            # 从activity_flag.csv读取
            csv_path = folder_path / f"{participant}_{folder_name}_activity_flag.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"文件不存在: {csv_path}")
            df = pd.read_csv(csv_path)

            # 确定读取left还是right列
            if feature_name.endswith('_l'):
                column_name = 'left'
            elif feature_name.endswith('_r'):
                column_name = 'right'
            else:
                raise ValueError(f"无法确定activity_flag的列: {feature_name}")

            if column_name not in df.columns:
                raise ValueError(f"列 {column_name} 不在文件 {csv_path} 中")
            combined_df[feature_name] = df[column_name]

        else:
            raise ValueError(f"无法确定特征 {feature_name} 的来源文件")

    # 添加participant_mass列（如果需要）
    if 'participant_mass' in feature_names:
        mass = config.participant_masses[participant]
        combined_df['participant_mass'] = mass

    return combined_df


def compute_norm_for_all_files(synthetic_df, participant, config, class_id,
                               feature_indices=None, output_dir=None, synthetic_id=None):
    """
    计算虚拟数据与所有真实数据文件的L1范数

    Args:
        synthetic_df: 虚拟数据DataFrame
        participant: 参与者名称
        config: 配置对象
        class_id: 类别ID
        feature_indices: 要计算的特征索引列表（None=除体重外所有）
        output_dir: 输出目录路径
        synthetic_id: 虚拟数据ID（用于生成文件名）

    Returns:
        list: 排序后的结果列表，每项为 (norm_mean, start_idx, folder_path)
    """
    print("\n" + "=" * 70)
    print("计算L1范数")
    print("=" * 70)

    # 确定要计算的特征
    all_features = synthetic_df.columns.tolist()

    if feature_indices is None:
        # 除了participant_mass外的所有特征
        feature_names = [f for f in all_features if f != 'participant_mass']
        feature_indices = [i for i, f in enumerate(all_features) if f != 'participant_mass']
    else:
        feature_names = [all_features[i] for i in feature_indices]

    print(f"参与者: {participant}")
    print(f"类别ID: {class_id}")
    print(f"计算特征数: {len(feature_names)}")
    print(f"特征列表: {feature_names[:5]}..." if len(feature_names) > 5 else f"特征列表: {feature_names}")

    # 提取虚拟数据的特征
    synthetic_data = synthetic_df[feature_names].values  # [seq_len, num_features]
    synthetic_len = len(synthetic_data)

    print(f"虚拟数据长度: {synthetic_len}")

    # 获取类别正则表达式
    class_pattern = config.action_patterns[class_id]
    print(f"类别正则表达式: {class_pattern}")

    # 获取所有匹配的文件夹
    all_results = []

    for mode in config.mode:
        data_dir = Path(config.data_dir) / mode
        matching_folders = get_matching_files(data_dir, participant, class_pattern)

        print(f"\n模式 '{mode}' 下找到 {len(matching_folders)} 个匹配文件夹")

        for folder_path in tqdm(matching_folders, desc=f"处理 {mode}"):
            try:
                # 读取真实数据
                real_df = read_real_data_features(folder_path, participant, feature_names, config)
                real_data = real_df.values  # [real_len, num_features]
                real_len = len(real_data)

                if real_len < synthetic_len:
                    # 真实数据太短，跳过
                    continue

                # 遍历所有可能的起始位置
                for start_idx in range(real_len - synthetic_len + 1):
                    real_subsequence = real_data[start_idx:start_idx + synthetic_len]

                    # 计算L1范数
                    l1_norm = np.mean(np.abs(synthetic_data - real_subsequence))

                    # 保存结果
                    relative_path = folder_path.relative_to(config.data_dir)
                    all_results.append((l1_norm, start_idx, str(relative_path)))

            except Exception as e:
                print(f"\n警告: 处理文件夹 {folder_path} 时出错: {e}")
                continue

    # 按范数排序
    all_results.sort(key=lambda x: x[0])

    print(f"\n✓ 共计算 {len(all_results)} 个子序列的范数")
    print(f"最小范数: {all_results[0][0]:.2f}" if all_results else "无结果")

    # 保存到文件
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名，包含特征索引信息
        feature_str = get_feature_indices_str(feature_indices)
        filename = f"norm_results_{participant}_features_{feature_str}.txt"
        output_file = output_path / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("序号\tL1范数和\t起始索引\t真实数据路径\n")
            for idx, (norm_sum, start_idx, folder_path) in enumerate(all_results):
                f.write(f"{idx}\t{norm_sum:.6f}\t{start_idx}\t{folder_path}\n")

        print(f"✓ 结果已保存到: {output_file}")

    print("=" * 70)

    return all_results


def visualize_data(synthetic_df, feature_indices=None, real_df=None, real_start_idx=0,
                   save_path=None, config=None):
    """
    可视化数据

    Args:
        synthetic_df: 虚拟数据DataFrame
        feature_indices: 要可视化的特征索引列表
        real_df: 真实数据DataFrame（可选）
        real_start_idx: 真实数据起始索引
        save_path: 保存图片路径（可选）
        config: 配置对象（可选，用于获取feature_display_names）
    """
    print("\n" + "=" * 70)
    print("数据可视化")
    print("=" * 70)

    # 确定要可视化的特征
    all_features = synthetic_df.columns.tolist()

    if feature_indices is None:
        # 除了participant_mass外的所有特征
        feature_names = [f for f in all_features if f != 'participant_mass']
        feature_indices = [i for i, f in enumerate(all_features) if f != 'participant_mass']
    else:
        feature_names = [all_features[i] for i in feature_indices]

    num_features = len(feature_names)
    print(f"可视化特征数: {num_features}")

    # 提取虚拟数据
    synthetic_data = synthetic_df[feature_names].values  # [seq_len, num_features]
    synthetic_len = len(synthetic_data)

    # 使用智能布局计算最佳行列数
    n_rows, n_cols = get_optimal_subplot_layout(num_features)

    print(f"子图布局: {n_rows}行 x {n_cols}列")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if num_features > 1 else [axes]

    # 绘制每个特征
    for idx, (feature_idx, feature_name) in enumerate(zip(feature_indices, feature_names)):
        ax = axes[idx]

        # 绘制虚拟数据
        ax.plot(synthetic_data[:, idx], label='虚拟数据', linewidth=2, alpha=0.8)

        # 如果提供了真实数据，也绘制
        if real_df is not None:
            real_data = real_df[feature_name].values
            real_subsequence = real_data[real_start_idx:real_start_idx + synthetic_len]
            ax.plot(real_subsequence, label='真实数据', linewidth=2, alpha=0.8, linestyle='--')

        # 设置标题：优先使用config中的display_names
        display_name = feature_name
        if config is not None:
            feature_display_names = getattr(config, 'feature_display_names', None)
            if feature_display_names and len(feature_display_names) > feature_idx:
                display_name = feature_display_names[feature_idx]

        ax.set_title(display_name, fontsize=10)
        ax.set_xlabel('时间步')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(num_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图片已保存到: {save_path}")
    else:
        plt.show()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='虚拟数据可视化和分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 查找最接近的参与者
  python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 --find_participant

  # 2. 计算范数并排序
  python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 \\
      --participant BT01 --compute_norm --feature_indices 0,1,2,3,4,5

  # 3. 可视化虚拟数据
  python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 \\
      --visualize --feature_indices 0,1,2,3,4,5

  # 4. 对比可视化（虚拟数据 vs 真实数据）
  python visualize_synthetic_data.py --config configs.default_config --class_id 0 --synthetic_id 0 \\
      --visualize --feature_indices 0,1,2,3,4,5 --real_data_path train/BT01/normal_walk_shuffle_0-6_01 --real_start_idx 100
        """
    )

    # 必需参数
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (例: configs.default_config)')
    parser.add_argument('--class_id', type=int, required=True,
                        help='类别ID (0-27)')
    parser.add_argument('--synthetic_id', type=int, required=True,
                        help='虚拟数据序号 (0-9999)')

    # 可选参数
    parser.add_argument('--synthetic_dir', type=str, default='../synthetic_data',
                        help='虚拟数据根目录 (默认: ../synthetic_data)')

    # 功能选项
    parser.add_argument('--find_participant', action='store_true',
                        help='根据体重查找最接近的参与者')
    parser.add_argument('--compute_norm', action='store_true',
                        help='计算与真实数据的L1范数')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化数据')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果保存根目录 (默认: results)，会在此目录下创建class_XX_synthetic_XXXX子文件夹')

    # 范数计算相关
    parser.add_argument('--participant', type=str,
                        help='参与者名称 (用于范数计算)')
    parser.add_argument('--feature_indices', type=str,
                        help='要计算/可视化的特征索引，逗号分隔 (例: 0,1,2,3,4,5)，不指定则为除体重外所有特征')

    # 可视化相关
    parser.add_argument('--real_data_path', type=str,
                        help='真实数据路径 (相对于data_dir，例: train/BT01/normal_walk_shuffle_0-6_01)')
    parser.add_argument('--real_start_idx', type=int, default=0,
                        help='真实数据起始索引 (默认: 0)')

    args = parser.parse_args()

    # 加载配置
    print("=" * 70)
    print("虚拟数据可视化和分析工具")
    print("=" * 70)
    print(f"\n加载配置文件: {args.config}")
    config = load_config(args.config)
    print(f"✓ 配置加载完成")

    # 加载虚拟数据
    synthetic_df = load_synthetic_data(args.synthetic_dir, args.class_id, args.synthetic_id)

    # 创建结果保存文件夹（基于class_id和synthetic_id）
    result_folder_name = f"class_{args.class_id:02d}_synthetic_{args.synthetic_id:04d}"
    result_dir = Path(args.output_dir) / result_folder_name
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 结果保存目录: {result_dir}")

    # 解析特征索引
    feature_indices = None
    if args.feature_indices:
        feature_indices = [int(x.strip()) for x in args.feature_indices.split(',')]

    # 执行功能
    if args.find_participant:
        find_closest_participant(synthetic_df, config)

    if args.compute_norm:
        if not args.participant:
            print("\n错误: --compute_norm 需要指定 --participant")
            return

        compute_norm_for_all_files(
            synthetic_df=synthetic_df,
            participant=args.participant,
            config=config,
            class_id=args.class_id,
            feature_indices=feature_indices,
            output_dir=result_dir,
            synthetic_id=args.synthetic_id
        )

    if args.visualize:
        real_df = None

        # 如果指定了真实数据路径，加载真实数据
        if args.real_data_path:
            # 解析路径: mode/participant/folder_name
            path_parts = args.real_data_path.split('/')
            if len(path_parts) != 3:
                print(f"\n错误: 真实数据路径格式应为 mode/participant/folder_name，得到: {args.real_data_path}")
                return

            mode, participant, folder_name = path_parts
            folder_path = Path(config.data_dir) / mode / participant / folder_name

            if not folder_path.exists():
                print(f"\n错误: 真实数据文件夹不存在: {folder_path}")
                return

            # 确定要读取的特征
            all_features = synthetic_df.columns.tolist()
            if feature_indices is None:
                feature_names = [f for f in all_features if f != 'participant_mass']
            else:
                feature_names = [all_features[i] for i in feature_indices]

            try:
                real_df = read_real_data_features(folder_path, participant, feature_names, config)
                print(f"\n✓ 真实数据加载成功: {folder_path}")
                print(f"  - 形状: {real_df.shape}")
            except Exception as e:
                print(f"\n错误: 加载真实数据失败: {e}")
                return

        # 可视化
        # 生成图片保存路径，包含特征索引信息
        feature_str = get_feature_indices_str(feature_indices)
        if real_df is not None:
            fig_name = f"comparison_vs_{participant}_features_{feature_str}.png"
        else:
            fig_name = f"synthetic_data_features_{feature_str}.png"
        save_path = result_dir / fig_name

        visualize_data(
            synthetic_df=synthetic_df,
            feature_indices=feature_indices,
            real_df=real_df,
            real_start_idx=args.real_start_idx,
            save_path=save_path,
            config=config
        )

    print("\n" + "=" * 70)
    print("✓ 所有任务完成")
    print("=" * 70)


if __name__ == '__main__':
    main()