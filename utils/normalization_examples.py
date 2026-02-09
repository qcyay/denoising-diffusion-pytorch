"""
数据归一化功能使用示例
"""

import torch
from dataset_loaders.sequence_dataloader import DiffusionSequenceDataset
import configs.default_config as config


def example_1_generate_statistics():
    """
    示例1: 生成特征统计文件

    在启用归一化之前，需要先运行这个步骤来计算并保存
    每个类别每个特征的最大最小值
    """
    print("=" * 70)
    print("示例1: 生成特征统计文件")
    print("=" * 70)

    # 创建数据集（不启用归一化）
    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=config.participant_masses,
        device=torch.device('cpu'),
        mode='train',
        remove_nan=True,
        remove_any_nan=True,
        activity_flag=config.activity_flag,
        use_participant_mass=config.use_participant_mass,
        enable_normalization=False  # 关闭归一化
    )

    # 计算并保存统计信息
    print("\n开始计算特征统计信息...")
    dataset.compute_and_save_statistics(output_dir='statistics')

    print("\n✓ 统计文件已生成: statistics/feature_statistics_old.json")
    print("  现在可以启用归一化功能了")


def example_2_load_normalized_data():
    """
    示例2: 加载归一化后的数据

    使用已生成的统计文件，创建归一化的数据集
    """
    print("\n" + "=" * 70)
    print("示例2: 加载归一化后的数据")
    print("=" * 70)

    # 创建数据集（启用归一化）
    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=config.participant_masses,
        device=torch.device('cpu'),
        mode='train',
        remove_nan=True,
        remove_any_nan=True,
        activity_flag=config.activity_flag,
        use_participant_mass=config.use_participant_mass,
        enable_normalization=True,  # 启用归一化
        feature_statistics_path=config.feature_statistics_path,
        normalization_method=config.normalization_method,
        normalization_params=config.normalization_params
    )

    # 查看归一化后的数据
    print("\n查看归一化后的数据:")
    data, label = dataset[0]
    print(f"  数据形状: {data.shape}")
    print(f"  类别标签: {label}")
    print(f"  数据范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  数据均值: {data.mean():.4f}")
    print(f"  数据标准差: {data.std():.4f}")

    # 对比前3个特征的原始值和归一化值
    print("\n前3个特征的统计:")
    for i in range(min(3, data.shape[0])):
        print(f"  特征{i}: min={data[i].min():.4f}, max={data[i].max():.4f}, "
              f"mean={data[i].mean():.4f}")


def example_3_denormalize_data():
    """
    示例3: 反归一化生成的数据

    模拟模型生成数据后，将归一化的数据还原到原始尺度
    """
    print("\n" + "=" * 70)
    print("示例3: 反归一化生成的数据")
    print("=" * 70)

    # 创建数据集（启用归一化）
    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=config.participant_masses,
        device=torch.device('cpu'),
        mode='train',
        remove_nan=True,
        remove_any_nan=True,
        activity_flag=config.activity_flag,
        use_participant_mass=config.use_participant_mass,
        enable_normalization=True,
        feature_statistics_path=config.feature_statistics_path,
        normalization_method=config.normalization_method,
        normalization_params=config.normalization_params
    )

    # 获取一个样本
    normalized_data, label = dataset[0]
    print(f"\n归一化数据:")
    print(f"  形状: {normalized_data.shape}")
    print(f"  范围: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    print(f"  类别: {label}")

    # 反归一化
    denormalized_data = dataset.denormalize_data(normalized_data, label.item())
    print(f"\n反归一化数据:")
    print(f"  形状: {denormalized_data.shape}")
    print(f"  范围: [{denormalized_data.min():.4f}, {denormalized_data.max():.4f}]")

    # 验证反归一化的正确性
    # 如果我们再次归一化，应该得到相同的结果
    renormalized_data = dataset._normalize_data(denormalized_data, label.item())
    diff = torch.abs(normalized_data - renormalized_data).max()
    print(f"\n反归一化正确性验证:")
    print(f"  最大差异: {diff:.6f}")
    print(f"  {'✓ 反归一化正确' if diff < 1e-4 else '✗ 反归一化存在问题'}")


def example_4_compare_normalization_methods():
    """
    示例4: 比较不同归一化方法的效果

    对比线性归一化、tanh归一化和幂函数归一化的效果
    """
    print("\n" + "=" * 70)
    print("示例4: 比较不同归一化方法的效果")
    print("=" * 70)

    methods = [
        ('linear', {}),
        ('tanh', {'tanh_scale': 2.0}),
        ('tanh', {'tanh_scale': 3.0}),
        ('tanh', {'tanh_scale': 5.0}),
        ('power', {'power_alpha': 2.0}),
    ]

    for method, params in methods:
        print(f"\n方法: {method}")
        if params:
            print(f"参数: {params}")

        # 创建数据集
        dataset = DiffusionSequenceDataset(
            data_dir=config.data_dir,
            input_names=config.input_names,
            label_names=config.label_names,
            side=config.side,
            diffusion_sequence_length=config.diffusion_sequence_length,
            action_patterns=config.action_patterns,
            participant_masses=config.participant_masses,
            device=torch.device('cpu'),
            mode='train',
            remove_nan=True,
            remove_any_nan=True,
            activity_flag=config.activity_flag,
            use_participant_mass=config.use_participant_mass,
            enable_normalization=True,
            feature_statistics_path=config.feature_statistics_path,
            normalization_method=method,
            normalization_params=params
        )

        # 获取多个样本的统计
        all_data = []
        for i in range(min(100, len(dataset))):
            data, _ = dataset[i]
            all_data.append(data)

        all_data = torch.stack(all_data)

        print(f"  数据范围: [{all_data.min():.4f}, {all_data.max():.4f}]")
        print(f"  数据均值: {all_data.mean():.4f}")
        print(f"  数据标准差: {all_data.std():.4f}")

        # 计算数据分布
        bins = torch.linspace(0, 1, 11)
        hist = torch.histc(all_data, bins=10, min=0, max=1)
        print(f"  数据分布 (10个区间):")
        for i, count in enumerate(hist):
            print(f"    [{bins[i]:.1f}-{bins[i + 1]:.1f}): {count.item():.0f} "
                  f"({'=' * int(count.item() / hist.max().item() * 20)})")


def example_5_batch_processing():
    """
    示例5: 批量处理归一化数据

    展示如何在DataLoader中使用归一化数据
    """
    print("\n" + "=" * 70)
    print("示例5: 批量处理归一化数据")
    print("=" * 70)

    from torch.utils.data import DataLoader

    # 创建数据集
    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=config.participant_masses,
        device=torch.device('cpu'),
        mode='train',
        remove_nan=True,
        remove_any_nan=True,
        activity_flag=config.activity_flag,
        use_participant_mass=config.use_participant_mass,
        enable_normalization=True,
        feature_statistics_path=config.feature_statistics_path,
        normalization_method=config.normalization_method,
        normalization_params=config.normalization_params
    )

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"\n批量数据统计 (batch_size=32):")
    for batch_idx, (data_batch, label_batch) in enumerate(dataloader):
        print(f"\n批次 {batch_idx}:")
        print(f"  数据形状: {data_batch.shape}")  # [32, num_features, seq_len]
        print(f"  标签形状: {label_batch.shape}")  # [32]
        print(f"  数据范围: [{data_batch.min():.4f}, {data_batch.max():.4f}]")
        print(f"  批次类别: {label_batch.unique().tolist()}")

        if batch_idx >= 2:  # 只显示前3个批次
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='数据归一化功能使用示例')
    parser.add_argument('--example', type=int, default=0,
                        help='运行指定的示例 (1-5), 0表示运行所有示例')
    args = parser.parse_args()

    examples = {
        1: example_1_generate_statistics,
        2: example_2_load_normalized_data,
        3: example_3_denormalize_data,
        4: example_4_compare_normalization_methods,
        5: example_5_batch_processing,
    }

    if args.example == 0:
        print("运行所有示例...\n")
        for i in sorted(examples.keys()):
            try:
                examples[i]()
            except Exception as e:
                print(f"\n✗ 示例{i}运行失败: {e}")
    else:
        if args.example in examples:
            examples[args.example]()
        else:
            print(f"错误: 无效的示例编号 {args.example}")
            print(f"可用示例: {list(examples.keys())}")