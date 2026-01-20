"""
扩散模型采样脚本 - 生成仿真数据用于数据增强

功能：
1. 从训练好的扩散模型生成指定类别的仿真数据
2. 反归一化到原始物理范围
3. 保存为CSV格式，与真实数据格式一致
4. 可直接用于力矩预测模型的训练数据增强
"""

import os
import sys
import argparse
import importlib
import importlib.util
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 设置 CUDA_VISIBLE_DEVICES 来指定可见的 GPU（例如 GPU 1 和 GPU 2）
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def load_config(config_name):
    """
    加载配置文件

    Args:
        config_name: 配置文件名或路径
    """
    # 支持两种格式：configs.default_config 或 configs/default_config.py
    if '/' in config_name or '\\' in config_name or config_name.endswith('.py'):
        # 文件路径形式
        filepath = config_name.replace('\\', '/').rstrip('.py') + '.py'
        spec = importlib.util.spec_from_file_location("config", filepath)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        # 模块名形式
        config = importlib.import_module(config_name)

    return config


def generate_synthetic_data(
    config_path,
    milestone,
    output_dir=None,
    device=None
):
    """
    生成仿真数据

    Args:
        config_path: 配置文件路径
        milestone: 模型milestone编号（如 50）
        output_dir: 输出目录（可选，如果不提供则使用config中的设置）
        device: 计算设备（可选，如果不提供则使用config中的设置）
    """
    print("=" * 70)
    print("扩散模型仿真数据生成")
    print("=" * 70)

    # 1. 加载配置文件
    print(f"\n加载配置文件: {config_path}")
    config = load_config(config_path)

    # 2. 从config读取参数（如果命令行未提供）
    results_folder = Path(config.results_folder)
    classes_to_generate = getattr(config, 'synthetic_classes_to_generate', None)
    samples_per_class = getattr(config, 'synthetic_samples_per_class', 100)
    batch_size = config.train_batch_size

    if output_dir is None:
        output_dir = getattr(config, 'synthetic_output_dir', 'synthetic_data')

    if device is None:
        device = getattr(config, 'device', 'cuda')

    print(f"\n从配置文件读取参数:")
    print(f"  - results_folder: {results_folder}")
    print(f"  - classes_to_generate: {classes_to_generate if classes_to_generate else '所有类别'}")
    print(f"  - samples_per_class: {samples_per_class}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - output_dir: {output_dir}")
    print(f"  - device: {device}")

    # 3. 构建模型路径
    model_path = results_folder / f'model-{milestone}.pt'

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            f"请检查 results_folder 和 milestone 是否正确"
        )

    print(f"\n模型路径: {model_path}")

    # 2. 加载数据集（用于获取denormalize_data方法和特征名称）
    print(f"\n创建数据集...")
    from dataset_loaders.sequence_dataloader import DiffusionSequenceDataset

    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=getattr(config, 'participant_masses', {}),
        mode='train',
        remove_nan=getattr(config, 'remove_nan', False),
        remove_any_nan=getattr(config, 'remove_any_nan', True),
        activity_flag=getattr(config, 'activity_flag', False),
        use_participant_mass=getattr(config, 'use_participant_mass', False),
        min_sequence_length=getattr(config, 'min_sequence_length', -1),
        enable_normalization=config.enable_normalization,
        feature_statistics_path=config.feature_statistics_path,
        normalization_method=getattr(config, 'normalization_method', 'linear'),
        normalization_params=getattr(config, 'normalization_params', None)
    )

    print(f"✓ 数据集加载完成")
    print(f"  - 特征维度: {dataset.get_num_features()}")
    print(f"  - 序列长度: {config.diffusion_sequence_length}")
    print(f"  - 类别数量: {len(config.action_patterns)}")

    # 获取特征名称
    feature_names = dataset._get_feature_names()
    print(f"  - 特征名称: {len(feature_names)} 个")

    # 3. 创建模型和训练器
    print(f"\n创建训练器和加载模型...")
    from denoising_diffusion_pytorch.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D, Trainer1D

    num_classes = dataset.get_num_classes()
    num_features = dataset.get_num_features()

    # 创建UNet模型
    model = Unet1D(
        dim=config.dim,
        num_classes=num_classes,
        cond_drop_prob=config.cond_drop_prob,
        dim_mults=config.dim_mults,
        channels=num_features
    )

    diffusion_model = GaussianDiffusion1D(
        model,
        seq_length=config.diffusion_sequence_length,
        timesteps=config.timesteps,
        channels=num_features
    )

    trainer = Trainer1D(
        diffusion_model=diffusion_model,
        dataset=dataset,
        train_batch_size=config.train_batch_size,
        gradient_accumulate_every=config.gradient_accumulate_every,
        train_lr=config.train_lr,
        train_num_steps=config.train_num_steps,
        ema_update_every=config.ema_update_every,
        ema_decay=config.ema_decay,
        save_and_sample_every=config.save_and_sample_every,
        num_samples=config.num_samples,
        results_folder=config.results_folder,
        amp=config.amp
    )

    # 加载模型
    print(f"加载模型权重: {model_path}")
    trainer.load(milestone)
    print(f"✓ 模型加载完成 (milestone {milestone})")

    # 4. 确定要生成的类别
    if classes_to_generate is None:
        classes_to_generate = list(range(len(config.action_patterns)))

    print(f"\n生成配置:")
    print(f"  - 生成类别: {classes_to_generate}")
    print(f"  - 每类样本数: {samples_per_class}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 输出目录: {output_dir}")

    # 5. 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. 为每个类别生成数据
    print(f"\n开始生成仿真数据...")
    print("=" * 70)

    for class_id in classes_to_generate:
        class_name = config.action_patterns[class_id] if class_id < len(config.action_patterns) else f"class_{class_id}"
        print(f"\n生成类别 {class_id}: {class_name}")

        # 创建类别目录
        class_dir = output_dir / f"class_{class_id:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # 计算需要的批次数
        num_batches = (samples_per_class + batch_size - 1) // batch_size
        samples_generated = 0

        for batch_idx in tqdm(range(num_batches), desc=f"类别 {class_id}"):
            # 确定当前批次的样本数
            current_batch_size = min(batch_size, samples_per_class - samples_generated)

            # 生成样本（使用EMA模型）
            with torch.no_grad():
                # 创建类别标签
                classes = torch.full((current_batch_size,), class_id, dtype=torch.long, device=device)

                # 使用EMA模型采样
                samples = trainer.ema.ema_model.sample(
                    classes=classes,
                    batch_size=current_batch_size
                )  # [batch_size, C, seq_length]

            # 反归一化每个样本
            if dataset.enable_normalization:
                denormalized_samples = []
                for i in range(samples.shape[0]):
                    sample = samples[i]  # [C, seq_length]
                    denormalized_sample = dataset.denormalize_data(sample, class_id)
                    denormalized_samples.append(denormalized_sample)
                samples = torch.stack(denormalized_samples, dim=0)

            # 保存每个样本为CSV文件
            for i in range(current_batch_size):
                sample = samples[i].cpu().numpy()  # [C, seq_length]

                # 转换为DataFrame (seq_length, C)
                df = pd.DataFrame(sample.T, columns=feature_names)

                # 保存文件
                file_idx = samples_generated + i
                csv_path = class_dir / f"synthetic_{file_idx:04d}.csv"
                df.to_csv(csv_path, index=False)

            samples_generated += current_batch_size

        print(f"✓ 类别 {class_id} 完成: 生成 {samples_generated} 个样本")

    print("\n" + "=" * 70)
    print(f"✓ 所有仿真数据生成完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  总样本数: {len(classes_to_generate) * samples_per_class}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='扩散模型仿真数据生成')

    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (例: configs.default_config 或 configs/default_config.py)')
    parser.add_argument('--milestone', type=int, required=True,
                       help='模型milestone编号 (例: 50)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (可选，默认使用config中的synthetic_output_dir)')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (可选，默认使用config中的device)')

    args = parser.parse_args()

    # 生成仿真数据
    generate_synthetic_data(
        config_path=args.config,
        milestone=args.milestone,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()