"""
扩散模型训练脚本

使用Classifier-Free Guidance的1D扩散模型训练生物力学序列数据
"""

import os
import sys
import argparse
import importlib
import torch
import numpy as np
import random

# 导入自定义模块
from dataset_loaders.sequence_dataloader import DiffusionSequenceDataset
from denoising_diffusion_pytorch.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D, Trainer1D

# 设置 CUDA_VISIBLE_DEVICES 来指定可见的 GPU（例如 GPU 1 和 GPU 2）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ==================== 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description='训练扩散模型')
    parser.add_argument('--config', type=str, default='default_config',
                        help='配置文件模块名 (default: default_config)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备 (default: cuda:0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练，指定milestone编号 (例如: 100)')

    args = parser.parse_args()

    # ==================== 加载配置文件 ====================
    print("=" * 70)
    print("加载配置文件...")
    print("=" * 70)

    config = importlib.import_module(args.config)
    print(f"✓ 配置文件已加载: {args.config}")

    # ==================== 设置随机种子 ====================
    set_seed(config.random_seed)
    print(f"✓ 随机种子已设置: {config.random_seed}")

    # 移除手动GPU设置，让Accelerate管理
    # # ==================== 设置设备 ====================
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print(f"✓ 使用设备: {device}")

    # ==================== 创建数据集 ====================
    print("\n" + "=" * 70)
    print("创建数据集...")
    print("=" * 70)

    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        participant_masses=config.participant_masses,
        mode=config.mode,
        remove_nan=True,
        remove_any_nan=True,
        activity_flag=config.activity_flag,
        use_participant_mass=config.use_participant_mass,
        min_sequence_length=getattr(config, 'min_sequence_length', -1),
        enable_normalization=getattr(config, 'enable_normalization', False),
        feature_statistics_path=getattr(config, 'feature_statistics_path', None),
        normalization_method=getattr(config, 'normalization_method', 'linear'),
        normalization_params=getattr(config, 'normalization_params', None)
    )

    # 获取数据集信息
    num_classes = dataset.get_num_classes()
    num_features = dataset.get_num_features()
    seq_length = config.diffusion_sequence_length

    print(f"\n✓ 数据集创建完成")
    print(f"  - 模式: {mode if isinstance(config.mode, str) else '/'.join(config.mode)}")
    print(f"  - 试验数量: {len(dataset.trial_names)}")
    print(f"  - 序列数量: {len(dataset)}")
    print(f"  - 类别数量: {num_classes}")
    print(f"  - 特征维度: {num_features}")
    print(f"  - 序列长度: {seq_length}")
    if getattr(config, 'enable_normalization', False):
        print(f"  - 归一化方法: {config.normalization_method}")

    # ==================== 创建模型 ====================
    print("\n" + "=" * 70)
    print("创建模型...")
    print("=" * 70)

    # 创建UNet模型
    model = Unet1D(
        dim=config.dim,
        num_classes=num_classes,
        cond_drop_prob=config.cond_drop_prob,
        dim_mults=config.dim_mults,
        channels=num_features
    )

    print(f"✓ UNet1D模型已创建")
    print(f"  - 基础维度: {config.dim}")
    print(f"  - 维度倍数: {config.dim_mults}")
    print(f"  - 输入通道: {num_features}")
    print(f"  - 类别数量: {num_classes}")
    print(f"  - Cond Drop概率: {config.cond_drop_prob}")

    # 创建扩散模型
    diffusion_model = GaussianDiffusion1D(
            model,
            seq_length=seq_length,
            timesteps=config.timesteps,
            channels=num_features
    )

    print(f"✓ GaussianDiffusion1D模型已创建")
    print(f"  - 扩散步数: {config.timesteps}")
    print(f"  - 序列长度: {seq_length}")

    # ==================== 创建训练器 ====================
    print("\n" + "=" * 70)
    print("创建训练器...")
    print("=" * 70)

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

    print(f"✓ Trainer1D已创建")
    print(f"  - Batch Size: {config.train_batch_size}")
    print(f"  - 学习率: {config.train_lr}")
    print(f"  - 训练步数: {config.train_num_steps}")
    print(f"  - 梯度累积: {config.gradient_accumulate_every}")
    print(f"  - EMA衰减: {config.ema_decay}")
    print(f"  - 保存间隔: {config.save_and_sample_every}")
    print(f"  - 采样数量: {config.num_samples}")
    print(f"  - 结果目录: {config.results_folder}")
    print(f"  - 混合精度: {config.amp}")

    # ==================== 恢复训练（可选）====================
    if args.resume is not None:
        print("\n" + "=" * 70)
        print(f"从检查点恢复训练...")
        print("=" * 70)

        milestone = int(args.resume)
        trainer.load(milestone)
        print(f"✓ 已从milestone {milestone}恢复训练")
        print(f"  - 当前步数: {trainer.step}")

    # ==================== 开始训练 ====================
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    print(f"训练将运行 {config.train_num_steps - trainer.step} 步")
    print(f"每 {config.save_and_sample_every} 步保存一次检查点\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        print("保存当前检查点...")
        milestone = trainer.step // config.save_and_sample_every
        trainer.save(milestone)
        print(f"✓ 检查点已保存: model-{milestone}.pt")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()