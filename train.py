"""
扩散模型训练脚本

使用Classifier-Free Guidance的1D扩散模型训练生物力学序列数据
"""

import os
import sys
import argparse
import importlib
import torch
import torch.distributed as dist
import numpy as np
import random
import shutil
from datetime import datetime
from pathlib import Path

# 导入自定义模块
from dataset_loaders.sequence_dataloader import print_rank_0
from dataset_loaders.sequence_dataloader import DiffusionSequenceDataset
from denoising_diffusion_pytorch.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D, Trainer1D

# 设置 CUDA_VISIBLE_DEVICES 来指定可见的 GPU（例如 GPU 1 和 GPU 2）
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def get_rank():
    """获取当前进程的rank，支持Accelerate和PyTorch原生分布式训练"""
    # 方法1: 检查Accelerate环境变量（优先级最高）
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'RANK' in os.environ:
        return int(os.environ['RANK'])
    # 方法2: 检查PyTorch原生分布式
    elif dist.is_initialized():
        return dist.get_rank()
    # 方法3: 默认为rank 0（单进程训练）
    else:
        return 0

def is_main_process():
    """检查当前进程是否为主进程"""
    return get_rank() == 0

class Logger:
    """同时输出到控制台和文件的日志记录器"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 实时写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backup_config_file(config_name, results_folder):
    """备份配置文件到结果目录

    Args:
        config_name: 配置文件模块名
        results_folder: 结果保存目录的Path对象
    """
    try:
        config_module = importlib.import_module(config_name)
        if hasattr(config_module, '__file__') and config_module.__file__:
            config_source = config_module.__file__
            config_dest = results_folder / "config.py"
            shutil.copy2(config_source, config_dest)
            print_rank_0(f"✓ 配置文件已备份: {config_dest}")
        else:
            print_rank_0(f"⚠ 无法找到配置文件路径，跳过备份")
    except Exception as e:
        print_rank_0(f"⚠ 备份配置文件时出错: {e}")


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
    print_rank_0("=" * 70)
    print_rank_0("加载配置文件...")
    print_rank_0("=" * 70)

    config = importlib.import_module(args.config)
    print_rank_0(f"✓ 配置文件已加载: {args.config}")

    # ==================== 创建结果目录并设置日志 ====================
    # 只在主进程创建结果目录和日志文件
    if is_main_process():
        results_folder = Path(config.results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)

        # 设置日志文件（使用时间戳避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_folder / f"training_log_{timestamp}.txt"

        # 重定向stdout到Logger
        sys.stdout = Logger(log_file)
        print_rank_0(f"✓ 日志文件已创建: {log_file}")
    else:
        # 非主进程也需要知道results_folder路径
        results_folder = Path(config.results_folder)

    # ==================== 备份配置文件 ====================
    if is_main_process():
        backup_config_file(args.config, results_folder)

    # ==================== 设置随机种子 ====================
    set_seed(config.random_seed)
    print_rank_0(f"✓ 随机种子已设置: {config.random_seed}")

    # 移除手动GPU设置，让Accelerate管理
    # # ==================== 设置设备 ====================
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print(f"✓ 使用设备: {device}")

    # ==================== 创建数据集 ====================
    print_rank_0("\n" + "=" * 70)
    print_rank_0("创建数据集...")
    print_rank_0("=" * 70)

    dataset = DiffusionSequenceDataset(
        data_dir=config.data_dir,
        input_names=config.input_names,
        label_names=config.label_names,
        side=config.side,
        diffusion_sequence_length=config.diffusion_sequence_length,
        action_patterns=config.action_patterns,
        selected_action_indices=config.selected_action_indices,
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

    print_rank_0(f"\n✓ 数据集创建完成")
    print_rank_0(f"  - 模式: {config.mode if isinstance(config.mode, str) else '/'.join(config.mode)}")
    print_rank_0(f"  - 试验数量: {len(dataset.trial_names)}")
    print_rank_0(f"  - 序列数量: {len(dataset)}")
    print_rank_0(f"  - 类别数量: {num_classes}")
    print_rank_0(f"  - 特征维度: {num_features}")
    print_rank_0(f"  - 序列长度: {seq_length}")
    if getattr(config, 'enable_normalization', False):
        print_rank_0(f"  - 归一化方法: {config.normalization_method}")

    # ==================== 创建模型 ====================
    print_rank_0("\n" + "=" * 70)
    print_rank_0("创建模型...")
    print_rank_0("=" * 70)

    # 创建UNet模型
    model = Unet1D(
        dim=config.dim,
        num_classes=num_classes,
        cond_drop_prob=config.cond_drop_prob,
        dim_mults=config.dim_mults,
        channels=num_features
    )

    print_rank_0(f"✓ UNet1D模型已创建")
    print_rank_0(f"  - 基础维度: {config.dim}")
    print_rank_0(f"  - 维度倍数: {config.dim_mults}")
    print_rank_0(f"  - 输入通道: {num_features}")
    print_rank_0(f"  - 类别数量: {num_classes}")
    print_rank_0(f"  - Cond Drop概率: {config.cond_drop_prob}")

    # 创建扩散模型
    diffusion_model = GaussianDiffusion1D(
            model,
            seq_length=seq_length,
            timesteps=config.timesteps,
            channels=num_features
    )

    print_rank_0(f"✓ GaussianDiffusion1D模型已创建")
    print_rank_0(f"  - 扩散步数: {config.timesteps}")
    print_rank_0(f"  - 序列长度: {seq_length}")

    # ==================== 创建训练器 ====================
    print_rank_0("\n" + "=" * 70)
    print_rank_0("创建训练器...")
    print_rank_0("=" * 70)

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
        log_file=log_file if is_main_process() else None,
        amp=config.amp
    )

    print_rank_0(f"✓ Trainer1D已创建")
    print_rank_0(f"  - Batch Size: {config.train_batch_size}")
    print_rank_0(f"  - 学习率: {config.train_lr}")
    print_rank_0(f"  - 训练步数: {config.train_num_steps}")
    print_rank_0(f"  - 梯度累积: {config.gradient_accumulate_every}")
    print_rank_0(f"  - EMA衰减: {config.ema_decay}")
    print_rank_0(f"  - 保存间隔: {config.save_and_sample_every}")
    print_rank_0(f"  - 采样数量: {config.num_samples}")
    print_rank_0(f"  - 结果目录: {config.results_folder}")
    print_rank_0(f"  - 混合精度: {config.amp}")

    # ==================== 恢复训练（可选）====================
    if args.resume is not None:
        print_rank_0("\n" + "=" * 70)
        print_rank_0(f"从检查点恢复训练...")
        print_rank_0("=" * 70)

        milestone = int(args.resume)
        trainer.load(milestone)
        print_rank_0(f"✓ 已从milestone {milestone}恢复训练")
        print_rank_0(f"  - 当前步数: {trainer.step}")

    # ==================== 开始训练 ====================
    print_rank_0("\n" + "=" * 70)
    print_rank_0("开始训练...")
    print_rank_0("=" * 70)
    print_rank_0(f"训练将运行 {config.train_num_steps - trainer.step} 步")
    print_rank_0(f"每 {config.save_and_sample_every} 步保存一次检查点\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print_rank_0("\n\n训练被用户中断")
        print_rank_0("保存当前检查点...")
        milestone = trainer.step // config.save_and_sample_every
        trainer.save(milestone)
        print_rank_0(f"✓ 检查点已保存: model-{milestone}.pt")

    print_rank_0("\n" + "=" * 70)
    print_rank_0("训练完成！")
    print_rank_0("=" * 70)
    # ==================== 关闭日志文件 ====================
    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal  # 恢复原始stdout


if __name__ == '__main__':
    main()