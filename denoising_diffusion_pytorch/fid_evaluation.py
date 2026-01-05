import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class FIDEvaluation:
    def __init__(
        self,
        batch_size, # 每次处理多少张图
        dl, # 真实数据的迭代器/数据源
        sampler, # 生成器/采样器对象
        channels=3, # 输入图片通道数
        accelerator=None, # 如果用 accelerate/分布式训练，它可以提供更安全的打印函数
        stats_dir="./results", # 保存/加载真实数据统计量的目录
        device="cuda", # GPU/CPU
        num_fid_samples=50000, # FID 统计使用的样本数
        inception_block_idx=2048, # InceptionV3 提取哪个层的特征维度（2048 最常用）
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        # 你指定的特征维度必须是 InceptionV3 支持的某个输出维度
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        # 根据维度（比如 2048）找到对应的 block index
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        # 构建 InceptionV3 特征提取器，只输出你要的那一层特征
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        # 真实数据集的统计量（均值/协方差）是否已加载或计算完成
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        '''
        给定一批samples（形状一般是[b, c, h, w]），输出对应的Inception特征向量（形状[b, 2048]等）
        '''
        # 如果输入是灰度图（1 通道），就复制成 3 通道，因为 InceptionV3 通常期望 3 通道输入
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        # 设置 InceptionV3 为 eval 模式
        self.inception_v3.eval()
        # 前向计算得到特征
        features = self.inception_v3(samples)[0]

        # 如果特征图不是 1×1 的空间尺寸，就做自适应平均池化到 1×1
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        # 把形状从 [b, d, 1, 1] 变成 [b, d]
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        '''
        负责得到真实数据集的特征均值 m2 和协方差 s2（FID 里 real 的统计量）
        '''
        # 拼出缓存文件路径前缀
        path = os.path.join(self.stats_dir, "dataset_stats")
        # 尝试从 dataset_stats.npz 读取已缓存的统计量
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        # 如果文件不存在或读取失败，就进入重新计算流程
        except OSError:
            # 计算需要从真实数据取多少个 batch 才能凑够 n_samples
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            # 用列表存每个 batch 的特征
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                try:
                    # 从 self.dl 取一个 batch 的真实图片
                    real_samples = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                # 提取 Inception 特征（形状 [b, d]）
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            # 把列表拼成一个大 tensor（形状 [N, d]），搬回 CPU，转成 numpy
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            # 计算真实特征的均值向量（长度 d）
            m2 = np.mean(stacked_real_features, axis=0)
            # 计算真实特征的协方差矩阵（d×d）
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        '''
        计算并返回 FID 分数
        '''
        # 如果真实数据统计量没准备好，就先加载/计算
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        # 把生成器/采样器设成 eval 模式
        self.sampler.eval()
        # 把总样本数拆成若干个 batch 大小的列表
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        for batch in tqdm(batches):
            # 调用采样器生成 batch 张假样本
            fake_samples = self.sampler.sample(batch_size=batch)
            # 提取生成样本的 Inception 特征
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        # 拼接成 [N, d]，搬回 CPU，转 numpy
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        # 计算生成特征均值 m1
        m1 = np.mean(stacked_fake_features, axis=0)
        # 计算生成特征协方差 s1
        s1 = np.cov(stacked_fake_features, rowvar=False)

        # 调用 Frechet Distance 计算函数, 使用生成分布 (m1, s1) 与真实分布 (m2, s2) 的距离, 返回即 FID（越低越好）
        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
