"""
本模块包含实现 DDPM（Denoising Diffusion Probabilistic Models）所需的全部核心组件：
U-Net 主干网络、扩散与采样流程、数据集封装以及训练调度器。使用者只需扩展此文件
即可完成从训练到推理的完整流水线。
"""

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# 常量定义
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# 辅助函数

def exists(x):
    """判断变量是否为 None。"""
    return x is not None

def default(val, d):
    """若 val 存在则返回，否则返回（或惰性求值）备用值 d。"""
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    """将输入统一转换为元组；当为标量时复制 length 次。"""
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    """判断分子是否可被分母整除。"""
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    """恒等映射，常用作占位回调。"""
    return t

def cycle(dl):
    """无限循环遍历 dataloader，便于加速器统一调度。"""
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    """判断某整数是否存在整数平方根。"""
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    """将整数 num 拆分为若干个大小为 divisor 的组，末尾允许余数。"""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    """仅在模式不同的时候，才将 PIL 图像转换为目标通道配置。"""
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 归一化函数

def normalize_to_neg_one_to_one(img):
    """将 [0, 1] 图像数据线性映射到模型期望的 [-1, 1]。"""
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """将 [-1, 1] 还原为 [0, 1]，方便保存或可视化。"""
    return (t + 1) * 0.5

# 小型组件

def Upsample(dim, dim_out = None):
    """先进行最邻近上采样，再用 3x3 卷积混合通道。"""
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    """通过像素重排实现的空间到通道下采样，接着 1x1 卷积压缩通道。"""
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    """二维 RMS 归一化层，逐通道归一并带可学习增益。"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # 可学习增益参数

    def forward(self, x):
        # F.normalize(x, dim=1) 等价于
        # rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-5)  # 计算均方根
        # normalized = x / rms  # 归一化
        # TODO:这里还需要再搞清楚
        return F.normalize(x, dim = 1) * self.g * self.scale

# 正弦位置/时间嵌入

class SinusoidalPosEmb(Module):
    """标准确定性正弦时间步编码，Imagen/DDPM U-Net 的常用配置。"""
    def __init__(self, dim, theta = 10000):
        super().__init__()
        # 嵌入向量的维度
        self.dim = dim
        # 频率缩放因子，控制不同维度的频率变化速率
        self.theta = theta

    def forward(self, x):
        # 获取输入张量所在的设备
        device = x.device
        # 计算一半维度数，用于生成正弦和余弦对
        half_dim = self.dim // 2
        # 计算频率衰减因子，按照指数规律分布
        emb = math.log(self.theta) / (half_dim - 1)
        # 生成递减的频率序列，每个维度有不同的频率
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将输入时间步扩展为列向量，与频率序列相乘得到角度值
        emb = x[:, None] * emb[None, :]
        # 分别计算正弦和余弦值，并沿最后一个维度拼接
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """
    参考 @crowsonkb 的 V-Diffusion，可学习或随机固定时间频率，实现更灵活的正弦嵌入。原实现见：
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim, is_random = False):
        super().__init__()
        # 确保维度是偶数，因为需要分成两半分别计算sin和cos
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        # 初始化权重参数，如果是随机模式则不进行梯度更新
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        # 将输入x从[b]变为[b, 1]以适应后续矩阵运算
        x = rearrange(x, 'b -> b 1')
        # 计算频率：x * weights * 2π，得到[b, d]的频率值
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 对频率分别计算正弦和余弦，并拼接在一起，形成Fourier特征
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # 在Fourier特征前加入原始输入x，增强表达能力
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# 核心网络模块
class Block(Module):
    """卷积 + RMSNorm + SiLU 的小模块，可接受类似 FiLM 的时间条件。"""
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        # 应用卷积操作
        x = self.proj(x)
        # 应用RMS归一化
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # 应用激活函数
        x = self.act(x)
        # 应用dropout并返回结果
        return self.dropout(x)

class ResnetBlock(Module):
    """由两个 Block 组成的 ResNet 残差块，可注入时间信息。"""
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        # 如果存在MLP层和时间嵌入，则计算scale和shift参数
        if exists(self.mlp) and exists(time_emb):
            # 通过MLP处理时间嵌入
            time_emb = self.mlp(time_emb)
            # 重塑时间嵌入张量以匹配图像张量的空间维度
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 将时间嵌入分成两部分：scale和shift
            scale_shift = time_emb.chunk(2, dim = 1)

        # 通过第一个Block处理输入，可选地应用时间条件调制
        h = self.block1(x, scale_shift = scale_shift)

        # 通过第二个Block进一步处理
        h = self.block2(h)

        # 返回残差连接的结果：Block处理结果 + 调整维度的输入
        return h + self.res_conv(x)

class LinearAttention(Module):
    """线性化注意力：softmax 分解 + 可学习记忆 token，降低复杂度。"""
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        # 缩放因子，用于稳定梯度
        self.scale = dim_head ** -0.5
        # 注意力头数
        self.heads = heads
        # 隐藏层总维度
        hidden_dim = dim_head * heads

        # 输入特征归一化
        self.norm = RMSNorm(dim)

        # 可学习的记忆键值对参数
        # 形状: [2, heads, dim_head, num_mem_kv]
        # 第一维2表示键(key)和值(value)两个部分
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        # 生成查询、键、值的卷积层
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        # 输出层：投影回原始维度并归一化
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量，形状为 [batch, channels, height, width]

        Returns:
            经过线性注意力处理后的张量，形状为 [batch, channels, height, width]
        """
        b, c, h, w = x.shape

        # 尺寸为[b,c,h,w]
        x = self.norm(x)

        # 通过卷积层生成查询、键、值，并将其分割为三部分
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 重新排列张量维度以适应多头注意力格式
        # 从 [b, dim_head*heads, h, w] 转换为 [b, heads, dim_head, h*w]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # 扩展可学习的记忆键值对到批次维度
        # 从 [heads, dim_head, num_mem_kv] 转换为 [b, heads, dim_head, num_mem_kv]
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        # 将记忆键值对与计算出的键值连接起来
        # k, v 的最终形状: [b, heads, dim_head, h*w + num_mem_kv]
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        # 线性注意力的关键步骤：对查询和键应用不同的softmax
        # 查询在 dim=-2 上做 softmax（对应 dim_head 维度）
        q = q.softmax(dim = -2)
        # 键在 dim=-1 上做 softmax（对应序列维度）
        k = k.softmax(dim = -1)

        # 应用缩放因子
        q = q * self.scale

        # 计算上下文矩阵: 键和值的点积，相当于K*V^T
        # context 形状: [b, heads, dim_head, dim_head]
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 基于上下文矩阵和查询计算输出，相当于Q^T*(K*V^T)
        # out 形状: [b, heads, dim_head, h*w]
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # 重新排列输出张量维度，恢复到卷积格式
        # 从 [b, heads, dim_head, h*w] 转换为 [b, dim_head*heads, h, w]
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # 通过输出层返回最终结果
        return self.to_out(out)

class Attention(Module):
    """标准自注意力，支持 FlashAttention 与可学习记忆 token。"""
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        """
        初始化Attention模块。

        Args:
            dim: 输入特征维度（通道数）
            heads: 注意力头的数量
            dim_head: 每个注意力头的维度
            num_mem_kv: 可学习记忆token的数量
            flash: 是否使用FlashAttention优化
        """
        self.heads = heads
        hidden_dim = dim_head * heads

        # 对输入进行归一化处理
        self.norm = RMSNorm(dim)
        # 实例化注意力计算模块，支持FlashAttention
        self.attend = Attend(flash = flash)

        # 可学习的记忆键值对参数，2表示键和值两个部分，尺寸为[2, heads, num_mem_kv, dim_head]
        # 作用：为每个注意力头添加可学习的全局记忆单元，增强模型的长期记忆能力
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        # 通过1x1卷积将输入映射为查询、键、值三个部分
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        # 输出投影层，将多头注意力结果映射回原始维度
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量，形状为 [batch, channels, height, width]

        Returns:
            经过多头自注意力处理后的张量，形状为 [batch, channels, height, width]
        """
        # 获取输入张量的形状信息：batch_size, channels, height, width
        b, c, h, w = x.shape

        # 对输入进行归一化
        x = self.norm(x)

        # 通过卷积层生成查询、键、值，并将其分割为三部分，每个部分尺寸为[b, dim_head * heads, h, w]
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 重新排列张量维度，将特征维度分拆到多头注意力格式: [b, heads, h * w, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # 扩展可学习的记忆键值对到批次维度
        # mk,尺寸为[b, heads, num_mem_kv, dim_head],mv,尺寸为[b, heads, num_mem_kv, dim_head]
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        # 将记忆键值对与计算出的键值连接起来
        # partial 创建了一个预设了部分参数的新函数
        # k, 尺寸为[b, heads, h * w + num_mem_kv, dim_head], v, 尺寸为[b, heads, h * w + num_mem_kv, dim_head]
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        #尺寸为[b, heads, h * w, dim_head]
        out = self.attend(q, k, v)

        #尺寸为[b, dim_head * heads, h, w]
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 模型结构
class Unet(Module):
    """
    多尺度 U-Net 主干，集成注意力、自条件、可学习噪声等扩展技巧。
    扩散模型的核心网络架构，基于U-Net设计，具有以下特点：
    1. 多尺度特征提取：通过下采样和上采样构建编码器-解码器结构
    2. 注意力机制：在不同层级集成注意力模块
    3. 时间条件：通过时间嵌入调节网络行为
    4. 自条件：利用先前预测结果改善去噪性能
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # 缺省仅在最深层使用全注意力
        flash_attn = False
    ):
        super().__init__()

        """
        初始化U-Net模型。

        Args:
            dim: 基础特征维度
            init_dim: 初始卷积层维度，默认为dim
            out_dim: 输出维度，默认为channels * (1 if not learned_variance else 2)
            dim_mults: 各层级维度倍数元组
            channels: 输入图像通道数
            self_condition: 是否启用自条件
            learned_variance: 是否学习方差
            learned_sinusoidal_cond: 是否使用可学习正弦位置编码
            random_fourier_features: 是否使用随机傅里叶特征
            learned_sinusoidal_dim: 学习正弦编码维度
            sinusoidal_pos_emb_theta: 正弦位置编码频率参数
            dropout: Dropout概率
            attn_dim_head: 注意力头维度
            attn_heads: 注意力头数量
            full_attn: 全注意力配置
            flash_attn: 是否使用FlashAttention
        """

        # 计算输入输出维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # 列表,包含各层的维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 列表,包含各层的输入输出维度
        in_out = list(zip(dims[:-1], dims[1:]))

        # 时间嵌入
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 注意力配置
        if not full_attn:
            # *((False,) * (len(dim_mults) - 1)) 中的 * 将这个元组解包
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        # (False, False, False, True)
        full_attn  = cast_tuple(full_attn, num_stages)
        # (4, 4, 4, 4)
        attn_heads = cast_tuple(attn_heads, num_stages)
        # (32, 32, 32, 32)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # 预配置常用 block 工厂
        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # 构建每个下采样/上采样阶段
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1) # 是否为最后一层

            # 根据配置选择注意力类型
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            # 添加下采样模块：两个ResnetBlock + 注意力 + 下采样
            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in), # 第一个残差块
                resnet_block(dim_in, dim_in), # 第二个残差块
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads), # 注意力模块
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1) # 下采样
            ]))

        # 构建中间瓶颈层
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim) # 中间第一个残差块
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1]) # 中间注意力
        self.mid_block2 = resnet_block(mid_dim, mid_dim) # 中间第二个残差块

        # 构建上采样路径（解码器）
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1) # 是否为最后一层

            # 根据配置选择注意力类型
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            # 添加上采样模块：两个ResnetBlock + 注意力 + 上采样
            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out), # 第一个残差块（注意输入维度增加了skip connection）
                resnet_block(dim_out + dim_in, dim_out), # 第二个残差块
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads), # 注意力模块
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1) # 上采样
            ]))

        # 输出层配置
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 最终残差块和卷积层
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        """获取总下采样因子（2的幂次）"""
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        """
        前向传播函数。

        Args:
            x: 输入图像张量，形状为 [batch, channels, height, width]
            time: 时间步张量，形状为 [batch]
            x_self_cond: 自条件张量，根据xt和预测值得到的x0，形状为 [batch, channels, height, width]

        Returns:
            去噪后的图像张量，形状为 [batch, out_channels, height, width]
        """
        # 多次下采样要求输入尺寸能被总体下采样因子整除
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # 处理自条件输入
        if self.self_condition:
            # 自条件：上一时刻预测的 x0 参与后续 denoising，默认全零
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1) # 拼接自条件

        # 初始卷积特征提取，尺寸为[b, init_dim, h, w]
        x = self.init_conv(x)
        r = x.clone() # 保存初始特征用于最后的skip connection

        # 时间嵌入处理，尺寸为[b, time_dim]
        t = self.time_mlp(time)

        h = [] # 存储skip connections的特征图

        # 编码器阶段（下采样）
        for block1, block2, attn, downsample in self.downs:
            # 每个阶段双残差块 -> 注意力 -> 下采样，且保存中间特征作 skip connection
            # 尺寸为[b, dim_in, h, w]
            x = block1(x, t) # 第一个残差块
            h.append(x) # 保存特征用于skip connection

            # 尺寸为[b, dim_in, h, w]
            x = block2(x, t) # 第二个残差块
            # 尺寸为[b, dim_in, h, w]
            x = attn(x) + x # 注意力模块（残差连接）
            h.append(x) # 保存特征用于skip connection

            # 尺寸为[b, dim_out, h/2, w/2]
            x = downsample(x) # 下采样

        # 瓶颈层处理
        # 尺寸为[b, mid_dim, h/s, w/s]
        x = self.mid_block1(x, t) # 中间第一个残差块
        # 尺寸为[b, mid_dim, h/s, w/s]
        x = self.mid_attn(x) + x # 中间注意力（残差连接）
        # 尺寸为[b, mid_dim, h/s, w/s]
        x = self.mid_block2(x, t) # 中间第二个残差块

        # 解码器阶段（上采样）
        for block1, block2, attn, upsample in self.ups:
            # 解码阶段逐步拼接 skip feature 并上采样复原空间尺寸，尺寸为[b, dim_out + dim_in, h/2, w/2]
            x = torch.cat((x, h.pop()), dim = 1) # 拼接skip connection特征
            # 尺寸为[b, dim_out, h/2, w/2]
            x = block1(x, t) # 第一个残差块

            # 尺寸为[b, dim_out + dim_in, h/2, w/2]
            x = torch.cat((x, h.pop()), dim = 1) # 拼接skip connection特征
            # 尺寸为[b, dim_out, h/2, w/2]
            x = block2(x, t) # 第二个残差块
            # 尺寸为[b, dim_out, h/2, w/2]
            x = attn(x) + x # 注意力模块（残差连接）

            # 尺寸为[b, dim_in, h, w]
            x = upsample(x) # 上采样

        # 拼接最早的高分辨率表示 r，帮助恢复低层语义，尺寸为[b, init_dim * 2, h, w]
        x = torch.cat((x, r), dim = 1)

        # # 最终残差块和卷积输出，尺寸为[b, init_dim, h, w]
        x = self.final_res_block(x, t)
        return self.final_conv(x)

# 高斯扩散核心实现
def extract(a, t, x_shape):
    """按 batch 索引从缓冲区张量 a 中提取对应时间步，并 reshape 适配目标形状。"""
    b, *_ = t.shape
    # 尺寸为[b]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    线性 beta 退火策略，来自原始 DDPM 论文，用于定义噪声方差。
    Args:
        timesteps: 时间步数，即扩散过程的总步数
    Returns:
        一个形状为[timesteps]的张量，包含每个时间步的beta值
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    余弦调度策略，参考 https://openreview.net/forum?id=-NEXDKk8gZ 提出的改进方案。
    该方法在训练初期添加较少噪声，在后期添加更多噪声，有助于提升训练稳定性。

    Args:
        timesteps: 时间步数
        s: 小的偏移量，避免在t=0时cos值为1导致beta为0

    Returns:
        一个形状为[timesteps]的张量，包含每个时间步的beta值
    """
    # 增加一个额外的步骤以方便计算
    steps = timesteps + 1
    # 生成[0, 1]之间的均匀时间点
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    # 使用余弦函数计算累积alpha值，添加s偏移量避免边界问题
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    # 归一化，使初始累积alpha值为1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # 从累积alpha值推导出每个时间步的beta值
    # 根据公式: alpha_t = alpha_bar_t / alpha_bar_{t-1}
    # 以及: beta_t = 1 - alpha_t
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # 将beta值限制在[0, 0.999]范围内，防止数值不稳定
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    Sigmoid 形状的 beta 调度，灵感来自 https://arxiv.org/abs/2212.11972，
    对 64x64 以上图像训练更稳定。
    Args:
        timesteps: 时间步数
        start: sigmoid函数的起始值
        end: sigmoid函数的结束值
        tau: 温度参数，控制曲线的陡峭程度
        clamp_min: beta值的最小限制值

    Returns:
        一个形状为[timesteps]的张量，包含每个时间步的beta值
    """
    # 增加一个额外的步骤以方便计算
    steps = timesteps + 1
    # 生成[0, 1]之间的均匀时间点
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    # 计算sigmoid函数的起始和结束值
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    # 使用sigmoid函数生成累积alpha值
    # 通过对时间进行线性变换并应用sigmoid函数创建平滑的过渡曲线
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    # 归一化，使初始累积alpha值为1
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # 从累积alpha值推导出每个时间步的beta值
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # 将beta值限制在[clamp_min, 0.999]范围内，防止数值不稳定
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    """
    负责前向扩散与反向生成的整体流程，包括时间调度、噪声预测目标、采样策略
    （DDPM / DDIM）以及诸多训练技巧，如 SNR reweight、自条件、offset noise 等。
    """
    def __init__(
        self,
        model, # 用于预测噪声或图像的神经网络模型（通常为UNet）
        *,
        image_size, # 输入图像的尺寸，可以是整数或元组
        timesteps = 1000, # 扩散过程中的总时间步数
        sampling_timesteps = None, # 采样时使用的时间步数，None表示与训练时相同
        objective = 'pred_v', # 训练目标类型：'pred_noise', 'pred_x0', 'pred_v'
        beta_schedule = 'sigmoid', # β值调度策略：'linear', 'cosine', 'sigmoid'
        schedule_fn_kwargs = dict(), # 调度函数的额外参数
        ddim_sampling_eta = 0., # DDIM采样中的η参数，控制随机性
        auto_normalize = True, # 是否自动将图像从[0,1]归一化到[-1,1]
        offset_noise_strength = 0.,  # 参考 https://www.crosslabs.org/blog/diffusion-with-offset-noise # 偏置噪声强度，参考相关博客文章
        min_snr_loss_weight = False, # 参考 https://arxiv.org/abs/2303.09556 # 是否使用权重平衡不同时间步的损失，参考论文
        min_snr_gamma = 5, # SNR裁剪参数，用于min-SNR-weighting
        immiscible = False # 是否启用不相容扩散（通过噪声重排减少成分混合）
    ):
        super().__init__()
        # 验证模型配置：确保通道数与输出维度匹配
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # 确保模型不使用随机或学习的正弦位置编码
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels # 模型输入通道数
        self.self_condition = self.model.self_condition # 是否启用自条件机制

        # 处理图像尺寸参数，统一为(width, height)元组形式
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        # 设置训练目标类型并验证其有效性
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # 根据配置选择β值调度函数
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 计算每个时间步的β值（噪声方差）
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # 这些扩散系数的递推与 DDPM 原始推导一致
        # 计算扩散过程中的关键系数
        alphas = 1. - betas # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0) # 累积乘积 ᾱ_t = Π_{s=1}^t α_s
        # 前一个时间步的累积乘积，第一个元素设为1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # 保存时间步数
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 采样相关参数

        # 采样步数可单独配置，缺省时与训练步数一致
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # 默认与训练步数一致

        assert self.sampling_timesteps <= timesteps
        # 判断是否使用DDIM采样（采样步数小于训练步数时）
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        # DDIM采样的η参数
        self.ddim_sampling_eta = ddim_sampling_eta

        # 封装 register_buffer，统一转为 float32
        # 定义辅助函数：注册缓冲区并统一转为float32类型
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 注册大量缓冲区以便在推理时自动迁移到正确设备
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 扩散前向链路所需的各种系数
        # 注册扩散前向链路所需的各种系数（用于采样和损失计算）
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 后验 q(x_{t-1} | x_t, x_0) 参数
        # 计算后验分布 q(x_{t-1} | x_t, x_0) 的参数（用于反向采样），参考论文DDPM中公式(7)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # 上式等价于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # 由于扩散链开头方差为 0，log 需做 clamp
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        # 后验均值系数，参考论文DDPM中公式(7)
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 不相容扩散：通过噪声重排减少成分混合
        self.immiscible = immiscible

        # 偏置噪声强度；博文建议 0.1 左右
        self.offset_noise_strength = offset_noise_strength

        # 基于 SNR 推导不同目标的 loss 权重（根据DDPM中xt的表达式得到）
        # 信噪比（SNR，signal-to-noise ratio）
        snr = alphas_cumprod / (1 - alphas_cumprod)

        # 相关论文：https://arxiv.org/abs/2303.09556
        # 该论文提出对 SNR 进行裁剪，缓解早期/晚期时间步梯度过大差异
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        # 根据训练目标类型设置损失权重
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # 自动将 [0,1] 正规化到 [-1,1]，可通过 auto_normalize 关闭
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        """便捷获取已注册缓冲区所在设备。"""
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        """
        依据噪声预测目标还原 x0。
        使用公式: x0 = (xt - sqrt(1-ᾱt) * noise) / sqrt(ᾱt)
        这是从噪声预测目标推导出 x0 的方法

        Args:
            x_t: 时间步 t 的噪声图像
            t: 时间步索引
            noise: 模型预测的噪声

        Returns:
            预测的初始图像 x0
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        已知 x0 反推出噪声项，用于 pred_x0 / pred_v 模式。
        使用公式: noise = (xt - sqrt(ᾱt) * x0) / sqrt(1-ᾱt)

        Args:
            x_t: 时间步 t 的噪声图像
            t: 时间步索引
            x0: 预测的初始图像

        Returns:
            反推出的噪声
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        """
        计算 Imagen 中使用的 v-parameterization。
        使用公式: v = sqrt(ᾱt) * noise - sqrt(1-ᾱt) * x0
        这是一种替代的参数化方法，在某些情况下表现更好

        Args:
            x_start: 初始图像 x0
            t: 时间步索引
            noise: 添加的噪声

        Returns:
            v 参数值
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """
        由 v 参数还原 x0。
        使用公式: x0 = sqrt(ᾱt) * xt - sqrt(1-ᾱt) * v

        Args:
            x_t: 时间步 t 的噪声图像
            t: 时间步索引
            v: v 参数值

        Returns:
            还原的初始图像 x0
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """计算 q(x_{t-1}|x_t,x0) 的均值与方差，用于采样推导。"""
        # 参考论文DDPM中公式(7)
        # q(xt-1|xt,x0)对应的均值，尺寸为 [b,c,h,w]
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # q(xt-1|xt,x0)对应的方差，尺寸为 [b,c,h,w]
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        """
        调用底层 U-Net 得到输出，并根据训练目标统一转成噪声预测和 x0 预测。
        clip_x_start/rederive_pred_noise 主要用于 DDIM 采样的数值稳定。
        Args:
            x: 输入的噪声图像
            t: 时间步索引
            x_self_cond: 自条件输入
            clip_x_start: 是否裁剪 x0 到 [-1, 1] 范围
            rederive_pred_noise: 是否重新推导噪声预测

        Returns:
            ModelPrediction 命名元组，包含 pred_noise 和 pred_x_start
        """
        # 调用模型获得输出，尺寸为[b,c,h,w]
        model_output = self.model(x, t, x_self_cond)
        # 定义裁剪函数
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            # 尺寸为[b,c,h,w]
            pred_noise = model_output
            # 从噪声预测推导出 x0,尺寸为[b,c,h,w]
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            # 如果需要重新推导噪声（用于数值稳定）
            if clip_x_start and rederive_pred_noise:
                # 尺寸为[b,c,h,w]
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            # 如果目标是预测 x0,尺寸为[b,c,h,w]
            x_start = model_output
            x_start = maybe_clip(x_start)
            # 从 x0 推导出噪声，尺寸为[b,c,h,w]
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            # 如果目标是预测 v 参数,尺寸为[b,c,h,w]
            v = model_output
            # 尺寸为[b,c,h,w]
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        """计算反向一步的均值/方差，若需要可裁剪 x0 防止自激发。"""
        # 根据当前xt、t和根据上一个时刻xt预测得到的x0得到预测的噪声和x0
        preds = self.model_predictions(x, t, x_self_cond)
        # 尺寸为[b,c,h,w]
        x_start = preds.pred_x_start

        # 限制当前得到的x0的范围
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # 根据预测得到的x0和当前的xt，计算 q(x_{t-1}|x_t,x0) 的均值、方差和对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        """
        执行单步随机采样（DDPM），返回下一个样本与最新的 x0 估计。
        给定当前时刻的噪声图像 x_t，
        根据模型预测的均值和方差，
        从 p(x_{t-1} | x_t) 中采样得到 x_{t-1}。

        参数说明：
        ----------
        x : Tensor
            当前时间步的样本 x_t，形状为 (B, C, H, W)

        t : int
            当前时间步（标量，范围 [0, T-1]）

        x_self_cond : Tensor or None
            self-conditioning 使用的 x_0 估计（可选）

        返回：
        ------
        pred_img : Tensor
            采样得到的 x_{t-1}

        x_start : Tensor
            当前模型预测的 x_0（干净图像估计）
        """
        b, *_, device = *x.shape, self.device
        # 尺寸为[b]
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        # 计算 q(x_{t-1}|x_t,x0) 的均值、方差和对数方差并得到预测的x0
        # p_mean_variance 会根据当前 x_t 和时间步 t：
        #   - 调用神经网络预测噪声 / x_0 / v
        #   - 计算反向分布 p(x_{t-1} | x_t) 的：
        #       * 均值 model_mean
        #       * 方差 model_variance（此处未用）
        #       * 对数方差 model_log_variance
        #   - 同时返回模型预测的 x_start（即 x_0）
        #
        # clip_denoised=True：
        #   将预测的 x_0 裁剪到合法范围（如 [-1, 1]），
        #   防止数值爆炸，提升采样稳定性
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        # 按 DDPM 公式：
        #   x_{t-1} = mean + std * noise
        #
        # 当 t > 0：
        #   从 N(0, I) 采样噪声，保持随机性
        #
        # 当 t == 0：
        #   不再加噪声（直接输出均值）
        #   因为这是最终生成的图像 x_0
        noise = torch.randn_like(x) if t > 0 else 0. # 当 t=0 时不再注入噪声
        # 参考论文DDPM中Algorithm 2
        # std = exp(0.5 * log_variance)
        # 使用 log_variance 而不是 variance，
        # 是为了数值稳定（避免直接预测/存储方差）
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # 返回：
        #   pred_img -> 下一时间步的样本 x_{t-1}
        #   x_start  -> 当前预测的 x_0（供 self-conditioning 或可视化）
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        """
        标准 DDPM 逐步采样循环，可选返回每个时间步生成的图像。
        从纯高斯噪声 x_T 开始，按时间步 t = T-1, T-2, ..., 0 逐步反向去噪，最终得到模型生成的图像 x_0。

        参数说明：
        ----------
        shape : tuple
            采样张量的形状，一般为 (batch_size, channels, height, width)

        return_all_timesteps : bool
            是否返回所有时间步的中间结果：
            - False（默认）：只返回最终生成的图像 x_0
            - True：返回形状为 (batch, num_timesteps+1, C, H, W) 的张量
        """
        batch, device = shape[0], self.device

        # 从标准正态分布采样，作为扩散过程的起点 x_T
        # 尺寸为[b,c,h,w]
        img = torch.randn(shape, device = device)
        # 用于存储每个时间步生成的图像
        # 初始时存入 x_T
        imgs = [img]

        # x_start 用于 self-conditioning（自条件）
        # 表示模型预测的 x_0（干净图像）
        # 第一次迭代时还不存在，因此设为 None
        x_start = None

        # 从 T-1 到 0 反向遍历所有时间步
        # reversed(range(...))：时间步倒序
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # 如果启用了 self-conditioning：
            #   将上一步预测的 x_start（即 x_0 估计）
            #   作为额外条件输入模型
            # 否则为 None
            self_cond = x_start if self.self_condition else None
            # 执行单步随机采样,得到xt-1和根据xt预测得到的x0
            # p_sample 执行一步 DDPM 反向采样：
            #   输入：
            #     img       -> 当前时刻的 x_t
            #     t         -> 当前时间步
            #     self_cond -> 可选的自条件 x_0 预测
            #
            #   输出：
            #     img       -> 采样得到的 x_{t-1}
            #     x_start   -> 模型预测的 x_0
            img, x_start = self.p_sample(img, t, self_cond)
            # 保存当前时间步的结果
            imgs.append(img)

        # 在return_all_timesteps为True时返回从xt到x0阶段生成的图像，尺寸为[b,T+1,c,h,w],return_all_timesteps为False时返回最终生成的图像，尺寸为[b,c,h,w]
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # 将图像范围从[-1,1]反归一化为[0,1]
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        """DDIM 采样实现，可用更少步数近似原始过程。"""
        # 基本采样相关参数
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 构造采样时间序列（当 sampling_timesteps == total_timesteps 时等价于完整 DDPM）
        # 尺寸为[N_sampling]
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # 从标准高斯噪声初始化 x_T
        # 尺寸为[b,c,h,w]
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None # self-conditioning 使用的 x_0 预测

        # 反向 DDIM 采样循环
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # 当前时间步构造成 batch 对齐的张量，尺寸为[b]
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # 根据xk预测得到的x0
            self_cond = x_start if self.self_condition else None
            # 得到预测噪声和根据当前xk预测得到的x0
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            # 若到达最终步（time_next == -1），直接输出 x_0
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            # 当前与下一时间步的 alpha 累积值
            # αk,尺寸为[b]
            alpha = self.alphas_cumprod[time]
            # αs(s<k),尺寸为[b]
            alpha_next = self.alphas_cumprod[time_next]

            # 参考论文DDIM中公式(16)
            # DDIM 中控制随机性的噪声尺度（eta=0 时为确定性采样），尺寸为[b]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # 参考论文DDIM中公式(12)
            # 尺寸为[b]
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # 采样噪声（仅在 eta > 0 时起作用），尺寸为[b,c,h,w]
            noise = torch.randn_like(img)

            # 参考论文DDIM中公式(12)，根据 DDIM 更新公式计算 x_{t_next}
            # 根据xk预测得到的x0和预测得到的噪声得到xs，尺寸为[b,c,h,w]
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img) # 保存当前时间步结果

        # 是否返回所有时间步结果
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # 将图像范围从[-1,1]反归一化为[0,1]
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        """根据配置选择 DDPM 或 DDIM 采样接口，对外提供统一入口。"""
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        """在扩散空间内对两幅图像进行线性插值并反向生成。"""
        # b是batch大小；device是张量所在设备（GPU/CPU）
        b, *_, device = *x1.shape, x1.device
        # 若未指定t，则默认用最大时间步（噪声最强）
        t = default(t, self.num_timesteps - 1)

        # 确保两张图形状一致，才能逐元素插值
        assert x1.shape == x2.shape

        # 构造形状为 [B] 的时间步张量，batch 中每个样本使用相同的 t
        t_batched = torch.full((b,), t, device = device)
        # 将 x1 和 x2 正向扩散到时间步 t，得到对应的带噪版本 xt1、xt2
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        # 在噪声空间 xt 上进行线性插值：lam=0 得到 xt1，lam=1 得到 xt2
        img = (1 - lam) * xt1 + lam * xt2

        # 初始化 self-conditioning 需要的 x_start（通常是上一轮预测的 x0）
        x_start = None

        # 从时间步 t-1 反向采样到 0，逐步去噪生成最终插值图
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            # 若启用 self-conditioning，则把上一轮的 x_start 作为条件输入；否则不使用
            self_cond = x_start if self.self_condition else None
            # 执行一步反向扩散采样：更新 img，并返回本步预测的 x_start 供下一步 self-conditioning 使用
            img, x_start = self.p_sample(img, i, self_cond)
        # 返回最终生成的插值结果图像
        return img

    def noise_assignment(self, x_start, noise):
        """若开启 immiscible 模式，使用匈牙利算法匹配噪声重排序。"""
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        """
        根据前向扩散定义生成 x_t，可传入自定义噪声。
        Args:
        x_start: 初始图像
        t: 时间步索引
        noise: 自定义噪声

        Returns:
            时间步 t 的噪声图像
        """
        # 尺寸为[b,c,h,w]
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 如果启用 immiscible 模式，重新排序噪声
        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        # 前向扩散公式: xt = sqrt(ᾱt) * x0 + sqrt(1-ᾱt) * noise
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        """
        计算训练损失，融合 offset noise、自条件与 SNR reweight。
        Args:
        x_start: 初始图像
        t: 时间步索引
        noise: 自定义噪声
        offset_noise_strength: 偏置噪声强度

        Returns:
            计算得到的损失值
        """
        b, c, h, w = x_start.shape

        # 尺寸为[b,c,h,w]
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 处理偏置噪声
        # 偏置噪声技巧可参考 https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            # 生成偏置噪声并添加到原始噪声中
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # 前向扩散得到 x_t，尺寸为[b,c,h,w]
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 若启用 self-conditioning，则以 50% 概率使用当前估计 x0 作为附加输入
        # 虽会增加计算，但可显著降低 FID
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # 根据xt和预测结果获得预测的x0，尺寸为[b,c,h,w]
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # 计算模型输出，尺寸为[b,c,h,w]
        model_out = self.model(x, t, x_self_cond)

        # 根据训练目标设置目标值
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            # 尺寸为[b,c,h,w]
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 计算 MSE 损失
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 应用损失权重（基于 SNR）
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Trainer/Accelerator 调用的统一入口，对输入图像随机采样 t 并计算损失。
        Args:
        img: 输入图像
        *args, **kwargs: 其他参数

        Returns:
            计算得到的损失值
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 归一化图像，从[0,1]到[-1,1]
        img = self.normalize(img)
        # 计算损失
        return self.p_losses(img, t, *args, **kwargs)

# 数据集封装
class Dataset(Dataset):
    """最小封装的数据集：递归遍历文件夹，按需增强、裁剪并转成张量。"""
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # 1. 遍历所有扩展名 (exts列表)
        # 2. 对每个扩展名递归查找文件 (**/*.{ext})
        # 3. 将所有找到的文件路径收集到列表中
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 预处理顺序：可选模式转换 -> resize -> 随机翻转 -> 中心裁剪 -> ToTensor
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn), # 图像模式转换
            T.Resize(image_size), # 尺寸缩放
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(), # 随机水平翻转
            T.CenterCrop(image_size), # 中心裁剪
            T.ToTensor() # 转成张量
        ])

    def __len__(self):
        """返回图像总数。"""
        return len(self.paths)

    def __getitem__(self, index):
        """读取单张图片并应用预处理管线。"""
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# 训练器
class Trainer:
    """整合数据、优化器、EMA、采样与评估逻辑的高阶训练脚手架。"""
    def __init__(
        self,
        diffusion_model, # 传入的扩散模型实例（已经定义好 UNet + 调度逻辑）
        folder, # 训练数据所在的图像文件夹路径
        *,
        train_batch_size = 16, # 每次迭代的 batch 大小（单次前向里实际使用的 batch）
        gradient_accumulate_every = 1, # 梯度累积的次数（用于等效放大 batch size）
        augment_horizontal_flip = True, # 是否对图像做随机水平翻转增强
        train_lr = 1e-4, # 学习率
        train_num_steps = 100000, # 总训练步数（以优化器 step 次数计）
        ema_update_every = 10, # 每隔多少个 step 更新一次 EMA 权重
        ema_decay = 0.995, # EMA 衰减系数（越接近 1，更新越平滑）
        adam_betas = (0.9, 0.99), # Adam 优化器的 betas 超参数
        save_and_sample_every = 1000, # 每多少 step 进行一次采样并保存模型
        num_samples = 25, # 每次采样要生成多少张图片
        results_folder = './results', # 训练结果（模型权重、采样图片等）保存目录
        amp = False, # 是否启用自动混合精度训练（Automatic Mixed Precision）
        mixed_precision_type = 'fp16', # 混合精度类型（fp16 / bf16）
        split_batches = True, # accelerate 是否在多 GPU 间拆分 batch
        convert_image_to = None, # 将 PIL 图像转换成什么模式（L/RGB/RGBA），None 则根据通道数推断
        calculate_fid = True, # 是否在训练过程中计算 FID 指标
        inception_block_idx = 2048, # Inception 网络中用于提取特征的 block 索引
        max_grad_norm = 1., # 梯度裁剪的最大范数（避免梯度爆炸）
        num_fid_samples = 50000, # 计算 FID 时要生成的样本量（通常越多越稳定）
        save_best_and_latest_only = False # 是否只保存 FID 最优和最新的模型，而不是每个 milestone 都存
    ):
        super().__init__()

        # --------------------------- 加速器初始化 ---------------------------
        # Accelerator 负责：
        #   - 多卡训练（分布式 / 数据并行）
        #   - 混合精度训练（fp16 / bf16）
        #   - 自动处理梯度同步、设备迁移等繁琐细节
        # 加速器负责多卡/混合精度调度
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # 模型引用与通道配置
        self.model = diffusion_model
        self.channels = diffusion_model.channels # 图像通道数（1/3/4）
        is_ddim_sampling = diffusion_model.is_ddim_sampling # 是否使用 DDIM 采样（影响 FID 采样速度）

        # 若用户未指定 convert_image_to，依据通道数推断默认值
        # 1 通道 -> 灰度 'L', 3 通道 -> 'RGB', 4 通道 -> 'RGBA'
        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # 采样与训练超参数
        # 验证采样图像数量是否有整数平方根（用于网格显示）
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # 确保有效批处理大小至少为16，以获得较好的训练效果
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size # 模型规定的输入图像尺寸（方形）

        self.max_grad_norm = max_grad_norm # 用于 clip_grad_norm_ 的阈值

        # 数据集与 DataLoader
        # 创建数据集实例
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        # 确保数据集足够大
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
        # 标准 DataLoader：shuffle 打乱，pin_memory 提升主机到 GPU 的拷贝效率
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # 使用accelerator准备数据加载器，使其支持多GPU和混合精度
        dl = self.accelerator.prepare(dl)
        # cycle(dl)：将 dataloader 包装成无限生成器，一直循环数据
        self.dl = cycle(dl)

        # 优化器
        # 标准 Adam 优化器，参数来自扩散模型
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # 定期保存模型 / 采样结果
        # 只有主进程（rank 0）负责维护 EMA 和保存模型，避免多进程重复写文件
        if self.accelerator.is_main_process:
            # 创建指数移动平均模型
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            # 将EMA模型移动到相应设备
            self.ema.to(self.device)

        # 创建结果保存目录
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # 记录当前训练步数
        self.step = 0

        # 使用 accelerator.prepare 适配模型与优化器到正确设备 / 分布式环境
        # 之后 self.model / self.opt 应该只通过 accelerator 调用
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # 可选 FID 评估模块
        # 只有在主进程且启用了 calculate_fid 时才真正计算 FID
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            # 延迟导入，避免未用到时占用资源或造成依赖问题
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            # 若未使用 DDIM 采样，提醒 FID 计算可能会非常慢
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            # FID 评分器：内部会从真实数据 + 生成器采样，计算统计量并得到 FID
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size, # 每次喂给 Inception 网络的 batch 大小
                dl=self.dl, # 真实数据 dataloader，用于计算真实分布统计
                sampler=self.ema.ema_model, # 用 EMA 模型作为生成器
                channels=self.channels, # 图像通道数
                accelerator=self.accelerator, # 共享同一个 accelerator
                stats_dir=results_folder, # 存放 FID 统计文件（.npz）的目录
                device=self.device, # 运行设备
                num_fid_samples=num_fid_samples, # 生成样本数
                inception_block_idx=inception_block_idx # Inception 特征层索引
            )

        # 若只想保留“最优 FID 模型”和“最新模型”，则需要有 FID 作为评价指标
        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # 作为“无穷大”起始值

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        """快捷获取 accelerator 当前设备。"""
        return self.accelerator.device

    def save(self, milestone):
        """在主进程保存检查点，包括模型/EMA/优化器/Scaler。"""
        # 只有 local_main_process 执行保存，避免多卡同时写入
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step, # 当前训练步数
            'model': self.accelerator.get_state_dict(self.model), # 从 accelerator 中取出“干净”的模型权重
            'opt': self.opt.state_dict(), # 优化器状态（包含动量等）
            'ema': self.ema.state_dict(), # EMA 模型权重
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None, # 混合精度的 GradScaler 状态（若存在）
            'version': __version__ # 代码版本号，方便兼容性检查
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        """从磁盘加载指定里程碑权重，并恢复优化器与 EMA。"""
        accelerator = self.accelerator
        device = accelerator.device

        # 加载 checkpoint，map_location 确保可以在当前设备上加载
        # weights_only=True 表示只加载权重张量，有助于安全和速度
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        # unwrap_model：从 accelerator 封装中取出原始模型（去掉 DDP / FP16 包装）
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model']) # 恢复模型参数

        # 恢复 step 和优化器状态
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # 只在主进程恢复 EMA（EMA 只在主进程真正使用）
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        # 打印版本信息（若存在）
        if 'version' in data:
            print(f"loading from version {data['version']}")

        # 若使用了混合精度，并且保存时也有 scaler，则恢复其状态
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        """标准训练循环：梯度累积、EMA、定期采样与可选 FID。"""
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 包装训练进度条
        # initial=self.step 支持从中途恢复训练
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            # 主训练循环，直到 step 达到 train_num_steps
            while self.step < self.train_num_steps:
                self.model.train() # 切换到训练模式（启用 dropout / BN 统计等）

                total_loss = 0. # 用于统计当前 step 内（含梯度累积）的 loss 和

                # --------------------------- 梯度累积 ---------------------------
                # 通过在多个 mini-batch 上累积梯度，再统一反向更新一次
                # 可达到“更大 batch 训练”的效果，而不需要增加显存
                for _ in range(self.gradient_accumulate_every):
                    # 从无限循环的 dataloader 中取出一个 batch，并移动到目标 device
                    data = next(self.dl).to(device)

                    # autocast：在混合精度下，自动推断使用 FP16 / FP32
                    with self.accelerator.autocast():
                        # 扩散模型通常将“输入 x -> 返回 loss”
                        loss = self.model(data)
                        # 除以累积次数，等价于对多个 batch 的 loss 取平均
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item() # 记录到 total_loss 统计值中

                    # 通过 accelerator.backward 统一处理分布式和混合精度的反向传播
                    self.accelerator.backward(loss)

                # 更新 tqdm 进度条头部显示当前 loss
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程同步，保证梯度等状态一致
                accelerator.wait_for_everyone()
                # 对模型参数做梯度裁剪，避免梯度爆炸
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # 优化器更新参数 + 清空梯度
                self.opt.step()
                self.opt.zero_grad()

                # 再次同步，确保所有进程在同一训练阶段
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    # 主进程更新 EMA 权重
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        # 切换 EMA 模型到 eval 模式（关掉 dropout / BN 训练行为）
                        self.ema.ema_model.eval()

                        with torch.inference_mode(): # 关闭梯度，加速推理
                            # 计算当前是第几个 milestone（从 1 开始）
                            milestone = self.step // self.save_and_sample_every
                            # 将 num_samples 拆成若干批（因为单次采样 batch_size 受显存限制）
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            # 对每个批次调用 EMA 模型的 sample 函数，生成图像，尺寸为[b,c,h,w]
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # 拼接所有生成图片为一个大 tensor：[N, C, H, W]
                        all_images = torch.cat(all_images_list, dim = 0)

                        # 保存为网格形式的 png 图像，nrow 为 sqrt(num_samples)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # 是否需要计算 FID
                        if self.calculate_fid:
                            # 使用事先构造好的 FID scorer 计算当前模型的 FID
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        # 模型保存策略
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            # 同时更新 "latest" 以记录最近一次训练状态
                            self.save("latest")
                        else:
                            # 否则按照 milestone 递增地保存（model-1.pt, model-2.pt, ...）
                            self.save(milestone)

                # 每完成一个 step（包含梯度累积+一次参数更新），进度条 +1
                pbar.update(1)

        # 所有训练完成后，在主进程打印提示
        accelerator.print('training complete')
