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
from torch.amp import autocast

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern=None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim=-1)

    parallel = (x * unit).sum(dim=-1, keepdim=True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    """卷积 + RMSNorm + SiLU 的小模块。"""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """由两个 Block 组成的 ResNet 残差块，可注入时间信息和类别信息。"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            # 把 iterable 中不满足条件的元素过滤掉
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            # 通过MLP处理条件嵌入
            cond_emb = self.mlp(cond_emb)
            # 重塑时间嵌入张量以匹配数据张量的维度
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            # 沿特征维将条件嵌入分成两部分：scale和shift
            scale_shift = cond_emb.chunk(2, dim=1)

        # 通过第一个Block处理输入，可选地应用条件嵌入调制
        h = self.block1(x, scale_shift=scale_shift)

        # 通过第二个Block进一步处理
        h = self.block2(h)

        # 返回残差连接的结果：Block处理结果 + 调整维度的输入
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """线性化注意力：softmax 分解 + 可学习记忆 token，降低复杂度。"""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        # 缩放因子，用于稳定梯度
        self.scale = dim_head ** -0.5
        # 注意力头数
        self.heads = heads
        # 隐藏层总维度
        hidden_dim = dim_head * heads
        # 生成查询、键、值的卷积层
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        # 输出层：投影回原始维度并归一化
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入张量，形状为 [batch, channels, length]

        Returns:
            经过线性注意力处理后的张量，形状为 [batch, channels, length]
        """
        b, c, n = x.shape
        # 通过卷积层生成查询、键、值，并将其分割为三部分
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # 重新排列张量维度以适应多头注意力格式
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        # 计算上下文矩阵: 键和值的点积，相当于K*V^T
        # context 形状: [b, heads, dim_head, dim_head]
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 基于上下文矩阵和查询计算输出，相当于(K*V^T)^T*Q
        # out 形状: [b, heads, dim_head, n]
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # 重新排列输出张量维度，恢复到卷积格式
        # 从 [b, heads, dim_head, n] 转换为 [b, dim_head*heads, n]
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        # 通过输出层返回最终结果
        return self.to_out(out)


class Attention(nn.Module):
    """标准自注意力，支持 FlashAttention 与可学习记忆 token。"""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        # 通过卷积层生成查询、键、值，并将其分割为三部分，每个部分尺寸为[b, dim_head * heads, n]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # 重新排列张量维度，将特征维度分拆到多头注意力格式: [b, heads, dim_head, n]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        # 计算查询和键的相似度矩阵 (Q^T*K)
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        # 应用softmax获得注意力权重
        attn = sim.softmax(dim=-1)
        # 使用注意力权重聚合值向量 (AV^T),相当于(Q^T*K)*V^T
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # 尺寸为[b, dim_head * heads, n]
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


# model

class Unet1D(nn.Module):
    def __init__(
            self,
            dim,
            num_classes,
            cond_drop_prob=0.5,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            attn_dim_head=32,
            attn_heads=4
    ):
        super().__init__()

        self.num_classes = num_classes

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        # 计算输入输出维度
        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        # 列表,包含各层的维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 列表,包含各层的输入输出维度
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        # 时间嵌入
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        # 类别嵌入
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        # 构建每个下采样/上采样阶段
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)  # 是否为最后一层

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            remove_parallel_component=True,
            keep_parallel_frac=0.,
            **kwargs
    ):
        """
        使用条件缩放进行前向传播，实现分类器自由指导(Classifier-Free Guidance)

        这个函数通过比较有条件和无条件的模型输出来增强生成质量。
        它支持多种改进技术，包括CFG++和平行分量移除。

        Args:
            *args: 传递给基础forward函数的位置参数
            cond_scale: 条件缩放因子，控制条件引导的强度
                       - 1.0: 不使用条件引导，返回原始输出
                       - >1.0: 增强条件引导效果
            rescaled_phi: 用于rescaled classifier-free guidance的插值参数
                         - 0.0: 不使用rescaled CFG
                         - >0.0: 应用rescaled CFG，值越大rescaled效果越强
            remove_parallel_component: 是否移除更新向量中与原始输出平行的分量
            keep_parallel_frac: 保留平行分量的比例(0.0-1.0)
            **kwargs: 传递给基础forward函数的关键字参数

        Returns:
            如果rescaled_phi为0: 返回(scaled_logits, null_logits)
            如果rescaled_phi>0: 返回(interpolated_rescaled_logits, null_logits)
        """
        # 获取有条件预测的结果 (cond_drop_prob=0 表示不丢弃条件信息)
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        # 如果条件缩放因子为1，不应用任何指导，直接返回原始输出
        if cond_scale == 1:
            return logits

        # 获取无条件预测的结果 (cond_drop_prob=1 表示完全丢弃条件信息)
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        # 计算条件引导的更新向量: 有条件输出 - 无条件输出
        # 这个更新向量代表了条件信息对输出的影响
        update = logits - null_logits

        # 如果启用移除平行分量功能(CFG++)
        if remove_parallel_component:
            # 将更新向量分解为平行和正交两个分量
            # parallel: 与原始logits平行的分量
            # orthog: 与原始logits正交的分量
            parallel, orthog = project(update, logits)
            # 重新组合更新向量，只保留正交分量和部分平行分量
            # 这是CFG++的核心思想，减少过度优化问题
            update = orthog + parallel * keep_parallel_frac

        # 应用条件缩放: 原始输出 + 缩放后的更新向量
        # 这是标准的Classifier-Free Guidance公式，参考Classifier-Free Guidance论文公式(6)
        scaled_logits = logits + update * (cond_scale - 1.)

        # 如果不使用rescaled CFG，直接返回结果
        if rescaled_phi == 0.:
            return scaled_logits, null_logits

        # 应用rescaled classifier-free guidance
        # 定义计算标准差的函数，沿除了批量维度外的所有维度计算
        # std_fn: 一个“计算张量标准差”的函数工厂（partial 相当于提前把部分参数固定住）
        # torch.std(..., dim=?, keepdim=True) 会沿指定维度求标准差。
        # 这里 dim = tuple(range(1, scaled_logits.ndim)) 的含义是：
        #   - 从维度 1 开始一直到最后一个维度（不包含 dim=0）
        #   - 通常 dim=0 是 batch 维度，所以我们“对每个样本单独计算标准差”，
        #     而不是把整个 batch 混在一起统计。
        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        # rescaled_logits: 对 scaled_logits 做“幅度重标定”
        # 解释：
        #   - scaled_logits 是经过 CFG 放大的结果：logits + (cond_scale-1)*update
        #     它的整体波动幅度（标准差）可能比原始 logits 更大/更小。
        #   - std_fn(logits) 给出“原始有条件预测 logits”的标准差（每个样本一个标量/或保留成可广播形状）
        #   - std_fn(scaled_logits) 给出“CFG 放大后 scaled_logits”的标准差
        #   - 两者相除得到一个“缩放系数”：
        #       如果 scaled_logits 的 std 比 logits 大 -> 比值 < 1 -> 把 scaled_logits 缩小
        #       如果 scaled_logits 的 std 比 logits 小 -> 比值 > 1 -> 把 scaled_logits 放大
        #   - 这样 rescaled_logits 的 std 会被拉回到接近 logits 的 std
        # 好处：
        #   - 避免 cond_scale 很大时输出幅度爆炸
        #   - 保持采样/生成过程的数值尺度更稳定，减少过度引导造成的崩坏
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        # interpolated_rescaled_logits: 在“原 scaled_logits”与“重标定后的 rescaled_logits”之间做线性插值
        # rescaled_phi 的作用（范围通常 0~1，当然也可以允许更大但一般不建议）：
        #   - rescaled_phi = 0  -> 完全不用重标定，等价于 scaled_logits
        #   - rescaled_phi = 1  -> 完全使用重标定后的 rescaled_logits
        #   - 0<rescaled_phi<1 -> 部分重标定：既保留 CFG 的效果，又缓和尺度校准带来的改变
        #
        # 直觉：
        #   - scaled_logits 可能“引导很强但容易过头”
        #   - rescaled_logits “尺度更稳、不过分膨胀”
        #   - 插值让你用一个旋钮(rescaled_phi)控制“稳健性 vs 引导强度表现”的折中
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits, null_logits

    def forward(
            self,
            x,
            time,
            classes,
            cond_drop_prob=None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        

        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        # 初始卷积特征提取，尺寸为[b, init_dim, n]
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        # 编码器阶段（下采样）
        for block1, block2, attn, downsample in self.downs:
            # 每个阶段双残差块 -> 注意力 -> 下采样，且保存中间特征作 skip connection
            # 尺寸为[b, dim_in, n]
            x = block1(x, t, c)  # 第一个残差块
            h.append(x)  # 保存特征用于skip connection

            # 尺寸为[b, dim_in, n]
            x = block2(x, t, c)  # 第二个残差块
            # 尺寸为[b, dim_in, n]
            x = attn(x)  # 注意力模块
            h.append(x)  # 保存特征用于skip connection

            # 尺寸为[b, dim_out, n/2]
            x = downsample(x)

        # 瓶颈层处理
        # 尺寸为[b, mid_dim, n/s]
        x = self.mid_block1(x, t, c)
        # 尺寸为[b, mid_dim, n/s]
        x = self.mid_attn(x)
        # 尺寸为[b, mid_dim, n/s]
        x = self.mid_block2(x, t, c)

        # 解码器阶段（上采样）
        for block1, block2, attn, upsample in self.ups:
            # 解码阶段逐步拼接 skip feature 并上采样复原空间尺寸，尺寸为[b, dim_out + dim_in, n/2]
            x = torch.cat((x, h.pop()), dim=1)  # 拼接skip connection特征
            # 尺寸为[b, dim_out, n/2]
            x = block1(x, t, c)  # 第一个残差块

            # 尺寸为[b, dim_out + dim_in, n/2]
            x = torch.cat((x, h.pop()), dim=1)  # 拼接skip connection特征
            # 尺寸为[b, dim_out, n/2]
            x = block2(x, t, c)  # 第二个残差块
            # 尺寸为[b, dim_out, n/2]
            x = attn(x)  # 注意力模块

            # 尺寸为[b, dim_out, n]
            x = upsample(x)

        # 拼接最早的高分辨率表示 r，帮助恢复低层语义，尺寸为[b, init_dim * 2, n]
        x = torch.cat((x, r), dim=1)

        # 最终残差块和卷积输出，尺寸为[b, init_dim, n]
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    """
    负责前向扩散与反向生成的整体流程，包括时间调度、噪声预测目标、采样策略
    （DDPM / DDIM）以及诸多训练技巧，如 SNR reweight、自条件、offset noise 等。
    """

    def __init__(
            self,
            model,  # 用于预测噪声或图像的神经网络模型（通常为UNet）
            *,
            seq_length,  # 输入数据的长度
            timesteps=1000,  # 扩散过程中的总时间步数
            sampling_timesteps=None,  # 采样时使用的时间步数，None表示与训练时相同
            objective='pred_noise',  # 训练目标类型：'pred_noise', 'pred_x0', 'pred_v'
            beta_schedule='cosine',  # β值调度策略：'linear', 'cosine', 'sigmoid'
            ddim_sampling_eta=1.,  # DDIM采样中的η参数，控制随机性
            channels=None,
            channel_first=True,
            offset_noise_strength=0.,  # 参考 https://www.crosslabs.org/blog/diffusion-with-offset-noise # 偏置噪声强度，参考相关博客文章
            min_snr_loss_weight=False,  # 参考 https://arxiv.org/abs/2303.09556 # 是否使用权重平衡不同时间步的损失，参考论文
            min_snr_gamma=5,  # SNR裁剪参数，用于min-SNR-weighting
            use_cfg_plus_plus=False  # https://arxiv.org/pdf/2406.08070
    ):
        super().__init__()
        # 验证模型配置：确保通道数与输出维度匹配
        assert not (type(self) == GaussianDiffusion1D and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = default(channels, lambda: self.model.channels)  # 模型输入通道数

        self.channel_first = channel_first
        self.seq_index = -2 if not channel_first else -1

        self.seq_length = seq_length

        # 设置训练目标类型并验证其有效性
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # 根据配置选择β值调度函数并计算每个时间步的β值（噪声方差）
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 这些扩散系数的递推与 DDPM 原始推导一致
        # 计算扩散过程中的关键系数
        alphas = 1. - betas  # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 累积乘积 ᾱ_t = Π_{s=1}^t α_s
        # 前一个时间步的累积乘积，第一个元素设为1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # 累积乘积 ᾱ_(t-1)

        # 保存时间步数
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling

        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters

        # 采样相关参数

        # 采样步数可单独配置，缺省时与训练步数一致
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        # 判断是否使用DDIM采样（采样步数小于训练步数时）
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        # DDIM采样的η参数
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        # 封装 register_buffer，统一转为 float32
        # 定义辅助函数：注册缓冲区并统一转为float32类型
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 注册大量缓冲区以便在推理时自动迁移到正确设备
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        # 扩散前向链路所需的各种系数
        # 注册扩散前向链路所需的各种系数（用于采样和损失计算）
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # 后验 q(x_{t-1} | x_t, x_0) 参数
        # 计算后验分布 q(x_{t-1} | x_t, x_0) 的方差系数（用于反向采样），参考论文DDPM中公式(7)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        # 上式等价于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        # 由于扩散链开头方差为0，log需做clamp
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        # 后验均值系数，参考论文DDPM中公式(7)
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        # 偏置噪声强度；博文建议 0.1 左右
        self.offset_noise_strength = offset_noise_strength

        # loss weight

        # 基于 SNR 推导不同目标的 loss 权重（根据DDPM中xt的表达式得到）
        # 信噪比（SNR，signal-to-noise ratio）
        snr = alphas_cumprod / (1 - alphas_cumprod)

        # 相关论文：https://arxiv.org/abs/2303.09556
        # 该论文提出对 SNR 进行裁剪，缓解早期/晚期时间步梯度过大差异
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        # 根据训练目标类型设置损失权重
        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
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
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """计算 q(x_{t-1}|x_t,x0) 的均值与方差，用于采样推导。"""
        # 参考论文DDPM中公式(7)
        # q(xt-1|xt,x0)对应的均值，尺寸为 [b,c,n]
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # q(xt-1|xt,x0)对应的方差，尺寸为 [b,c,n]
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale=6., rescaled_phi=0.7, clip_x_start=False):
        # model_output,条件引导后的模型输出,尺寸为[b,c,n],model_output_null,无条件情况下得到的模型输出,尺寸为[b,c,n]
        model_output, model_output_null = self.model.forward_with_cond_scale(x, t, classes, cond_scale=cond_scale,
                                                                             rescaled_phi=rescaled_phi)
        # 定义裁剪函数
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            # 如果不使用cfg++，pred_noise就是有条件预测结果；否则使用无条件预测结果，尺寸为[b,c,n]
            pred_noise = model_output if not self.use_cfg_plus_plus else model_output_null
            # 从噪声预测推导出 x0,尺寸为[b,c,n]
            x_start = self.predict_start_from_noise(x, t, model_output)
            # 根据需要裁剪x_start
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            # 裁剪x_start
            x_start = maybe_clip(x_start)
            # 如果不使用cfg++，使用裁剪后的x_start；否则使用裁剪后的无条件预测结果
            x_start_for_pred_noise = x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            # 从预测的x_start反推噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            # 目标是预测v值(v-parameterization)
            v = model_output
            # 从预测的v值反推初始图像x_start
            x_start = self.predict_start_from_v(x, t, v)
            # 裁剪x_start
            x_start = maybe_clip(x_start)

            # 为预测噪声准备x_start值
            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                # 如果使用cfg++，则使用从无条件预测的v值反推的x_start
                x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            # 从最终的x_start反推噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        # 返回包含预测噪声和预测初始图像的命名元组
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised=True):
        """计算反向一步的均值/方差，若需要可裁剪 x0 防止自激发。"""
        # 预测的噪声，尺寸为[b,c,n]，根据xt得到的预测的x0，尺寸为[b,c,n]
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        # 尺寸为[b,c,n]
        x_start = preds.pred_x_start

        # 限制当前得到的x0的范围
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # 根据预测得到的x0和当前的xt，计算 q(x_{t-1}|x_t,x0) 的均值、方差和对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale=6., rescaled_phi=0.7, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        # 创建一个包含当前时间步 t 的张量，尺寸为[b]，与输入图像的批量大小相匹配
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        # 调用 p_mean_variance 计算模型的预测均值、对数方差和预测的去噪图像 x0
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, classes=classes,
                                                                          cond_scale=cond_scale,
                                                                          rescaled_phi=rescaled_phi,
                                                                          clip_denoised=clip_denoised)
        # 如果当前时间步 t 大于 0，生成与图像 x 相同大小的随机噪声；否则，噪声为 0（不再添加噪声）
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        # 计算预测图像，参考论文DDPM公式(6)
        # 根据模型输出的均值 model_mean 和对数方差 model_log_variance，生成去噪后的图像
        # 使用噪声和对数方差的指数来加权噪声，得到预测图像 pred_img
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale=6., rescaled_phi=0.7):
        # 提取批量大小和设备信息
        batch, device = shape[0], self.betas.device

        # 初始化随机噪声图像，这将作为生成过程的起始点
        # 尺寸为[b,c,n]
        img = torch.randn(shape, device=device)

        x_start = None

        # 反向遍历每个时间步，进行图像生成
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # 对于每个时间步 t，调用 p_sample 函数生成一个新的图像
            # 更新图像 img 和初始图像 x_start
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        # 将图像范围从[-1,1]反归一化为[0,1]
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale=6., rescaled_phi=0.7, clip_denoised=True):
        """DDIM 采样实现，可用更少步数近似原始过程。"""
        # 基本采样相关参数，包括批大小、设备号、总采样次数、当前需要采样次数、σ值和网络预测变量
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 构造采样时间序列（当 sampling_timesteps == total_timesteps 时等价于完整 DDPM）
        # 尺寸为[N_sampling]
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # 从标准高斯噪声初始化 x_T，尺寸为[b,c,n]
        img = torch.randn(shape, device=device)

        x_start = None

        # 反向 DDIM 采样循环
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            # 当前时间步构造成 batch 对齐的张量，尺寸为[b]
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # 得到预测噪声和根据当前xk预测得到的x0
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale=cond_scale,
                                                             rescaled_phi=rescaled_phi, clip_x_start=clip_denoised)

            # 若到达最终步（time_next == -1），直接输出 x_0
            if time_next < 0:
                img = x_start
                continue

            # 当前与下一时间步的 alpha
            # 累积值αk,尺寸为[b]
            alpha = self.alphas_cumprod[time]
            # αs(s<k),尺寸为[b]
            alpha_next = self.alphas_cumprod[time_next]

            # 参考论文DDIM中公式(16)
            # DDIM 中控制随机性的噪声尺度（eta=0 时为确定性采样），尺寸为[b]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            # noise direction coefficient，参考论文DDIM中公式(12)
            # 尺寸为[b]
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # 采样噪声（仅在 eta > 0 时起作用），尺寸为[b,c,n]
            noise = torch.randn_like(img)

            # 参考论文DDIM中公式(12)，根据 DDIM 更新公式计算 x_{t_next}
            # 根据xk预测得到的x0和预测得到的噪声得到xs，尺寸为[b,c,n]
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # 将图像范围从[-1,1]反归一化为[0,1]
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, batch_size=16, cond_scale=6., rescaled_phi=0.7):
        """根据配置选择 DDPM 或 DDIM 采样接口，对外提供统一入口。"""
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        shape = (batch_size, channels, seq_length) if self.channel_first else (batch_size, seq_length, channels)
        return sample_fn(classes, shape, cond_scale, rescaled_phi)

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t=None, lam=0.5):
        """在扩散空间内对两幅图像进行线性插值并反向生成。"""
        # b是batch大小；device是张量所在设备（GPU/CPU）
        b, *_, device = *x1.shape, x1.device
        # 若未指定t，则默认用最大时间步（噪声最强）
        t = default(t, self.num_timesteps - 1)

        # 确保两张图形状一致，才能逐元素插值
        assert x1.shape == x2.shape

        # 为batch中每个样本构造同一个时间步t（形状[B]）
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        # 将x1/x2正向扩散到时间步t，得到带噪xt1/xt2
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        # 在噪声空间xt上做线性插值：lam=0取xt1，lam=1取xt2
        img = (1 - lam) * xt1 + lam * xt2

        # 从t-1到0逐步反向采样去噪
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            # 执行一步反向扩散：从x_i得到x_{i-1}（带条件classes）
            img, _ = self.p_sample(img, i, classes)

        # 返回最终插值生成的图像
        return img

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, noise=None):
        """
        根据前向扩散定义生成 x_t，可传入自定义噪声。
        Args:
        x_start: 初始数据
        t: 时间步索引
        noise: 自定义噪声

        Returns:
            时间步 t 的噪声图像
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            # 生成偏置噪声并添加到原始噪声中
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1')

        # 前向扩散公式: xt = sqrt(ᾱt) * x0 + sqrt(1-ᾱt) * noise
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise=None):
        """
        计算训练损失，融合 offset noise、自条件与 SNR reweight。
        Args:
        x_start: 初始数据
        t: 时间步索引
        noise: 自定义噪声
        offset_noise_strength: 偏置噪声强度

        Returns:
            计算得到的损失值（经过SNR加权的均方误差损失）
        """
        b = x_start.shape[0]  # 获取输入数据的批量大小
        n = x_start.shape[self.seq_index]

        # 尺寸为[b,c,n]
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        # 前向扩散得到 x_t，尺寸为[b,c,n]
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        # 使用模型对噪声图像进行预测，得到模型的输出，尺寸为[b,c,n]
        model_out = self.model(x, t, classes)

        # 根据训练目标设置目标值
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            # 如果目标是预测v（即v参数化的目标），则预测v值，尺寸为[b,c,n]
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 计算 MSE 损失，使用 reduction='none' 以便逐元素计算损失
        loss = F.mse_loss(model_out, target, reduction='none')
        # 对损失进行按批次求均值的操作
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 使用 SNR（信噪比）加权损失，调整不同时间步的损失权重
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, classes, *args, **kwargs):
        """
        Trainer/Accelerator 调用的统一入口，对输入数据随机采样 t 并计算损失。
        Args:
        img: 输入数据
        *args, **kwargs: 其他参数

        Returns:
            计算得到的损失值
        """
        b, n, device, seq_length, = img.shape[0], img.shape[self.seq_index], img.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 归一化图像，从[0,1]到[-1,1]
        img = normalize_to_neg_one_to_one(img)
        # 计算损失
        return self.p_losses(img, t, classes=classes, *args, **kwargs)


# example

# trainer class

class Trainer1D(object):
    def __init__(
            self,
            diffusion_model: GaussianDiffusion1D,
            dataset: Dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=1000000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.
    ):
        super().__init__()

        # accelerator

        # --------------------------- 加速器初始化 ---------------------------
        # Accelerator 负责：
        #   - 多卡训练（分布式 / 数据并行）
        #   - 混合精度训练（fp16 / bf16）
        #   - 自动处理梯度同步、设备迁移等繁琐细节
        # 加速器负责多卡/混合精度调度
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model

        # 模型引用与通道配置
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        # 采样与训练超参数
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        # 标准 DataLoader：shuffle 打乱，pin_memory 提升主机到 GPU 的拷贝效率
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        # 使用accelerator准备数据加载器，使其支持多GPU和混合精度
        dl = self.accelerator.prepare(dl)
        # cycle(dl)：将 dataloader 包装成无限生成器，一直循环数据
        self.dl = cycle(dl)

        # optimizer

        # 优化器
        # 标准 Adam 优化器，参数来自扩散模型
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        # 定期保存模型 / 采样结果
        # 只有主进程（rank 0）负责维护 EMA 和保存模型，避免多进程重复写文件
        if self.accelerator.is_main_process:
            # 创建指数移动平均模型
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            # 将EMA模型移动到相应设备
            self.ema.to(self.device)

        # 创建结果保存目录
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        # 记录当前训练步数
        self.step = 0

        # 记录每个step的total_loss用于后续分析
        self.loss_history = []

        # prepare model, dataloader, optimizer with accelerator

        # 使用 accelerator.prepare 适配模型与优化器到正确设备 / 分布式环境
        # 之后 self.model / self.opt 应该只通过 accelerator 调用
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        """快捷获取 accelerator 当前设备。"""
        return self.accelerator.device

    def save(self, milestone):
        """在主进程保存检查点，包括模型/EMA/优化器/Scaler。"""
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'loss_history': self.loss_history,  # 保存损失历史
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
        model.load_state_dict(data['model'])  # 恢复模型参数

        # 恢复 step 和优化器状态
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        # 恢复损失历史(如果存在)
        if 'loss_history' in data:
            self.loss_history = data['loss_history']

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
        """标准训练循环：梯度累积、EMA与定期采样。"""
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 包装训练进度条
        # initial=self.step 支持从中途恢复训练
        with (tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar):

            # 主训练循环，直到 step 达到 train_num_steps
            while self.step < self.train_num_steps:
                # 切换到训练模式（启用 dropout / BN 统计等）
                self.model.train()

                total_loss = 0.  # 用于统计当前 step 内（含梯度累积）的 loss 和

                # --------------------------- 梯度累积 ---------------------------
                # 通过在多个 mini-batch 上累积梯度，再统一反向更新一次
                # 可达到“更大 batch 训练”的效果，而不需要增加显存
                for _ in range(self.gradient_accumulate_every):
                    # 从无限循环的 dataloader 中取出一个 batch，并移动到目标 device
                    data, labels = next(self.dl)
                    # 尺寸为[B,C,N]
                    data = data.to(device)
                    # 尺寸为[B]
                    labels = labels.to(device)

                    # autocast：在混合精度下，自动推断使用 FP16 / FP32
                    with self.accelerator.autocast():
                        # 扩散模型通常将“输入 x -> 返回 loss”
                        loss = self.model(data, labels)
                        # 除以累积次数，等价于对多个 batch 的 loss 取平均
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()  # 记录到 total_loss 统计值中

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

                # 记录当前step的total_loss
                self.loss_history.append(total_loss)

                self.step += 1
                if accelerator.is_main_process:
                    # 主进程更新 EMA 权重
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # 切换 EMA 模型到 eval 模式（关掉 dropout / BN 训练行为）
                        self.ema.ema_model.eval()

                        with torch.no_grad():  # 关闭梯度，加速推理
                            # 计算当前是第几个 milestone（从 1 开始）
                            milestone = self.step // self.save_and_sample_every
                            # 将 num_samples 拆成若干批（因为单次采样 batch_size 受显存限制）
                            batches = num_to_groups(self.num_samples, self.batch_size)

                            # 为每个批次随机生成类别标签
                            all_samples_list = []
                            all_labels_list = []
                            for n in batches:
                                # 随机生成类别标签，范围为 [0, num_classes)
                                batch_labels = torch.randint(0, self.model.model.num_classes, (n,), device=device)
                                # 使用生成的类别标签进行采样
                                batch_samples = self.ema.ema_model.sample(classes=batch_labels, batch_size=n)
                                all_samples_list.append(batch_samples)
                                all_labels_list.append(batch_labels)

                        # 拼接所有生成样本为一个大 tensor：[N, C, seq_length]
                        all_samples = torch.cat(all_samples_list, dim=0)
                        # 拼接所有标签为一个大 tensor：[N]
                        all_labels = torch.cat(all_labels_list, dim=0)

                        # 保存样本和标签到.pt文件
                        save_dict = {
                            'samples': all_samples.cpu(),
                            'labels': all_labels.cpu()
                        }
                        torch.save(save_dict, str(self.results_folder / f'sample-{milestone}.pt'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


if __name__ == '__main__':
    num_classes = 10

    model = Unet1D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=num_classes,
        cond_drop_prob=0.5
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=128,
        timesteps=1000
    ).cuda()

    training_images = torch.randn(8, 3, 128, 128).cuda()  # images are normalized from 0 to 1
    image_classes = torch.randint(0, num_classes, (8,)).cuda()  # say 10 classes

    loss = diffusion(training_images, classes=image_classes)
    loss.backward()

    # do above for many steps

    sampled_images = diffusion.sample(
        classes=image_classes,
        cond_scale=6.
        # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    sampled_images.shape  # (8, 3, 128, 128)

    # interpolation

    interpolate_out = diffusion.interpolate(
        training_images[:1],
        training_images[:1],
        image_classes[:1]
    )