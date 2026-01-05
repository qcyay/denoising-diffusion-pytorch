from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from torch.nn.attention import SDPBackend

# constants

AttentionConfig = namedtuple('AttentionConfig', ['backends'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        # 断言：如果启用flash attention，PyTorch版本必须>=2.0
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        # 为CPU和CUDA配置不同的注意力后端
        self.cpu_config = AttentionConfig([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        self.cuda_config = None

        # 如果CUDA不可用或未启用flash，则直接返回
        if not torch.cuda.is_available() or not flash:
            return

        # 获取CUDA设备属性
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        # 解析设备版本号
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        # 根据GPU型号选择合适的注意力配置
        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.FLASH_ATTENTION])
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])

    def flash_attn(self, q, k, v):
        # 提取输入张量的形状信息，q的尺寸为[b, h, n, d]
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 如果设置了自定义缩放因子，则应用缩放
        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        # 确保张量在内存中连续存储
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        # 根据设备类型选择相应的配置
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        # 使用指定的注意力后端执行Flash Attention
        with torch.nn.attention.sdpa_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取序列长度和设备信息
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # 如果启用了flash attention，使用专门的实现
        if self.flash:
            return self.flash_attn(q, k, v)

        # 计算注意力缩放因子
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        # 计算查询和键的相似度矩阵 (QK^T)
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        # 应用softmax获得注意力权重
        attn = sim.softmax(dim = -1)
        # 对注意力权重应用dropout
        attn = self.attn_dropout(attn)

        # aggregate values

        # 使用注意力权重聚合值向量 (AV)
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
