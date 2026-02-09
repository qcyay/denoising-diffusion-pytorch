import os

from jupyterlab.extensions.pypi import xmlrpc_transport_override

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# # 验证不同的beta调度策略
#
# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule as proposed in https://arxiv.org/abs/2102.09672
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)  # [0, 1, 2, ..., timesteps]
#
#     # 核心公式：基于余弦函数的调度
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     # 解释：cos²(π/2 * (t/T + s)/(1 + s))
#
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化，使α₀=1
#
#     # 从累积α计算β
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#
#     return torch.clip(betas, 0.0001, 0.9999)  # 数值稳定性
#
# def linear_beta_schedule(timesteps):
#     beta_start = 0.0001  # 起始噪声很小
#     beta_end = 0.02      # 结束噪声较大
#     return torch.linspace(beta_start, beta_end, timesteps)
#
# def quadratic_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
#
# # def sigmoid_beta_schedule(timesteps):
# #     beta_start = 0.0001
# #     beta_end = 0.02
# #     betas = torch.linspace(-6, 6, timesteps)  # 对称范围
# #     return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
#
# def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
#     """
#     Sigmoid 形状的 beta 调度，灵感来自 https://arxiv.org/abs/2212.11972，
#     对 64x64 以上图像训练更稳定。
#     Args:
#         timesteps: 时间步数
#         start: sigmoid函数的起始值
#         end: sigmoid函数的结束值
#         tau: 温度参数，控制曲线的陡峭程度
#         clamp_min: beta值的最小限制值
#
#     Returns:
#         一个形状为[timesteps]的张量，包含每个时间步的beta值
#     """
#     # 增加一个额外的步骤以方便计算
#     steps = timesteps + 1
#     # 生成[0, 1]之间的均匀时间点
#     t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
#     # 计算sigmoid函数的起始和结束值
#     v_start = torch.tensor(start / tau).sigmoid()
#     v_end = torch.tensor(end / tau).sigmoid()
#     # 使用sigmoid函数生成累积alpha值
#     # 通过对时间进行线性变换并应用sigmoid函数创建平滑的过渡曲线
#     alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
#     # 归一化，使初始累积alpha值为1
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     # 从累积alpha值推导出每个时间步的beta值
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     # 将beta值限制在[clamp_min, 0.999]范围内，防止数值不稳定
#     return torch.clip(betas, 0, 0.999)
#
# timesteps = 1000
#
# # 计算各种调度
# cosine_betas = cosine_beta_schedule(timesteps)
# linear_betas = linear_beta_schedule(timesteps)
# quadratic_betas = quadratic_beta_schedule(timesteps)
# sigmoid_betas = sigmoid_beta_schedule(timesteps)
#
# # 绘制对比图
# plt.figure(figsize=(12, 8))
# plt.plot(cosine_betas, label='Cosine', linewidth=2)
# # plt.plot(linear_betas, label='Linear', linewidth=2)
# # plt.plot(quadratic_betas, label='Quadratic', linewidth=2)
# plt.plot(sigmoid_betas, label='Sigmoid', linewidth=2)
#
# plt.xlabel('Timestep')
# plt.ylabel('β_t')
# plt.title('Diffusion Model Noise Schedules')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# def get_alphas_cumprod(betas):
#     """从β计算累积α"""
#     alphas = 1 - betas
#     return torch.cumprod(alphas, dim=0)
#
# # 计算累积噪声比例
# cosine_alpha_bar = get_alphas_cumprod(cosine_betas)
# linear_alpha_bar = get_alphas_cumprod(linear_betas)
#
# print(f"最终噪声比例:")
# print(f"Cosine: {1 - cosine_alpha_bar[-1]:.4f}")  # 接近1
# print(f"Linear: {1 - linear_alpha_bar[-1]:.4f}")  # 接近1
#
# betas = linear_betas
#
# # define alphas
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
#
# # calculations for diffusion q(x_t | x_{t-1}) and others
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
#
# # calculations for posterior q(x_{t-1} | x_t, x_0)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#
# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = a.gather(-1, t.cpu())
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
#
# # 可视化关键参数
# plt.figure(figsize=(12, 8))
#
# plt.subplot(2, 2, 1)
# plt.plot(betas.numpy(), label='β_t')
# plt.title('Noise Schedule (β_t)')
# plt.xlabel('Timestep')
#
# plt.subplot(2, 2, 2)
# plt.plot(alphas_cumprod.numpy(), label='ᾱ_t')
# plt.title('Cumulative α product')
# plt.xlabel('Timestep')
#
# plt.subplot(2, 2, 3)
# plt.plot(sqrt_alphas_cumprod.numpy(), label='√ᾱ_t')
# plt.title('Signal coefficient')
# plt.xlabel('Timestep')
#
# plt.subplot(2, 2, 4)
# plt.plot(sqrt_one_minus_alphas_cumprod.numpy(), label='√(1-ᾱ_t)')
# plt.title('Noise coefficient')
# plt.xlabel('Timestep')
#
# plt.tight_layout()
# plt.show()


# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8),
#     flash_attn = True
# )
#
# diffusion = GaussianDiffusion(
#     model,
#     image_size = 128,
#     timesteps = 200    # number of steps
# )
#
# training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
# loss = diffusion(training_images)
# loss.backward()
#
# # after a lot of training
#
# sampled_images = diffusion.sample(batch_size = 4)
# sampled_images.shape # (4, 3, 128, 128)
# grid = make_grid(sampled_images, nrow = 2)
# plt.imshow(grid.permute(1, 2, 0).cpu())
# plt.show()


# init_dim = 32
# dim = 64
# dim_mults = (1, 2, 4, 8)
# dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
# print(dims)
# in_out = list(zip(dims[:-1], dims[1:]))
# print(in_out)
# full_attn = (*((False,) * (len(dim_mults) - 1)), True)
# print(full_attn)

def compute_and_save_statistics(self, output_dir: str = "statistics"):
    """
    计算并保存每个运动类别的特征统计信息（最大值和最小值）

    参数:
        output_dir: 统计信息保存目录

    保存格式:
    {
        "class_0": {
            "feature_name_1": {"max": xxx, "min": xxx},
            "feature_name_2": {"max": xxx, "min": xxx},
            ...
        },
        ...
        "participant_mass": {"max": xxx, "min": xxx}
    }
    """
    import json

    print(f"\n{'=' * 60}")
    print(f"开始计算特征统计信息...")
    print(f"{'=' * 60}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取特征名称列表
    feature_names = self._get_feature_names()
    num_features = len(feature_names)

    # 初始化统计字典：每个类别存储每个特征的最大值和最小值
    # {class_idx: {feature_name: {"max": [], "min": []}}}
    class_stats = {}
    for class_idx in range(len(self.action_patterns)):
        class_stats[class_idx] = {}
        for feat_name in feature_names:
            class_stats[class_idx][feat_name] = {
                "max": float('-inf'),
                "min": float('inf')
            }

    # 参与者体重统计（如果启用）
    mass_stats = None
    if self.use_participant_mass and self.participant_masses:
        # 直接从participant_masses字典中获取最大最小值
        mass_values = list(self.participant_masses.values())
        mass_stats = {
            "max": max(mass_values),
            "min": min(mass_values)
        }

    # 遍历所有试验，按类别收集统计信息
    print(f"正在处理 {len(self.all_data)} 个试验...")

    for trial_idx in tqdm(range(len(self.all_data)),
                          desc="计算特征统计",
                          unit="trial",
                          ncols=80):
        data = self.all_data[trial_idx]  # shape: [num_features, seq_len]
        class_label = self.all_labels[trial_idx]

        if class_label == -1:
            # 跳过未知类别
            continue

        # 对每个特征维度计算统计信息
        for feat_idx in range(num_features):
            feat_data = data[feat_idx, :]  # 该特征的所有时间步数据
            feat_name = feature_names[feat_idx]

            # 更新该类别该特征的最大值和最小值
            feat_max = np.max(feat_data)
            feat_min = np.min(feat_data)

            class_stats[class_label][feat_name]["max"] = max(
                class_stats[class_label][feat_name]["max"],
                feat_max
            )
            class_stats[class_label][feat_name]["min"] = min(
                class_stats[class_label][feat_name]["min"],
                feat_min
            )

    # 构建最终的统计字典
    final_stats = {}

    # 添加每个类别的统计信息
    for class_idx in range(len(self.action_patterns)):
        class_key = f"class_{class_idx}"

        # 只保存有数据的类别
        has_data = any(
            class_stats[class_idx][feat_name]["max"] != float('-inf')
            for feat_name in feature_names
        )

        if has_data:
            final_stats[class_key] = {}
            for feat_name in feature_names:
                # 只保存有有效值的特征
                if class_stats[class_idx][feat_name]["max"] != float('-inf'):
                    final_stats[class_key][feat_name] = {
                        "max": float(class_stats[class_idx][feat_name]["max"]),
                        "min": float(class_stats[class_idx][feat_name]["min"])
                    }

    # 添加参与者体重的全局统计（如果启用）
    if self.use_participant_mass and mass_stats is not None:
        final_stats["participant_mass"] = {
            "max": float(mass_stats["max"]),
            "min": float(mass_stats["min"])
        }

    # 保存到文件
    output_file = os.path.join(output_dir, "feature_statistics_old.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

    print(f"统计信息已保存到: {output_file}")
    print(f"包含 {len([k for k in final_stats.keys() if k.startswith('class_')])} 个类别的统计信息")
    print(f"每个类别包含 {num_features} 个特征")

    # 打印示例统计信息
    if len(final_stats) > 0:
        print(f"\n示例统计信息 (class_0 的前3个特征):")
        if "class_0" in final_stats:
            count = 0
            for feat_name, stats in final_stats["class_0"].items():
                print(f"  {feat_name}: max={stats['max']:.4f}, min={stats['min']:.4f}")
                count += 1
                if count >= 3:
                    break

    print(f"{'=' * 60}\n")

    return final_stats