import os
import torch

# ==================== 数据配置 ====================

# 相对路径:训练好的模型保存位置
model_path = os.path.join("logs", "diffusion_model.pt")

# 数据集配置
mode = ['train', 'test']

# 相对路径:数据目录
data_dir = 'data'

# 使用的身体侧别
# 'l': 左侧, 'r': 右侧, ['l', 'r']: 两侧
# 如果是列表，会同时加载两侧的数据并合并
side = ['l', 'r']

# ==================== Activity Flag 配置 ====================
# True: 读取activity_flag.csv文件，并将对应的left/right列与数据拼接
# False: 不使用activity_flag.csv文件
activity_flag = True

# ==================== 参与者体重配置 ====================
# True: 将参与者体重作为特征拼接到数据中（每个时间步重复该值）
# False: 不使用参与者体重特征
use_participant_mass = True

# 参与者体重字典(单位:kg)
participant_masses = {
    "BT01": 80.59, "BT02": 72.24, "BT03": 95.29, "BT04": 98.23,
    "BT06": 79.33, "BT07": 64.49, "BT08": 69.13, "BT09": 82.31,
    "BT10": 93.45, "BT11": 50.39, "BT12": 78.15, "BT13": 89.85,
    "BT14": 67.30, "BT15": 58.40, "BT16": 64.33, "BT17": 60.03,
    "BT18": 67.96, "BT19": 69.95, "BT20": 55.44, "BT21": 58.85,
    "BT22": 76.79, "BT23": 67.23, "BT24": 77.79
}

# ==================== 扩散模型生成的特征配置 ====================
# 扩散模型要生成的特征列表(* 会被替换为 side)
# 这里包含传感器数据和力矩数据
input_names = [
    # IMU传感器数据
    "foot_imu_*_gyro_x", "foot_imu_*_gyro_y", "foot_imu_*_gyro_z",
    "foot_imu_*_accel_x", "foot_imu_*_accel_y", "foot_imu_*_accel_z",
    "shank_imu_*_gyro_x", "shank_imu_*_gyro_y", "shank_imu_*_gyro_z",
    "shank_imu_*_accel_x", "shank_imu_*_accel_y", "shank_imu_*_accel_z",
    "thigh_imu_*_gyro_x", "thigh_imu_*_gyro_y", "thigh_imu_*_gyro_z",
    "thigh_imu_*_accel_x", "thigh_imu_*_accel_y", "thigh_imu_*_accel_z",
    "insole_*_cop_x", "insole_*_cop_z", "insole_*_force_y",
    "hip_angle_*", "hip_angle_*_velocity_filt",
    "knee_angle_*", "knee_angle_*_velocity_filt"
]

# ==================== 力矩特征配置 ====================
# 力矩特征名称(从moment_filt.csv读取，会与传感器数据合并)
label_names = ["hip_flexion_*_moment", "knee_angle_*_moment"]

# ==================== 序列长度配置 ====================
# 允许加载的最小序列长度（单位：采样点，200Hz，每点=5ms）
# -1 表示不限制序列长度（默认行为）
min_sequence_length = -1

# 扩散模型的子序列长度（每个样本的时间步数）
diffusion_sequence_length = 296

# 运动类型筛选模式（使用正则表达式）
# 每个元素可以是单个pattern或多个patterns的列表，同一元素内的patterns属于同一类别
# 总共28个类别（类别索引0-27）
action_patterns = [
    # === 按论文中重要性排序的动作筛选 ===
    [r"^normal_walk_.*_(shuffle|0-6|1-2|1-8).*"],  # 类别0: Level ground walk
    [r"^poses_.*"],  # 类别1: Standing poses
    [r"^dynamic_walk_.*(high-knees|butt-kicks).*", r"^normal_walk_.*skip.*", r"^tire_run_.*"],  # 类别2: Calisthenics
    [r"^push_.*"],  # 类别3: Push and pull recovery
    [r"^jump_.*_(hop|vertical|180|90-f|90-s).*"],  # 类别4: Jump in place
    [r"^turn_and_step_.*"],  # 类别5: Turns
    [r"^cutting_.*"],  # 类别6: Cut
    [r"^sit_to_stand_.*"],  # 类别7: Sit and stand
    [r"^walk_backward_.*"],  # 类别8: Backwards walk
    [r"^weighted_walk_.*"],  # 类别9: 25 lb Loaded walk
    [r"^lift_weight_.*"],  # 类别10: Lift and place weight
    [r"^tug_of_war_.*"],  # 类别11: Tug of war
    [r"^jump_.*_(fb|lateral).*", r"^side_shuffle_.*"],  # 类别12: Jump across
    [r"^normal_walk_.*_(2-0|2-5).*"],  # 类别13: Run
    [r"^dynamic_walk_.*(toe-walk|heel-walk).*"],  # 类别14: Toe and heel walk
    [r"^twister_.*"],  # 类别15: Twister
    [r"^meander_.*"],  # 类别16: Meander
    [r"^incline_walk_.*up.*"],  # 类别17: Inclined walk
    [r"^stairs_.*down.*"],  # 类别18: Stair descent
    [r"^lunges_.*"],  # 类别19: Lunge
    [r"^stairs_.*up.*"],  # 类别20: Stair ascent
    [r"^incline_walk_.*down.*"],  # 类别21: Declined walk
    [r"^start_stop_.*"],  # 类别22: Start and stop
    [r"^ball_toss_.*"],  # 类别23: Medicine ball toss
    [r"^obstacle_walk_.*"],  # 类别24: Step over
    [r"^squats_.*"],  # 类别25: Squat
    [r"^curb_.*"],  # 类别26: Curb
    [r"^step_ups_.*"],  # 类别27: Step up
]

# ==================== 归一化配置 ====================

# 是否启用数据归一化
enable_normalization = True

# 特征统计文件路径（包含每个类别每个特征的最大最小值）
# 该文件由 compute_and_save_statistics() 方法生成
feature_statistics_path = os.path.join("data", "feature_statistics.json")

# 归一化方法选择
# 可选值:
#   'linear': 线性归一化 (x - min) / (max - min)
#   'tanh': 基于tanh的S型归一化，两端变化慢，中间变化快
#   'power': 幂函数归一化 ((x - min) / (max - min)) ** alpha
normalization_method = 'tanh'

# 归一化方法的超参数
normalization_params = {
    # tanh方法的参数：控制S曲线的陡峭程度
    # 值越大，中间区域越陡峭，两端越平缓
    # 推荐范围: 2.0 - 6.0
    'tanh_scale': 3.0,

    # power方法的参数：幂指数
    # alpha > 1: 数据向1端集中
    # alpha < 1: 数据向0端集中
    # alpha = 1: 等价于线性归一化
    'power_alpha': 2.0
}

# ==================== 扩散模型配置 ====================

# 模型维度配置
dim = 64
dim_mults = (1, 2, 4, 8)

# 扩散步数
timesteps = 1000

# 训练配置
train_batch_size = 512
train_lr = 1e-4
train_num_steps = 700000

# 梯度累积步数
gradient_accumulate_every = 2

# EMA配置
ema_decay = 0.995
ema_update_every = 10

# 保存和采样间隔
save_and_sample_every = 10

# 采样数量
num_samples = 50

# 结果保存目录
results_folder = './logs'

# 混合精度训练
amp = True

# Classifier-free guidance配置
# 在训练时随机丢弃类别标签的概率
cond_drop_prob = 0.5

# 在采样时的guidance scale (>1 增强条件引导)
cond_scale = 3.0

# ==================== 其他配置 ====================

# 随机种子(保证结果可复现)
random_seed = 42