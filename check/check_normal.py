import torch
from pointcept.datasets import WindShearDataset  # 导入你的数据集类

# 1. 实例化 WindShearDataset（和训练时的配置一致，确保触发 __getitem__）
dataset = WindShearDataset(
    split='train',
    data_root='/mnt/d/model/wind_datas/csv_labels',
    transform=None,  # 先不加transform，方便看原始5维feat
    k_neighbors=16,  # 和配置一致，确保 _compute_neighborhood_features 正常执行
    radius=100.0
)

# 2. 取第一个样本，触发 __getitem__（此时会打印 feat 维度和各列均值）
sample_idx = 0
data_dict = dataset[sample_idx]  # 这一步会执行 __getitem__，看到你加的打印
feat = data_dict['feat']
coord = data_dict['coord']
label = data_dict['generate_label']
beamaz = data_dict['beamaz']

# 3. 打印真实的 feat 信息（验证5维结构）
print(f"\n通过 WindShearDataset 获取的 feat 信息：")
print(f"feat 维度：{feat.shape}")  # 应该输出 (N, 5)，N是点数
print("feat 各列均值（用于确认特征含义）：")
for i in range(feat.shape[1]):
    print(f"  第{i}列：{feat[:, i].mean():.2f}")

# 4. 现在执行归一化检查（适配5维 feat，不访问超出范围的索引）
from pointcept.datasets.transforms import NormalizeWind

# 初始化归一化（用你配置中的参数，或重新计算的统计量）
normalize = NormalizeWind(
    u_mean=-0.7983, u_std=2.6732,
    v_mean=2.0401, v_std=3.1915,
    beamaz_mean=184.0494, beamaz_std=100.2158
)

# 应用归一化（此时 feat 是5维，和训练时一致）
data_dict_norm = normalize(data_dict.copy())
feat_norm = data_dict_norm['feat']

# 5. 检查归一化结果（只访问 0-4 索引，避免超出范围）
print("\n归一化后 feat 各列均值（应接近0）：")
for i in range(feat_norm.shape[1]):
    print(f"  第{i}列：{feat_norm[:, i].mean():.4f}")

# 重点检查 BeamAz 对应的列（假设是第2列，根据前面的打印确认）
beamaz_col_idx = 2  # 从 __getitem__ 的打印中确认的 BeamAz 索引
print(f"\nBeamAz 对应列（第{beamaz_col_idx}列）归一化结果：")
print(f"  归一化前均值：{feat[:, beamaz_col_idx].mean():.2f}")
print(f"  归一化后均值：{feat_norm[:, beamaz_col_idx].mean():.4f}（应接近0）")
print(f"  归一化后标准差：{feat_norm[:, beamaz_col_idx].std():.4f}（应接近1）")