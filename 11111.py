import torch
import numpy as np
from pointcept.datasets.builder import build_dataset
import yaml

# 加载配置文件
with open("configs/wind_shear/pointtransformer_v3.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# 构建验证集和训练集（用于对比）
val_dataset = build_dataset(cfg['data']['val'])
train_dataset = build_dataset(cfg['data']['train'])  # 加载训练集用于对比


def check_sample(dataset, sample_idx, dataset_name="验证集"):
    """检查单个样本的详细信息"""
    try:
        sample = dataset[sample_idx]
        if sample is None:
            print(f"\n{dataset_name} 样本 {sample_idx} 为空（可能被过滤）")
            return

        # 1. 打印样本来源路径
        if 'path' in sample:
            print(f"\n{dataset_name} 样本 {sample_idx} 路径：{sample['path']}")
        else:
            print(f"\n{dataset_name} 样本 {sample_idx} 警告：未包含 'path' 字段")

        # 2. 检查特征 feat
        print("\n=== 特征 (feat) 检查 ===")
        print(f"形状: {sample['feat'].shape}")
        print(f"最小值: {sample['feat'].min().item():.4f}")
        print(f"最大值: {sample['feat'].max().item():.4f}")
        print(f"是否含 nan: {np.isnan(sample['feat']).any()}")
        print(f"是否含 inf: {np.isinf(sample['feat']).any()}")

        # 3. 检查坐标 coord（重点：类型、范围、是否整数）
        print("\n=== 坐标 (coord) 检查 ===")
        print(f"数据类型: {sample['coord'].dtype}（spconv要求为 int32/int64）")
        print(f"形状: {sample['coord'].shape}")
        print(f"坐标范围: min={sample['coord'].min().item()}, max={sample['coord'].max().item()}")
        # 检查是否为整数（允许±1e-6的浮点误差，避免因量化导致的微小偏差）
        is_integer = torch.allclose(sample['coord'], sample['coord'].round(), atol=1e-6)
        print(f"是否为整数（含微小误差）: {is_integer}")
        if not is_integer:
            # 打印非整数的坐标示例（前5个）
            non_integer_mask = ~torch.isclose(sample['coord'], sample['coord'].round(), atol=1e-6)
            if non_integer_mask.any():
                non_integer_examples = sample['coord'][non_integer_mask][:5]
                print(f"非整数坐标示例: {non_integer_examples.tolist()}")
        

        # 4. 检查标签（字段名、范围、合法性）
        print("\n=== 标签 (label) 检查 ===")
        label_key = 'generate_label'
        if label_key not in sample:
            print(f"错误：样本中未找到 '{label_key}' 字段，请检查数据集构建逻辑")
        else:
            labels = sample[label_key]
            print(f"标签字段: {label_key}")
            print(f"标签形状: {labels.shape}")
            print(f"标签范围: min={labels.min().item()}, max={labels.max().item()}")
            print(f"是否含 nan: {torch.isnan(labels).any().item()}")
            print(f"是否含越界值（假设0-4类）: {(labels < 0).any().item() or (labels > 4).any().item()}")
            # 打印标签分布（前5个值）
            print(f"标签示例（前5个）: {labels[:5].tolist()}")

        print("\n" + "=" * 50)

    except Exception as e:
        print(f"\n{dataset_name} 样本 {sample_idx} 检查失败: {str(e)}")


# 检查验证集第0个样本（重点检查）
check_sample(val_dataset, 0, dataset_name="验证集")

# 随机检查验证集另外2个样本（避免偶然性）
if len(val_dataset) > 3:
    check_sample(val_dataset, 10, dataset_name="验证集")  # 第10个样本
    check_sample(val_dataset, 100, dataset_name="验证集")  # 第100个样本

# 检查训练集第0个样本（用于对比coord范围和格式）
print("\n===== 与训练集样本对比 =====")
check_sample(train_dataset, 0, dataset_name="训练集")