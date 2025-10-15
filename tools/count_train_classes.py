import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt  # 新增：用于绘制点数分布直方图

# 解决模块导入问题：与train.py保持一致
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入自定义模块（仅保留数据加载相关）
from pointcept.datasets.builder import build_train_dataloader
from pointcept.utils.logging import setup_logging
from pointcept.utils.logger import get_logger


def parse_args():
    """解析命令行参数：仅需指定配置文件"""
    import argparse
    parser = argparse.ArgumentParser(description="统计训练集类别分布和样本点数分布")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/wind_shear/pointtransformer_v3.yaml',
        help='配置文件路径（与train.py使用的配置一致）'
    )
    parser.add_argument(
        '--histogram',
        action='store_true',
        help='是否生成点数分布直方图（保存到logs目录）'
    )
    return parser.parse_args()


def load_config(config_path):
    """加载yaml配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def count_train_statistics(train_loader, num_classes=5, save_histogram=False):
    """
    统计训练集：
    1. 类别分布（点级别）
    2. 每个样本的点数（样本级别）及分布特征
    """
    # 1. 初始化统计变量
    class_counts = defaultdict(int)  # 类别点数量
    sample_point_counts = {}  # 样本点数：{样本路径: 点数}
    total_points = 0  # 总点数
    all_sample_points = []  # 所有样本的点数列表（用于计算分布）

    print(f"\n开始遍历训练集（共{len(train_loader)}个batch）...")
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="统计中")):
        # 跳过空batch
        if batch is None or len(batch.get('path', [])) == 0:
            print(f"⚠️  跳过空batch {batch_idx}（无有效样本）")
            continue

        # 2. 提取当前batch的关键字段
        labels = batch['generate_label'].long()  # 点级别标签
        offsets = batch['offset']  # 样本分割偏移量（关键：用于计算每个样本的点数）
        paths = batch['path']  # 样本路径（唯一标识样本）

        # 3. 统计类别分布（点级别）
        flat_labels = labels.view(-1).cpu().numpy()
        batch_class_count = np.bincount(flat_labels, minlength=num_classes)
        for cls in range(num_classes):
            class_counts[cls] += batch_class_count[cls]
        total_points += flat_labels.shape[0]

        # 4. 统计样本点数（核心新增逻辑）
        # offsets格式：[0, p1, p1+p2, ..., total]，每个样本点数 = offsets[i+1] - offsets[i]
        offsets_np = offsets.cpu().numpy()
        for i in range(len(paths)):
            sample_path = paths[i]
            # 计算当前样本的点数（处理最后一个样本的边界情况）
            if i < len(offsets_np) - 1:
                point_num = offsets_np[i+1] - offsets_np[i]
            else:
                point_num = flat_labels.shape[0] - offsets_np[i]  # 兜底：避免索引越界
            # 记录样本点数
            sample_point_counts[sample_path] = point_num
            all_sample_points.append(point_num)

    # 5. 计算点数分布的关键指标
    if all_sample_points:
        point_stats = {
            'min': np.min(all_sample_points),
            'max': np.max(all_sample_points),
            'mean': np.mean(all_sample_points),
            'median': np.median(all_sample_points),
            'std': np.std(all_sample_points),  # 标准差：反映点数波动程度
            'total_samples': len(all_sample_points)
        }
    else:
        point_stats = None

    # 6. 生成点数分布直方图（可选）
    if save_histogram and all_sample_points:
        plt.figure(figsize=(10, 6))
        plt.hist(all_sample_points, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(point_stats['mean'], color='r', linestyle='--', label=f'均值：{point_stats["mean"]:.1f}')
        plt.axvline(point_stats['median'], color='g', linestyle='-', label=f'中位数：{point_stats["median"]:.1f}')
        plt.xlabel('样本点数')
        plt.ylabel('样本数量')
        plt.title('训练集样本点数分布')
        plt.legend()
        os.makedirs('./logs', exist_ok=True)
        plt.savefig('./logs/sample_point_count_histogram.png')
        plt.close()
        print(f"📊 点数分布直方图已保存到 ./logs/sample_point_count_histogram.png")

    # 整理结果
    final_class_counts = {cls: class_counts.get(cls, 0) for cls in range(num_classes)}
    return final_class_counts, total_points, sample_point_counts, point_stats


def main():
    # 1. 初始化日志
    setup_logging(log_dir="./logs")
    logger = get_logger("train_statistics")

    # 2. 解析参数+加载配置
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"✅ 加载配置文件：{args.config}")

    # 3. 构建训练集DataLoader（复用train.py逻辑）
    train_loader = build_train_dataloader(cfg)
    logger.info(f"✅ 训练集DataLoader构建完成：共{len(train_loader)}个batch，{len(train_loader.dataset)}个样本文件")

    # 4. 统计类别分布和样本点数
    num_classes = cfg['model']['num_classes']
    class_counts, total_points, sample_counts, point_stats = count_train_statistics(
        train_loader,
        num_classes,
        save_histogram=args.histogram  # 控制是否生成直方图
    )

    # 5. 输出类别分布结果（保持原有格式）
    logger.info("\n" + "=" * 60)
    logger.info("训练集类别数量统计结果（点级别）")
    logger.info("=" * 60)
    class_names = {
        0: "无风切变",
        1: "轻微风切变",
        2: "中度风切变",
        3: "重度风切变",
        4: "严重风切变"
    }
    for cls in range(num_classes):
        count = class_counts[cls]
        percentage = (count / total_points) * 100 if total_points > 0 else 0.0
        logger.info(f"类别{cls}（{class_names[cls]}）：{count:,} 个点（占比：{percentage:.2f}%）")
    logger.info("=" * 60)
    logger.info(f"训练集总点数：{total_points:,}")

    # 6. 新增：输出样本点数统计结果
    if point_stats:
        logger.info("\n" + "=" * 60)
        logger.info("训练集样本点数统计结果（样本级别）")
        logger.info("=" * 60)
        logger.info(f"样本总数：{point_stats['total_samples']}")
        logger.info(f"最小点数：{point_stats['min']}")
        logger.info(f"最大点数：{point_stats['max']}")
        logger.info(f"平均点数：{point_stats['mean']:.1f}")
        logger.info(f"中位数点数：{point_stats['median']:.1f}")
        logger.info(f"点数标准差：{point_stats['std']:.1f}（值越大，点数差异越显著）")
        logger.info("=" * 60)

        # 打印点数极端的样本（辅助分析）
        sorted_samples = sorted(sample_counts.items(), key=lambda x: x[1])
        logger.info("\n点数最少的5个样本：")
        for path, num in sorted_samples[:5]:
            logger.info(f"  {path.split('/')[-1]}: {num} 点")
        logger.info("\n点数最多的5个样本：")
        for path, num in sorted_samples[-5:]:
            logger.info(f"  {path.split('/')[-1]}: {num} 点")
    else:
        logger.warning("\n⚠️  未统计到有效样本点数（可能数据集为空）")

    logger.info("\n统计完成！可根据样本点数分布调整数据预处理策略")


if __name__ == "__main__":
    main()