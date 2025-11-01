# check/count_class_weights.py
# (最终修正版：修复了 TypeError: cfg must be a dict)

import os
import sys
import numpy as np
import torch
import logging
import math
import argparse
from tqdm import tqdm

# --- 🌟 修复1：PYTHONPATH 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# (将日志级别设为 ERROR，避免打印过多 INFO)
logging.basicConfig(level=logging.ERROR)
logging.info(f"已将项目根目录添加到路径: {project_root}")

# --- 🌟 修复2：安全导入 ---
try:
    from omegaconf import OmegaConf
    from pointcept.datasets.builder import build_train_dataloader
    from pointcept.utils.logger import get_root_logger
except ImportError as e:
    logging.error(f"导入 'pointcept' 模块失败。错误: {e}")
    logging.error("请确保您的 'pointcept' 目录位于: {project_root}")
    sys.exit(1)


def count_final_class_weights(config_file):
    """
    加载完整的训练数据管道，遍历所有样本（应用所有变换），
    并计算最终的类别分布和权重。
    """
    print(f"加载配置文件: {config_file}")

    # 1. 加载 OmegaConf 配置
    cfg_omega = OmegaConf.load(config_file)

    # --- 🌟 修复3：将 OmegaConf 转换为标准 dict ---
    # 这是解决 "TypeError: cfg must be a dict" 的关键
    try:
        cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    except Exception as e:
        print(f"OmegaConf 转换失败: {e}")
        return

    # 2. 初始化日志记录器
    get_root_logger(
        log_file=None,
        log_level=logging.INFO  # 日志器级别保持INFO
    )

    # 3. 构建训练数据加载器
    # (现在传递的是 dict 类型的 cfg，与 train.py 行为一致)
    print("构建训练数据加载器 (调用 build_train_dataloader)...")
    try:
        train_dataloader = build_train_dataloader(cfg)
    except Exception as e:
        print(f"构建 train_dataloader 失败: {e}")
        logging.exception("详细错误信息:")
        return

    print("开始遍历数据集... 这可能需要一些时间。")

    all_labels = []

    # 4. 遍历 Dataloader
    for data_dict in tqdm(train_dataloader, desc="遍历训练集"):
        if data_dict is None:
            continue
        labels = data_dict.get('generate_label')
        if labels is not None:
            all_labels.append(labels.cpu().numpy())

    if not all_labels:
        print("错误：没有从数据加载器中找到任何标签，请检查配置。")
        return

    # 5. 合并所有标签并计算
    all_labels = np.concatenate(all_labels)
    total_points = len(all_labels)

    # 动态获取类别数 (现在从 dict 中获取)
    NUM_CLASSES = 5  # 默认为5
    if 'model' in cfg and 'num_classes' in cfg['model']:
        NUM_CLASSES = cfg['model']['num_classes']

    print(f"\n遍历完成。共统计 {total_points} 个 *最终* (采样+SMOTE+补点后) 的标签点。")

    counts = np.bincount(all_labels, minlength=NUM_CLASSES)

    print("\n--- 最终类别点数统计 (采样+SMOTE+补点后) ---")
    print(f"总点数: {total_points}")
    print("---------------------------------------")
    for i in range(NUM_CLASSES):
        print(f"  类别 {i}: {counts[i]:>10d} 点")  # 调整格式以便对齐
    print("---------------------------------------")

    # 6. 计算类别权重
    weights = []
    print("\n--- 推荐的 class_weights (ENet 方法) ---")
    print("计算公式: 1.0 / log(1.02 + (类别点数 / 总点数))")

    for i in range(NUM_CLASSES):
        if counts[i] == 0:
            weight = 1.0
        else:
            proportion = counts[i] / total_points
            weight = 1.0 / math.log(1.02 + proportion)
        weights.append(weight)
        print(f"  类别 {i}: {weight:.4f}")

    weights_str = ", ".join([f"{w:.4f}" for w in weights])
    print("\n[请将以下列表复制到您的 .yaml 文件中]")
    print(f"  class_weights: [ {weights_str} ]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算最终训练集类别权重")
    parser.add_argument(
        'config',
        type=str,
        help='配置文件的路径 (例如 configs/wind_shear/pointtransformer_v3.yaml)'
    )
    args = parser.parse_args()

    count_final_class_weights(args.config)