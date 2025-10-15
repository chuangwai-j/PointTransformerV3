import os
import sys
import logging

# 解决模块导入问题：将项目根目录添加到Python搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
#from sklearn.metrics import recall_score, precision_score, f1_score
import warnings
import matplotlib.pyplot as plt

# 导入自定义模块
from pointcept.datasets.builder import build_train_dataloader, build_val_dataloader
from pointcept.models import build_model
from pointcept.utils.logger import get_logger
from pointcept.utils.checkpoint import save_checkpoint
from pointcept.utils.logging import setup_logging  # 导入工具函数

# 1. 配置全局日志（只调用1次！）
logger = setup_logging(log_dir="./logs")  # 日志文件存到项目根目录的logs文件夹


# -------------------------- 新增：统计训练集类别数量（点级别） --------------------------
def count_train_classes(train_loader):
    """
    统计训练集每个类别的点级别样本数量
    返回：类别数量字典 {0: 数量, 1: 数量, ..., 4: 数量}
    """
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # 固定5个类别
    logger.info("开始统计训练集类别数量（点级别）...")

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="统计训练集类别")):
        # 跳过空batch
        if batch is None or len(batch.get('path', [])) == 0:
            logger.warning(f"统计时跳过空batch {batch_idx}")
            continue

        # 获取当前batch的标签（点级别，字段为generate_label）
        labels = batch['generate_label'].cpu().numpy()  # 转移到CPU避免设备占用
        # 统计当前batch各类别数量
        batch_counts = np.bincount(labels, minlength=5)  # 确保返回5个类别（0-4）

        # 累加至总统计
        for cls in range(5):
            class_counts[cls] += batch_counts[cls]

    # 计算各类别占比
    total_points = sum(class_counts.values())
    logger.info("\n" + "=" * 60)
    logger.info("训练集类别数量统计结果（点级别）")
    logger.info("=" * 60)
    for cls, count in class_counts.items():
        ratio = (count / total_points) * 100 if total_points > 0 else 0
        if cls == 0:
            cls_name = "无风切变"
        elif cls == 1:
            cls_name = "轻微风切变"
        elif cls == 2:
            cls_name = "中度风切变"
        elif cls == 3:
            cls_name = "重度风切变"
        else:  # cls ==4
            cls_name = "严重风切变"
        logger.info(f"类别{cls}（{cls_name}）：{count:,} 个点（占比：{ratio:.2f}%）")
    logger.info(f"训练集总点数：{total_points:,}")
    logger.info("=" * 60)

    return class_counts

def plot_loss_curve(epochs, train_loss, val_loss, save_dir):
    """绘制训练/验证损失曲线并保存"""
    plt.figure(figsize=(10, 6))
    # 绘制训练损失 → 标签改为 "Train Loss"
    plt.plot(epochs, train_loss, color='#e74c3c', linewidth=2.5, marker='o', markersize=4, label='Train Loss')
    # 绘制验证损失 → 标签改为 "Val Loss"
    plt.plot(epochs, val_loss, color='#3498db', linewidth=2.5, marker='s', markersize=4, label='Val Loss')

    plt.title('Training and Validation Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, len(epochs) + 1, step=5))

    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_curve(epochs, train_f1, val_f1, save_dir):
    """绘制训练/验证F1曲线并保存"""
    plt.figure(figsize=(10, 6))
    # 绘制训练F1 → 标签改为 "Train F1"
    plt.plot(epochs, train_f1, color='#2ecc71', linewidth=2.5, marker='o', markersize=4, label='Train F1')
    # 绘制验证F1 → 标签改为 "Val F1"
    plt.plot(epochs, val_f1, color='#f39c12', linewidth=2.5, marker='s', markersize=4, label='Val F1')

    plt.title('Training and Validation F1 Score Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, len(epochs) + 1, step=5))

    save_path = os.path.join(save_dir, 'f1_curve.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics_gpu(logits, labels, criterion, num_classes=5):
    """
    GPU向量化计算指标（替代sklearn，无CPU转移，无循环）
    返回：当前batch的损失、TP、FP、FN、总点数
    """
    # 1. 计算当前batch的损失（带权重，与训练一致）
    loss = criterion(logits, labels)  # criterion已在main中定义（带类别权重）

    # 2. 预测结果（GPU上直接计算，不转移CPU）
    preds = torch.argmax(logits, dim=1)  # (N,)
    N = labels.shape[0]  # 当前batch的总点数

    # 3. 向量化计算混淆矩阵（GPU上用bincount，比sklearn快100倍）
    # 原理：用 (labels * num_classes + preds) 生成唯一索引，统计每个索引的数量
    confusion = torch.bincount(
        labels * num_classes + preds,
        minlength=num_classes * num_classes
    ).view(num_classes, num_classes)  # (num_classes, num_classes)

    # 4. 计算TP、FP、FN（GPU张量，无需循环）
    tp = torch.diag(confusion)  # 对角线上是TP（每个类的正确预测数）
    fp = confusion.sum(dim=1) - tp  # 行和 - TP = FP（预测对但真实错）
    fn = confusion.sum(dim=0) - tp  # 列和 - TP = FN（真实对但预测错）

    return {
        "loss": loss * N,  # 累计损失（乘以点数，后续求平均）
        "tp": tp,  # 每类TP（GPU张量）
        "fp": fp,  # 每类FP（GPU张量）
        "fn": fn,  # 每类FN（GPU张量）
        "total_points": N  # 当前batch总点数
    }


def main(config_path):
    # -------------------------- 1. 加载配置文件 --------------------------
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 初始化日志
    logger = get_logger('wind_shear_train', log_dir='./logs')
    logger.info(f"使用配置文件: {config_path}")
    logger.debug(f"配置详情: {cfg}")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # -------------------------- 2. 初始化数据集和DataLoader --------------------------
    # 关键：确保训练/验证用同一个collate_fn
    train_loader = build_train_dataloader(cfg)
    val_loader = build_val_dataloader(cfg)

    # 【调试】验证collate_fn是否生效
    try:
        train_iter = iter(train_loader)
        first_batch = next(train_iter)
        if first_batch is not None:
            logger.info("\n训练集第一个batch字段验证：")
            for key in first_batch:
                if isinstance(first_batch[key], torch.Tensor):
                    logger.info(f"  {key}: shape {first_batch[key].shape}, dtype {first_batch[key].dtype}")
                else:
                    logger.info(f"  {key}: type {type(first_batch[key])}")
        else:
            logger.warning("第一个batch为空，可能所有样本均被过滤")
    except Exception as e:
        logger.error(f"打印第一个batch失败: {e}")

    logger.info(f"训练集样本数: {len(train_loader.dataset)}, 验证集样本数: {len(val_loader.dataset)}")

    # -------------------------- 3. 统计训练集类别数量 + 计算类别权重 --------------------------
    # 步骤1：统计训练集各类别数量（点级别）
    #train_class_counts = count_train_classes(train_loader)
    #train_class_counts = {0: 201761, 1: 32251009, 2: 3509758, 3: 692463, 4: 1064945}
    #total_points = 37719936
    #num_classes = 5

    # 2. 计算逆频率权重
    #inverse_weights = []
    #for cls in range(num_classes):
    #    n_c = train_class_counts[cls]
    #    w_c = total_points / (num_classes * n_c)  # 核心公式
    #    inverse_weights.append(w_c)

    # 3. （可选）权重归一化（避免权重过大导致梯度爆炸）
    #max_weight = max(inverse_weights)
    #inverse_weights = [w / max_weight for w in inverse_weights]  # 归一到0~1

    #直接使用（1-3）计算好的结果
    #inverse_weights = [1.0, 0.006256, 0.057486, 0.291367, 0.189457]
    #权重设计与业务目标（风切变检测的核心任务）强绑定
    inverse_weights = [0.05, 0.15, 0.3, 0.6, 0.45]


    # 4. 转换为GPU张量
    weight_tensor = torch.tensor(inverse_weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # 转换为torch张量并移动到设备
    logger.info("\n" + "=" * 60)
    logger.info("最终类别权重")
    logger.info("=" * 60)
    for cls in range(5):
        if cls == 0:
            cls_name = "无风切变"
        elif cls == 1:
            cls_name = "轻度风切变"
        elif cls == 2:
            cls_name = "中度风切变"
        elif cls == 3:
            cls_name = "强烈风切变"
        else:
            cls_name = "严重风切变"
        logger.info(
            f"类别{cls}（{cls_name}）：权重={inverse_weights[cls]:.6f} ")
    logger.info("=" * 60)

    # -------------------------- 4. 初始化模型、优化器、损失函数（应用权重） --------------------------
    model = build_model(cfg['model']).to(device)
    logger.info(f"模型类型: {model.__class__.__name__}")

    # 优化器（AdamW，带权重衰减）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        weight_decay=cfg['train']['optimizer']['weight_decay']
    )

    # 学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs']
    )

    # 🌟 1. 初始化数据记录列表（存储每个epoch的指标）
    train_losses = []  # 训练损失
    train_f1s = []  # 训练F1
    val_losses = []  # 验证损失
    val_f1s = []  # 验证F1
    epochs_list = []  # epoch序号（用于x轴）

    # 🌟 2. 创建图片保存文件夹（不存在则自动创建）
    plot_save_dir = "./logs_photo/plots"
    os.makedirs(plot_save_dir, exist_ok=True)  # 自动创建多级目录

    # -------------------------- 5. 训练循环 --------------------------
    best_val_f1 = 0.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        epochs_list.append(epoch)  # 记录当前epoch
        logger.info(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")

        # -------------------------- 5.1 训练阶段 --------------------------
        model.train()
        # 初始化GPU张量用于累计（替代list，避免CPU转移）
        train_tp = torch.zeros(5, dtype=torch.long, device=device)  # 每类TP
        train_fp = torch.zeros(5, dtype=torch.long, device=device)  # 每类FP
        train_fn = torch.zeros(5, dtype=torch.long, device=device)  # 每类FN
        train_total_loss = 0.0  # 累计损失（带权重）
        train_total_points = 0  # 累计总点数
        abnormal_train_batches = []
        total_train_batches = 0
        normal_train_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="训练中")):
            total_train_batches += 1
            if batch is None or len(batch['path']) == 0:
                logger.warning(f"跳过空训练batch {batch_idx}")
                continue

            # 转移batch到设备（不变）
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                elif k == 'path':
                    batch_device[k] = v
            batch = batch_device
            labels = batch['generate_label'].long()  # (N,)
            logits = model(batch)  # (N, 5)

            # 异常检测（不变）
            loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                sample_paths = [os.path.basename(p) for p in batch.get('path', ['未知路径'])]
                abnormal_info = {"batch_idx": batch_idx, "sample_paths": sample_paths,
                                 "loss_value": loss.item() if not torch.isnan(loss) else "nan",
                                 "points_count": labels.shape[0]}
                abnormal_train_batches.append(abnormal_info)
                logger.error(
                    f"❌ 训练批次 {batch_idx} 异常: loss={abnormal_info['loss_value']}, 样本路径={sample_paths}, 点数={labels.shape[0]}")
                continue

            # 正常批次：反向传播（不变）
            normal_train_batches += 1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # 🌟 关键修改：传入 criterion 到 calculate_metrics_gpu
            metrics = calculate_metrics_gpu(logits, labels, criterion)
            train_total_loss += metrics["loss"].item()  # 累计损失（转为float避免显存占用）
            train_tp += metrics["tp"]
            train_fp += metrics["fp"]
            train_fn += metrics["fn"]
            train_total_points += metrics["total_points"]

        # 🌟 训练指标计算（GPU上完成，秒级）
        train_avg_loss = train_total_loss / train_total_points if train_total_points > 0 else 0.0
        # 计算每类精确率、召回率、F1（避免除0）
        epsilon = 1e-6
        train_precision = train_tp / (train_tp + train_fp + epsilon)  # (5,)
        train_recall = train_tp / (train_tp + train_fn + epsilon)  # (5,)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + epsilon)  # (5,)
        # 加权平均（按每类样本数加权，与sklearn的average='weighted'一致）
        class_counts = train_tp + train_fn  # 每类真实样本数（TP+FN）
        total_counts = class_counts.sum()
        train_weighted_f1 = (train_f1 * class_counts).sum() / (total_counts + epsilon)
        train_weighted_precision = (train_precision * class_counts).sum() / (total_counts + epsilon)
        train_weighted_recall = (train_recall * class_counts).sum() / (total_counts + epsilon)

        # 🌟 记录训练指标
        train_losses.append(train_avg_loss if train_total_points > 0 else 0.0)
        train_f1s.append(train_weighted_f1.item() if train_total_points > 0 else 0.0)

        # 日志打印（不变，仅将GPU张量转为CPU数值）
        if train_total_points > 0:
            logger.info(
                f"训练集: 损失={train_avg_loss:.4f}, "
                f"召回率={train_weighted_recall.item():.4f}, 精确率={train_weighted_precision.item():.4f}, F1={train_weighted_f1.item():.4f}"
            )
        else:
            logger.warning("本epoch无有效训练样本，跳过训练指标计算")

        # -------------------------- 5.2 验证阶段 --------------------------
        if epoch % cfg['evaluation']['interval'] == 0:
            model.eval()
            # 初始化GPU张量累计（替代list）
            val_tp = torch.zeros(5, dtype=torch.long, device=device)
            val_fp = torch.zeros(5, dtype=torch.long, device=device)
            val_fn = torch.zeros(5, dtype=torch.long, device=device)
            val_total_loss = 0.0
            val_total_points = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证中")):
                    if batch is None:
                        logger.warning(f"跳过空验证batch {batch_idx}")
                        continue

                    # 转移batch到设备（不变）
                    batch_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_device[k] = v.to(device)
                        elif k == 'path':
                            batch_device[k] = v
                    batch = batch_device
                    labels = batch['generate_label'].long()
                    logits = model(batch)

                    # 🌟 传入 criterion 到 calculate_metrics_gpu
                    metrics = calculate_metrics_gpu(logits, labels, criterion)
                    val_total_loss += metrics["loss"].item()
                    val_tp += metrics["tp"]
                    val_fp += metrics["fp"]
                    val_fn += metrics["fn"]
                    val_total_points += metrics["total_points"]

                    # 异常检测（不变）
                    if torch.isnan(metrics["loss"]) or torch.isinf(metrics["loss"]):
                        sample_paths = [os.path.basename(p) for p in batch.get('path', ['未知路径'])]
                        logger.error(
                            f"❌ 验证批次 {batch_idx} 异常: loss={metrics['loss'].item() if not torch.isnan(metrics['loss']) else 'nan'}, 样本路径={sample_paths}, 点数={labels.shape[0]}")

            # 🌟 验证指标计算（GPU上完成）
            val_avg_loss = val_total_loss / val_total_points if val_total_points > 0 else 0.0
            val_precision = val_tp / (val_tp + val_fp + epsilon)
            val_recall = val_tp / (val_tp + val_fn + epsilon)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + epsilon)
            val_class_counts = val_tp + val_fn
            val_total_counts = val_class_counts.sum()
            val_weighted_f1 = (val_f1 * val_class_counts).sum() / (val_total_counts + epsilon)
            val_weighted_precision = (val_precision * val_class_counts).sum() / (val_total_counts + epsilon)
            val_weighted_recall = (val_recall * val_class_counts).sum() / (val_total_counts + epsilon)

            # 🌟 记录验证指标
            val_losses.append(val_avg_loss if val_total_points > 0 else 0.0)
            val_f1s.append(val_weighted_f1.item() if val_total_points > 0 else 0.0)

            # 日志打印+最佳模型保存（不变，仅替换指标变量）
            if val_total_points > 0:
                logger.info(
                    f"验证集: 损失={val_avg_loss:.4f}, "
                    f"召回率={val_weighted_recall.item():.4f}, 精确率={val_weighted_precision.item():.4f}, F1={val_weighted_f1.item():.4f}"
                )
                if val_weighted_f1.item() > best_val_f1:
                    best_val_f1 = val_weighted_f1.item()
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                    save_path=f"./checkpoints/best_model_epoch{epoch}.pth")
                    logger.info(f"✅ 保存最佳模型 (F1={best_val_f1:.4f}) 到 ./checkpoints/")
            else:
                logger.warning("本epoch无有效验证样本，跳过验证指标计算和模型保存")
        else:
            # 不执行验证时，向 val_losses/val_f1s 追加默认值（保证列表长度一致）
            val_losses.append(0.0)
            val_f1s.append(0.0)

        # 🌟 4. 绘制并保存曲线（每个epoch更新）
        if epoch % 1 == 0:
            plot_loss_curve(epochs_list, train_losses, val_losses, plot_save_dir)
            plot_f1_curve(epochs_list, train_f1s, val_f1s, plot_save_dir)

        # 学习率调度器步进
        scheduler.step()

    logger.info(f"\n训练完成！最佳验证集F1分数: {best_val_f1:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/wind_shear/pointtransformer_v3.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    # 预先创建日志和检查点目录
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    main(args.config)