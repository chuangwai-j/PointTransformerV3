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
import warnings
from pointcept.utils.losses import MixedLoss

os.environ['MPLBACKEND'] = 'Agg'  # 全局强制使用无GUI后端，避免tkinter冲突

# 2. 再导入matplotlib
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


# 导入自定义模块
from pointcept.datasets.builder import build_train_dataloader, build_val_dataloader
from pointcept.models import build_model
from pointcept.utils.logger import get_logger
from pointcept.utils.checkpoint import save_checkpoint
from pointcept.utils.logging import setup_logging  # 导入工具函数

# 1. 配置全局日志（只调用1次！）
logger = setup_logging(log_dir="./logs")  # 日志文件存到项目根目录的logs文件夹


# -------------------------- 新增：早停策略类 --------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max', warmup=5):
        """
        早停策略
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善幅度
            mode: 'max' 表示指标越大越好, 'min' 表示越小越好
            warmup: 前几个epoch不进行早停判断
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.warmup = warmup
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, epoch):
        if epoch < self.warmup:
            return False

        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'max':
            improvement = current_score - self.best_score
        else:
            improvement = self.best_score - current_score

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'早停计数器: {self.counter}/{self.patience}')

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


def plot_loss_curve(epochs, train_loss, val_loss, save_dir):
    """绘制训练/验证损失曲线并保存"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, color='#e74c3c', linewidth=2.5, marker='o', markersize=4, label='Train Loss')
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
    plt.plot(epochs, train_f1, color='#2ecc71', linewidth=2.5, marker='o', markersize=4, label='Train F1')
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


def plot_lr_curve(epochs, learning_rates, save_dir):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, color='#9b59b6', linewidth=2.5, marker='o', markersize=4)

    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, len(epochs) + 1, step=5))

    save_path = os.path.join(save_dir, 'lr_curve.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_metrics_gpu(logits, labels, criterion, num_classes=5):
    """
    GPU向量化计算指标
    """
    loss = criterion(logits, labels)
    preds = torch.argmax(logits, dim=1)
    N = labels.shape[0]

    confusion = torch.bincount(
        labels * num_classes + preds,
        minlength=num_classes * num_classes
    ).view(num_classes, num_classes)

    tp = torch.diag(confusion)
    fp = confusion.sum(dim=1) - tp
    fn = confusion.sum(dim=0) - tp

    return {
        "loss": loss * N,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_points": N
    }


def main(config_path):
    # -------------------------- 1. 加载配置文件 --------------------------
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    logger = get_logger('wind_shear_train', log_dir='./logs')
    logger.info(f"使用配置文件: {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # -------------------------- 2. 初始化数据集和DataLoader --------------------------
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

    # -------------------------- 3. 优化后的类别权重设置 --------------------------
    # 🌟 使用更合理的权重设置
    weights = cfg['train']['class_weights']
    #inverse_weights = [0.05, 0.15, 0.3, 0.6, 0.45]

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    #criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # 2. 实例化混合损失（num_classes=5对应你的5分类任务，alpha=weight_tensor复用原权重）
    criterion = MixedLoss(num_classes=5, alpha=weight_tensor, gamma=2.0, focal_weight=1.0, dice_weight=1.0)

    logger.info("\n" + "=" * 60)
    logger.info("优化后的类别权重")
    logger.info("=" * 60)
    class_names = ["无风切变", "轻微风切变", "中度风切变", "重度风切变", "严重风切变"]
    for cls in range(5):
        logger.info(f"类别{cls}（{class_names[cls]}）：权重={weights[cls]:.6f}")
    logger.info("=" * 60)

    # -------------------------- 4. 初始化模型、优化器、学习率调度器 --------------------------
    model = build_model(cfg['model']).to(device)
    logger.info(f"模型类型: {model.__class__.__name__}")

    # 🌟 ===================== 新增代码开始 ===================== 🌟
    # 打印模型配置参数，这对于测试时复现模型至关重要
    logger.info("\n" + "=" * 60)
    logger.info("模型配置参数 (cfg['model']):")
    logger.info("=" * 60)
    for key, value in cfg['model'].items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # 打印模型完整结构
    logger.info("\n" + "=" * 60)
    logger.info("模型完整结构 (Model Structure):")
    logger.info("=" * 60)
    logger.info(str(model))  # str(model) 将捕获完整的 PyTorch 结构
    logger.info("=" * 60)
    # 🌟 ===================== 新增代码结束 ===================== 🌟

    # 🌟 从配置读取学习率设置
    initial_lr = cfg['train']['optimizer']['lr']

    # 优化器（AdamW，带权重衰减）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=cfg['train']['optimizer']['weight_decay']
    )

    # 🌟 使用传统的余弦退火调度器（100轮缓慢下降）
    scheduler_cfg = cfg['train']['scheduler']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_cfg['T_max'],
        eta_min=scheduler_cfg['eta_min']
    )

    # 🌟 新增：早停策略
    early_stopping = EarlyStopping(
        patience=15,  # 容忍15个epoch没有改善
        min_delta=0.001,  # 最小改善幅度
        mode='max',  # 监控验证F1（越大越好）
        warmup=10  # 前10个epoch不进行早停判断
    )

    # 🌟 初始化数据记录列表
    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []
    epochs_list = []
    learning_rates = []  # 记录学习率变化
    plot_save_dir = "./logs_photo/plots"
    os.makedirs(plot_save_dir, exist_ok=True)

    # -------------------------- 5. 训练循环 --------------------------
    best_val_f1 = 0.0
    total_epochs = cfg['train']['epochs']

    for epoch in range(1, total_epochs + 1):
        epochs_list.append(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        logger.info(f"\n===== Epoch {epoch}/{total_epochs} =====")
        logger.info(f"当前学习率: {current_lr:.6f}")

        # -------------------------- 5.1 训练阶段 --------------------------
        model.train()
        train_tp = torch.zeros(5, dtype=torch.long, device=device)
        train_fp = torch.zeros(5, dtype=torch.long, device=device)
        train_fn = torch.zeros(5, dtype=torch.long, device=device)
        train_total_loss = 0.0
        train_total_points = 0
        abnormal_train_batches = []
        total_train_batches = 0
        normal_train_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="训练中")):
            total_train_batches += 1
            if batch is None or len(batch['path']) == 0:
                logger.warning(f"跳过空训练batch {batch_idx}")
                continue

            # 转移batch到设备
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                elif k == 'path':
                    batch_device[k] = v
            batch = batch_device
            labels = batch['generate_label'].long()
            logits = model(batch)

            # 异常检测
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

            # 正常批次：反向传播
            normal_train_batches += 1
            optimizer.zero_grad()
            loss.backward()

            # 🌟 改进的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # 计算指标
            metrics = calculate_metrics_gpu(logits, labels, criterion)
            train_total_loss += metrics["loss"].item()
            train_tp += metrics["tp"]
            train_fp += metrics["fp"]
            train_fn += metrics["fn"]
            train_total_points += metrics["total_points"]

        # 训练指标计算
        train_avg_loss = train_total_loss / train_total_points if train_total_points > 0 else 0.0
        epsilon = 1e-6
        train_precision = train_tp / (train_tp + train_fp + epsilon)
        train_recall = train_tp / (train_tp + train_fn + epsilon)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + epsilon)
        class_counts = train_tp + train_fn
        total_counts = class_counts.sum()
        train_weighted_f1 = (train_f1 * class_counts).sum() / (total_counts + epsilon)
        train_weighted_precision = (train_precision * class_counts).sum() / (total_counts + epsilon)
        train_weighted_recall = (train_recall * class_counts).sum() / (total_counts + epsilon)

        # 记录训练指标
        train_losses.append(train_avg_loss if train_total_points > 0 else 0.0)
        train_f1s.append(train_weighted_f1.item() if train_total_points > 0 else 0.0)

        if train_total_points > 0:
            logger.info(
                f"训练集: 损失={train_avg_loss:.4f}, "
                f"召回率={train_weighted_recall.item():.4f}, 精确率={train_weighted_precision.item():.4f}, F1={train_weighted_f1.item():.4f}"
            )

            # 🌟 记录各类别F1分数
            logger.info("各类别训练F1分数:")
            for cls in range(5):
                logger.info(f"  类别{cls}({class_names[cls]}): {train_f1[cls].item():.4f}")
        else:
            logger.warning("本epoch无有效训练样本，跳过训练指标计算")

        # -------------------------- 5.2 验证阶段 --------------------------
        if epoch % cfg['evaluation']['interval'] == 0:
            model.eval()
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

                    batch_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_device[k] = v.to(device)
                        elif k == 'path':
                            batch_device[k] = v
                    batch = batch_device
                    labels = batch['generate_label'].long()
                    logits = model(batch)

                    metrics = calculate_metrics_gpu(logits, labels, criterion)
                    val_total_loss += metrics["loss"].item()
                    val_tp += metrics["tp"]
                    val_fp += metrics["fp"]
                    val_fn += metrics["fn"]
                    val_total_points += metrics["total_points"]

            # 验证指标计算
            val_avg_loss = val_total_loss / val_total_points if val_total_points > 0 else 0.0
            val_precision = val_tp / (val_tp + val_fp + epsilon)
            val_recall = val_tp / (val_tp + val_fn + epsilon)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + epsilon)
            val_class_counts = val_tp + val_fn
            val_total_counts = val_class_counts.sum()
            val_weighted_f1 = (val_f1 * val_class_counts).sum() / (val_total_counts + epsilon)
            val_weighted_precision = (val_precision * val_class_counts).sum() / (val_total_counts + epsilon)
            val_weighted_recall = (val_recall * val_class_counts).sum() / (val_total_counts + epsilon)

            # 记录验证指标
            val_losses.append(val_avg_loss if val_total_points > 0 else 0.0)
            val_f1s.append(val_weighted_f1.item() if val_total_points > 0 else 0.0)

            if val_total_points > 0:
                logger.info(
                    f"验证集: 损失={val_avg_loss:.4f}, "
                    f"召回率={val_weighted_recall.item():.4f}, 精确率={val_weighted_precision.item():.4f}, F1={val_weighted_f1.item():.4f}"
                )

                # 🌟 记录各类别验证F1分数
                logger.info("各类别验证F1分数:")
                for cls in range(5):
                    logger.info(f"  类别{cls}({class_names[cls]}): {val_f1[cls].item():.4f}")

                # 保存最佳模型
                if val_weighted_f1.item() > best_val_f1:
                    best_val_f1 = val_weighted_f1.item()
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                    save_path=f"./checkpoints/best_model_epoch{epoch}.pth")
                    logger.info(f"✅ 保存最佳模型 (F1={best_val_f1:.4f}) 到 ./checkpoints/")

                # 🌟 早停判断
                if early_stopping(val_weighted_f1.item(), epoch):
                    logger.info(f"🚨 触发早停！最佳验证F1: {best_val_f1:.4f}")
                    break
            else:
                logger.warning("本epoch无有效验证样本，跳过验证指标计算和模型保存")
        else:
            # 不执行验证时，追加默认值
            val_losses.append(0.0)
            val_f1s.append(0.0)

        # 🌟 绘制曲线（每个epoch更新）
        if epoch % 1 == 0:
            plot_loss_curve(epochs_list, train_losses, val_losses, plot_save_dir)
            plot_f1_curve(epochs_list, train_f1s, val_f1s, plot_save_dir)
            plot_lr_curve(epochs_list, learning_rates, plot_save_dir)

        # 🌟 学习率调度器步进（放在每个epoch最后）
        scheduler.step()

        # 🌟 早停检查（如果触发则跳出循环）
        if early_stopping.early_stop:
            break

    logger.info(f"\n训练完成！最佳验证集F1分数: {best_val_f1:.4f}")
    logger.info(f"总训练轮数: {len(epochs_list)}/{total_epochs}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/wind_shear/pointtransformer_v3.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    # 预先创建日志和检查点目录
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs_photo/plots', exist_ok=True)

    main(args.config)