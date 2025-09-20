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
from sklearn.metrics import recall_score, precision_score, f1_score

# 导入自定义模块
from pointcept.datasets.builder import build_train_dataloader, build_val_dataloader
from pointcept.models import build_model
from pointcept.utils.logger import get_logger
from pointcept.utils.checkpoint import save_checkpoint
from pointcept.utils.logging import setup_logging  # 导入工具函数
import warnings


# 1. 配置全局日志（只调用1次！）
logger = setup_logging(log_dir="./logs")  # 日志文件存到项目根目录的logs文件夹

def main(config_path):
    # --------------------------
    # 1. 加载配置文件
    # --------------------------
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 初始化日志
    logger = get_logger('wind_shear_train', log_dir='./logs')
    logger.info(f"使用配置文件: {config_path}")
    logger.debug(f"配置详情: {cfg}")

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # --------------------------
    # 2. 初始化数据集和DataLoader
    # --------------------------
    train_loader = build_train_dataloader(cfg)
    val_loader = build_val_dataloader(cfg)

    # 【调试】验证collate_fn是否生效
    try:
        train_iter = iter(train_loader)
        first_batch = next(train_iter)
        if first_batch is None:
            logger.warning("第一个batch为空，可能所有样本均被过滤")
        else:
            logger.info("First batch structure (验证collate_fn输出):")
            for key in first_batch:
                if isinstance(first_batch[key], torch.Tensor):
                    logger.info(f"  {key}: shape {first_batch[key].shape}, dtype {first_batch[key].dtype}")
                else:
                    logger.info(f"  {key}: type {type(first_batch[key])}")
    except Exception as e:
        logger.error(f"打印第一个batch失败: {e}")

    logger.info(f"训练集样本数: {len(train_loader.dataset)}, 验证集样本数: {len(val_loader.dataset)}")

    # --------------------------
    # 3. 初始化模型、优化器、损失函数
    # --------------------------
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

    # 损失函数（处理类别不平衡）
    pos_weight = torch.tensor([25.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --------------------------
    # 4. 训练循环
    # --------------------------
    best_val_f1 = 0.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        logger.info(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")

        # --------------------------
        # 4.1 训练阶段
        # --------------------------
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        total_train_points = 0  # 用于计算平均损失的实际总点数

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="训练中")):
            # 关键修改：跳过空batch
            if batch is None:
                logger.warning(f"跳过空训练batch {batch_idx}（无有效样本）")
                continue

            # 转移batch到设备
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch['label'].float()
            current_points = batch['coord'].size(0)  # 当前batch的实际点数
            total_train_points += current_points

            # 前向传播 + 反向传播
            optimizer.zero_grad()
            logits = model(batch)  # 关键：现在outputs直接是logits tensor
            # 调整维度：(N,1) → (N,)，适配labels的形状
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # 累计损失和预测结果
            train_loss += loss.item() * current_points
            train_preds.extend((logits.squeeze() > 0.0).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 计算训练集指标（处理可能的空训练集）
        if total_train_points == 0:
            logger.warning("本epoch无有效训练样本，跳过训练指标计算")
            train_loss, train_recall, train_precision, train_f1 = 0.0, 0.0, 0.0, 0.0
        else:
            train_loss /= total_train_points
            train_recall = recall_score(train_labels, train_preds, zero_division=0)
            train_precision = precision_score(train_labels, train_preds, zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        logger.info(
            f"训练集: 损失={train_loss:.4f}, "
            f"召回率={train_recall:.4f}, 精确率={train_precision:.4f}, F1={train_f1:.4f}"
        )

        # --------------------------
        # 4.2 验证阶段
        # --------------------------
        if epoch % cfg['evaluation']['interval'] == 0:
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            total_val_points = 0

            with torch.no_grad():
                for batch_idx, batch in tqdm(val_loader, desc="验证中"):
                    # 关键修改：跳过空batch
                    if batch is None:
                        logger.warning(f"跳过空验证batch {batch_idx}（无有效样本）")
                        continue

                    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    labels = batch['label'].float()
                    current_points = batch['coord'].size(0)
                    total_val_points += current_points

                    outputs = model(batch)
                    loss = criterion(outputs.squeeze(), labels)

                    val_loss += loss.item() * current_points
                    val_preds.extend((outputs.squeeze() > 0.0).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # 计算验证集指标（处理可能的空验证集）
            if total_val_points == 0:
                logger.warning("本epoch无有效验证样本，跳过验证指标计算")
                val_loss = 0.0
                val_recall = val_precision = val_f1 = 0.0
            else:
                val_loss /= total_val_points
                val_recall = recall_score(val_labels, val_preds, zero_division=0)
                val_precision = precision_score(val_labels, val_preds, zero_division=0)
                val_f1 = f1_score(val_labels, val_preds, zero_division=0)

            logger.info(
                f"验证集: 损失={val_loss:.4f}, "
                f"召回率={val_recall:.4f}, 精确率={val_precision:.4f}, F1={val_f1:.4f}"
            )

            # 保存最佳模型（处理无有效验证样本的情况）
            if total_val_points > 0 and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    save_path=f"./checkpoints/best_model_epoch{epoch}.pth"
                )
                logger.info(f"保存最佳模型 (F1={best_val_f1:.4f}) 到 ./checkpoints/")
            elif total_val_points == 0:
                logger.warning("未保存模型：无有效验证样本")

        # 学习率调度器步进
        scheduler.step()

    logger.info(f"训练完成！最佳验证集F1分数: {best_val_f1:.4f}")


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
