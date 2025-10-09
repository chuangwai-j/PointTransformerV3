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
import warnings

# 导入自定义模块
from pointcept.datasets.builder import build_train_dataloader, build_val_dataloader
from pointcept.models import build_model
from pointcept.utils.logger import get_logger
from pointcept.utils.checkpoint import save_checkpoint
from pointcept.utils.logging import setup_logging  # 导入工具函数


# 1. 配置全局日志（只调用1次！）
logger = setup_logging(log_dir="./logs")  # 日志文件存到项目根目录的logs文件夹

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
    # 关键：确保训练/验证用同一个collate_fn（需在build_train_dataloader/build_val_dataloader中指定）
    train_loader = build_train_dataloader(cfg)
    val_loader = build_val_dataloader(cfg)

    # 【调试】验证collate_fn是否生效
    try:
        train_iter = iter(train_loader)
        first_batch = next(train_iter)
        if first_batch is None:
            logger.warning("第一个batch为空，可能所有样本均被过滤")
            for key in first_batch:
                if isinstance(first_batch[key], torch.Tensor):
                    logger.info(f"  {key}: shape {first_batch[key].shape}, dtype {first_batch[key].dtype}")
                else:
                    logger.info(f"  {key}: type {type(first_batch[key])}")
    except Exception as e:
        logger.error(f"打印第一个batch失败: {e}")

    logger.info(f"训练集样本数: {len(train_loader.dataset)}, 验证集样本数: {len(val_loader.dataset)}")

    # -------------------------- 3. 初始化模型、优化器、损失函数 --------------------------
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

    # 多分类损失函数（CrossEntropyLoss适用于类别互斥的多分类）
    criterion = torch.nn.CrossEntropyLoss()

    # -------------------------- 4. 训练循环 --------------------------
    best_val_f1 = 0.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        logger.info(f"\n===== Epoch {epoch}/{cfg['train']['epochs']} =====")

        # -------------------------- 4.1 训练阶段 --------------------------
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        total_train_points = 0  # 用于计算平均损失的实际总点数
        # 新增：异常批次统计变量
        abnormal_train_batches = []  # 记录异常批次信息
        total_train_batches = 0  # 总训练批次
        normal_train_batches = 0  # 正常训练批次

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="训练中")):
            # 🌟 关键：打印传入模型前的 batch 字段
            #print(f"train.py 中 batch 的字段：{list(batch.keys())}")  # 必须加这行！
            #print(f"train.py 中 batch 的 path：{batch.get('path', '无')}")  # 查看 path 是否存在
            total_train_batches += 1  # 累计总批次
            # 关键修改：跳过空batch
            if batch is None or len(batch['path']) == 0:
                logger.warning(f"跳过空训练batch {batch_idx}（无有效样本）")
                continue

            # 转移batch到设备
            #batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            # 🌟 修改后：保留path，同时将张量转移到设备
            batch_device = {}
            # 先处理张量字段（转移到设备）
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                # 显式保留path（非张量）
                elif k == 'path':
                    batch_device[k] = v
            # 用处理后的batch替代原batch
            batch = batch_device
            labels = batch['generate_label'].long()  # 多分类标签需为长整数类型
            current_points = batch['coord'].size(0)  # 当前batch的实际点数

            # 前向传播+反向传播
            optimizer.zero_grad()
            logits = model(batch)  # 关键：现在outputs直接是logits tensor
            loss = criterion(logits, labels)

            # 异常检测
            if torch.isnan(loss) or torch.isinf(loss):
                sample_paths = [os.path.basename(p) for p in batch.get('path', ['未知路径'])]
                abnormal_info = {
                    "batch_idx": batch_idx,
                    "sample_paths": sample_paths,
                    "loss_value": loss.item() if not torch.isnan(loss) else "nan",
                    "points_count": current_points
                }
                abnormal_train_batches.append(abnormal_info)
                logger.error(
                    f"❌ 训练批次 {batch_idx} 异常: loss={abnormal_info['loss_value']}, 样本路径={sample_paths}, 点数={current_points}")
                continue

            # 正常批次：更新参数+累计指标
            normal_train_batches += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # 梯度裁剪
            optimizer.step()

            train_loss += loss.item() * current_points
            total_train_points += current_points
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())  # 多分类取argmax
            train_labels.extend(labels.cpu().numpy())

        # 训练阶段汇总
        logger.info("\n===== 训练阶段批次统计 =====")
        logger.info(
            f"总批次: {total_train_batches}, 正常批次: {normal_train_batches}, 异常批次: {len(abnormal_train_batches)}")
        if abnormal_train_batches:
            logger.error(f"异常批次索引列表: {[info['batch_idx'] for info in abnormal_train_batches]}")
            logger.error(f"首个异常批次详情: {abnormal_train_batches[0]}")
        else:
            logger.info("✅ 所有训练批次均正常")

        # 计算训练指标
        if total_train_points > 0:
            train_loss /= total_train_points
            train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            logger.info(
                f"训练集: 损失={train_loss:.4f}, "
                f"召回率={train_recall:.4f}, 精确率={train_precision:.4f}, F1={train_f1:.4f}"
            )
        else:
            logger.warning("本epoch无有效训练样本，跳过训练指标计算")

        # -------------------------- 4.2 验证阶段（核心修复） --------------------------
        if epoch % cfg['evaluation']['interval'] == 0:
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            total_val_points = 0
            first_val_batch = True

            with torch.no_grad():   # 关闭梯度，不影响数值计算
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证中")):
                    # 关键修改：跳过空batch
                    if batch is None:
                        logger.warning(f"跳过空验证batch {batch_idx}（无有效样本）")
                        continue

                    # 1. 转移设备+基础信息
                    #batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    # 🌟 修改后：保留path，同时将张量转移到设备
                    batch_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_device[k] = v.to(device)
                        elif k == 'path':
                            batch_device[k] = v
                    batch = batch_device
                    labels = batch['generate_label'].long()  # 多分类标签需为长整数类型
                    current_points = batch['coord'].size(0)
                    sample_paths = [os.path.basename(p) for p in batch.get('path', ['未知路径'])]

                    # 2. 只调用一次模型前向（修复重复调用bug）
                    logits = model(batch)
                    '''
                    # 3. 第一个验证batch：增强调试日志（对比训练集）
                    if first_val_batch:
                        logger.info("\n=== 验证集第一个batch关键信息（与训练集对比） ===")
                        # 打印logits统计
                        logger.info(
                            f"logits形状: {logits.shape}, 最小值: {logits.min().item():.4f}, 最大值: {logits.max().item():.4f}")
                        logger.info(
                            f"logits含nan: {torch.isnan(logits).any().item()}, 含inf: {torch.isinf(logits).any().item()}")
                        # 打印coord范围（关键：对比训练集是否一致）
                        coord_min = batch['coord'].min(axis=0).values
                        coord_max = batch['coord'].max(axis=0).values
                        logger.info(
                            f"coord范围: x[{coord_min[0]:.0f}~{coord_max[0]:.0f}], y[{coord_min[1]:.0f}~{coord_max[1]:.0f}], z[{coord_min[2]:.0f}~{coord_max[2]:.0f}]")
                        # 打印spatial_shape（临时计算：coord_max+1，后续模型返回后可替换）
                        spatial_shape = [int(coord_max[2].item()) + 1, int(coord_max[1].item()) + 1,
                                         int(coord_max[0].item()) + 1]  # z/y/x
                        logger.info(f"临时计算spatial_shape: {spatial_shape}（若某维度>2000，需限制coord范围）")
                        logger.info(f"标签范围: {labels.min().item()}~{labels.max().item()}, 样本路径: {sample_paths}")
                        first_val_batch = False
                    '''
                    # 4. 累计验证指标（修复未累计bug）
                    loss = criterion(logits, labels)  # 无多余squeeze()
                    val_loss += loss.item() * current_points
                    total_val_points += current_points
                    val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    # 5. 验证批次异常检测
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(
                            f"❌ 验证批次 {batch_idx} 异常: loss={loss.item() if not torch.isnan(loss) else 'nan'}, 样本路径={sample_paths}, 点数={current_points}")

            # 计算验证指标
            logger.info("\n===== 验证阶段汇总 =====")
            if total_val_points > 0:
                val_loss /= total_val_points
                val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
                val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
                val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
                logger.info(
                    f"验证集: 损失={val_loss:.4f}, "
                    f"召回率={val_recall:.4f}, 精确率={val_precision:.4f}, F1={val_f1:.4f}"
                )

                # 保存最佳模型
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        save_path=f"./checkpoints/best_model_epoch{epoch}.pth"
                    )
                    logger.info(f"✅ 保存最佳模型 (F1={best_val_f1:.4f}) 到 ./checkpoints/")
            else:
                logger.warning("本epoch无有效验证样本，跳过验证指标计算和模型保存")

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
