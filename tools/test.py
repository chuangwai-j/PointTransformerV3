import os
import argparse
import logging
import yaml
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import sys

# 解决模块导入问题：与train.py保持一致，添加项目根目录到Python搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入项目内模块（与train.py导入格式统一）
from pointcept.datasets.builder import build_test_dataloader  # 复用已实现的测试加载器
from pointcept.models import build_model  # 与train.py一致的模型构建函数
from pointcept.utils.logger import get_logger # 复用训练日志工具
from pointcept.utils.logging import setup_logging
from pointcept.utils.misc import AverageMeter

# 配置日志（与train.py保持一致，使用项目统一日志工具）
setup_logging(log_dir="./logs")  # 日志文件统一存到logs目录
logger = get_logger("wind_shear_test")  # 日志器名称与训练保持一致


def parse_args():
    """解析命令行参数：指定配置文件、模型权重、GPUID（与train.py风格一致）"""
    parser = argparse.ArgumentParser(description="PointTransformerV3 测试脚本")
    parser.add_argument(
        "--config",
        required=True,
        help="配置文件路径，例如：./configs/wind_shear/pointtransformer_v3.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="模型权重路径，例如：./checkpoints/best_model_epoch78.pth"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU设备号（-1表示使用CPU）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="测试批次大小（优先于配置文件）"
    )
    return parser.parse_args()


def load_config(config_path):
    """加载yaml配置文件（与train.py一致，使用safe_load，返回普通字典）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # 与train.py一致，使用safe_load更安全
    return config  # 不使用addict.Dict，与train.py的普通字典格式统一


def load_model_weight(model, checkpoint_path, device):
    """加载训练好的模型权重（适配 model_state_dict 键名，兼容训练保存格式）"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在：{checkpoint_path}")
    # 加载权重（处理CPU/GPU兼容）
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device  # 自动映射到当前设备
    )
    # 提取模型权重：优先用 model_state_dict（训练时常用键名），再用 model，最后用整个字典
    if "model_state_dict" in checkpoint:
        model_weights = checkpoint["model_state_dict"]
        logger.info("从 'model_state_dict' 键提取模型权重")
    elif "model" in checkpoint:
        model_weights = checkpoint["model"]
        logger.info("从 'model' 键提取模型权重")
    else:
        model_weights = checkpoint
        logger.info("从权重文件根目录提取模型权重（无嵌套键）")
    # 加载权重（strict=True 确保参数完全匹配，若仍报错需检查模型结构一致性）
    model.load_state_dict(model_weights, strict=True)
    logger.info(f"权重加载完成：{checkpoint_path}")
    return model


def calculate_metrics(logits, labels, num_classes=5):
    """优化版：与train.py一致，使用「加权平均」计算整体指标（按类别样本数加权）"""
    # 1. 计算损失（不变）
    loss = F.cross_entropy(logits, labels).item()
    # 2. 预测结果（不变）
    preds = torch.argmax(logits, dim=1)
    preds = preds.view(-1)  # 展平为1维（适配向量化计算）
    labels = labels.view(-1)  # 展平为1维

    # 3. 向量化计算混淆矩阵（不变）
    confusion_matrix = torch.bincount(
        labels * num_classes + preds,
        minlength=num_classes * num_classes
    ).view(num_classes, num_classes)  # 重塑为 (num_classes, num_classes)

    # 4. 向量化计算每类的 TP、FP、FN（不变）
    tp = torch.diag(confusion_matrix)  # 对角线=TP（真实=预测=cls）
    fp = confusion_matrix.sum(dim=0) - tp  # 列和 - TP = FP（预测=cls，真实≠cls）
    fn = confusion_matrix.sum(dim=1) - tp  # 行和 - TP = FN（真实=cls，预测≠cls）

    # 5. 计算每类精确率、召回率、F1（不变，保留每类细节）
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # 6. 关键修改：从「宏观平均」改为「加权平均」（与train.py一致）
    # 加权依据：每个类别的「真实样本数」（即 labels 中每个类别的数量）
    class_counts = torch.bincount(labels, minlength=num_classes)  # 每类的真实样本数
    total_samples = class_counts.sum()  # 总样本数

    # 加权平均精确率：(每类精确率 * 每类样本数) / 总样本数
    weighted_precision = (precision * class_counts).sum() / (total_samples + 1e-6)
    # 加权平均召回率：(每类召回率 * 每类样本数) / 总样本数
    weighted_recall = (recall * class_counts).sum() / (total_samples + 1e-6)
    # 加权平均F1：(每类F1 * 每类样本数) / 总样本数
    weighted_f1 = (f1 * class_counts).sum() / (total_samples + 1e-6)

    # 准确率（不变）
    accuracy = (preds == labels).float().mean().item()

    return {
        "loss": loss,
        "accuracy": accuracy,
        # 关键：返回「加权平均」指标（替换原来的宏观平均）
        "weighted_precision": weighted_precision.item(),
        "weighted_recall": weighted_recall.item(),
        "weighted_f1": weighted_f1.item(),
        "per_class_f1": f1.cpu().numpy(),  # 保留每类F1（方便分析少数类）
        "confusion_matrix": confusion_matrix.cpu().numpy()
    }

def test_one_epoch(model, dataloader, device, num_classes=5):
    """执行一轮测试，返回平均指标（同步更新为加权平均指标）"""
    # 初始化指标记录器（将 macro 改为 weighted）
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    precision_meter = AverageMeter()  # 现在记录加权精确率
    recall_meter = AverageMeter()     # 现在记录加权召回率
    f1_meter = AverageMeter()         # 现在记录加权F1
    per_class_f1_meters = [AverageMeter() for _ in range(num_classes)]
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="测试中")):
            # 数据移动到设备（不变）
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v
            batch = batch_device
            labels = batch["generate_label"].long()
            batch_size = labels.shape[0]

            # 模型前向传播（不变）
            logits = model(batch)

            # 计算指标（现在获取的是加权平均指标）
            metrics = calculate_metrics(logits, labels, num_classes)
            # 更新指标记录器（将 macro_xxx 改为 weighted_xxx）
            loss_meter.update(metrics["loss"], batch_size)
            accuracy_meter.update(metrics["accuracy"], batch_size)
            precision_meter.update(metrics["weighted_precision"], batch_size)  # 改这里
            recall_meter.update(metrics["weighted_recall"], batch_size)        # 改这里
            f1_meter.update(metrics["weighted_f1"], batch_size)                # 改这里
            # 更新每类F1（不变）
            for cls in range(num_classes):
                per_class_f1_meters[cls].update(metrics["per_class_f1"][cls], batch_size)
            # 累加混淆矩阵（不变）
            total_confusion_matrix += metrics["confusion_matrix"]

    # 返回结果（将 macro 改为 weighted）
    return {
        "avg_loss": loss_meter.avg,
        "avg_accuracy": accuracy_meter.avg,
        "avg_precision": precision_meter.avg,  # 现在是加权平均精确率
        "avg_recall": recall_meter.avg,        # 现在是加权平均召回率
        "avg_f1": f1_meter.avg,                # 现在是加权平均F1（与训练验证一致）
        "per_class_f1": [meter.avg for meter in per_class_f1_meters],
        "confusion_matrix": total_confusion_matrix
    }


def save_test_result(result, save_path, num_classes=5):
    """保存测试结果（格式与train.py的日志风格统一）"""
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + " 测试结果汇总（加权平均，与训练验证一致） " + "=" * 50 + "\n")
        f.write(f"平均损失：{result['avg_loss']:.4f}\n")
        f.write(f"平均准确率：{result['avg_accuracy']:.4f}\n")
        f.write(f"加权平均精确率：{result['avg_precision']:.4f}\n")
        f.write(f"加权平均召回率：{result['avg_recall']:.4f}\n")
        f.write(f"加权平均F1：{result['avg_f1']:.4f}\n")

        # 每类F1
        f.write("\n" + "=" * 50 + " 各类别F1分数 " + "=" * 50 + "\n")
        for cls in range(num_classes):
            f.write(f"类别{cls} F1分数：{result['per_class_f1'][cls]:.4f}\n")

        # 混淆矩阵
        f.write("\n" + "=" * 50 + " 混淆矩阵（行：真实类别，列：预测类别） " + "=" * 50 + "\n")
        cm = result["confusion_matrix"]
        for i in range(cm.shape[0]):
            f.write(f"真实类别{i}：" + " ".join([f"{int(x)}" for x in cm[i]]) + "\n")
    logger.info(f"测试结果已保存到：{save_path}")


def main():
    args = parse_args()

    # 设备设置（与train.py一致）
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"使用GPU设备：cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU设备（GPU不可用或指定--gpu=-1）")

    # 加载配置（与train.py一致的配置格式）
    config = load_config(args.config)
    num_classes = config["model"]["num_classes"]  # 从配置获取类别数（非硬编码）
    logger.info(f"测试配置加载完成：{args.config}，任务类别数：{num_classes}")

    # 构建测试数据集加载器（复用builder.py中的build_test_dataloader）
    test_dataloader = build_test_dataloader(
        cfg=config,
        batch_size=args.batch_size  # 支持外部传入batch_size
    )
    logger.info(f"测试 DataLoader 实际参数：num_workers={test_dataloader.num_workers}, "
                f"batch_size={test_dataloader.batch_size}, "
                f"pin_memory={test_dataloader.pin_memory}")
    logger.info(f"测试集加载完成：共{len(test_dataloader.dataset)}个样本")

    # 构建模型（与train.py一致，使用pointcept.models.build_model）
    model = build_model(config["model"]).to(device)
    model.eval()  # 测试模式
    logger.info(f"模型构建完成：{model.__class__.__name__}（设备：{device}）")

    # 加载模型权重
    model = load_model_weight(model, args.checkpoint, device)

    # 执行测试
    logger.info("开始测试...")
    test_result = test_one_epoch(model, test_dataloader, device, num_classes)

    # 打印测试结果（与train.py日志风格一致）
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总（与训练验证一致：加权平均）")
    logger.info("=" * 60)
    logger.info(f"平均损失：{test_result['avg_loss']:.4f}")
    logger.info(f"平均准确率：{test_result['avg_accuracy']:.4f}")
    logger.info(f"加权平均精确率：{test_result['avg_precision']:.4f}")
    logger.info(f"加权平均召回率：{test_result['avg_recall']:.4f}")
    logger.info(f"加权平均F1：{test_result['avg_f1']:.4f}")

    # 打印每类F1
    logger.info("\n" + "=" * 60)
    logger.info("各类别F1分数")
    logger.info("=" * 60)
    for cls in range(num_classes):
        logger.info(f"类别{cls} F1分数：{test_result['per_class_f1'][cls]:.4f}")

    # 打印混淆矩阵
    logger.info("\n" + "=" * 60)
    logger.info("混淆矩阵（行：真实类别，列：预测类别）")
    logger.info("=" * 60)
    print(test_result["confusion_matrix"].astype(int))

    # 保存测试结果（与train.py的日志目录风格统一）
    save_path = os.path.join("./logs", "test_result.txt")  # 保存到logs目录，与训练日志统一
    save_test_result(test_result, save_path, num_classes)


if __name__ == "__main__":
    main()