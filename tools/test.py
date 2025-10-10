import os
import argparse
import logging
import yaml
import torch
import numpy as np
from addict import Dict
from tqdm import tqdm
import torch.nn.functional as F

# 导入项目内模块（需确保路径正确，若报错需调整sys.path）
try:
    from pointcept.models.builder import MODELS
    from pointcept.datasets.builder import DATASETS
    from pointcept.models.utils.structure import Point
    from pointcept.utils.misc import average_meter
except ImportError:
    # 若导入失败，添加项目根目录到环境变量
    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from pointcept.models.builder import MODELS
    from pointcept.datasets.builder import DATASETS
    from pointcept.models.utils.structure import Point
    from pointcept.utils.misc import average_meter


# 配置日志（打印测试过程）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数：指定配置文件、模型权重、GPUID"""
    parser = argparse.ArgumentParser(description="PointTransformerV3 测试脚本")
    # 核心参数：配置文件路径（用find命令找到的路径）
    parser.add_argument(
        "--config",
        required=True,
        help="配置文件路径，例如：./configs/wind_shear/pointtransformer_v3.yaml"
    )
    # 核心参数：训练好的模型权重（通常在./checkpoints/下）
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="模型权重路径，例如：./checkpoints/best_model_epoch50.pth"
    )
    # 可选参数：GPU设备（默认用0号GPU，无GPU则设为-1）
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU设备号（-1表示使用CPU）"
    )
    # 可选参数：测试批次大小（默认用配置中的evaluation.batch_size）
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="测试批次大小（优先于配置文件）"
    )
    return parser.parse_args()


def load_config(config_path):
    """加载yaml配置文件，转为Dict格式（与train.py一致）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # 转为addict.Dict，支持属性访问（如config.model.type）
    return Dict(config)


def build_test_dataloader(config, batch_size=None):
    """构建测试数据集加载器（复用配置中的data.test参数）"""
    # 1. 从配置获取测试数据参数
    test_data_cfg = config.data.test
    # 批次大小：命令行指定优先，否则用配置中的evaluation.batch_size
    batch_size = batch_size or config.evaluation.batch_size
    # 数据加载参数（复用evaluation的配置，确保与训练/验证一致）
    num_workers = config.evaluation.num_workers
    prefetch_factor = config.evaluation.get("prefetch_factor", 2)
    pin_memory = config.evaluation.get("pin_memory", True)

    # 2. 构建数据集（WindShearDataset，与train.py一致）
    test_dataset = DATASETS.build(test_data_cfg)
    logger.info(f"测试集加载完成：共{len(test_dataset)}个样本")

    # 3. 构建DataLoader（无shuffle，测试时无需打乱）
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集禁用shuffle
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=False  # 测试集不丢弃最后一个不完整批次
    )
    return test_dataloader


def build_model(config, device):
    """构建PointTransformerV3模型（与train.py一致）"""
    # 从配置构建模型（注册名PT-v3m1）
    model = MODELS.build(config.model)
    # 移动模型到指定设备（GPU/CPU）
    model = model.to(device)
    # 测试模式：禁用Dropout、BatchNorm更新
    model.eval()
    logger.info(f"模型构建完成：{config.model.type}（设备：{device}）")
    return model


def load_model_weight(model, checkpoint_path, device):
    """加载训练好的模型权重"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在：{checkpoint_path}")
    # 加载权重（处理CPU/GPU兼容）
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device  # 自动映射到当前设备
    )
    # 提取模型权重（若权重字典有"model"键，取其值；否则直接用）
    if "model" in checkpoint:
        model_weights = checkpoint["model"]
    else:
        model_weights = checkpoint
    # 加载权重
    model.load_state_dict(model_weights, strict=True)
    logger.info(f"权重加载完成：{checkpoint_path}")
    return model


def calculate_metrics(logits, labels, num_classes=5):
    """计算多分类任务的核心指标：损失、准确率、精确率、召回率、F1"""
    # 1. 计算交叉熵损失
    loss = F.cross_entropy(logits, labels).item()
    # 2. 预测结果（取概率最大的类别）
    preds = torch.argmax(logits, dim=1)
    # 3. 混淆矩阵（用于计算多分类指标）
    confusion_matrix = torch.zeros(num_classes, num_classes, device=logits.device)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t, p] += 1

    # 4. 计算精确率、召回率、F1（按类别平均，支持多分类）
    precision = torch.zeros(num_classes, device=logits.device)
    recall = torch.zeros(num_classes, device=logits.device)
    f1 = torch.zeros(num_classes, device=logits.device)
    for cls in range(num_classes):
        # 精确率：该类预测正确的数量 / 该类被预测的总数量
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        precision[cls] = tp / (tp + fp + 1e-6)  # 加1e-6避免除0
        # 召回率：该类预测正确的数量 / 该类实际的总数量
        fn = confusion_matrix[cls, :].sum() - tp
        recall[cls] = tp / (tp + fn + 1e-6)
        # F1分数：2*(精确率*召回率)/(精确率+召回率)
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls] + 1e-6)

    # 5. 宏观平均（所有类别平均）
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    # 6. 准确率（所有样本预测正确的比例）
    accuracy = (preds == labels).float().mean().item()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "confusion_matrix": confusion_matrix.cpu().numpy()  # 转为numpy用于后续保存
    }


def test_one_epoch(model, dataloader, device, num_classes=5):
    """执行一轮测试，返回平均指标"""
    # 初始化指标记录器（平均多个批次的结果）
    loss_meter = average_meter()
    accuracy_meter = average_meter()
    precision_meter = average_meter()
    recall_meter = average_meter()
    f1_meter = average_meter()
    # 总混淆矩阵
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    # 无梯度测试（禁用反向传播，加速并节省内存）
    with torch.no_grad():
        # 遍历测试集（用tqdm显示进度）
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="测试中")):
            # 1. 数据移动到设备
            # （假设batch是Dict，包含"coord"、"feat"、"label"、"offset"等键，与train.py一致）
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]  # 标签

            # 2. 模型前向传播（生成logits）
            # （与train.py的model.forward一致，返回logits）
            logits = model(batch)

            # 3. 计算当前批次的指标
            metrics = calculate_metrics(logits, labels, num_classes)
            # 更新指标记录器（按批次样本数加权平均）
            batch_size = labels.shape[0]
            loss_meter.update(metrics["loss"], batch_size)
            accuracy_meter.update(metrics["accuracy"], batch_size)
            precision_meter.update(metrics["precision"], batch_size)
            recall_meter.update(metrics["recall"], batch_size)
            f1_meter.update(metrics["f1"], batch_size)
            # 累加混淆矩阵
            total_confusion_matrix += metrics["confusion_matrix"]

    # 返回平均指标
    return {
        "avg_loss": loss_meter.avg,
        "avg_accuracy": accuracy_meter.avg,
        "avg_precision": precision_meter.avg,
        "avg_recall": recall_meter.avg,
        "avg_f1": f1_meter.avg,
        "confusion_matrix": total_confusion_matrix
    }


def save_test_result(result, save_path):
    """保存测试结果到文件（txt格式，便于查看）"""
    # 创建保存目录
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    # 写入结果
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + " 测试结果 " + "=" * 50 + "\n")
        f.write(f"平均损失：{result['avg_loss']:.4f}\n")
        f.write(f"平均准确率：{result['avg_accuracy']:.4f}\n")
        f.write(f"平均精确率：{result['avg_precision']:.4f}\n")
        f.write(f"平均召回率：{result['avg_recall']:.4f}\n")
        f.write(f"平均F1分数：{result['avg_f1']:.4f}\n")
        f.write("\n" + "=" * 30 + " 混淆矩阵 " + "=" * 30 + "\n")
        # 混淆矩阵格式化输出
        cm = result["confusion_matrix"]
        for i in range(cm.shape[0]):
            f.write(f"类别{i}：" + " ".join([f"{int(x)}" for x in cm[i]]) + "\n")
    logger.info(f"测试结果已保存到：{save_path}")


def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 确定设备（GPU/CPU）
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"使用GPU设备：cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU设备（GPU不可用或指定--gpu=-1）")

    # 3. 加载配置文件
    config = load_config(args.config)
    # 从配置获取类别数（默认5类，与你的模型一致）
    num_classes = config.model.num_classes

    # 4. 构建测试数据集加载器
    test_dataloader = build_test_dataloader(config, args.batch_size)

    # 5. 构建模型并加载权重
    model = build_model(config, device)
    model = load_model_weight(model, args.checkpoint, device)

    # 6. 执行测试
    logger.info("开始测试...")
    test_result = test_one_epoch(model, test_dataloader, device, num_classes)

    # 7. 打印测试结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    logger.info(f"平均损失：{test_result['avg_loss']:.4f}")
    logger.info(f"平均准确率：{test_result['avg_accuracy']:.4f}")
    logger.info(f"平均精确率：{test_result['avg_precision']:.4f}")
    logger.info(f"平均召回率：{test_result['avg_recall']:.4f}")
    logger.info(f"平均F1分数：{test_result['avg_f1']:.4f}")
    logger.info("\n混淆矩阵（行：真实类别，列：预测类别）：")
    print(test_result["confusion_matrix"].astype(int))

    # 8. 保存测试结果（默认保存到./test_results/目录）
    save_path = os.path.join("./test_results", "test_result.txt")
    save_test_result(test_result, save_path)


if __name__ == "__main__":
    main()