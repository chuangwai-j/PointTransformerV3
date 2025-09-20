# pointcept/utils/checkpoint.py
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """
    保存训练检查点
    Args:
        model: 训练的模型（如PointTransformerV3）
        optimizer: 优化器（如AdamW）
        scheduler: 学习率调度器（如CosineAnnealingLR）
        epoch: 当前训练轮次
        save_path: 保存路径（如./checkpoints/best_model_epoch10.pth）
    """
    # 组装需要保存的信息
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # 模型权重
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'scheduler_state_dict': scheduler.state_dict(),  # 调度器状态
        # 可额外添加：最佳验证集指标（如best_val_f1）
        # 'best_val_f1': best_val_f1
    }
    # 保存文件（PyTorch标准格式）
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")