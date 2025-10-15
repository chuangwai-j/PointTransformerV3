import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    # 核心：抑制多数类（类别1）损失，放大少数类（2、3、4）损失
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 复用你原有的类别权重（weight_tensor）
        self.gamma = gamma  # 难例惩罚系数（默认2.0，无需改动）
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss  # 放大难例损失
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

class DiceLoss(nn.Module):
    # 核心：直接优化F1（计算预测与真实少数类的重叠度）
    def __init__(self, num_classes, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes  # 你的任务是5分类，后续传5
        self.eps = eps  # 避免分母为0

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)  # 预测概率
        one_hot = F.one_hot(labels, self.num_classes).float()  # 真实标签转独热码
        intersection = (probs * one_hot).sum(dim=0)  # 预测与真实的重叠数
        union = probs.sum(dim=0) + one_hot.sum(dim=0)  # 预测总数+真实总数
        dice = (2. * intersection + self.eps) / (union + self.eps)  # 重叠度（越近1越好）
        return 1 - dice.mean()  # 损失=1-重叠度，逼模型提升重叠度

class MixedLoss(nn.Module):
    # 核心：融合FocalLoss和DiceLoss（1:1加权，兼顾“关注少数类”和“提F1”）
    def __init__(self, num_classes, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight  # 权重1.0（无需改动）
        self.dice_weight = dice_weight    # 权重1.0（无需改动）

    def forward(self, logits, labels):
        return self.focal_weight * self.focal(logits, labels) + \
               self.dice_weight * self.dice(logits, labels)