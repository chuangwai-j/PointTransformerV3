import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes, eps=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.weight = weight

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        loss = -(targets * log_probs).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[labels]
        return loss.mean()
