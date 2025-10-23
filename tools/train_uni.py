"""
单文件搞定 0-1 / 2-4 两阶段，可开关
用法：
  # 阶段1
  python train_uni.py --cfg configs/wind/wind_ptv3.yaml stage=1
  # 阶段2
  python train_uni.py --cfg configs/wind/wind_ptv3.yaml stage=2
"""
import os, sys, torch, yaml, numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")
from pointcept.datasets.builder import build_dataset, build_dataloader
from pointcept.models import build_model
from pointcept.utils.logger import get_logger
from pointcept.utils.checkpoint import save_checkpoint, load_checkpoint
from pointcept.utils.losses import BalancedLoss   # 用我们新 loss
from pointcept.datasets.transforms import Compose


def move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def f1_score(tp, fp, fn):
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return f1.mean()  # 宏平均


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, stage):
    model.train()
    tp = fp = fn = loss_sum = 0
    for batch in tqdm(loader, desc=f'Stage{stage} Train'):
        batch = move_to_device(batch, device)
        labels = batch['label'].long()
        with autocast():
            logits = model(batch)
            loss = criterion(logits, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 统计
        preds = logits.argmax(1)
        for c in range(cfg['model']['num_classes']):
            tp += ((preds == c) & (labels == c)).sum()
            fp += ((preds == c) & (labels != c)).sum()
            fn += ((preds != c) & (labels == c)).sum()
        loss_sum += loss.item() * labels.size(0)
    return loss_sum / len(loader.dataset), f1_score(tp, fp, fn)


@torch.no_grad()
def validate(model, loader, criterion, device, stage):
    model.eval()
    tp = fp = fn = loss_sum = 0
    for batch in tqdm(loader, desc=f'Stage{stage} Val'):
        batch = move_to_device(batch, device)
        labels = batch['label'].long()
        logits = model(batch)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)
        for c in range(cfg['model']['num_classes']):
            tp += ((preds == c) & (labels == c)).sum()
            fp += ((preds == c) & (labels != c)).sum()
            fn += ((preds != c) & (labels == c)).sum()
        loss_sum += loss.item() * labels.size(0)
    return loss_sum / len(loader.dataset), f1_score(tp, fp, fn)


def run_stage(cfg, stage=1):
    device = torch.device('cuda')
    logger = get_logger('wind', log_dir='./logs')
    logger.info(f'========== STAGE {stage} START ==========')

    # 1. 数据集
    train_set = build_dataset(cfg['data']['train'])
    val_set = build_dataset(cfg['data']['val'])
    # 阶段2：只留 2/3/4
    if stage == 2:
        train_set = ClassFilterDataset(train_set, keep=[2, 3, 4])
        val_set = ClassFilterDataset(val_set, keep=[2, 3, 4])

    train_loader = build_dataloader(train_set, cfg['data']['train'])
    val_loader = build_dataloader(val_set, cfg['data']['val'])

    # 2. 模型
    model = build_model(cfg['model']).to(device)

    # 3. 损失 —— 用新 BalancedLoss
    num_classes = cfg['model']['num_classes']
    # 自动计算 inverse freq
    cls_cnt = np.bincount(train_set.label_stat(), minlength=num_classes)  # 自己实现
    weight = 1.0 / (cls_cnt + 1e-6)
    weight = torch.tensor(weight / weight.sum() * num_classes, device=device)
    criterion = BalancedLoss(num_classes=num_classes,
                             alpha=weight,
                             gamma=1.0 if stage == 2 else 0.5,
                             ce_weight=1.0,
                             focal_weight=1.0,
                             eps=0.05)

    # 4. 优化器 & 冻结
    if stage == 2:
        for name, p in model.named_parameters():
            p.requires_grad = 'cls_head' in name or 'block4' in name or 'block3' in name
    else:
        for p in model.parameters():
            p.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg['optimizer']['lr'] * (0.5 if stage == 2 else 1.0),
                                  weight_decay=cfg['optimizer']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    scaler = GradScaler()

    best_f1 = 0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, stage)
        val_loss, val_f1 = validate(model, val_loader, criterion, device, stage)
        logger.info(f'Epoch {epoch}  train_loss={train_loss:.4f}  train_f1={train_f1:.4f}  '
                    f'val_loss={val_loss:.4f}  val_f1={val_f1:.4f}')
        scheduler.step()
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, optimizer, scheduler, epoch,
                            save_path=f'./checkpoints/best_stage{stage}.pth')
    logger.info(f'========== STAGE {stage} END  best_f1={best_f1:.4f} ==========')


# -------------- 工具类 --------------
class ClassFilterDataset(torch.utils.data.Dataset):
    def __init__(self, base, keep):
        self.base = base
        self.keep = set(keep)
        self.indices = [i for i in range(len(base))
                        if any(l in self.keep for l in np.unique(base[i]['label']))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/wind/wind_ptv3.yaml')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['stage'] = args.stage
    run_stage(cfg, stage=args.stage)


if __name__ == '__main__':
    main()