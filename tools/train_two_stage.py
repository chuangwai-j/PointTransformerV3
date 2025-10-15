import os, sys, torch, yaml, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pointcept.datasets.builder import build_train_dataloader, build_val_dataloader
from pointcept.models import build_model
from pointcept.utils.logger import get_logger
from pointcept.utils.checkpoint import save_checkpoint
from pointcept.utils.losser_v2 import LabelSmoothingCrossEntropy
from pointcept.utils.losses import MixedLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

class Stage2Sampler(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_classes):
        self.base = base_dataset
        self.keep_classes = set(keep_classes)
        self.indices = [i for i in range(len(self.base))
                        if self.base[i]['generate_label'].numpy() in self.keep_classes]

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
        if self.best is None:
            self.best = score
            return False
        if self.mode == 'max':
            improved = score > self.best + self.min_delta
        else:
            improved = score < self.best - self.min_delta
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False



def run_stage(cfg, stage=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger('wind_shear_train', log_dir='./logs')
    logger.info(f"========== STAGE {stage} START ==========")

    train_loader = build_train_dataloader(cfg)
    val_loader   = build_val_dataloader(cfg)

    # 阶段2：只保留234，冻结backbone
    if stage == 2:
        train_loader.dataset = Stage2Sampler(train_loader.dataset, keep_classes=[2,3,4])
        logger.info("Stage2：只保留类别2/3/4，冻结backbone")

    model = build_model(cfg['model']).to(device)
    num_classes = 5
    weight_tensor = torch.tensor(cfg['train']['class_weights'], dtype=torch.float32, device=device)

    # 损失函数
    if stage == 1:
        criterion = MixedLoss(num_classes=num_classes, alpha=weight_tensor, gamma=2.0)
    else:
        criterion = LabelSmoothingCrossEntropy(num_classes=num_classes, eps=0.1, weight=weight_tensor)

    # 优化器
    if stage == 2:
        for name, p in model.named_parameters():
            p.requires_grad = "cls_head" in name or "classifier" in name
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg['train']['optimizer']['lr'] * 0.5,
                                      weight_decay=cfg['train']['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg['train']['optimizer']['lr'],
                                      weight_decay=cfg['train']['optimizer']['weight_decay'])

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'], eta_min=1e-6)

    early_stop = EarlyStopping(patience=15, min_delta=1e-4, mode='max')  # 可自己调

    best_val_f1 = 0.0
    for epoch in range(1, cfg['train']['epochs']+1):
        model.train()
        train_loss, train_total = 0.0, 0
        train_tp = torch.zeros(num_classes, dtype=torch.long, device=device)
        train_fn = torch.zeros(num_classes, dtype=torch.long, device=device)
        train_fp = torch.zeros(num_classes, dtype=torch.long, device=device)

        for batch in tqdm(train_loader, desc=f"Stage{stage} Train"):
            if batch is None: continue
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            labels = batch['generate_label'].long()
            logits = model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            preds = logits.argmax(1)
            train_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)
            for c in range(num_classes):
                train_tp[c] += ((preds == c) & (labels == c)).sum()
                train_fn[c] += ((preds != c) & (labels == c)).sum()
                train_fp[c] += ((preds == c) & (labels != c)).sum()

        # 训练指标
        train_precision = train_tp / (train_tp + train_fp + 1e-6)
        train_recall    = train_tp / (train_tp + train_fn + 1e-6)
        train_f1        = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-6)
        train_weight_f1 = (train_f1 * (train_tp + train_fn)).sum() / (train_tp + train_fn).sum()
        logger.info(f"Stage{stage} Epoch{epoch} Train F1={train_weight_f1:.4f}")

        # 验证
        model.eval()
        val_tp = torch.zeros(num_classes, dtype=torch.long, device=device)
        val_fn = torch.zeros(num_classes, dtype=torch.long, device=device)
        val_fp = torch.zeros(num_classes, dtype=torch.long, device=device)
        val_total, val_loss = 0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Stage{stage} Val"):
                if batch is None: continue
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                labels = batch['generate_label'].long()
                logits = model(batch)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_total += labels.size(0)
                preds = logits.argmax(1)
                for c in range(num_classes):
                    val_tp[c] += ((preds == c) & (labels == c)).sum()
                    val_fn[c] += ((preds != c) & (labels == c)).sum()
                    val_fp[c] += ((preds == c) & (labels != c)).sum()

        val_precision = val_tp / (val_tp + val_fp + 1e-6)
        val_recall    = val_tp / (val_tp + val_fn + 1e-6)
        val_f1        = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-6)
        val_weight_f1 = (val_f1 * (val_tp + val_fn)).sum() / (val_tp + val_fn).sum()
        logger.info(f"Stage{stage} Epoch{epoch} Val F1={val_weight_f1:.4f}")

        if early_stop(val_weight_f1):
            logger.info(f"🚨 Stage{stage} 早停触发，最佳 F1={early_stop.best:.4f}，结束训练！")
            break

        if val_weight_f1 > best_val_f1:
            best_val_f1 = val_weight_f1
            save_checkpoint(model, optimizer, scheduler, epoch,
                            save_path=f"./checkpoints/best_stage{stage}_model.pth")
            logger.info(f"✅ 保存最佳 Stage{stage} 模型 F1={best_val_f1:.4f}")

        scheduler.step()

    logger.info(f"========== STAGE {stage} END BEST F1={best_val_f1:.4f} ==========")

def main():
    cfg_path = "configs/wind_shear/pointtransformer_v3.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    # 阶段1：01
    cfg['train']['class_weights'] = [0.8, 0.1, 1.0, 4.5, 2.5]
    run_stage(cfg, stage=1)
    # 阶段2：234
    cfg['train']['epochs'] = 30  # 可适当减少
    run_stage(cfg, stage=2)

if __name__ == "__main__":
    main()
