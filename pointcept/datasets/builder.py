# pointcept/datasets/builder.py
import torch
from functools import partial
from torch.utils.data import DataLoader
from pointcept.utils.registry import Registry
from pointcept.utils.misc import collate_fn  # 保留自定义collate_fn导入（删除重复行）

DATASETS = Registry('datasets')

def build_dataset(cfg):
    """构建数据集：接收数据集配置（如cfg['data']['train']）"""
    return DATASETS.build(cfg)

def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True
):
    """构建数据加载器：强制使用自定义collate_fn"""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,  # 核心：使用自定义批处理函数
        persistent_workers=persistent_workers and (num_workers > 0)
    )
    return loader

# 关键修改1：build_train_dataloader传cfg['data']['train']给build_dataset
def build_train_dataloader(cfg):
    # 数据集配置：从cfg['data']['train']获取（含type、split等）
    train_dataset_cfg = cfg['data']['train']
    train_dataset = build_dataset(train_dataset_cfg)
    # 训练参数（batch_size、num_workers）：从cfg['train']获取
    return build_dataloader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        shuffle=True,
        drop_last=True
    )

# 关键修改2：build_val_dataloader传cfg['data']['val']给build_dataset
def build_val_dataloader(cfg):
    # 数据集配置：从cfg['data']['val']获取
    val_dataset_cfg = cfg['data']['val']
    val_dataset = build_dataset(val_dataset_cfg)
    # 验证参数：batch_size/num_workers优先用cfg['val']，没有则用默认值
    val_batch_size = cfg['val']['batch_size'] if ('val' in cfg and 'batch_size' in cfg['val']) else 1
    val_num_workers = cfg['val']['num_workers'] if ('val' in cfg and 'num_workers' in cfg['val']) else 0
    return build_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        shuffle=False,
        drop_last=False
    )