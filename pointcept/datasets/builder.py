# pointcept/datasets/builder.py
import torch
from functools import partial
from torch.utils.data import DataLoader
from pointcept.utils.registry import Registry

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
    from pointcept.utils.misc import collate_fn
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
    # 1. 数据集配置：从cfg['data']['train']获取（含type、split等）
    train_dataset_cfg = cfg['data']['train']
    train_dataset = build_dataset(train_dataset_cfg)

    # 2. 从训练配置中读取DataLoader参数
    return build_dataloader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],  # 从train配置取batch_size
        num_workers=cfg['train']['num_workers'],    # 从train配置取线程数
        shuffle=True,   # 训练集需要打乱
        drop_last=True  # 训练时丢弃最后一个不完整的batch
    )

def build_val_dataloader(cfg):
    """构建验证集DataLoader：从cfg['evaluation']读取batch_size和num_workers（与训练一致）"""
    # 1. 数据集配置：从cfg['data']['val']获取
    val_dataset_cfg = cfg['data']['val']
    val_dataset = build_dataset(val_dataset_cfg)

    # 2. 从evaluation配置中读取DataLoader参数（关键修改：与训练集保持一致的配置路径）
    # 优先使用evaluation中的配置，若未定义则复用训练集的配置（避免默认值1/0）
    val_batch_size = cfg['evaluation'].get('batch_size', cfg['train']['batch_size'])
    val_num_workers = cfg['evaluation'].get('num_workers', cfg['train']['num_workers'])

    return build_dataloader(
        val_dataset,
        batch_size=val_batch_size,  # 与训练集batch_size一致
        num_workers=val_num_workers,    # 与训练集线程数一致
        shuffle=False,  # 验证集不打乱
        drop_last=False # 验证时保留最后一个不完整的batch
    )