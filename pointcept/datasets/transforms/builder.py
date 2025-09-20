# pointcept/datasets/transforms/builder.py
from pointcept.utils.registry import Registry
from torchvision.transforms import Compose  # 导入组合多个transform的工具

TRANSFORMS = Registry('transforms')


def build_transform(cfg):
    """
    支持处理两种配置格式：
    1. 单个dict：对应一个预处理操作（如 {'type': 'ToTensor'}）
    2. list：对应多个预处理操作（如 [{'type': 'NormalizeWind', ...}, {'type': 'GridSample', ...}]）
    """
    if cfg is None:
        return None

    # 情况1：cfg是列表（多个预处理操作）
    if isinstance(cfg, list):
        transforms = []
        for sub_cfg in cfg:
            # 每个子配置必须是dict，且包含'type'键
            if not isinstance(sub_cfg, dict) or 'type' not in sub_cfg:
                raise ValueError(f"每个transform子配置必须是含'type'的dict，当前是 {type(sub_cfg)}")
            # 从注册表构建单个transform实例
            transform = TRANSFORMS.build(sub_cfg)
            transforms.append(transform)
        # 组合所有transform为一个流水线（依次执行）
        return Compose(transforms)

    # 情况2：cfg是单个dict（一个预处理操作）
    elif isinstance(cfg, dict):
        return TRANSFORMS.build(cfg)

    # 其他情况（不支持的类型）
    else:
        raise TypeError(f"transform配置必须是list或dict，当前是 {type(cfg)}")