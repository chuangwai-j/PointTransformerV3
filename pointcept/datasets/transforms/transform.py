# pointcept/datasets/transformers/transform.py
import numpy as np
import torch
from .builder import TRANSFORMS, build_transform


@TRANSFORMS.register_module()
class Compose(object):
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = []
        # 关键：遍历每个transform配置，用build_transform转为实例
        for tf_cfg in transforms:
            tf_instance = build_transform(tf_cfg)  # 把dict转为类实例（如NormalizeWind）
            self.transforms.append(tf_instance)

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    """Convert numpy arrays to torch tensors."""

    def __call__(self, data_dict):
        for key in data_dict:
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = torch.from_numpy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    """Grid sampling for point clouds（适配9维特征）"""

    def __init__(self, grid_size=0.02):
        self.grid_size = grid_size

    def __call__(self, data_dict):
        if 'coord' in data_dict:
            coord = data_dict['coord']
            if len(coord) > 0:
                voxel_indices = np.floor(coord / self.grid_size).astype(np.int32)
                unique_voxels, inverse_indices = np.unique(
                    voxel_indices, axis=0, return_inverse=True
                )

                sampled_indices = []
                for i in range(len(unique_voxels)):
                    point_indices = np.where(inverse_indices == i)[0]
                    if len(point_indices) > 0:
                        sampled_indices.append(point_indices[0])

                # 对9维特征进行采样
                for key in data_dict:
                    if isinstance(data_dict[key], np.ndarray) and len(data_dict[key]) == len(coord):
                        data_dict[key] = data_dict[key][sampled_indices]
        return data_dict