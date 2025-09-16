# pointcept/datasets/transforms/trnsformers.py (新建文件)
import numpy as np
from pointcept.datasets.trnsformers import TRANSFORMS


@TRANSFORMS.register()
class NormalizeWind(object):
    """标准化风速数据"""

    def __init__(self, u_mean=0.0, u_std=1.0, v_mean=0.0, v_std=1.0):
        self.u_mean = u_mean
        self.u_std = u_std
        self.v_mean = v_mean
        self.v_std = v_std

    def __call__(self, data_dict):
        if 'feat' in data_dict:
            # 假设特征中前两个通道是u和v风速分量
            feat = data_dict['feat'].copy()
            feat[:, 0] = (feat[:, 0] - self.u_mean) / self.u_std
            feat[:, 1] = (feat[:, 1] - self.v_mean) / self.v_std
            data_dict['feat'] = feat

        return data_dict


@TRANSFORMS.register()
class WindShearGridSample(object):
    """对风切变数据进行网格采样"""

    def __init__(self, grid_size=0.1):
        self.grid_size = grid_size

    def __call__(self, data_dict):
        coord = data_dict['coord']
        feat = data_dict['feat']
        label = data_dict['label']

        # 简单的网格采样
        voxel_coord = np.floor(coord / self.grid_size)
        unique_voxels, inverse_indices = np.unique(voxel_coord, axis=0, return_inverse=True)

        # 对每个体素内的点取平均
        sampled_coord = np.zeros((len(unique_voxels), 3))
        sampled_feat = np.zeros((len(unique_voxels), feat.shape[1]))
        sampled_label = np.zeros(len(unique_voxels))

        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            sampled_coord[i] = coord[mask].mean(axis=0)
            sampled_feat[i] = feat[mask].mean(axis=0)
            # 对于标签，取众数或根据业务逻辑处理
            sampled_label[i] = np.round(label[mask].mean())

        data_dict['coord'] = sampled_coord.astype(np.float32)
        data_dict['feat'] = sampled_feat.astype(np.float32)
        data_dict['label'] = sampled_label.astype(np.int64)

        return data_dict