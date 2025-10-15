# pointcept/datasets/transforms/wind_shear.py (新建文件)
import os
import logging
import numpy as np
from .builder import TRANSFORMS
from scipy.spatial import cKDTree
from .smote import smote_pointcloud
from pointcept.utils.registry import Registry

@TRANSFORMS.register_module()
class NormalizeWind(object):
    """标准化风速数据（新增beamaz归一化）"""

    def __init__(self, u_mean, u_std, v_mean, v_std,
                 beamaz_mean, beamaz_std):  # 新增beamaz参数
        self.u_mean = u_mean
        self.u_std = max(u_std, 1e-6)  # 🌟 防止std=0导致除零
        self.v_mean = v_mean
        self.v_std = max(v_std, 1e-6)
        self.beamaz_mean = beamaz_mean  # 新增
        self.beamaz_std = max(beamaz_std, 1e-6)    # 新增

    def __call__(self, data_dict):
        original_path = data_dict['path']
        if 'feat' not in data_dict:
            return data_dict

        feat = data_dict['feat'].copy()
        # 🌟 标准化前再次检查feat是否有异常（防止前面步骤漏网）
        feat = np.nan_to_num(feat, nan=0.0, posinf=1e3, neginf=-1e3)

        # 归一化u相关特征（索引0,3,6）
        try:
            feat[:, 0] = (feat[:, 0] - self.u_mean) / self.u_std
            feat[:, 3] = (feat[:, 3] - self.u_mean) / self.u_std
            feat[:, 6] = (feat[:, 6] - self.u_mean) / self.u_std
        except IndexError:
            logging.error(f"样本{original_path}feat维度异常（u相关），跳过标准化")
            data_dict['feat'] = feat
            data_dict['path'] = original_path
            return data_dict

        # 归一化v相关特征（索引1,4,7）
        try:
            feat[:, 1] = (feat[:, 1] - self.v_mean) / self.v_std
            feat[:, 4] = (feat[:, 4] - self.v_mean) / self.v_std
            feat[:, 7] = (feat[:, 7] - self.v_mean) / self.v_std
        except IndexError:
            logging.error(f"样本{original_path}feat维度异常（v相关），跳过标准化")
            data_dict['feat'] = feat
            data_dict['path'] = original_path
            return data_dict

        # 归一化beamaz相关特征（索引2,5,8）
        try:
            feat[:, 2] = (feat[:, 2] - self.beamaz_mean) / self.beamaz_std
            feat[:, 5] = (feat[:, 5] - self.beamaz_mean) / self.beamaz_std
            feat[:, 8] = (feat[:, 8] - self.beamaz_mean) / self.beamaz_std
        except IndexError:
            logging.error(f"样本{original_path}feat维度异常（beamaz相关），跳过标准化")
            data_dict['feat'] = feat
            data_dict['path'] = original_path
            return data_dict

        # 🌟 标准化后强制截断+处理异常值
        feat = np.clip(feat, -5.0, 5.0)  # 限制范围
        feat = np.nan_to_num(feat, nan=0.0, posinf=5.0, neginf=-5.0)  # 最终保险

        data_dict['feat'] = feat
        data_dict['path'] = original_path  # 强制保留path
        return data_dict

@TRANSFORMS.register_module()
class WindShearGridSample:
    def __init__(self, grid_size=80.0, min_points=50, adaptive=True):
        self.grid_size = grid_size  # 支持标量或三维列表（固定模式下）
        self.min_points = min_points
        self.adaptive = adaptive

    def __call__(self, data_dict):
        # 🌟 关键：先保存原始path（无论后续如何处理，都不丢失）
        original_path = data_dict.get('path', '未知路径')  # 读取原始path
        # 1. 提取原始数据（含beamaz）
        coord = data_dict["coord"]  # 原始坐标 (N, 3)
        feat = data_dict["feat"]  # 原始特征 (N, C)
        label = data_dict["generate_label"]  # 原始标签 (N,)
        beamaz = data_dict.get("beamaz", None)  # 提取beamaz（可能不存在）

        # 🌟 新增1：量化前再次检查coord（防止变换中引入异常）
        if np.isnan(coord).any() or np.isinf(coord).any():
            valid_mask = ~(np.isnan(coord).any(axis=1) | np.isinf(coord).any(axis=1))
            coord = coord[valid_mask]
            feat = feat[valid_mask]
            label = label[valid_mask]
            beamaz = beamaz[valid_mask] if beamaz is not None else None
            logging.warning(f"样本{original_path}量化前过滤{len(coord) - valid_mask.sum()}个异常点")

        # 坐标量化（强化处理）
        coord_min = coord.min(axis=0, keepdims=True)
        # 防止coord_min为NaN（理论上不会，保险措施）
        coord_min = np.nan_to_num(coord_min, nan=0.0)
        coord_quantized = (coord - coord_min).round().astype(np.int32)
        # 限制量化坐标范围（避免极端值，如>1e6）
        coord_quantized = np.clip(coord_quantized, 0, 100000)  # 合理范围上限
        coord = coord_quantized

        # 网格大小计算（防止除零）
        # 先计算坐标全局最小值（无论自适应与否都需要）
        x_min_global = coord[:, 0].min()
        y_min_global = coord[:, 1].min()
        z_min_global = coord[:, 2].min()
        x_max, y_max, z_max = coord[:, 0].max(), coord[:, 1].max(), coord[:, 2].max()

        # 2. 网格采样核心逻辑（获取采样点的索引）
        if self.adaptive:
            # 避免网格大小为0或负数（导致除零）
            grid_size_x = max(self.grid_size, (x_max - x_min_global) / 50)
            grid_size_y = max(self.grid_size, (y_max - y_min_global) / 50)
            grid_size_z = max(self.grid_size, (z_max - z_min_global) / 10)
        else:
            # 非自适应模式：使用配置的grid_size（支持三维输入）
            # 修复1：使用全局最小值作为坐标平移基准
            x_min, y_min, z_min = x_min_global, y_min_global, z_min_global
            # 修复2：支持三维grid_size（如[122.6, 118.0, 5.4]）
            if isinstance(self.grid_size, (list, np.ndarray)) and len(self.grid_size) == 3:
                grid_size_x, grid_size_y, grid_size_z = self.grid_size
            else:
                # 兼容标量grid_size的情况
                grid_size_x = grid_size_y = grid_size_z = self.grid_size
            # 强制网格大小为正
            grid_size_x = max(grid_size_x, 1e-3)
            grid_size_y = max(grid_size_y, 1e-3)
            grid_size_z = max(grid_size_z, 1e-3)

        # 计算每个点所属的网格索引（避免负数索引）
        grid_idx = np.floor(
            (coord - [x_min_global, y_min_global, z_min_global])
            / [grid_size_x, grid_size_y, grid_size_z]
        ).astype(int)

        # 网格ID非负化（防止负索引导致唯一ID计算错误）
        grid_idx = np.clip(grid_idx, 0, None)

        # 计算唯一网格ID（防止溢出）
        max_z = grid_idx[:, 2].max() + 1 if len(grid_idx) > 0 else 1
        max_y = grid_idx[:, 1].max() + 1 if len(grid_idx) > 0 else 1
        grid_id = grid_idx[:, 0] * max_y * max_z + grid_idx[:, 1] * max_z + grid_idx[:, 2]
        # 为每个网格分配唯一ID（避免冲突）
        #grid_id = grid_idx[:, 0] * (grid_idx[:, 1].max() + 1) * (grid_idx[:, 2].max() + 1) + \
        #          grid_idx[:, 1] * (grid_idx[:, 2].max() + 1) + grid_idx[:, 2]

        # 网格采样（处理空网格）
        unique_gids = np.unique(grid_id)
        if len(unique_gids) == 0:
            logging.warning(f"样本{original_path}无有效网格，返回空数据")
            return {'coord': np.empty((0, 3)), 'path': original_path}

        # 🌟 关键：获取每个网格的采样点索引（确保采样后beamaz可同步筛选）
        sampled_indices = []
        for gid in np.unique(grid_id):
            # 每个网格内随机选1个点
            grid_points = np.where(grid_id == gid)[0]
            if len(grid_points) == 0:
                continue  # 跳过空网格
            sampled_indices.append(np.random.choice(grid_points, 1))
        if len(sampled_indices) == 0:
            logging.warning(f"样本{original_path}采样后无有效点，返回空数据")
            return {'coord': np.empty((0, 3)), 'path': original_path}
        sampled_indices = np.concatenate(sampled_indices, axis=0)

        # 3. 🌟 同步采样：根据索引筛选coord/feat/generate_label/beamaz
        sampled_coord = coord[sampled_indices]
        sampled_feat = feat[sampled_indices]
        sampled_label = label[sampled_indices]
        sampled_beamaz = beamaz[sampled_indices] if beamaz is not None else None

        # 🌟 SMOTE增强：仅对少数类2、3、4生成新样本
        for cls in [2, 3, 4]:
            sampled_coord, sampled_feat, sampled_label, sampled_beamaz = smote_pointcloud(
                sampled_coord, sampled_feat, sampled_label, sampled_beamaz,
                target_class=cls, k=3, generate_ratio=0.3
            )

        # 4. 补点至384的倍数（同步补beamaz）—— 完整保留你的补点逻辑
        sampled_num = len(sampled_coord)
        if sampled_num < self.min_points:
            # 这里不直接报错，而是后续在__getitem__中返回None跳过（与之前的过滤逻辑呼应）
            # 保留警告信息便于调试
            logging.warning(f"样本{original_path}采样后点数={sampled_num} < {self.min_points}，将被过滤")
            # 返回空数据触发过滤
            return {'coord': np.empty((0,3)), 'path': original_path}

        min_multiple = 384  # 满足分块+下采样需求
        target_num = max(min_multiple, ((sampled_num + min_multiple - 1) // min_multiple) * min_multiple)
        pad_num = target_num - sampled_num

        if pad_num > 0:
            # 用KDTree找近邻点，补点时保持空间关联性
            tree = cKDTree(sampled_coord)
            _, pad_indices = tree.query(sampled_coord[:pad_num], k=1)  # 待补点的近邻索引

            # 同步补coord/feat/generate_label
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)
            sampled_label = np.concatenate([sampled_label, sampled_label[pad_indices]], axis=0)

            # 🌟 同步补beamaz
            if sampled_beamaz is not None:
                sampled_beamaz = np.concatenate([sampled_beamaz, sampled_beamaz[pad_indices]], axis=0)

        # 最终组装数据（强制保留path）
        return {
            "coord": sampled_coord,
            "feat": sampled_feat,
            "generate_label": sampled_label,
            "grid_size": np.array([grid_size_x, grid_size_y, grid_size_z]),
            "beamaz": sampled_beamaz,
            "path": original_path  # 全程不变的path
        }