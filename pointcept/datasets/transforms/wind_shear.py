# pointcept/datasets/transforms/wind_shear.py (更新版)
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
        self.beamaz_std = max(beamaz_std, 1e-6)  # 新增

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
    def __init__(self, grid_size=80.0, min_points=50, adaptive=True,
                 undersample_ratio=1.0,  # 🌟 新增：欠采样比例 (1.0 = 不执行)
                 smote_ratios=None):  # 🌟 新增：SMOTE 比例 (字典或列表)
        self.grid_size = grid_size
        self.min_points = min_points
        self.adaptive = adaptive
        self.undersample_ratio = undersample_ratio
        self.smote_ratios = smote_ratios

        # 将字典 {0: 0.0, 1: 0.0, ...} 转换为列表 [0.0, 0.0, ...]
        if isinstance(self.smote_ratios, dict):
            max_cls = max(self.smote_ratios.keys())
            ratios_list = [0.0] * (max_cls + 1)
            for k, v in self.smote_ratios.items():
                ratios_list[k] = v
            self.smote_ratios = ratios_list

        logging.info(f"WindShearGridSample: 类别1欠采样比例 = {self.undersample_ratio}")
        logging.info(f"WindShearGridSample: SMOTE 比例 = {self.smote_ratios}")

    def __call__(self, data_dict):
        # 🌟 关键：先保存原始path
        original_path = data_dict.get('path', '未知路径')

        # 1. 提取原始数据（含beamaz）
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        label = data_dict["generate_label"]
        beamaz = data_dict.get("beamaz", None)

        # 🌟 新增1：量化前再次检查coord
        if np.isnan(coord).any() or np.isinf(coord).any():
            valid_mask = ~(np.isnan(coord).any(axis=1) | np.isinf(coord).any(axis=1))
            coord = coord[valid_mask]
            feat = feat[valid_mask]
            label = label[valid_mask]
            beamaz = beamaz[valid_mask] if beamaz is not None else None
            logging.warning(f"样本{original_path}量化前过滤{len(coord) - valid_mask.sum()}个异常点")

        # ... (坐标量化和网格大小计算 - 保持不变)
        coord_min = coord.min(axis=0, keepdims=True)
        coord_min = np.nan_to_num(coord_min, nan=0.0)
        coord_quantized = (coord - coord_min).round().astype(np.int32)
        coord_quantized = np.clip(coord_quantized, 0, 100000)
        coord = coord_quantized

        x_min_global = coord[:, 0].min()
        y_min_global = coord[:, 1].min()
        z_min_global = coord[:, 2].min()
        x_max, y_max, z_max = coord[:, 0].max(), coord[:, 1].max(), coord[:, 2].max()

        if self.adaptive:
            grid_size_x = max(self.grid_size, (x_max - x_min_global) / 50)
            grid_size_y = max(self.grid_size, (y_max - y_min_global) / 50)
            grid_size_z = max(self.grid_size, (z_max - z_min_global) / 10)
        else:
            if isinstance(self.grid_size, (list, np.ndarray)) and len(self.grid_size) == 3:
                grid_size_x, grid_size_y, grid_size_z = self.grid_size
            else:
                grid_size_x = grid_size_y = grid_size_z = self.grid_size
            grid_size_x = max(grid_size_x, 1e-3)
            grid_size_y = max(grid_size_y, 1e-3)
            grid_size_z = max(grid_size_z, 1e-3)

        grid_idx = np.floor(
            (coord - [x_min_global, y_min_global, z_min_global])
            / [grid_size_x, grid_size_y, grid_size_z]
        ).astype(int)
        grid_idx = np.clip(grid_idx, 0, None)
        max_z = grid_idx[:, 2].max() + 1 if len(grid_idx) > 0 else 1
        max_y = grid_idx[:, 1].max() + 1 if len(grid_idx) > 0 else 1
        grid_id = grid_idx[:, 0] * max_y * max_z + grid_idx[:, 1] * max_z + grid_idx[:, 2]

        # 2. 网格采样（保持不变）
        unique_gids = np.unique(grid_id)
        if len(unique_gids) == 0:
            logging.warning(f"样本{original_path}无有效网格，返回空数据")
            return {'coord': np.empty((0, 3)), 'path': original_path}

        sampled_indices = []
        for gid in np.unique(grid_id):
            grid_points = np.where(grid_id == gid)[0]
            if len(grid_points) == 0:
                continue
            sampled_indices.append(np.random.choice(grid_points, 1))

        if len(sampled_indices) == 0:
            logging.warning(f"样本{original_path}采样后无有效点，返回空数据")
            return {'coord': np.empty((0, 3)), 'path': original_path}
        sampled_indices = np.concatenate(sampled_indices, axis=0)

        # 3. 同步采样：根据索引筛选
        sampled_coord = coord[sampled_indices]
        sampled_feat = feat[sampled_indices]
        sampled_label = label[sampled_indices]
        sampled_beamaz = beamaz[sampled_indices] if beamaz is not None else None

        # 4. 🌟 新增：对类别 1 进行欠采样
        if self.undersample_ratio < 1.0 and self.undersample_ratio >= 0.0:
            # 找出类别1 和 其他类别
            class1_mask = (sampled_label == 1)
            other_mask = (sampled_label != 1)

            class1_indices = np.where(class1_mask)[0]
            other_indices = np.where(other_mask)[0]

            n_class1 = len(class1_indices)
            n_keep = int(n_class1 * self.undersample_ratio)

            if n_keep < n_class1:
                # 随机选择要保留的类别1的索引
                class1_keep_indices = np.random.choice(class1_indices, n_keep, replace=False)
                # 合并索引
                combined_indices = np.concatenate([other_indices, class1_keep_indices])

                # 重新应用采样
                sampled_coord = sampled_coord[combined_indices]
                sampled_feat = sampled_feat[combined_indices]
                sampled_label = sampled_label[combined_indices]
                sampled_beamaz = sampled_beamaz[combined_indices] if sampled_beamaz is not None else None

                # logging.info(f"样本 {original_path} 类别1: {n_class1} -> {n_keep} (欠采样)")

        # 5. 🌟 修改：SMOTE增强 (使用可配置的比例)
        # (旧逻辑: for cls in [2, 3, 4]: ... ratio=0.3)
        if self.smote_ratios is not None:
            for cls, ratio in enumerate(self.smote_ratios):
                if ratio > 0:
                    sampled_coord, sampled_feat, sampled_label, sampled_beamaz = smote_pointcloud(
                        sampled_coord, sampled_feat, sampled_label, sampled_beamaz,
                        target_class=cls,
                        k=3,
                        generate_ratio=ratio  # 🌟 使用配置中的 ratio
                    )

        # 6. 补点（保持不变）
        sampled_num = len(sampled_coord)
        if sampled_num < self.min_points:
            logging.warning(f"样本{original_path}采样/SMOTE后点数={sampled_num} < {self.min_points}，将被过滤")
            return {'coord': np.empty((0, 3)), 'path': original_path}

        min_multiple = 384
        target_num = max(min_multiple, ((sampled_num + min_multiple - 1) // min_multiple) * min_multiple)
        pad_num = target_num - sampled_num

        if pad_num > 0:
            tree = cKDTree(sampled_coord)
            _, pad_indices = tree.query(sampled_coord[:pad_num], k=1)
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)
            sampled_label = np.concatenate([sampled_label, sampled_label[pad_indices]], axis=0)
            if sampled_beamaz is not None:
                sampled_beamaz = np.concatenate([sampled_beamaz, sampled_beamaz[pad_indices]], axis=0)

        # 7. 最终组装数据（保持不变）
        return {
            "coord": sampled_coord,
            "feat": sampled_feat,
            "generate_label": sampled_label,
            "grid_size": np.array([grid_size_x, grid_size_y, grid_size_z]),
            "beamaz": sampled_beamaz,
            "path": original_path
        }