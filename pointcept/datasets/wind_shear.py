# pointcept/datasets/wind_shear.py
import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
from pointcept.utils.logger import get_root_logger
from scipy.spatial import KDTree
from pointcept.datasets.builder import DATASETS
# 新增：导入构建transform的函数
from pointcept.datasets.transforms.builder import build_transform  # 注意路径是否正确（你的文件是transformers/builder.py）


# 关键：添加注册装饰器，让数据集被DATASETS注册表识别
@DATASETS.register_module()
class WindShearDataset(Dataset):
    def __init__(self, split='train', data_root="D:/model/wind_datas/csv_labels",
                 transform=None, k_neighbors=16, radius=0.5, min_points=50,
                 filter_full_paths=None):  # 新增min_points参数
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.transform = build_transform(transform)
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.min_points = min_points  # 新增：初始化min_points属性
        self.data_list = self._get_data_list()
        self.filter_full_paths = filter_full_paths if filter_full_paths is not None else []  # 存储完整路径列表
        if self.filter_full_paths:
            logging.info(f"将过滤{len(self.filter_full_paths)}个低价值样本（完整路径）")

        logger = get_root_logger()
        logging.info(f"WindShearDataset {split} split: {len(self.data_list)} scenes")

    def _get_data_list(self):
        # 根据日期划分数据集
        if self.split == 'train':
            dates = [f"202303{i:02d}" for i in range(1, 23)]
        elif self.split == 'val':
            dates = [f"202303{i:02d}" for i in range(23, 29)]
        elif self.split == 'test':
            dates = [f"202303{i:02d}" for i in range(29, 32)]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        data_list = []
        for date in dates:
            date_path = os.path.join(self.data_root, date)
            if not os.path.exists(date_path):
                continue

            # 查找所有datas文件夹
            datas_dirs = glob.glob(os.path.join(date_path, "datas*"))
            for datas_dir in datas_dirs:
                # 查找所有CSV文件
                csv_files = glob.glob(os.path.join(datas_dir, "*_labeled.csv"))
                data_list.extend(csv_files)

        return data_list

    def _compute_neighborhood_features(self, coord, beamaz, feat, label):
        """修正：结合coord和beamaz计算邻域，特征维度扩展为9维"""
        """结合coord实际跨度的beamaz归一化，确保与空间维度权重均衡"""
        # 1. 计算coord各维度跨度及平均跨度（基于你的实际数据：x≈1.17万、y≈1.11万、z≈420）
        coord_spans = [
            coord[:, 0].max() - coord[:, 0].min(),  # x跨度：~11713.2
            coord[:, 1].max() - coord[:, 1].min(),  # y跨度：~11050.2
            coord[:, 2].max() - coord[:, 2].min()  # z跨度：~420.6
        ]
        avg_coord_span = np.mean(coord_spans)  # 计算结果≈7728
        #print(
        #    f"  📏 coord各维度跨度：x={coord_spans[0]:.1f}, y={coord_spans[1]:.1f}, z={coord_spans[2]:.1f}，平均跨度≈{avg_coord_span:.0f}")

        # 2. Beamaz归一化：使其跨度与coord平均跨度同量级（核心修正）
        beamaz_original_span = 360.0  # Beamaz原始范围：0~360度
        if avg_coord_span > 0:
            norm_ratio = beamaz_original_span / avg_coord_span  # ≈360/7728≈0.0466
            beamaz_normalized = beamaz / norm_ratio  # 归一后范围：0~360/0.0466≈0~7725
        else:
            norm_ratio = 3.6  # 极端情况：coord无跨度时用默认比例
            beamaz_normalized = beamaz / norm_ratio
        #print(
        #    f"  🔄 Beamaz归一化：比例={norm_ratio:.4f}，归一后范围≈{beamaz_normalized.min():.0f}~{beamaz_normalized.max():.0f}")

        # 3. 组合“coord + 归一化Beamaz”构建KDTree（此时各维度权重均衡）
        spatial_features = np.hstack([coord, beamaz_normalized.reshape(-1, 1)])  # shape: (N, 4)

        # 4. 后续邻域计算逻辑（不变，保持9维特征）
        if len(spatial_features) < self.k_neighbors:
            mean_feat = np.zeros_like(feat)
            std_feat = np.zeros_like(feat)
            new_feat = np.concatenate([feat, mean_feat, std_feat], axis=1)
            return new_feat, label.copy()

        tree = KDTree(spatial_features)
        _, indices = tree.query(spatial_features, k=self.k_neighbors)

        new_feat = np.zeros((len(spatial_features), 9), dtype=np.float32)
        new_label = np.zeros(len(spatial_features), dtype=np.int64)

        for i in range(len(spatial_features)):
            neighbor_indices = indices[i]
            neighbor_feat = feat[neighbor_indices]
            mean_feat = np.mean(neighbor_feat, axis=0)
            std_feat = np.std(neighbor_feat, axis=0)
            # 新增：处理std=0的情况（替换为1e-6，避免除以0）
            std_feat = np.where(std_feat == 0, 1e-6, std_feat)
            feat_i = feat[i].squeeze()
            new_feat[i] = np.concatenate([feat_i, mean_feat, std_feat])

            # 2. 优化标签逻辑：邻域内风切变点占比≥0.3才标1（阈值可调整）
            #neighbor_labels = generate_label[neighbor_indices]
            #shear_ratio = np.sum(neighbor_labels == 1) / len(neighbor_labels)  # 计算邻域风切变占比
            #new_label[i] = 1 if shear_ratio >= 0.3 else 0  # 占比阈值设为0.3（可根据数据调整）

            # 新多分类逻辑：取邻域中出现次数最多的类别（多数投票）
            neighbor_labels = label[neighbor_indices]  # 邻域内所有点的原始标签（0-4）
            # 统计邻域中每个类别的出现次数
            counts = np.bincount(neighbor_labels, minlength=5)  # minlength=5确保0-4类都被统计
            # 取次数最多的类别作为当前点的标签（若有平局，取最小类别）
            most_common_label = np.argmax(counts)
            new_label[i] = most_common_label

        # 新增：邻域计算后检查是否引入NaN/inf
        if np.isnan(feat).any() or np.isinf(feat).any():
            nan_mask = np.isnan(feat).any(axis=1) | np.isinf(feat).any(axis=1)
            feat = feat[~nan_mask]
            coord = coord[~nan_mask]
            label = label[~nan_mask]
            beamaz = beamaz[~nan_mask]
            logging.warning(f"邻域计算后过滤了{nan_mask.sum()}个含NaN/inf的点")

        return new_feat, new_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, warnings=None):
        csv_path = self.data_list[idx]
        try:
            # 读取CSV数据
            data = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"读取样本{csv_path}失败：{str(e)}，已跳过")
            return None  # 读取失败直接跳过

        # 🌟 核心：按完整路径过滤，仅过滤低点数的那个样本
        if csv_path in self.filter_full_paths:
            #logging.warning(f"样本 {csv_path} 因路径匹配被过滤（低点数），已跳过加载")
            return None

        # 提取坐标（x,y,z），强化异常处理
        try:
            coord = data[["x", "y", "z"]].values.astype(np.float32)
        except KeyError:
            coord = data[[" x", " y", " z"]].values.astype(np.float32)
        # 🌟 新增1：检查坐标是否有NaN/inf（源头过滤）
        coord_nan = np.isnan(coord).any(axis=1)
        coord_inf = np.isinf(coord).any(axis=1)
        if np.any(coord_nan | coord_inf):
            valid_mask = ~(coord_nan | coord_inf)
            coord = coord[valid_mask]
            logging.warning(f"样本{csv_path}原始坐标含{len(coord) - valid_mask.sum()}个NaN/inf点，已过滤")

        # 提取特征（u, v, beamaz）
        try:
            u = data["u"].values.astype(np.float32)
            v = data["v"].values.astype(np.float32)
            beamaz = data["BeamAz"].values.astype(np.float32)
        except KeyError:
            u = data[" u"].values.astype(np.float32)
            v = data[" v"].values.astype(np.float32)
            beamaz = data["BeamAz"].values.astype(np.float32)
        # 🌟 新增2：检查u/v/beamaz是否有NaN/inf
        feat_nan = np.isnan(u) | np.isnan(v) | np.isnan(beamaz)
        feat_inf = np.isinf(u) | np.isinf(v) | np.isinf(beamaz)
        if np.any(feat_nan | feat_inf):
            valid_mask = ~(feat_nan | feat_inf)
            u = u[valid_mask]
            v = v[valid_mask]
            beamaz = beamaz[valid_mask]
            coord = coord[valid_mask]  # 同步过滤坐标
            logging.warning(f"样本{csv_path}原始特征含{len(u) - valid_mask.sum()}个NaN/inf点，已过滤")

        # 组合原始特征（u, v, beamaz）- 维度从2变为3
        feat = np.column_stack([u, v, beamaz])

        # 读取标签并检查有效性
        try:
            label = data["label"].values.astype(np.int64)
        except KeyError:
            label = data[" label"].values.astype(np.int64)
        # 过滤无效标签（0-4外）并同步过滤其他字段
        valid_label_mask = (label >= 0) & (label <= 4)
        if not np.all(valid_label_mask):
            invalid_count = len(label) - valid_label_mask.sum()
            label = label[valid_label_mask]
            feat = feat[valid_label_mask]
            coord = coord[valid_label_mask]
            beamaz = beamaz[valid_label_mask]
            logging.warning(f"样本{csv_path}含{invalid_count}个无效标签（非0-4），已过滤")

        # 若过滤后无有效点，直接跳过
        if len(coord) == 0:
            logging.warning(f"样本{csv_path}过滤后无有效点，已跳过")
            return None

        # 计算邻域特征（传入beamaz参与邻域计算）
        feat, label = self._compute_neighborhood_features(coord, beamaz, feat, label)

        # 构建数据字典
        data_dict = {
            'coord': coord,
            'feat': feat,  # 此时feat为9维
            'generate_label': label,
            'path': csv_path,
            'beamaz': beamaz  # 保留原始beamaz供调试
        }

        # 执行采样等变换后，添加点数校验
        if self.transform is not None:
            try:
                data_dict = self.transform(data_dict)
            except Exception as e:
                logging.error(f"样本{csv_path}变换失败：{str(e)}，已跳过")
                return None

        # 🌟 强制恢复path（防止变换中意外丢失）
        data_dict['path'] = csv_path

        # 新增：检查采样后点数是否满足最小要求
        sampled_num = len(data_dict['coord'])
        if sampled_num < self.min_points:  # 现在self.min_points已定义
            logging.warning(f"样本{data_dict['path']}采样后点数({sampled_num})不足，已跳过")  # 修正为data_dict['path']
            return None  # 返回None标记为无效样本

        # 最终校验：确保所有字段无NaN/inf
        final_nan = (np.isnan(data_dict['coord']).any()
                     | np.isnan(data_dict['feat']).any()
                     | np.isnan(data_dict['generate_label']).any())
        if final_nan:
            logging.error(f"样本{csv_path}最终数据含NaN，已跳过")
            return None

        return data_dict
