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
                 transform=None, k_neighbors=16, radius=0.5, min_points=50):  # 新增min_points参数
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.transform = build_transform(transform)
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.min_points = min_points  # 新增：初始化min_points属性
        self.data_list = self._get_data_list()

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
            feat_i = feat[i].squeeze()
            new_feat[i] = np.concatenate([feat_i, mean_feat, std_feat])

            # 2. 优化标签逻辑：邻域内风切变点占比≥0.3才标1（阈值可调整）
            neighbor_labels = label[neighbor_indices]
            shear_ratio = np.sum(neighbor_labels == 1) / len(neighbor_labels)  # 计算邻域风切变占比
            new_label[i] = 1 if shear_ratio >= 0.3 else 0  # 占比阈值设为0.3（可根据数据调整）

        return new_feat, new_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        csv_path = self.data_list[idx]

        # 读取CSV数据
        data = pd.read_csv(csv_path)

        # 新增：打印当前文件的列名（只在调试时用，之后可以删除）
        logging.debug(f"\nCSV文件路径：{csv_path}")
        #print("列名列表：", data.columns.tolist())  # 这行是关键

        # 提取坐标、风速和标签
        # 注意：列名可能有空格，也可能没有，这里尝试两种可能
        # 读取坐标（x,y,z）
        try:
            coord = data[["x", "y", "z"]].values.astype(np.float32)
        except KeyError:
            coord = data[[" x", " y", " z"]].values.astype(np.float32)

        # 读取特征（u, v, beamaz）- 新增beamaz
        try:
            u = data["u"].values.astype(np.float32)
            v = data["v"].values.astype(np.float32)
            beamaz = data["BeamAz"].values.astype(np.float32)  # 新增beamaz读取
        except KeyError:
            u = data[" u"].values.astype(np.float32)
            v = data[" v"].values.astype(np.float32)
            beamaz = data["BeamAz"].values.astype(np.float32)  # 处理带空格列名

        # 组合原始特征（u, v, beamaz）- 维度从2变为3
        feat = np.column_stack([u, v, beamaz])

        # 读取标签
        label = data["wind_shear_label"].values.astype(np.int64)

        # 计算邻域特征（传入beamaz参与邻域计算）
        feat, label = self._compute_neighborhood_features(coord, beamaz, feat, label)

        # 构建数据字典
        data_dict = {
            'coord': coord,
            'feat': feat,  # 此时feat为9维
            'label': label,
            'path': csv_path,
            'beamaz': beamaz  # 保留原始beamaz供调试
        }

        # 执行采样等变换后，添加点数校验
        data_dict = self.transform(data_dict)

        # 新增：检查采样后点数是否满足最小要求
        sampled_num = len(data_dict['coord'])
        if sampled_num < self.min_points:  # 现在self.min_points已定义
            # 打印警告信息（可选）
            import warnings
            warnings.warn(f"样本{data_dict['path']}采样后点数({sampled_num})不足，已跳过")  # 修正为data_dict['path']
            return None  # 返回None标记为无效样本

        return data_dict
