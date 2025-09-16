# pointcept/datasets/wind_shear.py

import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pointcept.utils.logger import get_root_logger


class WindShearDataset(Dataset):
    def __init__(self, split='train', data_root="D:/model/wind_datas/csv_labels", transform=None):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.transform = transform
        self.data_list = self._get_data_list()

        logger = get_root_logger()
        logger.info(f"WindShearDataset {split} split: {len(self.data_list)} scenes")

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
                csv_files = glob.glob(os.path.join(datas_dir, "*.csv"))
                data_list.extend(csv_files)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        csv_path = self.data_list[idx]

        # 读取CSV数据
        data = pd.read_csv(csv_path)

        # 提取坐标、风速和标签
        coord = data[["x", "y", "z"]].values.astype(np.float32)
        feat = data[["u", "v"]].values.astype(np.float32)
        label = data["wind_shear_label"].values.astype(np.int64)

        # 构建数据字典
        data_dict = {
            'coord': coord,
            'feat': feat,
            'label': label,
            'path': csv_path
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict