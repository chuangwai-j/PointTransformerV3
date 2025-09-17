import numpy as np
from . import transforms


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
            # 标准化前6个通道（原始+均值+标准差）
            feat = data_dict['feat'].copy()
            for i in range(0, 6, 2):  # 处理每对(u, v)特征
                feat[:, i] = (feat[:, i] - self.u_mean) / self.u_std
                feat[:, i + 1] = (feat[:, i + 1] - self.v_mean) / self.v_std
            data_dict['feat'] = feat

        return data_dict