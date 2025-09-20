# pointcept/datasets/transforms/wind_shear.py (新建文件)
import os
import logging
import numpy as np
from .builder import TRANSFORMS
from scipy.spatial import cKDTree
from pointcept.utils.registry import Registry

@TRANSFORMS.register_module()
class NormalizeWind(object):
    """标准化风速数据（新增beamaz归一化）"""

    def __init__(self, u_mean=0.0, u_std=1.0, v_mean=0.0, v_std=1.0,
                 beamaz_mean=0.0, beamaz_std=1.0):  # 新增beamaz参数
        self.u_mean = u_mean
        self.u_std = u_std
        self.v_mean = v_mean
        self.v_std = v_std
        self.beamaz_mean = beamaz_mean  # 新增
        self.beamaz_std = beamaz_std    # 新增

    '''def __call__(self, data_dict):
        if 'feat' in data_dict:
            feat = data_dict['feat'].copy()
            # 打印归一化前的各列均值（用于判断对应哪个特征）
            print("归一化前 feat 各列均值：")
            print(f"第0列：{feat[:, 0].mean():.2f}（应接近原始u的均值：0.59）")
            print(f"第1列：{feat[:, 1].mean():.2f}（应接近原始v的均值：-5.18）")
            print(f"第2列：{feat[:, 2].mean():.2f}（可能是dist的均值）")
            print(f"第3列：{feat[:, 3].mean():.2f}（应接近原始BeamAz的均值：192.23）")
            # 标准化u（第0维）
            feat[:, 0] = (feat[:, 0] - self.u_mean) / self.u_std
            # 标准化v（第1维）
            feat[:, 1] = (feat[:, 1] - self.v_mean) / self.v_std
            # 标准化beamaz（第2维）- 新增
            feat[:, 2] = (feat[:, 2] - self.beamaz_mean) / self.beamaz_std
            data_dict['feat'] = feat
        return data_dict
    '''

    def __call__(self, data_dict):
        if 'feat' in data_dict:
            feat = data_dict['feat'].copy()

            # 归一化u相关特征（索引0,3,6）
            feat[:, 0] = (feat[:, 0] - self.u_mean) / self.u_std  # 原始u
            feat[:, 3] = (feat[:, 3] - self.u_mean) / self.u_std  # u邻域均值
            feat[:, 6] = (feat[:, 6] - self.u_mean) / self.u_std  # u邻域方差

            # 归一化v相关特征（索引1,4,7）
            feat[:, 1] = (feat[:, 1] - self.v_mean) / self.v_std  # 原始v
            feat[:, 4] = (feat[:, 4] - self.v_mean) / self.v_std  # v邻域均值
            feat[:, 7] = (feat[:, 7] - self.v_mean) / self.v_std  # v邻域方差

            # 归一化beamaz相关特征（索引2,5,8）
            feat[:, 2] = (feat[:, 2] - self.beamaz_mean) / self.beamaz_std  # 原始beamaz
            feat[:, 5] = (feat[:, 5] - self.beamaz_mean) / self.beamaz_std  # beamaz邻域均值
            feat[:, 8] = (feat[:, 8] - self.beamaz_mean) / self.beamaz_std  # beamaz邻域方差

            data_dict['feat'] = feat
        return data_dict

@TRANSFORMS.register_module()
class WindShearGridSample:
    def __init__(self, grid_size=80.0, min_points=50, adaptive=True):
        self.grid_size = grid_size  # 支持标量或三维列表（固定模式下）
        self.min_points = min_points
        self.adaptive = adaptive

    def __call__(self, data_dict):
        # 1. 提取原始数据（含beamaz）
        coord = data_dict["coord"]  # 原始坐标 (N, 3)
        feat = data_dict["feat"]  # 原始特征 (N, C)
        label = data_dict["label"]  # 原始标签 (N,)
        beamaz = data_dict.get("beamaz", None)  # 提取beamaz（可能不存在）
        sample_path = data_dict.get("path", "未知样本")

        # -------------------------- 核心修复：确保x_min/y_min/z_min在所有模式下都被定义 --------------------------
        # 先计算坐标全局最小值（无论自适应与否都需要）
        x_min_global = coord[:, 0].min()
        y_min_global = coord[:, 1].min()
        z_min_global = coord[:, 2].min()
        # --------------------------------------------------------------------------------------------------

        # 2. 网格采样核心逻辑（获取采样点的索引）
        if self.adaptive:
            # 自适应网格采样：根据坐标范围调整网格大小（保持原逻辑）
            x_min, y_min, z_min = coord.min(axis=0)  # 这里复用全局最小值计算结果（等价）
            x_max, y_max, z_max = coord.max(axis=0)
            grid_size_x = max(self.grid_size, (x_max - x_min) / 50)
            grid_size_y = max(self.grid_size, (y_max - y_min) / 50)
            grid_size_z = max(self.grid_size, (z_max - z_min) / 10)
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

        # 计算每个点所属的网格索引（现在x_min/y_min/z_min在两种模式下都已定义）
        grid_idx = np.floor(
            (coord - [x_min, y_min, z_min]) / [grid_size_x, grid_size_y, grid_size_z]
        ).astype(int)

        # 为每个网格分配唯一ID（避免冲突）
        grid_id = grid_idx[:, 0] * (grid_idx[:, 1].max() + 1) * (grid_idx[:, 2].max() + 1) + \
                  grid_idx[:, 1] * (grid_idx[:, 2].max() + 1) + grid_idx[:, 2]

        # 🌟 关键：获取每个网格的采样点索引（确保采样后beamaz可同步筛选）
        sampled_indices = []
        for gid in np.unique(grid_id):
            # 每个网格内随机选1个点
            grid_points = np.where(grid_id == gid)[0]
            sampled_indices.append(np.random.choice(grid_points, 1))
        sampled_indices = np.concatenate(sampled_indices, axis=0)

        # 3. 🌟 同步采样：根据索引筛选coord/feat/label/beamaz
        sampled_coord = coord[sampled_indices]
        sampled_feat = feat[sampled_indices]
        sampled_label = label[sampled_indices]
        # 同步筛选beamaz（确保长度与采样后点数一致）
        if beamaz is not None:
            sampled_beamaz = beamaz[sampled_indices]
            assert len(sampled_beamaz) == len(sampled_coord), \
                f"{sample_path}采样后beamaz长度({len(sampled_beamaz)})与coord点数({len(sampled_coord)})不匹配"
        else:
            sampled_beamaz = None

        # 4. 补点至384的倍数（同步补beamaz）—— 完整保留你的补点逻辑
        sampled_num = len(sampled_coord)
        if sampled_num < self.min_points:
            # 这里不直接报错，而是后续在__getitem__中返回None跳过（与之前的过滤逻辑呼应）
            # 保留警告信息便于调试
            print(
                f"⚠️ {os.path.basename(sample_path)}采样后点数({sampled_num})小于最小要求({self.min_points})，将被跳过")
            # 返回空数据触发过滤
            data_dict["coord"] = np.empty((0, 3), dtype=np.float32)
            return data_dict

        min_multiple = 384  # 满足分块+下采样需求
        target_num = max(min_multiple, ((sampled_num + min_multiple - 1) // min_multiple) * min_multiple)
        pad_num = target_num - sampled_num

        if pad_num > 0:
            # 用KDTree找近邻点，补点时保持空间关联性
            tree = cKDTree(sampled_coord)
            _, pad_indices = tree.query(sampled_coord[:pad_num], k=1)  # 待补点的近邻索引

            # 同步补coord/feat/label
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)
            sampled_label = np.concatenate([sampled_label, sampled_label[pad_indices]], axis=0)
            # 🌟 同步补beamaz
            if sampled_beamaz is not None:
                sampled_beamaz = np.concatenate([sampled_beamaz, sampled_beamaz[pad_indices]], axis=0)

            logging.debug(f"[补点] {os.path.basename(sample_path)} | 点数{sampled_num}→{target_num}（384的倍数）")

        # 5. 更新数据字典（含同步处理后的beamaz）
        data_dict.update({
            "coord": sampled_coord,
            "feat": sampled_feat,
            "label": sampled_label,
            "grid_size": np.array([grid_size_x, grid_size_y, grid_size_z])  # 保留网格大小
        })
        # 放回同步处理后的beamaz
        if sampled_beamaz is not None:
            data_dict["beamaz"] = sampled_beamaz
            # 最终校验：beamaz与coord点数必须一致
            assert len(data_dict["beamaz"]) == len(data_dict["coord"]), \
                f"{sample_path}最终beamaz长度({len(data_dict['beamaz'])})与coord点数({len(data_dict['coord'])})不匹配"

        return data_dict
"""class WindShearGridSample(object):
    #对风切变数据进行网格采样（适配9维特征）

    def __init__(self, grid_size=80.0, min_points=10, adaptive=False):
        self.grid_size = grid_size
        self.min_points = min_points
        self.adaptive = adaptive  # 是否自适应调整网格大小

    def __call__(self, data_dict):
        coord = data_dict['coord']
        feat = data_dict['feat']  # 9维特征
        label = data_dict['label']
        sample_path = data_dict.get('path', '未知路径')
        original_num = coord.shape[0]

        # 🌟 新增：打印原始点数
        print(f"[采样前] {os.path.basename(sample_path)} | 原始点数：{original_num}")

        # 如果点数已经很少，直接返回，不进行采样
        if original_num <= self.min_points:
            data_dict['grid_size'] = np.array(0.0, dtype=np.float32)  # 标记为未采样
            return data_dict

        # 自适应调整网格大小
        effective_grid_size = self.grid_size
        if self.adaptive:
            # 根据数据范围动态调整网格大小
            data_range = np.ptp(coord, axis=0).max()
            effective_grid_size = max(self.grid_size, data_range / 100)  # 确保至少分成100个网格

        # 网格采样
        voxel_coord = np.floor(coord / effective_grid_size)
        unique_voxels, inverse_indices = np.unique(voxel_coord, axis=0, return_inverse=True)
        sampled_num = len(unique_voxels)

        # 🌟 新增：打印采样后点数
        print(
            f"[采样后] {os.path.basename(sample_path)} | 采样后点数：{sampled_num} | 实际grid_size：{effective_grid_size}")

        # 如果采样后点数太少，调整网格大小重新采样
        if sampled_num < self.min_points and self.adaptive:
            # 增大网格大小
            effective_grid_size *= 2
            voxel_coord = np.floor(coord / effective_grid_size)
            unique_voxels, inverse_indices = np.unique(voxel_coord, axis=0, return_inverse=True)
            sampled_num = len(unique_voxels)

        # 采样计算
        if sampled_num > 0:
            sampled_coord = np.zeros((sampled_num, 3))
            sampled_feat = np.zeros((sampled_num, feat.shape[1]))
            sampled_label = np.zeros(sampled_num)

            for i in range(sampled_num):
                mask = inverse_indices == i
                sampled_coord[i] = np.mean(coord[mask], axis=0)
                sampled_feat[i] = np.mean(feat[mask], axis=0)
                sampled_label[i] = np.round(np.mean(label[mask]))
        else:
            print(f"[兜底] {os.path.basename(sample_path)} | 采样后空，保留原始点")
            # 采样失败，使用原始数据
            sampled_coord = coord
            sampled_feat = feat
            sampled_label = label

            # 🌟 新增：确保采样后点数≥48（与enc_patch_size[0]一致）
        if sampled_num < 48:
            print(f"[补点] {os.path.basename(sample_path)} | 采样后点数{sampled_num} < 48，补至48")
            # 重复采样点至48个（简单兜底，不影响分布）
            repeat_times = (48 // sampled_num) + 1
            sampled_coord = np.repeat(sampled_coord, repeat_times, axis=0)[:48]
            sampled_feat = np.repeat(sampled_feat, repeat_times, axis=0)[:48]
            sampled_label = np.repeat(sampled_label, repeat_times, axis=0)[:48]
            sampled_num = 48

        sampled_num = sampled_coord.shape[0]
        '''# 🌟 关键：补点至48的倍数（enc_patch_size=48，同时覆盖头数整除需求）
        patch_size = 48  # 与配置中的enc_patch_size一致
        target_num = max(patch_size, ((sampled_num + patch_size - 1) // patch_size) * patch_size)
        if sampled_num != target_num:
            pad_num = target_num - sampled_num
            # 随机重复近邻点（更贴合空间分布）
            from scipy.spatial import cKDTree
            tree = cKDTree(sampled_coord)
            _, pad_indices = tree.query(sampled_coord[:pad_num], k=1)  # 补近邻点
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)
            sampled_label = np.concatenate([sampled_label, sampled_label[pad_indices]], axis=0)
            # 同步补beamaz（关键！避免维度不一致）
            if 'beamaz' in data_dict:
                data_dict['beamaz'] = np.concatenate([data_dict['beamaz'], data_dict['beamaz'][pad_indices]], axis=0)
            print(f"[补点] {os.path.basename(sample_path)} | 点数{sampled_num}→{target_num}（48的倍数）")
        '''

        min_multiple = 384  # 3次下采样后仍能被48整除
        target_num = max(min_multiple, ((sampled_num + min_multiple - 1) // min_multiple) * min_multiple)

        if sampled_num != target_num:
            pad_num = target_num - sampled_num
            # 用KDTree找近邻，确保补点的空间和射线关联性
            from scipy.spatial import cKDTree
            tree = cKDTree(sampled_coord)
            _, pad_indices = tree.query(sampled_coord[:pad_num], k=1)  # 为每个待补点找最近邻

            # 1. 补coord/feat/label（原有逻辑）
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)
            sampled_label = np.concatenate([sampled_label, sampled_label[pad_indices]], axis=0)

            # 2. 🌟 同步补beamaz，继承近邻点的方位角（关键！保持射线关联性）
            if "beamaz" in data_dict:
                # 确保原始beamaz长度与采样后点数一致
                assert data_dict["beamaz"].shape[
                           0] == sampled_num, f"beamaz长度({data_dict['beamaz'].shape[0]})与采样后点数({sampled_num})不匹配"
                # 补点的beamaz = 近邻点的beamaz
                data_dict["beamaz"] = np.concatenate(
                    [data_dict["beamaz"], data_dict["beamaz"][pad_indices]],
                    axis=0
                )

            # 打印补点后维度，验证一致性
            print(f"[补点后] {os.path.basename(data_dict['path'])} | 总点数: {target_num}")
            print(f"  coord: {sampled_coord.shape}, feat: {sampled_feat.shape}, label: {sampled_label.shape}")
            if "beamaz" in data_dict:
                print(f"  beamaz: {data_dict['beamaz'].shape} (与coord一致)")

        # 更新数据
        data_dict['coord'] = sampled_coord.astype(np.float32)
        data_dict['feat'] = sampled_feat.astype(np.float32)
        data_dict['label'] = sampled_label.astype(np.int64)
        data_dict['grid_size'] = np.array(effective_grid_size, dtype=np.float32)

        return data_dict"""