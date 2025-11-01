import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# --- 1. 配置您的路径和参数 ---

# 您的数据根目录
DATA_ROOT = "/mnt/d/model/wind_datas/csv_labels"

# 训练集对应的日期文件夹
# (与 WindShearDataset.__init__ 中的 'train' split 保持一致)
TRAIN_DATES = [f"202303{i:02d}" for i in range(1, 23)]

# 需要过滤的低点数样本的完整路径
# (从您的 yaml 配置文件中复制)
FILTER_PATHS_LIST = [
    "/mnt/d/model/wind_datas/csv_labels/20230310/datas4/period107_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230319/datas1/nn217_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230314/datas1/aa217_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230317/datas1/gg1_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230308/datas1/i1_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230310/datas1/period110_labeled.csv"
]
# 转换为 Set 结构以加快查找速度
FILTER_PATHS_SET = set(FILTER_PATHS_LIST)

# --- 🌟 修改1：添加最小高度限制 ---
MAX_HEIGHT = 1000.0
MIN_HEIGHT = 0.0  # 假设地面为 0 米

# --- 2. 辅助函数，用于安全读取列 ---

def get_columns(df, columns):
    """
    尝试读取列，兼容带空格和不带空格的列名
    """
    data = {}
    for col in columns:
        try:
            data[col] = df[col].values
        except KeyError:
            try:
                data[col] = df[" " + col].values
            except Exception as e:
                logging.error(f"无法读取列 {col} 或 ' {col}'. 错误: {e}")
                raise e
    return data

# --- 3. 主计算逻辑 ---

def recalculate_stats_corrected():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data_list = []
    logging.info("开始搜寻训练集文件...")
    for date in TRAIN_DATES:
        date_path = os.path.join(DATA_ROOT, date)
        if not os.path.exists(date_path):
            continue
        datas_dirs = glob.glob(os.path.join(date_path, "datas*"))
        for datas_dir in datas_dirs:
            csv_files = glob.glob(os.path.join(datas_dir, "*_labeled.csv"))
            data_list.extend(csv_files)

    logging.info(f"共找到 {len(data_list)} 个训练集文件。")

    all_u = []
    all_v = []
    all_beamaz = []

    pbar = tqdm(data_list, desc="处理文件")
    for csv_path in pbar:
        # 2.1 过滤低点数样本
        if csv_path in FILTER_PATHS_SET:
            logging.debug(f"跳过低点数样本: {csv_path}")
            continue

        try:
            # 2.2 读取数据 (确保读取所有相关列，包括带空格的)
            data = pd.read_csv(csv_path, usecols=['x', 'y', 'z', 'u', 'v', 'BeamAz'])
            if data.empty:
                logging.warning(f"文件为空: {csv_path}")
                continue

            # 2.3 提取所需列
            cols = get_columns(data, ['x', 'y', 'z', 'u', 'v', 'BeamAz'])

            # 2.4 🌟 修改2：应用高度过滤 (0 <= z <= 1000)
            height_mask = (cols['z'] <= MAX_HEIGHT) & (cols['z'] >= MIN_HEIGHT)

            # 2.5 过滤后数据
            u_filtered = cols['u'][height_mask]
            v_filtered = cols['v'][height_mask]
            beamaz_filtered = cols['BeamAz'][height_mask]

            if len(u_filtered) == 0:
                # logging.warning(f"文件 {csv_path} 在 {MIN_HEIGHT}m <= z <= {MAX_HEIGHT}m 过滤后无数据。")
                continue

            # 2.6 清洗 NaN/Inf (同 __getitem__)
            valid_mask_u = ~ (np.isnan(u_filtered) | np.isinf(u_filtered))
            valid_mask_v = ~ (np.isnan(v_filtered) | np.isinf(v_filtered))
            valid_mask_beamaz = ~ (np.isnan(beamaz_filtered) | np.isinf(beamaz_filtered))

            valid_mask_all = valid_mask_u & valid_mask_v & valid_mask_beamaz

            all_u.append(u_filtered[valid_mask_all])
            all_v.append(v_filtered[valid_mask_all])
            all_beamaz.append(beamaz_filtered[valid_mask_all])

        except Exception as e:
            logging.error(f"处理文件 {csv_path} 失败: {e}", exc_info=True)

    logging.info("所有文件处理完毕，开始合并数据...")

    # 3. 合并并计算统计数据
    # 使用 float64 以提高计算精度
    all_u = np.concatenate(all_u, dtype=np.float64)
    all_v = np.concatenate(all_v, dtype=np.float64)
    all_beamaz = np.concatenate(all_beamaz, dtype=np.float64)

    logging.info(f"在 {MIN_HEIGHT}m <= z <= {MAX_HEIGHT}m 条件下，共加载 {len(all_u)} 个有效点。")

    u_mean = np.mean(all_u)
    u_std = np.std(all_u)
    v_mean = np.mean(all_v)
    v_std = np.std(all_v)
    beamaz_mean = np.mean(all_beamaz)
    beamaz_std = np.std(all_beamaz)

    # 4. 打印结果
    print("\n--- 修正后的计算结果 (0 <= z <= 1000)，请使用这个更新 .yaml ---")
    print("    transform:")
    print("      - type: NormalizeWind")
    print(f"        u_mean: {u_mean:.4f}")
    print(f"        u_std: {u_std:.4f}")
    print(f"        v_mean: {v_mean:.4f}")
    print(f"        v_std: {v_std:.4f}")
    print(f"        beamaz_mean: {beamaz_mean:.4f}")
    print(f"        beamaz_std: {beamaz_std:.4f}")
    print("      - type: WindShearGridSample")
    print(f"        ...")


if __name__ == "__main__":
    recalculate_stats_corrected()