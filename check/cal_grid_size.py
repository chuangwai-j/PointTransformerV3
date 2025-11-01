import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# --- 1. 配置您的路径和参数 ---
# (与上一个脚本保持一致)

# 您的数据根目录
DATA_ROOT = "/mnt/d/model/wind_datas/csv_labels"

# 训练集对应的日期文件夹
TRAIN_DATES = [f"202303{i:02d}" for i in range(1, 23)]

# 需要过滤的低点数样本的完整路径
FILTER_PATHS_LIST = [
    "/mnt/d/model/wind_datas/csv_labels/20230310/datas4/period107_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230319/datas1/nn217_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230314/datas1/aa217_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230317/datas1/gg1_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230308/datas1/i1_labeled.csv",
    "/mnt/d/model/wind_datas/csv_labels/20230310/datas1/period110_labeled.csv"
]
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

def analyze_data_for_grid_size():
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

    # 存储所有样本的统计数据
    all_point_counts = []
    all_x_ranges = []
    all_y_ranges = []
    all_z_ranges = []

    pbar = tqdm(data_list, desc="分析文件")
    for csv_path in pbar:
        # 3.1 过滤低点数样本
        if csv_path in FILTER_PATHS_SET:
            continue

        try:
            # 3.2 读取数据 (只读坐标)
            # 修复：确保读取了 label 列用于后续过滤
            data = pd.read_csv(csv_path, usecols=['x', 'y', 'z', 'label'])
            if data.empty:
                continue

            # 3.3 提取坐标
            cols = get_columns(data, ['x', 'y', 'z', 'label'])
            coord = np.stack([cols['x'], cols['y'], cols['z']], axis=1)
            label = cols['label']

            # 3.4 清洗 NaN/Inf (同 __getitem__)
            coord_nan = np.isnan(coord).any(axis=1)
            coord_inf = np.isinf(coord).any(axis=1)
            label_valid = (label >= 0) & (label <= 4)
            valid_mask = ~(coord_nan | coord_inf) & label_valid

            coord = coord[valid_mask]

            # 3.5 🌟 修改2：应用高度过滤 (0 <= z <= 1000)
            height_mask = (coord[:, 2] <= MAX_HEIGHT) & (coord[:, 2] >= MIN_HEIGHT)
            coord = coord[height_mask]

            if coord.shape[0] == 0:
                logging.warning(f"文件 {csv_path} 过滤后无数据。")
                continue

            # 3.6 记录统计数据
            all_point_counts.append(coord.shape[0])
            # 修复：确保在计算范围前，点数大于0
            if coord.shape[0] > 0:
                all_x_ranges.append(coord[:, 0].max() - coord[:, 0].min())
                all_y_ranges.append(coord[:, 1].max() - coord[:, 1].min())
                all_z_ranges.append(coord[:, 2].max() - coord[:, 2].min())

        except Exception as e:
            logging.error(f"处理文件 {csv_path} 失败: {e}", exc_info=True)

    logging.info("所有文件分析完毕，开始计算统计数据...")

    # 4. 计算最终统计
    avg_point_count = np.mean(all_point_counts)
    min_point_count = np.min(all_point_counts)
    max_point_count = np.max(all_point_counts)

    avg_x_range = np.mean(all_x_ranges)
    avg_y_range = np.mean(all_y_ranges)
    avg_z_range = np.mean(all_z_ranges)

    # 5. 打印报告
    print(f"\n--- ( {MIN_HEIGHT}m <= z <= {MAX_HEIGHT}m ) 训练数据分析报告 ---")
    print("\n[点数统计 (过滤后, 采样前)]")
    print(f"  平均点数: {avg_point_count:.0f} (每个样本)")
    print(f"  点数范围: {min_point_count:.0f} (最少) - {max_point_count:.0f} (最多)")

    print("\n[空间范围统计 (平均值)]")
    print(f"  平均 X 轴范围: {avg_x_range:.1f} (米)")
    print(f"  平均 Y 轴范围: {avg_y_range:.1f} (米)")
    print(f"  平均 Z 轴范围: {avg_z_range:.1f} (米)  (最大为 {MAX_HEIGHT})")

    print("\n--- 如何选择新的 grid_size ---")
    print("`grid_size` 是一个超参数，您需要根据以上统计数据进行权衡。")
    print("目标：选择 (grid_x, grid_y, grid_z)，使采样后的点数在合理范围（如 1000 - 5000 点）。")
    print("\n[推荐的设置策略 (选择一种)]")

    # 策略1: 目标 X/Y 轴 100个体素, Z 轴 50个体素 (分辨率中等)
    rec_x_mid = avg_x_range / 100
    rec_y_mid = avg_y_range / 100
    rec_z_mid = avg_z_range / 50
    print("\n[选项1: 中等分辨率 (推荐起点)]")
    print("  目标: X/Y 轴约 100 个体素, Z 轴约 50 个体素")
    # 添加保护，防止 avg_z_range 为 0
    print(f"  - grid_size: [{rec_x_mid:.1f}, {rec_y_mid:.1f}, {max(0.1, rec_z_mid):.1f}]")
    print(f"  (计算: X={avg_x_range:.0f}/100, Y={avg_y_range:.0f}/100, Z={avg_z_range:.0f}/50)")

    # 策略2: 目标 X/Y 轴 150个体素, Z 轴 75个体素 (分辨率较高)
    rec_x_high = avg_x_range / 150
    rec_y_high = avg_y_range / 150
    rec_z_high = avg_z_range / 75
    print("\n[选项2: 较高分辨率 (点数更多, 显存占用高)]")
    print("  目标: X/Y 轴约 150 个体素, Z 轴约 75 个体素")
    print(f"  - grid_size: [{rec_x_high:.1f}, {rec_y_high:.1f}, {max(0.1, rec_z_high):.1f}]")
    print(f"  (计算: X={avg_x_range:.0f}/150, Y={avg_y_range:.0f}/150, Z={avg_z_range:.0f}/75)")

    # 策略3: 目标 X/Y 轴 80个体素, Z 轴 40个体素 (分辨率较低)
    rec_x_low = avg_x_range / 80
    rec_y_low = avg_y_range / 80
    rec_z_low = avg_z_range / 40
    print("\n[选项3: 较低分辨率 (点数更少, 速度快)]")
    print("  目标: X/Y 轴约 80 个体素, Z 轴约 40 个体素")
    print(f"  - grid_size: [{rec_x_low:.1f}, {rec_y_low:.1f}, {max(0.1, rec_z_low):.1f}]")
    print(f"  (计算: X={avg_x_range:.0f}/80, Y={avg_y_range:.0f}/80, Z={avg_z_range:.0f}/40)")

    print("\n[重要提示]")
    print(f"1. 您当前的 `grid_size` 是 [122.6, 118.0, 5.4]。请对比一下 '选项1' 的推荐值。")
    print(f"2. 选定 `grid_size` 后，请在训练时密切关注 `collate_fn` 打印的日志。")
    print(f"3. 查找 `Batch生成成功：... 总点数=XXXXX` 这条日志。")
    print(f"4. 如果总点数（补点后）经常是 1536, 1920 (即 384*4 或 384*5)，说明采样后点数较少。")
    print(f"5. 如果总点数非常大 (如 > 10000)，说明采样点过多，您可能需要增大 `grid_size` (使用 '选项3' 的思路)。")


if __name__ == "__main__":
    analyze_data_for_grid_size()