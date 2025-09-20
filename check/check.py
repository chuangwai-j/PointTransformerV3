# analyze_data.py
import pandas as pd
import numpy as np
import glob
import os


def analyze_data_distribution(data_root):
    csv_files = glob.glob(os.path.join(data_root, "**", "*_labeled.csv"), recursive=True)

    stats = {
        'point_counts': [],
        'x_ranges': [],
        'y_ranges': [],
        'z_ranges': []
    }

    for csv_file in csv_files[:100]:  # 分析前100个文件
        data = pd.read_csv(csv_file)

        # 获取坐标数据
        try:
            coord = data[["x", "y", "z"]].values
        except KeyError:
            coord = data[[" x", " y", " z"]].values

        # 统计信息
        stats['point_counts'].append(len(coord))
        stats['x_ranges'].append(np.ptp(coord[:, 0]))
        stats['y_ranges'].append(np.ptp(coord[:, 1]))
        stats['z_ranges'].append(np.ptp(coord[:, 2]))

    # 打印统计信息
    print("数据分布分析:")
    print(f"平均点数: {np.mean(stats['point_counts']):.2f}")
    print(f"点数范围: {min(stats['point_counts'])} - {max(stats['point_counts'])}")
    print(f"X轴平均范围: {np.mean(stats['x_ranges']):.2f}")
    print(f"Y轴平均范围: {np.mean(stats['y_ranges']):.2f}")
    print(f"Z轴平均范围: {np.mean(stats['z_ranges']):.2f}")

    # 建议网格大小
    avg_range = (np.mean(stats['x_ranges']) + np.mean(stats['y_ranges']) + np.mean(stats['z_ranges'])) / 3
    suggested_grid_size = avg_range / 100  # 将平均范围分成100份

    print(f"建议网格大小: {suggested_grid_size:.2f}")


# 运行分析
analyze_data_distribution("/mnt/d/model/wind_datas/csv_labels")