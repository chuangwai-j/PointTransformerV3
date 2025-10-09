import os
import pandas as pd
import numpy as np


# 风切变等级编码
def encode_shear_label(shear):
    if pd.isna(shear) or shear == 0:
        return 0
    elif shear <= 0.07:
        return 1
    elif shear <= 0.13:
        return 2
    elif shear <= 0.20:
        return 3
    else:
        return 4


# 计算相邻层风切变
def compute_shear(df):
    # 按无空格的'z'列排序
    df = df.sort_values('z').reset_index(drop=True)
    shear_list = [np.nan]

    for i in range(1, len(df)):
        # 使用无空格的列名计算
        du = df.at[i, 'u'] - df.at[i - 1, 'u']
        dv = df.at[i, 'v'] - df.at[i - 1, 'v']
        dz = df.at[i, 'z'] - df.at[i - 1, 'z']

        if dz == 0:
            shear = np.nan
        else:
            shear = np.sqrt(du ** 2 + dv ** 2) / dz

        shear_list.append(shear)

    df['shear'] = shear_list
    df['generate_label'] = df['shear'].apply(encode_shear_label)

    # 指定要保留的列（仅控制列名和顺序，不影响分隔符）
    return df[['x', 'y', 'z', 'u', 'v', 'BeamAz', 'shear', 'generate_label']]


# 主处理函数
def process_file(input_path, output_path):
    try:
        # 读取原始文件（逗号+空格分隔）
        df = pd.read_csv(input_path, sep=r',\s*', engine='python')
    except Exception as e:
        print(f"读取失败: {input_path} -> {e}")
        return

    # 打印原始列名（确认格式）
    print(f"处理文件: {input_path}")
    print(f"原始列名: {df.columns.tolist()}")

    # 检查必要列（无空格列名）
    required_columns = {'u', 'v', 'z', 'BeamAz', 'x', 'y'}
    if required_columns.issubset(df.columns):
        result = df.groupby('BeamAz', group_keys=False).apply(compute_shear)
    else:
        missing = required_columns - set(df.columns)
        print(f"缺少必要列: {input_path} -> 缺少 {missing}")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 关键：显式指定分隔符为单个逗号，确保输出格式正确
    result.to_csv(output_path, index=False, sep=',')
    print(f"已生成: {output_path}")


# 遍历所有文件
base_input = r"/mnt/d/model/wind_datas/csv_datas"
base_output = r"/mnt/d/model/wind_datas/csv_labels"

for day in range(1, 32):
    date_str = f"202303{day:02d}"
    date_input = os.path.join(base_input, date_str)

    if not os.path.isdir(date_input):
        continue

    for datas in os.listdir(date_input):
        datas_input = os.path.join(date_input, datas)

        if not os.path.isdir(datas_input):
            continue

        for file in os.listdir(datas_input):
            if file.endswith('.csv'):
                input_file = os.path.join(datas_input, file)
                output_file = os.path.join(base_output, date_str, datas, file.replace('.csv', '_labeled.csv'))
                process_file(input_file, output_file)
