import pandas as pd
import numpy as np

# 1. 加载CSV文件
csv_path = "/mnt/d/model/wind_datas/csv_labels/20230308/datas2/j101_labeled.csv"
df = pd.read_csv(csv_path)

# 2. 检查列名是否完整（带空格的列名）
expected_columns = [' x', ' y', ' z', ' u', ' v', ' dist', 'wind_shear_label', 'BeamAz']
assert list(df.columns) == expected_columns, f"列名不匹配：{df.columns} vs {expected_columns}"

# 3. 检查是否有缺失值
assert not df.isnull().any().any(), f"文件{csv_path}存在缺失值"

# 4. 检查数值是否合理（无异常值）
for col in [' x', ' y', ' z']:  # 坐标列（带空格）
    assert np.all(np.isfinite(df[col])), f"坐标{col}存在非有限值（NaN/Inf）"
for col in [' u', ' v', 'BeamAz']:  # 特征列（注意'u'和'v'带空格）
    assert np.all(np.isfinite(df[col])), f"特征{col}存在非有限值"

# 5. 打印基本统计量（正确引用带空格的列名）
print(f"文件 {csv_path} 统计：")
print(f"点数：{len(df)}")
print("坐标范围：")
print(f"x: [{df[' x'].min():.1f}, {df[' x'].max():.1f}]")
print(f"y: [{df[' y'].min():.1f}, {df[' y'].max():.1f}]")
print(f"z: [{df[' z'].min():.1f}, {df[' z'].max():.1f}]")
print("特征范围：")
print(f"u: [{df[' u'].min():.2f}, {df[' u'].max():.2f}]")  # 修正：' u'带空格
print(f"v: [{df[' v'].min():.2f}, {df[' v'].max():.2f}]")  # 修正：' v'带空格
print(f"BeamAz: [{df['BeamAz'].min():.2f}, {df['BeamAz'].max():.2f}]")