import pandas as pd
import glob
import numpy as np

# 遍历所有训练集CSV
train_csvs = glob.glob("/mnt/d/model/wind_datas/csv_labels/*/datas*/period*_labeled.csv")  # 根据实际train集路径调整
u_list, v_list, beamaz_list = [], [], []

for csv in train_csvs:
    df = pd.read_csv(csv)
    u_list.extend(df[' u'].dropna().values)
    v_list.extend(df[' v'].dropna().values)
    beamaz_list.extend(df['BeamAz'].dropna().values)

# 计算真实均值和标准差
u_mean, u_std = np.mean(u_list), np.std(u_list)
v_mean, v_std = np.mean(v_list), np.std(v_list)
beamaz_mean, beamaz_std = np.mean(beamaz_list), np.std(beamaz_list)

print(f"u_mean={u_mean:.4f}, u_std={u_std:.4f}")
print(f"v_mean={v_mean:.4f}, v_std={v_std:.4f}")
print(f"beamaz_mean={beamaz_mean:.4f}, beamaz_std={beamaz_std:.4f}")