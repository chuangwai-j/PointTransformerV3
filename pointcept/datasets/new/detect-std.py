import pandas as pd
import numpy as np
from io import StringIO

# 这是多普勒雷达风切变检测的经典方法之一。

# 1.  **风切变的定义**: 风切变强度（Wind Shear Intensity）是风速矢量在空间上的变化率。$\text{Shear} = |\frac{\vec{V}_2 - \vec{V}_1}{\Delta \vec{r}}| \approx \frac{|\Delta V|}{|\Delta r|}$。
# 2.  **数据选择**: 由于您提供的风场数据是沿着单一光束（`BeamAz=42`, `BeamEl=6`）采集的，因此我们主要计算**沿着光束方向**（即径向）和**垂直方向**上的切变。
# 3.  **径向风切变**: 在多普勒雷达中，$\text{Radial Shear} = \frac{\Delta V_r}{\Delta r}$ 是最直接的量测，其中 $V_r$ 是径向速度。由于您提供的 $u, v, w$ 是转换后的三维风速分量，`speed` 是合成风速 $|V|$，所以我们计算 $\frac{\Delta |\vec{V}|}{\Delta r}$ 作为近似的径向总风速切变。
# 4.  **实际应用中的复杂性**:
#     * **下滑道扫描 (Glide Path Scanning)**: 机场实际使用的算法（如香港天文台和国内主流算法）通常会采用**下滑道扫描**模式，将径向风速投影到飞机下滑道上，检测**下滑道迎/顺风分量**的急剧变化（称为斜坡检测 Ramps Detection）。
#     * **微爆流/阵风锋**: 对于由雷暴引起的风切变，需要使用更复杂的算法，如识别风场中的辐散/辐合模式（例如微爆流的径向速度差 $\Delta V_r$ 超过阈值），或识别阵风锋的特征。
#     * **矢量切变**: 严格的水平和垂直风切变应分别计算为 $\frac{\partial \vec{V}}{\partial x}$ 和 $\frac{\partial \vec{V}}{\partial z}$，涉及到对 $u, v, w$ 矢量分量的偏导数计算，这需要更密集的多光束或体积扫描数据。上面的示例使用了简化版的标量切变强度计算。
    
# 常用的低空风切变判据是：在小于 600 米（或 1600 英尺）的高度内，如果风速（或径向风速）在水平或垂直距离上的变化超过一定阈值，则被判定为风切变。
# 空间位置划分: 风切变通常在垂直（如下滑道方向）或水平方向上进行检测。
# 这里我们按距离 $dist$ 和高度 $z$ 划分，并计算相邻点之间的风切变强度。
# 风切变强度计算: 风切变强度 $\text{Shear} = \frac{\Delta \text{WindSpeed}}{\Delta \text{Distance}}$。
# 垂直风切变强度：$\text{Vertical Shear} = \frac{\Delta \text{Total Wind Speed}}{\Delta z}$。
# 水平风切变强度：$\text{Horizontal Shear} = \frac{\Delta \text{Total Wind Speed}}{\Delta \text{Horizontal Distance}}$。
# 告警阈值: 根据国际民航组织的规定或经验值设定告警阈值（例如 $\text{X} \text{m/s}$ per $100 \text{m}$）。
# 在实际应用中，阈值会根据机场标准和雷达类型进行调整。

# --- 1. 模拟数据加载 ---
# 您的文件内容，为了代码的完整性和可运行性，我们用一个字符串来模拟文件内容
file_content = """
BeamAz, BeamEl, dist, x, y, z, speed, direction, u, v, w
42, 6, 300.750000, 200.138611, 222.276443, 31.436935, 1.080000, 329.399994, 0.929601, -0.549765, 0.000000
42, 6, 401.250000, 267.017853, 296.553375, 41.942047, 0.810000, 330.000000, 0.701481, -0.405000, 0.000000
42, 6, 501.750000, 333.897095, 370.830292, 52.447155, 0.460000, 326.600006, 0.384030, -0.253221, 0.000000
42, 6, 602.250000, 400.776306, 445.107208, 62.952267, 0.200000, 109.599998, -0.067090, 0.188411, 0.000000
42, 6, 702.750000, 467.655548, 519.384094, 73.457375, 0.500000, 145.100006, -0.410076, 0.286073, 0.000000
42, 6, 803.250000, 534.534790, 593.661011, 83.962486, 0.510000, 175.800003, -0.508630, 0.037351, 0.000000
"""
df = pd.read_csv(StringIO(file_content))
df.columns = df.columns.str.strip()
# --- 2. 参数设置 ---
# 定义低空风切变检测的高度范围（单位：米）。
# 通常低空风切变定义在600米以下，但在机场跑道附近常关注更低的高度。
LOW_LEVEL_THRESHOLD_Z = 600 

# 定义风切变告警的阈值（单位：(m/s) / m）。
# 行业经验：约 0.04 m/s/m (即 4 m/s per 100m) 为中等强度风切变的一般参考。
WIND_SHEAR_THRESHOLD = 0.04 

# --- 3. 预处理和计算 ---

# 筛选低于阈值高度的数据点
df_low_level = df[df['z'] < LOW_LEVEL_THRESHOLD_Z].copy()

if df_low_level.empty:
    print(f"在 {LOW_LEVEL_THRESHOLD_Z} 米以下没有检测到数据点。")
else:
    # 按照距离（dist）进行排序，确保计算是在光束路径上连续进行的
    df_low_level = df_low_level.sort_values(by='dist').reset_index(drop=True)

    # 计算相邻点的距离差（Delta Distance）
    # 垂直距离差 (dz)
    df_low_level['dz'] = df_low_level['z'].diff().abs() 
    # 水平距离差 (dh = sqrt(dx^2 + dy^2))
    df_low_level['dx'] = df_low_level['x'].diff().abs()
    df_low_level['dy'] = df_low_level['y'].diff().abs()
    df_low_level['dh'] = np.sqrt(df_low_level['dx']**2 + df_low_level['dy']**2)
    # 空间总距离差 (dr = dist.diff())
    df_low_level['dr'] = df_low_level['dist'].diff().abs()

    # 计算相邻点的风速差 (Delta Wind Speed)
    # 使用总风速 (speed) 的变化
    df_low_level['d_speed'] = df_low_level['speed'].diff().abs()
    
    # 也可以使用风速矢量（u, v, w）的模变化
    # Total_Wind_Vector = np.sqrt(df_low_level['u']**2 + df_low_level['v']**2 + df_low_level['w']**2)
    # df_low_level['d_total_vec_speed'] = Total_Wind_Vector.diff().abs()

    # --- 4. 风切变强度计算 ---

    # 垂直风切变强度（单位：(m/s)/m）
    # 仅在 dz > 0 的地方计算，避免除以零
    df_low_level['vertical_shear_intensity'] = np.where(
        df_low_level['dz'] > 0.1, # 设定一个小的非零阈值
        df_low_level['d_speed'] / df_low_level['dz'], 
        0.0
    )

    # 水平风切变强度（单位：(m/s)/m）
    # 仅在 dh > 0 的地方计算
    df_low_level['horizontal_shear_intensity'] = np.where(
        df_low_level['dh'] > 0.1, 
        df_low_level['d_speed'] / df_low_level['dh'],
        0.0
    )

    # 沿光束方向（径向）的风切变强度 (主要用于径向速度切变分析，这里使用总风速)
    df_low_level['radial_shear_intensity'] = np.where(
        df_low_level['dr'] > 0.1, 
        df_low_level['d_speed'] / df_low_level['dr'],
        0.0
    )

    # --- 5. 风切变检测与告警 ---

    # 标记检测到的风切变事件
    df_low_level['is_wind_shear'] = (
        (df_low_level['vertical_shear_intensity'] >= WIND_SHEAR_THRESHOLD) |
        (df_low_level['horizontal_shear_intensity'] >= WIND_SHEAR_THRESHOLD) |
        (df_low_level['radial_shear_intensity'] >= WIND_SHEAR_THRESHOLD)
    )

    # 筛选出风切变事件
    wind_shear_events = df_low_level[df_low_level['is_wind_shear']].copy()

    # 结果输出
    print(f"\n--- 低空风切变检测结果 (阈值: {WIND_SHEAR_THRESHOLD:.2f} m/s/m) ---")
    if wind_shear_events.empty:
        print("未检测到超过阈值的低空风切变事件。")
    else:
        print(f"检测到 {len(wind_shear_events)} 个潜在的低空风切变事件。")
        
        # 为了清晰，只展示关键信息
        events_summary = wind_shear_events[[
            'dist', 'z', 'speed', 'd_speed', 'dr', 
            'vertical_shear_intensity', 'horizontal_shear_intensity', 'radial_shear_intensity',
            'is_wind_shear'
        ]].rename(columns={'dist': '距离(m)', 'z': '高度(m)', 'speed': '风速(m/s)', 'd_speed': '风速差(m/s)', 
                           'dr': '径向距离差(m)', 'vertical_shear_intensity': '垂直切变强度(s^-1)',
                           'horizontal_shear_intensity': '水平切变强度(s^-1)', 'radial_shear_intensity': '径向切变强度(s^-1)'})
        
        # 垂直切变超过阈值的事件
        vertical_events = events_summary[events_summary['垂直切变强度(s^-1)'] >= WIND_SHEAR_THRESHOLD]
        if not vertical_events.empty:
            print("\n--- 垂直风切变告警 (z < 600m) ---")
            print(vertical_events)
            
        # 水平切变超过阈值的事件
        horizontal_events = events_summary[events_summary['水平切变强度(s^-1)'] >= WIND_SHEAR_THRESHOLD]
        if not horizontal_events.empty:
            print("\n--- 水平风切变告警 (z < 600m) ---")
            print(horizontal_events)
            
        # 径向切变超过阈值的事件
        radial_events = events_summary[events_summary['径向切变强度(s^-1)'] >= WIND_SHEAR_THRESHOLD]
        if not radial_events.empty:
            print("\n--- 径向风切变告警 (z < 600m) ---")
            print(radial_events)

