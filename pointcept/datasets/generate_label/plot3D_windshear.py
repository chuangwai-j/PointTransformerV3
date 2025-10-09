import pandas as pd
import matplotlib.pyplot as plt
# 强制使用交互式后端（关键修改）
import matplotlib
matplotlib.use('TkAgg')  # 确保使用支持交互的后端（TkAgg适用于大多数环境）
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------
# 1. 读入数据
# -------------------------------------------------
file = r'/mnt/d/model/wind_datas/csv_labels/20230301/datas1/period2_labeled.csv'
df = pd.read_csv(file)

# 确保label列非空
df = df.dropna(subset=['label'])

# -------------------------------------------------
# 2. 颜色映射
# -------------------------------------------------
color_map = {
    0: '#ffffff',   # 无
    1: '#00ff00',   # 轻度
    2: '#ffff00',   # 中度
    3: '#ff8800',   # 强烈
    4: '#ff0000'    # 严重
}
df['color'] = df['label'].map(color_map)

# -------------------------------------------------
# 3. 绘图（保持原有逻辑，增强交互提示）
# -------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
sc = ax.scatter(
    df['x'], df['y'], df['z'],
    c=df['color'],
    s=15,
    edgecolors='k',
    linewidths=0.2,
    alpha=0.8
)

# 坐标轴设置
ax.set_xlabel('X (m)', fontsize=10)
ax.set_ylabel('Y (m)', fontsize=10)
ax.set_zlabel('Z (m)', fontsize=10)
ax.set_title('3-D Wind-Shear Intensity (拖动鼠标旋转视角)', fontsize=12)

# 图例
for lbl, col in color_map.items():
    ax.scatter([], [], [], c=col, label=f'等级 {lbl}', s=40)
ax.legend(title='风切变等级', loc='upper right')

# 增强交互体验：显示提示文本
plt.figtext(0.5, 0.01, "提示：拖动鼠标旋转视角 | 滚轮缩放 | 右键平移",
            ha='center', fontsize=8, style='italic')

# 启动交互模式并显示
plt.tight_layout()
plt.show()
