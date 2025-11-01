import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from io import StringIO
from typing import Tuple

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 模拟数据加载和 PyTorch Dataset ---

# 模拟您的雷达点云数据
simulated_data_content = """
# 正常风场 (Volume 0)
x, y, z, u, v, w, label
10, 10, 30, 2.0, 0.5, 0.0, 0
11, 10, 31, 2.1, 0.4, 0.0, 0
10, 11, 30, 2.0, 0.6, 0.0, 0
11, 11, 31, 2.1, 0.5, 0.0, 0
12, 12, 32, 2.2, 0.5, 0.0, 0
13, 13, 33, 2.3, 0.4, 0.0, 0
14, 14, 34, 2.4, 0.5, 0.0, 0
15, 15, 35, 2.5, 0.6, 0.0, 0
# 强风切变风场 (Volume 1)
20, 20, 30, 5.0, 0.0, 0.0, 1 # P1 (强风)
21, 20, 31, 5.1, -0.1, 0.0, 1
22, 20, 32, 5.2, -0.2, 0.0, 1
23, 20, 33, 1.0, 0.0, 0.0, 1 # P4 (风速突降)
24, 20, 34, 1.1, 0.1, 0.0, 1
25, 20, 35, 1.2, 0.2, 0.0, 1
26, 20, 36, 1.3, 0.3, 0.0, 1
27, 20, 37W, 1.4, 0.4, 0.0, 1
"""
data = pd.read_csv(StringIO(simulated_data_content), comment='#')
data.columns = data.columns.str.strip()
data['volume_id'] = np.repeat([0, 1], 8)

FEATURES = ['x', 'y', 'z', 'u', 'v', 'w']
NUM_FEATURES = len(FEATURES)
NUM_POINTS = 8
NUM_VOLUMES = 2

# 格式化为 (Volume, Points, Features)
X_data_list = []
Y_labels_list = []
for i in range(NUM_VOLUMES):
    volume_data = data[data['volume_id'] == i]
    X_data_list.append(volume_data[FEATURES].values)
    Y_labels_list.append(volume_data['label'].iloc[0])

X_data = np.array(X_data_list, dtype=np.float32)
Y_labels = np.array(Y_labels_list, dtype=np.float32)


# 数据标准化（与 Keras 代码保持一致）
def normalize_data(X: np.ndarray) -> np.ndarray:
    # 对所有坐标 (x, y, z) 进行全局归一化
    xyz_data = X[:, :, :3]
    xyz_min = np.min(xyz_data)
    xyz_max = np.max(xyz_data)
    normalized_xyz = (xyz_data - xyz_min) / (xyz_max - xyz_min + 1e-6)

    # 对所有风速矢量 (u, v, w) 进行全局归一化
    uvw_data = X[:, :, 3:]
    uvw_max = np.max(np.abs(uvw_data))
    normalized_uvw = uvw_data / (uvw_max + 1e-6)

    return np.concatenate([normalized_xyz, normalized_uvw], axis=-1)


X_data_normalized = normalize_data(X_data)


# 自定义 PyTorch Dataset
class WindShearDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = WindShearDataset(X_data_normalized, Y_labels)
dataloader = DataLoader(dataset, batch_size=NUM_VOLUMES, shuffle=False)


# --- 2. DGCNN 核心组件 EdgeConv 的 PyTorch 实现 ---

def knn_graph_torch(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 k-NN 图的索引。
    输入 x: (batch_size, num_points, num_features)
    输出: 邻居索引 (batch_size, num_points, k)
    """
    # 形状: (B, N, F) -> (B, N, N) 距离矩阵
    distance_matrix = torch.cdist(x, x, p=2)

    # 找到 k 个最近邻居的索引
    # torch.topk 默认找最大值，因此不需要取负号，但 DGCNN 通常排除自身
    # 由于我们需要相对坐标，因此需要找到 k 个最近邻居的索引
    # 这里我们使用 sort 找到最小的 k+1 个，然后排除自身 (索引 0)

    # topk 返回 (values, indices)
    # distance_matrix 是距离，越小越近。
    # 我们希望找到 k 个最小距离的索引。

    # 查找 k+1 个最近邻（包含自身）
    _, indices = torch.topk(distance_matrix, k=k + 1, largest=False, sorted=True)

    # 排除第一个（自身）
    nn_index = indices[:, :, 1:]

    # nn_index 形状: (batch_size, num_points, k)
    return nn_index.to(device)


def get_edge_features_torch(point_features: torch.Tensor, nn_index: torch.Tensor, k: int) -> torch.Tensor:
    """
    根据 k-NN 索引生成 EdgeConv 所需的边特征。
    输入: point_features (B, N, F), nn_index (B, N, K)
    输出: edge_features (B, F*2, N, K) - PyTorch 卷积需要这个格式
    """
    batch_size, num_points, num_features = point_features.shape

    # 1. 收集邻居点的特征 (j)
    # nn_index: (B, N, K)

    # 扩展 batch 索引
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
    batch_indices = batch_indices.repeat(1, num_points, k).view(-1)

    # 邻居点的所有索引 (B * N * K)
    neighbor_indices = nn_index.contiguous().view(-1)

    # 构建 gather 索引 (B*N*K, 2)
    # torch.gather 只能沿一个维度操作，使用 advanced indexing (张量索引)

    # 由于高级索引操作复杂且效率低，我们使用一个更 PyTorch 的方法：
    # 将 B * N * F 转换为 B * F * N, 然后使用 F.fold/unfold 的思路

    # 使用 PyG 的思想：将点特征展平到 (B*N, F)，然后利用索引

    # 更直接的方法：
    # 1. 扩展中心点特征 (i)
    point_features_i = point_features.unsqueeze(2).repeat(1, 1, k, 1)  # (B, N, K, F)

    # 2. 收集邻居点特征 (j)
    # point_features_j: (B, N, K, F)
    point_features_j = torch.gather(
        point_features.unsqueeze(1).repeat(1, num_points, 1, 1),  # (B, N, N, F)
        dim=2,
        index=nn_index.unsqueeze(-1).repeat(1, 1, 1, num_features)  # (B, N, K, F)
    )

    # 3. 计算边特征
    edge_features = torch.cat([
        point_features_i,  # 中心点特征 (F)
        point_features_j - point_features_i  # 相对特征 (F)
    ], dim=-1)  # (B, N, K, 2*F)

    # PyTorch Conv2D 期望的形状: (B, C, H, W) -> (B, 2*F, N, K)
    edge_features = edge_features.permute(0, 3, 1, 2)  # (B, 2*F, N, K)

    return edge_features.contiguous()


# 封装为 EdgeConv 层
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        # EdgeConv 内部的 MLP (1x1 卷积) 输入是 2*in_channels
        self.conv = nn.Conv2d(2 * in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, N, F_in)

        # 1. 构建 k-NN 图
        # nn_index 形状: (B, N, K)
        nn_index = knn_graph_torch(x, self.k)

        # 2. 提取边特征
        # edge_features 形状: (B, 2*F_in, N, K)
        edge_features = get_edge_features_torch(x, nn_index, self.k)

        # 3. EdgeConv 操作 (1x1 卷积)
        x_conv = self.conv(edge_features)
        x_conv = self.bn(x_conv)
        x_conv = F.relu(x_conv)

        # 4. 聚合 (对 k 个邻居进行 Max Pooling)
        # 形状: (B, F_out, N)
        x_aggregated = torch.max(x_conv, dim=3)[0]

        # 转换回 (B, N, F_out) 形状，用于下一层输入
        return x_aggregated.permute(0, 2, 1).contiguous()


# --- 3. DGCNN 模型构建 ---

class DGCNN_WindShear(nn.Module):
    def __init__(self, num_points, num_features, k=4):
        super(DGCNN_WindShear, self).__init__()
        self.k = k

        # EdgeConv 块
        self.edge_conv1 = EdgeConv(num_features, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)

        # 全局特征 MLP (用于分类)
        # 聚合特征维度: 64 + 64 + 128 + 256 = 512
        # 实际 DGCNN 通常拼接所有层的全局特征，这里简化为只拼接 Max/Avg Pool
        # MaxPool 形状: 256, AvgPool 形状: 256 -> Concatenate 512
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256),  # MaxPool(256) + AvgPool(256)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出为二分类
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, N, F)

        # EdgeConv 块
        x1 = self.edge_conv1(x)  # (B, N, 64)
        x2 = self.edge_conv2(x1)  # (B, N, 64)
        x3 = self.edge_conv3(x2)  # (B, N, 128)
        x4 = self.edge_conv4(x3)  # (B, N, 256)

        # 全局特征聚合 (Max Pooling and Avg Pooling)
        # torch.max(input, dim) 返回 (values, indices)

        # Max Pool: (B, N, F) -> (B, F)
        max_pool = torch.max(x4, dim=1)[0]
        # Avg Pool: (B, N, F) -> (B, F)
        avg_pool = torch.mean(x4, dim=1)

        global_feature = torch.cat([max_pool, avg_pool], dim=1)  # (B, 512)

        # 分类器
        logits = self.classifier(global_feature)

        # Sigmoid 输出 (用于二分类)
        return torch.sigmoid(logits)


# --- 4. 训练与预测演示 ---

K_NEIGHBORS = 4
model = DGCNN_WindShear(NUM_POINTS, NUM_FEATURES, k=K_NEIGHBORS).to(device)

# 损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n--- DGCNN PyTorch 模型结构摘要 (设备: {device}) ---")
# 仅打印模型结构 (不打印形状)
print(model)

# 模拟训练
EPOCHS = 50
print(f"\n--- 开始模型训练 (模拟训练，数据量太小，结果无意义) ---")
model.train()
try:
    for epoch in range(EPOCHS):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"DGCNN 模拟训练完成。最终 Loss: {loss.item():.4f}")

    # 模拟预测
    model.eval()
    with torch.no_grad():
        X_test = dataset.data.to(device)
        Y_test = dataset.labels

        predictions = model(X_test)

        print("\n--- 模拟预测结果 ---")
        for i in range(NUM_VOLUMES):
            pred_score = predictions[i].item()
            is_wind_shear = pred_score > 0.5
            true_label = Y_test[i].item()

            print(
                f"Volume {i} (真实标签: {int(true_label)}): 预测分数 {pred_score:.4f} -> {'风切变' if is_wind_shear else '正常'}")

except Exception as e:
    print(f"DGCNN PyTorch 训练或预测过程中发生错误：{e}")
    print("DGCNN 模型对数据量和计算资源要求较高。请确保 PyTorch 和 CUDA（如果使用 GPU）环境正确。")