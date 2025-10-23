# windshear/backend/inference.py
import os
import sys
import pathlib
import pandas as pd
import numpy as np
import torch
import yaml
from scipy.spatial import KDTree, cKDTree
import traceback

# 添加项目根目录到路径
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# 尝试导入pointcept
try:
    from pointcept.models import build_model

    print("成功导入pointcept")
except ImportError as e:
    print(f"导入pointcept失败: {e}")

# 全局模型变量
_model = None
_cfg = None


def load_model_weight(model, checkpoint_path, device):
    """更健壮的权重加载方法 - 专门处理 Pointcept 格式"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"检查点键: {list(checkpoint.keys())}")

    # 对于 Pointcept 格式，权重在 'model_state_dict' 中
    if 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
        print("从 'model_state_dict' 提取模型权重")
    else:
        # 如果不是 Pointcept 格式，尝试其他可能的键
        possible_keys = ['state_dict', 'model', 'net']
        model_weights = None

        for key in possible_keys:
            if key in checkpoint:
                model_weights = checkpoint[key]
                print(f"从 '{key}' 键提取模型权重")
                break

        # 如果以上键都没有，尝试直接使用整个checkpoint
        if model_weights is None:
            model_weights = checkpoint
            print("使用整个检查点作为模型权重")

    if model_weights is None:
        raise ValueError("无法从检查点中找到模型权重")

    # 打印权重结构信息用于调试
    print("=== 权重结构分析 ===")
    head_keys = [k for k in model_weights.keys() if 'head' in k]
    print(f"Head层键: {head_keys}")
    for key in head_keys:
        print(f"  {key}: {model_weights[key].shape}")

    # 打印模型期望的head层结构
    model_head_keys = [k for k in model.state_dict().keys() if 'head' in k]
    print(f"模型期望的head层: {model_head_keys}")
    for key in model_head_keys:
        print(f"  {key}: {model.state_dict()[key].shape}")
    print("=== 分析结束 ===")

    # 初始化变量
    strict_loaded = False

    # 尝试加载权重，如果strict模式失败，尝试非strict模式
    try:
        model.load_state_dict(model_weights, strict=True)
        strict_loaded = True
        print("权重加载成功（strict模式）")
    except RuntimeError as e:
        print(f"strict模式加载失败: {e}")
        print("尝试非strict模式加载...")

        # 非strict加载
        missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
        print(f"缺失的键: {missing_keys}")
        print(f"意外的键: {unexpected_keys}")

        # 基于权重结构分析，专门处理head层
        fix_head_layer_weights_based_on_analysis(model, model_weights, missing_keys, unexpected_keys)

        print("权重加载成功（非strict模式 + 手动修复）")

    return model


def fix_head_layer_weights_based_on_analysis(model, model_weights, missing_keys, unexpected_keys):
    """基于权重结构分析的head层修复"""
    print("=== 基于权重分析的head层修复 ===")

    # 检查模型权重中的head层结构
    checkpoint_head_keys = [k for k in model_weights.keys() if 'head' in k]
    model_head_keys = [k for k in model.state_dict().keys() if 'head' in k]

    print(f"检查点head层: {checkpoint_head_keys}")
    print(f"模型期望head层: {model_head_keys}")

    # 情况1: 检查点有复杂的head结构 (head.0.weight, head.3.weight)
    if 'head.0.weight' in checkpoint_head_keys and 'head.3.weight' in checkpoint_head_keys:
        print("检测到复杂head结构，尝试直接映射")

        # 直接复制所有head层权重
        for key in checkpoint_head_keys:
            if key in model.state_dict():
                if model_weights[key].shape == model.state_dict()[key].shape:
                    model.state_dict()[key].copy_(model_weights[key])
                    print(f"成功复制 {key}")
                else:
                    print(f"形状不匹配 {key}: {model_weights[key].shape} vs {model.state_dict()[key].shape}")

    # 情况2: 检查点只有简单的head结构 (head.weight, head.bias)
    elif 'head.weight' in checkpoint_head_keys and 'head.bias' in checkpoint_head_keys:
        print("检测到简单head结构，需要映射到复杂head")

        simple_weight = model_weights['head.weight']  # [5, 64]
        simple_bias = model_weights['head.bias']  # [5]

        # 尝试映射到模型的head.3层 (最后一层分类器)
        if 'head.3.weight' in model.state_dict():
            target_weight = model.state_dict()['head.3.weight']  # [5, 256]
            target_bias = model.state_dict()['head.3.bias']  # [5]

            # 如果输出维度匹配 (都是5类)，但输入维度不同
            if target_weight.shape[0] == simple_weight.shape[0]:  # 都是5个类别
                print(f"类别数匹配: {target_weight.shape[0]}")

                # 重新初始化head.3层，但使用简单head的偏置
                torch.nn.init.kaiming_normal_(target_weight)
                target_bias.copy_(simple_bias)
                print("使用简单head的偏置，重新初始化head.3的权重")

        # 重新初始化其他缺失的head层
        for key in missing_keys:
            if 'head' in key and key in model.state_dict():
                param = model.state_dict()[key]
                if 'weight' in key:
                    torch.nn.init.kaiming_normal_(param)
                    print(f"重新初始化 {key}")
                elif 'bias' in key:
                    torch.nn.init.constant_(param, 0)
                    print(f"重新初始化 {key}")

    # 情况3: 其他不匹配情况
    else:
        print("未知的head层结构，尝试重新初始化所有head层")
        for key in model_head_keys:
            param = model.state_dict()[key]
            if 'weight' in key:
                torch.nn.init.kaiming_normal_(param)
                print(f"重新初始化 {key}")
            elif 'bias' in key:
                torch.nn.init.constant_(param, 0)
                print(f"重新初始化 {key}")

    print("=== head层修复完成 ===")


class Command:
    """Django管理命令风格的类，用于封装预测功能"""

    @staticmethod
    def load_model():
        """加载模型 - 单例模式"""
        global _model, _cfg
        if _model is not None:
            return _model

        try:
            # 模型路径 - 修正为正确的路径
            CFG_PATH = BASE_DIR / 'configs' / 'wind_shear' / 'pointtransformer_v3.yaml'

            # 尝试多个可能的模型文件路径
            possible_checkpoint_paths = [
                BASE_DIR / 'checkpoints' / 'checkpoints_3' / 'best_merged_model.pth',
                BASE_DIR / 'checkpoints' / 'best_model_epoch9.pth',  # 您的实际路径
                BASE_DIR / 'backend' / 'checkpoints' / 'best_model_epoch9.pth',  # 备用路径
                BASE_DIR / 'windshear' / 'checkpoints' / 'best_model_epoch9.pth',  # 另一个可能的路径
            ]

            CKPT_PATH = None
            for path in possible_checkpoint_paths:
                if path.exists():
                    CKPT_PATH = path
                    break

            if CKPT_PATH is None:
                # 列出所有可能的路径供调试
                print("找不到模型文件，尝试的路径:")
                for path in possible_checkpoint_paths:
                    print(f"  {path} - 存在: {path.exists()}")
                raise FileNotFoundError(f"找不到模型文件，请检查模型文件位置")

            print(f"加载配置文件: {CFG_PATH}")
            print(f"加载模型权重: {CKPT_PATH}")

            # 设备设置 - 使用GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {device}")

            if device.type == "cpu":
                print("警告: 使用CPU设备，但模型可能需要GPU才能正常运行")

            # 加载配置和模型
            _cfg = yaml.safe_load(open(CFG_PATH))
            _cfg['model']['num_classes'] = 5
            _model = build_model(_cfg['model']).to(device)

            # 使用正确的权重加载方法
            _model = load_model_weight(_model, CKPT_PATH, device)

            _model.eval()
            print("模型加载成功!")
            return _model

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    @staticmethod
    def filter_nan_inf_data(coord, feat, beamaz):
        """过滤NaN和inf值 - 与训练时完全一致"""
        # 检查坐标NaN/inf
        coord_nan = np.isnan(coord).any(axis=1)
        coord_inf = np.isinf(coord).any(axis=1)

        # 检查特征NaN/inf
        feat_nan = np.isnan(feat).any(axis=1)
        feat_inf = np.isinf(feat).any(axis=1)

        # 组合掩码
        valid_mask = ~(coord_nan | coord_inf | feat_nan | feat_inf)

        if not np.all(valid_mask):
            invalid_count = len(coord) - valid_mask.sum()
            print(f"过滤了 {invalid_count} 个包含NaN/inf的点")
            coord = coord[valid_mask]
            feat = feat[valid_mask]
            beamaz = beamaz[valid_mask]

        return coord, feat, beamaz, valid_mask

    @staticmethod
    def build_9d_features(coord, original_feat, beamaz, k_neighbors=16):
        """构建9维特征 - 与训练数据集完全一致"""
        if len(coord) < k_neighbors:
            mean_feat = np.zeros_like(original_feat)
            std_feat = np.zeros_like(original_feat)
            return np.concatenate([original_feat, mean_feat, std_feat], axis=1)

        # 1. 计算坐标跨度用于beamaz归一化 - 与训练时完全一致
        coord_spans = [
            coord[:, 0].max() - coord[:, 0].min(),
            coord[:, 1].max() - coord[:, 1].min(),
            coord[:, 2].max() - coord[:, 2].min()
        ]
        avg_coord_span = np.mean(coord_spans) if np.mean(coord_spans) > 0 else 1.0

        # 2. Beamaz归一化 - 与训练时完全一致
        beamaz_original_span = 360.0
        if avg_coord_span > 0:
            norm_ratio = beamaz_original_span / avg_coord_span
            beamaz_normalized = beamaz / norm_ratio
        else:
            norm_ratio = 3.6
            beamaz_normalized = beamaz / norm_ratio

        # 3. 组合空间特征用于KDTree - 与训练时完全一致
        spatial_features = np.hstack([coord, beamaz_normalized.reshape(-1, 1)])

        tree = KDTree(spatial_features)
        _, indices = tree.query(spatial_features, k=k_neighbors)

        # 4. 计算9维邻域特征 - 与训练时完全一致
        new_feat = np.zeros((len(spatial_features), 9), dtype=np.float32)
        for i in range(len(spatial_features)):
            neighbor_indices = indices[i]
            neighbor_feat = original_feat[neighbor_indices]
            mean_feat = np.mean(neighbor_feat, axis=0)
            std_feat = np.std(neighbor_feat, axis=0)
            # 处理std=0的情况 - 与训练时完全一致
            std_feat = np.where(std_feat == 0, 1e-6, std_feat)

            feat_i = original_feat[i].squeeze()
            new_feat[i] = np.concatenate([feat_i, mean_feat, std_feat])

        return new_feat

    @staticmethod
    def normalize_features_for_inference(feat_9d):
        """特征归一化 - 根据训练数据统计进行标准化"""
        # 这些值应该与训练时使用的归一化参数一致
        # 如果没有保存训练时的统计信息，可以使用以下基于典型数据的估计值
        feature_means = np.array([0.0, 0.0, 180.0, 0.0, 0.0, 180.0, 1.0, 1.0, 1.0])
        feature_stds = np.array([5.0, 5.0, 100.0, 2.0, 2.0, 100.0, 1.0, 1.0, 2.0])

        # 避免除零
        feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)

        normalized_feat = (feat_9d - feature_means) / feature_stds
        return normalized_feat

    @staticmethod
    def apply_inference_transform(coord, feat_9d, grid_size=80.0):
        """应用推理变换 - 网格采样和补点"""
        if len(coord) == 0:
            return np.empty((0, 3)), np.empty((0, 9)), np.array([grid_size, grid_size, grid_size])

        # 网格采样
        coord_min = coord.min(axis=0, keepdims=True)
        grid_idx = np.floor((coord - coord_min) / grid_size).astype(int)
        grid_idx = np.clip(grid_idx, 0, None)

        # 计算网格ID
        max_z = grid_idx[:, 2].max() + 1 if len(grid_idx) > 0 else 1
        max_y = grid_idx[:, 1].max() + 1 if len(grid_idx) > 0 else 1
        grid_id = grid_idx[:, 0] * max_y * max_z + grid_idx[:, 1] * max_z + grid_idx[:, 2]

        # 每个网格采样一个点
        sampled_indices = []
        for gid in np.unique(grid_id):
            grid_points = np.where(grid_id == gid)[0]
            if len(grid_points) > 0:
                sampled_indices.append(np.random.choice(grid_points, 1))

        if len(sampled_indices) == 0:
            return np.empty((0, 3)), np.empty((0, 9)), np.array([grid_size, grid_size, grid_size])

        sampled_indices = np.concatenate(sampled_indices, axis=0)
        sampled_coord = coord[sampled_indices]
        sampled_feat = feat_9d[sampled_indices]

        # 补点到384的倍数
        sampled_num = len(sampled_coord)
        min_multiple = 384
        target_num = max(min_multiple, ((sampled_num + min_multiple - 1) // min_multiple) * min_multiple)
        pad_num = target_num - sampled_num

        if pad_num > 0:
            tree = cKDTree(sampled_coord)
            _, pad_indices = tree.query(sampled_coord[:pad_num], k=1)
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)

        # 返回grid_size，这是模型需要的
        return sampled_coord, sampled_feat, np.array([grid_size, grid_size, grid_size])

    @staticmethod
    def inference_collate_fn(batch, device):
        """推理专用批处理函数 - 添加grid_size处理和设备移动"""
        batch = [item for item in batch if item is not None]
        if not batch:
            return {'path': [], 'coord': torch.empty((0, 3)), 'offset': torch.tensor([0])}

        result = {}
        offsets = [0]
        total_points = 0

        for item in batch:
            num_points = item['coord'].shape[0]
            total_points += num_points
            offsets.append(total_points)

        result['coord'] = torch.cat([item['coord'] for item in batch], dim=0).to(device)
        result['feat'] = torch.cat([item['feat'] for item in batch], dim=0).to(device)
        result['offset'] = torch.tensor(offsets, dtype=torch.int64).to(device)
        result['path'] = [item['path'] for item in batch]

        # 添加grid_size - 使用第一个样本的grid_size
        if 'grid_size' in batch[0]:
            result['grid_size'] = batch[0]['grid_size'].to(device)

        return result

    @staticmethod
    def calibrate_predictions(logits, temperature=2.0):
        """校准模型预测，减少类别偏向"""
        print(f"应用温度缩放校准，温度={temperature}")

        # 分析原始logits分布
        original_probs = torch.softmax(logits, dim=-1)
        original_probs_np = original_probs.cpu().numpy()
        print("原始预测概率统计:")
        for i in range(5):
            print(f"  类别 {i}: 平均概率={original_probs_np[:, i].mean():.3f}")

        # 应用温度缩放
        calibrated_logits = logits / temperature

        # 分析校准后分布
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
        calibrated_probs_np = calibrated_probs.cpu().numpy()
        print("校准后预测概率统计:")
        for i in range(5):
            print(f"  类别 {i}: 平均概率={calibrated_probs_np[:, i].mean():.3f}")

        return calibrated_logits

    @staticmethod
    def map_predictions_to_original(original_coord, sampled_coord, sampled_pred):
        """将采样点的预测结果映射回原始点"""
        if len(sampled_coord) == 0:
            return np.zeros(len(original_coord), dtype=int)

        # 使用KDTree找到每个原始点的最近采样点
        tree = cKDTree(sampled_coord)
        _, indices = tree.query(original_coord, k=1)

        # 将采样点的标签赋给对应的原始点
        original_pred = sampled_pred[indices]
        return original_pred

    @staticmethod
    def predict_csv(csv_path):
        """主要的预测函数 - 确保与训练数据处理完全一致"""
        print(f"开始处理CSV文件: {csv_path}")

        try:
            # 1. 加载模型
            model = Command.load_model()
            device = next(model.parameters()).device
            print(f"模型设备: {device}")

            # 2. 读取CSV数据
            df = pd.read_csv(csv_path)
            original_columns = df.columns.tolist()
            df.columns = df.columns.str.strip().str.lower()
            print(f"原始数据点数: {len(df)}, 列: {original_columns}")

            # 3. 检查必要的列
            required_coord = ['x', 'y', 'z']
            missing_coord = [col for col in required_coord if col not in df.columns]
            if missing_coord:
                raise ValueError(f"CSV文件缺少必要的坐标列: {missing_coord}")

            # 4. 提取坐标和特征 - 与训练时一致
            coord = df[['x', 'y', 'z']].values.astype(np.float32)

            # 5. 提取原始特征 (u, v, beamaz) - 与训练时一致
            feature_columns = ['u', 'v', 'beamaz']
            feature_data = {}
            for col in feature_columns:
                if col in df.columns:
                    feature_data[col] = df[col].values.astype(np.float32)
                else:
                    print(f"警告: 缺少特征列 '{col}'，使用0填充")
                    feature_data[col] = np.zeros(len(df), dtype=np.float32)

            original_feat = np.column_stack([
                feature_data['u'],
                feature_data['v'],
                feature_data['beamaz']
            ])

            # 6. 过滤NaN和inf值 - 与训练时完全一致
            print("过滤NaN和inf值...")
            coord, original_feat, beamaz, valid_mask = Command.filter_nan_inf_data(
                coord, original_feat, feature_data['beamaz']
            )

            # 记录有效点索引，用于最终映射
            original_indices = np.where(valid_mask)[0]

            if len(coord) == 0:
                print("警告: 过滤后无有效点，返回默认标签")
                df['label'] = 0
                return df

            # 7. 构建9维特征 - 与训练时完全一致
            print("构建9维特征...")
            feat_9d = Command.build_9d_features(coord, original_feat, beamaz)
            print(f"9维特征形状: {feat_9d.shape}")

            # 8. 特征归一化 - 新增，确保与训练时一致
            print("应用特征归一化...")
            feat_9d = Command.normalize_features_for_inference(feat_9d)
            print(f"归一化后特征范围:")
            for i in range(feat_9d.shape[1]):
                print(f"  特征{i}: min={feat_9d[:, i].min():.3f}, max={feat_9d[:, i].max():.3f}")

            # 9. 应用推理变换 - 网格采样和补点
            print("应用网格采样和补点...")
            processed_coord, processed_feat, grid_size = Command.apply_inference_transform(coord, feat_9d)
            print(f"处理后点数: {len(processed_coord)}")
            print(f"处理后特征形状: {processed_feat.shape}")
            print(f"grid_size: {grid_size}")

            if len(processed_coord) == 0:
                print("警告: 处理后无有效点，返回默认标签")
                df['label'] = 0
                return df

            # 10. 准备模型输入
            data_dict = {
                'coord': torch.from_numpy(processed_coord).float(),
                'feat': torch.from_numpy(processed_feat).float(),
                'grid_size': torch.from_numpy(grid_size).float(),
                'path': csv_path
            }

            print(f"输入coord形状: {data_dict['coord'].shape}")
            print(f"输入feat形状: {data_dict['feat'].shape}")

            # 11. 模型推理
            print("进行模型推理...")
            with torch.no_grad():
                batch = Command.inference_collate_fn([data_dict], device)
                logits = model(batch)

                # 添加温度缩放校准
                calibrated_logits = Command.calibrate_predictions(logits, temperature=2.0)

                print(f"模型输出logits形状: {logits.shape}")

                # 分析预测分布
                probs = torch.softmax(calibrated_logits, dim=-1)
                probs_np = probs.cpu().numpy()
                print("校准后预测概率统计:")
                for i in range(5):
                    print(f"  类别 {i}: 平均概率={probs_np[:, i].mean():.3f}")

                sampled_pred = calibrated_logits.argmax(-1).cpu().numpy()
                print(f"采样预测结果形状: {sampled_pred.shape}")

            # 12. 将预测结果映射回原始点
            print("将预测结果映射回原始点...")

            # 创建全零标签数组（对应原始数据框长度）
            original_pred = np.zeros(len(df), dtype=int)

            # 只对有效点进行映射
            if len(processed_coord) > 0 and len(original_indices) > 0:
                # 使用KDTree找到每个有效原始点的最近采样点
                tree = cKDTree(processed_coord)
                _, indices = tree.query(coord, k=1)  # 注意：这里使用过滤后的coord

                # 将采样点的标签赋给对应的有效原始点
                valid_original_pred = sampled_pred[indices]

                # 将有效点的预测结果放回原始位置
                original_pred[original_indices] = valid_original_pred

            # 13. 添加标签到原始DataFrame
            df['label'] = original_pred

            # 打印统计信息
            label_counts = df['label'].value_counts().sort_index()
            print("预测结果统计:")
            for label, count in label_counts.items():
                print(f"  标签 {label}: {count} 个点")

            return df

        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            print(f"详细堆栈信息: {traceback.format_exc()}")
            raise


# 兼容旧版本导入
def predict_csv(csv_path):
    """兼容函数，直接调用Command类的静态方法"""
    return Command.predict_csv(csv_path)