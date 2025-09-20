import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pointcept.datasets import build_dataset, build_dataloader
from pointcept.datasets.transforms import build_transform
from torch.utils.data.sampler import RandomSampler, SequentialSampler

def load_config(config_path):
    """加载YAML配置文件"""
    assert os.path.exists(config_path), f"配置文件不存在：{config_path}"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 关键：Linux环境下修正data_root路径（原配置是Windows路径D:/...）
    for split in ["train", "val", "test"]:
        if "data_root" in config["data"][split]:
            # 示例：将D:/model/wind_datas/ 改为 Linux路径（需根据你的实际路径调整！）
            config["data"][split]["data_root"] = config["data"][split]["data_root"].replace(
                "D:/model/wind_datas/csv_labels",
                "/home/wai/PointTransformerV3/data/wind_datas/csv_labels"  # 你的Linux数据路径
            )
    return config


def verify_dataset_config(dataset, split, data_cfg):
    print(f"\n📋 验证[{split}]数据集配置")
    print("-" * 50)

    assert dataset.split == split, f"split配置错误：预期{split}，实际{dataset.split}"
    assert dataset.data_root == data_cfg["data_root"], f"data_root配置错误"
    assert dataset.k_neighbors == data_cfg["k_neighbors"], f"k_neighbors配置错误"
    print("✅ 基础参数（split/data_root/k_neighbors）配置正确")

    # 验证transform（含beamaz归一化）
    transform_class_names = [t.__class__.__name__ for t in dataset.transform.transforms]
    config_transform_types = [t["type"] for t in data_cfg["transform"]]
    assert transform_class_names == config_transform_types, \
        f"transform顺序或名称错误：预期{config_transform_types}，实际{transform_class_names}"
    print(f"✅ Transform配置正确：{transform_class_names}")

    # 验证NormalizeWind（含beamaz参数）
    normalize_tf = next(t for t in dataset.transform.transforms if t.__class__.__name__ == "NormalizeWind")
    normalize_cfg = next(t for t in data_cfg["transform"] if t["type"] == "NormalizeWind")
    assert normalize_tf.u_mean == normalize_cfg["u_mean"], f"u_mean配置错误"
    assert normalize_tf.u_std == normalize_cfg["u_std"], f"u_std配置错误"
    assert normalize_tf.v_mean == normalize_cfg["v_mean"], f"v_mean配置错误"
    assert normalize_tf.v_std == normalize_cfg["v_std"], f"v_std配置错误"
    assert normalize_tf.beamaz_mean == normalize_cfg["beamaz_mean"], f"beamaz_mean配置错误"  # 新增
    assert normalize_tf.beamaz_std == normalize_cfg["beamaz_std"], f"beamaz_std配置错误"      # 新增
    print("✅ 风速+beamaz归一化参数配置正确")

    # 验证GridSample
    grid_tf = next(t for t in dataset.transform.transforms if t.__class__.__name__ in ["GridSample", "WindShearGridSample"])
    grid_cfg = next(t for t in data_cfg["transform"] if t["type"] in ["GridSample", "WindShearGridSample"])
    assert grid_tf.grid_size == grid_cfg["grid_size"], f"grid_size配置错误"
    print("✅ 网格采样参数配置正确")


def verify_data_compatibility(data_dict, model_cfg, csv_path):
    print("\n📐 验证数据与模型兼容性")
    print("-" * 50)

    # 特征维度从6变为9（3原始+3均值+3标准差）
    feat = data_dict["feat"]
    expected_in_channels = model_cfg["in_channels"]  # 需在配置中设为9
    feat_np = feat.numpy() if torch.is_tensor(feat) else feat
    assert feat_np.shape[1] == expected_in_channels, \
        f"特征维度与模型不匹配：模型期望{expected_in_channels}维，实际{feat_np.shape[1]}维"
    print(f"✅ 特征维度正确：{feat_np.shape[1]}维（与模型in_channels一致）")

    # 标签验证
    label = data_dict["label"]
    expected_num_classes = model_cfg["num_classes"]
    label_np = label.numpy() if torch.is_tensor(label) else label
    assert np.all(np.isin(label_np, range(expected_num_classes))), \
        f"标签类别与模型不匹配：模型期望{expected_num_classes}类，实际标签包含{np.unique(label_np)}"

    # 新增：打印全1标签的样本路径
    label_np = label.numpy() if torch.is_tensor(label) else label
    if np.all(label_np == 1):
        print(f"⚠️ 警告：当前样本全为风切变点！路径：{csv_path}")

    shear_ratio = np.sum(label_np == 1) / len(label_np)
    print(f"✅ 标签类别正确：{expected_num_classes}类，风切变点占比={shear_ratio:.3f}")


def verify_dataloader_config(dataloader, train_cfg, split):
    print(f"\n🔄 验证[{split}]DataLoader配置")
    print("-" * 50)

    assert dataloader.batch_size == train_cfg["batch_size"], f"batch_size配置错误"
    assert dataloader.num_workers == train_cfg["num_workers"], f"num_workers配置错误"

    if split == "train":
        assert isinstance(dataloader.sampler, RandomSampler), "训练集未开启shuffle！"
    else:
        assert isinstance(dataloader.sampler, SequentialSampler), f"{split}集不应开启shuffle！"
    print("✅ DataLoader参数（batch_size/num_workers/shuffle）配置正确")


def visualize_config_impact(raw_data, processed_data, save_path="config_impact.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(raw_data["coord"][:, 0], raw_data["coord"][:, 1],
                    c=raw_data["label"], cmap="coolwarm", s=10, alpha=0.6)
    axes[0].set_title(f"原始数据（无Transform）\n点数：{len(raw_data['coord'])}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].scatter(processed_data["coord"][:, 0], processed_data["coord"][:, 1],
                    c=processed_data["label"], cmap="coolwarm", s=10, alpha=0.6)
    axes[1].set_title("Processed Data (GridSample Applied)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 配置效果可视化已保存：{save_path}（左：原始数据，右：处理后数据）")


if __name__ == "__main__":
    CONFIG_PATH = "configs/wind_shear/pointtransformer_v3.yaml"
    config = load_config(CONFIG_PATH)
    print(f"✅ 成功加载配置文件：{CONFIG_PATH}")
    print("=" * 70)

    model_cfg = config["model"]
    print(f"🎯 模型配置：in_channels={model_cfg['in_channels']}, num_classes={model_cfg['num_classes']}")

    for split in ["train", "val", "test"]:
        print("\n" + "=" * 70)
        print(f"===== 开始验证[{split}]集配置 =====")

        data_cfg = config["data"][split]
        transform = build_transform({
            "type": "Compose",
            "transforms": data_cfg["transform"]
        })
        dataset_cfg = data_cfg.copy()
        dataset_cfg["transform"] = transform
        dataset = build_dataset(dataset_cfg)

        assert len(dataset) > 0, f"{split}集无数据！请检查路径和日期范围"
        print(f"✅ {split}集数据量：{len(dataset)}个CSV样本")

        verify_dataset_config(dataset, split, data_cfg)

        data_dict = dataset[0]
        required_keys = ["coord", "feat", "label", "path", "beamaz"]  # 新增beamaz检查
        for key in required_keys:
            assert key in data_dict, f"数据字典缺少关键字段：{key}"
        print("✅ 数据字典字段完整（含新增beamaz）")

        verify_data_compatibility(data_dict, model_cfg, data_dict["path"])

        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=config["train"]["batch_size"],
            num_workers=config["train"]["num_workers"],
            shuffle=(split == "train")
        )
        verify_dataloader_config(dataloader, config["train"], split)

        batch_data = next(iter(dataloader))
        assert batch_data["coord"].shape[0] == batch_data["feat"].shape[0] == batch_data["label"].shape[0], \
            "批量数据拼接错误：coord/feat/label点数不匹配"
        print("✅ 批量数据处理正确")

    print("\n" + "=" * 70)
    print("===== 可视化配置效果 =====")
    raw_data_cfg = config["data"]["train"].copy()
    raw_data_cfg["transform"] = None
    raw_dataset = build_dataset(raw_data_cfg)
    visualize_config_impact(raw_dataset[2], dataset[2])

    print("\n" + "=" * 70)
    print("🎉 所有验证步骤完成！数据加载模块正常，可进入模型训练阶段")
    print("=" * 70)