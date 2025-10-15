import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_pointcloud(coords, feats, labels, beamaz=None, target_class=2, k=3, generate_ratio=0.3):
    """
    核心：给少数类生成“逼真新样本”（非复制，避免过拟合）
    coords: 点云坐标（你的数据格式）
    feats: 点云特征（你的数据格式）
    labels: 样本标签
    target_class: 要增强的类别（仅传2、3、4）
    k: 找5个近邻样本（无需改动）
    generate_ratio: 新增样本比例（0.3=新增原样本数的30%，无需改动）
    """
    mask = labels == target_class
    n_samples = mask.sum()
    if n_samples < k:
        return coords, feats, labels, beamaz

    coords_cls = coords[mask]
    feats_cls = feats[mask]
    beamaz_cls = beamaz[mask] if beamaz is not None else None
    n_generate = max(1, int(len(coords_cls) * generate_ratio))

    nn = NearestNeighbors(n_neighbors=min(k, n_samples)).fit(coords_cls)
    synth_coords, synth_feats, synth_labels, synth_beamaz = [], [], [], []

    for _ in range(n_generate):
        idx = np.random.randint(0, len(coords_cls))
        neighbor_idx = np.random.choice(nn.kneighbors([coords_cls[idx]], return_distance=False)[0][1:])
        alpha = np.random.rand()
        new_coord = coords_cls[idx] * alpha + coords_cls[neighbor_idx] * (1 - alpha)
        new_feat = feats_cls[idx] * alpha + feats_cls[neighbor_idx] * (1 - alpha)
        new_label = target_class
        synth_coords.append(new_coord)
        synth_feats.append(new_feat)
        synth_labels.append(new_label)
        if beamaz_cls is not None:
            new_beamaz = beamaz_cls[idx] * alpha + beamaz_cls[neighbor_idx] * (1 - alpha)
            synth_beamaz.append(new_beamaz)

    coords = np.vstack([coords, np.array(synth_coords)])
    feats = np.vstack([feats, np.array(synth_feats)])
    labels = np.hstack([labels, np.array(synth_labels)])
    if beamaz is not None:
        beamaz = np.hstack([beamaz, np.array(synth_beamaz)])

    return coords, feats, labels, beamaz
