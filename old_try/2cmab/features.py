"""
features.py — 手工特征提取 (用于 LinUCB)

参考 NSS: 9nss论文/neural-solver-selection/dataset.py 的 manual_features()
和 model.py 中 scale 的单独拼接逻辑。

TSP: 9维几何 + scale = 10维
CVRP: 9维几何 + scale + demand_mean + demand_std = 12维
统一: one-hot(2) + 特征(12, TSP补零) = 14维
LinUCB交互特征: 14 + is_tsp×geo_scale(10) + is_cvrp×geo_scale(10) = 34维
"""
import numpy as np

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None


def manual_features_single(nodes: np.ndarray, scale: float, problem_type: str = "tsp") -> np.ndarray:
    """
    单个实例的手工特征，对齐 NSS dataset.py:manual_features()

    返回: [14] = [one_hot(2), geo(9), scale(1), demand_mean(1), demand_std(1)]
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if problem_type.lower() == "cvrp" and nodes.shape[-1] >= 3:
        demands = nodes[:, 2]
        coords = nodes[:, :2]
    else:
        demands = None
        coords = nodes[:, :2] if nodes.shape[-1] >= 2 else nodes

    N = coords.shape[0]

    # 1. dist_std
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    dist_std = float(np.std(dist_mat.ravel()))

    # 2-3. centroid
    centroid = coords.mean(axis=0)

    # 4. radius
    radius = float(np.linalg.norm(coords - centroid, axis=1).mean())

    # 5. count_distinct
    count_distinct = float(len(np.unique(np.round(dist_mat * 100).astype(int).ravel())))

    # 6. nn_dist_std
    nn_dist = (dist_mat + 1e3 * np.eye(N)).min(axis=-1)
    nn_dist_std = float(np.std(nn_dist))

    # 7-9. HDBSCAN 聚类特征
    cluster_ratio = outlier_ratio = cluster_radius = 0.0
    if HDBSCAN is not None:
        try:
            labels = HDBSCAN().fit(coords).labels_
            max_label = int(np.max(labels))
            if max_label >= 0:
                cluster_ratio = max_label / N
                outlier_ratio = float((labels == -1).sum()) / N
                radii = []
                for cl in range(max_label + 1):
                    pts = coords[labels == cl]
                    if len(pts) > 0:
                        radii.append(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())
                cluster_radius = float(np.mean(radii)) if radii else 0.0
        except Exception:
            pass

    # 组装 [14]
    feat = np.zeros(14, dtype=np.float64)
    feat[0] = 1.0 if problem_type.lower() == "tsp" else 0.0   # one-hot TSP
    feat[1] = 1.0 if problem_type.lower() == "cvrp" else 0.0  # one-hot CVRP
    feat[2] = dist_std
    feat[3] = centroid[0]
    feat[4] = centroid[1]
    feat[5] = radius
    feat[6] = count_distinct
    feat[7] = nn_dist_std
    feat[8] = cluster_ratio
    feat[9] = outlier_ratio
    feat[10] = cluster_radius
    feat[11] = float(scale)
    if demands is not None:
        feat[12] = float(np.mean(demands))
        feat[13] = float(np.std(demands))
    return feat


def build_interaction_features(base: np.ndarray) -> np.ndarray:
    """
    LinUCB 交互特征: base(14) + is_tsp*geo_scale(10) + is_cvrp*geo_scale(10) = 34

    让线性模型能学到 "同一几何特征在 TSP 和 CVRP 上的不同影响"。
    """
    is_tsp, is_cvrp = base[0], base[1]
    geo_scale = base[2:12]  # 9维几何 + 1维scale
    return np.concatenate([base, is_tsp * geo_scale, is_cvrp * geo_scale])  # 34


def extract_features_batch(instances: list, problem_type: str, scale: int = 100,
                           use_interaction: bool = True) -> np.ndarray:
    """批量提取特征。返回 [N, 34](交互) 或 [N, 14](基础)"""
    feats = []
    for inst in instances:
        b = manual_features_single(inst, scale, problem_type)
        feats.append(build_interaction_features(b) if use_interaction else b)
    return np.array(feats)
