from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


try:
    from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - 环境依赖不稳定
    try:
        from hdbscan import HDBSCAN  # type: ignore
    except Exception:  # pragma: no cover - 环境依赖不稳定
        HDBSCAN = None


@dataclass(frozen=True)
class ManualFeatureBundle:
    """
    手工特征打包结构。

    这里刻意保留两套版本：
    - manual_raw: 不含 scale 的原始手工特征
    - manual_with_scale: 在 manual_raw 基础上再拼一个 scale

    这样做是因为：
    - NSS 的 manual_features() 本身不直接把 scale 放进去
    - 但 NSS 的分类器阶段会再把 scale 拼进去
    - 我们这里既想保留“原始手工特征”的语义，也想保留“可直接喂给模型”的版本
    """

    manual_raw: np.ndarray
    manual_with_scale: np.ndarray


def _to_numpy(nodes: torch.Tensor | np.ndarray) -> np.ndarray:
    """统一把输入转成 numpy，方便后面做统计计算。"""
    if isinstance(nodes, torch.Tensor):
        return nodes.detach().cpu().numpy()
    return np.asarray(nodes)


def _cluster_features(coords: np.ndarray) -> tuple[float, float, float]:
    """
    计算和聚类结构相关的 3 个特征。

    输出分别是：
    - cluster_ratio
    - outlier_ratio
    - cluster_radius

    这部分直接借鉴 NSS / evolved-instance 难度分析那条思路。
    """
    # 小图或缺依赖时，聚类特征直接退化成 0
    if coords.shape[0] < 4 or HDBSCAN is None:
        return 0.0, 0.0, 0.0

    try:
        labels = HDBSCAN().fit(coords).labels_
    except Exception:
        return 0.0, 0.0, 0.0

    max_label = int(labels.max())
    if max_label < 0:
        return 0.0, float((labels == -1).mean()), 0.0

    cluster_ratio = max_label / float(coords.shape[0])
    outlier_ratio = float((labels == -1).mean())

    radii = []
    for cluster_id in range(max_label + 1):
        cluster_nodes = coords[labels == cluster_id]
        if len(cluster_nodes) == 0:
            continue
        centroid = cluster_nodes.mean(axis=0, keepdims=True)
        radii.append(float(np.linalg.norm(cluster_nodes - centroid, axis=1).mean()))
    cluster_radius = float(np.mean(radii)) if radii else 0.0
    return cluster_ratio, outlier_ratio, cluster_radius


def _extract_geometry_features(coords: np.ndarray) -> np.ndarray:
    """
    提取 9 维几何特征。

    这 9 维基本对应 NSS 里的 manual_features：
    1. 所有点对距离的标准差
    2. 质心 x
    3. 质心 y
    4. 平均半径
    5. 距离离散值统计
    6. 最近邻距离标准差
    7. 聚类比例
    8. 离群点比例
    9. 簇半径
    """
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    std_all = float(dist_mat.reshape(-1).std())
    centroid = coords.mean(axis=0)
    radius = float(np.linalg.norm(coords - centroid[None, :], axis=1).mean())
    rounded = np.round(dist_mat * 100).astype(np.int64).reshape(-1)
    bincount = np.bincount(rounded) if rounded.size > 0 else np.zeros(1, dtype=np.int64)
    count_distinct = float((bincount == 0).sum())

    masked = dist_mat + np.eye(coords.shape[0]) * 1e6
    nearest = masked.min(axis=1)
    std_nn = float(nearest.std())

    cluster_ratio, outlier_ratio, cluster_radius = _cluster_features(coords)

    geo = np.zeros(9, dtype=np.float32)
    geo[0] = std_all
    geo[1:3] = centroid.astype(np.float32)
    geo[3] = radius
    geo[4] = count_distinct
    geo[5] = std_nn
    geo[6] = cluster_ratio
    geo[7] = outlier_ratio
    geo[8] = cluster_radius
    return geo


def extract_manual_feature_bundle(problem: str, nodes: torch.Tensor | np.ndarray) -> ManualFeatureBundle:
    """
    复用 NSS 的手工特征思路：
    - TSP: 9 维几何特征
    - CVRP: 9 维几何特征 + 2 维需求统计
    - 另外单独拼一个 scale，得到统一 12 维

    特别注意：
    - TSP 的原始手工特征只有 9 维
    - CVRP 的原始手工特征是 11 维
    - 为了统一输入维度，这里把 TSP 的 demand 位置补 0

    最终统一成：
    - manual_raw: 11 维
    - manual_with_scale: 12 维

    再加问题 one-hot(2) 后，就得到 LinUCB 的 14 维上下文。
    """
    problem = problem.lower()
    nodes_np = _to_numpy(nodes).astype(np.float32)

    if problem == "tsp":
        # TSP 节点只有坐标，没有 demand
        coords = nodes_np[:, :2]
        geo = _extract_geometry_features(coords)
        raw = np.zeros(11, dtype=np.float32)
        raw[:9] = geo
        scale = float(nodes_np.shape[0])
    elif problem == "cvrp":
        # CVRP 的每个节点有 [x, y, demand]
        # 这里默认 nodes 已经包含 depot，且 depot 的 demand 为 0
        coords = nodes_np[:, :2]
        demands = nodes_np[:, 2]
        geo = _extract_geometry_features(coords)
        raw = np.zeros(11, dtype=np.float32)
        raw[:9] = geo
        raw[9] = float(demands.mean())
        raw[10] = float(demands.std())
        scale = float(nodes_np.shape[0])
    else:
        raise ValueError(f"未知问题类型: {problem}")

    with_scale = np.zeros(12, dtype=np.float32)
    with_scale[:11] = raw
    # scale 直接取节点数。
    # 对 TSP 是 n，对 CVRP 因为包含 depot，这里是 1+n。
    # 当前实现优先保持“与当前输入张量一致”，便于排查问题。
    with_scale[11] = scale
    return ManualFeatureBundle(manual_raw=raw, manual_with_scale=with_scale)
