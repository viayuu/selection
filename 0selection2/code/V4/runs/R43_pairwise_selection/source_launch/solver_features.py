"""User-supplied solver metadata: 0 means unmarked, not confirmed absent.

Field order is fixed. Each checkpoint stores the complete specification in
model_params and the ID-aligned matrix as a persistent, non-trainable buffer.
"""

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS


SOLVER_FEATURE_DIM = 34

FEATURE_ORDER = (
    "构造式", "扩散式", "坐标与节点属性", "成本矩阵", "图与边",
    "MHA／Transformer", "GNN", "矩阵双流注意力", "静态编码复用",
    "动态状态更新", "动态局部更新", "迭代去噪", "实例条件", "属性条件",
    "约束条件", "元学习任务条件", "FFN 专家", "LoRA 专家", "监督学习",
    "强化学习", "扩散训练", "单任务", "多任务", "元学习", "子问题重编码",
    "重解码器", "梯度引导搜索", "全局—局部双策略", "近邻成本输入",
    "伪 one-hot 输入", "统一矩阵表示", "全局分区与局部重构",
    "解码器 FFN 残差", "全局属性嵌入",
)

SOLVER_ACTIVE_INDICES = {
    "BQ": [0, 2, 5, 9, 18, 21, 24],
    "LEHD": [0, 2, 5, 9, 18, 21, 25],
    "DIFUSCO": [1, 4, 6, 11, 20, 21],
    "DIFUSCO500": [1, 4, 6, 11, 20, 21],
    "T2T": [1, 4, 6, 11, 20, 21, 26],
    "T2T500": [1, 4, 6, 11, 20, 21, 26],
    "ELG": [0, 2, 5, 8, 10, 19, 21, 27],
    "OMNI": [0, 2, 5, 8, 15, 19, 23],
    "ICAM": [0, 2, 5, 8, 12, 19, 21],
    "ICAM_ATSP": [0, 3, 7, 8, 12, 19, 21, 28],
    "MATNET": [0, 3, 7, 8, 19, 21],
    "MATPOENET": [0, 3, 7, 8, 19, 21, 29],
    "UNICO_MatPOENet": [0, 3, 7, 8, 19, 22, 29, 30],
    "GLOP": [0, 3, 4, 6, 7, 19, 31],
    "MTPOMO": [0, 2, 5, 8, 13, 19, 22],
    "MVMOE": [0, 2, 5, 8, 13, 16, 19, 22],
    "RELD_CVRP": [0, 2, 5, 8, 19, 21, 32],
    "RELD_MTL": [0, 2, 5, 8, 13, 19, 22, 32],
    "RELD_MOEL": [0, 2, 5, 8, 13, 16, 19, 22, 32],
    "RouteFinder": [0, 2, 5, 8, 13, 19, 22, 33],
    "MoSES_RF": [0, 2, 5, 8, 13, 17, 22, 33],
    "MoSES_CaDA": [0, 2, 5, 8, 14, 17, 22],
}


def get_solver_feature_spec():
    return {
        "feature_dim": SOLVER_FEATURE_DIM,
        "index_base": 0,
        "feature_order": list(FEATURE_ORDER),
        "solver_active_indices": {name: indices[:] for name, indices in SOLVER_ACTIVE_INDICES.items()},
        "solver_names": list(GLOBAL_SOLVERS),
    }


def build_solver_features(spec):
    if spec["feature_dim"] != SOLVER_FEATURE_DIM or spec["index_base"] != 0:
        raise ValueError("Solver features require 34 dimensions with zero-based indices")
    fields = spec["feature_order"]
    if (
        len(fields) != SOLVER_FEATURE_DIM
        or any(not isinstance(name, str) or not name for name in fields)
        or len(set(fields)) != SOLVER_FEATURE_DIM
    ):
        raise ValueError("Solver features require 34 unique, ordered field names")
    if spec["solver_names"] != list(GLOBAL_SOLVERS):
        raise ValueError("Solver feature row order differs from the current global solver IDs")

    active = spec["solver_active_indices"]
    missing = set(GLOBAL_SOLVERS) - set(active)
    if missing:
        raise ValueError(f"Missing solver feature rows: {sorted(missing)}")
    features = torch.zeros(len(GLOBAL_SOLVERS), SOLVER_FEATURE_DIM)
    for row, name in enumerate(GLOBAL_SOLVERS):
        indices = active[name]
        if (
            not isinstance(indices, list)
            or any(type(i) is not int or not 0 <= i < SOLVER_FEATURE_DIM for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"{name}: feature indices must be unique integers in [0, 33]")
        features[row, indices] = 1.0
    return features
