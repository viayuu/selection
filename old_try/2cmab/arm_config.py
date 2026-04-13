"""
arm_config.py — 统一方法注册表 + action mask

TSP 和 CVRP 共用一个 bandit，共有方法共享同一个臂。
TSP: 12 个方法全部可选
CVRP: 前 8 个可选，后 4 个 TSP-only 方法不可选
"""
import numpy as np

# 统一臂空间 = TSP方法 ∪ CVRP方法（去重），共有方法在前
ALL_METHODS = [
    "lehd", "elg", "invit", "icam",     # 0-3: TSP + CVRP 共有
    "dact", "lih", "udc", "omni",       # 4-7: TSP + CVRP 共有
    "pointerformer",                      # 8:   TSP only
    "difusco", "t2t", "glop",           # 9-11: TSP only
]
N_ARMS = len(ALL_METHODS)  # 12
METHOD_TO_IDX = {m: i for i, m in enumerate(ALL_METHODS)}

# 可用 mask: 1=可选, 0=不可选
TSP_MASK  = np.array([1,1,1,1, 1,1,1,1, 1, 1,1,1], dtype=np.float32)
CVRP_MASK = np.array([1,1,1,1, 1,1,1,1, 0, 0,0,0], dtype=np.float32)

def get_mask(problem_type: str) -> np.ndarray:
    """根据问题类型返回 action mask"""
    return TSP_MASK.copy() if problem_type.lower() == "tsp" else CVRP_MASK.copy()
