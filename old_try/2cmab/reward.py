"""
reward.py — 排名奖励 (AutoSAEA 论文公式24 风格)

r = (K + 1 - rank) / K
K = 可用方法数, rank = 1 表示最好 (cost 最低)
性质: 有界 [1/K, 1], 非稀疏, 不受 cost 量级影响
"""
import numpy as np
from scipy.stats import rankdata


def rank_reward(arm_cost: float, all_costs: np.ndarray, mask: np.ndarray) -> float:
    """计算单个臂的排名奖励"""
    feasible = np.where(mask > 0)[0]
    costs = all_costs[feasible]
    valid = ~np.isnan(costs)
    costs = costs[valid]
    K = len(costs)
    if K == 0:
        return 0.0
    ranks = rankdata(costs, method="average")
    idx = np.argmin(np.abs(costs - arm_cost))
    return float((K + 1 - ranks[idx]) / K)


def compute_all_rewards(costs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """批量计算所有可用方法的排名奖励，不可用方法为 0"""
    rewards = np.zeros(len(costs))
    feasible = np.where(mask > 0)[0]
    fc = costs[feasible]
    valid = ~np.isnan(fc)
    if valid.sum() == 0:
        return rewards
    fc_valid = fc[valid]
    idx_valid = feasible[valid]
    K = len(fc_valid)
    ranks = rankdata(fc_valid, method="average")
    for i, arm_idx in enumerate(idx_valid):
        rewards[arm_idx] = (K + 1 - ranks[i]) / K
    return rewards
