from __future__ import annotations

import math

import numpy as np


def _average_ranks(costs: np.ndarray) -> np.ndarray:
    """
    对成本做升序排名，ties 使用平均名次。
    例如 [1, 1, 3] -> [1.5, 1.5, 3.0]

    这里“成本越小越好”，所以是升序排名：
    - 最小 cost 的 rank 最小
    - 最大 cost 的 rank 最大

    为什么要单独写平均名次？
    因为实例中不同方法可能出现完全一样的分数。
    如果 ties 不处理好，reward 会不稳定，也不利于后续分析。
    """
    order = np.argsort(costs, kind="mergesort")
    sorted_costs = costs[order]
    ranks_sorted = np.zeros(len(costs), dtype=np.float64)

    start = 0
    while start < len(costs):
        end = start + 1
        # 找到当前这段相同 cost 的结束位置
        while end < len(costs) and math.isclose(sorted_costs[end], sorted_costs[start], rel_tol=1e-12, abs_tol=1e-12):
            end += 1
        # 平均名次公式：
        # 如果第 2、3 名并列，那么两者 rank 都记为 2.5
        avg_rank = (start + 1 + end) / 2.0
        ranks_sorted[start:end] = avg_rank
        start = end

    ranks = np.empty(len(costs), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def compute_rank_based_rewards(
    costs: np.ndarray,
    mask: np.ndarray | None = None,
    mode: str = "linear_zero_one",
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据实例内方法排名把 cost 转成 reward。

    参数
    - mode="linear_zero_one": 最好=1, 最差=0
    - mode="autosaea": 参考 AutoSAEA 公式风格, 最差=1/K

    为什么这里默认不用原始 cost 直接当 reward？
    因为不同方法、不同问题、不同实例上的 cost 量级可能差异很大。
    直接用原始 cost 会让学习目标不稳定。

    排名 reward 的优点是：
    1. 有界
    2. 对量纲不敏感
    3. 更接近“方法选择”的本质：我们关心的是谁更好，而不是绝对值差多少
    """
    costs = np.asarray(costs, dtype=np.float64)
    if mask is None:
        # 没有 mask 时，只要 cost 是有效数值，就认为该 arm 可用
        valid = np.isfinite(costs)
    else:
        # 有 mask 时，必须同时满足：
        # 1. 这个方法对该问题可用
        # 2. 这个方法当前确实有结果
        valid = np.asarray(mask, dtype=bool) & np.isfinite(costs)

    rewards = np.zeros_like(costs, dtype=np.float64)
    ranks = np.full_like(costs, np.nan, dtype=np.float64)

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return rewards, ranks

    valid_costs = costs[valid]
    valid_ranks = _average_ranks(valid_costs)
    ranks[valid] = valid_ranks
    k = len(valid_costs)

    if mode == "linear_zero_one":
        # 我们当前主线默认采用这一种。
        #
        # 公式：
        # r = 1 - (rank - 1) / (K - 1)
        #
        # 性质：
        # - 最好 rank=1 -> reward=1
        # - 最差 rank=K -> reward=0
        # - 中间线性插值
        if k == 1:
            # 如果当前实例只有一个可行动作，那它天然就是最优
            rewards[valid] = 1.0
        else:
            rewards[valid] = 1.0 - (valid_ranks - 1.0) / float(k - 1)
    elif mode == "autosaea":
        # 参考 AutoSAEA 公式(24) 风格：
        # r = (K + 1 - rank) / K
        #
        # 与上面的区别：
        # - 最好仍然是 1
        # - 最差不是 0，而是 1/K
        rewards[valid] = (k + 1.0 - valid_ranks) / float(k)
    else:
        raise ValueError(f"未知 reward mode: {mode}")

    return rewards, ranks
