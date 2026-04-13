from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomSelector:
    """随机 baseline：在可行动作里均匀随机选一个。"""

    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select(self, sample) -> int:
        # 只在当前实例可选的方法里采样
        feasible = np.flatnonzero(sample.feasible_mask > 0)
        return int(self.rng.choice(feasible))


@dataclass
class SingleBestGlobalSelector:
    """
    全局 single-best：
    在训练集上统计一个“整体上平均最强”的固定方法。

    这个 baseline 很重要，因为它回答的是：
    “如果完全不用实例特征，只固定选一个方法，效果有多好？”
    """

    arm: int

    @classmethod
    def fit(cls, samples, train_indices, arm_names):
        arm_scores = np.zeros(len(arm_names), dtype=np.float64)
        arm_counts = np.zeros(len(arm_names), dtype=np.int64)
        for idx in train_indices:
            sample = samples[idx]
            feasible = sample.feasible_mask > 0
            # 用 reward 而不是原始 cost 做平均，便于跨实例比较
            arm_scores[feasible] += sample.rewards[feasible]
            arm_counts[feasible] += 1

        mean_scores = np.divide(
            arm_scores,
            np.maximum(arm_counts, 1),
            out=np.full_like(arm_scores, -np.inf, dtype=np.float64),
            where=arm_counts > 0,
        )
        return cls(arm=int(np.argmax(mean_scores)))

    def select(self, sample) -> int:
        if sample.feasible_mask[self.arm] > 0:
            return self.arm
        feasible = np.flatnonzero(sample.feasible_mask > 0)
        return int(feasible[0])


@dataclass
class SingleBestPerProblemSelector:
    """
    按问题分别统计 single-best。

    例如：
    - TSP 上固定选一个方法
    - CVRP 上固定选另一个方法

    它比 global single-best 更公平，因为两个问题上的最佳固定方法很可能不同。
    """

    best_by_problem: dict[str, int]

    @classmethod
    def fit(cls, samples, train_indices, arm_names):
        best_by_problem: dict[str, int] = {}
        for problem in sorted({samples[idx].problem for idx in train_indices}):
            arm_scores = np.zeros(len(arm_names), dtype=np.float64)
            arm_counts = np.zeros(len(arm_names), dtype=np.int64)
            for idx in train_indices:
                sample = samples[idx]
                if sample.problem != problem:
                    continue
                feasible = sample.feasible_mask > 0
                arm_scores[feasible] += sample.rewards[feasible]
                arm_counts[feasible] += 1
            mean_scores = np.divide(
                arm_scores,
                np.maximum(arm_counts, 1),
                out=np.full_like(arm_scores, -np.inf, dtype=np.float64),
                where=arm_counts > 0,
            )
            best_by_problem[problem] = int(np.argmax(mean_scores))
        return cls(best_by_problem=best_by_problem)

    def select(self, sample) -> int:
        arm = self.best_by_problem[sample.problem]
        if sample.feasible_mask[arm] > 0:
            return arm
        feasible = np.flatnonzero(sample.feasible_mask > 0)
        return int(feasible[0])


@dataclass
class OracleSelector:
    """oracle 上界：每个实例都直接选事后最优方法。"""

    def select(self, sample) -> int:
        return int(sample.best_arm)
