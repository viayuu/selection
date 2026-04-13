from __future__ import annotations

import numpy as np


class LinUCB:
    """
    经典 disjoint LinUCB。
    每个 arm 一套 A / b，支持 action mask。

    这是最标准、最容易解释的 contextual bandit 基线之一。

    当前实现中：
    - 上下文 x 是 14 维
    - 每个 arm 都有独立的线性参数
    - 选择规则是 mean + alpha * uncertainty
    - uncertainty 由 x^T A^{-1} x 给出
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        reg: float = 1.0,
        initial_pulls: int = 1,
    ) -> None:
        self.n_arms = int(n_arms)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.initial_pulls = int(initial_pulls)

        # 这里直接维护 A^{-1}，而不是每次重新求逆。
        # 这样更新时可以用 Sherman-Morrison 公式做增量更新，速度更快。
        self.A_inv = [np.eye(self.context_dim, dtype=np.float64) / reg for _ in range(self.n_arms)]
        self.b = [np.zeros(self.context_dim, dtype=np.float64) for _ in range(self.n_arms)]
        self.arm_counts = np.zeros(self.n_arms, dtype=np.int64)

    def _theta(self, arm: int) -> np.ndarray:
        """当前 arm 的线性参数估计：theta = A^{-1} b"""
        return self.A_inv[arm] @ self.b[arm]

    def predict_means(self, context: np.ndarray) -> np.ndarray:
        """只计算每个 arm 的均值项，不加探索 bonus。"""
        x = np.asarray(context, dtype=np.float64)
        return np.asarray([self._theta(arm) @ x for arm in range(self.n_arms)], dtype=np.float64)

    def score_ucb(self, context: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        计算所有 arm 的 UCB 分数。

        UCB_k = theta_k^T x + alpha * sqrt(x^T A_k^{-1} x)

        其中：
        - 第一项是利用项（当前估计均值）
        - 第二项是探索项（当前不确定性）
        """
        x = np.asarray(context, dtype=np.float64)
        scores = np.full(self.n_arms, -np.inf, dtype=np.float64)
        if mask is None:
            feasible = np.ones(self.n_arms, dtype=bool)
        else:
            feasible = np.asarray(mask, dtype=bool)

        for arm in np.flatnonzero(feasible):
            theta = self._theta(int(arm))
            mean = float(theta @ x)
            bonus = float(np.sqrt(x @ self.A_inv[int(arm)] @ x))
            scores[int(arm)] = mean + self.alpha * bonus
        return scores

    def decision_details(self, context: np.ndarray, mask: np.ndarray | None = None, use_ucb: bool = True) -> dict:
        """
        返回一次选臂决策的详细信息，便于落盘分析。

        这个接口是“无副作用”的：
        - 不会更新 A / b
        - 不会修改 arm_counts

        主要用于：
        - 训练/评测 trace 记录
        - 之后复盘“为什么选了这个 arm”
        """
        x = np.asarray(context, dtype=np.float64)
        if mask is None:
            feasible = np.ones(self.n_arms, dtype=bool)
        else:
            feasible = np.asarray(mask, dtype=bool)
        feasible_arms = np.flatnonzero(feasible)
        if len(feasible_arms) == 0:
            raise ValueError("没有可选 arm")

        under_sampled = []
        if self.initial_pulls > 0:
            under_sampled = [int(arm) for arm in feasible_arms if self.arm_counts[int(arm)] < self.initial_pulls]

        arm_details = []
        for arm in range(self.n_arms):
            if not feasible[arm]:
                arm_details.append(
                    {
                        "arm_id": int(arm),
                        "feasible": False,
                        "pull_count": int(self.arm_counts[arm]),
                        "mean": None,
                        "bonus": None,
                        "score": None,
                    }
                )
                continue

            theta = self._theta(int(arm))
            mean = float(theta @ x)
            bonus = float(np.sqrt(x @ self.A_inv[int(arm)] @ x))
            score = mean + self.alpha * bonus if use_ucb else mean
            arm_details.append(
                {
                    "arm_id": int(arm),
                    "feasible": True,
                    "pull_count": int(self.arm_counts[arm]),
                    "mean": mean,
                    "bonus": bonus,
                    "score": score,
                }
            )

        if under_sampled:
            selected_arm = int(sorted(under_sampled, key=lambda arm: (self.arm_counts[int(arm)], int(arm)))[0])
            selection_reason = "warm_start"
        else:
            selected_arm = int(max((detail for detail in arm_details if detail["feasible"]), key=lambda item: float(item["score"]))["arm_id"])
            selection_reason = "score"

        return {
            "selected_arm": selected_arm,
            "selection_reason": selection_reason,
            "use_ucb": bool(use_ucb),
            "feasible_arms": [int(arm) for arm in feasible_arms.tolist()],
            "under_sampled_arms": [int(arm) for arm in under_sampled],
            "arm_details": arm_details,
        }

    def select(self, context: np.ndarray, mask: np.ndarray | None = None, use_ucb: bool = True) -> int:
        """
        选择一个动作。

        额外包含一个 very practical 的 warm-start 规则：
        - 如果某些 arm 还没被拉够 initial_pulls 次，就优先把它们先试一遍

        这样做是为了避免一开始完全靠随机噪声决定后续走势。
        """
        return int(self.decision_details(context, mask=mask, use_ucb=use_ucb)["selected_arm"])

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        用一次新的 (x, a, r) 观测更新指定 arm。

        对 LinUCB 来说，更新公式就是：
        - A <- A + x x^T
        - b <- b + r x

        但这里为了效率不显式维护 A，而是维护 A^{-1}。
        """
        x = np.asarray(context, dtype=np.float64)
        arm = int(arm)

        a_inv = self.A_inv[arm]
        ax = a_inv @ x
        denom = 1.0 + float(x @ ax)
        self.A_inv[arm] = a_inv - np.outer(ax, ax) / denom
        self.b[arm] = self.b[arm] + float(reward) * x
        self.arm_counts[arm] += 1
