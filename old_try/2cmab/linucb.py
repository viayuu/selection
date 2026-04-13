"""
linucb.py — LinUCB (Disjoint 版) [Li et al., 2010]

使用 34 维交互特征 (14维基础 + 20维问题类型×几何交互)。
每个臂维护独立的 A_k (d×d) 和 b_k (d,)。

UCB_k = θ_k^T · x + α · √(x^T · A_k^{-1} · x)
其中 θ_k = A_k^{-1} · b_k
"""
import numpy as np


class LinUCB:
    """Disjoint LinUCB, 支持 action mask"""

    def __init__(self, n_arms: int, context_dim: int = 34, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        # 每个臂的精度矩阵 A_k 和奖励累积向量 b_k
        self.A = [np.eye(self.d) for _ in range(n_arms)]         # A_0 = I
        self.b = [np.zeros(self.d) for _ in range(n_arms)]       # b_0 = 0
        self._A_inv = [np.eye(self.d) for _ in range(n_arms)]    # 缓存逆矩阵
        self._dirty = [False] * n_arms  # 标记是否需要重新计算逆

    def select(self, context: np.ndarray, mask: np.ndarray) -> int:
        """
        选择 UCB 值最大的可用臂。
        context: [d], mask: [n_arms] (1=可用)
        """
        x = context.flatten()
        best_ucb, best_arm = -np.inf, 0

        for k in range(self.n_arms):
            if mask[k] < 0.5:
                continue
            if self._dirty[k]:
                self._A_inv[k] = np.linalg.inv(self.A[k])
                self._dirty[k] = False
            A_inv = self._A_inv[k]
            theta = A_inv @ self.b[k]                         # θ_k = A_k^{-1} b_k
            mu = float(theta @ x)                              # 预测奖励
            bonus = self.alpha * np.sqrt(float(x @ A_inv @ x)) # 不确定性
            ucb = mu + bonus
            if ucb > best_ucb:
                best_ucb, best_arm = ucb, k
        return best_arm

    def update(self, arm: int, context: np.ndarray, reward: float):
        """更新臂的统计量: A += x x^T, b += r x"""
        x = context.flatten()
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self._dirty[arm] = True

    def get_theta(self) -> np.ndarray:
        """返回 [n_arms, d] 参数矩阵 (可解释性)"""
        thetas = []
        for k in range(self.n_arms):
            if self._dirty[k]:
                self._A_inv[k] = np.linalg.inv(self.A[k])
                self._dirty[k] = False
            thetas.append(self._A_inv[k] @ self.b[k])
        return np.array(thetas)
