from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# 统一动作空间定义
# ============================================================
# 这里采用“共享 arm 空间”的思路：
# - 同名方法在 TSP 和 CVRP 中共用同一个 arm id
# - 例如 lehd 在两个问题上都映射到同一个 arm
#
# 这样做的原因是：
# 1. 我们希望 bandit 学到的“方法偏好”可以在问题间共享一部分经验
# 2. 具体某个实例最终能不能选某个方法，不由 arm id 决定，而由 feasible mask 决定
# 3. 共享的是“方法身份”，不是“所有问题都能选所有方法”
#
# 因此，真正的决策逻辑是：
# - 先进入统一 arm 空间
# - 再按问题类型用 mask 过滤不可用方法
#
# 例如：
# - TSP 可以选 12 个方法
# - CVRP 只能选前 8 个方法
# ============================================================

# 统一动作空间：共有方法共享一个 arm id
ALL_METHODS = [
    "lehd",
    "elg",
    "invit",
    "icam",
    "dact",
    "lih",
    "udc",
    "omni",
    "pointerformer",
    "difusco",
    "t2t",
    "glop",
]

TSP_METHODS = [
    "lehd",
    "elg",
    "invit",
    "icam",
    "dact",
    "lih",
    "udc",
    "omni",
    "pointerformer",
    "difusco",
    "t2t",
    "glop",
]

CVRP_METHODS = [
    "lehd",
    "elg",
    "invit",
    "icam",
    "dact",
    "lih",
    "udc",
    "omni",
]

PROBLEM_TO_METHODS = {
    "tsp": TSP_METHODS,
    "cvrp": CVRP_METHODS,
}

# 方法名到统一 arm id 的映射
METHOD_TO_IDX = {name: idx for idx, name in enumerate(ALL_METHODS)}


@dataclass(frozen=True)
class ArmSpace:
    """
    ArmSpace 用来管理“统一动作空间 + 问题可行动作约束”。

    它回答两个问题：
    1. 全局一共有多少个候选方法（arm）
    2. 对于某个具体问题，哪些 arm 是可选的
    """

    arm_names: list[str]
    problem_to_methods: dict[str, list[str]]

    @property
    def n_arms(self) -> int:
        """统一动作空间大小。"""
        return len(self.arm_names)

    @property
    def method_to_idx(self) -> dict[str, int]:
        """根据当前 arm_names 动态生成方法名到索引的映射。"""
        return {name: idx for idx, name in enumerate(self.arm_names)}

    def get_mask(self, problem: str) -> np.ndarray:
        """
        根据问题类型返回可行动作 mask。

        返回的是一个长度等于 n_arms 的向量：
        - 1.0 表示这个 arm 对该问题可用
        - 0.0 表示这个 arm 对该问题不可用
        """
        problem = problem.lower()
        if problem not in self.problem_to_methods:
            raise ValueError(f"未知问题类型: {problem}")
        allowed = set(self.problem_to_methods[problem])
        return np.asarray([1.0 if name in allowed else 0.0 for name in self.arm_names], dtype=np.float32)


DEFAULT_ARM_SPACE = ArmSpace(arm_names=ALL_METHODS, problem_to_methods=PROBLEM_TO_METHODS)


def get_mask(problem: str, arm_names: list[str] | None = None, problem_to_methods: dict[str, list[str]] | None = None) -> np.ndarray:
    """
    便捷接口：
    - 如果不传自定义空间，就使用默认 arm 空间
    - 如果传了 arm_names/problem_to_methods，就临时构造一个 ArmSpace
    """
    if arm_names is None and problem_to_methods is None:
        return DEFAULT_ARM_SPACE.get_mask(problem)
    arm_space = ArmSpace(
        arm_names=arm_names or ALL_METHODS,
        problem_to_methods=problem_to_methods or PROBLEM_TO_METHODS,
    )
    return arm_space.get_mask(problem)


def get_problem_onehot(problem: str) -> np.ndarray:
    """
    把问题类型编码成 one-hot。

    当前只支持两类问题：
    - TSP  -> [1, 0]
    - CVRP -> [0, 1]

    后续如果要扩展到更多问题，可以在这里继续往下加。
    """
    problem = problem.lower()
    if problem == "tsp":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if problem == "cvrp":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    raise ValueError(f"未知问题类型: {problem}")
