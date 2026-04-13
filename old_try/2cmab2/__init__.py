"""
2cmab2

TSP + CVRP 初始化方法选择的共享 contextual bandit 实现。
"""

from .arm_config import ALL_METHODS, CVRP_METHODS, TSP_METHODS

__all__ = [
    "ALL_METHODS",
    "TSP_METHODS",
    "CVRP_METHODS",
]
