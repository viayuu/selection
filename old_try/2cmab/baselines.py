"""
baselines.py — 四个 baseline 选择器

RandomSelector: 均匀随机选可用方法
SingleBestGlobalSelector: 训练集上全局最优的单一方法
SingleBestPerProblemSelector: TSP/CVRP 分别的最优方法
OracleSelector: 每个实例选事后最优 (上界)
"""
import numpy as np
from reward import compute_all_rewards


class RandomSelector:
    """均匀随机选可用方法"""
    def select(self, context, mask, **kwargs):
        feasible = np.where(mask > 0)[0]
        return int(np.random.choice(feasible)) if len(feasible) > 0 else 0

    def update(self, *args, **kwargs):
        pass


class SingleBestGlobalSelector:
    """始终选训练集上 mean rank reward 最高的单一方法 (全局)"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.best_arm = 0

    def fit(self, all_costs, all_masks):
        """从训练集统计每个臂的平均排名奖励, 选最好的"""
        arm_rewards = np.zeros(self.n_arms)
        arm_counts = np.zeros(self.n_arms)
        for i in range(len(all_costs)):
            rewards = compute_all_rewards(all_costs[i], all_masks[i])
            for k in range(self.n_arms):
                if all_masks[i][k] > 0 and not np.isnan(all_costs[i][k]):
                    arm_rewards[k] += rewards[k]
                    arm_counts[k] += 1
        mean_rewards = np.divide(arm_rewards, arm_counts, where=arm_counts > 0, out=np.zeros(self.n_arms))
        self.best_arm = int(np.argmax(mean_rewards))
        self.mean_rewards = mean_rewards

    def select(self, context, mask, **kwargs):
        if mask[self.best_arm] > 0:
            return self.best_arm
        # 如果最优臂不可用, 选可用中最好的
        feasible = np.where(mask > 0)[0]
        return int(feasible[np.argmax(self.mean_rewards[feasible])])

    def update(self, *args, **kwargs):
        pass


class SingleBestPerProblemSelector:
    """TSP 和 CVRP 分别选各自训练集上最优的方法"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.best_arm_tsp = 0
        self.best_arm_cvrp = 0
        self.mean_rewards_tsp = np.zeros(n_arms)
        self.mean_rewards_cvrp = np.zeros(n_arms)

    def fit(self, all_costs, all_masks, problem_types):
        """分问题统计"""
        for ptype, attr_r, attr_b in [("tsp", "mean_rewards_tsp", "best_arm_tsp"),
                                       ("cvrp", "mean_rewards_cvrp", "best_arm_cvrp")]:
            arm_rewards = np.zeros(self.n_arms)
            arm_counts = np.zeros(self.n_arms)
            for i in range(len(all_costs)):
                if problem_types[i] != ptype:
                    continue
                rewards = compute_all_rewards(all_costs[i], all_masks[i])
                for k in range(self.n_arms):
                    if all_masks[i][k] > 0 and not np.isnan(all_costs[i][k]):
                        arm_rewards[k] += rewards[k]
                        arm_counts[k] += 1
            mr = np.divide(arm_rewards, arm_counts, where=arm_counts > 0, out=np.zeros(self.n_arms))
            setattr(self, attr_r, mr)
            setattr(self, attr_b, int(np.argmax(mr)))

    def select(self, context, mask, problem_type="tsp", **kwargs):
        best = self.best_arm_tsp if problem_type == "tsp" else self.best_arm_cvrp
        mr = self.mean_rewards_tsp if problem_type == "tsp" else self.mean_rewards_cvrp
        if mask[best] > 0:
            return best
        feasible = np.where(mask > 0)[0]
        return int(feasible[np.argmax(mr[feasible])])

    def update(self, *args, **kwargs):
        pass


class OracleSelector:
    """每个实例选事后 cost 最小的方法 (性能上界)"""
    def select_with_costs(self, costs, mask):
        feasible = np.where(mask > 0)[0]
        fc = costs[feasible]
        valid = ~np.isnan(fc)
        if valid.sum() == 0:
            return int(feasible[0])
        best_local = np.argmin(fc[valid])
        return int(feasible[np.where(valid)[0][best_local]])
