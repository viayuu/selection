from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    from encoders import HybridContextEncoder  # type: ignore
else:
    from .encoders import HybridContextEncoder

logger = logging.getLogger("neural_ucb_diag")


class NeuralUCBDiag(nn.Module):
    """
    参考 Zhou 2020 官方 diag 代码的共享 action-conditioned 版本。

    这里采用的是“工程可跑版本”，不是原论文最昂贵的全协方差实现。

    具体来说：
    - Zhou 2020 原论文是基于网络梯度构造不确定性
    - 官方实用代码里常用 diagonal approximation
    - 我们这里继续沿用 diag 思路

    因此它的定位是：
    - 一个更贴近 NeuralUCB 思想的补充算法
    - 但不是对原论文最严格定义的逐字复现
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        use_arm_onehot: bool = True,
        use_arm_bias: bool = True,
        lamdba: float = 1.0,
        nu: float = 1.0,
        lr: float = 1e-2,
        train_every: int = 100,
        train_batch_size: int = 256,
        train_steps_per_update: int = 20,
        initial_pulls: int = 1,
        use_problem_arm_safety: bool = True,
        safety_min_pulls: int = 20,
        safety_reward_gap: float = 0.15,
        safety_penalty_scale: float = 3.0,
        safety_penalty_cap: float = 2.0,
        arm_names: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_arms = int(n_arms)
        self.nu = float(nu)
        self.lamdba = float(lamdba)
        self.lr = float(lr)
        self.arm_embed_dim = int(arm_embed_dim)
        self.use_arm_onehot = bool(use_arm_onehot)
        self.use_arm_bias = bool(use_arm_bias)
        self.train_every = max(1, int(train_every))
        self.train_batch_size = int(train_batch_size)
        self.train_steps_per_update = int(train_steps_per_update)
        self.initial_pulls = int(initial_pulls)
        self.use_problem_arm_safety = bool(use_problem_arm_safety)
        self.safety_min_pulls = max(1, int(safety_min_pulls))
        self.safety_reward_gap = float(safety_reward_gap)
        self.safety_penalty_scale = float(safety_penalty_scale)
        self.safety_penalty_cap = float(safety_penalty_cap)
        self.arm_names = list(arm_names) if arm_names is not None else [str(idx) for idx in range(self.n_arms)]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.context_encoder = HybridContextEncoder()
        self.arm_embedding = nn.Embedding(self.n_arms, self.arm_embed_dim)
        if self.use_arm_bias:
            # 显式的 arm 偏置项。
            # 当 reward_net 还没学出足够强的实例级差异时，
            # 这个偏置至少能帮助模型更稳地表达“不同方法的整体均值不同”。
            self.arm_bias = nn.Parameter(torch.zeros(self.n_arms, dtype=torch.float32))
        else:
            self.register_parameter("arm_bias", None)

        # 一个共享的 reward 网络，输入是 (sample, arm) 的联合表示。
        # 这里的 arm 条件化不再只靠一个稠密 embedding，
        # 还会额外拼上 one-hot：
        # - embedding: 便于学“方法之间的相似性”
        # - one-hot:   强制保留离散身份，避免网络把不同 arm 完全混在一起
        arm_feature_dim = self.arm_embed_dim + (self.n_arms if self.use_arm_onehot else 0)
        self.reward_net = nn.Sequential(
            nn.Linear(self.context_encoder.output_dim + arm_feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(self.device)

        self._refresh_trainable_state(reset_uncertainty=True)
        self.history: list[tuple[object, int, float]] = []
        self.arm_counts = np.zeros(self.n_arms, dtype=np.int64)
        self.arm_reward_sums = np.zeros(self.n_arms, dtype=np.float64)
        # 方案 A：
        # 不再只看“全局 arm 被拉了几次”，而是看“某个问题上的某个 arm 被拉了几次”。
        # 例如：
        # - ("tsp", 0) 和 ("cvrp", 0) 分开统计
        # 这样更符合当前 TSP/CVRP 共享框架的语义。
        self.problem_arm_counts: dict[tuple[str, int], int] = {}
        self.problem_arm_reward_sums: dict[tuple[str, int], float] = {}
        # 当前 replay 采用 problem-arm 平衡采样，避免历史数据被 lehd 一家独大淹没。
        self.replay_mode = "balanced_problem_arm"
        self.total_steps = 0

    def _refresh_trainable_state(self, reset_uncertainty: bool = True) -> None:
        """
        重新扫描当前可训练参数，并重建与之配套的优化器 / 不确定性状态。

        这样做主要是为了兼容 `freeze_graph_encoder()`：
        - 一旦某些参数被设成 `requires_grad=False`
        - 梯度向量长度就会变化
        - 与之对应的 `U` 也必须同步重建
        """
        self._trainable_params = [param for param in self.parameters() if param.requires_grad]
        self.total_param = sum(param.numel() for param in self._trainable_params)
        # 这里继续保持和官方 diag 代码一致的做法：SGD + weight decay。
        self.optimizer = torch.optim.SGD(self._trainable_params, lr=self.lr, weight_decay=self.lamdba)
        if reset_uncertainty:
            # U 是对角近似下的“参数不确定性统计量”
            self.U = torch.full((self.total_param,), self.lamdba, dtype=torch.float32, device=self.device)

    def _arm_features_from_ids(self, arm_tensor: torch.Tensor) -> torch.Tensor:
        """
        根据 arm id 构造显式的 arm 条件特征。

        返回：
        - embedding 部分: [B, arm_embed_dim]
        - 如果启用 one-hot，再拼 [B, n_arms]

        这样做的目的，是让 reward_net 在输入层就能看见“这是哪一个 arm”，
        从而更容易学出真正的 arm 区分能力。
        """
        arm_emb = self.arm_embedding(arm_tensor)
        if not self.use_arm_onehot:
            return arm_emb
        arm_onehot = torch.nn.functional.one_hot(arm_tensor, num_classes=self.n_arms).to(dtype=arm_emb.dtype)
        return torch.cat((arm_emb, arm_onehot), dim=1)

    def _encode_actions(self, sample, arm_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        把“一个实例 + 多个候选 arm”编码成 reward_net 的输入。

        旧实现里，每评估一个 arm 都会重新跑一遍图编码器，
        这会让同一个样本的共享上下文被重复计算很多次。

        这里改成：
        - 先把 sample 编成统一 context
        - 再与多个 arm embedding 拼接

        这样语义不变，但速度与稳定性都更好。
        """
        base = self.context_encoder.encode_samples([sample]).to(self.device)
        arm_tensor = torch.as_tensor(arm_ids, dtype=torch.long, device=self.device)
        arm_features = self._arm_features_from_ids(arm_tensor)
        base = base.expand(len(arm_ids), -1)
        return torch.cat((base, arm_features), dim=1), arm_tensor

    def _encode_record_batch(self, records: list[tuple[object, int, float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        把一批 history records 编码成 reward_net 输入。

        当前 NeuralUCB 的训练仍然是 reward regression，
        但这里支持 mini-batch，而不是像旧版那样每次只抽 1 条。
        """
        if not records:
            arm_feature_dim = self.arm_embedding.embedding_dim + (self.n_arms if self.use_arm_onehot else 0)
            input_dim = self.context_encoder.output_dim + arm_feature_dim
            empty_inputs = torch.zeros((0, input_dim), dtype=torch.float32, device=self.device)
            empty_arms = torch.zeros((0,), dtype=torch.long, device=self.device)
            return empty_inputs, empty_arms

        samples = [sample for sample, _arm, _reward in records]
        arms = [int(arm) for _sample, arm, _reward in records]
        base = self.context_encoder.encode_samples(samples).to(self.device)
        arm_tensor = torch.as_tensor(arms, dtype=torch.long, device=self.device)
        arm_features = self._arm_features_from_ids(arm_tensor)
        return torch.cat((base, arm_features), dim=1), arm_tensor

    def _forward_encoded(self, encoded_inputs: torch.Tensor, arm_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """对已经拼好的 (context, arm) 联合输入做 reward 前向。"""
        output = self.reward_net(encoded_inputs).squeeze(-1)
        if self.arm_bias is not None and arm_tensor is not None:
            output = output + self.arm_bias[arm_tensor]
        return output

    def freeze_graph_encoder(self) -> None:
        """
        冻结底层图编码器，并同步刷新 NeuralUCB 的参数统计状态。

        注意：
        - NeuralUCB-Diag 的 `U` 长度取决于“当前可训练参数总数”
        - 所以冻结后不能只改 requires_grad，而不刷新 `U`
        """
        self.context_encoder.freeze_graph_encoder()
        self._refresh_trainable_state(reset_uncertainty=True)

    def _forward_arm(self, sample, arm: int) -> torch.Tensor:
        """前向计算某个 (sample, arm) 的 reward 预测值。"""
        encoded, arm_tensor = self._encode_actions(sample, [int(arm)])
        return self._forward_encoded(encoded, arm_tensor=arm_tensor).squeeze(0)

    def _arm_name(self, arm: int) -> str:
        """把 arm id 转成更友好的名字，便于日志阅读。"""
        if 0 <= int(arm) < len(self.arm_names):
            return str(self.arm_names[int(arm)])
        return str(int(arm))

    def _sample_meta(self, sample) -> tuple[str, str, str]:
        """提取一个样本在日志里最常用的标识信息。"""
        problem = str(getattr(sample, "problem", "")).upper()
        uid = str(getattr(sample, "uid", "unknown"))
        global_index = str(getattr(sample, "global_index", "unknown"))
        return problem, uid, global_index

    def _problem_key(self, sample) -> str:
        """把 sample 的问题类型规范成内部统一的小写字符串。"""
        return str(getattr(sample, "problem", "")).strip().lower()

    def _problem_arm_key(self, sample, arm: int) -> tuple[str, int]:
        """一个问题上的一个 arm，作为 warmup / replay 的最小统计单元。"""
        return (self._problem_key(sample), int(arm))

    def _problem_arm_pull_count(self, sample, arm: int) -> int:
        """返回 (problem, arm) 这个组合当前已经被拉了多少次。"""
        return int(self.problem_arm_counts.get(self._problem_arm_key(sample, int(arm)), 0))

    def _problem_arm_reward_mean(self, sample, arm: int) -> float | None:
        """返回当前 (problem, arm) 的经验平均 reward。"""
        key = self._problem_arm_key(sample, int(arm))
        pulls = int(self.problem_arm_counts.get(key, 0))
        if pulls <= 0:
            return None
        reward_sum = float(self.problem_arm_reward_sums.get(key, 0.0))
        return reward_sum / float(pulls)

    def _problem_arm_safety_penalty(self, sample, arm: int, feasible_arms: list[int]) -> tuple[float, float | None, float | None]:
        """
        对“在当前问题上经验上明显偏差的 arm”施加一个软惩罚。

        返回：
        - safety_penalty
        - 当前 arm 在该问题上的经验平均 reward
        - 当前问题里可靠 arm 的最好经验平均 reward
        """
        if not self.use_problem_arm_safety:
            return 0.0, None, None

        pulls = self._problem_arm_pull_count(sample, int(arm))
        arm_mean_reward = self._problem_arm_reward_mean(sample, int(arm))
        if arm_mean_reward is None or pulls < self.safety_min_pulls:
            return 0.0, arm_mean_reward, None

        reliable_means = []
        for candidate_arm in feasible_arms:
            candidate_pulls = self._problem_arm_pull_count(sample, int(candidate_arm))
            candidate_mean = self._problem_arm_reward_mean(sample, int(candidate_arm))
            if candidate_mean is None or candidate_pulls < self.safety_min_pulls:
                continue
            reliable_means.append(float(candidate_mean))
        if len(reliable_means) < 2:
            return 0.0, arm_mean_reward, None

        best_mean_reward = float(max(reliable_means))
        gap = best_mean_reward - float(arm_mean_reward)
        if gap <= self.safety_reward_gap:
            return 0.0, arm_mean_reward, best_mean_reward

        penalty = self.safety_penalty_scale * (gap - self.safety_reward_gap)
        penalty = min(float(self.safety_penalty_cap), float(max(0.0, penalty)))
        return float(penalty), arm_mean_reward, best_mean_reward

    def _selected_result_text(self, sample, arm: int) -> str:
        """
        把选中 arm 对应的原始结果记录压成一行文本。

        这样两种神经方法的详细日志都能看到同样的 result 摘要。
        """
        result_records = getattr(sample, "result_records", None)
        if result_records is None or not (0 <= int(arm) < len(result_records)):
            return "result: unavailable"

        record = result_records[int(arm)]
        if record is None:
            return "result: unavailable"

        score_key = None
        score_value = None
        for candidate in ("aug_score", "score", "no_aug_score", "picked_score"):
            value = record.get(candidate)
            if value is not None:
                score_key = candidate
                score_value = float(value)
                break

        parts = [
            f"record_global_index={record.get('global_index')}",
            f"name={record.get('name')}",
        ]
        if score_key is not None and score_value is not None:
            parts.append(f"{score_key}={score_value:.4f}")
        return "result: " + " | ".join(parts)

    def _u_stats(self) -> tuple[float, float, float, float]:
        """返回当前对角不确定性统计量 U 的几个摘要指标。"""
        return (
            float(torch.linalg.norm(self.U).item()),
            float(torch.mean(self.U).item()),
            float(torch.min(self.U).item()),
            float(torch.max(self.U).item()),
        )

    def _format_update_log(self, sample, arm: int, reward: float) -> str:
        """把一次 bandit update 组织成和 Neural-LinUCB 同风格的详细日志。"""
        problem, uid, global_index = self._sample_meta(sample)
        feasible_mask = np.asarray(sample.feasible_mask) > 0
        selected_cost = float(sample.costs[int(arm)])
        best_cost = float(np.nanmin(sample.costs[feasible_mask]))
        regret = selected_cost - best_cost
        pulls = int(self.arm_counts[int(arm)])
        avg_reward = self.arm_reward_sums[int(arm)] / max(1, pulls)
        problem_arm_pulls = self._problem_arm_pull_count(sample, int(arm))
        problem_arm_avg_reward = self._problem_arm_reward_mean(sample, int(arm))
        u_norm, u_mean, u_min, u_max = self._u_stats()
        lines = [
            f"t={self.total_steps} | UPDATE | problem={problem} | uid={uid} | global_index={global_index} | arm={int(arm)}({self._arm_name(int(arm))})",
            (
                f"    outcome: reward={float(reward):.4f} | selected_cost={selected_cost:.4f} | "
                f"best_cost={best_cost:.4f} | regret={regret:.4f}"
            ),
            self._selected_result_text(sample, int(arm)).replace("result:", "    result:", 1),
            (
                f"    arm_stats: global_pulls={pulls} | "
                f"problem_arm_pulls={problem_arm_pulls} | avg_reward={float(avg_reward):.4f} | "
                f"problem_arm_avg_reward={'None' if problem_arm_avg_reward is None else f'{float(problem_arm_avg_reward):.4f}'}"
            ),
            (
                f"    state: history={len(self.history)} | U_norm={u_norm:.4f} | "
                f"U_mean={u_mean:.6f} | U_min={u_min:.6f} | U_max={u_max:.6f}"
            ),
        ]
        return "\n".join(lines)

    def _format_train_log(self, train_stats: dict) -> str:
        """把一次 reward_net 训练整理成统一格式。"""
        u_norm, u_mean, u_min, u_max = self._u_stats()
        lines = [
            f"t={self.total_steps} | TRAIN | module=reward_net | trigger={train_stats['trigger']}",
            (
                f"    optimize: epochs={int(train_stats['num_epochs'])} | steps={int(train_stats['num_steps'])} | "
                f"batch_size={train_stats['batch_size']} | replay={train_stats['replay_mode']} | "
                f"history={len(self.history)} | "
                f"lr={float(self.optimizer.param_groups[0]['lr']):.6g}"
            ),
            (
                f"    loss: avg={float(train_stats['avg_loss']):.6f} | min={float(train_stats['min_loss']):.6f} | "
                f"max={float(train_stats['max_loss']):.6f} | last={float(train_stats['last_loss']):.6f}"
            ),
            (
                f"    state: U_norm={u_norm:.4f} | U_mean={u_mean:.6f} | "
                f"U_min={u_min:.6f} | U_max={u_max:.6f}"
            ),
        ]
        return "\n".join(lines)

    def _flatten_grad_list(self, grad_list: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None]) -> torch.Tensor:
        """把 autograd.grad 返回的梯度列表拉平成一个向量。"""
        grads = []
        for param, grad in zip(self._trainable_params, grad_list):
            if grad is None:
                grads.append(torch.zeros_like(param).reshape(-1))
            else:
                grads.append(grad.reshape(-1))
        return torch.cat(grads, dim=0)

    def _train_trigger_reason(self) -> str:
        """
        返回当前这一步为什么触发训练。

        这里采用“前期更密、后期按间隔触发”的策略：
        - 早期 history 很少时，reward_net 需要更频繁地跟上 bandit 数据
        - 后期再切到 train_every 的周期训练，避免每一步都重训
        """
        if self.train_every <= 1:
            return "per_update"
        warmup_steps = max(self.train_every, self.n_arms * max(1, self.initial_pulls))
        if self.total_steps <= warmup_steps:
            return f"warmup(<= {warmup_steps})"
        return f"train_every({self.train_every})"

    def _should_train_now(self) -> bool:
        """判断当前 bandit update 之后是否应该训练 reward_net。"""
        if self.total_steps <= 0:
            return False
        if self.train_every <= 1:
            return True
        warmup_steps = max(self.train_every, self.n_arms * max(1, self.initial_pulls))
        return self.total_steps <= warmup_steps or (self.total_steps % self.train_every == 0)

    def _history_buckets(self, history_records: list[tuple[object, int, float]]) -> dict[tuple[str, int], list[tuple[object, int, float]]]:
        """
        按 (problem, arm) 对历史记录分桶。

        这是方案 A 的核心：
        - TSP 的 lehd 和 CVRP 的 lehd 分开
        - 稀有 arm 不会再因为 replay 完全被高频 lehd 淹没
        """
        buckets: dict[tuple[str, int], list[tuple[object, int, float]]] = {}
        for record in history_records:
            sample, arm, _reward = record
            key = self._problem_arm_key(sample, int(arm))
            buckets.setdefault(key, []).append(record)
        return buckets

    def _sample_balanced_batch(
        self,
        history_buckets: dict[tuple[str, int], list[tuple[object, int, float]]],
        batch_size: int,
    ) -> list[tuple[object, int, float]]:
        """
        从 (problem, arm) 桶里做平衡采样。

        采样策略：
        - 先在“非空桶”之间均匀选桶
        - 再在桶内均匀抽记录

        这样做以后：
        - 高频的 lehd 仍然可以出现
        - 但低频 arm 会被显式过采样
        - 更有机会学到“什么时候别的方法会赢过 lehd”
        """
        non_empty_keys = [key for key, bucket in history_buckets.items() if bucket]
        if not non_empty_keys:
            return []

        chosen_bucket_indices = np.random.randint(0, len(non_empty_keys), size=max(1, int(batch_size)))
        batch_records: list[tuple[object, int, float]] = []
        for bucket_idx in chosen_bucket_indices:
            bucket = history_buckets[non_empty_keys[int(bucket_idx)]]
            record_idx = int(np.random.randint(0, len(bucket)))
            batch_records.append(bucket[record_idx])
        return batch_records

    def _decision_details_with_best_grad(self, sample, use_ucb: bool = True) -> tuple[dict, torch.Tensor | None]:
        """
        计算一次决策的完整细节。

        返回：
        - decision details
        - 如果 use_ucb=True 且最终按 score 选择，则返回被选 arm 的梯度向量

        这样 select() 可以在不重复前向的前提下更新 U，
        同时 trace 记录也能拿到完整的每 arm 分数。
        """
        feasible_arms = [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)]
        if not feasible_arms:
            raise ValueError("没有可选 arm")

        under_sampled = []
        if self.initial_pulls > 0:
            # 方案 A：
            # warmup 改成按 (problem, arm) 统计，而不是全局 arm 统计。
            # 否则如果 TSP 上先把某个 arm 拉满，CVRP 上这个 arm 可能就再也得不到冷启动机会。
            under_sampled = [arm for arm in feasible_arms if self._problem_arm_pull_count(sample, arm) < self.initial_pulls]

        detail_map: dict[int, dict] = {}
        grad_map: dict[int, torch.Tensor] = {}
        best_arm = feasible_arms[0]
        best_score = -float("inf")
        best_grad = None

        encoded, arm_tensor = self._encode_actions(sample, feasible_arms)
        outputs = self._forward_encoded(encoded, arm_tensor=arm_tensor)

        for idx, arm in enumerate(feasible_arms):
            fx = outputs[idx]
            if use_ucb:
                grad_list = torch.autograd.grad(
                    fx,
                    self._trainable_params,
                    retain_graph=(idx + 1 < len(feasible_arms)),
                    allow_unused=True,
                )
                g = self._flatten_grad_list(grad_list).detach()
                grad_map[int(arm)] = g
                sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g * g / self.U.clamp_min(1e-12)))
                mean = float(fx.item())
                bonus = float(sigma.item())
                raw_score = mean + bonus
            else:
                g = None
                mean = float(fx.detach().item())
                bonus = None
                raw_score = mean

            safety_penalty, problem_arm_mean_reward, best_problem_arm_mean_reward = self._problem_arm_safety_penalty(
                sample,
                int(arm),
                feasible_arms,
            )
            score = float(raw_score - safety_penalty)

            detail_map[int(arm)] = {
                "arm_id": int(arm),
                "feasible": True,
                "pull_count": self._problem_arm_pull_count(sample, arm),
                "mean": mean,
                "bonus": bonus,
                "raw_score": float(raw_score),
                "safety_penalty": float(safety_penalty),
                "problem_arm_mean_reward": problem_arm_mean_reward,
                "best_problem_arm_mean_reward": best_problem_arm_mean_reward,
                "score": score,
            }

            if score > best_score:
                best_score = score
                best_arm = int(arm)
                best_grad = g

        arm_details = []
        for arm in range(self.n_arms):
            if arm in detail_map:
                arm_details.append(detail_map[arm])
            else:
                arm_details.append(
                    {
                        "arm_id": int(arm),
                        "feasible": False,
                        "pull_count": self._problem_arm_pull_count(sample, arm),
                        "mean": None,
                        "bonus": None,
                        "score": None,
                    }
                )

        if under_sampled:
            # warmup 阶段优先补齐当前 problem 上还没拉满的 arm。
            # 排序优先级：
            # 1. 当前 (problem, arm) 被拉的次数更少
            # 2. 全局该 arm 被拉的次数更少
            # 3. arm id 更小（仅用于打破并列）
            selected_arm = int(
                sorted(
                    under_sampled,
                    key=lambda arm: (
                        self._problem_arm_pull_count(sample, arm),
                        int(self.arm_counts[arm]),
                        int(arm),
                    ),
                )[0]
            )
            selection_reason = "warm_start"
            selected_grad = grad_map.get(selected_arm)
        else:
            selected_arm = int(best_arm)
            selection_reason = "score"
            selected_grad = best_grad

        return (
            {
                "selected_arm": selected_arm,
                "selection_reason": selection_reason,
                "use_ucb": bool(use_ucb),
                "feasible_arms": [int(arm) for arm in feasible_arms],
                "under_sampled_arms": [int(arm) for arm in under_sampled],
                "arm_details": arm_details,
            },
            selected_grad,
        )

    def decision_details(self, sample, use_ucb: bool = True) -> dict:
        """无副作用版本，供 trace 记录使用。"""
        details, _ = self._decision_details_with_best_grad(sample, use_ucb=use_ucb)
        return details

    def select_with_details(self, sample, use_ucb: bool = True) -> dict:
        """
        有副作用版本，供训练主循环使用。

        和 `decision_details()` 的区别是：
        - 会在真正完成一次“选择”之后更新对角不确定性统计量 `U`
        - 这样才能符合 NeuralUCB / 官方 diag 代码里的语义：
          选中的 arm 一旦被拉取，就应该把对应梯度并入不确定性统计量
        """
        details, selected_grad = self._decision_details_with_best_grad(sample, use_ucb=use_ucb)
        if use_ucb and selected_grad is not None:
            self.U = self.U + selected_grad * selected_grad
        return details

    def select(self, sample, use_ucb: bool = True) -> int:
        """
        对所有可行动作计算：
        fx + sigma

        其中 sigma 使用对角近似：
        sigma^2 ≈ sum(lambda * nu * g^2 / U)

        这和官方 diag 代码是一致的。
        """
        details = self.select_with_details(sample, use_ucb=use_ucb)
        return int(details["selected_arm"])

    def update(self, sample, arm: int, reward: float) -> None:
        """把新样本加入缓存，并在需要时继续训练 reward 网络。"""
        self.history.append((sample, int(arm), float(reward)))
        self.arm_counts[int(arm)] += 1
        self.arm_reward_sums[int(arm)] += float(reward)
        problem_arm_key = self._problem_arm_key(sample, int(arm))
        self.problem_arm_counts[problem_arm_key] = int(self.problem_arm_counts.get(problem_arm_key, 0)) + 1
        self.problem_arm_reward_sums[problem_arm_key] = float(self.problem_arm_reward_sums.get(problem_arm_key, 0.0)) + float(reward)
        self.total_steps += 1
        train_stats = None
        if self._should_train_now():
            train_stats = self.train_on_history(trigger=self._train_trigger_reason())
        logger.info(self._format_update_log(sample, int(arm), float(reward)))
        if train_stats is not None:
            logger.info(self._format_train_log(train_stats))

    def train_on_history(self, trigger: str = "manual") -> dict | None:
        """
        使用历史 bandit 观测训练 reward_net。

        这部分现在改成了更接近论文 / 官方 diag 代码的训练方式：
        - 训练目标仍然是 reward regression
        - 但不再是旧版“随机抽 1 条样本做 SGD”
        - 而是对 history 做打乱后 mini-batch 优化

        这样做的理由：
        1. 更接近“用历史 bandit 数据更新神经网络”的原始思路
        2. 单样本 SGD 容易高方差、容易让网络只追着最近样本抖动
        3. mini-batch 也更容易把 GPU 利用起来
        """
        if not self.history:
            return None

        self.train()
        losses: list[float] = []
        num_epochs = 0
        max_steps = max(1, int(self.train_steps_per_update))
        history_records = list(self.history)
        batch_size = len(history_records) if self.train_batch_size <= 0 else min(int(self.train_batch_size), len(history_records))
        history_buckets = self._history_buckets(history_records)
        # 用“按样本量估计的一轮步数”做一个近似 epoch 概念，便于日志理解。
        steps_per_epoch = max(1, int(np.ceil(len(history_records) / max(1, batch_size))))

        while len(losses) < max_steps:
            num_epochs += 1
            epoch_total_loss = 0.0
            epoch_total_count = 0

            for _ in range(steps_per_epoch):
                batch_records = self._sample_balanced_batch(history_buckets, batch_size)
                if not batch_records:
                    break
                encoded, arm_tensor = self._encode_record_batch(batch_records)
                pred = self._forward_encoded(encoded, arm_tensor=arm_tensor)
                target = torch.as_tensor(
                    [float(reward) for _sample, _arm, reward in batch_records],
                    dtype=torch.float32,
                    device=self.device,
                )
                loss = torch.mean((pred - target) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value = float(loss.item())
                losses.append(loss_value)
                batch_len = len(batch_records)
                epoch_total_loss += loss_value * batch_len
                epoch_total_count += batch_len

                if len(losses) >= max_steps:
                    break

            # 当前 epoch 的平均 loss 已经很低时就提前停，避免无意义多跑。
            if epoch_total_count > 0 and epoch_total_loss / epoch_total_count <= 1e-4:
                break

        self.eval()
        if not losses:
            return None
        return {
            "trigger": str(trigger),
            "num_epochs": num_epochs,
            "num_steps": len(losses),
            "batch_size": int(batch_size),
            "replay_mode": str(self.replay_mode),
            "avg_loss": float(np.mean(losses)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "last_loss": float(losses[-1]),
        }
