from __future__ import annotations

import logging
import math
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    from encoders import HybridContextEncoder  # type: ignore
else:
    from .encoders import HybridContextEncoder

logger = logging.getLogger("neural_linucb")


class NeuralLinUCB(nn.Module):
    """
    共享 action-conditioned Neural-LinUCB。
    这里把 arm 信息拼到输入里，用共享表示层做 phi(x, a)，
    再在最后一层做 UCB 探索。

    这不是 Xu 2022 原封不动的监督数据集版本，
    而是结合当前“组合优化初始化方法选择”任务做的适配版。

    核心保留了 Xu 2022 的两层思想：
    1. deep representation
    2. shallow exploration

    具体到这里：
    - deep representation: context_encoder + arm_embedding + phi_net
    - shallow exploration: 只在最后的线性空间上维护 A / b / theta

    也就是说，我们学的是 phi(x, a)，
    UCB 发生在最后的线性层，而不是整个网络参数空间。

    当前改进版默认还做了一件事：
    - TSP 和 CVRP 不再共享同一套线性头
    - 而是共享 encoder / phi_net，但按 problem 分开维护线性 UCB 统计量
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        alpha: float = 1.0,
        reg: float = 1.0,
        lr: float = 1e-3,
        train_every: int = 20,
        representation_steps: int = 10,
        representation_buffer_size: int = 1000,
        representation_batch_size: int = 32,
        representation_balance_mode: str = "none",
        representation_balance_power: float = 0.0,
        representation_loss_reweight: bool = False,
        linear_head_buffer_size: int = -1,
        initial_pulls: int = 1,
        train_epsilon: float = 0.0,
        problem_specific_heads: bool = True,
        arm_names: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_arms = int(n_arms)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.reg = float(reg)
        self.train_every = int(train_every)
        self.representation_steps = int(representation_steps)
        self.representation_buffer_size = int(representation_buffer_size)
        self.representation_batch_size = int(representation_batch_size)
        self.representation_balance_mode = str(representation_balance_mode).lower()
        self.representation_balance_power = float(representation_balance_power)
        self.representation_loss_reweight = bool(representation_loss_reweight)
        # 线性头缓存默认跟随表示层缓存，只有显式传参时才分开。
        self.linear_head_buffer_size = (
            int(representation_buffer_size) if int(linear_head_buffer_size) < 0 else int(linear_head_buffer_size)
        )
        self.initial_pulls = int(initial_pulls)
        self.train_epsilon = float(train_epsilon)
        self.problem_specific_heads = bool(problem_specific_heads)
        self.arm_names = list(arm_names) if arm_names is not None else [str(idx) for idx in range(self.n_arms)]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.representation_balance_mode not in {"none", "arm", "problem_arm"}:
            raise ValueError(
                "representation_balance_mode 必须是 none / arm / problem_arm 之一，"
                f"当前收到: {self.representation_balance_mode}"
            )

        self.context_encoder = HybridContextEncoder()
        self.arm_embedding = nn.Embedding(self.n_arms, arm_embed_dim)
        # phi_net 输出论文意义上的表示 phi(x, a)
        self.phi_net = nn.Sequential(
            nn.Linear(self.context_encoder.output_dim + arm_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.hidden_dim),
        )
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 这里维护的是“最后一层线性 UCB”的状态。
        #
        # 改进点：
        # - 旧版本是一个全局线性头
        # - 现在默认按 problem 拆头：TSP 一套，CVRP 一套
        #
        # 这样可以避免：
        # - 共享方法在两个问题上的统计量被强行平均
        # - TSP 学到的线性偏好直接污染 CVRP，反之亦然
        self.problem_heads: dict[str, dict[str, np.ndarray]] = {}
        self.problem_arm_counts: dict[str, np.ndarray] = {}
        self.problem_arm_reward_sums: dict[str, np.ndarray] = {}
        self.arm_counts = np.zeros(self.n_arms, dtype=np.int64)
        self.arm_reward_sums = np.zeros(self.n_arms, dtype=np.float64)
        # 这里显式拆成两个缓存：
        # 1. representation_history: 给表示层训练使用，通常取最近一个较小窗口
        # 2. linear_head_history: 给线性头 rebuild 使用，可以比表示层窗口更大
        #
        # 这么做的原因是：
        # - 表示层训练最贵，应该偏“recent buffer”
        # - 线性头 rebuild 相对便宜，可以吃到更多历史信息
        #
        # 如果某个 buffer_size <= 0，则该缓存退化为“保留全部历史”。
        rep_maxlen = None if self.representation_buffer_size <= 0 else self.representation_buffer_size
        linear_maxlen = None if self.linear_head_buffer_size <= 0 else self.linear_head_buffer_size
        self.representation_history = deque(maxlen=rep_maxlen)
        self.linear_head_history = deque(maxlen=linear_maxlen)
        self.total_steps = 0
        for head_key in self._initial_head_keys():
            self._ensure_head_state(head_key)

    def _arm_name(self, arm: int) -> str:
        """把 arm id 转成更友好的字符串，方便文本日志直接阅读。"""
        if 0 <= int(arm) < len(self.arm_names):
            return str(self.arm_names[int(arm)])
        return str(int(arm))

    def _initial_head_keys(self) -> list[str]:
        """
        初始化时默认创建哪些线性头。

        当前任务固定是 TSP + CVRP。
        - 如果开启 problem-specific heads，就先建这两套头
        - 否则只建一个共享头 ALL
        """
        if self.problem_specific_heads:
            return ["TSP", "CVRP"]
        return ["ALL"]

    def _new_head_state(self) -> dict[str, np.ndarray]:
        """创建一套新的线性 UCB 统计量。"""
        return {
            "A_inv": np.eye(self.hidden_dim, dtype=np.float64) / self.reg,
            "b": np.zeros(self.hidden_dim, dtype=np.float64),
            "theta": np.zeros(self.hidden_dim, dtype=np.float64),
        }

    def _normalize_problem_key(self, problem: str) -> str:
        """把 problem 名字规整成内部统一键。"""
        normalized = str(problem).upper().strip()
        if not normalized:
            normalized = "ALL"
        if not self.problem_specific_heads:
            return "ALL"
        return normalized

    def _head_key_from_sample(self, sample) -> str:
        """给一个样本确定它应该走哪一套线性头。"""
        problem, _uid, _global_index = self._sample_meta(sample)
        return self._normalize_problem_key(problem)

    def _ensure_head_state(self, head_key: str) -> None:
        """
        确保某个 problem 对应的线性头 / 统计数组已经存在。

        这里同时会准备：
        - A_inv / b / theta
        - problem 局部的 arm pull 计数
        - problem 局部的 arm reward 累加
        """
        key = self._normalize_problem_key(head_key)
        if key not in self.problem_heads:
            self.problem_heads[key] = self._new_head_state()
        if key not in self.problem_arm_counts:
            self.problem_arm_counts[key] = np.zeros(self.n_arms, dtype=np.int64)
        if key not in self.problem_arm_reward_sums:
            self.problem_arm_reward_sums[key] = np.zeros(self.n_arms, dtype=np.float64)

    def _head_state(self, head_key: str) -> dict[str, np.ndarray]:
        """取出某个 problem 的线性头状态，不存在就自动创建。"""
        normalized = self._normalize_problem_key(head_key)
        self._ensure_head_state(normalized)
        return self.problem_heads[normalized]

    def _problem_arm_count_array(self, head_key: str) -> np.ndarray:
        """取出某个 problem 的局部 arm 计数数组。"""
        normalized = self._normalize_problem_key(head_key)
        self._ensure_head_state(normalized)
        return self.problem_arm_counts[normalized]

    def _problem_arm_reward_sum_array(self, head_key: str) -> np.ndarray:
        """取出某个 problem 的局部 arm reward 累加数组。"""
        normalized = self._normalize_problem_key(head_key)
        self._ensure_head_state(normalized)
        return self.problem_arm_reward_sums[normalized]

    def export_problem_head_state(self) -> dict:
        """
        导出所有 problem 线性头的状态。

        这是给 checkpoint 保存用的。
        """
        return {
            "problem_specific_heads": bool(self.problem_specific_heads),
            "problem_heads": {
                key: {
                    "A_inv": state["A_inv"].copy(),
                    "b": state["b"].copy(),
                    "theta": state["theta"].copy(),
                }
                for key, state in self.problem_heads.items()
            },
            "problem_arm_counts": {key: value.copy() for key, value in self.problem_arm_counts.items()},
            "problem_arm_reward_sums": {key: value.copy() for key, value in self.problem_arm_reward_sums.items()},
        }

    def load_problem_head_state(self, payload: dict | None) -> None:
        """
        从 checkpoint 恢复所有 problem 线性头。

        如果拿到的是老版本单头 checkpoint，
        会把那一套头复制到当前所有默认 head 上，尽量兼容旧格式。
        """
        self.problem_heads = {}
        self.problem_arm_counts = {}
        self.problem_arm_reward_sums = {}

        if payload is None or "problem_heads" not in payload:
            for key in self._initial_head_keys():
                self._ensure_head_state(key)
            return

        saved_problem_specific_heads = bool(payload.get("problem_specific_heads", self.problem_specific_heads))
        self.problem_specific_heads = bool(saved_problem_specific_heads)
        for key, state in payload.get("problem_heads", {}).items():
            normalized = self._normalize_problem_key(key)
            self.problem_heads[normalized] = {
                "A_inv": np.asarray(state["A_inv"], dtype=np.float64),
                "b": np.asarray(state["b"], dtype=np.float64),
                "theta": np.asarray(state["theta"], dtype=np.float64),
            }
        for key, counts in payload.get("problem_arm_counts", {}).items():
            normalized = self._normalize_problem_key(key)
            self.problem_arm_counts[normalized] = np.asarray(counts, dtype=np.int64)
        for key, reward_sums in payload.get("problem_arm_reward_sums", {}).items():
            normalized = self._normalize_problem_key(key)
            self.problem_arm_reward_sums[normalized] = np.asarray(reward_sums, dtype=np.float64)
        for key in self._initial_head_keys():
            self._ensure_head_state(key)

    def _window_text(self, buffer_size: int) -> str:
        """把窗口大小转成更适合日志阅读的文本。"""
        return "ALL" if int(buffer_size) <= 0 else str(int(buffer_size))

    def _sample_meta(self, sample) -> tuple[str, str, str]:
        """提取一个样本最常用的标识信息。"""
        problem = str(getattr(sample, "problem", "")).upper()
        uid = str(getattr(sample, "uid", "unknown"))
        global_index = str(getattr(sample, "global_index", "unknown"))
        return problem, uid, global_index

    def _selected_result_text(self, sample, arm: int) -> str:
        """
        把当前选中 arm 对应的原始结果记录压成一行文本。

        这样在详细日志里可以直接看到：
        - 这个 arm 原始结果记录的 global_index
        - 使用的是哪一个 score 字段
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

    def _format_update_log(self, sample, arm: int, reward: float) -> str:
        """把一次 bandit update 组织成统一的多行详细日志。"""
        problem, uid, global_index = self._sample_meta(sample)
        head_key = self._head_key_from_sample(sample)
        head_state = self._head_state(head_key)
        problem_arm_counts = self._problem_arm_count_array(head_key)
        problem_arm_reward_sums = self._problem_arm_reward_sum_array(head_key)
        feasible_mask = np.asarray(sample.feasible_mask) > 0
        selected_cost = float(sample.costs[int(arm)])
        best_cost = float(np.nanmin(sample.costs[feasible_mask]))
        regret = selected_cost - best_cost
        pulls_problem = int(problem_arm_counts[int(arm)])
        pulls_global = int(self.arm_counts[int(arm)])
        avg_reward_problem = problem_arm_reward_sums[int(arm)] / max(1, pulls_problem)
        avg_reward_global = self.arm_reward_sums[int(arm)] / max(1, pulls_global)
        diag = np.diag(head_state["A_inv"])
        lines = [
            (
                f"t={self.total_steps} | UPDATE | problem={problem} | head={head_key} | "
                f"uid={uid} | global_index={global_index} | arm={int(arm)}({self._arm_name(int(arm))})"
            ),
            (
                f"    outcome: reward={float(reward):.4f} | selected_cost={selected_cost:.4f} | "
                f"best_cost={best_cost:.4f} | regret={regret:.4f}"
            ),
            self._selected_result_text(sample, int(arm)).replace("result:", "    result:", 1),
            (
                f"    arm_stats: problem_pulls={pulls_problem} | problem_avg_reward={float(avg_reward_problem):.4f} | "
                f"global_pulls={pulls_global} | global_avg_reward={float(avg_reward_global):.4f}"
            ),
            (
                f"    state: rep_history={len(self.representation_history)}/{self._window_text(self.representation_buffer_size)} | "
                f"linear_history={len(self.linear_head_history)}/{self._window_text(self.linear_head_buffer_size)} | "
                f"head_mode={'problem_specific' if self.problem_specific_heads else 'shared'} | "
                f"theta_norm={float(np.linalg.norm(head_state['theta'])):.4f} | "
                f"A_inv_diag_mean={float(np.mean(diag)):.6f} | "
                f"A_inv_diag_min={float(np.min(diag)):.6f} | "
                f"A_inv_diag_max={float(np.max(diag)):.6f}"
            ),
        ]
        return "\n".join(lines)

    def _format_train_log(self, train_stats: dict) -> str:
        """把一次表示层训练整理成统一格式。"""
        lines = [
            f"t={self.total_steps} | TRAIN | module=representation | trigger=train_every({self.train_every})",
            (
                f"    optimize: epochs={int(train_stats['num_epochs'])} | opt_steps={int(train_stats['num_steps'])} | "
                f"batch_size={self.representation_batch_size if self.representation_batch_size > 0 else 'FULL'} | "
                f"lr={float(self.optimizer.param_groups[0]['lr']):.6g}"
            ),
            (
                f"    buffers: rep_history={len(self.representation_history)}/{self._window_text(self.representation_buffer_size)} | "
                f"linear_history={len(self.linear_head_history)}/{self._window_text(self.linear_head_buffer_size)}"
            ),
            (
                f"    sampling: balance_mode={train_stats['balance_mode']} | power={float(train_stats['balance_power']):.3f} | "
                f"loss_reweight={bool(train_stats['loss_reweight'])} | groups={int(train_stats['num_groups'])} | "
                f"group_count_min={int(train_stats['min_group_count'])} | group_count_max={int(train_stats['max_group_count'])}"
            ),
            (
                f"    heads: mode={'problem_specific' if self.problem_specific_heads else 'shared'} | "
                f"active={','.join(sorted(self.problem_heads.keys()))}"
            ),
            (
                f"    loss: avg={float(train_stats['avg_loss']):.6f} | min={float(train_stats['min_loss']):.6f} | "
                f"max={float(train_stats['max_loss']):.6f} | last={float(train_stats['last_loss']):.6f}"
            ),
        ]
        return "\n".join(lines)

    def _representation_records(self) -> list[tuple[object, int, float, np.ndarray]]:
        """
        返回当前用于表示层训练的缓存窗口。

        当前默认语义：
        - 不是“全部历史”
        - 而是最近一个窗口 recent buffer
        """
        return list(self.representation_history)

    def _linear_head_records(self) -> list[tuple[object, int, float, np.ndarray]]:
        """
        返回当前用于线性头重建的缓存窗口。

        这个窗口可以和表示层训练窗口不同。
        一般希望它更大一些，这样线性头能保留更多 bandit 历史信息。
        """
        return list(self.linear_head_history)

    def _history_records(self) -> list[tuple[object, int, float, np.ndarray]]:
        """
        向后兼容接口。

        旧测试和旧分析代码里有时会直接取 _history_records()，
        这里默认返回“表示层训练窗口”。
        """
        return self._representation_records()

    def _representation_balance_key(self, record: tuple[object, int, float, np.ndarray]):
        """
        为一条表示层 history record 生成“均衡采样”的分组键。

        可选分组方式：
        - none: 所有样本看成同一组
        - arm: 只按 arm 分组
        - problem_arm: 按 (problem, arm) 联合分组
        """
        sample, arm, _reward, _theta_at_pull = record
        mode = self.representation_balance_mode
        if mode == "arm":
            return ("arm", int(arm))
        if mode == "problem_arm":
            problem, _uid, _global_index = self._sample_meta(sample)
            return (problem, int(arm))
        return ("all", 0)

    def _representation_sampling_info(self, history_records: list[tuple[object, int, float, np.ndarray]]):
        """
        计算表示层训练用的采样概率与 loss 权重。

        返回：
        - sampling_probs: 用于“这个 epoch 如何抽 batch”
        - loss_weights:   用于 batch 内 loss 重加权
        - group_stats:    方便写训练日志
        """
        num_records = len(history_records)
        if num_records == 0:
            ones = np.zeros(0, dtype=np.float64)
            return ones, ones, {"num_groups": 0, "min_group_count": 0, "max_group_count": 0}

        keys = [self._representation_balance_key(record) for record in history_records]
        group_counts = Counter(keys)
        count_values = list(group_counts.values())
        group_stats = {
            "num_groups": len(group_counts),
            "min_group_count": min(count_values),
            "max_group_count": max(count_values),
        }

        if self.representation_balance_mode == "none" or self.representation_balance_power <= 0.0:
            sampling_probs = np.full(num_records, 1.0 / num_records, dtype=np.float64)
            loss_weights = np.ones(num_records, dtype=np.float64)
            return sampling_probs, loss_weights, group_stats

        raw_weights = np.asarray(
            [float(group_counts[key]) ** (-self.representation_balance_power) for key in keys],
            dtype=np.float64,
        )
        raw_weights = np.maximum(raw_weights, 1e-12)
        sampling_probs = raw_weights / np.sum(raw_weights)
        # loss 权重归一化到均值约为 1，避免纯数值尺度把优化器搞乱。
        loss_weights = raw_weights / np.mean(raw_weights)
        return sampling_probs, loss_weights, group_stats

    def _encode_actions(self, sample, arm_ids: list[int]) -> torch.Tensor:
        """
        把一个 sample 和多个候选 arm 编码成 phi(x, a)。

        这里采用 action-conditioned 设计：
        - 先编码实例 x
        - 再取 arm embedding
        - 最后拼接得到 (x, a) 的联合表示
        """
        base = self.context_encoder.encode_samples([sample]).to(self.device)
        arm_tensor = torch.as_tensor(arm_ids, dtype=torch.long, device=self.device)
        arm_emb = self.arm_embedding(arm_tensor)
        base = base.expand(len(arm_ids), -1)
        return self.phi_net(torch.cat((base, arm_emb), dim=1))

    def _encode_record_batch(self, records: list[tuple[object, int, float, np.ndarray]]) -> torch.Tensor:
        """
        把一批 history records 一次性编码成 phi(x, a)。

        这里是给表示层 mini-batch 训练准备的批量版本。
        和 _encode_actions() 的区别在于：
        - _encode_actions() 处理“一个 sample + 多个候选 arm”
        - 这里处理“多个 sample，每个 sample 对应一个已选 arm”

        这样做以后，表示层训练就能真正使用 mini-batch，
        而不再是逐条样本 SGD。
        """
        if not records:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=self.device)

        samples = [sample for sample, _arm, _reward, _theta_at_pull in records]
        arms = [int(arm) for _sample, arm, _reward, _theta_at_pull in records]
        base = self.context_encoder.encode_samples(samples).to(self.device)
        arm_tensor = torch.as_tensor(arms, dtype=torch.long, device=self.device)
        arm_emb = self.arm_embedding(arm_tensor)
        return self.phi_net(torch.cat((base, arm_emb), dim=1))

    def _decision_details(self, sample, use_ucb: bool = True, allow_train_epsilon: bool = False) -> dict:
        """
        返回当前样本的一次完整决策细节。

        `allow_train_epsilon=False` 时：
        - 这是纯评测 / 纯复盘版本
        - 不会引入训练期额外随机探索

        `allow_train_epsilon=True` 时：
        - 只在训练阶段使用
        - warm start 结束后，会额外按 `train_epsilon` 做一次随机探索
        """
        feasible_arms = [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)]
        if not feasible_arms:
            raise ValueError("没有可选 arm")
        head_key = self._head_key_from_sample(sample)
        head_state = self._head_state(head_key)
        problem_arm_counts = self._problem_arm_count_array(head_key)

        under_sampled = []
        if self.initial_pulls > 0:
            under_sampled = [arm for arm in feasible_arms if problem_arm_counts[arm] < self.initial_pulls]

        with torch.no_grad():
            phi = self._encode_actions(sample, feasible_arms).detach().cpu().numpy()

        detail_map: dict[int, dict] = {}
        for idx, arm in enumerate(feasible_arms):
            mean = float(head_state["theta"] @ phi[idx])
            bonus = float(math.sqrt(float(phi[idx] @ head_state["A_inv"] @ phi[idx])))
            score = mean + self.alpha * bonus if use_ucb else mean
            detail_map[int(arm)] = {
                "arm_id": int(arm),
                "feasible": True,
                "pull_count": int(problem_arm_counts[arm]),
                "global_pull_count": int(self.arm_counts[arm]),
                "mean": mean,
                "bonus": bonus,
                "score": score,
            }

        arm_details = []
        for arm in range(self.n_arms):
            if arm in detail_map:
                arm_details.append(detail_map[arm])
            else:
                arm_details.append(
                    {
                        "arm_id": int(arm),
                        "feasible": False,
                        "pull_count": int(problem_arm_counts[arm]),
                        "global_pull_count": int(self.arm_counts[arm]),
                        "mean": None,
                        "bonus": None,
                        "score": None,
                    }
                )

        epsilon_applied = False
        if under_sampled:
            selected_arm = int(sorted(under_sampled, key=lambda arm: (self.arm_counts[arm], arm))[0])
            selection_reason = "warm_start"
        elif allow_train_epsilon and self.train_epsilon > 0.0 and float(np.random.random()) < self.train_epsilon:
            selected_arm = int(np.random.choice(feasible_arms))
            selection_reason = "epsilon_random"
            epsilon_applied = True
        else:
            selected_arm = int(max((detail_map[arm] for arm in feasible_arms), key=lambda item: float(item["score"]))["arm_id"])
            selection_reason = "score"

        return {
            "selected_arm": selected_arm,
            "selection_reason": selection_reason,
            "use_ucb": bool(use_ucb),
            "head_key": head_key,
            "feasible_arms": [int(arm) for arm in feasible_arms],
            "under_sampled_arms": [int(arm) for arm in under_sampled],
            "arm_details": arm_details,
            "train_epsilon": float(self.train_epsilon if allow_train_epsilon else 0.0),
            "epsilon_applied": bool(epsilon_applied),
        }

    def decision_details(self, sample, use_ucb: bool = True) -> dict:
        """
        评测 / 复盘用的纯决策版本。

        这里不启用训练期 epsilon，保证：
        - val/test 可重复
        - trace 里看到的是“模型真实打分后会怎么选”
        """
        return self._decision_details(sample, use_ucb=use_ucb, allow_train_epsilon=False)

    def select_with_details(self, sample, use_ucb: bool = True) -> dict:
        """
        训练阶段专用的决策版本。

        这里只返回“这一次真正做出的训练决策”，
        因此允许引入训练期的 epsilon 探索。
        """
        return self._decision_details(sample, use_ucb=use_ucb, allow_train_epsilon=True)

    def select(self, sample, use_ucb: bool = True) -> int:
        """
        在当前实例可行动作里选择一个 arm。

        评分公式：
        score(a) = theta^T phi(x, a) + alpha * sqrt(phi(x,a)^T A^{-1} phi(x,a))
        """
        return int(self.decision_details(sample, use_ucb=use_ucb)["selected_arm"])

    def update(self, sample, arm: int, reward: float) -> None:
        """
        收到一次 bandit 反馈后更新。

        当前分两部分：
        1. 先把最后一层线性 UCB 状态更新掉
        2. 再把这次样本分别缓存到表示层窗口和线性头窗口中

        这里特别缓存了 theta_at_pull：
        - 这是因为表示学习阶段要用“当时那一刻的 theta”来构造训练目标
        - 这是更接近 Xu 2022 论文精神的写法
        """
        head_key = self._head_key_from_sample(sample)
        head_state = self._head_state(head_key)
        theta_at_pull = head_state["theta"].copy()
        with torch.no_grad():
            phi = self._encode_actions(sample, [arm])[0].detach().cpu().numpy()

        # last-layer LinUCB 的标准增量更新
        a_inv_phi = head_state["A_inv"] @ phi
        denom = 1.0 + float(phi @ a_inv_phi)
        head_state["A_inv"] = head_state["A_inv"] - np.outer(a_inv_phi, a_inv_phi) / denom
        head_state["b"] = head_state["b"] + float(reward) * phi
        head_state["theta"] = head_state["A_inv"] @ head_state["b"]
        self.arm_counts[int(arm)] += 1
        self.arm_reward_sums[int(arm)] += float(reward)
        self._problem_arm_count_array(head_key)[int(arm)] += 1
        self._problem_arm_reward_sum_array(head_key)[int(arm)] += float(reward)
        record = (sample, int(arm), float(reward), theta_at_pull)
        self.representation_history.append(record)
        self.linear_head_history.append(record)
        self.total_steps += 1

        logger.info(self._format_update_log(sample, int(arm), float(reward)))

        if self.train_every > 0 and self.total_steps % self.train_every == 0:
            train_stats = self.fit_representation()
            if train_stats is not None:
                logger.info(self._format_train_log(train_stats))

    def rebuild_linear_head_from_history(self) -> None:
        """
        用“当前表示网络 + 线性头专用缓存窗口”重新构建最后一层线性 UCB 统计量。

        这一步很重要。
        因为一旦 phi_net 被更新，旧的 A / b / theta 对应的其实是“旧特征空间”。
        如果不重建，最后一层线性模型和当前表示层就不再一致。

        因此在表示学习结束后，需要把线性头缓存里的样本重新过一遍当前 phi，
        再恢复出新的：
        - A^{-1}
        - b
        - theta
        """
        rebuilt_heads: dict[str, dict[str, np.ndarray]] = {}
        for head_key in self.problem_heads.keys():
            rebuilt_heads[head_key] = self._new_head_state()

        history_records = self._linear_head_records()
        with torch.no_grad():
            for sample, arm, reward, _theta_at_pull in history_records:
                head_key = self._head_key_from_sample(sample)
                if head_key not in rebuilt_heads:
                    rebuilt_heads[head_key] = self._new_head_state()
                phi = self._encode_actions(sample, [arm])[0].detach().cpu().numpy()
                a_inv_phi = rebuilt_heads[head_key]["A_inv"] @ phi
                denom = 1.0 + float(phi @ a_inv_phi)
                rebuilt_heads[head_key]["A_inv"] = (
                    rebuilt_heads[head_key]["A_inv"] - np.outer(a_inv_phi, a_inv_phi) / denom
                )
                rebuilt_heads[head_key]["b"] = rebuilt_heads[head_key]["b"] + float(reward) * phi
        for head_key, state in rebuilt_heads.items():
            state["theta"] = state["A_inv"] @ state["b"]
        self.problem_heads = rebuilt_heads
        for head_key in self._initial_head_keys():
            self._ensure_head_state(head_key)

    def fit_representation(self) -> dict | None:
        """
        周期性训练表示网络。

        当前采用的训练目标是：
        - 用缓存下来的 theta_at_pull 与当前 phi(x, a) 做内积
        - 拟合真实 reward

        这对应的是：
        “让表示层学出一个更适合最后一层线性 UCB 的特征空间”

        当前这里已经支持 mini-batch：
        - bandit 主循环仍然是一条样本一步一步 online 更新
        - 只有表示层训练部分使用 batch

        这样做的好处是：
        - 不改变 contextual bandit 的在线决策语义
        - 但能显著减少表示层训练时的 Python 循环开销
        - 也能让 GPU 更容易吃满一些
        """
        history_records = self._representation_records()
        if not history_records:
            return None

        self.train()
        losses: list[float] = []
        num_epochs = 0
        sampling_probs, loss_weights, group_stats = self._representation_sampling_info(history_records)
        for _ in range(self.representation_steps):
            num_epochs += 1
            total_loss = 0.0
            count = 0
            if self.representation_balance_mode == "none" or self.representation_balance_power <= 0.0:
                order = np.random.permutation(len(history_records))
            else:
                # 这里采用“带放回的加权采样”而不是普通 permutation。
                # 直观上，它会让低频 group 在一个优化轮次里反复出现，
                # 从而减弱“高频 arm 完全主导表示层”的问题。
                order = np.random.choice(
                    len(history_records),
                    size=len(history_records),
                    replace=True,
                    p=sampling_probs,
                )
            batch_size = len(history_records) if self.representation_batch_size <= 0 else self.representation_batch_size
            for start in range(0, len(order), batch_size):
                batch_indices = order[start : start + batch_size]
                batch_records = [history_records[int(idx)] for idx in batch_indices]
                phi = self._encode_record_batch(batch_records)
                theta_tensor = torch.as_tensor(
                    np.stack([theta_at_pull for _sample, _arm, _reward, theta_at_pull in batch_records], axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                target = torch.as_tensor(
                    [float(reward) for _sample, _arm, reward, _theta_at_pull in batch_records],
                    dtype=torch.float32,
                    device=self.device,
                )
                pred = torch.sum(phi * theta_tensor, dim=1)
                mse = (pred - target) ** 2
                if self.representation_loss_reweight and len(batch_indices) > 0:
                    batch_weights = torch.as_tensor(
                        loss_weights[np.asarray(batch_indices, dtype=np.int64)],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    loss = torch.sum(mse * batch_weights) / torch.sum(batch_weights)
                else:
                    loss = torch.mean(mse)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(float(loss.item()))
                batch_len = len(batch_records)
                total_loss += float(loss.item()) * batch_len
                count += batch_len
            if count > 0 and total_loss / count <= 1e-4:
                break
        self.eval()
        self.rebuild_linear_head_from_history()
        if not losses:
            return None
        return {
            "num_epochs": num_epochs,
            "num_steps": len(losses),
            "avg_loss": float(np.mean(losses)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "last_loss": float(losses[-1]),
            "balance_mode": self.representation_balance_mode,
            "balance_power": float(self.representation_balance_power),
            "loss_reweight": bool(self.representation_loss_reweight),
            "num_groups": int(group_stats["num_groups"]),
            "min_group_count": int(group_stats["min_group_count"]),
            "max_group_count": int(group_stats["max_group_count"]),
        }
