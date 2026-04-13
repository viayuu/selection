from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    from encoders import HybridContextEncoder  # type: ignore
else:
    from .encoders import HybridContextEncoder


class SupervisedRanker(nn.Module):
    """
    监督式 learning-to-rank 模型。

    设计思路和当前 2cmab2/neural_linucb.py 尽量保持一致：
    1. 先用 NSS 风格 encoder 编码实例上下文 x
    2. 再把统一 arm id 映射成 arm embedding e(a)
    3. 拼成 action-conditioned 表示
    4. 最后输出每个 arm 的一个实数 score

    与 Neural-LinUCB 的核心差异是：
    - 这里没有 UCB 线性头
    - 也没有 A / b / theta 的 bandit 统计量
    - 整个模型直接用监督式 ranking loss 训练
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_arms = int(n_arms)
        self.hidden_dim = int(hidden_dim)
        self.arm_embed_dim = int(arm_embed_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.context_encoder = HybridContextEncoder()
        self.arm_embedding = nn.Embedding(self.n_arms, self.arm_embed_dim)
        self.problem_to_prior_index = {"tsp": 0, "cvrp": 1}
        self.use_score_priors = False
        self.register_buffer("score_priors", torch.zeros((2, self.n_arms), dtype=torch.float32))

        # 这里的输入维度与 2cmab2 的 Neural-LinUCB 对齐：
        # context(270) + arm_embedding(16) = 286
        self.phi_net = nn.Sequential(
            nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(self.hidden_dim, 1)

        self.to(self.device)

    def freeze_graph_encoder(self) -> None:
        """只冻结底层 NSS 图编码器，保留后面的排序头可训练。"""
        self.context_encoder.freeze_graph_encoder()

    def set_problem_arm_priors(self, prior_by_problem: dict[str, np.ndarray | list[float]]) -> None:
        """
        设置 problem-specific 的 arm 先验分数。

        当前先验的作用是：
        - 不让网络从 0 开始重新学习“每个问题默认更偏好哪个 arm”
        - 而是把学习重点转移到“哪些实例应该偏离这个默认策略”
        """
        prior_tensor = torch.zeros_like(self.score_priors)
        for problem, values in prior_by_problem.items():
            key = str(problem).lower()
            if key not in self.problem_to_prior_index:
                continue
            prior_tensor[self.problem_to_prior_index[key]] = torch.as_tensor(values, dtype=torch.float32)
        self.score_priors.copy_(prior_tensor)
        self.use_score_priors = True

    def clear_problem_arm_priors(self) -> None:
        """关闭 score prior，退回纯监督 ranker。"""
        self.score_priors.zero_()
        self.use_score_priors = False

    def zero_initialize_residual_head(self) -> None:
        """
        当我们采用 prior + residual 设计时，
        让残差头从 0 开始通常更合理。

        否则随机初始化的 residual score 可能一开始就把 prior 冲掉，
        使“围绕先验做小偏移”的设计意图失效。
        """
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def _batch_prior_scores(self, samples) -> torch.Tensor:
        if not self.use_score_priors or len(samples) == 0:
            return torch.zeros((len(samples), self.n_arms), dtype=torch.float32, device=self.device)
        prior_indices = [self.problem_to_prior_index[str(sample.problem).lower()] for sample in samples]
        index_tensor = torch.as_tensor(prior_indices, dtype=torch.long, device=self.device)
        return self.score_priors[index_tensor]

    def encode_all_arms(self, samples) -> tuple[torch.Tensor, torch.Tensor]:
        """
        一次性为一个 batch 的所有统一 arm 打分。

        输入：
        - samples: list[InstanceSample]，长度为 B

        输出：
        - phi:    [B, A, H]
        - scores: [B, A]

        这里即使某个 arm 对某个问题不可行，也仍然会先算出一个 score；
        真正做 loss 和选臂时再由 feasible_mask 过滤。
        """
        if len(samples) == 0:
            empty_phi = torch.zeros((0, self.n_arms, self.hidden_dim), dtype=torch.float32, device=self.device)
            empty_scores = torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device)
            return empty_phi, empty_scores

        base = self.context_encoder.encode_samples(samples).to(self.device)
        batch_size = base.shape[0]

        arm_ids = torch.arange(self.n_arms, dtype=torch.long, device=self.device)
        arm_emb = self.arm_embedding(arm_ids)[None, :, :].expand(batch_size, -1, -1)
        base_expanded = base[:, None, :].expand(-1, self.n_arms, -1)

        fused = torch.cat((base_expanded, arm_emb), dim=2)
        phi = self.phi_net(fused.reshape(batch_size * self.n_arms, -1)).reshape(batch_size, self.n_arms, self.hidden_dim)
        residual_scores = self.score_head(phi).squeeze(-1)
        scores = residual_scores + self._batch_prior_scores(samples)
        return phi, scores

    def score_samples(self, samples) -> torch.Tensor:
        """
        只返回 [B, A] 分数矩阵，供训练和评测使用。
        """
        _phi, scores = self.encode_all_arms(samples)
        return scores

    def predict_details(self, sample, arm_names: list[str]) -> dict:
        """
        返回一个样本上全部 arm 的预测细节，便于评测 trace 复盘。
        """
        with torch.no_grad():
            scores = self.score_samples([sample])[0].detach().cpu().numpy()

        feasible_arms = [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)]
        masked_scores = np.full_like(scores, -np.inf, dtype=np.float64)
        masked_scores[feasible_arms] = scores[feasible_arms]
        selected_arm = int(np.argmax(masked_scores))

        arm_details = []
        for arm_id in range(self.n_arms):
            feasible = bool(sample.feasible_mask[arm_id] > 0)
            arm_details.append(
                {
                    "arm_id": int(arm_id),
                    "arm_name": arm_names[int(arm_id)],
                    "feasible": feasible,
                    "score": None if not feasible else float(scores[arm_id]),
                    "true_reward": None if not feasible else float(sample.rewards[arm_id]),
                    "true_cost": None if not feasible else float(sample.costs[arm_id]),
                    "true_rank": None if not feasible else float(sample.ranks[arm_id]),
                }
            )

        return {
            "selected_arm": selected_arm,
            "feasible_arms": feasible_arms,
            "arm_details": arm_details,
        }


class DefaultGainRanker(SupervisedRanker):
    """
    默认策略相对收益回归器。

    含义：
    - 模型不再直接输出“谁是 best arm”
    - 而是输出“当前 arm 相对默认 arm 值不值得切过去”

    推理时：
    - 默认 arm 的隐式 gain 视为 0
    - 如果某个 alternative arm 的预测 gain > threshold
      就切到 gain 最大的那个
    - 否则 stay 在默认 arm
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.default_arm_by_problem = {"tsp": 0, "cvrp": 0}
        self.decision_threshold = 0.0
        self.decision_threshold_by_problem = {"tsp": 0.0, "cvrp": 0.0}

        # 虽然底层图编码器已经按问题分成了 TSP / CVRP 两套，
        # 但如果收益头仍然完全共享，模型还是容易被迫学成“平均策略”。
        #
        # 这里进一步把默认收益回归的上层 head 也拆成 problem-specific：
        # - TSP 学自己的“默认 arm -> 备选 arm 收益映射”
        # - CVRP 学自己的
        self.problem_phi_nets = nn.ModuleDict(
            {
                "tsp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, self.hidden_dim),
                    nn.GELU(),
                ),
                "cvrp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, self.hidden_dim),
                    nn.GELU(),
                ),
            }
        )
        self.problem_score_heads = nn.ModuleDict(
            {
                "tsp": nn.Linear(self.hidden_dim, 1),
                "cvrp": nn.Linear(self.hidden_dim, 1),
            }
        )
        self.to(self.device)

    def set_default_arms(self, best_by_problem: dict[str, int]) -> None:
        normalized = {str(problem).lower(): int(arm) for problem, arm in best_by_problem.items()}
        for problem in ("tsp", "cvrp"):
            if problem in normalized:
                self.default_arm_by_problem[problem] = normalized[problem]

    def candidate_arms_for_problem(self, problem: str) -> list[int] | None:
        """
        默认收益回归器默认允许所有可行动作都作为替代臂。

        子类如果想只保留少数候选臂，可以覆写这个方法。
        """
        return None

    def set_decision_thresholds(self, threshold_by_problem: dict[str, float]) -> None:
        for problem in ("tsp", "cvrp"):
            if problem in threshold_by_problem:
                self.decision_threshold_by_problem[problem] = float(threshold_by_problem[problem])
        # 保留一个标量兼容旧代码/旧日志。
        if self.decision_threshold_by_problem:
            self.decision_threshold = float(np.mean(list(self.decision_threshold_by_problem.values())))

    def decision_threshold_for_problem(self, problem: str) -> float:
        return float(self.decision_threshold_by_problem.get(str(problem).lower(), self.decision_threshold))

    def score_samples(self, samples) -> torch.Tensor:
        """
        返回 `[B, A]` 的默认相对收益预测。

        与父类不同的是，这里会按 problem 路由到不同的 head。
        """
        if len(samples) == 0:
            return torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device)

        base = self.context_encoder.encode_samples(samples).to(self.device)
        batch_size = base.shape[0]
        arm_ids = torch.arange(self.n_arms, dtype=torch.long, device=self.device)
        arm_emb = self.arm_embedding(arm_ids)[None, :, :].expand(batch_size, -1, -1)
        base_expanded = base[:, None, :].expand(-1, self.n_arms, -1)
        fused = torch.cat((base_expanded, arm_emb), dim=2)

        outputs = torch.zeros((batch_size, self.n_arms), dtype=torch.float32, device=self.device)
        for problem in ("tsp", "cvrp"):
            idxs = [idx for idx, sample in enumerate(samples) if str(sample.problem).lower() == problem]
            if not idxs:
                continue
            fused_problem = fused[idxs].reshape(len(idxs) * self.n_arms, -1)
            phi_problem = self.problem_phi_nets[problem](fused_problem)
            score_problem = self.problem_score_heads[problem](phi_problem).reshape(len(idxs), self.n_arms)
            outputs[idxs] = score_problem
        return outputs


class OneVsDefaultGainRanker(DefaultGainRanker):
    """
    one-vs-default 候选分解版本。

    与 `DefaultGainRanker` 的区别：
    - 不再给所有 arm 共用同一套“收益回归”头
    - 而是先为每个问题筛出少数候选替代 arm
    - 然后每个候选 arm 各自拥有独立 head，专门学习：
      “在什么实例上，这个 arm 会优于默认 arm”

    这更贴近当前数据结构：
    - TSP 默认是 `icam`
    - 真正值得比较的是 `lehd/pointerformer/elg/difusco` 这类少数候选
    - CVRP 默认是 `lehd`
    - 真正值得比较的是 `icam/elg/omni`
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.candidate_arm_by_problem: dict[str, list[int]] = {"tsp": [], "cvrp": []}
        self.candidate_gain_prior_by_problem: dict[str, dict[int, float]] = {"tsp": {}, "cvrp": {}}
        self.output_mode = "gain"
        # 候选臂打分改成显式 action-conditioned 结构：
        # - 输入是 [context(x), arm_embedding(a)]
        # - 这样候选 arm 的身份不再只体现在“最后一个线性头”
        # - 而是能在上层非线性层里和实例上下文产生交互
        self.problem_candidate_phi_nets = nn.ModuleDict(
            {
                "tsp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, self.hidden_dim),
                    nn.GELU(),
                ),
                "cvrp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, self.hidden_dim),
                    nn.GELU(),
                ),
            }
        )
        self.problem_candidate_score_heads = nn.ModuleDict(
            {
                "tsp": nn.Linear(self.hidden_dim, 1),
                "cvrp": nn.Linear(self.hidden_dim, 1),
            }
        )
        self.to(self.device)

    def set_candidate_arms(self, candidate_arm_by_problem: dict[str, list[int]]) -> None:
        """
        注入每个问题的候选替代 arm 集。

        这里会按候选 arm 动态创建独立 head。
        """
        normalized = {
            str(problem).lower(): [int(arm) for arm in arms]
            for problem, arms in candidate_arm_by_problem.items()
        }
        for problem in ("tsp", "cvrp"):
            arms = normalized.get(problem, [])
            self.candidate_arm_by_problem[problem] = list(arms)
            self.candidate_gain_prior_by_problem[problem] = {int(arm): 0.0 for arm in arms}
            module_dict = nn.ModuleDict()
            for arm in arms:
                head = nn.Linear(self.hidden_dim, 1)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                module_dict[str(int(arm))] = head
            self.candidate_heads[problem] = module_dict
        self.to(self.device)

    def set_candidate_gain_priors(self, candidate_gain_prior_by_problem: dict[str, dict[int, float]]) -> None:
        normalized: dict[str, dict[int, float]] = {}
        for problem, mapping in candidate_gain_prior_by_problem.items():
            normalized[str(problem).lower()] = {int(arm): float(value) for arm, value in mapping.items()}
        for problem in ("tsp", "cvrp"):
            if problem in normalized:
                self.candidate_gain_prior_by_problem[problem] = dict(normalized[problem])

    def candidate_arms_for_problem(self, problem: str) -> list[int] | None:
        return list(self.candidate_arm_by_problem.get(str(problem).lower(), []))

    def score_samples(self, samples) -> torch.Tensor:
        """
        只为候选替代 arm 输出预测 gain。

        非候选 arm 的输出保持 0，
        后续由 run_l2r 里的 candidate mask 决定它们不会参与训练和决策。
        """
        if len(samples) == 0:
            return torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device)

        base = self.context_encoder.encode_samples(samples).to(self.device)
        outputs = torch.zeros((len(samples), self.n_arms), dtype=torch.float32, device=self.device)

        for problem in ("tsp", "cvrp"):
            idxs = [idx for idx, sample in enumerate(samples) if str(sample.problem).lower() == problem]
            candidate_arms = self.candidate_arm_by_problem.get(problem, [])
            if not idxs or not candidate_arms:
                continue
            hidden = self.problem_trunks[problem](base[idxs])
            for arm in candidate_arms:
                residual = self.candidate_heads[problem][str(int(arm))](hidden).squeeze(-1)
                prior = float(self.candidate_gain_prior_by_problem.get(problem, {}).get(int(arm), 0.0))
                outputs[idxs, int(arm)] = residual + prior
        return outputs


class OneVsDefaultBeatRanker(OneVsDefaultGainRanker):
    """
    候选级 beat-default 二分类器。

    结构上复用 one-vs-default 的 candidate-specific heads，
    但每个 head 的输出解释为：

        logit = log P(arm beats default | x) / (1 - P)

    因此：
    - 不使用 gain prior
    - 推理时按 sigmoid(logit) 排序
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.output_mode = "binary"

    def set_candidate_gain_priors(self, candidate_gain_prior_by_problem: dict[str, dict[int, float]]) -> None:
        """
        beat-default 版本不用 gain prior。
        """
        self.candidate_gain_prior_by_problem = {"tsp": {}, "cvrp": {}}


class OneVsDefaultMarginRanker(OneVsDefaultGainRanker):
    """
    候选级 margin / pairwise policy 排序器。

    这里每个候选 arm 输出一个实数 score(a)，它的语义不是：
    - 精确回归 gain
    - 也不是二分类概率

    而是更直接的策略分数：
    - score(a) > 0: 倾向于“该 arm 应该胜过默认 arm”
    - score(a) < 0: 倾向于“默认 arm 更好”

    训练时再用 default-vs-candidate 的相对 cost gap
    作为 signed margin 去监督它。
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.output_mode = "margin"
        self.decision_threshold = 0.0
        self.decision_threshold_by_problem = {"tsp": 0.0, "cvrp": 0.0}

    def set_candidate_gain_priors(self, candidate_gain_prior_by_problem: dict[str, dict[int, float]]) -> None:
        """
        margin policy 版本不使用 gain prior，
        直接学“相对默认 arm 的策略符号和分离边界”。
        """
        self.candidate_gain_prior_by_problem = {"tsp": {}, "cvrp": {}}


class OneVsDefaultTwoStageRanker(nn.Module):
    """
    候选约束的两阶段 one-vs-default 排序器。

    核心思路：
    1. 第一阶段只回答一个问题：
       当前实例是否值得从默认 arm 切走
    2. 第二阶段只在少数候选替代 arm 中做条件选择

    和 `StaySwitchRanker` 的主要区别是：
    - `StaySwitchRanker` 第二阶段会在“所有可行非默认 arm”里归一化
    - 这里第二阶段只看训练集自动筛出的候选替代 arm

    这样更贴近当前数据：
    - 默认策略很强
    - 真正有意义的偏离，通常只发生在少数几个强替代臂之间
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_arms = int(n_arms)
        self.hidden_dim = int(hidden_dim)
        self.arm_embed_dim = int(arm_embed_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.context_encoder = HybridContextEncoder()
        self.arm_embedding = nn.Embedding(self.n_arms, self.arm_embed_dim)
        self.default_arm_by_problem = {"tsp": 0, "cvrp": 0}
        self.candidate_arm_by_problem: dict[str, list[int]] = {"tsp": [], "cvrp": []}

        self.problem_switch_heads = nn.ModuleDict(
            {
                "tsp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                ),
                "cvrp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                ),
            }
        )
        self.problem_candidate_phi_nets = nn.ModuleDict(
            {
                "tsp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, self.hidden_dim),
                    nn.GELU(),
                ),
                "cvrp": nn.Sequential(
                    nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, self.hidden_dim),
                    nn.GELU(),
                ),
            }
        )
        self.problem_candidate_score_heads = nn.ModuleDict(
            {
                "tsp": nn.Linear(self.hidden_dim, 1),
                "cvrp": nn.Linear(self.hidden_dim, 1),
            }
        )

        # 让第一阶段初始时更接近“保守 stay”而不是随机乱切。
        for problem in ("tsp", "cvrp"):
            final_layer = self.problem_switch_heads[problem][-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

        self.to(self.device)

    def freeze_graph_encoder(self) -> None:
        self.context_encoder.freeze_graph_encoder()

    def set_default_arms(self, best_by_problem: dict[str, int]) -> None:
        normalized = {str(problem).lower(): int(arm) for problem, arm in best_by_problem.items()}
        for problem in ("tsp", "cvrp"):
            if problem in normalized:
                self.default_arm_by_problem[problem] = normalized[problem]

    def set_candidate_arms(self, candidate_arm_by_problem: dict[str, list[int]]) -> None:
        normalized = {
            str(problem).lower(): [int(arm) for arm in arms]
            for problem, arms in candidate_arm_by_problem.items()
        }
        for problem in ("tsp", "cvrp"):
            self.candidate_arm_by_problem[problem] = list(normalized.get(problem, []))
        self.to(self.device)

    def candidate_arms_for_problem(self, problem: str) -> list[int] | None:
        return list(self.candidate_arm_by_problem.get(str(problem).lower(), []))

    def _default_arm_tensor(self, samples) -> torch.Tensor:
        default_arms = [int(self.default_arm_by_problem[str(sample.problem).lower()]) for sample in samples]
        return torch.as_tensor(default_arms, dtype=torch.long, device=self.device)

    def _candidate_alt_logits(
        self,
        *,
        base_problem: torch.Tensor,
        candidate_arms: list[int],
        problem: str,
    ) -> torch.Tensor:
        """
        计算某个问题下，一批样本对候选 arm 的 action-conditioned 打分。

        输入：
        - base_problem: [B, C]
        - candidate_arms: 当前问题允许参与 one-vs-default 竞争的少数候选 arm

        输出：
        - logits: [B, len(candidate_arms)]
        """
        if base_problem.numel() == 0 or not candidate_arms:
            return torch.zeros((base_problem.shape[0], 0), dtype=torch.float32, device=self.device)

        candidate_tensor = torch.as_tensor(candidate_arms, dtype=torch.long, device=self.device)
        arm_emb = self.arm_embedding(candidate_tensor)[None, :, :].expand(base_problem.shape[0], -1, -1)
        base_expanded = base_problem[:, None, :].expand(-1, len(candidate_arms), -1)
        fused = torch.cat((base_expanded, arm_emb), dim=2)
        phi = self.problem_candidate_phi_nets[problem](fused.reshape(base_problem.shape[0] * len(candidate_arms), -1))
        logits = self.problem_candidate_score_heads[problem](phi).reshape(base_problem.shape[0], len(candidate_arms))
        return logits

    def forward_with_details(self, samples) -> dict[str, torch.Tensor]:
        if len(samples) == 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=self.device)
            empty_float = torch.zeros((0,), dtype=torch.float32, device=self.device)
            empty_matrix = torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device)
            return {
                "switch_logits": empty_float,
                "switch_probs": empty_float,
                "default_arms": empty_long,
                "alt_logits": empty_matrix,
                "alt_log_probs": empty_matrix,
                "final_scores": empty_matrix,
            }

        base = self.context_encoder.encode_samples(samples).to(self.device)
        batch_size = len(samples)
        default_arms = self._default_arm_tensor(samples)

        switch_logits = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
        alt_logits = torch.zeros((batch_size, self.n_arms), dtype=torch.float32, device=self.device)

        for problem in ("tsp", "cvrp"):
            idxs = [idx for idx, sample in enumerate(samples) if str(sample.problem).lower() == problem]
            candidate_arms = self.candidate_arm_by_problem.get(problem, [])
            if not idxs:
                continue

            base_problem = base[idxs]
            default_emb = self.arm_embedding(default_arms[idxs])
            switch_logits[idxs] = self.problem_switch_heads[problem](torch.cat((base_problem, default_emb), dim=1)).squeeze(-1)

            if candidate_arms:
                candidate_logits = self._candidate_alt_logits(
                    base_problem=base_problem,
                    candidate_arms=candidate_arms,
                    problem=problem,
                )
                for col_idx, arm in enumerate(candidate_arms):
                    alt_logits[idxs, int(arm)] = candidate_logits[:, col_idx]

        switch_probs = torch.sigmoid(switch_logits)
        alt_log_probs = torch.full((batch_size, self.n_arms), float("-inf"), dtype=torch.float32, device=self.device)
        final_scores = torch.full((batch_size, self.n_arms), float("-inf"), dtype=torch.float32, device=self.device)

        log_switch = F.logsigmoid(switch_logits)
        log_stay = F.logsigmoid(-switch_logits)

        for row_idx, sample in enumerate(samples):
            problem = str(sample.problem).lower()
            default_arm = int(default_arms[row_idx].detach().cpu().item())
            feasible = np.asarray(sample.feasible_mask) > 0
            candidate_arms = [
                int(arm)
                for arm in self.candidate_arm_by_problem.get(problem, [])
                if feasible[int(arm)] and int(arm) != default_arm
            ]

            final_scores[row_idx, default_arm] = log_stay[row_idx]
            if candidate_arms:
                candidate_mask = torch.zeros((self.n_arms,), dtype=torch.bool, device=self.device)
                candidate_mask[candidate_arms] = True
                masked_alt_logits = alt_logits[row_idx].masked_fill(~candidate_mask, float("-inf"))
                row_alt_log_probs = F.log_softmax(masked_alt_logits, dim=0)
                alt_log_probs[row_idx, candidate_mask] = row_alt_log_probs[candidate_mask]
                final_scores[row_idx, candidate_mask] = log_switch[row_idx] + row_alt_log_probs[candidate_mask]
            else:
                final_scores[row_idx, default_arm] = 0.0

        return {
            "switch_logits": switch_logits,
            "switch_probs": switch_probs,
            "default_arms": default_arms,
            "alt_logits": alt_logits,
            "alt_log_probs": alt_log_probs,
            "final_scores": final_scores,
        }

    def score_samples(self, samples) -> torch.Tensor:
        return self.forward_with_details(samples)["final_scores"]

    def predict_details(self, sample, arm_names: list[str]) -> dict:
        with torch.no_grad():
            outputs = self.forward_with_details([sample])
        final_scores = outputs["final_scores"][0].detach().cpu().numpy()
        alt_logits = outputs["alt_logits"][0].detach().cpu().numpy()
        switch_logit = float(outputs["switch_logits"][0].detach().cpu().item())
        switch_prob = float(outputs["switch_probs"][0].detach().cpu().item())
        default_arm = int(outputs["default_arms"][0].detach().cpu().item())
        feasible_arms = [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)]
        selected_arm = int(np.argmax(final_scores))

        arm_details = []
        for arm_id in range(self.n_arms):
            feasible = bool(sample.feasible_mask[arm_id] > 0)
            arm_details.append(
                {
                    "arm_id": int(arm_id),
                    "arm_name": arm_names[int(arm_id)],
                    "feasible": feasible,
                    "is_default_arm": bool(int(arm_id) == default_arm),
                    "is_candidate_arm": bool(int(arm_id) in self.candidate_arm_by_problem.get(str(sample.problem).lower(), [])),
                    "final_score": None if not feasible else float(final_scores[arm_id]),
                    "alt_logit": None if not feasible else float(alt_logits[arm_id]),
                    "true_reward": None if not feasible else float(sample.rewards[arm_id]),
                    "true_cost": None if not feasible else float(sample.costs[arm_id]),
                    "true_rank": None if not feasible else float(sample.ranks[arm_id]),
                }
            )

        return {
            "selected_arm": selected_arm,
            "default_arm": default_arm,
            "default_arm_name": arm_names[default_arm],
            "switch_logit": switch_logit,
            "switch_prob": switch_prob,
            "feasible_arms": feasible_arms,
            "arm_details": arm_details,
        }


class OneVsDefaultSwitchRanker(OneVsDefaultTwoStageRanker):
    """
    显式 stay/switch + candidate ranking 版本。

    结构上完全复用 `OneVsDefaultTwoStageRanker`：
    - 一条 switch head
    - 一组 candidate-specific heads

    区别只在于训练与推理语义：
    - switch head 单独负责“要不要离开默认 arm”
    - alt_logits 只负责“如果已经 switch，候选 arm 里谁更好”
    - 最终推理不是比较 joint log-prob，
      而是：
        1. 先看 switch_prob 是否超过阈值
        2. 再在候选臂里取 alt_score 最大者
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.output_mode = "switch_rank"
        self.decision_threshold = 0.5
        self.decision_threshold_by_problem = {"tsp": 0.5, "cvrp": 0.5}

    def set_decision_thresholds(self, threshold_by_problem: dict[str, float]) -> None:
        for problem in ("tsp", "cvrp"):
            if problem in threshold_by_problem:
                self.decision_threshold_by_problem[problem] = float(threshold_by_problem[problem])
        if self.decision_threshold_by_problem:
            self.decision_threshold = float(np.mean(list(self.decision_threshold_by_problem.values())))

    def decision_threshold_for_problem(self, problem: str) -> float:
        return float(self.decision_threshold_by_problem.get(str(problem).lower(), self.decision_threshold))


class OneVsDefaultCoupledSwitchRanker(OneVsDefaultSwitchRanker):
    """
    耦合版 switch-rank。

    核心区别：
    - 原版 `OneVsDefaultSwitchRanker` 的 switch head 只看 context + default arm
    - 这里的 switch head 还会额外看到 candidate 头当前给出的摘要信息：
      1. 候选分数的 softmax 加权 arm embedding
      2. 候选最大分数
      3. 候选平均分数
      4. top1-top2 gap

    直觉上，这更像真实决策：
    - 不是“盲猜该不该切”
    - 而是“先看看最强候选有多强，再决定要不要切”
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.output_mode = "coupled_switch_rank"
        switch_input_dim = self.context_encoder.output_dim + self.arm_embed_dim + self.arm_embed_dim + 3
        self.problem_coupled_switch_heads = nn.ModuleDict(
            {
                "tsp": nn.Sequential(
                    nn.Linear(switch_input_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                ),
                "cvrp": nn.Sequential(
                    nn.Linear(switch_input_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                ),
            }
        )
        for problem in ("tsp", "cvrp"):
            final_layer = self.problem_coupled_switch_heads[problem][-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        self.to(self.device)

    def forward_with_details(self, samples) -> dict[str, torch.Tensor]:
        if len(samples) == 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=self.device)
            empty_float = torch.zeros((0,), dtype=torch.float32, device=self.device)
            empty_matrix = torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device)
            return {
                "switch_logits": empty_float,
                "switch_probs": empty_float,
                "default_arms": empty_long,
                "alt_logits": empty_matrix,
                "alt_log_probs": empty_matrix,
                "final_scores": empty_matrix,
            }

        base = self.context_encoder.encode_samples(samples).to(self.device)
        batch_size = len(samples)
        default_arms = self._default_arm_tensor(samples)

        alt_logits = torch.zeros((batch_size, self.n_arms), dtype=torch.float32, device=self.device)
        for problem in ("tsp", "cvrp"):
            idxs = [idx for idx, sample in enumerate(samples) if str(sample.problem).lower() == problem]
            candidate_arms = self.candidate_arm_by_problem.get(problem, [])
            if not idxs or not candidate_arms:
                continue
            candidate_logits = self._candidate_alt_logits(
                base_problem=base[idxs],
                candidate_arms=candidate_arms,
                problem=problem,
            )
            for col_idx, arm in enumerate(candidate_arms):
                alt_logits[idxs, int(arm)] = candidate_logits[:, col_idx]

        switch_logits = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
        alt_log_probs = torch.full((batch_size, self.n_arms), float("-inf"), dtype=torch.float32, device=self.device)
        final_scores = torch.full((batch_size, self.n_arms), float("-inf"), dtype=torch.float32, device=self.device)

        for row_idx, sample in enumerate(samples):
            problem = str(sample.problem).lower()
            default_arm = int(default_arms[row_idx].detach().cpu().item())
            default_emb = self.arm_embedding(default_arms[row_idx : row_idx + 1]).squeeze(0)
            feasible = np.asarray(sample.feasible_mask) > 0
            candidate_arms = [
                int(arm)
                for arm in self.candidate_arm_by_problem.get(problem, [])
                if feasible[int(arm)] and int(arm) != default_arm
            ]

            if candidate_arms:
                candidate_tensor = torch.as_tensor(candidate_arms, dtype=torch.long, device=self.device)
                cand_scores = alt_logits[row_idx, candidate_tensor]
                cand_emb = self.arm_embedding(candidate_tensor)
                cand_weights = F.softmax(cand_scores, dim=0)
                pooled_emb = torch.sum(cand_weights[:, None] * cand_emb, dim=0)

                topk = torch.topk(cand_scores, k=min(2, cand_scores.numel()), dim=0).values
                max_score = topk[0]
                second_score = topk[1] if cand_scores.numel() >= 2 else cand_scores.new_zeros(())
                mean_score = torch.mean(cand_scores)
                gap_score = max_score - second_score

                masked_alt_logits = alt_logits[row_idx].masked_fill(
                    ~torch.as_tensor(
                        np.isin(np.arange(self.n_arms), np.asarray(candidate_arms, dtype=np.int64)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    float("-inf"),
                )
                row_alt_log_probs = F.log_softmax(masked_alt_logits, dim=0)
                alt_log_probs[row_idx, candidate_tensor] = row_alt_log_probs[candidate_tensor]
            else:
                pooled_emb = torch.zeros((self.arm_embed_dim,), dtype=torch.float32, device=self.device)
                max_score = alt_logits.new_zeros(())
                second_score = alt_logits.new_zeros(())
                mean_score = alt_logits.new_zeros(())
                gap_score = alt_logits.new_zeros(())

            switch_features = torch.cat(
                (
                    base[row_idx],
                    default_emb,
                    pooled_emb,
                    torch.stack((max_score, mean_score, gap_score)),
                ),
                dim=0,
            ).unsqueeze(0)
            switch_logits[row_idx] = self.problem_coupled_switch_heads[problem](switch_features).squeeze(0).squeeze(0)

        switch_probs = torch.sigmoid(switch_logits)
        log_switch = F.logsigmoid(switch_logits)
        log_stay = F.logsigmoid(-switch_logits)

        for row_idx, sample in enumerate(samples):
            problem = str(sample.problem).lower()
            default_arm = int(default_arms[row_idx].detach().cpu().item())
            feasible = np.asarray(sample.feasible_mask) > 0
            candidate_arms = [
                int(arm)
                for arm in self.candidate_arm_by_problem.get(problem, [])
                if feasible[int(arm)] and int(arm) != default_arm
            ]

            final_scores[row_idx, default_arm] = log_stay[row_idx]
            if candidate_arms:
                candidate_tensor = torch.as_tensor(candidate_arms, dtype=torch.long, device=self.device)
                final_scores[row_idx, candidate_tensor] = log_switch[row_idx] + alt_log_probs[row_idx, candidate_tensor]
            else:
                final_scores[row_idx, default_arm] = 0.0

        return {
            "switch_logits": switch_logits,
            "switch_probs": switch_probs,
            "default_arms": default_arms,
            "alt_logits": alt_logits,
            "alt_log_probs": alt_log_probs,
            "final_scores": final_scores,
        }


class OneVsDefaultCoupledAdvantageRanker(OneVsDefaultCoupledSwitchRanker):
    """
    耦合版 switch-advantage ranker。

    与 `OneVsDefaultCoupledSwitchRanker` 的唯一区别是语义：
    - `switch_logits` 不再解释成“switch 概率的 logit”
    - 而是解释成“当前实例上，偏离默认 arm 能带来的预测收益幅度”

    推理时：
    - 如果 predicted_advantage > threshold，就 switch
    - 否则 stay

    这样可以直接学习连续监督信号，而不是把“有一点点收益”和“收益很大”
    都压成同一个二元标签。
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            hidden_dim=hidden_dim,
            arm_embed_dim=arm_embed_dim,
            device=device,
        )
        self.output_mode = "coupled_advantage_rank"
        self.decision_threshold = 0.0
        self.decision_threshold_by_problem = {"tsp": 0.0, "cvrp": 0.0}

class StaySwitchRanker(nn.Module):
    """
    两阶段门控排序器。

    设计目标：
    1. 不再让模型直接在所有 arm 上“从零学一个统一排序”
    2. 而是先把 `single_best_per_problem` 当作默认动作
    3. 模型只学习两个问题：
       - 当前实例要不要偏离默认动作（stay / switch）
       - 如果要偏离，应该切到哪个 arm

    这比直接 listwise/pairwise 排全表更贴合当前数据特点：
    - 默认固定策略本身已经很强
    - 真正有价值的是“少量高收益的偏离决策”
    """

    def __init__(
        self,
        n_arms: int,
        hidden_dim: int = 64,
        arm_embed_dim: int = 16,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_arms = int(n_arms)
        self.hidden_dim = int(hidden_dim)
        self.arm_embed_dim = int(arm_embed_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.context_encoder = HybridContextEncoder()
        self.arm_embedding = nn.Embedding(self.n_arms, self.arm_embed_dim)

        # 当前默认动作来自训练集上拟合得到的 single_best_per_problem。
        # 这里先放一个占位值，后续由 set_default_arms() 注入。
        self.default_arm_by_problem = {"tsp": 0, "cvrp": 0}

        # 第一阶段：判定 stay / switch。
        # 输入是：
        # - 当前实例上下文 context(x)
        # - 默认 arm 的 embedding e(a_default)
        # 输出一个标量 switch logit。
        self.switch_head = nn.Sequential(
            nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # 第二阶段：只在“已经决定 switch”的前提下，
        # 对候选替代 arm 打分。
        self.alt_phi_net = nn.Sequential(
            nn.Linear(self.context_encoder.output_dim + self.arm_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.hidden_dim),
            nn.GELU(),
        )
        self.alt_head = nn.Linear(self.hidden_dim, 1)

        self.to(self.device)

    def freeze_graph_encoder(self) -> None:
        """冻结底层 NSS 图编码器，保留上层 gate / arm head 可训练。"""
        self.context_encoder.freeze_graph_encoder()

    def set_default_arms(self, best_by_problem: dict[str, int]) -> None:
        """
        设置每个问题的默认 arm。

        例如：
        - TSP 默认选 icam
        - CVRP 默认选 lehd
        """
        normalized = {str(problem).lower(): int(arm) for problem, arm in best_by_problem.items()}
        for problem in ("tsp", "cvrp"):
            if problem in normalized:
                self.default_arm_by_problem[problem] = normalized[problem]

    def _default_arm_tensor(self, samples) -> torch.Tensor:
        default_arms = [int(self.default_arm_by_problem[str(sample.problem).lower()]) for sample in samples]
        return torch.as_tensor(default_arms, dtype=torch.long, device=self.device)

    def _encode_alt_logits(self, samples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码所有 arm 的替代打分 logits。

        返回：
        - base:      [B, 270]
        - phi:       [B, A, H]
        - alt_logits:[B, A]
        """
        if len(samples) == 0:
            empty_base = torch.zeros((0, self.context_encoder.output_dim), dtype=torch.float32, device=self.device)
            empty_phi = torch.zeros((0, self.n_arms, self.hidden_dim), dtype=torch.float32, device=self.device)
            empty_logits = torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device)
            return empty_base, empty_phi, empty_logits

        base = self.context_encoder.encode_samples(samples).to(self.device)
        batch_size = base.shape[0]

        arm_ids = torch.arange(self.n_arms, dtype=torch.long, device=self.device)
        arm_emb = self.arm_embedding(arm_ids)[None, :, :].expand(batch_size, -1, -1)
        base_expanded = base[:, None, :].expand(-1, self.n_arms, -1)

        fused = torch.cat((base_expanded, arm_emb), dim=2)
        phi = self.alt_phi_net(fused.reshape(batch_size * self.n_arms, -1)).reshape(batch_size, self.n_arms, self.hidden_dim)
        alt_logits = self.alt_head(phi).squeeze(-1)
        return base, phi, alt_logits

    def forward_with_details(self, samples) -> dict[str, torch.Tensor]:
        """
        前向同时返回两阶段门控的完整细节。

        输出里的 `final_scores` 不是普通的 raw score，而是：
        - default arm: log P(stay)
        - alternative arm: log P(switch) + log P(alt_arm | switch)

        这样评测时直接取 `argmax(final_scores)`，
        就相当于在比较最终动作概率。
        """
        base, phi, alt_logits = self._encode_alt_logits(samples)
        if len(samples) == 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=self.device)
            return {
                "base": base,
                "phi": phi,
                "alt_logits": alt_logits,
                "switch_logits": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "switch_probs": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "default_arms": empty_long,
                "alt_log_probs": torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device),
                "final_scores": torch.zeros((0, self.n_arms), dtype=torch.float32, device=self.device),
            }

        default_arms = self._default_arm_tensor(samples)
        default_emb = self.arm_embedding(default_arms)
        switch_logits = self.switch_head(torch.cat((base, default_emb), dim=1)).squeeze(-1)
        switch_probs = torch.sigmoid(switch_logits)

        batch_size = len(samples)
        alt_log_probs = torch.full((batch_size, self.n_arms), float("-inf"), dtype=torch.float32, device=self.device)
        final_scores = torch.full((batch_size, self.n_arms), float("-inf"), dtype=torch.float32, device=self.device)

        log_switch = F.logsigmoid(switch_logits)
        log_stay = F.logsigmoid(-switch_logits)

        for row_idx, sample in enumerate(samples):
            feasible = torch.as_tensor(np.asarray(sample.feasible_mask) > 0, dtype=torch.bool, device=self.device)
            default_arm = int(default_arms[row_idx].detach().cpu().item())

            # stay 分支只对应默认 arm。
            final_scores[row_idx, default_arm] = log_stay[row_idx]

            # switch 分支只在“可行且不是默认 arm”的动作上归一化。
            alt_mask = feasible.clone()
            alt_mask[default_arm] = False
            if torch.any(alt_mask):
                masked_alt_logits = alt_logits[row_idx].masked_fill(~alt_mask, float("-inf"))
                row_alt_log_probs = F.log_softmax(masked_alt_logits, dim=0)
                alt_log_probs[row_idx, alt_mask] = row_alt_log_probs[alt_mask]
                final_scores[row_idx, alt_mask] = log_switch[row_idx] + row_alt_log_probs[alt_mask]
            else:
                # 理论上当前任务里一般不会走到这里；
                # 但如果某个样本只有一个可行动作，就强制 stay。
                final_scores[row_idx, default_arm] = 0.0

        return {
            "base": base,
            "phi": phi,
            "alt_logits": alt_logits,
            "switch_logits": switch_logits,
            "switch_probs": switch_probs,
            "default_arms": default_arms,
            "alt_log_probs": alt_log_probs,
            "final_scores": final_scores,
        }

    def score_samples(self, samples) -> torch.Tensor:
        """
        返回最终动作分数 `[B, A]`。

        这里的分数是“最终动作概率的对数”，不是普通回归 score；
        但对 greedy 选臂来说，直接 argmax 是完全等价的。
        """
        return self.forward_with_details(samples)["final_scores"]

    def predict_details(self, sample, arm_names: list[str]) -> dict:
        """
        返回一条样本上的详细预测信息，便于 trace 复盘。
        """
        with torch.no_grad():
            outputs = self.forward_with_details([sample])
        final_scores = outputs["final_scores"][0].detach().cpu().numpy()
        alt_logits = outputs["alt_logits"][0].detach().cpu().numpy()
        switch_logit = float(outputs["switch_logits"][0].detach().cpu().item())
        switch_prob = float(outputs["switch_probs"][0].detach().cpu().item())
        default_arm = int(outputs["default_arms"][0].detach().cpu().item())

        feasible_arms = [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)]
        selected_arm = int(np.argmax(final_scores))

        arm_details = []
        for arm_id in range(self.n_arms):
            feasible = bool(sample.feasible_mask[arm_id] > 0)
            arm_details.append(
                {
                    "arm_id": int(arm_id),
                    "arm_name": arm_names[int(arm_id)],
                    "feasible": feasible,
                    "is_default_arm": bool(int(arm_id) == default_arm),
                    "final_score": None if not feasible else float(final_scores[arm_id]),
                    "alt_logit": None if not feasible else float(alt_logits[arm_id]),
                    "true_reward": None if not feasible else float(sample.rewards[arm_id]),
                    "true_cost": None if not feasible else float(sample.costs[arm_id]),
                    "true_rank": None if not feasible else float(sample.ranks[arm_id]),
                }
            )

        return {
            "selected_arm": selected_arm,
            "default_arm": default_arm,
            "default_arm_name": arm_names[default_arm],
            "switch_logit": switch_logit,
            "switch_prob": switch_prob,
            "feasible_arms": feasible_arms,
            "arm_details": arm_details,
        }
