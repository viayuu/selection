from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_logistic_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    feasible_mask: torch.Tensor,
    gap_weight: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    pairwise learning-to-rank loss。

    对同一实例中的两个 arm：
    - 如果 reward_i > reward_j
    - 就要求 score_i > score_j

    公式可以写成：

        L = softplus(-(s_i - s_j))

    如果打开 gap_weight，则会再乘上 reward gap，
    让“差距更大的 pair”对梯度贡献更大。
    """
    batch_losses: list[torch.Tensor] = []
    pair_count = 0
    weighted_gap_sum = 0.0

    for row_scores, row_rewards, row_mask in zip(scores, rewards, feasible_mask):
        valid_scores = row_scores[row_mask]
        valid_rewards = row_rewards[row_mask]
        if valid_scores.numel() <= 1:
            continue

        reward_diff = valid_rewards[:, None] - valid_rewards[None, :]
        pair_mask = reward_diff > eps
        if not torch.any(pair_mask):
            continue

        score_diff = valid_scores[:, None] - valid_scores[None, :]
        pair_losses = F.softplus(-score_diff[pair_mask])

        if gap_weight:
            weights = reward_diff[pair_mask]
            sample_loss = torch.sum(pair_losses * weights) / torch.clamp(weights.sum(), min=eps)
            weighted_gap_sum += float(weights.sum().detach().cpu().item())
        else:
            sample_loss = torch.mean(pair_losses)

        pair_count += int(pair_mask.sum().detach().cpu().item())
        batch_losses.append(sample_loss)

    if not batch_losses:
        zero = scores.sum() * 0.0
        return zero, {"pair_count": 0, "avg_gap_weight_sum": 0.0}

    loss = torch.mean(torch.stack(batch_losses))
    stats = {
        "pair_count": pair_count,
        "avg_gap_weight_sum": 0.0 if len(batch_losses) == 0 else weighted_gap_sum / max(len(batch_losses), 1),
    }
    return loss, stats


def listwise_cross_entropy_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    feasible_mask: torch.Tensor,
    target_scale: float = 8.0,
) -> tuple[torch.Tensor, dict]:
    """
    listwise cross entropy。

    思路：
    - 先把模型输出 scores 在可行 arm 上做 softmax，得到预测排序分布
    - 再把真实 reward 经过 softmax(target_scale * reward) 变成目标分布
    - 最后最小化两者的交叉熵

    它关注的是“整张候选列表”的相对顺序，而不只是单个 arm 的绝对回归值。
    """
    if scores.ndim != 2:
        raise ValueError("scores 必须是 [B, A] 二维张量")

    masked_scores = scores.masked_fill(~feasible_mask, float("-inf"))
    pred_log_probs = F.log_softmax(masked_scores, dim=1)
    # 无效 arm 上的 log prob 设成 0，避免后面出现 0 * (-inf) -> NaN
    pred_log_probs = torch.where(feasible_mask, pred_log_probs, torch.zeros_like(pred_log_probs))

    target_logits = (rewards * float(target_scale)).masked_fill(~feasible_mask, float("-inf"))
    target_probs = F.softmax(target_logits, dim=1)
    target_probs = torch.where(feasible_mask, target_probs, torch.zeros_like(target_probs))

    per_sample_loss = -(target_probs * pred_log_probs).sum(dim=1)
    loss = per_sample_loss.mean()
    stats = {
        "target_scale": float(target_scale),
        "mean_target_entropy": float(
            (
                -(
                    target_probs
                    * torch.where(
                        feasible_mask,
                        torch.log(torch.clamp(target_probs, min=1e-12)),
                        torch.zeros_like(target_probs),
                    )
                ).sum(dim=1)
            ).mean().detach().cpu().item()
        ),
    }
    return loss, stats


def best_vs_all_regret_loss(
    scores: torch.Tensor,
    normalized_regrets: torch.Tensor,
    feasible_mask: torch.Tensor,
    best_arms: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    只关注 top1 选择是否正确的监督损失。

    对于每个实例 i：
    - 先拿到真实最优 arm: a_i^*
    - 再要求 score(best) > score(other)
    - 并用 normalized regret 作为负样本权重

    公式可以写成：

        L_i = sum_a w_{i,a} * softplus(-(s_{i,best} - s_{i,a})) / sum_a w_{i,a}

    其中：
    - a 遍历该实例中除 best 之外的所有 feasible arm
    - w_{i,a} 取该 arm 相对 best 的 normalized regret

    这样做和 pairwise/listwise 的最大区别是：
    - 梯度几乎全部集中在“best arm 能不能压过其他 arm”
    - 不再把大量训练信号浪费在中游 arm 之间的排序上
    """
    if scores.ndim != 2:
        raise ValueError("scores 必须是 [B, A] 二维张量")

    batch_size = scores.shape[0]
    if sample_weights is None:
        sample_weights = torch.ones(batch_size, dtype=scores.dtype, device=scores.device)
    else:
        sample_weights = sample_weights.to(device=scores.device, dtype=scores.dtype)

    weighted_loss_sum = scores.sum() * 0.0
    total_sample_weight = scores.sum() * 0.0
    pair_count = 0
    valid_sample_count = 0
    regret_weight_sum = 0.0

    for row_idx in range(batch_size):
        row_mask = feasible_mask[row_idx]
        valid_arm_ids = torch.nonzero(row_mask, as_tuple=False).flatten()
        if valid_arm_ids.numel() <= 1:
            continue

        best_arm = int(best_arms[row_idx].detach().cpu().item())
        neg_arm_ids = valid_arm_ids[valid_arm_ids != best_arm]
        if neg_arm_ids.numel() == 0:
            continue

        best_score = scores[row_idx, best_arm]
        neg_scores = scores[row_idx, neg_arm_ids]
        neg_regrets = normalized_regrets[row_idx, neg_arm_ids]

        # 如果某些 arm 和 best 完全并列，normalized regret 可能为 0。
        # 这时退化成均匀权重，避免出现除 0。
        regret_sum = torch.sum(neg_regrets)
        if float(regret_sum.detach().cpu().item()) <= eps:
            pair_weights = torch.ones_like(neg_regrets)
        else:
            pair_weights = neg_regrets

        pair_losses = torch.nn.functional.softplus(-(best_score - neg_scores))
        sample_loss = torch.sum(pair_losses * pair_weights) / torch.clamp(torch.sum(pair_weights), min=eps)

        sample_weight = sample_weights[row_idx]
        weighted_loss_sum = weighted_loss_sum + sample_loss * sample_weight
        total_sample_weight = total_sample_weight + sample_weight

        pair_count += int(neg_arm_ids.numel())
        valid_sample_count += 1
        regret_weight_sum += float(torch.sum(pair_weights).detach().cpu().item())

    if valid_sample_count == 0:
        zero = scores.sum() * 0.0
        return zero, {
            "pair_count": 0,
            "valid_sample_count": 0,
            "mean_sample_weight": 0.0,
            "avg_pair_regret_weight_sum": 0.0,
        }

    loss = weighted_loss_sum / torch.clamp(total_sample_weight, min=eps)
    stats = {
        "pair_count": pair_count,
        "valid_sample_count": valid_sample_count,
        "mean_sample_weight": float(
            torch.mean(sample_weights).detach().cpu().item() if sample_weights.numel() > 0 else 0.0
        ),
        "avg_pair_regret_weight_sum": regret_weight_sum / max(valid_sample_count, 1),
    }
    return loss, stats


def two_stage_default_gate_loss(
    switch_logits: torch.Tensor,
    alt_logits: torch.Tensor,
    feasible_mask: torch.Tensor,
    default_arms: torch.Tensor,
    best_arms: torch.Tensor,
    default_improvements: torch.Tensor,
    switch_thresholds: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    两阶段门控损失。

    先定义伪标签：
    - 如果默认 arm 相对 oracle 的 improvement 小于阈值，就记作 `stay`
    - 否则记作 `switch`，并要求切到真实 best arm

    损失分成两种情况：
    1. stay 样本：
       L_i = -log(1 - sigmoid(z_i))
    2. switch 样本：
       L_i = -log(sigmoid(z_i)) - log P(best_arm | switch, x_i)

    这比“对所有 arm 统一排序”更像当前任务真正需要的行为：
    - 默认策略已经很强
    - 模型应该只在值得的时候偏离它
    """
    if switch_logits.ndim != 1:
        raise ValueError("switch_logits 必须是 [B] 一维张量")
    if alt_logits.ndim != 2:
        raise ValueError("alt_logits 必须是 [B, A] 二维张量")

    batch_size = switch_logits.shape[0]
    if sample_weights is None:
        sample_weights = torch.ones(batch_size, dtype=switch_logits.dtype, device=switch_logits.device)
    else:
        sample_weights = sample_weights.to(device=switch_logits.device, dtype=switch_logits.dtype)

    weighted_loss_sum = switch_logits.sum() * 0.0
    total_sample_weight = switch_logits.sum() * 0.0
    positive_count = 0
    mean_positive_improvement = 0.0

    for row_idx in range(batch_size):
        default_arm = int(default_arms[row_idx].detach().cpu().item())
        best_arm = int(best_arms[row_idx].detach().cpu().item())
        improvement = float(default_improvements[row_idx].detach().cpu().item())
        threshold = float(switch_thresholds[row_idx].detach().cpu().item())
        row_weight = sample_weights[row_idx]

        should_switch = improvement > threshold + eps and best_arm != default_arm
        if should_switch:
            positive_count += 1
            mean_positive_improvement += improvement

            alt_mask = feasible_mask[row_idx].clone()
            alt_mask[default_arm] = False
            masked_alt_logits = alt_logits[row_idx].masked_fill(~alt_mask, float("-inf"))
            alt_log_probs = F.log_softmax(masked_alt_logits, dim=0)
            sample_loss = -F.logsigmoid(switch_logits[row_idx]) - alt_log_probs[best_arm]
        else:
            sample_loss = -F.logsigmoid(-switch_logits[row_idx])

        weighted_loss_sum = weighted_loss_sum + sample_loss * row_weight
        total_sample_weight = total_sample_weight + row_weight

    loss = weighted_loss_sum / torch.clamp(total_sample_weight, min=eps)
    stats = {
        "positive_count": int(positive_count),
        "positive_rate": float(positive_count / max(batch_size, 1)),
        "mean_positive_improvement": 0.0 if positive_count == 0 else float(mean_positive_improvement / positive_count),
        "mean_sample_weight": float(
            torch.mean(sample_weights).detach().cpu().item() if sample_weights.numel() > 0 else 0.0
        ),
    }
    return loss, stats


def default_gain_regression_loss(
    scores: torch.Tensor,
    target_gains: torch.Tensor,
    feasible_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    positive_gain_weight: torch.Tensor | None = None,
    zero_gain_weight: float = 0.2,
    loss_type: str = "smooth_l1",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    学习“相对默认 arm 的收益”。

    target_gains[i, a] 的语义：
    - > 0: 这个 arm 比默认 arm 好，值越大越值得切换
    - = 0: 这个 arm 不优于默认 arm，或者就是默认 arm 自己

    之所以不是直接回归原始 cost / rank，而是回归 default-relative gain：
    - 更贴近当前真正的决策问题：要不要偏离默认策略
    - 所有“比默认更好”的 arm 都能提供正监督，不只 best arm 一条信号
    """
    if sample_weights is None:
        sample_weights = torch.ones(scores.shape[0], dtype=scores.dtype, device=scores.device)
    else:
        sample_weights = sample_weights.to(device=scores.device, dtype=scores.dtype)

    if positive_gain_weight is None:
        positive_gain_weight = torch.ones_like(target_gains)
    else:
        positive_gain_weight = positive_gain_weight.to(device=scores.device, dtype=scores.dtype)

    if loss_type == "smooth_l1":
        per_entry_loss = F.smooth_l1_loss(scores, target_gains, reduction="none")
    elif loss_type == "mse":
        per_entry_loss = (scores - target_gains) ** 2
    else:
        raise ValueError(f"未知 gain regression loss_type: {loss_type}")

    # 正收益项给予更大权重，零收益项降权，避免模型被大量 0 淹没。
    entry_weights = torch.where(
        target_gains > 0,
        positive_gain_weight,
        torch.full_like(target_gains, float(zero_gain_weight)),
    )
    weighted_loss = per_entry_loss * entry_weights
    weighted_loss = weighted_loss.masked_fill(~feasible_mask, 0.0)

    sample_denominator = torch.sum(entry_weights.masked_fill(~feasible_mask, 0.0), dim=1)
    sample_numerator = torch.sum(weighted_loss, dim=1)
    sample_losses = sample_numerator / torch.clamp(sample_denominator, min=eps)
    loss = torch.sum(sample_losses * sample_weights) / torch.clamp(torch.sum(sample_weights), min=eps)

    stats = {
        "positive_gain_rate": float(torch.mean((target_gains > 0).to(torch.float32)).detach().cpu().item()),
        "mean_target_gain": float(torch.mean(target_gains.masked_fill(~feasible_mask, 0.0)).detach().cpu().item()),
        "mean_positive_target_gain": float(
            target_gains[target_gains > 0].mean().detach().cpu().item() if torch.any(target_gains > 0) else 0.0
        ),
        "mean_sample_weight": float(torch.mean(sample_weights).detach().cpu().item()),
    }
    return loss, stats


def one_vs_default_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
    negative_weight: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    候选级 beat-default 二分类损失。

    含义：
    - 每个候选 arm 都是一个独立二分类任务
    - label=1 表示“该 arm 在这个实例上优于默认 arm”
    - label=0 表示“不优于默认 arm”

    与 gain 回归相比，这个目标更直接：
    - 不要求模型拟合精确的 gain 数值
    - 只要求先学会“该不该切”
    """
    if sample_weights is None:
        sample_weights = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)
    else:
        sample_weights = sample_weights.to(device=logits.device, dtype=logits.dtype)

    if positive_weights is None:
        positive_weights = torch.ones_like(labels)
    else:
        positive_weights = positive_weights.to(device=logits.device, dtype=logits.dtype)

    entry_weights = torch.where(
        labels > 0.5,
        positive_weights,
        torch.full_like(labels, float(negative_weight)),
    )
    per_entry_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weighted_loss = per_entry_loss * entry_weights
    weighted_loss = weighted_loss.masked_fill(~candidate_mask, 0.0)

    sample_denominator = torch.sum(entry_weights.masked_fill(~candidate_mask, 0.0), dim=1)
    sample_numerator = torch.sum(weighted_loss, dim=1)
    sample_losses = sample_numerator / torch.clamp(sample_denominator, min=eps)
    loss = torch.sum(sample_losses * sample_weights) / torch.clamp(torch.sum(sample_weights), min=eps)

    stats = {
        "positive_label_rate": float(torch.mean(labels.masked_fill(~candidate_mask, 0.0)).detach().cpu().item()),
        "mean_sample_weight": float(torch.mean(sample_weights).detach().cpu().item()),
        "mean_positive_weight": float(
            positive_weights[candidate_mask].mean().detach().cpu().item() if torch.any(candidate_mask) else 0.0
        ),
    }
    return loss, stats


def one_vs_default_margin_loss(
    scores: torch.Tensor,
    target_margins: torch.Tensor,
    candidate_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    entry_weights: torch.Tensor | None = None,
    negative_weight: float = 0.2,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    候选级 margin / pairwise policy loss。

    对每个候选 arm:
    - target_margin > 0: 该 arm 比默认 arm 更好
    - target_margin < 0: 默认 arm 更好
    - |target_margin|: 两者在相对 cost gap 上相差多少

    我们希望模型学到：
    - 更好的 arm 得到更大的正分数
    - 更差的 arm 得到更小的负分数
    - gap 越大，要求分离边界越明显

    这里采用平滑的 margin 形式：

        loss = softplus(margin - sign * score)

    其中：
    - sign = sign(target_margin)
    - margin = |target_margin|
    """
    if sample_weights is None:
        sample_weights = torch.ones(scores.shape[0], dtype=scores.dtype, device=scores.device)
    else:
        sample_weights = sample_weights.to(device=scores.device, dtype=scores.dtype)

    if entry_weights is None:
        entry_weights = torch.where(
            target_margins > 0,
            1.0 + torch.abs(target_margins),
            torch.full_like(target_margins, float(negative_weight)),
        )
    else:
        entry_weights = entry_weights.to(device=scores.device, dtype=scores.dtype)

    signed_direction = torch.where(target_margins > 0, torch.ones_like(target_margins), -torch.ones_like(target_margins))
    margin = torch.abs(target_margins)
    per_entry_loss = F.softplus(margin - signed_direction * scores)
    weighted_loss = per_entry_loss * entry_weights
    weighted_loss = weighted_loss.masked_fill(~candidate_mask, 0.0)

    sample_denominator = torch.sum(entry_weights.masked_fill(~candidate_mask, 0.0), dim=1)
    sample_numerator = torch.sum(weighted_loss, dim=1)
    sample_losses = sample_numerator / torch.clamp(sample_denominator, min=eps)
    loss = torch.sum(sample_losses * sample_weights) / torch.clamp(torch.sum(sample_weights), min=eps)

    positive_mask = candidate_mask & (target_margins > 0)
    negative_mask = candidate_mask & (target_margins <= 0)
    stats = {
        "positive_margin_rate": float(
            positive_mask.to(torch.float32).sum().detach().cpu().item() / max(candidate_mask.to(torch.float32).sum().detach().cpu().item(), 1.0)
        ),
        "mean_abs_margin": float(
            torch.abs(target_margins).masked_fill(~candidate_mask, 0.0).sum().detach().cpu().item()
            / max(candidate_mask.to(torch.float32).sum().detach().cpu().item(), 1.0)
        ),
        "mean_positive_margin": float(torch.abs(target_margins[positive_mask]).mean().detach().cpu().item() if torch.any(positive_mask) else 0.0),
        "mean_negative_score": float(scores[negative_mask].mean().detach().cpu().item() if torch.any(negative_mask) else 0.0),
        "mean_positive_score": float(scores[positive_mask].mean().detach().cpu().item() if torch.any(positive_mask) else 0.0),
        "mean_sample_weight": float(torch.mean(sample_weights).detach().cpu().item()),
    }
    return loss, stats


def one_vs_default_hard_rank_loss(
    scores: torch.Tensor,
    target_margins: torch.Tensor,
    candidate_mask: torch.Tensor,
    hard_sample_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    entry_weights: torch.Tensor | None = None,
    negative_weight: float = 0.2,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    hard-case candidate ranking 版本。

    目标分两部分：
    1. policy margin:
       继续学习“候选 arm 相对默认 arm 是正还是负”
    2. hard-case ranking:
       只在 default 真正不稳的样本上，
       再要求候选 arm 之间的排序与 target_margin 一致

    这样做的动机是：
    - 避免模型只学成“每个问题一个常数 winner”
    - 强制它在 hard cases 上分清不同候选臂
    """
    policy_loss, policy_stats = one_vs_default_margin_loss(
        scores=scores,
        target_margins=target_margins,
        candidate_mask=candidate_mask,
        sample_weights=sample_weights,
        entry_weights=entry_weights,
        negative_weight=negative_weight,
        eps=eps,
    )

    if sample_weights is None:
        sample_weights = torch.ones(scores.shape[0], dtype=scores.dtype, device=scores.device)
    else:
        sample_weights = sample_weights.to(device=scores.device, dtype=scores.dtype)

    weighted_rank_sum = scores.sum() * 0.0
    total_weight = scores.sum() * 0.0
    hard_sample_count = 0
    pair_count = 0
    margin_gap_sum = 0.0

    hard_sample_mask = hard_sample_mask.to(device=scores.device, dtype=torch.bool)

    for row_idx in range(scores.shape[0]):
        if not bool(hard_sample_mask[row_idx].detach().cpu().item()):
            continue

        valid_ids = torch.nonzero(candidate_mask[row_idx], as_tuple=False).flatten()
        if valid_ids.numel() <= 1:
            continue

        row_scores = scores[row_idx, valid_ids]
        row_targets = target_margins[row_idx, valid_ids]
        margin_diff = row_targets[:, None] - row_targets[None, :]
        pair_mask = margin_diff > eps
        if not torch.any(pair_mask):
            continue

        score_diff = row_scores[:, None] - row_scores[None, :]
        pair_weights = margin_diff[pair_mask]
        rank_loss_row = torch.sum(F.softplus(-score_diff[pair_mask]) * pair_weights) / torch.clamp(
            torch.sum(pair_weights), min=eps
        )

        row_weight = sample_weights[row_idx]
        weighted_rank_sum = weighted_rank_sum + rank_loss_row * row_weight
        total_weight = total_weight + row_weight
        hard_sample_count += 1
        pair_count += int(pair_mask.sum().detach().cpu().item())
        margin_gap_sum += float(torch.sum(pair_weights).detach().cpu().item())

    if hard_sample_count == 0:
        rank_loss = scores.sum() * 0.0
    else:
        rank_loss = weighted_rank_sum / torch.clamp(total_weight, min=eps)

    total_loss = policy_loss + rank_loss
    stats = {
        **policy_stats,
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "hard_sample_rate": float(
            hard_sample_mask.to(torch.float32).mean().detach().cpu().item() if hard_sample_mask.numel() > 0 else 0.0
        ),
        "hard_sample_count": int(hard_sample_mask.to(torch.int64).sum().detach().cpu().item()),
        "ranking_pair_count": int(pair_count),
        "avg_margin_gap_sum": float(margin_gap_sum / max(hard_sample_count, 1)),
    }
    return total_loss, stats


def one_vs_default_switch_rank_loss(
    switch_logits: torch.Tensor,
    alt_logits: torch.Tensor,
    target_margins: torch.Tensor,
    candidate_mask: torch.Tensor,
    switch_labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    switch_positive_weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    显式 stay/switch gate + 候选排序损失。

    训练目标被拆成两部分：
    1. gate loss:
       用 BCE 直接监督 “当前实例要不要 switch”
    2. rank loss:
       只在 switch 正样本上，对候选 arm 做 pairwise ranking

    这样比 margin-only 版本更贴近真实决策流程：
    - 先过“值不值得切”的门
    - 再决定“切到谁”
    """
    if sample_weights is None:
        sample_weights = torch.ones(switch_logits.shape[0], dtype=switch_logits.dtype, device=switch_logits.device)
    else:
        sample_weights = sample_weights.to(device=switch_logits.device, dtype=switch_logits.dtype)

    if switch_positive_weights is None:
        switch_positive_weights = torch.ones_like(switch_labels, dtype=switch_logits.dtype, device=switch_logits.device)
    else:
        switch_positive_weights = switch_positive_weights.to(device=switch_logits.device, dtype=switch_logits.dtype)

    switch_labels = switch_labels.to(device=switch_logits.device, dtype=switch_logits.dtype)
    gate_entry_weights = torch.where(
        switch_labels > 0.5,
        switch_positive_weights,
        torch.ones_like(switch_labels, dtype=switch_logits.dtype, device=switch_logits.device),
    )
    gate_loss_per_sample = F.binary_cross_entropy_with_logits(switch_logits, switch_labels, reduction="none")
    gate_weights = gate_entry_weights * sample_weights
    gate_loss = torch.sum(gate_loss_per_sample * gate_weights) / torch.clamp(torch.sum(gate_weights), min=eps)

    weighted_rank_sum = alt_logits.sum() * 0.0
    total_rank_weight = alt_logits.sum() * 0.0
    positive_sample_count = 0
    pair_count = 0
    margin_gap_sum = 0.0

    for row_idx in range(alt_logits.shape[0]):
        if float(switch_labels[row_idx].detach().cpu().item()) <= 0.5:
            continue

        valid_ids = torch.nonzero(candidate_mask[row_idx], as_tuple=False).flatten()
        if valid_ids.numel() <= 1:
            continue

        row_scores = alt_logits[row_idx, valid_ids]
        row_targets = target_margins[row_idx, valid_ids]
        margin_diff = row_targets[:, None] - row_targets[None, :]
        pair_mask = margin_diff > eps
        if not torch.any(pair_mask):
            continue

        score_diff = row_scores[:, None] - row_scores[None, :]
        pair_weights = margin_diff[pair_mask]
        rank_loss_row = torch.sum(F.softplus(-score_diff[pair_mask]) * pair_weights) / torch.clamp(
            torch.sum(pair_weights),
            min=eps,
        )

        row_weight = sample_weights[row_idx] * gate_entry_weights[row_idx]
        weighted_rank_sum = weighted_rank_sum + rank_loss_row * row_weight
        total_rank_weight = total_rank_weight + row_weight
        positive_sample_count += 1
        pair_count += int(pair_mask.sum().detach().cpu().item())
        margin_gap_sum += float(torch.sum(pair_weights).detach().cpu().item())

    if positive_sample_count == 0:
        rank_loss = alt_logits.sum() * 0.0
    else:
        rank_loss = weighted_rank_sum / torch.clamp(total_rank_weight, min=eps)

    total_loss = gate_loss + rank_loss
    switch_probs = torch.sigmoid(switch_logits)
    stats = {
        "gate_loss": float(gate_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "switch_positive_rate": float(torch.mean((switch_labels > 0.5).to(torch.float32)).detach().cpu().item()),
        "mean_switch_prob": float(torch.mean(switch_probs).detach().cpu().item()),
        "mean_positive_switch_prob": float(
            switch_probs[switch_labels > 0.5].mean().detach().cpu().item() if torch.any(switch_labels > 0.5) else 0.0
        ),
        "mean_negative_switch_prob": float(
            switch_probs[switch_labels <= 0.5].mean().detach().cpu().item() if torch.any(switch_labels <= 0.5) else 0.0
        ),
        "mean_gate_positive_weight": float(
            gate_entry_weights[switch_labels > 0.5].mean().detach().cpu().item()
            if torch.any(switch_labels > 0.5)
            else 0.0
        ),
        "ranking_pair_count": int(pair_count),
        "avg_margin_gap_sum": float(margin_gap_sum / max(positive_sample_count, 1)),
    }
    return total_loss, stats


def one_vs_default_switch_advantage_loss(
    switch_scores: torch.Tensor,
    alt_logits: torch.Tensor,
    target_margins: torch.Tensor,
    candidate_mask: torch.Tensor,
    switch_targets: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    switch_positive_weights: torch.Tensor | None = None,
    zero_weight: float = 1.0,
    gate_loss_type: str = "smooth_l1",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """
    显式 switch advantage 回归 + 候选排序损失。

    1. gate:
       直接回归 “默认臂相对最强候选臂能提升多少”
    2. rank:
       只在 switch_target > 0 的样本上，对候选臂做排序
    """
    if sample_weights is None:
        sample_weights = torch.ones(switch_scores.shape[0], dtype=switch_scores.dtype, device=switch_scores.device)
    else:
        sample_weights = sample_weights.to(device=switch_scores.device, dtype=switch_scores.dtype)

    if switch_positive_weights is None:
        switch_positive_weights = torch.ones_like(switch_targets, dtype=switch_scores.dtype, device=switch_scores.device)
    else:
        switch_positive_weights = switch_positive_weights.to(device=switch_scores.device, dtype=switch_scores.dtype)

    switch_targets = switch_targets.to(device=switch_scores.device, dtype=switch_scores.dtype)
    gate_entry_weights = torch.where(
        switch_targets > eps,
        switch_positive_weights,
        torch.full_like(switch_targets, float(zero_weight)),
    )
    if gate_loss_type == "smooth_l1":
        gate_loss_per_sample = F.smooth_l1_loss(switch_scores, switch_targets, reduction="none")
    elif gate_loss_type == "mse":
        gate_loss_per_sample = (switch_scores - switch_targets) ** 2
    else:
        raise ValueError(f"未知 gate_loss_type: {gate_loss_type}")
    gate_weights = gate_entry_weights * sample_weights
    gate_loss = torch.sum(gate_loss_per_sample * gate_weights) / torch.clamp(torch.sum(gate_weights), min=eps)

    weighted_rank_sum = alt_logits.sum() * 0.0
    total_rank_weight = alt_logits.sum() * 0.0
    positive_sample_count = 0
    pair_count = 0
    margin_gap_sum = 0.0

    for row_idx in range(alt_logits.shape[0]):
        if float(switch_targets[row_idx].detach().cpu().item()) <= eps:
            continue

        valid_ids = torch.nonzero(candidate_mask[row_idx], as_tuple=False).flatten()
        if valid_ids.numel() <= 1:
            continue

        row_scores = alt_logits[row_idx, valid_ids]
        row_targets = target_margins[row_idx, valid_ids]
        margin_diff = row_targets[:, None] - row_targets[None, :]
        pair_mask = margin_diff > eps
        if not torch.any(pair_mask):
            continue

        score_diff = row_scores[:, None] - row_scores[None, :]
        pair_weights = margin_diff[pair_mask]
        rank_loss_row = torch.sum(F.softplus(-score_diff[pair_mask]) * pair_weights) / torch.clamp(
            torch.sum(pair_weights),
            min=eps,
        )

        row_weight = sample_weights[row_idx] * gate_entry_weights[row_idx]
        weighted_rank_sum = weighted_rank_sum + rank_loss_row * row_weight
        total_rank_weight = total_rank_weight + row_weight
        positive_sample_count += 1
        pair_count += int(pair_mask.sum().detach().cpu().item())
        margin_gap_sum += float(torch.sum(pair_weights).detach().cpu().item())

    if positive_sample_count == 0:
        rank_loss = alt_logits.sum() * 0.0
    else:
        rank_loss = weighted_rank_sum / torch.clamp(total_rank_weight, min=eps)

    total_loss = gate_loss + rank_loss
    positive_mask = switch_targets > eps
    negative_mask = ~positive_mask
    stats = {
        "gate_loss": float(gate_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "switch_positive_rate": float(torch.mean(positive_mask.to(torch.float32)).detach().cpu().item()),
        "mean_switch_score": float(torch.mean(switch_scores).detach().cpu().item()),
        "mean_positive_switch_score": float(
            switch_scores[positive_mask].mean().detach().cpu().item() if torch.any(positive_mask) else 0.0
        ),
        "mean_negative_switch_score": float(
            switch_scores[negative_mask].mean().detach().cpu().item() if torch.any(negative_mask) else 0.0
        ),
        "mean_switch_target": float(torch.mean(switch_targets).detach().cpu().item()),
        "mean_positive_switch_target": float(
            switch_targets[positive_mask].mean().detach().cpu().item() if torch.any(positive_mask) else 0.0
        ),
        "mean_gate_positive_weight": float(
            gate_entry_weights[positive_mask].mean().detach().cpu().item() if torch.any(positive_mask) else 0.0
        ),
        "ranking_pair_count": int(pair_count),
        "avg_margin_gap_sum": float(margin_gap_sum / max(positive_sample_count, 1)),
    }
    return total_loss, stats


def compute_ranking_loss(
    *,
    loss_name: str,
    scores: torch.Tensor,
    rewards: torch.Tensor,
    feasible_mask: torch.Tensor,
    normalized_regrets: torch.Tensor | None = None,
    best_arms: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    switch_logits: torch.Tensor | None = None,
    alt_logits: torch.Tensor | None = None,
    default_arms: torch.Tensor | None = None,
    default_improvements: torch.Tensor | None = None,
    switch_thresholds: torch.Tensor | None = None,
    target_gains: torch.Tensor | None = None,
    positive_gain_weight: torch.Tensor | None = None,
    hard_sample_mask: torch.Tensor | None = None,
    switch_labels: torch.Tensor | None = None,
    switch_positive_weights: torch.Tensor | None = None,
    switch_targets: torch.Tensor | None = None,
    zero_gain_weight: float = 0.2,
    gain_loss_type: str = "smooth_l1",
    pairwise_gap_weight: bool = True,
    listwise_target_scale: float = 8.0,
) -> tuple[torch.Tensor, dict]:
    """
    统一的 ranking loss 分发入口。
    """
    if loss_name == "pairwise_logistic":
        loss, stats = pairwise_logistic_loss(
            scores=scores,
            rewards=rewards,
            feasible_mask=feasible_mask,
            gap_weight=pairwise_gap_weight,
        )
    elif loss_name == "listwise_ce":
        loss, stats = listwise_cross_entropy_loss(
            scores=scores,
            rewards=rewards,
            feasible_mask=feasible_mask,
            target_scale=listwise_target_scale,
        )
    elif loss_name == "best_vs_all_regret":
        if normalized_regrets is None:
            raise ValueError("best_vs_all_regret 需要 normalized_regrets")
        if best_arms is None:
            raise ValueError("best_vs_all_regret 需要 best_arms")
        loss, stats = best_vs_all_regret_loss(
            scores=scores,
            normalized_regrets=normalized_regrets,
            feasible_mask=feasible_mask,
            best_arms=best_arms,
            sample_weights=sample_weights,
        )
    elif loss_name == "two_stage_default_gate":
        if switch_logits is None or alt_logits is None:
            raise ValueError("two_stage_default_gate 需要 switch_logits 和 alt_logits")
        if default_arms is None:
            raise ValueError("two_stage_default_gate 需要 default_arms")
        if best_arms is None:
            raise ValueError("two_stage_default_gate 需要 best_arms")
        if default_improvements is None or switch_thresholds is None:
            raise ValueError("two_stage_default_gate 需要 default_improvements 和 switch_thresholds")
        loss, stats = two_stage_default_gate_loss(
            switch_logits=switch_logits,
            alt_logits=alt_logits,
            feasible_mask=feasible_mask,
            default_arms=default_arms,
            best_arms=best_arms,
            default_improvements=default_improvements,
            switch_thresholds=switch_thresholds,
            sample_weights=sample_weights,
        )
    elif loss_name == "one_vs_default_two_stage":
        if switch_logits is None or alt_logits is None:
            raise ValueError("one_vs_default_two_stage 需要 switch_logits 和 alt_logits")
        if default_arms is None:
            raise ValueError("one_vs_default_two_stage 需要 default_arms")
        if best_arms is None:
            raise ValueError("one_vs_default_two_stage 需要 best_arms 作为候选最优臂")
        if default_improvements is None or switch_thresholds is None:
            raise ValueError("one_vs_default_two_stage 需要 default_improvements 和 switch_thresholds")
        loss, stats = two_stage_default_gate_loss(
            switch_logits=switch_logits,
            alt_logits=alt_logits,
            feasible_mask=feasible_mask,
            default_arms=default_arms,
            best_arms=best_arms,
            default_improvements=default_improvements,
            switch_thresholds=switch_thresholds,
            sample_weights=sample_weights,
        )
    elif loss_name == "default_gain_regression":
        if target_gains is None:
            raise ValueError("default_gain_regression 需要 target_gains")
        loss, stats = default_gain_regression_loss(
            scores=scores,
            target_gains=target_gains,
            feasible_mask=feasible_mask,
            sample_weights=sample_weights,
            positive_gain_weight=positive_gain_weight,
            zero_gain_weight=zero_gain_weight,
            loss_type=gain_loss_type,
        )
    elif loss_name == "one_vs_default_gain":
        if target_gains is None:
            raise ValueError("one_vs_default_gain 需要 target_gains")
        loss, stats = default_gain_regression_loss(
            scores=scores,
            target_gains=target_gains,
            feasible_mask=feasible_mask,
            sample_weights=sample_weights,
            positive_gain_weight=positive_gain_weight,
            zero_gain_weight=zero_gain_weight,
            loss_type=gain_loss_type,
        )
    elif loss_name == "one_vs_default_beat":
        if target_gains is None:
            raise ValueError("one_vs_default_beat 需要 target_gains 作为 0/1 labels")
        loss, stats = one_vs_default_binary_loss(
            logits=scores,
            labels=target_gains,
            candidate_mask=feasible_mask,
            sample_weights=sample_weights,
            positive_weights=positive_gain_weight,
            negative_weight=zero_gain_weight,
        )
    elif loss_name == "one_vs_default_margin":
        if target_gains is None:
            raise ValueError("one_vs_default_margin 需要 target_gains 作为 signed margins")
        loss, stats = one_vs_default_margin_loss(
            scores=scores,
            target_margins=target_gains,
            candidate_mask=feasible_mask,
            sample_weights=sample_weights,
            entry_weights=positive_gain_weight,
            negative_weight=zero_gain_weight,
        )
    elif loss_name == "one_vs_default_hard_rank":
        if target_gains is None:
            raise ValueError("one_vs_default_hard_rank 需要 target_gains 作为 signed margins")
        if hard_sample_mask is None:
            raise ValueError("one_vs_default_hard_rank 需要 hard_sample_mask")
        loss, stats = one_vs_default_hard_rank_loss(
            scores=scores,
            target_margins=target_gains,
            candidate_mask=feasible_mask,
            hard_sample_mask=hard_sample_mask,
            sample_weights=sample_weights,
            entry_weights=positive_gain_weight,
            negative_weight=zero_gain_weight,
        )
    elif loss_name in ("one_vs_default_switch_rank", "one_vs_default_coupled_switch_rank"):
        if switch_logits is None:
            raise ValueError(f"{loss_name} 需要 switch_logits")
        if alt_logits is None:
            raise ValueError(f"{loss_name} 需要 alt_logits")
        if target_gains is None:
            raise ValueError(f"{loss_name} 需要 target_gains 作为 candidate margins")
        if switch_labels is None:
            raise ValueError(f"{loss_name} 需要 switch_labels")
        loss, stats = one_vs_default_switch_rank_loss(
            switch_logits=switch_logits,
            alt_logits=alt_logits,
            target_margins=target_gains,
            candidate_mask=feasible_mask,
            switch_labels=switch_labels,
            sample_weights=sample_weights,
            switch_positive_weights=switch_positive_weights,
        )
    elif loss_name == "one_vs_default_switch_advantage":
        if switch_logits is None:
            raise ValueError("one_vs_default_switch_advantage 需要 switch_logits")
        if alt_logits is None:
            raise ValueError("one_vs_default_switch_advantage 需要 alt_logits")
        if target_gains is None:
            raise ValueError("one_vs_default_switch_advantage 需要 target_gains 作为 candidate margins")
        if switch_targets is None:
            raise ValueError("one_vs_default_switch_advantage 需要 switch_targets")
        loss, stats = one_vs_default_switch_advantage_loss(
            switch_scores=switch_logits,
            alt_logits=alt_logits,
            target_margins=target_gains,
            candidate_mask=feasible_mask,
            switch_targets=switch_targets,
            sample_weights=sample_weights,
            switch_positive_weights=switch_positive_weights,
            zero_weight=max(float(zero_gain_weight), 1e-6),
            gate_loss_type=gain_loss_type,
        )
    else:
        raise ValueError(f"未知 loss_name: {loss_name}")

    return loss, {"loss_name": loss_name, **stats}
