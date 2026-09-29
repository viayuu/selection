import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import PROBLEMS

from .V4Model import ProblemToSolverSelector, get_default_model_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(problem, split, batch_size, num_workers, coord_augment=0, shuffle=True):
    dataset = UnifiedProblemDataset(problem, split, coord_augment=coord_augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_single_problem,
        drop_last=(split == "train"),
        pin_memory=(num_workers > 0),
        persistent_workers=False,
    )


def get_split_sbs_vbs(problem, split):
    dataset = UnifiedProblemDataset(problem, split, coord_augment=0)
    costs = []
    for i in range(dataset.base_N):
        costs.append(np.asarray(dataset.labels[str(i)]["cost"][: dataset.K_p], dtype=np.float32))
    costs = np.stack(costs, axis=0)
    method_mean = costs.mean(axis=0)
    return float(method_mean.min()), float(costs.min(axis=1).mean())


def build_sbs_indices(problems, split="train"):
    indices = {}
    for problem in problems:
        dataset = UnifiedProblemDataset(problem, split, coord_augment=0)
        costs = []
        for i in range(dataset.base_N):
            costs.append(np.asarray(dataset.labels[str(i)]["cost"][: dataset.K_p], dtype=np.float32))
        costs = np.stack(costs, axis=0)
        indices[problem] = int(costs.mean(axis=0).argmin())
    return indices


def build_winner_weights(problems, max_weight=3.0):
    weights = {}
    for problem in problems:
        ds = UnifiedProblemDataset(problem, "train", coord_augment=0)
        counts = np.zeros(ds.K_p, dtype=np.float64)
        for i in range(ds.base_N):
            c = np.asarray(ds.labels[str(i)]["cost"][: ds.K_p], dtype=np.float32)
            counts[int(c.argmin())] += 1
        positive = counts[counts > 0]
        mean_freq = positive.mean() if len(positive) else 1.0
        w = np.sqrt(mean_freq / np.maximum(counts, 1.0))
        w = np.clip(w, 0.25, max_weight).astype(np.float32)
        weights[problem] = torch.tensor(w, dtype=torch.float32)
    return weights


def pairwise_order_loss(score, costs):
    diff_score = score[:, :, None] - score[:, None, :]
    diff_cost = costs[:, None, :] - costs[:, :, None]
    sign = diff_cost.sign()
    norm = costs.abs().mean(dim=1, keepdim=True)[:, None, :].clamp_min(1.0e-9)
    weight = (diff_cost.abs() / norm).clamp(0.05, 5.0)
    valid = sign != 0
    loss = F.softplus(-sign * diff_score) * weight
    return loss[valid].mean() if valid.any() else score.sum() * 0


def top_focused_pair_loss(score, costs, sbs_idx=None):
    diff_score = score[:, :, None] - score[:, None, :]
    diff_cost = costs[:, None, :] - costs[:, :, None]
    rank = costs.argsort(dim=1)
    topk = min(3, costs.size(1))

    top_mask = torch.zeros_like(costs, dtype=torch.bool)
    top_mask.scatter_(1, rank[:, :topk], True)
    best_mask = torch.zeros_like(costs, dtype=torch.bool)
    best_mask.scatter_(1, rank[:, :1], True)

    pred = score.argmax(dim=1, keepdim=True)
    pred_mask = torch.zeros_like(costs, dtype=torch.bool)
    pred_mask.scatter_(1, pred, True)
    if sbs_idx is None:
        sbs = costs.mean(dim=0).argmin().view(1, 1).expand(costs.size(0), 1)
    else:
        sbs = torch.full((costs.size(0), 1), int(sbs_idx), dtype=torch.long, device=costs.device)
    sbs_mask = torch.zeros_like(costs, dtype=torch.bool)
    sbs_mask.scatter_(1, sbs, True)

    important_pair = (top_mask[:, :, None] & ~top_mask[:, None, :]) | (
        best_mask[:, :, None] & (pred_mask | sbs_mask)[:, None, :]
    )
    valid = important_pair & (diff_cost > 0)
    norm = costs.abs().mean(dim=1, keepdim=True)[:, None, :].clamp_min(1.0e-9)
    weight = (diff_cost.abs() / norm).clamp(0.25, 10.0)
    loss = F.softplus(-diff_score) * weight
    return loss[valid].mean() if valid.any() else score.sum() * 0


def parse_rank_weights(text):
    return [float(x) for x in text.split(",") if x.strip()]


def ranked_topk_ce_loss(logits, costs, rank_weights):
    rank = costs.argsort(dim=1)
    total = logits.sum() * 0
    denom = 0.0
    for k, weight in enumerate(rank_weights):
        if k >= logits.size(1):
            break
        total = total + float(weight) * F.cross_entropy(logits, rank[:, k])
        denom += float(weight)
    return total / max(denom, 1.0e-9)


def sequential_topk_ce_loss(logits, costs, rank_weights):
    rank = costs.argsort(dim=1)
    available = torch.ones_like(logits, dtype=torch.bool)
    total = logits.sum() * 0
    denom = 0.0
    for k, weight in enumerate(rank_weights):
        if k >= logits.size(1):
            break
        target = rank[:, k]
        masked_logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        total = total + float(weight) * F.cross_entropy(masked_logits, target)
        available.scatter_(1, target[:, None], False)
        denom += float(weight)
    return total / max(denom, 1.0e-9)


def gap_loss(pred_gap, costs):
    best = costs.min(dim=1, keepdim=True).values
    target = ((costs - best) / best.abs().clamp_min(1.0e-9)).clamp(max=1.0)
    return F.huber_loss(pred_gap, target, delta=0.05)


def risk_loss(score, costs):
    prob = F.softmax(score, dim=1)
    pred_cost = (prob * costs).sum(dim=1)
    best = costs.min(dim=1).values
    return ((pred_cost - best) / best.abs().clamp_min(1.0e-9)).clamp(max=1.0).mean()


def support_targets_from_costs(costs, eps=0.01, topk=3):
    best = costs.min(dim=1, keepdim=True).values
    rel_gap = (costs - best) / best.abs().clamp_min(1.0e-9)
    target = rel_gap <= eps
    if topk > 0:
        kk = min(max(1, topk), costs.size(1))
        idx = costs.topk(kk, dim=1, largest=False).indices
        top_mask = torch.zeros_like(target)
        top_mask.scatter_(1, idx, True)
        target = target | top_mask
    return target.to(costs.dtype)


def support_rank_partition_targets(costs, generators, eps=0.01, topk=3, target_mode="rank_partition"):
    if generators <= 0:
        raise ValueError("generators must be positive")
    base_target = support_targets_from_costs(costs, eps=eps, topk=topk)
    if target_mode != "rank_partition" or generators == 1:
        return base_target[:, None, :].expand(-1, generators, -1).contiguous()

    batch, pool = costs.shape
    targets = costs.new_zeros(batch, generators, pool)
    kk = min(max(1, topk), pool)
    top_idx = costs.topk(kk, dim=1, largest=False).indices
    top_mask = torch.zeros_like(base_target)
    top_mask.scatter_(1, top_idx, 1.0)
    for g in range(generators):
        if g < kk:
            targets[:, g].scatter_(1, top_idx[:, g : g + 1], 1.0)
        elif g == kk:
            targets[:, g] = (base_target - top_mask).clamp_min(0.0)
    return targets


def support_branch_loss(
    support_logits,
    costs,
    eps=0.01,
    topk=3,
    gamma_neg=4.0,
    pos_weight=2.0,
    diversity_weight=0.0,
    budget_weight=0.0,
    target_mode="rank_partition",
):
    if support_logits is None:
        zero = costs.sum() * 0
        return zero, {"support": 0.0, "support_bce": 0.0, "support_div": 0.0, "support_budget": 0.0}
    if support_logits.dim() == 2:
        support_logits = support_logits[:, None, :]
    targets = support_rank_partition_targets(
        costs,
        support_logits.size(1),
        eps=eps,
        topk=topk,
        target_mode=target_mode,
    )
    probs = torch.sigmoid(support_logits)
    bce = F.binary_cross_entropy_with_logits(support_logits, targets, reduction="none")
    pos_scale = costs.new_ones(support_logits.size(1))
    if target_mode == "rank_partition" and support_logits.size(1) > 1:
        pos_scale = costs.new_tensor([1.75 if g == 0 else 1.5 if g < topk else 1.0 for g in range(support_logits.size(1))])
    pos_w = pos_weight * pos_scale.view(1, -1, 1)
    neg_w = probs.pow(gamma_neg)
    weights = torch.where(targets > 0.5, pos_w, neg_w)
    bce_loss = (bce * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    bce_loss = bce_loss.mean()

    budget_loss = costs.sum() * 0
    if budget_weight > 0:
        target_ratio = targets.mean(dim=2)
        budget_loss = (probs.mean(dim=2) - target_ratio).pow(2).mean()

    div_loss = costs.sum() * 0
    if diversity_weight > 0 and support_logits.size(1) > 1:
        overlaps = []
        for i in range(support_logits.size(1)):
            for j in range(i + 1, support_logits.size(1)):
                overlaps.append((probs[:, i] * probs[:, j]).mean())
        div_loss = torch.stack(overlaps).mean() if overlaps else div_loss

    loss = bce_loss + diversity_weight * div_loss + budget_weight * budget_loss
    return loss, {
        "support": float(loss.detach().cpu()),
        "support_bce": float(bce_loss.detach().cpu()),
        "support_div": float(div_loss.detach().cpu()),
        "support_budget": float(budget_loss.detach().cpu()),
    }


def query_diversity_loss(query_logits):
    attn = F.softmax(query_logits, dim=-1)
    sim = torch.matmul(attn, attn.transpose(1, 2))
    r = sim.size(1)
    off_diag = sim.masked_select(~torch.eye(r, dtype=torch.bool, device=sim.device)[None, :, :])
    return off_diag.mean() if off_diag.numel() else query_logits.sum() * 0


def score_correlation(a, b):
    a = a.reshape(a.size(0), -1)
    b = b.reshape(b.size(0), -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = a.norm(dim=1) * b.norm(dim=1)
    corr = (a * b).sum(dim=1) / denom.clamp_min(1.0e-9)
    return corr.mean().item()


def support_recall_stats(support_logits, costs, topk=3):
    if support_logits is None:
        return None
    if support_logits.dim() == 2:
        support_logits = support_logits[:, None, :]
    true_rank = torch.argsort(costs, dim=1)
    best = true_rank[:, 0]
    true_top3 = true_rank[:, : min(3, costs.size(1))]
    probs = torch.sigmoid(support_logits)

    def union_stats(k):
        kk = min(max(1, k), support_logits.size(-1))
        top_idx = probs.topk(kk, dim=2).indices
        shortlist = torch.zeros(costs.size(0), costs.size(1), dtype=torch.bool, device=costs.device)
        for g in range(support_logits.size(1)):
            shortlist.scatter_(1, top_idx[:, g], True)
        return dict(
            top1=shortlist.gather(1, best[:, None]).float().sum().item(),
            top3=shortlist.gather(1, true_top3).any(dim=1).float().sum().item(),
            arm_seen=shortlist.any(dim=0).detach().cpu(),
            size=shortlist.float().sum(dim=1).mean().item(),
        )

    union1 = union_stats(1)
    union2 = union_stats(2)
    union3 = union_stats(topk)
    g0_top1 = probs[:, 0].argmax(dim=1).eq(best).float().sum().item()
    return {
        "top1": union3["top1"],
        "top3": union3["top3"],
        "arm_seen": union3["arm_seen"],
        "g0_top1": g0_top1,
        "union1_top1": union1["top1"],
        "union2_top1": union2["top1"],
        "union_size": union3["size"],
    }


def selector_loss(out, costs, class_weight, args, sbs_idx=None):
    logits = out["logits"]
    winner = costs.argmin(dim=1)
    ce = F.cross_entropy(logits, winner, weight=class_weight)
    pair = top_focused_pair_loss(logits, costs, sbs_idx=sbs_idx) if args.top_focused_pair else pairwise_order_loss(logits, costs)
    rank_weights = parse_rank_weights(args.topk_ce_rank_weights)
    if args.topk_ce_mode == "sequential":
        topk_ce = sequential_topk_ce_loss(logits, costs, rank_weights)
    else:
        topk_ce = ranked_topk_ce_loss(logits, costs, rank_weights)
    gap = gap_loss(out["pred_gap"], costs)
    pre = F.cross_entropy(out["pre_score"], winner, weight=class_weight)
    div = query_diversity_loss(out["query_logits"])
    risk = risk_loss(logits, costs)
    loss = (
        args.ce_weight * ce
        + args.pair_weight * pair
        + args.gap_weight * gap
        + args.pre_ce_weight * pre
        + args.topk_ce_weight * topk_ce
        + args.query_div_weight * div
        + args.risk_weight * risk
    )
    support_parts = {}
    if args.support_loss_weight > 0 and out.get("support_logits") is not None:
        support, support_parts = support_branch_loss(
            out["support_logits"],
            costs,
            eps=args.support_eps,
            topk=args.support_topk,
            gamma_neg=args.support_focal_gamma_neg,
            pos_weight=args.support_pos_weight,
            diversity_weight=args.support_diversity_weight,
            budget_weight=args.support_budget_weight,
            target_mode=args.support_target_mode,
        )
        loss = loss + args.support_loss_weight * support
    parts = {
        "ce": ce.item(),
        "pair": pair.item(),
        "gap": gap.item(),
        "pre": pre.item(),
        "topk_ce": topk_ce.item(),
        "div": div.item(),
        "risk": risk.item(),
    }
    parts.update(support_parts)
    return loss, parts


@torch.no_grad()
def evaluate(model, problems, split, batch_size, num_workers, device):
    model.eval()
    per_problem = {}
    for problem in problems:
        loader = make_loader(problem, split, batch_size, num_workers, coord_augment=0, shuffle=False)
        comp = {
            "final": {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0},
            "base": {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0},
            "pre": {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0},
            "gap": {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0},
            "utility": {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0},
            "support_g0": {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0},
        }
        n_total = 0
        pick_count = None
        support_top1 = 0.0
        support_top3 = 0.0
        support_g0_top1 = 0.0
        support_union1_top1 = 0.0
        support_union2_top1 = 0.0
        support_union_size_sum = 0.0
        support_arm_seen = None
        std_meter = {k: [] for k in comp}
        corr_meter = {
            "base_final": [],
            "pre_final": [],
            "gap_final": [],
            "utility_final": [],
            "support_g0_final": [],
        }
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            scores = {
                "final": out["logits"],
                "base": out.get("base_logits", out["logits"]),
                "pre": out["pre_score"],
                "gap": -out["pred_gap"],
                "utility": out["utility"],
            }
            if out.get("support_logits") is not None:
                support_logits = out["support_logits"]
                if support_logits.dim() == 2:
                    support_logits = support_logits[:, None, :]
                scores["support_g0"] = support_logits[:, 0]
            costs = batch["costs"]
            true_rank = torch.argsort(costs, dim=1)
            best = true_rank[:, 0]
            for name, score in scores.items():
                pred_rank = torch.argsort(-score, dim=1)
                pred = pred_rank[:, 0]
                comp[name]["top1"] += (pred == best).sum().item()
                comp[name]["top2"] += (pred_rank[:, : min(2, score.size(1))] == best[:, None]).any(dim=1).sum().item()
                comp[name]["top3"] += (pred_rank[:, : min(3, score.size(1))] == best[:, None]).any(dim=1).sum().item()
                comp[name]["cost_sum"] += costs.gather(1, pred[:, None]).sum().item()
                std_meter[name].append(score.std(dim=1).mean().item())
            corr_meter["pre_final"].append(score_correlation(scores["pre"], scores["final"]))
            corr_meter["gap_final"].append(score_correlation(scores["gap"], scores["final"]))
            corr_meter["utility_final"].append(score_correlation(scores["utility"], scores["final"]))
            corr_meter["base_final"].append(score_correlation(scores["base"], scores["final"]))
            if "support_g0" in scores:
                corr_meter["support_g0_final"].append(score_correlation(scores["support_g0"], scores["final"]))
            n_total += costs.size(0)
            pred = torch.argsort(-scores["final"], dim=1)[:, 0]
            cur = torch.bincount(pred.detach().cpu(), minlength=scores["final"].size(1)).float()
            pick_count = cur if pick_count is None else pick_count + cur
            support_stats = support_recall_stats(out.get("support_logits"), costs, topk=3)
            if support_stats is not None:
                support_top1 += support_stats["top1"]
                support_top3 += support_stats["top3"]
                support_g0_top1 += support_stats["g0_top1"]
                support_union1_top1 += support_stats["union1_top1"]
                support_union2_top1 += support_stats["union2_top1"]
                support_union_size_sum += support_stats["union_size"] * costs.size(0)
                s_seen = support_stats["arm_seen"]
                support_arm_seen = s_seen if support_arm_seen is None else (support_arm_seen | s_seen)
        sbs, vbs = get_split_sbs_vbs(problem, split)
        mean_cost = comp["final"]["cost_sum"] / max(1, n_total)
        per_problem[problem] = dict(
            top1=comp["final"]["top1"] / n_total,
            top2=comp["final"]["top2"] / n_total,
            top3=comp["final"]["top3"] / n_total,
            mean_cost=mean_cost,
            sbs=sbs,
            vbs=vbs,
            vs_sbs_pct=(mean_cost - sbs) / (abs(sbs) + 1.0e-9) * 100,
            vbs_gap_closed_pct=((sbs - mean_cost) / (sbs - vbs + 1.0e-9)) * 100 if sbs != vbs else 0.0,
            pick_dist=(pick_count / pick_count.sum().clamp_min(1)).tolist(),
            n=n_total,
        )
        if support_arm_seen is not None:
            per_problem[problem]["support_top1_recall"] = support_top1 / max(1, n_total)
            per_problem[problem]["support_top3_recall"] = support_top3 / max(1, n_total)
            per_problem[problem]["support_g0_top1_recall"] = support_g0_top1 / max(1, n_total)
            per_problem[problem]["support_union1_top1_recall"] = support_union1_top1 / max(1, n_total)
            per_problem[problem]["support_union2_top1_recall"] = support_union2_top1 / max(1, n_total)
            per_problem[problem]["support_avg_shortlist"] = support_union_size_sum / max(1, n_total)
            per_problem[problem]["support_arm_coverage"] = float(support_arm_seen.float().mean().item())
        for name in sorted(k for k in comp if k != "final"):
            c = comp[name]
            mean_cost_name = c["cost_sum"] / max(1, n_total)
            per_problem[problem][f"top1_{name}"] = c["top1"] / n_total
            per_problem[problem][f"mean_cost_{name}"] = mean_cost_name
            per_problem[problem][f"vs_sbs_pct_{name}"] = (mean_cost_name - sbs) / (abs(sbs) + 1.0e-9) * 100
        for name, values in std_meter.items():
            per_problem[problem][f"std_{name}"] = float(np.mean(values)) if values else 0.0
        for name, values in corr_meter.items():
            per_problem[problem][f"corr_{name}"] = float(np.mean(values)) if values else 0.0
    macro = dict(
        macro_top1=float(np.mean([v["top1"] for v in per_problem.values()])),
        macro_top2=float(np.mean([v["top2"] for v in per_problem.values()])),
        macro_top3=float(np.mean([v["top3"] for v in per_problem.values()])),
        macro_vs_sbs_pct=float(np.mean([v["vs_sbs_pct"] for v in per_problem.values()])),
        macro_vbs_gap_closed_pct=float(np.mean([v["vbs_gap_closed_pct"] for v in per_problem.values()])),
        macro_top1_pre=float(np.mean([v["top1_pre"] for v in per_problem.values()])),
        macro_top1_gap=float(np.mean([v["top1_gap"] for v in per_problem.values()])),
        macro_top1_utility=float(np.mean([v["top1_utility"] for v in per_problem.values()])),
        macro_top1_base=float(np.mean([v["top1_base"] for v in per_problem.values()])),
        macro_top1_support_g0=float(np.mean([v["top1_support_g0"] for v in per_problem.values()])),
        macro_vs_sbs_pct_pre=float(np.mean([v["vs_sbs_pct_pre"] for v in per_problem.values()])),
        macro_vs_sbs_pct_gap=float(np.mean([v["vs_sbs_pct_gap"] for v in per_problem.values()])),
        macro_vs_sbs_pct_utility=float(np.mean([v["vs_sbs_pct_utility"] for v in per_problem.values()])),
        macro_corr_pre_final=float(np.mean([v["corr_pre_final"] for v in per_problem.values()])),
        macro_corr_gap_final=float(np.mean([v["corr_gap_final"] for v in per_problem.values()])),
        macro_corr_utility_final=float(np.mean([v["corr_utility_final"] for v in per_problem.values()])),
        macro_corr_base_final=float(np.mean([v["corr_base_final"] for v in per_problem.values()])),
        macro_corr_support_g0_final=float(np.mean([v["corr_support_g0_final"] for v in per_problem.values()])),
    )
    if all("support_top1_recall" in v for v in per_problem.values()):
        macro.update(
            macro_support_top1_recall=float(np.mean([v["support_top1_recall"] for v in per_problem.values()])),
            macro_support_top3_recall=float(np.mean([v["support_top3_recall"] for v in per_problem.values()])),
            macro_support_g0_top1_recall=float(np.mean([v["support_g0_top1_recall"] for v in per_problem.values()])),
            macro_support_union1_top1_recall=float(np.mean([v["support_union1_top1_recall"] for v in per_problem.values()])),
            macro_support_union2_top1_recall=float(np.mean([v["support_union2_top1_recall"] for v in per_problem.values()])),
            macro_support_avg_shortlist=float(np.mean([v["support_avg_shortlist"] for v in per_problem.values()])),
            macro_support_arm_coverage=float(np.mean([v["support_arm_coverage"] for v in per_problem.values()])),
        )
    model.train()
    return per_problem, macro


def print_eval(tag, per_problem, macro):
    print(
        f"[{tag}] macro_top1={macro['macro_top1']:.4f} top2={macro['macro_top2']:.4f} "
        f"top3={macro['macro_top3']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% "
        f"vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}%"
    )
    support_g0_msg = (
        f" support_g0_top1={macro['macro_top1_support_g0']:.4f}"
        if "macro_support_top1_recall" in macro
        else ""
    )
    print(
        f"          components: pre_top1={macro['macro_top1_pre']:.4f} "
        f"gap_top1={macro['macro_top1_gap']:.4f} utility_top1={macro['macro_top1_utility']:.4f} "
        f"base_top1={macro['macro_top1_base']:.4f}{support_g0_msg} "
        f"corr(pre/final)={macro['macro_corr_pre_final']:+.3f} "
        f"corr(gap/final)={macro['macro_corr_gap_final']:+.3f} "
        f"corr(util/final)={macro['macro_corr_utility_final']:+.3f}"
    )
    if "macro_support_top1_recall" in macro:
        print(
            f"          support: g0@1={macro['macro_support_g0_top1_recall']:.4f} "
            f"u1@1={macro['macro_support_union1_top1_recall']:.4f} "
            f"u2@1={macro['macro_support_union2_top1_recall']:.4f} "
            f"u3@1={macro['macro_support_top1_recall']:.4f} "
            f"top3_recall={macro['macro_support_top3_recall']:.4f} "
            f"avg_shortlist={macro['macro_support_avg_shortlist']:.2f} "
            f"arm_coverage={macro['macro_support_arm_coverage']:.4f}"
        )
    for p in per_problem:
        r = per_problem[p]
        print(
            f"{p:>10}: top1={r['top1']:.3f} mean_cost={r['mean_cost']:.4f} "
            f"(sbs={r['sbs']:.4f}) vs_sbs={r['vs_sbs_pct']:+.2f}% "
            f"pre={r['top1_pre']:.3f} gap={r['top1_gap']:.3f} util={r['top1_utility']:.3f}"
        )


def build_model(args):
    params = get_default_model_params()
    params.update(
        embedding_dim=args.d,
        head_num=args.heads,
        qkv_dim=args.d // args.heads,
        ff_hidden_dim=args.ff_hidden,
        head_hidden_dim=args.head_hidden,
        encoder_layer_num=args.encoder_layers,
        solver_set_layer_num=args.set_layers,
        query_num=args.query_num,
        dropout=args.dropout,
        rezero=args.rezero,
        gap_score_weight=args.gap_score_weight,
        pre_score_weight=args.pre_score_weight,
        support_branch=args.support_branch,
        support_generators=args.support_generators if args.support_branch else 0,
        support_hidden=args.support_hidden,
        support_score_weight=args.support_score_weight if args.support_branch else 0.0,
        support_score_mode=args.support_score_mode,
        support_g0_as_main=args.support_g0_as_main,
        support_feature_to_token=not args.no_support_feature_token,
        support_main_utility_weight=args.support_main_utility_weight,
        support_main_pre_weight=args.support_main_pre_weight,
        support_main_gap_weight=args.support_main_gap_weight,
    )
    return ProblemToSolverSelector(**params), params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-per-problem", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--wd", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coord-augment", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--problems", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--query-num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rezero", action="store_true")
    parser.add_argument("--ce-weight", type=float, default=0.35)
    parser.add_argument("--pair-weight", type=float, default=0.30)
    parser.add_argument("--gap-weight", type=float, default=0.20)
    parser.add_argument("--pre-ce-weight", type=float, default=0.10)
    parser.add_argument("--top-focused-pair", action="store_true")
    parser.add_argument("--topk-ce-weight", type=float, default=0.0)
    parser.add_argument("--topk-ce-rank-weights", default="1.0,0.4,0.2")
    parser.add_argument("--topk-ce-mode", choices=["legacy", "sequential"], default="legacy")
    parser.add_argument("--query-div-weight", type=float, default=0.03)
    parser.add_argument("--risk-weight", type=float, default=0.02)
    parser.add_argument("--gap-score-weight", type=float, default=0.25)
    parser.add_argument("--pre-score-weight", type=float, default=0.25)
    parser.add_argument("--winner-balance", action="store_true")
    parser.add_argument("--support-branch", action="store_true")
    parser.add_argument("--support-generators", type=int, default=4)
    parser.add_argument("--support-hidden", type=int, default=128)
    parser.add_argument("--support-score-weight", type=float, default=0.10)
    parser.add_argument("--support-score-mode", choices=["g0", "max"], default="g0")
    parser.add_argument("--support-g0-as-main", action="store_true")
    parser.add_argument("--no-support-feature-token", action="store_true")
    parser.add_argument("--support-main-utility-weight", type=float, default=0.0)
    parser.add_argument("--support-main-pre-weight", type=float, default=0.0)
    parser.add_argument("--support-main-gap-weight", type=float, default=0.0)
    parser.add_argument("--support-loss-weight", type=float, default=0.50)
    parser.add_argument("--support-eps", type=float, default=0.01)
    parser.add_argument("--support-topk", type=int, default=3)
    parser.add_argument("--support-focal-gamma-neg", type=float, default=4.0)
    parser.add_argument("--support-pos-weight", type=float, default=2.0)
    parser.add_argument("--support-diversity-weight", type=float, default=0.05)
    parser.add_argument("--support-budget-weight", type=float, default=0.02)
    parser.add_argument("--support-target-mode", choices=["legacy", "rank_partition"], default="rank_partition")
    args = parser.parse_args()

    set_seed(args.seed)
    configure_torch()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else list(PROBLEMS)

    model, model_params = build_model(args)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    start_epoch = 0
    best_score = -1.0e9
    best_top1 = -1.0e9
    best_top3_safe = -1.0e9
    best_cost = 1.0e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_score", best_score))

    train_loaders = {
        p: make_loader(p, "train", args.batch_per_problem, args.num_workers, args.coord_augment, shuffle=True)
        for p in problems
    }
    class_weights = build_winner_weights(problems) if args.winner_balance else {p: None for p in problems}
    sbs_indices = build_sbs_indices(problems, split="train")

    config = vars(args)
    config["model_params"] = model_params
    (save_dir / "args.json").write_text(json.dumps(config, indent=2))
    model_name = "V4 R32a R31c + R25-style support branch" if args.support_branch else "V4 R31c multi-query problem-to-solver attention + solver-set transformer"
    print(f"[train] {model_name}")
    print(f"[train] problems={problems}")
    print(f"[train] model_params={model_params}")

    def save_checkpoint(name, epoch, macro, score):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": config,
                "epoch": epoch,
                "macro": macro,
                "best_score": score,
            },
            save_dir / name,
        )

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    global_step = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        meter = []
        part_meter = {
            k: []
            for k in [
                "ce",
                "pair",
                "gap",
                "pre",
                "topk_ce",
                "div",
                "risk",
                "support",
                "support_bce",
                "support_div",
                "support_budget",
            ]
        }
        train_iters = {p: iter(train_loaders[p]) for p in problems}
        active = list(problems)
        while active:
            random.shuffle(active)
            for problem in list(active):
                try:
                    batch = next(train_iters[problem])
                except StopIteration:
                    active.remove(problem)
                    continue
                weight = class_weights[problem]
                weight = weight.to(args.device, non_blocking=True) if weight is not None else None
                batch = to_device(batch, args.device)
                with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
                    out = model(batch)
                    loss, parts = selector_loss(out, batch["costs"], weight, args, sbs_idx=sbs_indices.get(problem))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                meter.append(float(loss.detach().cpu()))
                for k, v in parts.items():
                    part_meter[k].append(v)
                if global_step % args.log_every == 0:
                    msg = " ".join([f"{k}={np.mean(v):.4f}" for k, v in part_meter.items() if v])
                    print(
                        f"ep{epoch:03d} step{global_step:06d} loss={np.mean(meter):.4f} "
                        f"{msg} time={(time.time() - start) / 60:.1f}m"
                    )

        if epoch % args.eval_every == 0:
            per_problem, macro = evaluate(model, problems, "val", args.batch_per_problem, args.num_workers, args.device)
            print_eval(f"eval epoch {epoch}", per_problem, macro)
            (save_dir / f"eval_epoch{epoch}.json").write_text(json.dumps({"per_problem": per_problem, "macro": macro}, indent=2))
            score = macro["macro_top1"] - 0.25 * max(0.0, macro["macro_vs_sbs_pct"])
            if score > best_score:
                best_score = score
                save_checkpoint("best.pt", epoch, macro, best_score)
                print(f"[save] best.pt epoch={epoch} score={score:.4f}")
            if macro["macro_top1"] > best_top1:
                best_top1 = macro["macro_top1"]
                save_checkpoint("best_top1.pt", epoch, macro, best_top1)
                print(f"[save] best_top1.pt epoch={epoch} top1={best_top1:.4f}")
            top3_safe_score = macro["macro_top3"] - 0.25 * max(0.0, macro["macro_vs_sbs_pct"])
            if top3_safe_score > best_top3_safe:
                best_top3_safe = top3_safe_score
                save_checkpoint("best_top3_safe.pt", epoch, macro, best_top3_safe)
                print(f"[save] best_top3_safe.pt epoch={epoch} score={best_top3_safe:.4f}")
            if macro["macro_vs_sbs_pct"] < best_cost:
                best_cost = macro["macro_vs_sbs_pct"]
                save_checkpoint("best_cost.pt", epoch, macro, best_cost)
                print(f"[save] best_cost.pt epoch={epoch} vs_sbs={best_cost:+.3f}%")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": config,
                    "epoch": epoch,
                    "macro": macro,
                    "best_score": best_score,
                },
                save_dir / "last.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": config,
            "epoch": args.epochs - 1,
            "best_score": best_score,
        },
        save_dir / "last.pt",
    )

    per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device)
    print_eval("test last", per_problem, macro)
    (save_dir / "test_last.json").write_text(json.dumps({"per_problem": per_problem, "macro": macro}, indent=2))

    best_path = save_dir / "best.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(best_ckpt["model"])
        per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device)
        print_eval("test best", per_problem, macro)
        (save_dir / "test_best.json").write_text(
            json.dumps(
                {
                    "per_problem": per_problem,
                    "macro": macro,
                    "best_epoch": best_ckpt.get("epoch"),
                    "best_val_macro": best_ckpt.get("macro"),
                },
                indent=2,
            )
        )
    for ckpt_name in ["best_top1", "best_top3_safe", "best_cost"]:
        ckpt_path = save_dir / f"{ckpt_name}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(ckpt["model"])
            per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device)
            print_eval(f"test {ckpt_name}", per_problem, macro)
            (save_dir / f"test_{ckpt_name}.json").write_text(
                json.dumps(
                    {
                        "per_problem": per_problem,
                        "macro": macro,
                        "best_epoch": ckpt.get("epoch"),
                        "best_val_macro": ckpt.get("macro"),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
