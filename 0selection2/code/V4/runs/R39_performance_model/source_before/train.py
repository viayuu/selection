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

from .V4Model import get_default_model_params, make_selector, score_components
from .tensor_loader import TensorBatchLoader, make_tensor_loader
from .training_monitor import capture_rng, restore_rng, plot_history


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


def make_loader(problem, split, batch_size, num_workers, coord_augment=0, shuffle=True, cache_device=None):
    if cache_device is not None:
        loader = make_tensor_loader(problem, split, batch_size, shuffle, cache_device, coord_augment)
        loader.drop_last = split == "train" and shuffle
        return loader
    dataset = UnifiedProblemDataset(problem, split, coord_augment=coord_augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_single_problem,
        drop_last=(split == "train" and shuffle),
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


def build_winner_weights(problems, max_weight=3.0, native_winner=False):
    weights = {}
    for problem in problems:
        ds = UnifiedProblemDataset(problem, "train", coord_augment=0)
        counts = np.zeros(ds.K_p, dtype=np.float64)
        for i in range(ds.base_N):
            label = ds.labels[str(i)]
            winner = int(label["ind"]) if native_winner else int(np.asarray(label["cost"][: ds.K_p], dtype=np.float32).argmin())
            counts[winner] += 1
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


def winner_first_rank(costs, winner=None):
    rank = costs.argsort(dim=1)
    if winner is not None:
        rest = rank[rank != winner[:, None]].reshape(len(costs), costs.size(1) - 1)
        rank = torch.cat([winner[:, None], rest], dim=1)
    return rank


def sequential_topk_ce_loss(logits, costs, rank_weights, winner=None):
    rank = winner_first_rank(costs, winner)
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


def winner_pair_loss(score, costs, winner, margin_scale=0.01):
    """Compare the winner to every strictly worse arm, including the runner-up."""
    best_cost = costs.gather(1, winner[:, None])
    gain = (costs - best_cost) / best_cost.abs().clamp_min(1e-9)
    valid = gain > 0
    weight = (gain / margin_scale).clamp(0.1, 5.0)
    margin = score.gather(1, winner[:, None]) - score
    return (F.softplus(-margin) * weight)[valid].mean() if valid.any() else score.sum() * 0


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
        "support": loss.detach(),
        "support_bce": bce_loss.detach(),
        "support_div": div_loss.detach(),
        "support_budget": budget_loss.detach(),
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


def selector_loss(out, costs, class_weight, args, sbs_idx=None, winner=None):
    logits = out["logits"]
    winner = costs.argmin(dim=1) if winner is None else winner
    if getattr(args, "loss_mode", "original") != "original":
        ce = F.cross_entropy(logits, winner)
        if args.loss_mode == "ce":
            return args.ce_weight * ce, {"ce": ce.detach()}
        pair = winner_pair_loss(logits, costs, winner, args.cost_scale)
        risk = risk_loss(logits, costs) / args.cost_scale
        loss = args.ce_weight * ce + args.pair_weight * pair + args.risk_weight * risk
        return loss, {"ce": ce.detach(), "pair": pair.detach(), "risk": risk.detach()}
    ce = F.cross_entropy(logits, winner, weight=class_weight)
    pair = top_focused_pair_loss(logits, costs, sbs_idx=sbs_idx) if args.top_focused_pair else pairwise_order_loss(logits, costs)
    rank_weights = parse_rank_weights(args.topk_ce_rank_weights)
    if args.topk_ce_mode == "sequential":
        topk_ce = sequential_topk_ce_loss(logits, costs, rank_weights, winner if getattr(args, "native_winner", False) else None)
    else:
        topk_ce = ranked_topk_ce_loss(logits, costs, rank_weights)
    risk = risk_loss(logits, costs)
    if getattr(args, "architecture", "legacy") == "dual_stream":
        loss = args.ce_weight * ce + args.pair_weight * pair + args.topk_ce_weight * topk_ce + args.risk_weight * risk
        return loss, {"ce": ce.detach(), "pair": pair.detach(), "topk_ce": topk_ce.detach(), "risk": risk.detach()}
    gap = gap_loss(out["pred_gap"], costs)
    pre = F.cross_entropy(out["pre_score"], winner, weight=class_weight)
    div = query_diversity_loss(out["query_logits"])
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
        "ce": ce.detach(),
        "pair": pair.detach(),
        "gap": gap.detach(),
        "pre": pre.detach(),
        "topk_ce": topk_ce.detach(),
        "div": div.detach(),
        "risk": risk.detach(),
    }
    parts.update(support_parts)
    return loss, parts


@torch.no_grad()
def evaluate(model, problems, split, batch_size, num_workers, device, cache_device=None):
    was_training = model.training
    model.eval()
    per_problem = {}
    for problem in problems:
        loader = make_loader(problem, split, batch_size, num_workers, coord_augment=0, shuffle=False, cache_device=cache_device)
        comp = {}
        n_total = 0
        ce_sum = tie_sum = 0.0
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
            scores = score_components(out)
            costs = batch["costs"]
            true_rank = torch.argsort(costs, dim=1)
            best = true_rank[:, 0]
            if model.params.get("native_winner", False):
                best = batch["ind"]
            ce_sum += F.cross_entropy(out["logits"], best, reduction="sum").item()
            pred_cost = costs.gather(1, out["logits"].argmax(1)[:, None]).squeeze(1)
            tie_sum += pred_cost.eq(costs.min(1).values).sum().item()
            for name, score in scores.items():
                comp.setdefault(name, {"top1": 0, "top2": 0, "top3": 0, "cost_sum": 0.0})
                std_meter.setdefault(name, [])
                pred_rank = torch.argsort(-score, dim=1)
                pred = pred_rank[:, 0]
                comp[name]["top1"] += (pred == best).sum().item()
                comp[name]["top2"] += (pred_rank[:, : min(2, score.size(1))] == best[:, None]).any(dim=1).sum().item()
                comp[name]["top3"] += (pred_rank[:, : min(3, score.size(1))] == best[:, None]).any(dim=1).sum().item()
                comp[name]["cost_sum"] += costs.gather(1, pred[:, None]).sum().item()
                std_meter[name].append(score.std(dim=1).mean().item())
            for name, score in scores.items():
                if name != "final":
                    corr_meter[f"{name}_final"].append(score_correlation(score, scores["final"]))
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
        if isinstance(loader, TensorBatchLoader):
            costs_np = loader.batch["costs"].cpu().numpy()
            sbs, vbs = float(costs_np.mean(axis=0).min()), float(costs_np.min(axis=1).mean())
        else:
            sbs, vbs = get_split_sbs_vbs(problem, split)
        mean_cost = comp["final"]["cost_sum"] / max(1, n_total)
        per_problem[problem] = dict(
            ce=ce_sum / n_total,
            top1_tie_aware=tie_sum / n_total,
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
            if values:
                per_problem[problem][f"corr_{name}"] = float(np.mean(values))
    macro = dict(
        macro_ce=float(np.mean([v["ce"] for v in per_problem.values()])),
        macro_top1_tie_aware=float(np.mean([v["top1_tie_aware"] for v in per_problem.values()])),
        macro_top1=float(np.mean([v["top1"] for v in per_problem.values()])),
        macro_top2=float(np.mean([v["top2"] for v in per_problem.values()])),
        macro_top3=float(np.mean([v["top3"] for v in per_problem.values()])),
        macro_vs_sbs_pct=float(np.mean([v["vs_sbs_pct"] for v in per_problem.values()])),
        macro_vbs_gap_closed_pct=float(np.mean([v["vbs_gap_closed_pct"] for v in per_problem.values()])),
    )
    for key in next(iter(per_problem.values())):
        if key.startswith(("top1_", "vs_sbs_pct_", "corr_")):
            macro[f"macro_{key}"] = float(np.mean([v[key] for v in per_problem.values()]))
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
    model.train(was_training)
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
    if "macro_top1_pre" in macro:
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
        components = f" pre={r['top1_pre']:.3f} gap={r['top1_gap']:.3f} util={r['top1_utility']:.3f}" if "top1_pre" in r else ""
        print(
            f"{p:>10}: top1={r['top1']:.3f} mean_cost={r['mean_cost']:.4f} "
            f"(sbs={r['sbs']:.4f}) vs_sbs={r['vs_sbs_pct']:+.2f}%{components}"
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
        solver_feature_spec=None if args.no_solver_features else params["solver_feature_spec"],
        solver_feature_weight=args.solver_feature_weight,
        solver_feature_hidden=args.solver_feature_hidden,
        sdpa=args.sdpa,
        architecture=getattr(args, "architecture", "legacy"),
        joint_layer_num=getattr(args, "joint_layers", 2),
        ignore_coord_dist=getattr(args, "ignore_coord_dist", False),
        native_winner=getattr(args, "native_winner", False),
    )
    return make_selector(params), params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--architecture", choices=["legacy", "dual_stream"], default="legacy")
    parser.add_argument("--joint-layers", type=int, default=2)
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
    parser.add_argument("--ignore-coord-dist", action="store_true")
    parser.add_argument("--native-winner", action="store_true")
    parser.add_argument("--loss-mode", choices=["original", "ce", "winner_cost"], default="original")
    parser.add_argument("--cost-scale", type=float, default=0.01)
    parser.add_argument("--lr-schedule", choices=["constant", "plateau"], default="constant")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-updates", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--train-eval-every", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true", help="Select the experiment on validation before testing")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--cache-gpu", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--wandb-project", default="selector")
    parser.add_argument("--wandb-entity", default="yjkds-southern-university-of-science-technology")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--query-num", type=int, default=4)
    parser.add_argument("--no-solver-features", action="store_true", help="Use ID-only solver embeddings (legacy V4)")
    parser.add_argument("--solver-feature-weight", type=float, default=0.3)
    parser.add_argument("--solver-feature-hidden", type=int, default=128)
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
    if args.cost_scale <= 0:
        parser.error("cost-scale must be positive")
    if args.loss_mode != "original" and args.architecture != "dual_stream":
        parser.error("simplified losses currently require dual_stream (no unused auxiliary heads)")

    if args.architecture == "dual_stream":
        if args.no_solver_features:
            parser.error("dual_stream requires the fixed solver feature table")
        args.support_branch = args.support_g0_as_main = False
        args.support_generators = 0
        for name in ("gap_weight", "pre_ce_weight", "query_div_weight", "support_loss_weight",
                     "gap_score_weight", "pre_score_weight", "support_score_weight",
                     "support_main_utility_weight", "support_main_pre_weight", "support_main_gap_weight"):
            setattr(args, name, 0.0)

    set_seed(args.seed)
    configure_torch()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else list(PROBLEMS)

    model, model_params = build_model(args)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=args.min_lr,
    ) if args.lr_schedule == "plateau" else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 0
    global_step = successful_updates = 0
    stop_best, stale_evals = -1e9, 0
    resume_rng = None
    winner_weight_policy = "native" if args.native_winner else "fp32_argmin"
    best_score = -1.0e9
    best_top1 = -1.0e9
    best_top3_safe = -1.0e9
    best_cost = 1.0e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        winner_weight_policy = ckpt.get("winner_weight_policy", "fp32_argmin")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        best_top1 = ckpt.get("best_top1", best_top1)
        best_top3_safe = ckpt.get("best_top3_safe", best_top3_safe)
        best_cost = ckpt.get("best_cost", best_cost)
        resume_macro = ckpt.get("macro", {})
        best_top1 = max(best_top1, resume_macro.get("macro_top1", -1e9))
        best_cost = min(best_cost, resume_macro.get("macro_vs_sbs_pct", 1e9))
        if "macro_top3" in resume_macro:
            best_top3_safe = max(best_top3_safe, resume_macro["macro_top3"] - 0.25 * max(0.0, resume_macro["macro_vs_sbs_pct"]))
        global_step = ckpt.get("global_step", 0)
        successful_updates = ckpt.get("successful_updates", global_step)
        stop_best, stale_evals = ckpt.get("stop_best", -1e9), ckpt.get("stale_evals", 0)
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        resume_rng = ckpt.get("rng")

    history = json.loads((save_dir / "history.json").read_text()) if args.resume and (save_dir / "history.json").exists() else []
    if history and max(r["epoch"] for r in history) >= start_epoch:
        raise ValueError("History is newer than the resume checkpoint; resume last.pt or use a new save-dir")
    cache_device = args.device if args.cache_gpu else None
    train_loaders = {
        p: make_loader(p, "train", args.batch_per_problem, args.num_workers, args.coord_augment, shuffle=True, cache_device=cache_device)
        for p in problems
    }
    class_weights = build_winner_weights(problems, native_winner=winner_weight_policy == "native") if args.winner_balance else {p: None for p in problems}
    sbs_indices = build_sbs_indices(problems, split="train")

    config = vars(args)
    config["model_params"] = model_params
    config["winner_weight_policy"] = winner_weight_policy
    (save_dir / "args.json").write_text(json.dumps(config, indent=2))
    model_name = "V4 R32a R31c + R25-style support branch" if args.support_branch else "V4 R31c multi-query problem-to-solver attention + solver-set transformer"
    if args.architecture == "dual_stream":
        model_name = "V4 dual-stream joint encoder + instance-query solver decoder (logits only)"
    print(f"[train] {model_name}")
    print(f"[train] problems={problems}")
    print(f"[train] model_params={model_params}")
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=save_dir.name,
            config=config, dir=str(save_dir), mode=args.wandb_mode,
        )
    if resume_rng is not None:
        restore_rng(resume_rng)

    def track_eval(tag, per_problem, macro):
        if wandb_run is not None:
            metrics = {f"{tag}/{k}": v for k, v in macro.items()}
            for problem, result in per_problem.items():
                metrics.update({f"{tag}/{problem}/{k}": v for k, v in result.items() if isinstance(v, (int, float))})
            wandb_run.log(metrics, step=global_step)

    def save_checkpoint(name, epoch, macro, score):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": config,
                "epoch": epoch,
                "macro": macro,
                "best_score": best_score,
                "selection_score": score,
                "best_top1": best_top1,
                "best_top3_safe": best_top3_safe,
                "best_cost": best_cost,
                "global_step": global_step,
                "successful_updates": successful_updates,
                "winner_weight_policy": winner_weight_policy,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "rng": capture_rng(),
                "stop_best": stop_best,
                "stale_evals": stale_evals,
            },
            save_dir / name,
        )

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    start = time.time()
    last_log_time = start
    interval_examples = 0
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (epoch + 1) / args.warmup_epochs
        epoch_lr = optimizer.param_groups[0]["lr"]
        gradient_norms = []
        updates_before = successful_updates
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
                    loss, parts = selector_loss(out, batch["costs"], weight, args, sbs_idx=sbs_indices.get(problem),
                                                winner=batch["ind"] if args.native_winner else None)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful_updates += int(scaler.get_scale() >= old_scale)
                gradient_norms.append(grad_norm.detach())
                global_step += 1
                meter.append(loss.detach())
                interval_examples += batch["costs"].size(0)
                for k, v in parts.items():
                    part_meter[k].append(v)
                if global_step % args.log_every == 0:
                    keys = [k for k, v in part_meter.items() if v]
                    means = torch.stack([torch.stack(meter).mean()] + [torch.stack(part_meter[k]).mean() for k in keys]).float().cpu().tolist()
                    stats = dict(zip(["loss"] + keys, means))
                    now = time.time()
                    throughput = interval_examples / max(now - last_log_time, 1.0e-9)
                    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if str(args.device).startswith("cuda") else 0.0
                    msg = " ".join(f"{k}={stats[k]:.4f}" for k in keys)
                    print(
                        f"ep{epoch:03d} step{global_step:06d} loss={stats['loss']:.4f} "
                        f"{msg} samples/s={throughput:.0f} peak_GiB={peak_gb:.2f} time={(now - start) / 60:.1f}m"
                    )
                    if wandb_run is not None:
                        wandb_run.log({**{f"train/{k}": v for k, v in stats.items()}, "epoch": epoch,
                                       "train/lr": optimizer.param_groups[0]["lr"],
                                       "speed/samples_per_sec": throughput, "gpu/peak_memory_gib": peak_gb}, step=global_step)
                    last_log_time, interval_examples = now, 0

        train_stats = {"loss": torch.stack(meter).mean().item()}
        train_stats.update({k: torch.stack(v).mean().item() for k, v in part_meter.items() if v})
        norms = torch.stack(gradient_norms)
        train_stats["grad_norm"] = norms[torch.isfinite(norms)].mean().item() if torch.isfinite(norms).any() else None
        train_stats["skipped_updates"] = len(meter) - (successful_updates - updates_before)
        print(f"[epoch done {epoch}] loss={train_stats['loss']:.5f} lr={epoch_lr:.2g} "
              f"updates={successful_updates} skipped={train_stats['skipped_updates']} grad={train_stats['grad_norm']}")
        record = dict(epoch=epoch, step=global_step, updates=successful_updates, lr=epoch_lr, train=train_stats)
        should_stop = False
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            per_problem, macro = evaluate(model, problems, "val", args.batch_per_problem, args.num_workers, args.device, cache_device)
            print_eval(f"eval epoch {epoch}", per_problem, macro)
            track_eval("val", per_problem, macro)
            (save_dir / f"eval_epoch{epoch}.json").write_text(json.dumps({"per_problem": per_problem, "macro": macro}, indent=2))
            score = macro["macro_top1"] - 0.25 * max(0.0, macro["macro_vs_sbs_pct"])
            record["val"] = macro
            if score > stop_best + args.min_delta:
                stop_best, stale_evals = score, 0
            else:
                stale_evals += 1
            if scheduler is not None and epoch + 1 >= args.warmup_epochs:
                scheduler.step(score)
            should_stop = bool(args.early_stop_patience and stale_evals >= args.early_stop_patience
                               and successful_updates >= args.min_updates)
            top3_safe_score = macro["macro_top3"] - 0.25 * max(0.0, macro["macro_vs_sbs_pct"])
            improve_score = score > best_score
            improve_top1 = macro["macro_top1"] > best_top1
            improve_top3 = top3_safe_score > best_top3_safe
            improve_cost = macro["macro_vs_sbs_pct"] < best_cost
            best_score = max(best_score, score)
            best_top1 = max(best_top1, macro["macro_top1"])
            best_top3_safe = max(best_top3_safe, top3_safe_score)
            best_cost = min(best_cost, macro["macro_vs_sbs_pct"])
            if improve_score:
                save_checkpoint("best.pt", epoch, macro, best_score)
                print(f"[save] best.pt epoch={epoch} score={score:.4f}")
            if improve_top1:
                save_checkpoint("best_top1.pt", epoch, macro, best_top1)
                print(f"[save] best_top1.pt epoch={epoch} top1={best_top1:.4f}")
            if improve_top3:
                save_checkpoint("best_top3_safe.pt", epoch, macro, best_top3_safe)
                print(f"[save] best_top3_safe.pt epoch={epoch} score={best_top3_safe:.4f}")
            if improve_cost:
                save_checkpoint("best_cost.pt", epoch, macro, best_cost)
                print(f"[save] best_cost.pt epoch={epoch} vs_sbs={best_cost:+.3f}%")
        if args.train_eval_every and ((epoch + 1) % args.train_eval_every == 0 or should_stop or epoch == args.epochs - 1):
            train_problem, train_macro = evaluate(model, problems, "train", args.batch_per_problem,
                                                  args.num_workers, args.device, cache_device)
            print_eval(f"train_eval epoch {epoch}", train_problem, train_macro)
            track_eval("train_eval", train_problem, train_macro)
            record["train_eval"] = train_macro
            (save_dir / f"train_eval_epoch{epoch}.json").write_text(json.dumps({"per_problem": train_problem, "macro": train_macro}, indent=2))
        history.append(record)
        if wandb_run is not None:
            wandb_run.log({**{f"epoch_train/{k}": v for k, v in train_stats.items()},
                           "train/successful_updates": successful_updates, "train/amp_scale": scaler.get_scale()}, step=global_step)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_checkpoint("last.pt", epoch, record.get("val", {}), best_score)
        if (epoch + 1) % 5 == 0 or should_stop or epoch == args.epochs - 1:
            plot_history(history, save_dir)
        if should_stop:
            print(f"[early stop] epoch={epoch} stale_evals={stale_evals} successful_updates={successful_updates}")
            break

    if args.skip_test:
        print("[done] validation-selected checkpoints saved; test deferred until experiment selection")
        if wandb_run is not None:
            wandb_run.finish()
        return

    per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device, cache_device)
    print_eval("test last", per_problem, macro)
    track_eval("test_last", per_problem, macro)
    (save_dir / "test_last.json").write_text(json.dumps({"per_problem": per_problem, "macro": macro}, indent=2))

    best_path = save_dir / "best.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(best_ckpt["model"])
        per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device, cache_device)
        print_eval("test best", per_problem, macro)
        track_eval("test_best", per_problem, macro)
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
            per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device, cache_device)
            print_eval(f"test {ckpt_name}", per_problem, macro)
            track_eval(f"test_{ckpt_name}", per_problem, macro)
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


    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
