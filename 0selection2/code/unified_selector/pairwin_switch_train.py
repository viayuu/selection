from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import UnifiedProblemDataset, collate_single_problem
from .eval_log import format_eval_block
from .model import load_partial_state_dict
from .registry import PROBLEMS
from .rerank_train import (
    set_seed,
    to_device,
    weighted_mean,
    parse_float_grid,
    build_base_model,
    load_or_build_problem_stats,
    summarize_macro,
    evaluate_base_all,
)
from .solver_context_rerank_train import (
    ResidualCrossBlock,
    build_context_meta,
    load_or_build_solver_profiles,
)


def binary_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.zeros_like(scores, dtype=np.float64)
    start = 0
    rank = 1.0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (rank + (rank + (end - start) - 1))
        ranks[order[start:end]] = avg_rank
        rank += end - start
        start = end
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


class PairWinSwitchReranker(nn.Module):
    def __init__(
        self,
        cand_dim: int,
        inst_dim: int,
        solver_dim: int,
        profile_dim: int,
        cond_dim: int,
        support_dim: int,
        d_model: int = 128,
        heads: int = 4,
        cross_layers: int = 1,
        self_layers: int = 2,
        dropout: float = 0.1,
        max_topk: int = 10,
    ):
        super().__init__()
        self.cand_in = nn.Sequential(nn.Linear(cand_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.inst_in = nn.Sequential(nn.Linear(inst_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.solver_profile = nn.Sequential(
            nn.Linear(solver_dim + profile_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.inst_cond_proj = nn.Linear(d_model, d_model)
        self.cond_film = nn.Linear(d_model, 2 * d_model)
        self.query_ln = nn.LayerNorm(d_model)
        self.inst_ln = nn.LayerNorm(d_model)
        self.pos_emb = nn.Embedding(max_topk, d_model)
        self.cross_blocks = nn.ModuleList([ResidualCrossBlock(d_model, heads, dropout) for _ in range(cross_layers)])
        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=2 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(self_layers)
        ])
        pair_num_dim = 9 + support_dim
        pair_in_dim = 4 * d_model + pair_num_dim
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pair_win_head = nn.Linear(d_model, 1)
        self.gain_head = nn.Linear(d_model, 1)
        self.switch_head = nn.Linear(d_model, 1)

    def _build_query(self, cand_feats, inst_tokens, inst_mask, solver_embs, solver_profiles, cond_vec):
        bsz, k, _ = cand_feats.shape
        pos = torch.arange(k, device=cand_feats.device).unsqueeze(0).expand(bsz, k)
        cand = self.cand_in(cand_feats) + self.pos_emb(pos)
        inst = self.inst_in(inst_tokens)
        solver_ctx = self.solver_profile(torch.cat([solver_embs, solver_profiles], dim=-1))
        cond = self.cond_proj(cond_vec)
        film_gain, film_bias = torch.tanh(self.cond_film(cond)).chunk(2, dim=-1)
        q = self.query_ln(cand + solver_ctx + cond.unsqueeze(1))
        q = q * (1.0 + 0.10 * film_gain.unsqueeze(1)) + 0.10 * film_bias.unsqueeze(1)
        inst = self.inst_ln(inst + self.inst_cond_proj(cond).unsqueeze(1))
        for block in self.cross_blocks:
            q = block(q, inst, inst_mask)
        for block in self.self_blocks:
            q = block(q)
        return q

    def forward(self, meta, cand_feats, inst_tokens, inst_mask, solver_embs, solver_profiles, cond_vec):
        q = self._build_query(cand_feats, inst_tokens, inst_mask, solver_embs, solver_profiles, cond_vec)
        bsz, k, d = q.shape
        rows = torch.arange(bsz, device=q.device)
        base_pos = meta["base_pred_cand_pos"]
        base_q = q[rows, base_pos].unsqueeze(1).expand(-1, k, -1)
        base_logit = meta["cand_base"][rows, base_pos].unsqueeze(1)
        base_prob = meta["cand_prob"][rows, base_pos].unsqueeze(1)
        base_support_max = meta["cand_support_max"][rows, base_pos].unsqueeze(1)
        if meta["cand_support_per_gen"].shape[-1] > 0:
            base_support_gen = meta["cand_support_per_gen"][rows, base_pos].unsqueeze(1)
            support_diff = meta["cand_support_per_gen"] - base_support_gen
        else:
            support_diff = q.new_zeros((bsz, k, 0))
        num_parts = [
            (meta["cand_base"] - base_logit).unsqueeze(-1),
            (meta["cand_prob"] - base_prob).unsqueeze(-1),
            (meta["cand_support_max"] - base_support_max).unsqueeze(-1),
            meta["cand_oracle_prior"].unsqueeze(-1),
            meta["cand_hidden_flag"].unsqueeze(-1),
            meta["cand_rank_frac"].unsqueeze(-1),
            meta["cand_base_delta_sbs"].unsqueeze(-1),
            meta["cand_prob_delta_sbs"].unsqueeze(-1),
            meta["cand_is_sbs"].unsqueeze(-1),
            support_diff,
        ]
        pair_feats = torch.cat([
            q,
            base_q,
            q - base_q,
            q * base_q,
            *num_parts,
        ], dim=-1)
        pair_h = self.pair_proj(pair_feats)
        pair_logit = self.pair_win_head(pair_h).squeeze(-1)
        gain_pred = self.gain_head(pair_h).squeeze(-1)
        switch_logit = self.switch_head(pair_h).squeeze(-1)
        return pair_logit, gain_pred, switch_logit


def pairwise_pool_loss(pred_gain: torch.Tensor, cand_costs: torch.Tensor, pair_weight: torch.Tensor | None = None) -> torch.Tensor:
    sign = torch.sign(cand_costs.unsqueeze(2) - cand_costs.unsqueeze(1))
    score_diff = pred_gain.unsqueeze(1) - pred_gain.unsqueeze(2)
    margin = (cand_costs.unsqueeze(2) - cand_costs.unsqueeze(1)).abs()
    weight = torch.clamp(margin / margin.mean().clamp_min(1e-6), min=0.25, max=5.0)
    tri = torch.triu(torch.ones(pred_gain.shape[1], pred_gain.shape[1], dtype=torch.bool, device=pred_gain.device), diagonal=1)
    mask = (sign != 0) & tri.unsqueeze(0)
    if not mask.any():
        return pred_gain.new_tensor(0.0)
    terms = F.softplus(-sign * score_diff) * weight
    if pair_weight is None:
        return terms.masked_select(mask).mean()
    per_ex = []
    for i in range(pred_gain.shape[0]):
        cur_mask = mask[i]
        if cur_mask.any():
            per_ex.append(terms[i].masked_select(cur_mask).mean())
        else:
            per_ex.append(pred_gain.new_tensor(0.0))
    return weighted_mean(torch.stack(per_ex), pair_weight)


def bce_weighted(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weight).sum() / weight.sum().clamp_min(1e-6)


def brier_weighted(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = (prob - target).pow(2)
    return (raw * weight).sum() / weight.sum().clamp_min(1e-6)


def build_candidate_weights(meta, true_gain_pct: torch.Tensor, args) -> torch.Tensor:
    weight = torch.ones_like(true_gain_pct)
    non_base = torch.ones_like(true_gain_pct, dtype=torch.bool)
    non_base.scatter_(1, meta["base_pred_cand_pos"].unsqueeze(1), False)
    weight = weight * non_base.float()
    if args.hidden_candidate_weight > 0:
        weight = weight * (1.0 + args.hidden_candidate_weight * meta["cand_hidden_flag"])
    if args.decisive_candidate_weight > 0:
        decisive = (true_gain_pct >= args.decisive_gain_pct).float()
        weight = weight * (1.0 + args.decisive_candidate_weight * decisive)
    return weight


def pairwin_switch_loss(meta, pair_logit, gain_pred, switch_logit, args):
    base_cost = meta["base_selected_cost"].unsqueeze(1)
    true_gain_pct = ((base_cost - meta["cand_costs"]) / (base_cost.abs() + 1e-9)) * 100.0
    rows = torch.arange(true_gain_pct.shape[0], device=true_gain_pct.device)
    true_gain_pct = true_gain_pct.clone()
    true_gain_pct[rows, meta["base_pred_cand_pos"]] = 0.0
    non_base = torch.ones_like(true_gain_pct, dtype=torch.bool)
    non_base.scatter_(1, meta["base_pred_cand_pos"].unsqueeze(1), False)
    pair_target = (true_gain_pct > args.pair_eps_pct).float()
    switch_target = (true_gain_pct > args.switch_eps_pct).float()
    cand_weight = build_candidate_weights(meta, true_gain_pct, args)
    active_weight = cand_weight * non_base.float()

    pair_loss = bce_weighted(pair_logit, pair_target, active_weight)
    gain_loss = weighted_mean(
        F.smooth_l1_loss(gain_pred, true_gain_pct, reduction="none")[non_base],
        active_weight[non_base],
    )
    listwise_ex_weight = meta["base_wrong"].float() * (1.0 + args.switch_gain_weight * (meta["switch_gain_pct"] >= args.switch_eps_pct).float())
    listwise_loss = pairwise_pool_loss(gain_pred, meta["cand_costs"], pair_weight=listwise_ex_weight)

    switch_loss = pair_logit.new_tensor(0.0)
    calib_loss = pair_logit.new_tensor(0.0)
    if args.use_switch_head:
        switch_loss = bce_weighted(switch_logit, switch_target, active_weight)
        switch_prob = torch.sigmoid(switch_logit)
        calib_loss = brier_weighted(switch_prob, switch_target, active_weight)

    total = (
        args.pair_loss_weight * pair_loss
        + args.gain_loss_weight * gain_loss
        + args.listwise_loss_weight * listwise_loss
        + args.switch_loss_weight * switch_loss
        + args.calibration_loss_weight * calib_loss
    )
    parts = {
        "pair": float(pair_loss.item()),
        "gain": float(gain_loss.item()),
        "list": float(listwise_loss.item()),
        "switch": float(switch_loss.item()),
        "calib": float(calib_loss.item()),
    }
    return total, parts


@torch.no_grad()
def collect_problem_predictions(
    base_model,
    reranker,
    problem: str,
    split: str,
    device: str,
    audit: dict,
    problem_stats: dict,
    solver_profiles: dict,
    topk: int,
    instance_max_tokens: int,
    base_topk: int | None,
    support_topk: int,
    full_pool: bool,
    support_feature_mode: str,
):
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
    out = {
        "problem": problem,
        "pool_names": audit[problem]["pool"],
        "sbs_cost": audit[problem][split]["sbs_mean"],
        "vbs_mean": audit[problem][split]["vbs_mean"],
    }
    bucket = {
        "pair_prob": [],
        "gain_pred": [],
        "switch_prob": [],
        "cand_costs": [],
        "cand_local_ids": [],
        "cand_base": [],
        "cand_hidden_flag": [],
        "base_pred_local": [],
        "base_pred_cand_pos": [],
        "best_local": [],
        "base_selected_cost": [],
        "switch_gain_pct": [],
    }
    for batch in dl:
        batch = to_device(batch, device)
        meta = build_context_meta(
            base_model,
            batch,
            problem_stats=problem_stats,
            solver_profiles=solver_profiles,
            audit=audit,
            topk=topk,
            instance_max_tokens=instance_max_tokens,
            base_topk=base_topk,
            support_topk=support_topk,
            full_pool=full_pool,
            support_feature_mode=support_feature_mode,
        )
        pair_logit, gain_pred, switch_logit = reranker(
            meta,
            meta["token_feats"],
            meta["inst_tokens"],
            meta["inst_mask"],
            meta["solver_embs"],
            meta["solver_profiles"],
            meta["cond_vec"],
        )
        bucket["pair_prob"].append(torch.sigmoid(pair_logit).detach().cpu().numpy())
        bucket["gain_pred"].append(gain_pred.detach().cpu().numpy())
        bucket["switch_prob"].append(torch.sigmoid(switch_logit).detach().cpu().numpy())
        bucket["cand_costs"].append(meta["cand_costs"].detach().cpu().numpy())
        bucket["cand_local_ids"].append(meta["top_idx"].detach().cpu().numpy())
        bucket["cand_base"].append(meta["cand_base"].detach().cpu().numpy())
        bucket["cand_hidden_flag"].append(meta["cand_hidden_flag"].detach().cpu().numpy())
        bucket["base_pred_local"].append(meta["base_pred_local"].detach().cpu().numpy())
        bucket["base_pred_cand_pos"].append(meta["base_pred_cand_pos"].detach().cpu().numpy())
        bucket["best_local"].append(meta["winner_local"].detach().cpu().numpy())
        bucket["base_selected_cost"].append(meta["base_selected_cost"].detach().cpu().numpy())
        bucket["switch_gain_pct"].append(meta["switch_gain_pct"].detach().cpu().numpy())

    out.update({k: np.concatenate(v, axis=0) for k, v in bucket.items()})
    return out


def metrics_from_problem_collection(col: dict, prob_thr: float, gain_thr: float, use_switch_head: bool) -> dict:
    rows = np.arange(col["gain_pred"].shape[0])
    base_pos = col["base_pred_cand_pos"]
    cand_choice = col["gain_pred"].copy()
    cand_choice[rows, base_pos] = -1e18
    cand_pos = cand_choice.argmax(axis=1)
    gate_prob = col["switch_prob"] if use_switch_head else col["pair_prob"]
    gate_score = gate_prob[rows, cand_pos]
    gain_score = col["gain_pred"][rows, cand_pos]
    switch_mask = (gate_score >= prob_thr) & (gain_score >= gain_thr)
    chosen_pos = np.where(switch_mask, cand_pos, base_pos)
    pred_local = col["cand_local_ids"][rows, chosen_pos]
    sel_cost = col["cand_costs"][rows, chosen_pos]
    best_local = col["best_local"]
    base_local = col["base_pred_local"]
    base_wrong = base_local != best_local
    safe = ~base_wrong

    final_scores = col["gain_pred"].copy()
    if (~switch_mask).any():
        max_scores = final_scores.max(axis=1)
        final_scores[~switch_mask, base_pos[~switch_mask]] = np.maximum(max_scores[~switch_mask] + 1e-6, 0.0)
    rank_order = np.argsort(-final_scores, axis=1)
    ranked_local = np.take_along_axis(col["cand_local_ids"], rank_order, axis=1)

    decisive = col["switch_gain_pct"]
    result = {
        "top1": float(np.mean(pred_local == best_local)),
        "top2": float(np.mean(np.any(ranked_local[:, : min(2, ranked_local.shape[1])] == best_local[:, None], axis=1))),
        "top3": float(np.mean(np.any(ranked_local[:, : min(3, ranked_local.shape[1])] == best_local[:, None], axis=1))),
        "mean_cost": float(sel_cost.mean()),
        "sbs_cost": float(col["sbs_cost"]),
        "vbs_mean": float(col["vbs_mean"]),
        "vs_sbs_pct": float((sel_cost.mean() - col["sbs_cost"]) / (abs(col["sbs_cost"]) + 1e-9) * 100.0),
        "vbs_gap_closed_pct": float(((col["sbs_cost"] - sel_cost.mean()) / (col["sbs_cost"] - col["vbs_mean"] + 1e-9)) * 100.0) if col["sbs_cost"] != col["vbs_mean"] else 0.0,
        "rerank_apply_rate": float(np.mean(switch_mask)),
        "fixable_recovery_rate": float(np.mean(pred_local[base_wrong] == best_local[base_wrong])) if base_wrong.any() else 0.0,
        "safe_harm_rate": float(np.mean(pred_local[safe] != best_local[safe])) if safe.any() else 0.0,
        "oracle_rank_mean": float(np.mean(np.argmax(ranked_local == best_local[:, None], axis=1) + 1)),
        "oracle_rank_top2_rate": float(np.mean((np.argmax(ranked_local == best_local[:, None], axis=1) + 1) <= 2)),
        "oracle_rank_top3_rate": float(np.mean((np.argmax(ranked_local == best_local[:, None], axis=1) + 1) <= 3)),
        "top1_decisive_gap_0.1": float(np.mean((pred_local == best_local)[base_wrong & (decisive > 0.1)])) if np.any(base_wrong & (decisive > 0.1)) else 0.0,
        "top1_decisive_gap_0.5": float(np.mean((pred_local == best_local)[base_wrong & (decisive > 0.5)])) if np.any(base_wrong & (decisive > 0.5)) else 0.0,
        "top1_decisive_gap_1.0": float(np.mean((pred_local == best_local)[base_wrong & (decisive > 1.0)])) if np.any(base_wrong & (decisive > 1.0)) else 0.0,
    }
    pool_names = col["pool_names"]
    method_top1 = {name: float(np.mean(best_local == i)) for i, name in enumerate(pool_names)}
    arm_dist = {name: float(np.mean(pred_local == i)) for i, name in enumerate(pool_names)}
    hidden = [name for i, name in enumerate(pool_names) if method_top1[name] > 0 and arm_dist[name] <= 0]
    result.update({
        "arm_distribution": arm_dist,
        "final_arm_coverage": float(np.mean([v > 0 for v in arm_dist.values()])),
        "pick_entropy": float(-(np.asarray(list(arm_dist.values())) * np.log(np.asarray(list(arm_dist.values())) + 1e-12)).sum() / np.log(max(2, len(pool_names)))),
        "zero_pick_count": int(sum(v <= 0 for v in arm_dist.values())),
        "hidden_winner_count": int(len(hidden)),
        "hidden_winner_mass": float(sum(method_top1[name] for name in hidden)),
        "oracle_support_mass_recall": 1.0,
        "method_top1": method_top1,
        "switch_precision": float(np.mean((col["cand_costs"][rows, cand_pos] < col["base_selected_cost"] - 1e-9)[switch_mask])) if np.any(switch_mask) else 0.0,
        "switch_gain_mean_pct": float(np.mean(((col["base_selected_cost"] - col["cand_costs"][rows, cand_pos]) / (np.abs(col["base_selected_cost"]) + 1e-9) * 100.0)[switch_mask])) if np.any(switch_mask) else 0.0,
        "chosen_threshold_prob": float(prob_thr),
        "chosen_threshold_gain": float(gain_thr),
    })
    return result


def metrics_all_from_collections(collections: dict[str, dict], prob_thr: float, gain_thr: float, use_switch_head: bool):
    per = {problem: metrics_from_problem_collection(col, prob_thr, gain_thr, use_switch_head) for problem, col in collections.items()}
    return {"per_problem": per, "macro": summarize_macro(per)}


def build_frontier(curves: list[dict], safe_harm_budgets: list[float]):
    frontier = {}
    for budget in safe_harm_budgets:
        feasible = [c for c in curves if c["macro"]["macro_safe_harm_rate"] <= budget + 1e-9]
        if not feasible:
            frontier[f"{budget:.3f}"] = None
            continue
        best = max(
            feasible,
            key=lambda c: (
                c["macro"]["macro_fixable_recovery_rate"],
                c["macro"]["macro_top1"],
                -c["macro"]["macro_vs_sbs_pct"],
                -c["macro"]["macro_hidden_winner_mass"],
            ),
        )
        frontier[f"{budget:.3f}"] = best
    return frontier


def choose_best_frontier(curves: list[dict], safe_harm_budget: float):
    feasible = [c for c in curves if c["macro"]["macro_safe_harm_rate"] <= safe_harm_budget + 1e-9]
    if feasible:
        return max(
            feasible,
            key=lambda c: (
                c["macro"]["macro_top1"],
                -c["macro"]["macro_vs_sbs_pct"],
                c["macro"]["macro_fixable_recovery_rate"],
                -c["macro"]["macro_hidden_winner_mass"],
            ),
        )
    return max(
        curves,
        key=lambda c: (
            c["macro"]["macro_top1"] - 5.0 * max(0.0, c["macro"]["macro_safe_harm_rate"] - safe_harm_budget),
            -c["macro"]["macro_vs_sbs_pct"],
            c["macro"]["macro_fixable_recovery_rate"],
        ),
    )


def compute_required_delta_to_flip(collections: dict[str, dict]):
    out = {}
    ratios = []
    frac_hits = []
    for problem, col in collections.items():
        rows = np.arange(col["cand_base"].shape[0])
        wrong = col["base_pred_local"] != col["best_local"]
        if not np.any(wrong):
            out[problem] = {"count": 0, "frac_gain_margin_beats_required": None, "mean_margin_ratio": None}
            continue
        oracle_pos = (col["cand_local_ids"] == col["best_local"][:, None]).argmax(axis=1)
        req = col["cand_base"][rows, col["base_pred_cand_pos"]] - col["cand_base"][rows, oracle_pos]
        actual = col["gain_pred"][rows, oracle_pos] - col["gain_pred"][rows, col["base_pred_cand_pos"]]
        req = req[wrong]
        actual = actual[wrong]
        frac = float(np.mean(actual > req))
        ratio = float(np.mean(actual / (np.abs(req) + 1e-6)))
        out[problem] = {
            "count": int(wrong.sum()),
            "frac_gain_margin_beats_required": frac,
            "mean_margin_ratio": ratio,
        }
        frac_hits.append(frac)
        ratios.append(ratio)
    out["macro"] = {
        "macro_frac_gain_margin_beats_required": float(np.mean(frac_hits)) if frac_hits else None,
        "macro_mean_margin_ratio": float(np.mean(ratios)) if ratios else None,
    }
    return out


def compute_hidden_arm_auc(collections: dict[str, dict], switch_eps_pct: float, use_switch_head: bool):
    out = {}
    macro_auc = []
    for problem, col in collections.items():
        gate_prob = col["switch_prob"] if use_switch_head else col["pair_prob"]
        arm_metrics = {}
        for arm_idx, name in enumerate(col["pool_names"]):
            hidden_mask = col["cand_hidden_flag"][col["cand_local_ids"] == arm_idx]
            if hidden_mask.size == 0 or np.max(hidden_mask) <= 0:
                continue
            pos = (col["cand_local_ids"] == arm_idx)
            rows, cols = np.where(pos)
            score = gate_prob[rows, cols]
            true_gain = ((col["base_selected_cost"][rows] - col["cand_costs"][rows, cols]) / (np.abs(col["base_selected_cost"][rows]) + 1e-9)) * 100.0
            label = (true_gain > switch_eps_pct).astype(np.int64)
            auc = binary_roc_auc(score, label)
            pr = binary_pr_auc(score, label)
            arm_metrics[name] = {
                "n": int(len(score)),
                "positives": int(label.sum()),
                "roc_auc": auc,
                "pr_auc": pr,
            }
            if auc is not None:
                macro_auc.append(auc)
        out[problem] = arm_metrics
    out["macro"] = {"macro_hidden_arm_auc": float(np.mean(macro_auc)) if macro_auc else None}
    return out


def sweep_thresholds(collections: dict[str, dict], prob_grid: list[float], gain_grid: list[float], use_switch_head: bool):
    curves = []
    for prob_thr in prob_grid:
        for gain_thr in gain_grid:
            metrics = metrics_all_from_collections(collections, prob_thr, gain_thr, use_switch_head)
            curves.append({
                "config": {
                    "switch_prob_thr": float(prob_thr),
                    "gain_thr_pct": float(gain_thr),
                    "gate_source": "switch" if use_switch_head else "pair",
                },
                "macro": metrics["macro"],
                "per_problem": metrics["per_problem"],
            })
    return curves


def collect_all_predictions(base_model, reranker, split: str, device: str, audit: dict, problem_stats: dict,
                            solver_profiles: dict, args, problems: list[str]):
    return {
        problem: collect_problem_predictions(
            base_model,
            reranker,
            problem,
            split,
            device,
            audit,
            problem_stats,
            solver_profiles,
            topk=args.rerank_topk,
            instance_max_tokens=args.instance_max_tokens,
            base_topk=args.base_topk,
            support_topk=args.support_candidate_topk,
            full_pool=args.full_pool,
            support_feature_mode=args.support_feature_mode,
        )
        for problem in problems
    }


def evaluate_checkpoint(tag: str, save_dir: Path, base_model, reranker, device: str, audit: dict,
                        problem_stats: dict, solver_profiles: dict, args):
    problems = [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else list(PROBLEMS)
    for split in ["val", "test"]:
        collections = collect_all_predictions(base_model, reranker, split, device, audit, problem_stats, solver_profiles, args, problems)
        curves = sweep_thresholds(
            collections,
            prob_grid=parse_float_grid(args.switch_prob_grid, [0.55, 0.65, 0.75, 0.85, 0.90]),
            gain_grid=parse_float_grid(args.switch_gain_grid, [0.0, 0.03, 0.05, 0.10, 0.20]),
            use_switch_head=bool(args.use_switch_head),
        )
        frontier = build_frontier(curves, safe_harm_budgets=[0.0, 0.01, 0.03, 0.05, 0.10])
        best = choose_best_frontier(curves, safe_harm_budget=args.safe_harm_budget)
        payload = {
            "best": best,
            "frontier": frontier,
            "n_points": len(curves),
        }
        (save_dir / f"switch_curves_{split}_{tag}.json").write_text(json.dumps(curves, indent=2))
        (save_dir / f"frontier_{split}_{tag}.json").write_text(json.dumps(payload, indent=2))
        (save_dir / f"required_delta_to_flip_{split}_{tag}.json").write_text(
            json.dumps(compute_required_delta_to_flip(collections), indent=2)
        )
        (save_dir / f"hidden_arm_auc_{split}_{tag}.json").write_text(
            json.dumps(compute_hidden_arm_auc(collections, switch_eps_pct=args.switch_eps_pct, use_switch_head=bool(args.use_switch_head)), indent=2)
        )
        analysis = {
            "config": best["config"],
            "per_problem": best["per_problem"],
            "macro": best["macro"],
        }
        print(
            format_eval_block(
                f"[{split} {tag}]",
                best["macro"],
                best["per_problem"],
                problem_order=problems,
                extra_macro_fields=[
                    ("gate_prob", f"{best['config']['switch_prob_thr']:.3f}"),
                    ("gain_thr", f"{best['config']['gain_thr_pct']:.3f}"),
                ],
            ),
            flush=True,
        )
        (save_dir / f"analysis_{split}_{tag}.json").write_text(json.dumps(analysis, indent=2))
        if tag == "best_frontier_safe":
            (save_dir / f"analysis_{split}.json").write_text(json.dumps(analysis, indent=2))
            (save_dir / f"frontier_{split}.json").write_text(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-per-problem", type=int, default=128)
    ap.add_argument("--instance-max-tokens", type=int, default=64)
    ap.add_argument("--profile-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rerank-topk", type=int, default=10)
    ap.add_argument("--rerank-d-model", type=int, default=128)
    ap.add_argument("--rerank-heads", type=int, default=4)
    ap.add_argument("--rerank-layers", type=int, default=2)
    ap.add_argument("--rerank-cross-layers", type=int, default=1)
    ap.add_argument("--pair-loss-weight", type=float, default=0.35)
    ap.add_argument("--gain-loss-weight", type=float, default=0.25)
    ap.add_argument("--listwise-loss-weight", type=float, default=0.20)
    ap.add_argument("--switch-loss-weight", type=float, default=0.10)
    ap.add_argument("--calibration-loss-weight", type=float, default=0.10)
    ap.add_argument("--base-topk", type=int, default=2)
    ap.add_argument("--support-candidate-topk", type=int, default=10)
    ap.add_argument("--hidden-candidate-weight", type=float, default=1.5)
    ap.add_argument("--switch-gain-weight", type=float, default=0.20)
    ap.add_argument("--decisive-gain-pct", type=float, default=0.20)
    ap.add_argument("--decisive-candidate-weight", type=float, default=1.0)
    ap.add_argument("--pair-eps-pct", type=float, default=0.05)
    ap.add_argument("--switch-eps-pct", type=float, default=0.10)
    ap.add_argument("--switch-prob-grid", default="0.55,0.65,0.75,0.85,0.90")
    ap.add_argument("--switch-gain-grid", default="0.0,0.03,0.05,0.10,0.20")
    ap.add_argument("--safe-harm-budget", type=float, default=0.03)
    ap.add_argument("--coord-augment", type=int, default=0)
    ap.add_argument("--max-batches-per-problem", type=int, default=0)
    ap.add_argument("--problems", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full-pool", action="store_true")
    ap.add_argument("--support-feature-mode", choices=["none", "max", "per_generator"], default="per_generator")
    ap.add_argument("--prior-cache", type=str, default="")
    ap.add_argument("--profile-cache", type=str, default="")
    ap.add_argument("--warmstart-r27a-ckpt", type=str, default="")
    ap.add_argument("--use-switch-head", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    problems = [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else list(PROBLEMS)

    base_model, _ = build_base_model(args.base_ckpt, args.device)
    audit = json.loads(Path("code/unified_selector/runs/audit.json").read_text())
    prior_cache = Path(args.prior_cache) if args.prior_cache else save_dir / "problem_solver_stats.json"
    problem_stats = load_or_build_problem_stats(base_model, audit, args.device, prior_cache)
    profile_cache = Path(args.profile_cache) if args.profile_cache else save_dir / "solver_profiles.pt"
    profile_payload = load_or_build_solver_profiles(
        base_model, args.device, problem_stats, profile_cache, profile_batch_size=args.profile_batch_size
    )
    solver_profiles = profile_payload["profiles"]
    base_val = evaluate_base_all(base_model, "val", args.device, audit)
    (save_dir / "base_val_metrics.json").write_text(json.dumps(base_val, indent=2))

    sample_ds = UnifiedProblemDataset(PROBLEMS[0], "train")
    sample_batch = collate_single_problem([sample_ds[0], sample_ds[1]])
    sample_batch = to_device(sample_batch, args.device)
    sample_meta = build_context_meta(
        base_model,
        sample_batch,
        problem_stats=problem_stats,
        solver_profiles=solver_profiles,
        audit=audit,
        topk=args.rerank_topk,
        instance_max_tokens=args.instance_max_tokens,
        base_topk=args.base_topk,
        support_topk=args.support_candidate_topk,
        full_pool=args.full_pool,
        support_feature_mode=args.support_feature_mode,
    )
    support_dim = sample_meta["cand_support_per_gen"].shape[-1]
    reranker = PairWinSwitchReranker(
        cand_dim=sample_meta["token_feats"].shape[-1],
        inst_dim=sample_meta["inst_tokens"].shape[-1],
        solver_dim=sample_meta["solver_embs"].shape[-1],
        profile_dim=sample_meta["solver_profiles"].shape[-1],
        cond_dim=sample_meta["cond_vec"].shape[-1],
        support_dim=support_dim,
        d_model=args.rerank_d_model,
        heads=args.rerank_heads,
        cross_layers=args.rerank_cross_layers,
        self_layers=args.rerank_layers,
        max_topk=max(args.rerank_topk, 2),
    ).to(args.device)
    if args.warmstart_r27a_ckpt:
        init_ckpt = torch.load(args.warmstart_r27a_ckpt, map_location=args.device)
        init_state = init_ckpt.get("reranker", init_ckpt)
        missing, unexpected, skipped, remapped = load_partial_state_dict(reranker, init_state)
        print(
            f"[warmstart] loaded {args.warmstart_r27a_ckpt} missing={missing} unexpected={unexpected} "
            f"skipped={skipped} remapped={remapped}",
            flush=True,
        )
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.wd)

    best_frontier_score = -1e18
    best_cost = 1e18
    step = 0

    for epoch in range(args.epochs):
        for problem in problems:
            ds = UnifiedProblemDataset(problem, "train", coord_augment=args.coord_augment)
            dl = DataLoader(
                ds,
                batch_size=args.batch_per_problem,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_single_problem,
                drop_last=True,
            )
            for i, batch in enumerate(dl):
                if args.max_batches_per_problem > 0 and i >= args.max_batches_per_problem:
                    break
                batch = to_device(batch, args.device)
                meta = build_context_meta(
                    base_model,
                    batch,
                    problem_stats=problem_stats,
                    solver_profiles=solver_profiles,
                    audit=audit,
                    topk=args.rerank_topk,
                    instance_max_tokens=args.instance_max_tokens,
                    base_topk=args.base_topk,
                    support_topk=args.support_candidate_topk,
                    full_pool=args.full_pool,
                    support_feature_mode=args.support_feature_mode,
                )
                pair_logit, gain_pred, switch_logit = reranker(
                    meta,
                    meta["token_feats"],
                    meta["inst_tokens"],
                    meta["inst_mask"],
                    meta["solver_embs"],
                    meta["solver_profiles"],
                    meta["cond_vec"],
                )
                loss, parts = pairwin_switch_loss(meta, pair_logit, gain_pred, switch_logit, args)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    print(
                        f"ep{epoch} step{step} prob={problem} loss={loss.item():.4f} "
                        f"pair={parts['pair']:.4f} gain={parts['gain']:.4f} "
                        f"list={parts['list']:.4f} switch={parts['switch']:.4f} "
                        f"calib={parts['calib']:.4f}",
                        flush=True,
                    )

        val_collections = collect_all_predictions(base_model, reranker, "val", args.device, audit, problem_stats, solver_profiles, args, problems)
        val_curves = sweep_thresholds(
            val_collections,
            prob_grid=parse_float_grid(args.switch_prob_grid, [0.55, 0.65, 0.75, 0.85, 0.90]),
            gain_grid=parse_float_grid(args.switch_gain_grid, [0.0, 0.03, 0.05, 0.10, 0.20]),
            use_switch_head=bool(args.use_switch_head),
        )
        best_val = choose_best_frontier(val_curves, safe_harm_budget=args.safe_harm_budget)
        frontier_score = best_val["macro"]["macro_top1"] - 2.0 * max(0.0, best_val["macro"]["macro_safe_harm_rate"] - args.safe_harm_budget)
        print(
            format_eval_block(
                f"[eval epoch {epoch}]",
                best_val["macro"],
                best_val["per_problem"],
                problem_order=problems,
                extra_macro_fields=[
                    ("gate_prob", f"{best_val['config']['switch_prob_thr']:.3f}"),
                    ("gain_thr", f"{best_val['config']['gain_thr_pct']:.3f}"),
                    ("score", f"{frontier_score:.6f}"),
                ],
            ),
            flush=True,
        )
        payload = {
            "reranker": reranker.state_dict(),
            "base_ckpt": args.base_ckpt,
            "args": vars(args),
            "val_best": best_val,
            "frontier_score": frontier_score,
        }
        if best_val["macro"]["macro_vs_sbs_pct"] < best_cost:
            best_cost = best_val["macro"]["macro_vs_sbs_pct"]
            torch.save(payload, save_dir / "best_cost.pt")
            print(f"  -> saved best_cost (macro_vs_sbs={best_cost:+.3f}%)", flush=True)
        if frontier_score > best_frontier_score:
            best_frontier_score = frontier_score
            torch.save(payload, save_dir / "best_frontier_safe.pt")
            torch.save(payload, save_dir / "best.pt")
            print(f"  -> saved best_frontier_safe (score={best_frontier_score:.6f})", flush=True)

    for tag in ["best_frontier_safe", "best_cost"]:
        ckpt_path = save_dir / f"{tag}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=args.device)
        reranker.load_state_dict(ckpt["reranker"])
        evaluate_checkpoint(tag, save_dir, base_model, reranker, args.device, audit, problem_stats, solver_profiles, args)


if __name__ == "__main__":
    main()
