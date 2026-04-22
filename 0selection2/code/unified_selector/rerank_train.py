from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import UnifiedProblemDataset, collate_single_problem
from .eval_log import format_eval_block
from .model import UnifiedSelector, load_partial_state_dict
from .registry import PROBLEMS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def weighted_mean(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if weight is None:
        return values.mean()
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def parse_float_grid(spec: str | None, default: list[float]) -> list[float]:
    if spec is None:
        return list(default)
    vals = []
    for token in spec.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        vals.append(float("inf") if tok in {"inf", "+inf"} else float(tok))
    return vals or list(default)


class CandidateSetReranker(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, max_topk: int = 10, delta_init_scale: float = 1e-3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pos_emb = nn.Embedding(max_topk, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=2 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(layers)
        ])
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.gap_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.delta_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(max(delta_init_scale, 1e-6))), dtype=torch.float32))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, token_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, k, _ = token_feats.shape
        x = self.in_proj(token_feats)
        pos = torch.arange(k, device=token_feats.device).unsqueeze(0).expand(bsz, k)
        x = x + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        delta_scale = F.softplus(self.delta_scale_raw)
        delta = delta_scale * torch.tanh(self.delta_head(x).squeeze(-1))
        gap = self.gap_head(x).squeeze(-1)
        return delta, gap


def build_base_model(base_ckpt: str, device: str):
    ckpt = torch.load(base_ckpt, map_location=device)
    args = ckpt.get("args", {})
    model = UnifiedSelector(
        d=args.get("d", 128),
        depth=args.get("depth", 4),
        dropout=args.get("dropout", 0.1),
        use_mvrp_factorized=not args.get("no_fact", False),
        use_problem_solver_bias="problem_solver_bias" in ckpt["model"],
        use_problem_film=bool(args.get("problem_film", False)),
        use_size_feature=bool(args.get("size_feature", False)),
        rich_pool=bool(args.get("rich_pool", False)),
        use_global_stats=bool(args.get("global_stats", False)),
        use_manual_features=bool(args.get("manual_features", False)),
        use_constraint_experts=bool(args.get("constraint_experts", False)),
        use_problem_residual_head=bool(args.get("problem_residual_head", False)),
        use_problem_adapter=bool(args.get("problem_adapter", False)),
        use_support_head=bool(args.get("support_head", False)),
        support_hidden=int(args.get("support_hidden", 128)),
        support_generators=int(args.get("support_generators", 1)),
        adapter_hidden=int(args.get("adapter_hidden", 64)),
        coord_hier_pool=bool(args.get("coord_hier_pool", False)),
        coord_downsample_ratio=float(args.get("coord_downsample_ratio", 0.8)),
        deep_encoder_overhaul=bool(args.get("deep_encoder_overhaul", False)),
        expanded_mbm=bool(args.get("expanded_mbm", False)),
        encoder_rezero=bool(args.get("encoder_rezero", False)),
        encoder_constraint_experts=bool(args.get("encoder_constraint_experts", False)),
        encoder_constraint_hidden=int(args.get("encoder_constraint_hidden", 128)),
        solver_query_scorer=bool(args.get("solver_query_scorer", False)),
        full_pool_set_scorer=bool(args.get("full_pool_set_scorer", False)),
        hidden_local_head=bool(args.get("hidden_local_weight", 0.0) > 0),
        hidden_local_inference_weight=float(args.get("hidden_local_inference_weight", 0.0)),
    ).to(device)
    missing, unexpected, skipped, remapped = load_partial_state_dict(model, ckpt["model"])
    if any(k.startswith("support_problem_heads.") for k in missing):
        for name, param in model.named_parameters():
            if name.startswith("support_problem_heads."):
                nn.init.zeros_(param)
    if any(k.startswith("support_constraint_expert_heads.") for k in missing):
        for name, param in model.named_parameters():
            if name.startswith("support_constraint_expert_heads."):
                nn.init.zeros_(param)
    if missing or unexpected or skipped or remapped:
        print(f"[load base] missing={missing} unexpected={unexpected} skipped={skipped} remapped={remapped}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt


def build_problem_solver_stats(problem: str, audit: dict) -> dict[str, list[float] | int]:
    ds = UnifiedProblemDataset(problem, "train")
    all_costs = []
    for i in range(ds.base_N):
        all_costs.append(torch.tensor(ds.labels[str(i)]["cost"][:ds.K_p], dtype=torch.float32))
    costs = torch.stack(all_costs, dim=0)
    sbs_idx = int(audit[problem]["sbs_pool_idx"])
    sbs_cost = costs[:, sbs_idx:sbs_idx + 1]
    gap = (costs - sbs_cost) / (sbs_cost.abs() + 1e-9)
    winner = costs.argmin(dim=1)
    rank = costs.argsort(dim=1)
    counts = torch.bincount(winner, minlength=ds.K_p).float()
    oracle_win_prior = (counts / counts.sum().clamp_min(1.0)).tolist()
    mean_gap_vs_sbs = gap.mean(dim=0).tolist()
    rank1 = []
    rank2 = []
    rank3 = []
    for k in range(ds.K_p):
        rank1.append(float((rank[:, 0] == k).float().mean().item()))
        rank2.append(float((rank[:, :min(2, ds.K_p)] == k).any(dim=1).float().mean().item()))
        rank3.append(float((rank[:, :min(3, ds.K_p)] == k).any(dim=1).float().mean().item()))
    return {
        "sbs_pool_idx": sbs_idx,
        "oracle_win_prior": oracle_win_prior,
        "mean_gap_vs_sbs": mean_gap_vs_sbs,
        "rank_hist": [rank1, rank2, rank3],
        "pick_rate": [0.0] * ds.K_p,
        "hidden_winner_flag": [0.0] * ds.K_p,
    }


@torch.no_grad()
def add_base_pick_stats(base_model: UnifiedSelector, problem: str, device: str,
                        stats: dict[str, list[float] | int], batch_size: int = 64):
    ds = UnifiedProblemDataset(problem, "train")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
    pick_counts = torch.zeros(ds.K_p, dtype=torch.float32)
    for batch in dl:
        batch = to_device(batch, device)
        logits = base_model(batch)
        logits_pool = logits[:, batch["pool_ids"]]
        pred = logits_pool.argmax(dim=1).cpu()
        pick_counts += torch.bincount(pred, minlength=ds.K_p).float()
    pick_rate = pick_counts / pick_counts.sum().clamp_min(1.0)
    oracle_win_prior = torch.tensor(stats["oracle_win_prior"], dtype=torch.float32)
    hidden = ((oracle_win_prior > 0) & (pick_rate <= 0)).float()
    stats["pick_rate"] = pick_rate.tolist()
    stats["hidden_winner_flag"] = hidden.tolist()


def load_or_build_problem_stats(base_model: UnifiedSelector, audit: dict, device: str,
                                cache_path: Path | None) -> dict[str, dict]:
    if cache_path is not None and cache_path.exists():
        print(f"[prior-cache] loading {cache_path}", flush=True)
        return json.loads(cache_path.read_text())
    stats = {}
    for problem in PROBLEMS:
        print(f"[prior-cache] building stats for {problem}", flush=True)
        cur = build_problem_solver_stats(problem, audit)
        add_base_pick_stats(base_model, problem, device, cur)
        stats[problem] = cur
    if cache_path is not None:
        cache_path.write_text(json.dumps(stats, indent=2))
        print(f"[prior-cache] wrote {cache_path}", flush=True)
    return stats


def gather_problem_stats(problem: str, problem_stats: dict, device: str) -> dict[str, torch.Tensor | int]:
    cur = problem_stats[problem]
    rank_hist = torch.tensor(cur["rank_hist"], dtype=torch.float32, device=device).t().contiguous()
    return {
        "sbs_pool_idx": int(cur["sbs_pool_idx"]),
        "oracle_win_prior": torch.tensor(cur["oracle_win_prior"], dtype=torch.float32, device=device),
        "mean_gap_vs_sbs": torch.tensor(cur["mean_gap_vs_sbs"], dtype=torch.float32, device=device),
        "rank_hist": rank_hist,
        "pick_rate": torch.tensor(cur["pick_rate"], dtype=torch.float32, device=device),
        "hidden_winner_flag": torch.tensor(cur["hidden_winner_flag"], dtype=torch.float32, device=device),
    }


def build_candidate_set(base_model: UnifiedSelector, batch, problem_stats: dict, audit: dict,
                        topk: int, base_topk: int | None = None, support_topk: int = 0,
                        full_pool: bool = False, support_feature_mode: str = "max"):
    z, _ = base_model.compute_pair_features(batch)
    if getattr(base_model, "use_support_head", False):
        logits, support_logits = base_model(batch, return_support=True)
    else:
        logits = base_model(batch)
        support_logits = None

    problem = PROBLEMS[batch["problem_id"]]
    stats = gather_problem_stats(problem, problem_stats, logits.device)
    logits_pool = logits[:, batch["pool_ids"]]
    z_pool = z[:, batch["pool_ids"]]
    probs_pool = logits_pool.softmax(dim=1)
    bsz, k_pool = logits_pool.shape
    topk = min(max(1, topk), k_pool)
    base_topk = min(k_pool, max(1, base_topk or topk))

    base_idx = logits_pool.topk(base_topk, dim=1).indices
    sbs_idx = stats["sbs_pool_idx"]

    if support_logits is not None:
        if support_logits.dim() == 2:
            support_per_gen = torch.sigmoid(support_logits[:, batch["pool_ids"]]).unsqueeze(1)
        else:
            support_per_gen = torch.sigmoid(support_logits[:, :, batch["pool_ids"]])
    else:
        support_per_gen = logits_pool.new_zeros((bsz, 0, k_pool))

    support_prob_max = support_per_gen.max(dim=1).values if support_per_gen.shape[1] > 0 else torch.zeros_like(probs_pool)
    if support_topk > 0 and support_prob_max.shape[1] > 0:
        support_idx = support_prob_max.topk(min(k_pool, support_topk), dim=1).indices
    else:
        support_idx = None

    if full_pool:
        top_idx = torch.argsort(logits_pool, dim=1, descending=True)
    else:
        rows = []
        for i in range(bsz):
            row = []
            for src in (base_idx[i], support_idx[i] if support_idx is not None else []):
                for val in src.tolist():
                    if val not in row:
                        row.append(int(val))
                    if len(row) >= topk:
                        break
                if len(row) >= topk:
                    break
            if len(row) < topk:
                for val in logits_pool[i].argsort(descending=True).tolist():
                    if val not in row:
                        row.append(int(val))
                    if len(row) >= topk:
                        break
            rows.append(torch.tensor(row, dtype=torch.long, device=logits_pool.device))
        top_idx = torch.stack(rows, dim=0)

    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, z_pool.shape[-1])
    cand_z = torch.gather(z_pool, 1, gather_idx)
    cand_base = torch.gather(logits_pool, 1, top_idx)
    cand_prob = torch.gather(probs_pool, 1, top_idx)

    if support_per_gen.shape[1] > 0:
        gather_support = top_idx.unsqueeze(1).expand(-1, support_per_gen.shape[1], -1)
        cand_support_per_gen = torch.gather(support_per_gen, 2, gather_support)
        cand_support_max = cand_support_per_gen.max(dim=1).values
    else:
        cand_support_per_gen = logits_pool.new_zeros((bsz, 0, top_idx.shape[1]))
        cand_support_max = torch.zeros_like(cand_prob)

    support_src = []
    rescued_src = []
    for i in range(bsz):
        base_set = set(base_idx[i].tolist())
        support_set = set(support_idx[i].tolist()) if support_idx is not None else set()
        row = top_idx[i].tolist()
        support_src.append(torch.tensor([float(v in support_set) for v in row], device=logits_pool.device))
        rescued_src.append(torch.tensor([float((v in support_set) and (v not in base_set)) for v in row], device=logits_pool.device))
    support_src = torch.stack(support_src, dim=0)
    rescued_src = torch.stack(rescued_src, dim=0)

    sbs_logit = logits_pool[:, sbs_idx:sbs_idx + 1]
    sbs_prob = probs_pool[:, sbs_idx:sbs_idx + 1]
    base_delta_sbs = cand_base - sbs_logit
    prob_delta_sbs = cand_prob - sbs_prob
    is_sbs = (top_idx == sbs_idx).float()
    rank_frac = torch.arange(top_idx.shape[1], device=logits_pool.device, dtype=logits_pool.dtype)
    rank_frac = rank_frac / max(1, top_idx.shape[1] - 1)
    rank_frac = rank_frac.view(1, -1).expand(bsz, -1)

    oracle_prior = stats["oracle_win_prior"].unsqueeze(0).expand(bsz, -1).gather(1, top_idx)
    mean_gap = stats["mean_gap_vs_sbs"].unsqueeze(0).expand(bsz, -1).gather(1, top_idx)
    hidden_flag = stats["hidden_winner_flag"].unsqueeze(0).expand(bsz, -1).gather(1, top_idx)
    rank_hist = stats["rank_hist"].unsqueeze(0).expand(bsz, -1, -1)
    rank_hist = torch.gather(rank_hist, 1, top_idx.unsqueeze(-1).expand(-1, -1, rank_hist.shape[-1]))

    prob_id = torch.full((bsz,), batch["problem_id"], dtype=torch.long, device=logits_pool.device)
    prob_emb = base_model.prob_emb(prob_id).unsqueeze(1).expand(-1, top_idx.shape[1], -1)
    global_stats = base_model.compute_global_stats(batch).unsqueeze(1).expand(-1, top_idx.shape[1], -1)
    manual_feats = base_model.compute_manual_features(batch).unsqueeze(1).expand(-1, top_idx.shape[1], -1)
    cbits = batch["cbits"].unsqueeze(1).expand(-1, top_idx.shape[1], -1)

    token_parts = [
        cand_z,
        cand_base.unsqueeze(-1),
        cand_prob.unsqueeze(-1),
        rank_frac.unsqueeze(-1),
        base_delta_sbs.unsqueeze(-1),
        prob_delta_sbs.unsqueeze(-1),
        is_sbs.unsqueeze(-1),
        cand_support_max.unsqueeze(-1),
    ]
    if support_feature_mode == "per_generator" and cand_support_per_gen.shape[1] > 0:
        token_parts.append(cand_support_per_gen.transpose(1, 2))
    token_parts.extend([
        support_src.unsqueeze(-1),
        rescued_src.unsqueeze(-1),
        oracle_prior.unsqueeze(-1),
        mean_gap.unsqueeze(-1),
        hidden_flag.unsqueeze(-1),
        rank_hist,
        cbits,
        prob_emb,
        global_stats,
        manual_feats,
    ])
    token_feats = torch.cat(token_parts, dim=-1)

    cand_costs = torch.gather(batch["costs"], 1, top_idx)
    all_gap = (batch["costs"] - batch["costs"][:, sbs_idx:sbs_idx + 1]) / (batch["costs"][:, sbs_idx:sbs_idx + 1].abs() + 1e-9)
    cand_gap_target = torch.gather(all_gap, 1, top_idx)
    best_idx = batch["costs"].argmin(dim=1)
    base_pred = logits_pool.argmax(dim=1)
    base_pred_cand_pos = (top_idx == base_pred.unsqueeze(1)).float().argmax(dim=1)
    base_wrong = base_pred != best_idx
    base_selected_cost = batch["costs"].gather(1, base_pred.unsqueeze(1)).squeeze(1)
    oracle_cost = batch["costs"].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    switch_gain = (base_selected_cost - oracle_cost).clamp_min(0.0)
    switch_gain_pct = switch_gain / (base_selected_cost.abs() + 1e-9) * 100.0
    oracle_in_topk = (top_idx == best_idx.unsqueeze(1)).any(dim=1)
    base_contains_oracle = (base_idx == best_idx.unsqueeze(1)).any(dim=1)
    support_contains_oracle = (support_idx == best_idx.unsqueeze(1)).any(dim=1) if support_idx is not None else torch.zeros_like(oracle_in_topk)
    probs_sorted = probs_pool.topk(min(2, k_pool), dim=1).values
    if probs_sorted.shape[1] == 1:
        base_prob_gap = torch.ones_like(probs_sorted[:, 0])
    else:
        base_prob_gap = probs_sorted[:, 0] - probs_sorted[:, 1]

    return {
        "problem": problem,
        "pool_names": audit[problem]["pool"],
        "sbs_pool_idx": sbs_idx,
        "logits_pool": logits_pool,
        "top_idx": top_idx,
        "cand_base": cand_base,
        "cand_prob": cand_prob,
        "cand_support_max": cand_support_max,
        "cand_support_per_gen": cand_support_per_gen.transpose(1, 2).contiguous(),
        "cand_support_src": support_src,
        "cand_rescued_src": rescued_src,
        "cand_rank_frac": rank_frac,
        "cand_is_sbs": is_sbs,
        "cand_base_delta_sbs": base_delta_sbs,
        "cand_prob_delta_sbs": prob_delta_sbs,
        "cand_gap_target": cand_gap_target,
        "cand_oracle_prior": oracle_prior,
        "cand_hidden_flag": hidden_flag,
        "token_feats": token_feats,
        "cand_costs": cand_costs,
        "oracle_in_topk": oracle_in_topk,
        "base_contains_oracle": base_contains_oracle,
        "support_contains_oracle": support_contains_oracle,
        "base_wrong": base_wrong,
        "base_pred_local": base_pred,
        "base_pred_cand_pos": base_pred_cand_pos,
        "winner_local": best_idx,
        "base_selected_cost": base_selected_cost,
        "oracle_cost": oracle_cost,
        "switch_gain": switch_gain,
        "switch_gain_pct": switch_gain_pct,
        "base_prob_gap": base_prob_gap,
        "support_rescued": (support_contains_oracle & ~base_contains_oracle) if support_idx is not None else torch.zeros_like(oracle_in_topk),
    }


def rerank_loss(delta_scores: torch.Tensor, gap_scores: torch.Tensor,
                cand_base: torch.Tensor, cand_gap_target: torch.Tensor, cand_costs: torch.Tensor,
                example_weight: torch.Tensor | None, pair_mask: torch.Tensor, ce_mask: torch.Tensor,
                safe_mask: torch.Tensor, pair_weight: float, gap_weight: float,
                ce_weight: float, anchor_weight: float, anchor_mode: str) -> tuple[torch.Tensor, dict[str, float]]:
    final_scores = cand_base + delta_scores
    winner = cand_costs.argmin(dim=1)

    loss_ce = final_scores.new_tensor(0.0)
    if ce_weight > 0.0 and ce_mask.any():
        ce_raw = F.cross_entropy(final_scores, winner, reduction="none")
        ce_weight_vec = example_weight[ce_mask] if example_weight is not None else None
        loss_ce = weighted_mean(ce_raw[ce_mask], ce_weight_vec)

    gap_raw = F.smooth_l1_loss(gap_scores, cand_gap_target, reduction="none").mean(dim=1)
    loss_gap = weighted_mean(gap_raw, example_weight)

    loss_pair = final_scores.new_tensor(0.0)
    if pair_mask.any():
        final_fix = final_scores[pair_mask]
        gap_fix = cand_gap_target[pair_mask]
        sign = torch.sign(gap_fix.unsqueeze(1) - gap_fix.unsqueeze(2))
        score_diff = final_fix.unsqueeze(2) - final_fix.unsqueeze(1)
        gap_diff = (gap_fix.unsqueeze(1) - gap_fix.unsqueeze(2)).abs()
        weight = torch.clamp(gap_diff / 0.05, max=5.0)
        tri = torch.triu(torch.ones(final_fix.shape[1], final_fix.shape[1], dtype=torch.bool, device=final_fix.device), diagonal=1)
        pairwise_mask = (sign != 0) & tri.unsqueeze(0)
        if pairwise_mask.any():
            pair_terms = F.softplus(-sign * score_diff) * weight
            if example_weight is not None:
                per_ex = []
                ex_w = example_weight[pair_mask].to(final_fix.dtype)
                for i in range(final_fix.shape[0]):
                    mask_i = pairwise_mask[i]
                    if mask_i.any():
                        per_ex.append(pair_terms[i].masked_select(mask_i).mean())
                    else:
                        per_ex.append(final_fix.new_tensor(0.0))
                loss_pair = weighted_mean(torch.stack(per_ex), ex_w)
            else:
                loss_pair = pair_terms.masked_select(pairwise_mask).mean()

    loss_anchor = final_scores.new_tensor(0.0)
    if anchor_weight > 0.0 and safe_mask.any():
        if anchor_mode == "delta":
            anchor_raw = F.smooth_l1_loss(
                delta_scores[safe_mask],
                torch.zeros_like(delta_scores[safe_mask]),
                reduction="none",
            ).mean(dim=1)
        elif anchor_mode == "kl":
            base_prob = cand_base[safe_mask].softmax(dim=1)
            final_logprob = final_scores[safe_mask].log_softmax(dim=1)
            anchor_raw = F.kl_div(final_logprob, base_prob, reduction="none").sum(dim=1)
        else:
            raise ValueError(f"Unsupported anchor_mode={anchor_mode}")
        safe_weight = example_weight[safe_mask] if example_weight is not None else None
        loss_anchor = weighted_mean(anchor_raw, safe_weight)

    total = pair_weight * loss_pair + gap_weight * loss_gap + ce_weight * loss_ce + anchor_weight * loss_anchor
    return total, {
        "pair": float(loss_pair.item()),
        "gap": float(loss_gap.item()),
        "ce": float(loss_ce.item()),
        "anchor": float(loss_anchor.item()),
    }


def apply_rerank(meta, delta: torch.Tensor, blend_alpha: float = 1.0,
                 base_gap_max: float = float("inf"), rerank_margin_min: float = 0.0):
    rerank_scores = meta["cand_base"] + blend_alpha * delta
    if rerank_scores.shape[1] >= 2:
        top2 = rerank_scores.topk(2, dim=1).values
        rerank_margin = top2[:, 0] - top2[:, 1]
    else:
        rerank_margin = torch.full_like(meta["base_prob_gap"], float("inf"))
    gate = meta["base_prob_gap"] <= base_gap_max
    if rerank_margin_min > 0:
        gate = gate & (rerank_margin >= rerank_margin_min)
    final_scores = meta["cand_base"] + gate.unsqueeze(1).to(delta.dtype) * blend_alpha * delta
    final_pool = meta["logits_pool"].clone()
    final_pool.scatter_(1, meta["top_idx"], final_scores)
    return final_pool, gate, rerank_margin


@torch.no_grad()
def evaluate_problem(base_model, reranker, problem: str, split: str, device: str, audit: dict,
                     problem_stats: dict, topk: int, blend_alpha: float = 1.0,
                     base_gap_max: float = float("inf"), rerank_margin_min: float = 0.0,
                     base_topk: int | None = None, support_topk: int = 0,
                     full_pool: bool = False, support_feature_mode: str = "max") -> dict:
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_single_problem)

    top1 = top2 = top3 = 0
    cost_sum = 0.0
    total = 0
    applied = 0
    fixable_recovered = 0
    fixable_total = 0
    safe_harm = 0
    safe_total = 0
    decisive_hit = {0.1: [0, 0], 0.5: [0, 0], 1.0: [0, 0]}
    picks = []
    support_seen = None
    oracle_rank_sum = 0.0
    oracle_rank_le2 = 0
    oracle_rank_le3 = 0
    oracle_support_mass = 0.0

    for batch in dl:
        batch = to_device(batch, device)
        meta = build_candidate_set(
            base_model,
            batch,
            problem_stats=problem_stats,
            audit=audit,
            topk=topk,
            base_topk=base_topk,
            support_topk=support_topk,
            full_pool=full_pool,
            support_feature_mode=support_feature_mode,
        )
        delta, _ = reranker(meta["token_feats"])
        final_pool, gate, _ = apply_rerank(
            meta,
            delta,
            blend_alpha=blend_alpha,
            base_gap_max=base_gap_max,
            rerank_margin_min=rerank_margin_min,
        )

        pred = final_pool.argmax(dim=1)
        best = batch["costs"].argmin(dim=1)
        pred_rank = final_pool.argsort(dim=1, descending=True)
        top1 += (pred == best).sum().item()
        top2 += (pred_rank[:, :min(2, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        top3 += (pred_rank[:, :min(3, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        sel_cost = batch["costs"].gather(1, pred.unsqueeze(1)).squeeze(1)
        cost_sum += sel_cost.sum().item()
        total += batch["costs"].shape[0]
        applied += int(gate.sum().item())

        fixable = meta["base_wrong"]
        safe = ~fixable
        if fixable.any():
            fixable_recovered += (pred[fixable] == best[fixable]).sum().item()
            fixable_total += int(fixable.sum().item())
        if safe.any():
            safe_harm += (pred[safe] != best[safe]).sum().item()
            safe_total += int(safe.sum().item())

        switch_gain_pct = meta["switch_gain_pct"]
        for thr in decisive_hit:
            mask = fixable & (switch_gain_pct > thr)
            if mask.any():
                decisive_hit[thr][0] += int(mask.sum().item())
                decisive_hit[thr][1] += int((pred[mask] == best[mask]).sum().item())

        rank_pos = (pred_rank == best.unsqueeze(1)).float().argmax(dim=1) + 1
        oracle_rank_sum += rank_pos.float().sum().item()
        oracle_rank_le2 += int((rank_pos <= 2).sum().item())
        oracle_rank_le3 += int((rank_pos <= 3).sum().item())

        oracle_support_mass += float(meta["oracle_in_topk"].float().sum().item())
        picks.append(pred.cpu().numpy())

        if support_seen is None:
            support_seen = torch.zeros(batch["costs"].shape[1], dtype=torch.bool)
        if meta["support_contains_oracle"] is not None:
            support_seen |= meta["support_contains_oracle"].new_zeros(batch["costs"].shape[1], dtype=torch.bool).cpu()

    pred = np.concatenate(picks)
    pool_names = audit[problem]["pool"]
    method_top1 = {}
    method_mean = {}

    ds_costs = []
    for i in range(ds.base_N):
        ds_costs.append(ds.labels[str(i)]["cost"][:ds.K_p])
    costs_np = np.asarray(ds_costs, dtype=np.float32)
    best_idx_np = costs_np.argmin(axis=1)
    for k, name in enumerate(pool_names):
        method_top1[name] = float((best_idx_np == k).mean())
        method_mean[name] = float(costs_np[:, k].mean())

    arm_dist = {name: float((pred == k).mean()) for k, name in enumerate(pool_names)}
    final_arm_coverage = float(np.mean([v > 0 for v in arm_dist.values()]))
    pick_entropy = float(-(np.array(list(arm_dist.values())) * np.log(np.array(list(arm_dist.values())) + 1e-12)).sum() / np.log(max(2, len(pool_names))))
    zero_pick_count = int(sum(v <= 0 for v in arm_dist.values()))
    hidden_winners = [name for name in pool_names if method_top1[name] > 0 and arm_dist[name] <= 0]
    hidden_winner_count = len(hidden_winners)
    hidden_winner_mass = float(sum(method_top1[name] for name in hidden_winners))

    mean_cost = cost_sum / max(1, total)
    sbs_cost = audit[problem][split]["sbs_mean"]
    vbs_mean = audit[problem][split]["vbs_mean"]
    return {
        "top1": top1 / total,
        "top2": top2 / total,
        "top3": top3 / total,
        "mean_cost": mean_cost,
        "sbs_cost": sbs_cost,
        "vbs_mean": vbs_mean,
        "vs_sbs_pct": (mean_cost - sbs_cost) / (abs(sbs_cost) + 1e-9) * 100,
        "vbs_gap_closed_pct": ((sbs_cost - mean_cost) / (sbs_cost - vbs_mean + 1e-9)) * 100 if sbs_cost != vbs_mean else 0.0,
        "rerank_apply_rate": applied / total,
        "fixable_recovery_rate": fixable_recovered / fixable_total if fixable_total > 0 else 0.0,
        "safe_harm_rate": safe_harm / safe_total if safe_total > 0 else 0.0,
        "n_fixable": fixable_total,
        "n_safe": safe_total,
        "oracle_rank_mean": oracle_rank_sum / total,
        "oracle_rank_top2_rate": oracle_rank_le2 / total,
        "oracle_rank_top3_rate": oracle_rank_le3 / total,
        "top1_decisive_gap_0.1": decisive_hit[0.1][1] / decisive_hit[0.1][0] if decisive_hit[0.1][0] > 0 else 0.0,
        "top1_decisive_gap_0.5": decisive_hit[0.5][1] / decisive_hit[0.5][0] if decisive_hit[0.5][0] > 0 else 0.0,
        "top1_decisive_gap_1.0": decisive_hit[1.0][1] / decisive_hit[1.0][0] if decisive_hit[1.0][0] > 0 else 0.0,
        "arm_distribution": arm_dist,
        "final_arm_coverage": final_arm_coverage,
        "pick_entropy": pick_entropy,
        "zero_pick_count": zero_pick_count,
        "hidden_winner_count": hidden_winner_count,
        "hidden_winner_mass": hidden_winner_mass,
        "oracle_support_mass_recall": oracle_support_mass / total,
        "method_top1": method_top1,
        "method_mean": method_mean,
    }


def summarize_macro(per_problem: dict[str, dict]) -> dict[str, float]:
    vals = list(per_problem.values())
    return {
        "macro_top1": float(np.mean([r["top1"] for r in vals])),
        "macro_top2": float(np.mean([r["top2"] for r in vals])),
        "macro_top3": float(np.mean([r["top3"] for r in vals])),
        "macro_vs_sbs_pct": float(np.mean([r["vs_sbs_pct"] for r in vals])),
        "macro_vbs_gap_closed_pct": float(np.mean([r["vbs_gap_closed_pct"] for r in vals])),
        "macro_rerank_apply_rate": float(np.mean([r["rerank_apply_rate"] for r in vals])),
        "macro_fixable_recovery_rate": float(np.mean([r["fixable_recovery_rate"] for r in vals])),
        "macro_safe_harm_rate": float(np.mean([r["safe_harm_rate"] for r in vals])),
        "macro_oracle_rank_mean": float(np.mean([r["oracle_rank_mean"] for r in vals])),
        "macro_oracle_rank_top2_rate": float(np.mean([r["oracle_rank_top2_rate"] for r in vals])),
        "macro_oracle_rank_top3_rate": float(np.mean([r["oracle_rank_top3_rate"] for r in vals])),
        "macro_top1_decisive_gap_0.1": float(np.mean([r["top1_decisive_gap_0.1"] for r in vals])),
        "macro_top1_decisive_gap_0.5": float(np.mean([r["top1_decisive_gap_0.5"] for r in vals])),
        "macro_top1_decisive_gap_1.0": float(np.mean([r["top1_decisive_gap_1.0"] for r in vals])),
        "macro_final_arm_coverage": float(np.mean([r["final_arm_coverage"] for r in vals])),
        "macro_pick_entropy": float(np.mean([r["pick_entropy"] for r in vals])),
        "macro_zero_pick_count": float(np.mean([r["zero_pick_count"] for r in vals])),
        "macro_hidden_winner_count": float(np.mean([r["hidden_winner_count"] for r in vals])),
        "macro_hidden_winner_mass": float(np.mean([r["hidden_winner_mass"] for r in vals])),
        "macro_oracle_support_mass_recall": float(np.mean([r["oracle_support_mass_recall"] for r in vals])),
    }


@torch.no_grad()
def evaluate_all(base_model, reranker, split: str, device: str, audit: dict, problem_stats: dict,
                 topk: int, blend_alpha: float = 1.0, base_gap_max: float = float("inf"),
                 rerank_margin_min: float = 0.0, base_topk: int | None = None,
                 support_topk: int = 0, full_pool: bool = False,
                 support_feature_mode: str = "max"):
    per = {}
    for problem in PROBLEMS:
        per[problem] = evaluate_problem(
            base_model,
            reranker,
            problem,
            split,
            device,
            audit,
            problem_stats,
            topk=topk,
            blend_alpha=blend_alpha,
            base_gap_max=base_gap_max,
            rerank_margin_min=rerank_margin_min,
            base_topk=base_topk,
            support_topk=support_topk,
            full_pool=full_pool,
            support_feature_mode=support_feature_mode,
        )
    return {"per_problem": per, "macro": summarize_macro(per)}


@torch.no_grad()
def evaluate_base_problem(base_model, problem: str, split: str, device: str, audit: dict) -> dict:
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
    top1 = top2 = top3 = 0
    cost_sum = 0.0
    total = 0
    picks = []
    all_costs = []
    for batch in dl:
        batch = to_device(batch, device)
        logits = base_model(batch)
        logits_pool = logits[:, batch["pool_ids"]]
        pred = logits_pool.argmax(dim=1)
        best = batch["costs"].argmin(dim=1)
        pred_rank = logits_pool.argsort(dim=1, descending=True)
        top1 += (pred == best).sum().item()
        top2 += (pred_rank[:, :min(2, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        top3 += (pred_rank[:, :min(3, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        sel_cost = batch["costs"].gather(1, pred.unsqueeze(1)).squeeze(1)
        cost_sum += sel_cost.sum().item()
        total += batch["costs"].shape[0]
        picks.append(pred.cpu().numpy())
        all_costs.append(batch["costs"].cpu().numpy())
    pred = np.concatenate(picks)
    costs_np = np.concatenate(all_costs)
    best_idx_np = costs_np.argmin(axis=1)
    pool_names = audit[problem]["pool"]
    method_top1 = {name: float((best_idx_np == i).mean()) for i, name in enumerate(pool_names)}
    arm_dist = {name: float((pred == i).mean()) for i, name in enumerate(pool_names)}
    hidden_winners = [name for name in pool_names if method_top1[name] > 0 and arm_dist[name] <= 0]
    hidden_winner_mass = float(sum(method_top1[name] for name in hidden_winners))
    mean_cost = cost_sum / total
    sbs_cost = audit[problem][split]["sbs_mean"]
    vbs_mean = audit[problem][split]["vbs_mean"]
    return {
        "top1": top1 / total,
        "top2": top2 / total,
        "top3": top3 / total,
        "mean_cost": mean_cost,
        "vs_sbs_pct": (mean_cost - sbs_cost) / (abs(sbs_cost) + 1e-9) * 100,
        "vbs_gap_closed_pct": ((sbs_cost - mean_cost) / (sbs_cost - vbs_mean + 1e-9)) * 100 if sbs_cost != vbs_mean else 0.0,
        "hidden_winner_mass": hidden_winner_mass,
    }


@torch.no_grad()
def evaluate_base_all(base_model, split: str, device: str, audit: dict):
    per = {problem: evaluate_base_problem(base_model, problem, split, device, audit) for problem in PROBLEMS}
    return {
        "per_problem": per,
        "macro": {
            "macro_top1": float(np.mean([r["top1"] for r in per.values()])),
            "macro_top2": float(np.mean([r["top2"] for r in per.values()])),
            "macro_top3": float(np.mean([r["top3"] for r in per.values()])),
            "macro_vs_sbs_pct": float(np.mean([r["vs_sbs_pct"] for r in per.values()])),
            "macro_vbs_gap_closed_pct": float(np.mean([r["vbs_gap_closed_pct"] for r in per.values()])),
            "macro_hidden_winner_mass": float(np.mean([r["hidden_winner_mass"] for r in per.values()])),
        },
    }


@torch.no_grad()
def sweep_gate(base_model, reranker, device: str, audit: dict, problem_stats: dict, topk: int,
               blend_alphas: list[float], base_gap_grid: list[float], rerank_margin_grid: list[float],
               base_topk: int | None = None, support_topk: int = 0,
               full_pool: bool = False, support_feature_mode: str = "max"):
    best = None
    for alpha in blend_alphas:
        for gap in base_gap_grid:
            for rerank_margin in rerank_margin_grid:
                out = evaluate_all(
                    base_model,
                    reranker,
                    "val",
                    device,
                    audit,
                    problem_stats,
                    topk=topk,
                    blend_alpha=alpha,
                    base_gap_max=gap,
                    rerank_margin_min=rerank_margin,
                    base_topk=base_topk,
                    support_topk=support_topk,
                    full_pool=full_pool,
                    support_feature_mode=support_feature_mode,
                )
                macro = out["macro"]
                key = (
                    -macro["macro_top1"],
                    macro["macro_vs_sbs_pct"],
                    macro["macro_safe_harm_rate"],
                    -macro["macro_fixable_recovery_rate"],
                )
                if best is None or key < best["key"]:
                    best = {
                        "key": key,
                        "config": {
                            "blend_alpha": alpha,
                            "base_gap_max": gap,
                            "rerank_margin_min": rerank_margin,
                        },
                        "val": out,
                    }
    assert best is not None
    return best


def top1_safe_score(macro: dict, baseline_vs_sbs: float, baseline_hidden_mass: float, cost_slack: float) -> float:
    hidden_recovery = max(0.0, baseline_hidden_mass - macro["macro_hidden_winner_mass"])
    cost_penalty = max(0.0, macro["macro_vs_sbs_pct"] - (baseline_vs_sbs + cost_slack))
    return (
        macro["macro_top1"]
        - 0.50 * cost_penalty
        + 0.05 * hidden_recovery
        - 0.05 * macro["macro_safe_harm_rate"]
    )


def evaluate_checkpoint(tag: str, save_dir: Path, base_model, reranker, device: str, audit: dict,
                        problem_stats: dict, args):
    best_gate = sweep_gate(
        base_model,
        reranker,
        device,
        audit,
        problem_stats,
        topk=args.rerank_topk,
        blend_alphas=parse_float_grid(args.blend_alpha_grid, [0.25, 0.5, 0.75, 1.0]),
        base_gap_grid=parse_float_grid(args.base_gap_grid, [0.03, 0.05, 0.08, 0.12, 0.20, float("inf")]),
        rerank_margin_grid=parse_float_grid(args.rerank_margin_grid, [0.0, 0.02, 0.05, 0.10]),
        base_topk=args.base_topk,
        support_topk=args.support_candidate_topk,
        full_pool=args.full_pool,
        support_feature_mode=args.support_feature_mode,
    )
    (save_dir / f"best_gate_{tag}.json").write_text(json.dumps(best_gate, indent=2))

    for split in ["val", "test"]:
        out = evaluate_all(
            base_model,
            reranker,
            split,
            device,
            audit,
            problem_stats,
            topk=args.rerank_topk,
            blend_alpha=best_gate["config"]["blend_alpha"],
            base_gap_max=best_gate["config"]["base_gap_max"],
            rerank_margin_min=best_gate["config"]["rerank_margin_min"],
            base_topk=args.base_topk,
            support_topk=args.support_candidate_topk,
            full_pool=args.full_pool,
            support_feature_mode=args.support_feature_mode,
        )
        print(
            format_eval_block(
                f"[{split} {tag}]",
                out["macro"],
                out["per_problem"],
                problem_order=PROBLEMS,
            ),
            flush=True,
        )
        (save_dir / f"analysis_{split}_{tag}.json").write_text(json.dumps(out, indent=2))
        if tag == "best_top1_safe":
            (save_dir / f"analysis_{split}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-per-problem", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rerank-topk", type=int, default=10)
    ap.add_argument("--rerank-d-model", type=int, default=128)
    ap.add_argument("--rerank-heads", type=int, default=4)
    ap.add_argument("--rerank-layers", type=int, default=2)
    ap.add_argument("--rerank-pair-weight", type=float, default=0.45)
    ap.add_argument("--gap-weight", type=float, default=0.25)
    ap.add_argument("--rerank-ce-weight", type=float, default=0.20)
    ap.add_argument("--anchor-weight", type=float, default=0.10)
    ap.add_argument("--anchor-mode", choices=["kl", "delta"], default="kl")
    ap.add_argument("--base-topk", type=int, default=2)
    ap.add_argument("--support-candidate-topk", type=int, default=10)
    ap.add_argument("--rare-winner-weight", type=float, default=1.0)
    ap.add_argument("--hidden-winner-weight", type=float, default=1.5)
    ap.add_argument("--switch-gain-weight", type=float, default=0.25)
    ap.add_argument("--coord-augment", type=int, default=0)
    ap.add_argument("--blend-alpha-grid", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--base-gap-grid", default="0.03,0.05,0.08,0.12,0.20,inf")
    ap.add_argument("--rerank-margin-grid", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--max-batches-per-problem", type=int, default=0)
    ap.add_argument("--cost-slack", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full-pool", action="store_true")
    ap.add_argument("--support-feature-mode", choices=["none", "max", "per_generator"], default="per_generator")
    ap.add_argument("--prior-cache", type=str, default="")
    ap.add_argument("--delta-init-scale", type=float, default=1e-3)
    ap.add_argument("--ce-fixable-only", action="store_true")
    ap.add_argument("--pair-min-gain-pct", type=float, default=0.0)
    ap.add_argument("--ce-min-gain-pct", type=float, default=0.0)
    ap.add_argument("--decisive-gain-pct", type=float, default=0.0)
    ap.add_argument("--decisive-upweight", type=float, default=0.0)
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    base_model, _ = build_base_model(args.base_ckpt, args.device)
    audit = json.loads(Path("code/unified_selector/runs/audit.json").read_text())
    cache_path = Path(args.prior_cache) if args.prior_cache else save_dir / "problem_solver_stats.json"
    problem_stats = load_or_build_problem_stats(base_model, audit, args.device, cache_path)
    base_val = evaluate_base_all(base_model, "val", args.device, audit)
    (save_dir / "base_val_metrics.json").write_text(json.dumps(base_val, indent=2))

    support_generators = 0
    if args.support_feature_mode == "per_generator" and getattr(base_model, "use_support_head", False):
        support_generators = int(getattr(base_model, "support_generators", 0))
    token_dim = 5 * base_model.d + 68 + support_generators
    reranker = CandidateSetReranker(
        in_dim=token_dim,
        d_model=args.rerank_d_model,
        heads=args.rerank_heads,
        layers=args.rerank_layers,
        max_topk=max(args.rerank_topk, 2),
        delta_init_scale=args.delta_init_scale,
    ).to(args.device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.wd)

    winner_weights = {}
    hidden_weights = {}
    for problem in PROBLEMS:
        cur = gather_problem_stats(problem, problem_stats, args.device)
        w = (cur["oracle_win_prior"] + 0.01).pow(-0.5)
        winner_weights[problem] = w / w.mean().clamp_min(1e-6)
        hidden_weights[problem] = 1.0 + args.hidden_winner_weight * cur["hidden_winner_flag"]

    best_top1_safe = -1e18
    best_cost = 1e18
    step = 0

    for epoch in range(args.epochs):
        for problem in PROBLEMS:
            ds = UnifiedProblemDataset(problem, "train", coord_augment=args.coord_augment)
            dl = DataLoader(ds, batch_size=args.batch_per_problem, shuffle=True, num_workers=0,
                            collate_fn=collate_single_problem, drop_last=True)
            hidden_flag = gather_problem_stats(problem, problem_stats, args.device)["hidden_winner_flag"]
            for i, batch in enumerate(dl):
                if args.max_batches_per_problem > 0 and i >= args.max_batches_per_problem:
                    break
                batch = to_device(batch, args.device)
                with torch.no_grad():
                    meta = build_candidate_set(
                        base_model,
                        batch,
                        problem_stats=problem_stats,
                        audit=audit,
                        topk=args.rerank_topk,
                        base_topk=args.base_topk,
                        support_topk=args.support_candidate_topk,
                        full_pool=args.full_pool,
                        support_feature_mode=args.support_feature_mode,
                    )
                delta, gap_pred = reranker(meta["token_feats"])
                base_wrong = meta["base_wrong"] if args.full_pool else meta["base_wrong"] & meta["oracle_in_topk"]
                pair_focus = base_wrong & (meta["switch_gain_pct"] >= args.pair_min_gain_pct)
                if args.ce_fixable_only:
                    ce_focus = base_wrong.clone()
                else:
                    ce_focus = torch.ones_like(base_wrong, dtype=torch.bool)
                if args.ce_min_gain_pct > 0:
                    ce_focus = ce_focus & (meta["switch_gain_pct"] >= args.ce_min_gain_pct)
                safe = ~base_wrong
                winner = batch["costs"].argmin(dim=1)
                ex_weight = winner_weights[problem][winner]
                ex_weight = 1.0 + args.rare_winner_weight * (ex_weight - 1.0)
                ex_weight = ex_weight * hidden_weights[problem][winner]
                if args.switch_gain_weight > 0:
                    gain_scale = meta["switch_gain_pct"] / meta["switch_gain_pct"].median().clamp_min(1e-6)
                    ex_weight = ex_weight * (1.0 + args.switch_gain_weight * gain_scale * base_wrong.float())
                if args.decisive_upweight > 0:
                    decisive = base_wrong & (meta["switch_gain_pct"] >= args.decisive_gain_pct)
                    ex_weight = ex_weight * (1.0 + args.decisive_upweight * decisive.float())
                loss, parts = rerank_loss(
                    delta,
                    gap_pred,
                    meta["cand_base"],
                    meta["cand_gap_target"],
                    meta["cand_costs"],
                    example_weight=ex_weight,
                    pair_mask=pair_focus,
                    ce_mask=ce_focus,
                    safe_mask=safe,
                    pair_weight=args.rerank_pair_weight,
                    gap_weight=args.gap_weight,
                    ce_weight=args.rerank_ce_weight,
                    anchor_weight=args.anchor_weight,
                    anchor_mode=args.anchor_mode,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    delta_scale = float(F.softplus(reranker.delta_scale_raw).item())
                    print(
                        f"ep{epoch} step{step} prob={problem} "
                        f"loss={loss.item():.4f} pair={parts['pair']:.4f} "
                        f"gap={parts['gap']:.4f} ce={parts['ce']:.4f} anchor={parts['anchor']:.4f} "
                        f"dscale={delta_scale:.6f}",
                        flush=True,
                    )

        val_out = evaluate_all(
            base_model,
            reranker,
            "val",
            args.device,
            audit,
            problem_stats,
            topk=args.rerank_topk,
            base_topk=args.base_topk,
            support_topk=args.support_candidate_topk,
            full_pool=args.full_pool,
            support_feature_mode=args.support_feature_mode,
        )
        macro = val_out["macro"]
        score = top1_safe_score(
            macro,
            baseline_vs_sbs=base_val["macro"]["macro_vs_sbs_pct"],
            baseline_hidden_mass=base_val["macro"]["macro_hidden_winner_mass"],
            cost_slack=args.cost_slack,
        )
        print(
            format_eval_block(
                f"[eval epoch {epoch}]",
                macro,
                val_out["per_problem"],
                problem_order=PROBLEMS,
                extra_macro_fields=[("score", f"{score:.6f}")],
            ),
            flush=True,
        )

        payload = {
            "reranker": reranker.state_dict(),
            "base_ckpt": args.base_ckpt,
            "args": vars(args),
            "val_macro": macro,
            "top1_safe_score": score,
        }
        if macro["macro_vs_sbs_pct"] < best_cost:
            best_cost = macro["macro_vs_sbs_pct"]
            torch.save(payload, save_dir / "best_cost.pt")
            print(f"  -> saved best_cost (macro_vs_sbs={best_cost:+.3f}%)", flush=True)
        if score > best_top1_safe:
            best_top1_safe = score
            torch.save(payload, save_dir / "best_top1_safe.pt")
            torch.save(payload, save_dir / "best.pt")
            print(f"  -> saved best_top1_safe (score={best_top1_safe:.6f})", flush=True)

    for tag in ["best_top1_safe", "best_cost"]:
        ckpt_path = save_dir / f"{tag}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=args.device)
        reranker.load_state_dict(ckpt["reranker"])
        evaluate_checkpoint(tag, save_dir, base_model, reranker, args.device, audit, problem_stats, args)


if __name__ == "__main__":
    main()
