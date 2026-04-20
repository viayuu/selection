from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, remap_legacy_state_dict


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


class CandidateSetReranker(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, heads: int = 4, layers: int = 1, dropout: float = 0.1, max_topk: int = 8):
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
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_feats: torch.Tensor) -> torch.Tensor:
        # token_feats: (B, K, F)
        B, K, _ = token_feats.shape
        x = self.in_proj(token_feats)
        pos = torch.arange(K, device=token_feats.device).unsqueeze(0).expand(B, K)
        x = x + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        return self.out(x).squeeze(-1)  # (B, K)


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
        encoder_rezero=bool(args.get("encoder_rezero", False)),
        encoder_constraint_experts=bool(args.get("encoder_constraint_experts", False)),
        encoder_constraint_hidden=int(args.get("encoder_constraint_hidden", 128)),
    ).to(device)
    missing, unexpected = model.load_state_dict(remap_legacy_state_dict(ckpt["model"]), strict=False)
    if any(k.startswith("support_problem_heads.") for k in missing):
        for name, param in model.named_parameters():
            if name.startswith("support_problem_heads."):
                nn.init.zeros_(param)
    if any(k.startswith("support_constraint_expert_heads.") for k in missing):
        for name, param in model.named_parameters():
            if name.startswith("support_constraint_expert_heads."):
                nn.init.zeros_(param)
    if missing or unexpected:
        print(f"[load base] missing={missing} unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt


@torch.no_grad()
def build_candidate_set(base_model: UnifiedSelector, batch, topk: int,
                        base_topk: int | None = None,
                        support_topk: int = 0):
    z, _ = base_model.compute_pair_features(batch)
    if getattr(base_model, "use_support_head", False):
        logits, support_logits = base_model(batch, return_support=True)
    else:
        logits = base_model(batch)
        support_logits = None
    logits_pool = logits[:, batch["pool_ids"]]
    z_pool = z[:, batch["pool_ids"]]
    probs_pool = logits_pool.softmax(dim=1)
    k_pool = logits_pool.shape[1]
    topk = min(max(1, topk), k_pool)
    base_topk = min(topk, max(1, base_topk or topk))
    base_idx = logits_pool.topk(base_topk, dim=1).indices
    if support_topk > 0 and support_logits is not None:
        support_pool = support_logits[:, batch["pool_ids"]] if support_logits.dim() == 2 else support_logits[:, :, batch["pool_ids"]]
        support_prob = torch.sigmoid(support_pool)
        if support_prob.dim() == 3:
            support_prob = support_prob.max(dim=1).values
        support_idx = support_prob.topk(min(topk, support_topk), dim=1).indices
    else:
        support_idx = None
        support_prob = None
    top_rows = []
    cand_is_support = []
    cand_is_rescued = []
    for i in range(logits_pool.shape[0]):
        row = []
        base_set = set(base_idx[i].tolist())
        support_set = set(support_idx[i].tolist()) if support_idx is not None else set()
        for src in (base_idx[i], support_idx[i] if support_idx is not None else []):
            for val in src.tolist():
                if val not in row:
                    row.append(int(val))
                if len(row) >= topk:
                    break
            if len(row) >= topk:
                break
        if len(row) < topk:
            extra = logits_pool[i].argsort(descending=True).tolist()
            for val in extra:
                if val not in row:
                    row.append(int(val))
                if len(row) >= topk:
                    break
        row_t = torch.tensor(row, device=logits_pool.device, dtype=torch.long)
        top_rows.append(row_t)
        cand_is_support.append(torch.tensor([float(v in support_set) for v in row], device=logits_pool.device))
        cand_is_rescued.append(torch.tensor([float((v in support_set) and (v not in base_set)) for v in row], device=logits_pool.device))
    top_idx = torch.stack(top_rows, dim=0)
    support_src = torch.stack(cand_is_support, dim=0)
    rescued_src = torch.stack(cand_is_rescued, dim=0)
    gather_feat = top_idx.unsqueeze(-1).expand(-1, -1, z_pool.shape[-1])
    cand_z = torch.gather(z_pool, 1, gather_feat)
    cand_base = torch.gather(logits_pool, 1, top_idx)
    cand_prob = torch.gather(probs_pool, 1, top_idx)
    cand_support_prob = torch.gather(support_prob, 1, top_idx) if support_prob is not None else torch.zeros_like(cand_prob)
    best_base = cand_base.max(dim=1, keepdim=True).values
    delta = cand_base - best_base
    best_prob = cand_prob.max(dim=1, keepdim=True).values
    prob_delta = cand_prob - best_prob
    rank_frac = torch.arange(topk, device=logits_pool.device, dtype=logits_pool.dtype)
    rank_frac = rank_frac / max(1, topk - 1)
    rank_frac = rank_frac.view(1, topk, 1).expand(logits_pool.shape[0], -1, -1)
    token_feats = torch.cat([
        cand_z,
        cand_base.unsqueeze(-1),
        delta.unsqueeze(-1),
        cand_prob.unsqueeze(-1),
        cand_support_prob.unsqueeze(-1),
        support_src.unsqueeze(-1),
        rescued_src.unsqueeze(-1),
        prob_delta.unsqueeze(-1),
        rank_frac,
    ], dim=-1)
    cand_costs = torch.gather(batch["costs"], 1, top_idx)
    best_idx = batch["costs"].argmin(dim=1)
    oracle_in_topk = (top_idx == best_idx.unsqueeze(1)).any(dim=1)
    support_contains_oracle = (support_idx == best_idx.unsqueeze(1)).any(dim=1) if support_idx is not None else torch.zeros_like(oracle_in_topk)
    base_contains_oracle = (base_idx == best_idx.unsqueeze(1)).any(dim=1)
    base_pred = logits_pool.argmax(dim=1)
    base_wrong = base_pred != best_idx
    base_selected_cost = batch["costs"].gather(1, base_pred.unsqueeze(1)).squeeze(1)
    oracle_cost = batch["costs"].gather(1, best_idx.unsqueeze(1)).squeeze(1)
    switch_gain = (base_selected_cost - oracle_cost).clamp_min(0.0)
    probs_sorted = probs_pool.topk(min(2, probs_pool.shape[1]), dim=1).values
    if probs_sorted.shape[1] == 1:
        base_prob_gap = torch.ones_like(probs_sorted[:, 0])
    else:
        base_prob_gap = probs_sorted[:, 0] - probs_sorted[:, 1]
    entropy = -(probs_pool * probs_pool.clamp_min(1e-12).log()).sum(dim=1)
    base_entropy = entropy / math.log(max(2, probs_pool.shape[1]))
    return {
        "logits_pool": logits_pool,
        "top_idx": top_idx,
        "cand_base": cand_base,
        "cand_prob": cand_prob,
        "cand_support_prob": cand_support_prob,
        "token_feats": token_feats,
        "cand_costs": cand_costs,
        "oracle_in_topk": oracle_in_topk,
        "base_contains_oracle": base_contains_oracle,
        "support_contains_oracle": support_contains_oracle,
        "base_wrong": base_wrong,
        "base_selected_cost": base_selected_cost,
        "oracle_cost": oracle_cost,
        "switch_gain": switch_gain,
        "base_prob_gap": base_prob_gap,
        "base_entropy": base_entropy,
        "support_rescued": (support_contains_oracle & ~base_contains_oracle) if support_idx is not None else torch.zeros_like(oracle_in_topk),
    }


def rerank_loss(delta_scores: torch.Tensor, cand_base: torch.Tensor, cand_costs: torch.Tensor,
                fixable_mask: torch.Tensor | None = None, pair_weight: float = 1.0, ce_weight: float = 0.25,
                example_weight: torch.Tensor | None = None,
                anchor_mask: torch.Tensor | None = None,
                anchor_weight: float = 0.0,
                anchor_mode: str = "kl",
                anchor_example_weight: torch.Tensor | None = None):
    if fixable_mask is None:
        fixable_mask = torch.ones(delta_scores.shape[0], dtype=torch.bool, device=delta_scores.device)
    total = delta_scores.new_tensor(0.0)
    if fixable_mask.any():
        final_scores = cand_base[fixable_mask] + delta_scores[fixable_mask]
        costs = cand_costs[fixable_mask]
        winner = costs.argmin(dim=1)
        ce = F.cross_entropy(final_scores, winner, reduction="none")
        if example_weight is not None:
            ex_w = example_weight[fixable_mask].to(ce.dtype)
            loss_ce = (ce * ex_w).sum() / ex_w.sum().clamp_min(1e-6)
        else:
            loss_ce = ce.mean()

        gap_ji = costs.unsqueeze(1) - costs.unsqueeze(2)
        sign = torch.sign(gap_ji)
        weight = gap_ji.abs()
        score_ij = final_scores.unsqueeze(2) - final_scores.unsqueeze(1)
        K = final_scores.shape[1]
        tri = torch.triu(torch.ones(K, K, dtype=torch.bool, device=final_scores.device), diagonal=1)
        pair_mask = (sign != 0) & tri.unsqueeze(0)
        if pair_mask.any():
            pair = F.softplus(-sign * score_ij)
            pair_val = (pair * weight).masked_select(pair_mask).sum() / weight.masked_select(pair_mask).sum().clamp_min(1e-6)
            if example_weight is not None:
                pair_per = []
                tri_mask = tri.unsqueeze(0).expand(final_scores.shape[0], -1, -1)
                for b in range(final_scores.shape[0]):
                    mask_b = (sign[b] != 0) & tri_mask[b]
                    if mask_b.any():
                        pair_b = (pair[b] * weight[b]).masked_select(mask_b).sum() / weight[b].masked_select(mask_b).sum().clamp_min(1e-6)
                        pair_per.append(pair_b)
                    else:
                        pair_per.append(final_scores.new_tensor(0.0))
                pair_per = torch.stack(pair_per)
                ex_w = example_weight[fixable_mask].to(pair_per.dtype)
                pair = (pair_per * ex_w).sum() / ex_w.sum().clamp_min(1e-6)
            else:
                pair = pair_val
        else:
            pair = final_scores.new_tensor(0.0)
        total = total + pair_weight * pair + ce_weight * loss_ce

    if anchor_weight > 0.0:
        if anchor_mask is None:
            anchor_mask = ~fixable_mask
        if anchor_mask.any():
            if anchor_mode == "delta":
                anchor_raw = F.smooth_l1_loss(
                    delta_scores[anchor_mask],
                    torch.zeros_like(delta_scores[anchor_mask]),
                    reduction="none",
                ).mean(dim=1)
            elif anchor_mode == "kl":
                base_prob = cand_base[anchor_mask].softmax(dim=1)
                final_logprob = (cand_base[anchor_mask] + delta_scores[anchor_mask]).log_softmax(dim=1)
                anchor_raw = F.kl_div(final_logprob, base_prob, reduction="none").sum(dim=1)
            else:
                raise ValueError(f"Unsupported anchor_mode={anchor_mode}")
            if anchor_example_weight is not None:
                anchor_w = anchor_example_weight[anchor_mask].to(anchor_raw.dtype)
                anchor = (anchor_raw * anchor_w).sum() / anchor_w.sum().clamp_min(1e-6)
            else:
                anchor = anchor_raw.mean()
            total = total + anchor_weight * anchor
    return total


def parse_float_grid(spec: str | None, default: Iterable[float]) -> list[float]:
    if spec is None:
        return list(default)
    vals = []
    for token in spec.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok in {"inf", "+inf"}:
            vals.append(float("inf"))
        else:
            vals.append(float(tok))
    return vals or list(default)


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
def evaluate(base_model, reranker, problem: str, split: str, device: str, topk: int,
             blend_alpha: float = 1.0, base_gap_max: float = float("inf"),
             rerank_margin_min: float = 0.0,
             base_topk: int | None = None,
             support_topk: int = 0):
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
    top1 = top2 = top3 = 0
    total = 0
    cost_sum = 0.0
    top1_fixable = 0
    total_fixable = 0
    applied = 0
    fixable_applied = 0
    base_gap_sum = 0.0
    rerank_margin_sum = 0.0
    for batch in dl:
        batch = to_device(batch, device)
        meta = build_candidate_set(base_model, batch, topk=topk, base_topk=base_topk, support_topk=support_topk)
        delta = reranker(meta["token_feats"])
        final_pool, gate, rerank_margin = apply_rerank(
            meta,
            delta,
            blend_alpha=blend_alpha,
            base_gap_max=base_gap_max,
            rerank_margin_min=rerank_margin_min,
        )

        pred = final_pool.argmax(dim=1)
        true_rank = batch["costs"].argsort(dim=1)
        best = true_rank[:, 0]
        pred_rank = final_pool.argsort(dim=1, descending=True)
        top1 += (pred == best).sum().item()
        top2 += (pred_rank[:, :min(2, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        top3 += (pred_rank[:, :min(3, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        sel_cost = batch["costs"].gather(1, pred.unsqueeze(1)).squeeze(1)
        cost_sum += sel_cost.sum().item()
        total += batch["costs"].shape[0]
        applied += int(gate.sum().item())
        fixable_applied += int((gate & meta["oracle_in_topk"] & meta["base_wrong"]).sum().item())
        base_gap_sum += meta["base_prob_gap"].sum().item()
        rerank_margin_sum += rerank_margin.sum().item()

        fixable = meta["oracle_in_topk"] & meta["base_wrong"]
        if fixable.any():
            top1_fixable += (pred[fixable] == best[fixable]).sum().item()
            total_fixable += int(fixable.sum().item())

    return {
        "top1": top1 / total,
        "top2": top2 / total,
        "top3": top3 / total,
        "mean_cost": cost_sum / total,
        "top1_given_fixable": top1_fixable / total_fixable if total_fixable > 0 else None,
        "n_fixable": total_fixable,
        "rerank_apply_rate": applied / total,
        "rerank_apply_rate_on_fixable": fixable_applied / total_fixable if total_fixable > 0 else None,
        "base_prob_gap_mean": base_gap_sum / total,
        "rerank_margin_mean": rerank_margin_sum / total,
    }


def summarize_macro(per_problem: dict[str, dict]) -> dict[str, float]:
    vals = list(per_problem.values())
    return {
        "macro_top1": float(np.mean([r["top1"] for r in vals])),
        "macro_top2": float(np.mean([r["top2"] for r in vals])),
        "macro_top3": float(np.mean([r["top3"] for r in vals])),
        "macro_vs_sbs_pct": float(np.mean([r["vs_sbs_pct"] for r in vals])),
        "macro_vbs_gap_closed_pct": float(np.mean([r["vbs_gap_closed_pct"] for r in vals])),
        "macro_top1_given_fixable": float(np.mean([r["top1_given_fixable"] for r in vals if r["top1_given_fixable"] is not None] or [0.0])),
        "macro_rerank_apply_rate": float(np.mean([r["rerank_apply_rate"] for r in vals])),
        "macro_rerank_apply_rate_on_fixable": float(np.mean([r["rerank_apply_rate_on_fixable"] for r in vals if r["rerank_apply_rate_on_fixable"] is not None] or [0.0])),
        "macro_base_prob_gap_mean": float(np.mean([r["base_prob_gap_mean"] for r in vals])),
        "macro_rerank_margin_mean": float(np.mean([r["rerank_margin_mean"] for r in vals])),
    }


@torch.no_grad()
def evaluate_all(base_model, reranker, split: str, device: str, topk: int, audit: dict,
                 blend_alpha: float = 1.0, base_gap_max: float = float("inf"),
                 rerank_margin_min: float = 0.0,
                 base_topk: int | None = None,
                 support_topk: int = 0):
    per = {}
    for problem in PROBLEMS:
        res = evaluate(
            base_model,
            reranker,
            problem,
            split,
            device,
            topk=topk,
            blend_alpha=blend_alpha,
            base_gap_max=base_gap_max,
            rerank_margin_min=rerank_margin_min,
            base_topk=base_topk,
            support_topk=support_topk,
        )
        sbs = audit[problem][split]["sbs_mean"]
        vbs = audit[problem][split]["vbs_mean"]
        mean_cost = res["mean_cost"]
        per[problem] = {
            **res,
            "sbs_cost": sbs,
            "vbs_mean": vbs,
            "vs_sbs_pct": (mean_cost - sbs) / (abs(sbs) + 1e-9) * 100,
            "vbs_gap_closed_pct": ((sbs - mean_cost) / (sbs - vbs + 1e-9)) * 100 if sbs != vbs else 0.0,
        }
    return {"per_problem": per, "macro": summarize_macro(per)}


@torch.no_grad()
def sweep_gate(base_model, reranker, device: str, topk: int, audit: dict,
               blend_alphas: list[float], base_gap_grid: list[float], rerank_margin_grid: list[float],
               base_topk: int | None = None, support_topk: int = 0):
    best = None
    for alpha in blend_alphas:
        for gap in base_gap_grid:
            for rerank_margin in rerank_margin_grid:
                out = evaluate_all(
                    base_model,
                    reranker,
                    "val",
                    device,
                    topk=topk,
                    audit=audit,
                    blend_alpha=alpha,
                    base_gap_max=gap,
                    rerank_margin_min=rerank_margin,
                    base_topk=base_topk,
                    support_topk=support_topk,
                )
                macro = out["macro"]
                key = (
                    macro["macro_vs_sbs_pct"],
                    -macro["macro_top1"],
                    -macro["macro_vbs_gap_closed_pct"],
                    macro["macro_rerank_apply_rate"],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-per-problem", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rerank-topk", type=int, default=3)
    ap.add_argument("--rerank-d-model", type=int, default=128)
    ap.add_argument("--rerank-heads", type=int, default=4)
    ap.add_argument("--rerank-layers", type=int, default=1)
    ap.add_argument("--rerank-pair-weight", type=float, default=1.0)
    ap.add_argument("--rerank-ce-weight", type=float, default=0.25)
    ap.add_argument("--fixable-only", action="store_true")
    ap.add_argument("--base-topk", type=int, default=2,
                    help="How many candidates to keep from the base selector before support augmentation.")
    ap.add_argument("--support-candidate-topk", type=int, default=0,
                    help="Extra candidates proposed by the base shortlist/support head.")
    ap.add_argument("--rare-winner-weight", type=float, default=0.0,
                    help="Extra multiplier for inverse-frequency oracle-winner weighting.")
    ap.add_argument("--fixable-upweight", type=float, default=0.0,
                    help="Extra multiplier for fixable / support-rescued examples.")
    ap.add_argument("--switch-gain-weight", type=float, default=0.0,
                    help="Extra multiplier on fixable examples proportional to base-to-oracle switch gain.")
    ap.add_argument("--anchor-weight", type=float, default=0.0,
                    help="Weight for do-no-harm anchor on non-fixable examples.")
    ap.add_argument("--anchor-mode", choices=["kl", "delta"], default="kl",
                    help="Anchor mode: KL between base/final candidate distributions, or SmoothL1 on delta scores.")
    ap.add_argument("--coord-augment", type=int, default=0)
    ap.add_argument("--blend-alpha-grid", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--base-gap-grid", default="0.03,0.05,0.08,0.12,0.20,inf")
    ap.add_argument("--rerank-margin-grid", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--max-batches-per-problem", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    base_model, base_ckpt = build_base_model(args.base_ckpt, args.device)
    token_dim = 4 * base_model.d + 8
    reranker = CandidateSetReranker(
        in_dim=token_dim,
        d_model=args.rerank_d_model,
        heads=args.rerank_heads,
        layers=args.rerank_layers,
        max_topk=max(args.rerank_topk, 2),
    ).to(args.device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.wd)

    step = 0
    best_macro_vs_sbs = 1e9
    audit = json.loads(Path("code/unified_selector/runs/audit.json").read_text())
    winner_weights = {}
    for problem in PROBLEMS:
        ds = UnifiedProblemDataset(problem, "train")
        counts = torch.zeros(ds.K_p, dtype=torch.float32)
        for i in range(ds.base_N):
            costs = torch.tensor(ds.labels[str(i)]["cost"][:ds.K_p], dtype=torch.float32)
            counts[int(costs.argmin().item())] += 1.0
        weights = (counts + 0.01).pow(-0.5)
        winner_weights[problem] = (weights / weights.mean().clamp_min(1e-6))

    for epoch in range(args.epochs):
        for problem in PROBLEMS:
            ds = UnifiedProblemDataset(problem, "train", coord_augment=args.coord_augment)
            dl = DataLoader(ds, batch_size=args.batch_per_problem, shuffle=True, num_workers=0, collate_fn=collate_single_problem, drop_last=True)
            for i, batch in enumerate(dl):
                if args.max_batches_per_problem > 0 and i >= args.max_batches_per_problem:
                    break
                batch = to_device(batch, args.device)
                with torch.no_grad():
                    meta = build_candidate_set(
                        base_model, batch, topk=args.rerank_topk,
                        base_topk=args.base_topk,
                        support_topk=args.support_candidate_topk,
                    )
                delta = reranker(meta["token_feats"])
                fixable = meta["oracle_in_topk"]
                if args.fixable_only:
                    fixable = fixable & meta["base_wrong"]
                safe_mask = ~fixable
                if not fixable.any() and args.anchor_weight <= 0.0:
                    continue
                winner = batch["costs"].argmin(dim=1)
                ex_weight = winner_weights[problem].to(batch["costs"].device)[winner]
                if args.rare_winner_weight > 0:
                    ex_weight = 1.0 + args.rare_winner_weight * (ex_weight - 1.0)
                if args.fixable_upweight > 0:
                    boost = fixable | meta["support_rescued"]
                    ex_weight = ex_weight * (1.0 + args.fixable_upweight * boost.float())
                if args.switch_gain_weight > 0:
                    gain = meta["switch_gain"] / meta["switch_gain"].mean().clamp_min(1e-6)
                    ex_weight = ex_weight * (1.0 + args.switch_gain_weight * gain * fixable.float())
                loss = rerank_loss(
                    delta,
                    meta["cand_base"],
                    meta["cand_costs"],
                    fixable_mask=fixable,
                    pair_weight=args.rerank_pair_weight,
                    ce_weight=args.rerank_ce_weight,
                    example_weight=ex_weight,
                    anchor_mask=safe_mask,
                    anchor_weight=args.anchor_weight,
                    anchor_mode=args.anchor_mode,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    print(f"ep{epoch} step{step} prob={problem} loss={loss.item():.4f}", flush=True)

        val_out = evaluate_all(
            base_model, reranker, "val", args.device, topk=args.rerank_topk, audit=audit,
            base_topk=args.base_topk, support_topk=args.support_candidate_topk,
        )
        macro = val_out["macro"]
        print(f"[eval epoch {epoch}] macro_top1={macro['macro_top1']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% top1_fixable={macro['macro_top1_given_fixable']:.4f}")
        if macro["macro_vs_sbs_pct"] < best_macro_vs_sbs:
            best_macro_vs_sbs = macro["macro_vs_sbs_pct"]
            torch.save({
                "reranker": reranker.state_dict(),
                "base_ckpt": args.base_ckpt,
                "args": vars(args),
                "val_macro": macro,
            }, save_dir / "best.pt")
            print(f"  -> saved best reranker (macro_vs_sbs={best_macro_vs_sbs:+.3f}%)")

    ckpt = torch.load(save_dir / "best.pt", map_location=args.device)
    reranker.load_state_dict(ckpt["reranker"])
    best_gate = sweep_gate(
        base_model,
        reranker,
        args.device,
        topk=args.rerank_topk,
        audit=audit,
        blend_alphas=parse_float_grid(args.blend_alpha_grid, [0.25, 0.5, 0.75, 1.0]),
        base_gap_grid=parse_float_grid(args.base_gap_grid, [0.03, 0.05, 0.08, 0.12, 0.20, float("inf")]),
        rerank_margin_grid=parse_float_grid(args.rerank_margin_grid, [0.0, 0.02, 0.05, 0.10]),
        base_topk=args.base_topk,
        support_topk=args.support_candidate_topk,
    )
    print(
        "[best gate] "
        f"alpha={best_gate['config']['blend_alpha']:.3f} "
        f"base_gap_max={best_gate['config']['base_gap_max']:.3f} "
        f"rerank_margin_min={best_gate['config']['rerank_margin_min']:.3f} "
        f"val_top1={best_gate['val']['macro']['macro_top1']:.4f} "
        f"val_vs_sbs={best_gate['val']['macro']['macro_vs_sbs_pct']:+.3f}%"
    )
    (save_dir / "best_gate.json").write_text(json.dumps(best_gate, indent=2))

    out = {"val": best_gate["val"], "test": {}}
    for split in ["test"]:
        split_out = evaluate_all(
            base_model,
            reranker,
            split,
            args.device,
            topk=args.rerank_topk,
            audit=audit,
            blend_alpha=best_gate["config"]["blend_alpha"],
            base_gap_max=best_gate["config"]["base_gap_max"],
            rerank_margin_min=best_gate["config"]["rerank_margin_min"],
            base_topk=args.base_topk,
            support_topk=args.support_candidate_topk,
        )
        out[split] = split_out
        macro = split_out["macro"]
        print(
            f"[{split}] macro_top1={macro['macro_top1']:.4f} "
            f"vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% "
            f"top1_fixable={macro['macro_top1_given_fixable']:.4f} "
            f"apply={macro['macro_rerank_apply_rate']:.4f}"
        )
        (save_dir / f"analysis_{split}.json").write_text(json.dumps(split_out, indent=2))
    (save_dir / "analysis_val.json").write_text(json.dumps(out["val"], indent=2))


if __name__ == "__main__":
    main()
