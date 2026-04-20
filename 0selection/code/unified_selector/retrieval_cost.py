"""Cost-based retrieval: instead of using neighbors' oracle-argmax as prior,
use their per-arm COST vector directly. For each test instance, estimate
expected per-arm cost as a similarity-weighted mean of neighbors' costs.
Pick argmin. Blend with base logits.

Motivation: oracle-argmax retrieval throws away cost magnitude information.
A neighbor whose oracle was arm-2 by a 0.01% margin gives the same vote as
a neighbor whose oracle was arm-2 by a 5% margin. The cost vector captures
the full distribution.

Usage:
  python -m code.unified_selector.retrieval_cost \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R27_cost_retrieval \
    --split test --k 32 --temperature 0.1 \
    --device cuda:0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .registry import PROBLEMS
from .data import UnifiedProblemDataset, collate_single_problem
from .retrieval_prior import (
    load_base, encode_instances, bootstrap_vs_base,
)


def compute_cost_prior(train_p, test_p, k=32, eps=1e-6, temperature=0.1,
                        normalize: str = "zscore"):
    """For each test instance, estimate per-arm expected cost as
    sim-weighted mean of training neighbors' per-arm cost vector.

    normalize options:
      - "none": raw cost
      - "zscore": each neighbor's cost vector z-scored within its own arm set
                  (makes scales comparable across problems with different |cost|)
      - "rank": convert to within-instance rank [0..K-1]

    Returns: per-problem (N_test, K_p) expected-cost vector.
    """
    priors = {}
    for p in PROBLEMS:
        t = train_p[p]; q = test_p[p]
        t_emb = t["pooled"] / (np.linalg.norm(t["pooled"], axis=1, keepdims=True) + eps)
        q_emb = q["pooled"] / (np.linalg.norm(q["pooled"], axis=1, keepdims=True) + eps)
        sim = q_emb @ t_emb.T
        topk_idx = np.argpartition(-sim, k, axis=1)[:, :k]
        t_costs = t["costs"]  # (N_train, K_p)

        # Normalize training costs
        if normalize == "zscore":
            mu = t_costs.mean(axis=1, keepdims=True)
            sd = t_costs.std(axis=1, keepdims=True) + eps
            t_costs_n = (t_costs - mu) / sd
        elif normalize == "rank":
            # Rank within each row [0, K-1]; lower cost = lower rank
            t_costs_n = np.argsort(np.argsort(t_costs, axis=1), axis=1).astype(np.float32)
        else:
            t_costs_n = t_costs

        K_p = q["costs"].shape[1]
        cost_prior = np.zeros((q_emb.shape[0], K_p), dtype=np.float32)
        for i in range(q_emb.shape[0]):
            sims = sim[i, topk_idx[i]]
            w = np.exp(sims / temperature)
            w /= w.sum() + eps
            # Weighted mean of neighbors' normalized cost vectors
            cost_prior[i] = (w[:, None] * t_costs_n[topk_idx[i]]).sum(axis=0)
        priors[p] = cost_prior
    return priors


def eval_blend(test_p, cost_priors, alpha_grid):
    """Blend: score = alpha · base_logit - (1-alpha) · cost_prior (since we want argmax of score).
    Cost is negated because low cost is better; we want to select argmax of -cost."""
    results = {a: {"per_p": {}} for a in alpha_grid}
    picks_per_alpha = {a: {} for a in alpha_grid}
    for p in PROBLEMS:
        q = test_p[p]
        costs = q["costs"]
        oracle = q["oracle"]
        sbs_idx = int(costs.mean(axis=0).argmin())
        sbs_mean = costs[:, sbs_idx].mean()
        base_logits = q["logits"]
        base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
        # Negate cost prior so argmax picks low cost
        cp = cost_priors[p]
        cp_norm = cp.mean(axis=1, keepdims=True) - cp  # low cost → high value
        # Standardize
        cp_norm = cp_norm / (cp_norm.std(axis=1, keepdims=True) + 1e-6)
        for alpha in alpha_grid:
            score = alpha * base_norm + (1 - alpha) * cp_norm
            pred = score.argmax(axis=1)
            top1 = float((pred == oracle).mean())
            mean_c = float(costs[np.arange(costs.shape[0]), pred].mean())
            vs_sbs = 100 * (mean_c - sbs_mean) / (abs(sbs_mean) + 1e-9)
            results[alpha]["per_p"][p] = dict(top1=top1, mean_cost=mean_c, vs_sbs_pct=vs_sbs,
                                              N=int(costs.shape[0]), K_p=int(costs.shape[1]))
            picks_per_alpha[alpha][p] = dict(picks=pred.astype(np.int32), costs=costs,
                                              oracle=oracle.astype(np.int32))
    for a in alpha_grid:
        pp = results[a]["per_p"]
        results[a]["macro_top1"] = float(np.mean([v["top1"] for v in pp.values()]))
        results[a]["macro_vs_sbs_pct"] = float(np.mean([v["vs_sbs_pct"] for v in pp.values()]))
    return results, picks_per_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--normalize", choices=["none", "zscore", "rank"], default="zscore")
    ap.add_argument("--alpha-grid", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[cost-retrieval] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)
    print(f"[cost-retrieval] encoding train + {args.split}")
    train_p = encode_instances(base_model, "train", device)
    test_p = encode_instances(base_model, args.split, device)
    print(f"[cost-retrieval] computing k={args.k}-NN cost priors (τ={args.temperature}, norm={args.normalize})")
    cost_priors = compute_cost_prior(train_p, test_p, k=args.k,
                                      temperature=args.temperature,
                                      normalize=args.normalize)
    res, picks_per_alpha = eval_blend(test_p, cost_priors, args.alpha_grid)

    # Bootstrap each α vs α=1.0 (pure base)
    boot = {}
    if 1.0 in args.alpha_grid:
        print(f"[cost-retrieval] bootstrapping n_boot={args.n_boot}")
        boot = bootstrap_vs_base(picks_per_alpha, ref_alpha=1.0, n_boot=args.n_boot)

    lines = [f"# Cost-retrieval α-sweep ({args.split})\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}, norm={args.normalize}\n",
             "| α | macro top1 | Δtop1 [95% CI] | macro vs_sbs | Δcost% [95% CI] |",
             "|---:|---:|---:|---:|---:|"]
    for a in args.alpha_grid:
        if a == 1.0 or a not in boot:
            boot_str = "—"
            cost_str = "—"
        else:
            b = boot[a]
            sig_t = "*" if (b["top1_lo"] > 0 or b["top1_hi"] < 0) else ""
            sig_c = "*" if (b["cost_lo"] > 0 or b["cost_hi"] < 0) else ""
            boot_str = f"{b['top1_mean']:+.4f} [{b['top1_lo']:+.4f}, {b['top1_hi']:+.4f}]{sig_t}"
            cost_str = f"{b['cost_mean']:+.4f}% [{b['cost_lo']:+.4f}, {b['cost_hi']:+.4f}]{sig_c}"
        lines.append(f"| {a:.2f} | {res[a]['macro_top1']:.4f} | {boot_str} | {res[a]['macro_vs_sbs_pct']:+.4f}% | {cost_str} |")
    report = "\n".join(lines)
    (out / f"cost_retrieval_{args.split}.md").write_text(report)
    (out / f"cost_retrieval_{args.split}.json").write_text(json.dumps(res, indent=2, default=float))
    (out / f"cost_retrieval_{args.split}_bootstrap.json").write_text(json.dumps(boot, indent=2, default=float))
    print(report)


if __name__ == "__main__":
    main()
