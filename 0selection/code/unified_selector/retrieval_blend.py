"""Retrieval-as-tiebreaker: blend base logits with retrieval prior, but restricted
to base's top-K candidates. Motivated by topk_coverage diagnostic:
92% of zero-pick+oracle-wins mass is within base top-3 — so restricting the
retrieval tiebreaker to that set should be close to an upper-bound rescue.

Variants:
  - full: blend over all K_p arms (same as retrieval_prior.py)
  - top3: restrict blend argmax to base top-3 candidates
  - top5: restrict blend argmax to base top-5 candidates

Also compares to per-instance GATED blend: if base margin (top1-top2) > δ,
use pure base; otherwise use blend. Tune δ on val, apply on test.

Usage:
  python -m code.unified_selector.retrieval_blend \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R26_blend \
    --modes full top3 top5 gated \
    --alpha-grid 0.25 0.5 0.75 \
    --gate-delta-grid 0.0 0.05 0.1 0.2 0.5 \
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
    load_base, encode_instances, compute_retrieval_prior, bootstrap_vs_base,
)


def blend_and_pick(base_logits, log_prior, alpha, top_idx=None):
    """Return argmax of alpha·base_norm + (1-alpha)·prior_norm.
    If top_idx is provided (N, K), restrict argmax to those K candidates.
    """
    base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
    prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
    blend = alpha * base_norm + (1 - alpha) * prior_norm
    if top_idx is None:
        return blend.argmax(axis=1)
    # Restrict to top_idx columns
    restricted = np.take_along_axis(blend, top_idx, axis=1)  # (N, K)
    best_k = restricted.argmax(axis=1)                        # (N,)
    return np.take_along_axis(top_idx, best_k[:, None], axis=1).squeeze(1)


def margin(base_logits):
    """Return top1-top2 margin per instance (N,) on normalized logits."""
    sorted_logits = np.sort(base_logits, axis=1)[:, ::-1]
    return sorted_logits[:, 0] - sorted_logits[:, 1]


def eval_mode(test_p, priors, alpha, mode, gate_delta=None):
    """Compute picks under specified mode; returns per-problem dict."""
    per_p = {}
    for p in PROBLEMS:
        q = test_p[p]
        K_p = q["costs"].shape[1]
        base_logits = q["logits"]
        log_prior = np.log(priors[p] + 1e-6)
        if mode == "full":
            picks = blend_and_pick(base_logits, log_prior, alpha)
        elif mode in ("top3", "top5"):
            K = int(mode[3:])
            K = min(K, K_p)
            top_idx = np.argsort(-base_logits, axis=1)[:, :K]
            picks = blend_and_pick(base_logits, log_prior, alpha, top_idx=top_idx)
        elif mode == "gated":
            assert gate_delta is not None
            base_margin = margin(base_logits)
            blend_picks = blend_and_pick(base_logits, log_prior, alpha)
            base_picks = base_logits.argmax(axis=1)
            picks = np.where(base_margin >= gate_delta, base_picks, blend_picks)
        else:
            raise ValueError(mode)
        costs = q["costs"]; oracle = q["oracle"]
        sbs_idx = int(costs.mean(axis=0).argmin())
        sbs_mean = costs[:, sbs_idx].mean()
        top1 = float((picks == oracle).mean())
        mean_c = float(costs[np.arange(costs.shape[0]), picks].mean())
        vs_sbs = 100 * (mean_c - sbs_mean) / (abs(sbs_mean) + 1e-9)
        per_p[p] = dict(picks=picks.astype(np.int32), costs=costs,
                        oracle=oracle.astype(np.int32),
                        top1=top1, mean_cost=mean_c, vs_sbs_pct=vs_sbs)
    macro_top1 = float(np.mean([v["top1"] for v in per_p.values()]))
    macro_vs_sbs = float(np.mean([v["vs_sbs_pct"] for v in per_p.values()]))
    return per_p, macro_top1, macro_vs_sbs


def run_config(test_p, priors, mode, alpha_grid, gate_grid=None):
    """Return (list_of_configs, per_config_dict)."""
    configs = []
    if mode == "gated":
        for a in alpha_grid:
            for g in gate_grid:
                configs.append((a, g))
    else:
        for a in alpha_grid:
            configs.append((a, None))
    per_cfg = {}
    for (a, g) in configs:
        per_p, top1, vs_sbs = eval_mode(test_p, priors, a, mode, gate_delta=g)
        key = f"a={a:.2f}" if g is None else f"a={a:.2f}_d={g:.2f}"
        per_cfg[key] = dict(mode=mode, alpha=a, gate=g, per_p=per_p,
                            macro_top1=top1, macro_vs_sbs_pct=vs_sbs)
    return per_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--modes", nargs="+", default=["full", "top3", "top5", "gated"])
    ap.add_argument("--alpha-grid", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--gate-delta-grid", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0])
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[blend] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[blend] encoding train+{args.split}")
    train_p = encode_instances(base_model, "train", device)
    test_p = encode_instances(base_model, args.split, device)

    print(f"[blend] computing k={args.k}-NN retrieval priors (τ={args.temperature})")
    priors = compute_retrieval_prior(train_p, test_p, k=args.k, temperature=args.temperature)

    # Baseline: pure base (α=1.0) picks
    base_picks = {}
    for p in PROBLEMS:
        q = test_p[p]
        base_picks[p] = dict(
            picks=q["logits"].argmax(axis=1).astype(np.int32),
            costs=q["costs"], oracle=q["oracle"].astype(np.int32),
        )

    all_results = {}
    for mode in args.modes:
        print(f"[blend] mode={mode}")
        gate_grid = args.gate_delta_grid if mode == "gated" else None
        cfgs = run_config(test_p, priors, mode, args.alpha_grid, gate_grid=gate_grid)
        all_results[mode] = cfgs

    # Bootstrap each config vs pure base
    print(f"[blend] bootstrapping vs base, n_boot={args.n_boot}")
    all_boot = {}
    for mode, cfgs in all_results.items():
        for key, r in cfgs.items():
            per_p = {p: dict(picks=r["per_p"][p]["picks"],
                             costs=r["per_p"][p]["costs"],
                             oracle=r["per_p"][p]["oracle"]) for p in PROBLEMS}
            # Use bootstrap_vs_base from retrieval_prior
            boot = bootstrap_vs_base({1.0: base_picks, 0.0: per_p}, ref_alpha=1.0, n_boot=args.n_boot)[0.0]
            all_boot[f"{mode}/{key}"] = boot

    lines = [f"# Retrieval-blend sweep ({args.split})\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}, n_boot={args.n_boot}\n",
             "| Config | macro top1 | Δtop1 [95% CI] | macro vs_sbs | Δcost% [95% CI] |",
             "|:---|---:|---:|---:|---:|"]
    # Baseline row
    base_top1 = float(np.mean([(base_picks[p]["picks"] == base_picks[p]["oracle"]).mean()
                               for p in PROBLEMS]))
    sbs_means = {p: float(test_p[p]["costs"][:, int(test_p[p]["costs"].mean(axis=0).argmin())].mean())
                 for p in PROBLEMS}
    base_vs_sbs = float(np.mean([100 * (base_picks[p]["costs"][np.arange(1000), base_picks[p]["picks"]].mean() - sbs_means[p]) / (abs(sbs_means[p]) + 1e-9)
                                 for p in PROBLEMS]))
    lines.append(f"| BASE (α=1.0 pure) | {base_top1:.4f} | — | {base_vs_sbs:+.4f}% | — |")
    for mode, cfgs in all_results.items():
        for key, r in cfgs.items():
            b = all_boot[f"{mode}/{key}"]
            sig_t = "*" if (b["top1_lo"] > 0 or b["top1_hi"] < 0) else ""
            sig_c = "*" if (b["cost_lo"] > 0 or b["cost_hi"] < 0) else ""
            lines.append(f"| {mode}/{key} | {r['macro_top1']:.4f} | {b['top1_mean']:+.4f} [{b['top1_lo']:+.4f}, {b['top1_hi']:+.4f}]{sig_t} | {r['macro_vs_sbs_pct']:+.4f}% | {b['cost_mean']:+.4f}% [{b['cost_lo']:+.4f}, {b['cost_hi']:+.4f}]{sig_c} |")

    report = "\n".join(lines)
    (out / f"blend_{args.split}.md").write_text(report)
    # Save summarized results
    save_results = {}
    for mode, cfgs in all_results.items():
        save_results[mode] = {}
        for key, r in cfgs.items():
            save_results[mode][key] = dict(
                alpha=r["alpha"], gate=r["gate"],
                macro_top1=r["macro_top1"], macro_vs_sbs_pct=r["macro_vs_sbs_pct"],
                bootstrap=all_boot[f"{mode}/{key}"],
            )
    (out / f"blend_{args.split}.json").write_text(json.dumps(save_results, indent=2, default=float))
    print(report)


if __name__ == "__main__":
    main()
