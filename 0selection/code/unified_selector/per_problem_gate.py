"""Per-problem gated blend — different α and δ for each problem family.

Motivation: retrieval helps most on TW families (+0.03-0.05) but hurts L families
(-0.01). A per-problem gate can maximize net gain by using retrieval only where
it helps.

Strategy:
  1. On val, find best (α, δ) per-problem.
  2. Evaluate on test with those fixed per-problem configs.

Usage:
  python -m code.unified_selector.per_problem_gate \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R30_per_prob \
    --k 32 --temperature 0.1 \
    --alpha-grid 0.3 0.5 0.7 1.0 \
    --gate-delta-grid 0.05 0.08 0.1 0.15 0.20 \
    --device cuda:0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from .registry import PROBLEMS
from .retrieval_prior import (
    load_base, encode_instances, compute_retrieval_prior, bootstrap_vs_base,
)


def margin_per_instance(base_logits):
    sorted_logits = np.sort(base_logits, axis=1)[:, ::-1]
    return sorted_logits[:, 0] - sorted_logits[:, 1]


def pick_for_problem(base_logits, prior, alpha, gate_delta):
    """Apply gated blend for a single problem.
    If alpha >= 1.0 or gate_delta >= infinity, use pure base.
    """
    if alpha >= 1.0:
        return base_logits.argmax(axis=1)
    base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
    log_prior = np.log(prior + 1e-6)
    prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
    blend = alpha * base_norm + (1 - alpha) * prior_norm
    blend_picks = blend.argmax(axis=1)
    base_picks = base_logits.argmax(axis=1)
    if gate_delta <= 0:
        return blend_picks
    margins = margin_per_instance(base_logits)
    return np.where(margins >= gate_delta, base_picks, blend_picks)


def eval_per_problem(data_enc, priors, alpha, gate_delta):
    """Return per-problem {picks, top1, mean_cost, vs_sbs_pct, N}."""
    per_p = {}
    for p in PROBLEMS:
        q = data_enc[p]
        base_logits = q["logits"]
        prior = priors[p]
        costs = q["costs"]
        oracle = q["oracle"]
        picks = pick_for_problem(base_logits, prior, alpha, gate_delta)
        top1 = float((picks == oracle).mean())
        mean_c = float(costs[np.arange(costs.shape[0]), picks].mean())
        sbs_idx = int(costs.mean(axis=0).argmin())
        sbs_mean = costs[:, sbs_idx].mean()
        vs_sbs = 100 * (mean_c - sbs_mean) / (abs(sbs_mean) + 1e-9)
        per_p[p] = dict(picks=picks.astype(np.int32), top1=top1, mean_cost=mean_c,
                        vs_sbs_pct=vs_sbs, costs=costs, oracle=oracle.astype(np.int32))
    return per_p


def find_best_per_problem(val_enc, val_priors, alpha_grid, gate_grid):
    """For each problem, pick (α, δ) that maximizes top1 on val."""
    best = {}
    for p in PROBLEMS:
        best[p] = dict(alpha=1.0, gate=0.0, top1=0.0)
        # Include α=1.0 (pure base) as a candidate
        for a in list(alpha_grid) + [1.0]:
            for g in list(gate_grid) + [0.0]:
                q = val_enc[p]
                picks = pick_for_problem(q["logits"], val_priors[p], a, g)
                top1 = float((picks == q["oracle"]).mean())
                if top1 > best[p]["top1"]:
                    best[p] = dict(alpha=a, gate=g, top1=top1)
    return best


def apply_per_problem(data_enc, priors, best_cfg):
    per_p = {}
    for p in PROBLEMS:
        q = data_enc[p]
        a = best_cfg[p]["alpha"]; g = best_cfg[p]["gate"]
        picks = pick_for_problem(q["logits"], priors[p], a, g)
        top1 = float((picks == q["oracle"]).mean())
        mean_c = float(q["costs"][np.arange(q["costs"].shape[0]), picks].mean())
        sbs_idx = int(q["costs"].mean(axis=0).argmin())
        sbs_mean = q["costs"][:, sbs_idx].mean()
        vs_sbs = 100 * (mean_c - sbs_mean) / (abs(sbs_mean) + 1e-9)
        per_p[p] = dict(picks=picks.astype(np.int32), top1=top1, mean_cost=mean_c,
                        vs_sbs_pct=vs_sbs, costs=q["costs"],
                        oracle=q["oracle"].astype(np.int32))
    return per_p


def get_base(data_enc):
    per_p = {}
    for p in PROBLEMS:
        q = data_enc[p]
        picks = q["logits"].argmax(axis=1).astype(np.int32)
        per_p[p] = dict(picks=picks, costs=q["costs"],
                        oracle=q["oracle"].astype(np.int32))
    return per_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--alpha-grid", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--gate-delta-grid", nargs="+", type=float,
                    default=[0.05, 0.08, 0.1, 0.12, 0.15, 0.20])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[pp-gate] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[pp-gate] encoding train+val+test")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    print(f"[pp-gate] computing priors (k={args.k})")
    val_priors = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    test_priors = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)

    print(f"[pp-gate] selecting best (α, δ) per problem on val")
    best_cfg = find_best_per_problem(val_enc, val_priors, args.alpha_grid, args.gate_delta_grid)
    for p, c in best_cfg.items():
        print(f"  {p}: α={c['alpha']:.2f}, δ={c['gate']:.2f}, val_top1={c['top1']:.4f}")

    print(f"[pp-gate] applying per-problem cfg to test")
    test_pp = apply_per_problem(test_enc, test_priors, best_cfg)
    val_pp = apply_per_problem(val_enc, val_priors, best_cfg)
    test_base = get_base(test_enc)
    val_base = get_base(val_enc)

    # Macro metrics
    macro_test_top1 = float(np.mean([v["top1"] for v in test_pp.values()]))
    macro_test_vs_sbs = float(np.mean([v["vs_sbs_pct"] for v in test_pp.values()]))
    macro_val_top1 = float(np.mean([v["top1"] for v in val_pp.values()]))

    base_test_top1 = float(np.mean([(test_base[p]["picks"] == test_base[p]["oracle"]).mean()
                                     for p in PROBLEMS]))
    base_val_top1 = float(np.mean([(val_base[p]["picks"] == val_base[p]["oracle"]).mean()
                                    for p in PROBLEMS]))

    # Bootstrap on test
    print(f"[pp-gate] bootstrap test, n_boot={args.n_boot}")
    boot = bootstrap_vs_base({1.0: test_base, 0.0: test_pp}, ref_alpha=1.0, n_boot=args.n_boot)[0.0]

    lines = [f"# Per-problem gated blend\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}\n",
             "## Per-problem val-selected config",
             "| Problem | α | δ | val top1 | test top1 | base test top1 | Δtest top1 |",
             "|:---|---:|---:|---:|---:|---:|---:|"]
    for p in PROBLEMS:
        c = best_cfg[p]
        t_top1 = test_pp[p]["top1"]
        b_top1 = float((test_base[p]["picks"] == test_base[p]["oracle"]).mean())
        lines.append(f"| {p} | {c['alpha']:.2f} | {c['gate']:.2f} | {c['top1']:.4f} | "
                     f"{t_top1:.4f} | {b_top1:.4f} | {t_top1 - b_top1:+.4f} |")
    lines.append("")
    lines.append("## Macro")
    lines.append(f"| split | pp-gated top1 | base top1 | delta |")
    lines.append(f"|:---|---:|---:|---:|")
    lines.append(f"| val | {macro_val_top1:.4f} | {base_val_top1:.4f} | {macro_val_top1 - base_val_top1:+.4f} |")
    lines.append(f"| test | {macro_test_top1:.4f} | {base_test_top1:.4f} | {macro_test_top1 - base_test_top1:+.4f} |")
    lines.append("")
    lines.append(f"## Bootstrap (test)")
    lines.append(f"- Δtop1: {boot['top1_mean']:+.4f} [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}]")
    lines.append(f"- Δcost%: {boot['cost_mean']:+.4f}% [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}]")
    lines.append(f"- Significant: {'YES' if (boot['top1_lo'] > 0 or boot['top1_hi'] < 0) else 'NO'}")

    (out / "per_problem_gate.md").write_text("\n".join(lines))
    (out / "per_problem_gate.json").write_text(json.dumps(dict(
        best_cfg=best_cfg, macro_val_top1=macro_val_top1,
        base_val_top1=base_val_top1, macro_test_top1=macro_test_top1,
        base_test_top1=base_test_top1, bootstrap=boot,
        args=vars(args),
    ), indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
