"""Subset analysis: on the 17-20% test mass where R18's TOP-1 is a zero-pick arm
AND the oracle IS a zero-pick arm, what is the top1 lift from retrieval-based methods?

This is a TARGETED claim: even if overall top1 doesn't move, if we can show that on
the rescuable subset, we actually rescue, it's a real method contribution.

Usage:
  python -m code.unified_selector.zero_pick_rescue \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R29_zp_rescue \
    --split test --k 32 --temperature 0.1 \
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


def margin(base_logits):
    """Top1-top2 margin per instance."""
    sorted_logits = np.sort(base_logits, axis=1)[:, ::-1]
    return sorted_logits[:, 0] - sorted_logits[:, 1]


def analyze(test_p, priors, alpha, gate_delta=None, mode="gated"):
    """Compute picks + zero-pick subset lift."""
    res = {}
    total_zp = 0
    total_zp_base_correct = 0
    total_zp_blend_correct = 0
    total_N = 0
    for p in PROBLEMS:
        q = test_p[p]
        K_p = q["costs"].shape[1]
        base_logits = q["logits"]
        log_prior = np.log(priors[p] + 1e-6)
        base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
        prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
        if mode == "gated":
            base_margin = margin(base_logits)
            base_picks = base_logits.argmax(axis=1)
            blend = alpha * base_norm + (1 - alpha) * prior_norm
            blend_picks = blend.argmax(axis=1)
            picks = np.where(base_margin >= gate_delta, base_picks, blend_picks)
        else:
            blend = alpha * base_norm + (1 - alpha) * prior_norm
            picks = blend.argmax(axis=1)
        base_picks = base_logits.argmax(axis=1)
        oracle = q["oracle"]
        costs = q["costs"]
        N = costs.shape[0]
        # Zero-pick arms for base on this problem
        unique_base = set(base_picks.tolist())
        zp_arms = [k for k in range(K_p) if k not in unique_base]
        # Zero-pick mass: oracle arm is a zero-pick arm
        zp_mask = np.isin(oracle, zp_arms)
        n_zp = zp_mask.sum()
        # On zero-pick subset, compare base vs blend
        if n_zp > 0:
            base_correct_zp = (base_picks[zp_mask] == oracle[zp_mask]).sum()
            blend_correct_zp = (picks[zp_mask] == oracle[zp_mask]).sum()
        else:
            base_correct_zp = 0
            blend_correct_zp = 0
        total_zp += n_zp
        total_zp_base_correct += int(base_correct_zp)
        total_zp_blend_correct += int(blend_correct_zp)
        total_N += N
        # Overall top1
        base_top1 = float((base_picks == oracle).mean())
        blend_top1 = float((picks == oracle).mean())
        res[p] = dict(
            K_p=K_p, N=N,
            zp_arms=zp_arms,
            n_zp=int(n_zp),
            zp_base_top1=float(base_correct_zp / max(n_zp, 1)),
            zp_blend_top1=float(blend_correct_zp / max(n_zp, 1)),
            base_top1=base_top1,
            blend_top1=blend_top1,
            delta_top1=blend_top1 - base_top1,
        )
    return res, dict(total_N=int(total_N), total_zp=int(total_zp),
                      total_zp_base_correct=int(total_zp_base_correct),
                      total_zp_blend_correct=int(total_zp_blend_correct),
                      global_zp_base_top1=total_zp_base_correct / max(total_zp, 1),
                      global_zp_blend_top1=total_zp_blend_correct / max(total_zp, 1),
                      global_zp_mass_pct=100 * total_zp / max(total_N, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gate-delta", type=float, default=0.1)
    ap.add_argument("--mode", choices=["full", "gated"], default="gated")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[zp-rescue] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)
    print(f"[zp-rescue] encoding train+{args.split}")
    train_p = encode_instances(base_model, "train", device)
    test_p = encode_instances(base_model, args.split, device)
    print(f"[zp-rescue] computing k={args.k} retrieval priors")
    priors = compute_retrieval_prior(train_p, test_p, k=args.k,
                                      temperature=args.temperature)
    per_p, globals_ = analyze(test_p, priors, args.alpha,
                               gate_delta=args.gate_delta, mode=args.mode)

    lines = [f"# Zero-pick rescue analysis ({args.split})\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}",
             f"Method: mode={args.mode}, α={args.alpha}, δ={args.gate_delta}\n",
             "## Per-problem",
             "| Problem | K_p | N | zp_arms | zp_mass | base top1 on zp | blend top1 on zp | overall Δtop1 |",
             "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for p, r in per_p.items():
        lines.append(f"| {p} | {r['K_p']} | {r['N']} | {len(r['zp_arms'])} | "
                     f"{r['n_zp']} ({100*r['n_zp']/r['N']:.1f}%) | "
                     f"{r['zp_base_top1']:.4f} | {r['zp_blend_top1']:.4f} | "
                     f"{r['delta_top1']:+.4f} |")
    lines.append("")
    lines.append("## Global")
    lines.append(f"- Total N: {globals_['total_N']}")
    lines.append(f"- Zero-pick mass: {globals_['total_zp']} ({globals_['global_zp_mass_pct']:.2f}%)")
    lines.append(f"- Base top1 on zp subset: **{globals_['global_zp_base_top1']:.4f}**")
    lines.append(f"- Blend top1 on zp subset: **{globals_['global_zp_blend_top1']:.4f}**")
    lines.append(f"- Δtop1 on zp subset: **{globals_['global_zp_blend_top1'] - globals_['global_zp_base_top1']:+.4f}**")
    rescue_rate = globals_['global_zp_blend_top1'] - globals_['global_zp_base_top1']
    lift_overall = rescue_rate * globals_['global_zp_mass_pct'] / 100
    lines.append(f"- Expected overall top1 lift from zp rescue alone: ≈ Δtop1(zp) × mass = **{lift_overall:+.4f}**")
    (out / f"zp_rescue_{args.split}.md").write_text("\n".join(lines))
    (out / f"zp_rescue_{args.split}.json").write_text(
        json.dumps(dict(per_p=per_p, globals=globals_, args=vars(args)),
                   indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
