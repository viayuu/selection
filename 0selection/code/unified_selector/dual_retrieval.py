"""R35: Dual-retrieval agreement.

Per oracle-pro R1.3:
Use TWO independent retrieval priors:
  - oracle-vote prior (argmax-based): neighbors vote for their oracle arm
  - cost-rank prior (cost-based): neighbors' per-arm inverse ranks

Override base only when both priors agree on a same non-base arm,
base margin < δ, AND oracle-vote top1 prob > γ.

All hyperparameters selected on val, report once on test.

Usage:
  python -m code.unified_selector.dual_retrieval \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R35_dual \
    --k 32 --temperature 0.1 \
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
from .retrieval_cost import compute_cost_prior


def margin_of(x):
    s = np.sort(x, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def apply_dual(enc, priors_vote, priors_cost, delta_gate, gamma_conf):
    per_p = {}; flips = dict(wr=0, rw=0, unchanged=0, modified=0)
    for p in PROBLEMS:
        d = enc[p]
        logits = d["logits"]; costs = d["costs"]; oracle = d["oracle"]
        pv = priors_vote[p]; pc = priors_cost[p]
        base_pick = logits.argmax(axis=1)
        vote_pick = pv.argmax(axis=1)
        # Cost-based: for z-score/rank normalization, argmax of NEGATED cost (low cost = good)
        # cost_prior is stored as the cost vector (low=good)
        cost_pick = pc.argmin(axis=1)  # low cost = best arm
        base_margin = margin_of(logits)
        gate_ok = base_margin < delta_gate
        pv_norm = pv / (pv.sum(axis=1, keepdims=True) + 1e-6)
        pv_top1_prob = pv_norm.max(axis=1)
        conf_ok = pv_top1_prob > gamma_conf
        agree = (vote_pick == cost_pick)
        disagree_with_base = (vote_pick != base_pick)
        override = gate_ok & conf_ok & agree & disagree_with_base
        picks = np.where(override, vote_pick, base_pick)
        # Stats
        changed = (picks != base_pick)
        base_correct = (base_pick == oracle)
        new_correct = (picks == oracle)
        flips["wr"] += int((changed & ~base_correct & new_correct).sum())
        flips["rw"] += int((changed & base_correct & ~new_correct).sum())
        flips["modified"] += int(changed.sum())
        flips["unchanged"] += int((~changed).sum())
        per_p[p] = dict(picks=picks.astype(np.int32), costs=costs,
                        oracle=oracle.astype(np.int32))
    return per_p, flips


def macro_top1(per_p):
    return float(np.mean([(per_p[p]["picks"] == per_p[p]["oracle"]).mean() for p in PROBLEMS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--cost-normalize", default="rank")
    ap.add_argument("--delta-grid", nargs="+", type=float,
                    default=[0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0, 10.0])
    ap.add_argument("--gamma-grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[dual] loading base")
    base_model = load_base(args.base_ckpt, device)
    print(f"[dual] encoding splits")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    print(f"[dual] oracle-vote priors")
    val_priors_vote = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    test_priors_vote = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)

    print(f"[dual] cost-rank priors")
    val_priors_cost = compute_cost_prior(train_enc, val_enc, k=args.k, temperature=args.temperature, normalize=args.cost_normalize)
    test_priors_cost = compute_cost_prior(train_enc, test_enc, k=args.k, temperature=args.temperature, normalize=args.cost_normalize)

    base_val = {p: dict(picks=val_enc[p]["logits"].argmax(axis=1).astype(np.int32),
                         costs=val_enc[p]["costs"], oracle=val_enc[p]["oracle"].astype(np.int32))
                for p in PROBLEMS}
    base_test = {p: dict(picks=test_enc[p]["logits"].argmax(axis=1).astype(np.int32),
                          costs=test_enc[p]["costs"], oracle=test_enc[p]["oracle"].astype(np.int32))
                 for p in PROBLEMS}
    val_base_top1 = macro_top1(base_val)
    test_base_top1 = macro_top1(base_test)

    # Grid search on val
    best = dict(top1=-1)
    for delta in args.delta_grid:
        for gamma in args.gamma_grid:
            val_pp, val_flips = apply_dual(val_enc, val_priors_vote, val_priors_cost, delta, gamma)
            t1 = macro_top1(val_pp)
            if t1 > best["top1"]:
                best = dict(top1=t1, delta=delta, gamma=gamma, val_flips=val_flips)

    print(f"[dual] val-best: delta={best['delta']}, gamma={best['gamma']}, val_top1={best['top1']:.4f}")
    test_pp, test_flips = apply_dual(test_enc, test_priors_vote, test_priors_cost,
                                       best["delta"], best["gamma"])
    test_top1 = macro_top1(test_pp)
    boot = bootstrap_vs_base({1.0: base_test, 0.0: test_pp}, ref_alpha=1.0, n_boot=args.n_boot)[0.0]

    lines = [f"# Dual-retrieval agreement (R35)\n",
             f"Base: {args.base_ckpt}, k={args.k}, cost_norm={args.cost_normalize}",
             f"Val-selected: δ={best['delta']}, γ={best['gamma']}",
             "",
             f"VAL: base {val_base_top1:.4f} → dual {best['top1']:.4f} (Δ={best['top1']-val_base_top1:+.4f})",
             f"TEST: base {test_base_top1:.4f} → dual {test_top1:.4f} (Δ={test_top1-test_base_top1:+.4f})",
             f"Test wrong→right: {test_flips['wr']}, right→wrong: {test_flips['rw']}, modified: {test_flips['modified']} ({test_flips['modified']/18000*100:.1f}%)",
             "",
             f"Bootstrap Δtop1: {boot['top1_mean']:+.4f} [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}]",
             f"Bootstrap Δcost%: {boot['cost_mean']:+.4f}% [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}]",
             f"**SIGNIFICANT** (95% two-sided): {'YES' if boot['top1_lo'] > 0 else 'NO'}"]
    (out / "dual.md").write_text("\n".join(lines))
    (out / "dual.json").write_text(json.dumps(dict(
        best=best, test_base_top1=test_base_top1, test_top1=test_top1,
        bootstrap=boot, test_flips=test_flips, args=vars(args),
    ), indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
