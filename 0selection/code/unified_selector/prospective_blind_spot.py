"""R34: Prospective blind-spot override.

Per oracle-pro R1.2:
Define blind-spot arms by VALIDATION base-model pick frequency < τ.
Override base only when:
  - retrieval top1 is a val-defined blind-spot arm
  - base margin < δ
  - retrieval concentration (top1 prior prob) > γ

All hyperparameters (τ, δ, γ, α) selected on val, evaluated once on test.

Usage:
  python -m code.unified_selector.prospective_blind_spot \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R34_prospective \
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


def compute_val_pick_freq(val_enc):
    """Return per-problem per-arm val pick frequency."""
    freqs = {}
    for p in PROBLEMS:
        logits = val_enc[p]["logits"]
        picks = logits.argmax(axis=1)
        K_p = logits.shape[1]
        f = np.bincount(picks, minlength=K_p) / len(picks)
        freqs[p] = f
    return freqs


def margin_of(base_logits):
    s = np.sort(base_logits, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def apply_config(enc, priors, val_freqs, tau_bs, delta_gate, gamma_conf, alpha):
    """Override only when: retrieval top1 is blind-spot (val freq < tau_bs),
    base margin < delta_gate, retrieval max prob > gamma_conf."""
    per_p = {}; flips = dict(wr=0, rw=0, unchanged=0, modified=0)
    for p in PROBLEMS:
        d = enc[p]
        logits = d["logits"]; costs = d["costs"]; oracle = d["oracle"]
        prior = priors[p]
        base_pick = logits.argmax(axis=1)
        prior_pick = prior.argmax(axis=1)
        base_correct = (base_pick == oracle)
        # Conditions
        bs_mask = val_freqs[p] < tau_bs  # (K_p,) blind-spot arms
        prior_top1_is_bs = bs_mask[prior_pick]
        base_margin = margin_of(logits)
        gate_ok = base_margin < delta_gate
        prior_p_norm = prior / (prior.sum(axis=1, keepdims=True) + 1e-6)
        prior_max = prior_p_norm.max(axis=1)
        conf_ok = prior_max > gamma_conf
        disagree = (base_pick != prior_pick)
        override = prior_top1_is_bs & gate_ok & conf_ok & disagree
        # Final pick: when override, pick the retrieval-top1 arm directly (HARD override)
        picks = np.where(override, prior_pick, base_pick)
        changed = (picks != base_pick)
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
    ap.add_argument("--tau-grid", nargs="+", type=float, default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--delta-grid", nargs="+", type=float, default=[0.05, 0.08, 0.10, 0.15, 0.20, 1.0, 10.0])
    ap.add_argument("--gamma-grid", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[prospective] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[prospective] encoding splits")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    val_freqs = compute_val_pick_freq(val_enc)

    print(f"[prospective] priors")
    val_priors = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    test_priors = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)

    # Base reference
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
    for tau in args.tau_grid:
        for delta in args.delta_grid:
            for gamma in args.gamma_grid:
                val_pp, val_flips = apply_config(val_enc, val_priors, val_freqs,
                                                   tau, delta, gamma, args.alpha)
                t1 = macro_top1(val_pp)
                if t1 > best["top1"]:
                    best = dict(top1=t1, tau=tau, delta=delta, gamma=gamma,
                                 val_flips=val_flips)

    print(f"[prospective] val-best: tau={best['tau']}, delta={best['delta']}, gamma={best['gamma']}, val_top1={best['top1']:.4f}")

    # Apply on test
    test_pp, test_flips = apply_config(test_enc, test_priors, val_freqs,
                                         best["tau"], best["delta"], best["gamma"], args.alpha)
    test_top1 = macro_top1(test_pp)
    boot = bootstrap_vs_base({1.0: base_test, 0.0: test_pp}, ref_alpha=1.0, n_boot=args.n_boot)[0.0]

    lines = [f"# Prospective blind-spot override (R34)\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}, alpha={args.alpha}",
             f"Val-selected: τ_bs={best['tau']}, δ_gate={best['delta']}, γ_conf={best['gamma']}",
             "",
             f"VAL: base {val_base_top1:.4f} → method {best['top1']:.4f} (Δ={best['top1']-val_base_top1:+.4f})",
             f"TEST: base {test_base_top1:.4f} → method {test_top1:.4f} (Δ={test_top1-test_base_top1:+.4f})",
             f"Test wrong→right: {test_flips['wr']}, right→wrong: {test_flips['rw']}, modified: {test_flips['modified']}",
             f"Test modified fraction: {test_flips['modified'] / 18000:.3f}",
             "",
             f"Bootstrap Δtop1: {boot['top1_mean']:+.4f} [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}]",
             f"Bootstrap Δcost%: {boot['cost_mean']:+.4f}% [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}]",
             f"**SIGNIFICANT** (95% two-sided, Δtop1>0): {'YES' if boot['top1_lo'] > 0 else 'NO'}",]
    (out / "prospective.md").write_text("\n".join(lines))
    (out / "prospective.json").write_text(json.dumps(dict(
        best=best, test_base_top1=test_base_top1, test_top1=test_top1,
        bootstrap=boot, test_flips=test_flips, args=vars(args),
    ), indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
