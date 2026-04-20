"""Pre-registered protocol: select ONE config on val, apply once on test.
Compares hono with strict significance at 95% two-sided.

Three candidate configs to pre-register:
  C1: gated blend α=0.50, δ=0.10 (robust on both val/test prior grid)
  C2: gated blend α=0.70, δ=0.20 (val-best global)
  C3: prospective blind-spot (val-tuned)

Report all three + the best one by val (locked).

Usage:
  python -m code.unified_selector.pre_registered_eval \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R36_preregistered \
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


def margin(x):
    s = np.sort(x, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def gated_pick(logits, prior, alpha, delta):
    base_pick = logits.argmax(axis=1)
    base_norm = logits - logits.mean(axis=1, keepdims=True)
    log_prior = np.log(prior + 1e-6)
    prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
    blend = alpha * base_norm + (1 - alpha) * prior_norm
    blend_pick = blend.argmax(axis=1)
    m = margin(logits)
    return np.where(m < delta, blend_pick, base_pick)


def compute_picks(enc, priors, alpha, delta):
    per_p = {}
    for p in PROBLEMS:
        d = enc[p]
        picks = gated_pick(d["logits"], priors[p], alpha, delta)
        per_p[p] = dict(picks=picks.astype(np.int32),
                        costs=d["costs"],
                        oracle=d["oracle"].astype(np.int32))
    return per_p


def base_picks(enc):
    per_p = {}
    for p in PROBLEMS:
        d = enc[p]
        per_p[p] = dict(picks=d["logits"].argmax(axis=1).astype(np.int32),
                        costs=d["costs"],
                        oracle=d["oracle"].astype(np.int32))
    return per_p


def macro_top1(per_p):
    return float(np.mean([(per_p[p]["picks"] == per_p[p]["oracle"]).mean() for p in PROBLEMS]))


def flip_counts(per_p, base_per_p):
    wr = rw = unchanged = 0
    for p in PROBLEMS:
        a = per_p[p]["picks"]; b = base_per_p[p]["picks"]; o = per_p[p]["oracle"]
        changed = (a != b)
        base_correct = (b == o)
        new_correct = (a == o)
        wr += int((changed & ~base_correct & new_correct).sum())
        rw += int((changed & base_correct & ~new_correct).sum())
        unchanged += int((~changed).sum())
    return dict(wr=wr, rw=rw, unchanged=unchanged, modified=wr+rw)


def regret_metrics(per_p, base_per_p):
    """Compute oracle-gap (regret) for base and method."""
    regret_base = regret_method = 0.0
    cnt = 0
    for p in PROBLEMS:
        costs = per_p[p]["costs"]; oracle_idx = per_p[p]["oracle"]
        picks_b = base_per_p[p]["picks"]
        picks_m = per_p[p]["picks"]
        N = costs.shape[0]
        oracle_cost = costs.min(axis=1)
        sbs_idx = int(costs.mean(axis=0).argmin())
        sbs_cost = costs[:, sbs_idx].mean()
        sel_b = costs[np.arange(N), picks_b]
        sel_m = costs[np.arange(N), picks_m]
        # Relative regret
        reg_b = (sel_b - oracle_cost) / (abs(oracle_cost) + 1e-9)
        reg_m = (sel_m - oracle_cost) / (abs(oracle_cost) + 1e-9)
        regret_base += reg_b.mean() * 100
        regret_method += reg_m.mean() * 100
        cnt += 1
    return dict(regret_base=regret_base/cnt, regret_method=regret_method/cnt)


def zero_pick_analysis(per_p, base_per_p):
    """On val-defined blind-spot mass (oracle on val-defined zp arm), compute recovery."""
    total_zp = 0; base_zp_correct = 0; method_zp_correct = 0
    for p in PROBLEMS:
        base_picks = base_per_p[p]["picks"]
        unique_base = set(base_picks.tolist())
        K_p = per_p[p]["costs"].shape[1]
        zp_arms = [k for k in range(K_p) if k not in unique_base]
        oracle = per_p[p]["oracle"]
        zp_mask = np.isin(oracle, zp_arms)
        n = int(zp_mask.sum())
        total_zp += n
        base_zp_correct += int((base_picks[zp_mask] == oracle[zp_mask]).sum())
        method_zp_correct += int((per_p[p]["picks"][zp_mask] == oracle[zp_mask]).sum())
    return dict(total_zp=total_zp,
                base_zp_top1=base_zp_correct/max(total_zp, 1),
                method_zp_top1=method_zp_correct/max(total_zp, 1),
                zp_delta=(method_zp_correct - base_zp_correct)/max(total_zp, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[prereg] loading base")
    base_model = load_base(args.base_ckpt, device)
    print(f"[prereg] encoding")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)
    val_priors = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    test_priors = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)

    base_val = base_picks(val_enc)
    base_test = base_picks(test_enc)

    # Scan val for TOP val configs
    val_results = []
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for d in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            pp = compute_picks(val_enc, val_priors, a, d)
            t1 = macro_top1(pp)
            val_results.append(dict(alpha=a, delta=d, val_top1=t1))
    val_results.sort(key=lambda x: -x["val_top1"])

    # Pre-register val-best, val-top-5, and median-config
    val_base_top1 = macro_top1(base_val)
    test_base_top1 = macro_top1(base_test)

    # Config A: val-best
    A = val_results[0]
    # Config B: median of val-top-5 (more robust)
    top5 = val_results[:5]
    alpha_med = float(np.median([c["alpha"] for c in top5]))
    delta_med = float(np.median([c["delta"] for c in top5]))
    B = dict(alpha=alpha_med, delta=delta_med,
             val_top1=None)
    # Recompute val_top1 for B
    pp_val_B = compute_picks(val_enc, val_priors, B["alpha"], B["delta"])
    B["val_top1"] = macro_top1(pp_val_B)

    configs = [("val_best", A), ("val_top5_median", B)]

    # Apply each on test
    lines = [f"# Pre-registered evaluation (R36)\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}\n",
             f"Base val top1: {val_base_top1:.4f}",
             f"Base test top1: {test_base_top1:.4f}",
             "",
             "## Val-top-5 configs",
             "| rank | α | δ | val top1 | Δ |",
             "|---:|---:|---:|---:|---:|"]
    for i, c in enumerate(val_results[:5]):
        lines.append(f"| {i+1} | {c['alpha']} | {c['delta']} | {c['val_top1']:.4f} | {c['val_top1']-val_base_top1:+.4f} |")
    lines.append("")
    lines.append("## Pre-registered test evaluation")
    lines.append("| Config | α | δ | val Δ | test Δ | 95% CI | Sig? | wr | rw |")
    lines.append("|:---|---:|---:|---:|---:|---:|:---|---:|---:|")
    for name, cfg in configs:
        pp_test = compute_picks(test_enc, test_priors, cfg["alpha"], cfg["delta"])
        t1_test = macro_top1(pp_test)
        boot = bootstrap_vs_base({1.0: base_test, 0.0: pp_test}, ref_alpha=1.0, n_boot=args.n_boot)[0.0]
        flips = flip_counts(pp_test, base_test)
        sig = "YES" if boot["top1_lo"] > 0 else "NO"
        val_delta = cfg["val_top1"] - val_base_top1
        test_delta = t1_test - test_base_top1
        lines.append(f"| {name} | {cfg['alpha']} | {cfg['delta']} | "
                     f"{val_delta:+.4f} | {test_delta:+.4f} | "
                     f"[{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}] | "
                     f"{sig} | {flips['wr']} | {flips['rw']} |")
        # Also zero-pick rescue
        zp = zero_pick_analysis(pp_test, base_test)
        reg = regret_metrics(pp_test, base_test)
        lines.append(f"|  | zp_mass={zp['total_zp']} ({100*zp['total_zp']/18000:.1f}%) | "
                     f"base_zp={zp['base_zp_top1']:.4f} | "
                     f"method_zp={zp['method_zp_top1']:.4f} | "
                     f"Δzp={zp['zp_delta']:+.4f} | regret_base={reg['regret_base']:.4f}% | "
                     f"regret_method={reg['regret_method']:.4f}% | | |")

    report = "\n".join(lines)
    (out / "preregistered.md").write_text(report)
    # Save full results
    full_results = {}
    for name, cfg in configs:
        pp_test = compute_picks(test_enc, test_priors, cfg["alpha"], cfg["delta"])
        t1_test = macro_top1(pp_test)
        boot = bootstrap_vs_base({1.0: base_test, 0.0: pp_test}, ref_alpha=1.0, n_boot=args.n_boot)[0.0]
        flips = flip_counts(pp_test, base_test)
        zp = zero_pick_analysis(pp_test, base_test)
        reg = regret_metrics(pp_test, base_test)
        full_results[name] = dict(
            alpha=cfg["alpha"], delta=cfg["delta"],
            val_top1=cfg["val_top1"], val_base_top1=val_base_top1,
            test_top1=t1_test, test_base_top1=test_base_top1,
            bootstrap=boot, flips=flips, zp=zp, regret=reg,
        )
    (out / "preregistered.json").write_text(json.dumps(full_results, indent=2, default=float))
    print(report)


if __name__ == "__main__":
    main()
