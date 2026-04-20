"""Disagreement-gated blend: apply retrieval blend ONLY when retrieval's top1
disagrees with base's top1. Rationale: if base and retrieval already agree,
no need to change anything. Disagreement is a (noisy) signal that base might
be wrong and retrieval might be right.

Usage:
  python -m code.unified_selector.disagree_blend \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R32_disagree \
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


def margin(x):
    s = np.sort(x, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def pick_disagree(base_logits, prior, alpha, gate_delta,
                   require_confident_prior=False, prior_conf_delta=0.0):
    """Base picks. When base.top1 != prior.top1 AND base_margin < gate_delta,
    use α-blend. Else use base.

    If require_confident_prior, additionally require prior top1-top2 mass > prior_conf_delta.
    """
    base_picks = base_logits.argmax(axis=1)
    prior_picks = prior.argmax(axis=1)
    base_margin = margin(base_logits)
    disagree = (base_picks != prior_picks)
    gate_ok = (base_margin < gate_delta)
    use_blend = disagree & gate_ok
    if require_confident_prior:
        prior_sorted = np.sort(prior, axis=1)[:, ::-1]
        prior_margin = prior_sorted[:, 0] - prior_sorted[:, 1]
        prior_conf = (prior_margin > prior_conf_delta)
        use_blend = use_blend & prior_conf
    base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
    log_prior = np.log(prior + 1e-6)
    prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
    blend = alpha * base_norm + (1 - alpha) * prior_norm
    blend_picks = blend.argmax(axis=1)
    return np.where(use_blend, blend_picks, base_picks), use_blend.mean()


def eval_config(data_enc, priors, alpha, gate_delta, require_conf=False, prior_conf=0.0):
    per_p = {}
    frac_blended = []
    for p in PROBLEMS:
        q = data_enc[p]
        picks, frac = pick_disagree(q["logits"], priors[p], alpha, gate_delta,
                                     require_conf, prior_conf)
        top1 = float((picks == q["oracle"]).mean())
        mean_c = float(q["costs"][np.arange(q["costs"].shape[0]), picks].mean())
        sbs_idx = int(q["costs"].mean(axis=0).argmin())
        sbs_mean = q["costs"][:, sbs_idx].mean()
        vs_sbs = 100 * (mean_c - sbs_mean) / (abs(sbs_mean) + 1e-9)
        per_p[p] = dict(picks=picks.astype(np.int32), top1=top1, mean_cost=mean_c,
                        vs_sbs_pct=vs_sbs, frac_blended=float(frac),
                        costs=q["costs"], oracle=q["oracle"].astype(np.int32))
        frac_blended.append(frac)
    macro_top1 = float(np.mean([v["top1"] for v in per_p.values()]))
    macro_vs_sbs = float(np.mean([v["vs_sbs_pct"] for v in per_p.values()]))
    return per_p, macro_top1, macro_vs_sbs, float(np.mean(frac_blended))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--alpha-grid", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    ap.add_argument("--gate-delta-grid", nargs="+", type=float,
                    default=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0, 10.0])
    ap.add_argument("--prior-conf-grid", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[disagree] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[disagree] encoding train+{args.split}")
    train_enc = encode_instances(base_model, "train", device)
    test_enc = encode_instances(base_model, args.split, device)
    print(f"[disagree] computing k={args.k} retrieval priors")
    priors = compute_retrieval_prior(train_enc, test_enc, k=args.k,
                                      temperature=args.temperature)

    base_picks = {p: dict(picks=test_enc[p]["logits"].argmax(axis=1).astype(np.int32),
                           costs=test_enc[p]["costs"],
                           oracle=test_enc[p]["oracle"].astype(np.int32)) for p in PROBLEMS}
    base_top1 = float(np.mean([(base_picks[p]["picks"] == base_picks[p]["oracle"]).mean()
                                for p in PROBLEMS]))
    sbs_means = {p: float(test_enc[p]["costs"][:, int(test_enc[p]["costs"].mean(axis=0).argmin())].mean())
                 for p in PROBLEMS}
    base_vs_sbs = float(np.mean([100 * (base_picks[p]["costs"][np.arange(1000), base_picks[p]["picks"]].mean() - sbs_means[p]) / (abs(sbs_means[p]) + 1e-9)
                                 for p in PROBLEMS]))

    results = []
    for a in args.alpha_grid:
        for g in args.gate_delta_grid:
            for pc in args.prior_conf_grid:
                per_p, m_top1, m_vs_sbs, frac = eval_config(
                    test_enc, priors, a, g,
                    require_conf=(pc > 0), prior_conf=pc)
                pp = {p: dict(picks=per_p[p]["picks"],
                               costs=per_p[p]["costs"],
                               oracle=per_p[p]["oracle"]) for p in PROBLEMS}
                boot = bootstrap_vs_base({1.0: base_picks, 0.0: pp},
                                          ref_alpha=1.0, n_boot=args.n_boot)[0.0]
                key = f"a={a:.2f}_d={g:.2f}_pc={pc:.2f}"
                results.append(dict(
                    key=key, alpha=a, gate_delta=g, prior_conf=pc,
                    frac_blended=frac, macro_top1=m_top1,
                    macro_vs_sbs=m_vs_sbs, bootstrap=boot,
                ))

    # Sort by macro_top1
    results.sort(key=lambda r: -r["macro_top1"])
    lines = [f"# Disagreement-gated blend\n",
             f"Base: {args.base_ckpt}, k={args.k}\n",
             f"BASE top1: {base_top1:.4f}, vs_sbs: {base_vs_sbs:+.4f}%\n",
             "| Config | frac blended | top1 | Δtop1 [95% CI] | vs_sbs | Δcost% [95% CI] |",
             "|:---|---:|---:|---:|---:|---:|"]
    for r in results[:30]:
        b = r["bootstrap"]
        sig_t = "*" if (b["top1_lo"] > 0 or b["top1_hi"] < 0) else ""
        sig_c = "*" if (b["cost_lo"] > 0 or b["cost_hi"] < 0) else ""
        lines.append(f"| {r['key']} | {r['frac_blended']:.3f} | {r['macro_top1']:.4f} | "
                     f"{b['top1_mean']:+.4f} [{b['top1_lo']:+.4f}, {b['top1_hi']:+.4f}]{sig_t} | "
                     f"{r['macro_vs_sbs']:+.4f}% | "
                     f"{b['cost_mean']:+.4f}% [{b['cost_lo']:+.4f}, {b['cost_hi']:+.4f}]{sig_c} |")
    (out / f"disagree_{args.split}.md").write_text("\n".join(lines))
    (out / f"disagree_{args.split}.json").write_text(json.dumps(dict(
        results=results, base_top1=base_top1, base_vs_sbs=base_vs_sbs,
        args=vars(args),
    ), indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
