"""Margin-bucket decomposition (codex R14 must-have).

For each model, partition test instances by oracle-gap (how far the SBS is from oracle):
  bucket 0: <0.1%       (near-tie between SBS and oracle; labels noisy)
  bucket 1: 0.1-0.5%    (small gap; tie-break region)
  bucket 2: 0.5-1%      (moderate gap; fuzzy@1% boundary)
  bucket 3: >1%         (clear gap; easy region)

For each bucket, report:
  - N (count)
  - strict top1 / top2 / top3
  - fuzzy top1 @ 1% (picks within 1% of oracle)
  - mean regret = mean (sel_cost - oracle_cost) / |oracle_cost|  (percent)

This turns the "wall" claim from narrative to evidence: if errors live in
buckets 0+1 (low-margin) and all methods are tied there, the wall is
structural, not a method failure.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, _migrate_state_dict


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("args", {})
    has_bias = "problem_solver_bias" in ck["model"]
    m = UnifiedSelector(
        d=cfg.get("d", 128), depth=cfg.get("depth", 4),
        dropout=cfg.get("dropout", 0.1),
        use_mvrp_factorized=not cfg.get("no_fact", False),
        use_problem_solver_bias=has_bias,
        use_film=cfg.get("use_film", False),
        use_arm_attn=cfg.get("use_arm_attn", False),
        use_coe=cfg.get("use_coe", False),
        arm_attn_heads=cfg.get("arm_attn_heads", 4),
        coe_experts=cfg.get("coe_experts", None),
    ).to(device)
    m.load_state_dict(_migrate_state_dict(ck["model"]), strict=False)
    m.eval()
    return m


@torch.no_grad()
def collect_predictions(model, split: str, device):
    per_p = {}
    for p in PROBLEMS:
        ds = UnifiedProblemDataset(p, split)
        dl = DataLoader(ds, batch_size=32, collate_fn=collate_single_problem, num_workers=0)
        picks = []; costs_all = []
        for b in dl:
            for k, v in b.items():
                if torch.is_tensor(v):
                    b[k] = v.to(device)
            logits = model(b)[:, b["pool_ids"]]
            pick = logits.argmax(dim=1).cpu().numpy()
            picks.append(pick)
            costs_all.append(b["costs"].cpu().numpy())
        picks = np.concatenate(picks)
        costs = np.concatenate(costs_all, axis=0)
        per_p[p] = dict(picks=picks, costs=costs)
    return per_p


BUCKETS = [
    ("<0.1%",   0.0,   0.001),
    ("0.1-0.5%",0.001, 0.005),
    ("0.5-1%",  0.005, 0.01),
    (">1%",     0.01,  1.0),
]


def per_instance_metrics(picks: np.ndarray, costs: np.ndarray):
    """Returns per-instance arrays: pick_rank, regret (relative), sel_cost, oracle_cost, sbs_gap."""
    N, K_p = costs.shape
    oracle_idx = costs.argmin(axis=1)
    oracle_cost = costs[np.arange(N), oracle_idx]
    sel_cost = costs[np.arange(N), picks]
    # rank of selector's pick in oracle's sorted order
    order = np.argsort(costs, axis=1)
    inv_rank = np.argsort(order, axis=1)
    pick_rank = inv_rank[np.arange(N), picks]
    # relative regret
    regret = (sel_cost - oracle_cost) / (np.abs(oracle_cost) + 1e-9)
    # SBS gap = (sbs_cost - oracle_cost) / oracle — characterizes the INSTANCE, not the model
    sbs_idx = int(costs.mean(axis=0).argmin())
    sbs_cost = costs[:, sbs_idx]
    sbs_gap = (sbs_cost - oracle_cost) / (np.abs(oracle_cost) + 1e-9)
    return pick_rank, regret, sel_cost, oracle_cost, sbs_gap


def bucket_metrics(pick_rank, regret, sbs_gap, K_p):
    """For each oracle-gap bucket (characterized by sbs_gap), report metrics."""
    out = {}
    for name, lo, hi in BUCKETS:
        mask = (sbs_gap >= lo) & (sbs_gap < hi)
        if mask.sum() == 0:
            out[name] = None; continue
        r = pick_rank[mask]
        reg = regret[mask]
        out[name] = dict(
            N=int(mask.sum()),
            top1=float((r == 0).mean()),
            top2=float((r < min(2, K_p)).mean()),
            top3=float((r < min(3, K_p)).mean()),
            fuzzy_1pct=float((reg < 0.01).mean()),
            fuzzy_01pct=float((reg < 0.001).mean()),
            mean_regret_pct=float(reg.mean() * 100),
        )
    return out


def macro_bucket(per_p_bucket: Dict, bucket: str):
    """Aggregate one bucket's metrics across all problems (macro average)."""
    vals = [v[bucket] for v in per_p_bucket.values() if v[bucket] is not None]
    if not vals:
        return None
    total_N = sum(v["N"] for v in vals)
    return dict(
        N=total_N,
        num_problems=len(vals),
        top1=float(np.mean([v["top1"] for v in vals])),
        top2=float(np.mean([v["top2"] for v in vals])),
        top3=float(np.mean([v["top3"] for v in vals])),
        fuzzy_1pct=float(np.mean([v["fuzzy_1pct"] for v in vals])),
        mean_regret_pct=float(np.mean([v["mean_regret_pct"] for v in vals])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="ckpt_name=path pairs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model_pairs = []
    for kv in args.models:
        name, path = kv.split("=", 1)
        model_pairs.append((name, path))

    all_results = {}
    ref_model_picks = None
    for name, path in model_pairs:
        print(f"[bucket] loading {name}={path}")
        m = load_model(path, device)
        picks_all = collect_predictions(m, args.split, device)
        per_p_bucket = {}
        for p in PROBLEMS:
            picks = picks_all[p]["picks"]
            costs = picks_all[p]["costs"]
            pr, reg, sc, oc, sbs_gap = per_instance_metrics(picks, costs)
            per_p_bucket[p] = bucket_metrics(pr, reg, sbs_gap, costs.shape[1])
        # Macro over problems
        macro = {b[0]: macro_bucket(per_p_bucket, b[0]) for b in BUCKETS}
        all_results[name] = dict(per_problem=per_p_bucket, macro=macro)
        if ref_model_picks is None:
            ref_model_picks = picks_all

    # Global bucket N counts (should be identical across models since they depend on dataset only)
    bucket_counts = {}
    for b in BUCKETS:
        first_model = list(all_results.values())[0]
        bucket_counts[b[0]] = first_model["macro"][b[0]]["N"] if first_model["macro"][b[0]] else 0

    total = sum(bucket_counts.values())
    mass_pct = {b: 100 * n / total if total > 0 else 0 for b, n in bucket_counts.items()}

    # Build report
    lines = [f"# Margin-Bucket Decomposition — {args.split}\n",
             f"Test instances bucketed by *SBS-vs-oracle gap* = (sbs_cost - oracle_cost)/oracle.\n",
             f"Low gap = SBS already near-optimal; high gap = clear room to beat SBS.\n",
             f"## Bucket mass (total {total} instances)\n"]
    lines.append("| Bucket | N | Mass |")
    lines.append("|:---|---:|---:|")
    for b, lo, hi in BUCKETS:
        lines.append(f"| {b} | {bucket_counts[b]} | {mass_pct[b]:.1f}% |")
    lines.append("")
    lines.append("## Macro metrics per bucket — all models side by side\n")

    model_names = [n for n, _ in model_pairs]
    for b, lo, hi in BUCKETS:
        lines.append(f"### Bucket: {b}  (N={bucket_counts[b]}, {mass_pct[b]:.1f}% of test)")
        lines.append("")
        lines.append("| Model | top1 | top2 | top3 | fuzzy@1% | mean_regret |")
        lines.append("|:---|---:|---:|---:|---:|---:|")
        for name in model_names:
            m = all_results[name]["macro"][b]
            if m:
                lines.append(f"| {name} | {m['top1']:.4f} | {m['top2']:.4f} | {m['top3']:.4f} | {m['fuzzy_1pct']:.4f} | {m['mean_regret_pct']:+.4f}% |")
            else:
                lines.append(f"| {name} | — | — | — | — | — |")
        lines.append("")

    (out / "margin_bucket_report.md").write_text("\n".join(lines))
    (out / "margin_bucket_summary.json").write_text(json.dumps(dict(
        bucket_counts=bucket_counts, mass_pct=mass_pct,
        results={n: {"macro": r["macro"]} for n, r in all_results.items()},
    ), indent=2, default=float))

    # Also save the full per-problem breakdown (for appendix)
    (out / "margin_bucket_full.json").write_text(json.dumps(all_results, indent=2, default=float))
    print("\n".join(lines))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
