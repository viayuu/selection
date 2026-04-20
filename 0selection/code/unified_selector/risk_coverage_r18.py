"""Risk-coverage analysis for R18 and R21 (codex R14 supporting figure).

Unlike the original risk_coverage.py (for R5 zeroshot on held-out composites),
this version computes risk-coverage on the unified-selector's own picks (not
SBS-gate), across all 18 problems × val and test.

Idea:
  Take base R18 + optional pairwise R21 reranker, produce per-instance
  (pick_arm, confidence), where confidence = base top1 margin = logit_top1 - logit_top2.
  Order instances by confidence descending; for each coverage level τ (fraction
  of instances the model "speaks" on, high-confidence first), compute the
  mean regret on that covered subset.

  If the model is well-calibrated, low-confidence = high regret (good). And
  if pairwise reranker helps only on low-confidence instances (= tie-break
  cases), that's the "boundary-helps-val-not-test" pattern codex demanded.

Output: CSV + plot showing (coverage, mean_regret) curves for R18 and for
R21-blend-α0.5, on val and on test, per problem and macro.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .registry import PROBLEMS
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, _migrate_state_dict
from .pairwise_train import PairwiseComparator, build_base_topk, build_base


def load_base(path, device):
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


def load_comparator(pair_ckpt, device):
    ck = torch.load(pair_ckpt, map_location=device, weights_only=False)
    d_token = ck.get("d_token", 128)
    d_solver = ck.get("d_solver", 128)
    d_hidden = ck.get("d_hidden", 256)
    comp = PairwiseComparator(d_token=d_token, d_solver=d_solver, d_hidden=d_hidden,
                              heads=ck["args"].get("heads", 4), dropout=0.0,
                              cbits_dim=5).to(device)
    comp.load_state_dict(ck["model"])
    comp.eval()
    return comp


@torch.no_grad()
def collect_base_features(base_model, split, device, topk=3):
    """Returns per-problem arrays: margins (top1 - top2), picks, regrets, and top_idx for blend later."""
    per_p = {}
    for p in PROBLEMS:
        ds = UnifiedProblemDataset(p, split)
        dl = DataLoader(ds, batch_size=32, collate_fn=collate_single_problem, num_workers=0)
        margins = []; picks = []; costs_all = []; top_idx_all = []
        for b in dl:
            for k, v in b.items():
                if torch.is_tensor(v):
                    b[k] = v.to(device)
            logits = base_model(b)
            lp = logits[:, b["pool_ids"]]
            K_p = lp.shape[1]
            topk_eff = min(topk, K_p)
            vals, idx = lp.topk(topk_eff, dim=1)
            margin = (vals[:, 0] - vals[:, 1]) if K_p > 1 else vals[:, 0]
            margins.append(margin.cpu().numpy())
            picks.append(idx[:, 0].cpu().numpy())
            top_idx_all.append(idx.cpu().numpy())
            costs_all.append(b["costs"].cpu().numpy())
        per_p[p] = dict(
            margin=np.concatenate(margins),
            base_pick=np.concatenate(picks),
            costs=np.concatenate(costs_all, axis=0),
            top_idx=np.concatenate(top_idx_all, axis=0),
        )
    return per_p


@torch.no_grad()
def collect_blend_picks(base_model, comparator, split, device, alpha, topk):
    """Run R18 + pairwise blend at alpha; return per-problem picks."""
    per_p = {}
    for p in PROBLEMS:
        ds = UnifiedProblemDataset(p, split)
        dl = DataLoader(ds, batch_size=16, collate_fn=collate_single_problem, num_workers=0)
        picks = []; costs_all = []
        for b in dl:
            for k, v in b.items():
                if torch.is_tensor(v):
                    b[k] = v.to(device)
            info = build_base_topk(base_model, b, topk)
            B, K = info["top_idx"].shape
            base_logits = base_model(b)[:, b["pool_ids"]]
            base_topk_logits = base_logits.gather(1, info["top_idx"])
            base_topk_logits = base_topk_logits - base_topk_logits.mean(dim=1, keepdim=True)
            solver_topk = info["solver_emb_topk"]
            S = torch.zeros(B, K, K, device=device)
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    s_ij = comparator(info["tokens"], info["tok_mask"],
                                      solver_topk[:, i], solver_topk[:, j], b["cbits"])
                    S[:, i, j] = s_ij
            arm_score = S.sum(dim=2)
            arm_score = arm_score - arm_score.mean(dim=1, keepdim=True)
            scale = arm_score.std(dim=1, keepdim=True).clamp_min(1e-6)
            base_scale = base_topk_logits.std(dim=1, keepdim=True).clamp_min(1e-6)
            pair_norm = arm_score / scale * base_scale
            blend = alpha * base_topk_logits + (1 - alpha) * pair_norm
            pick_pos = blend.argmax(dim=1)
            pick_pool = info["top_idx"].gather(1, pick_pos.unsqueeze(1)).squeeze(1)
            picks.append(pick_pool.cpu().numpy())
            costs_all.append(b["costs"].cpu().numpy())
        per_p[p] = dict(pick=np.concatenate(picks), costs=np.concatenate(costs_all, axis=0))
    return per_p


def compute_regret(costs, picks):
    N = costs.shape[0]
    oracle_cost = costs.min(axis=1)
    sel_cost = costs[np.arange(N), picks]
    return (sel_cost - oracle_cost) / (np.abs(oracle_cost) + 1e-9)  # relative regret


def risk_coverage_curve(regrets, margins, n_bins=30):
    """Sort by margin (descending) = most-confident first.
    For each coverage level (top-k fraction), compute mean regret on that subset.
    Well-calibrated: low-coverage has near-zero regret; high-coverage has higher regret.
    """
    order = np.argsort(-margins)  # highest margin first
    sorted_regrets = regrets[order]
    N = len(regrets)
    cov_points = np.linspace(0.02, 1.0, n_bins)  # 2% to 100% coverage
    curve_cov = []
    curve_reg = []
    for cov in cov_points:
        k = int(cov * N)
        if k == 0:
            continue
        mean_reg = sorted_regrets[:k].mean()
        curve_cov.append(cov)
        curve_reg.append(mean_reg * 100)  # percent
    return np.array(curve_cov), np.array(curve_reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--pair-ckpt", default=None, help="R21 pairwise comparator (optional)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Blend α if pair-ckpt given")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[rc] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    comparator = None
    if args.pair_ckpt:
        print(f"[rc] loading pairwise {args.pair_ckpt}")
        comparator = load_comparator(args.pair_ckpt, device)

    results = {}
    for split in args.splits:
        print(f"[rc] collecting base features on {split}")
        base_feat = collect_base_features(base_model, split, device, topk=args.topk)

        # Base picks
        all_margins = []; all_base_regret = []
        for p in PROBLEMS:
            d = base_feat[p]
            all_margins.append(d["margin"])
            all_base_regret.append(compute_regret(d["costs"], d["base_pick"]))
        margins = np.concatenate(all_margins)
        base_regret = np.concatenate(all_base_regret)

        # Pairwise-blend picks (if available)
        blend_regret = None
        if comparator is not None:
            print(f"[rc] running pairwise blend (α={args.alpha}) on {split}")
            blend = collect_blend_picks(base_model, comparator, split, device, args.alpha, args.topk)
            all_blend_regret = []
            for p in PROBLEMS:
                all_blend_regret.append(compute_regret(blend[p]["costs"], blend[p]["pick"]))
            blend_regret = np.concatenate(all_blend_regret)

        # Sort/risk-coverage curves — sort by MARGINS (confidence), not by regret, so coverage is honest
        cov_base, reg_base = risk_coverage_curve(base_regret, margins)
        curves = {"base_R18": (cov_base, reg_base)}
        if blend_regret is not None:
            cov_b, reg_b = risk_coverage_curve(blend_regret, margins)
            curves["blend_a0.5"] = (cov_b, reg_b)

        results[split] = dict(
            curves={k: {"coverage": v[0].tolist(), "mean_regret_pct": v[1].tolist()} for k, v in curves.items()},
            base_mean_regret_pct=float(base_regret.mean() * 100),
            blend_mean_regret_pct=float(blend_regret.mean() * 100) if blend_regret is not None else None,
        )

    # Plot
    fig, axes = plt.subplots(1, len(args.splits), figsize=(6 * len(args.splits), 4.5), sharey=False)
    if len(args.splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, args.splits):
        for name, data in results[split]["curves"].items():
            ax.plot(np.array(data["coverage"]) * 100, data["mean_regret_pct"], label=name, linewidth=1.8)
        ax.set_xlabel("Coverage (% of test, high-confidence first)")
        ax.set_ylabel("Mean relative regret (%) on covered subset")
        ax.set_title(f"Risk-Coverage — {split}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (out / "risk_coverage.json").write_text(json.dumps(results, indent=2, default=float))

    # Also write a markdown report
    lines = [f"# Risk-Coverage Analysis\n",
             f"Base: {args.base_ckpt}",
             f"Pairwise: {args.pair_ckpt if args.pair_ckpt else '(not used)'}",
             f"Blend α: {args.alpha if comparator is not None else 'n/a'}",
             f"Top-K for pair: {args.topk}\n",
             "## Mean relative regret (all instances, %)"]
    for split, r in results.items():
        lines.append(f"- **{split}**: base_R18={r['base_mean_regret_pct']:.4f}%"
                     + (f", blend_a{args.alpha}={r['blend_mean_regret_pct']:.4f}%" if r['blend_mean_regret_pct'] is not None else ""))
    lines.append("\n## Risk-coverage curve values (coverage fraction, mean regret %)\n")
    for split, r in results.items():
        lines.append(f"### {split}")
        for name, data in r["curves"].items():
            lines.append(f"- {name}: " + "; ".join(f"({c:.2f},{reg:.3f})" for c, reg in zip(data["coverage"][::5], data["mean_regret_pct"][::5])))
        lines.append("")
    (out / "risk_coverage_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nPlot: {out}/risk_coverage.png")


if __name__ == "__main__":
    main()
