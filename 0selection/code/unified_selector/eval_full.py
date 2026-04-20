"""Unified evaluation + ensemble inference with full 观测指标/观测指标.md metrics.

Usage:
    # single ckpt
    python -m code.unified_selector.eval_full --ckpts code/unified_selector/runs/R4_soft_bias/best.pt \
           --split test --out code/unified_selector/runs/eval_R4_soft_bias_test

    # 3-seed ensemble
    python -m code.unified_selector.eval_full --ckpts ckpt1 ckpt2 ckpt3 \
           --split test --out code/unified_selector/runs/ensemble_R4_test --ensemble logit
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, P2I, POOLS, M_GLOBAL, problem_to_pool_mask
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, _migrate_state_dict


def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("args", {})
    has_bias = "problem_solver_bias" in ck["model"]
    model = UnifiedSelector(
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
    sd = _migrate_state_dict(ck["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, cfg


@torch.no_grad()
def per_problem_eval(models: List[torch.nn.Module], problem: str, split: str, device: str,
                     ensemble: str = "logit", tta_8fold: bool = False):
    """Evaluate one or many models on a single problem × split, with requested ensemble mode.
    Returns per-problem metrics plus raw arrays for downstream tables.

    If ``tta_8fold=True``, applies the 8 D4 transforms to coord problems at test time and
    averages logits across folds (no retraining).  Skipped for ATSP (no canonical 2D).
    """
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, collate_fn=collate_single_problem, num_workers=0, shuffle=False)
    preds_all = []
    costs_all = []
    logp_all = []
    best_idx_all = []

    from .data import _augment_xy
    apply_tta = tta_8fold and problem != "ATSP"

    for b in dl:
        for k, v in b.items():
            if torch.is_tensor(v):
                b[k] = v.to(device)
        # TTA: run 8 D4 transforms on the coord columns [0:2] and average logits
        if apply_tta and "node" in b:
            fold_logits = []
            original_node = b["node"].clone()
            for fold in range(8):
                b["node"] = original_node.clone()
                b["node"][..., 0:2] = _augment_xy(original_node[..., 0:2], fold)
                logits_list = [m(b) for m in models]
                if ensemble == "logit":
                    fl = torch.stack(logits_list, dim=0).mean(0)
                elif ensemble == "prob":
                    fp = torch.stack([F.softmax(l, dim=1) for l in logits_list], dim=0).mean(0)
                    fl = torch.log(fp.clamp_min(1e-20))
                else:
                    fl = logits_list[0]
                fold_logits.append(fl)
            logits = torch.stack(fold_logits, dim=0).mean(0)
            # Restore for any downstream
            b["node"] = original_node
        else:
            logits_list = [m(b) for m in models]
            if ensemble == "logit":
                logits = torch.stack(logits_list, dim=0).mean(0)
            elif ensemble == "prob":
                probs = torch.stack([F.softmax(l, dim=1) for l in logits_list], dim=0).mean(0)
                logits = torch.log(probs.clamp_min(1e-20))
            else:
                logits = logits_list[0]
        log_p = F.log_softmax(logits, dim=1)
        log_p_pool = log_p[:, b["pool_ids"]]  # (B, K_p)
        pred = log_p_pool.argmax(dim=1)       # (B,)
        preds_all.append(pred.cpu().numpy())
        costs_all.append(b["costs"].cpu().numpy())
        logp_all.append(log_p_pool.cpu().numpy())
        best_idx_all.append(b["costs"].argmin(dim=1).cpu().numpy())

    pred = np.concatenate(preds_all)
    costs = np.concatenate(costs_all)     # (N, K_p)
    log_p = np.concatenate(logp_all)      # (N, K_p)
    best_idx = np.concatenate(best_idx_all)
    N, K_p = costs.shape

    # top1/2/3 = fraction where selector's pick is in oracle top-k (per 观测指标 spec).
    true_rank = np.argsort(costs, axis=1)                  # (N, K_p) best-first oracle order
    inv_rank = np.argsort(true_rank, axis=1)               # rank of each pool slot
    pick_rank = inv_rank[np.arange(N), pred]
    top1 = float((pick_rank == 0).mean())
    top2 = float((pick_rank < min(2, K_p)).mean())
    top3 = float((pick_rank < min(3, K_p)).mean())

    sel_costs = costs[np.arange(N), pred]
    sbs_pool_idx = int(np.bincount(best_idx, minlength=K_p).argmax())  # proxy: most-frequent winner (fallback)
    # use audit sbs_pool_idx below from caller

    return dict(
        problem=problem, split=split, N=N, K_p=K_p,
        pool_names=[s for s in POOLS[problem]],
        pred=pred.tolist(),
        costs=costs.tolist(),
        top1=top1, top2=top2, top3=top3,
        mean_cost=float(sel_costs.mean()),
        # per-method columns
        method_top1_winrate=[float((best_idx == k).mean()) for k in range(K_p)],
        method_mean_cost=[float(costs[:, k].mean()) for k in range(K_p)],
        # arm distribution = how often each method is picked
        arm_pick_count=[int((pred == k).sum()) for k in range(K_p)],
    )


def build_tables(per_p: Dict[str, dict], audit: dict, split: str, out_dir: Path):
    """Produce the tables listed in 观测指标/观测指标.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    macro = dict(top1=[], top2=[], top3=[], mean_cost=[], vs_sbs=[])
    # =========== Table A: top1/2/3 selector vs oracle, and per-method winrate =============
    lines.append("# 观测指标报表\n")
    lines.append(f"Split: **{split}**  ·  18 problems\n")
    lines.append("## A. Selector top-k vs oracle, and per-method top1 winrate\n")
    lines.append("| Problem | selector top1 | top2 | top3 | best top1 (oracle) | best top2 | best top3 | beat_top1 | lose_top1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for p, r in per_p.items():
        names = r["pool_names"]
        mt1 = r["method_top1_winrate"]
        best_top1 = max(mt1)
        # oracle top2/top3 by definition = 1.0 (best always has rank 0) — keep for explicit comparison
        # selector beat / lose vs each method on top1:
        beat = [names[k] for k in range(r["K_p"]) if r["top1"] > mt1[k] + 1e-6]
        lose = [names[k] for k in range(r["K_p"]) if r["top1"] < mt1[k] - 1e-6]
        lines.append(f"| {p} | {r['top1']:.3f} | {r['top2']:.3f} | {r['top3']:.3f} | {best_top1:.3f} | 1.000 | 1.000 | {', '.join(beat) or '—'} | {', '.join(lose) or '—'} |")
        macro["top1"].append(r["top1"]); macro["top2"].append(r["top2"]); macro["top3"].append(r["top3"])
    lines.append(f"| **MACRO** | **{np.mean(macro['top1']):.4f}** | **{np.mean(macro['top2']):.4f}** | **{np.mean(macro['top3']):.4f}** | | | | | |\n")

    # =========== Table B: mean_cost per problem, selector vs each method, beat/lose lists =======
    lines.append("## B. Mean cost — selector vs SBS, and per-method mean cost\n")
    lines.append("| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |")
    lines.append("|---|---:|---:|---:|---|---|")
    for p, r in per_p.items():
        names = r["pool_names"]
        mmc = r["method_mean_cost"]
        sbs = audit[p][split]["sbs_mean"]
        vs_sbs = (r["mean_cost"] - sbs) / (abs(sbs) + 1e-9) * 100
        beat = [names[k] for k in range(r["K_p"]) if r["mean_cost"] < mmc[k] - 1e-6]
        lose = [names[k] for k in range(r["K_p"]) if r["mean_cost"] > mmc[k] + 1e-6]
        macro["mean_cost"].append(r["mean_cost"]); macro["vs_sbs"].append(vs_sbs)
        lines.append(f"| {p} | {r['mean_cost']:.4f} | {sbs:.4f} | {vs_sbs:+.3f}% | {', '.join(beat) or '—'} | {', '.join(lose) or '—'} |")
    lines.append(f"| **MACRO vs_sbs** | | | **{np.mean(macro['vs_sbs']):+.4f}%** | | |\n")

    # =========== Table C: per-method comparison matrix ==========================
    lines.append("## C. Per-method comparison (mean cost on val/test)\n")
    for p, r in per_p.items():
        names = r["pool_names"]
        mmc = r["method_mean_cost"]
        mt1 = r["method_top1_winrate"]
        lines.append(f"### {p}  (selector top1={r['top1']:.3f}, mean_cost={r['mean_cost']:.4f})")
        lines.append("| Method | oracle winrate | mean_cost | pick count |")
        lines.append("|---|---:|---:|---:|")
        for k, nm in enumerate(names):
            lines.append(f"| {nm} | {mt1[k]:.3f} | {mmc[k]:.4f} | {r['arm_pick_count'][k]} |")
        lines.append("")

    # =========== Table D: arm distribution =======================================
    lines.append("## D. Arm distribution (how often selector picks each method)\n")
    for p, r in per_p.items():
        total = sum(r["arm_pick_count"]) or 1
        shares = [f"{n}:{c/total*100:.1f}%" for n, c in zip(r["pool_names"], r["arm_pick_count"])]
        lines.append(f"- **{p}**: {' | '.join(shares)}")

    # ============================================================================
    (out_dir / "metrics_report.md").write_text("\n".join(lines))
    summary = dict(
        split=split,
        macro_top1=float(np.mean(macro["top1"])),
        macro_top2=float(np.mean(macro["top2"])),
        macro_top3=float(np.mean(macro["top3"])),
        macro_mean_cost=float(np.mean(macro["mean_cost"])),
        macro_vs_sbs_pct=float(np.mean(macro["vs_sbs"])),
        per_problem=per_p,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ensemble", default="logit", choices=["logit", "prob", "none"])
    ap.add_argument("--tta-8fold", action="store_true", help="Apply 8-fold D4 TTA on coord problems.")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    audit = json.loads(Path(args.audit).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    models = [load_model(c, args.device)[0] for c in args.ckpts]
    print(f"[eval] loaded {len(models)} model(s), ensemble={args.ensemble}")

    per_p = {}
    for p in PROBLEMS:
        r = per_problem_eval(models, p, args.split, args.device, ensemble=args.ensemble,
                              tta_8fold=args.tta_8fold)
        per_p[p] = r
        sbs = audit[p][args.split]["sbs_mean"]
        vs_sbs = (r["mean_cost"] - sbs) / (abs(sbs) + 1e-9) * 100
        print(f"  {p:>10}: top1={r['top1']:.3f} top2={r['top2']:.3f} top3={r['top3']:.3f} mc={r['mean_cost']:.4f} vs_sbs={vs_sbs:+.3f}%")

    summary = build_tables(per_p, audit, args.split, out)
    print(f"\n[MACRO] top1={summary['macro_top1']:.4f} top2={summary['macro_top2']:.4f} top3={summary['macro_top3']:.4f} vs_sbs={summary['macro_vs_sbs_pct']:+.4f}%")
    print(f"[done] tables written to {out}/metrics_report.md and {out}/summary.json")


if __name__ == "__main__":
    main()
