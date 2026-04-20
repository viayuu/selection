"""Analyze a trained selector checkpoint — produces ALL metrics from 观测指标/观测指标.md:

For each of the 18 problems (on val + test):
    1) selector top1/2/3 accuracy; best top1/2/3 (= oracle / SBS)
    2) selector mean_cost; best mean_cost (= SBS mean cost on that split)
    3) per-method top1 win rate & mean_cost; lists of methods selector beats / loses
       on (a) top1 and (b) mean_cost
    4) 3 figures per split:
        - top1_vs_single_methods.png  (per-problem stacked bar: selector vs each method)
        - mean_cost_vs_single_methods.png
        - arm_distribution.png  (per-problem histogram of selector's picks)

Optional: apply SBS gate (gate.py-produced per-problem gamma calibration).

Usage:
    python -m code.unified_selector.analyze --ckpt runs/R3/best.pt --split val \
            --out runs/R3/analysis/ [--gate runs/R3/gate/gate.json]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .registry import PROBLEMS, P2I, problem_to_pool_mask
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector


@torch.no_grad()
def evaluate_problem(model, problem: str, split: str, device, audit: dict,
                     gate_gamma: Optional[float] = None) -> dict:
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    collate_fn=collate_single_problem)
    pool_names = audit[problem]["pool"]
    pool_ids = audit[problem]["pool_global_ids"]
    sbs_idx = audit[problem]["sbs_pool_idx"]
    K_p = len(pool_names)

    all_costs = []
    all_pred = []        # selector's chosen index within pool
    all_log_p = []
    for b in dl:
        for k, v in b.items():
            if torch.is_tensor(v):
                b[k] = v.to(device)
        logits = model(b)                                          # (B, M_global)
        log_p = F.log_softmax(logits, dim=1)
        log_p_pool = log_p[:, b["pool_ids"]]                       # (B, K_p)
        if gate_gamma is not None:
            # SBS gate: if log_p_pool[:, sbs_idx] > -gamma (i.e. p >= exp(-gamma)) -> force pick SBS
            max_lp = log_p_pool.max(dim=1).values
            # Alternative: margin-based gate — if best beats SBS by <gamma (log-prob), pick SBS.
            best_idx = log_p_pool.argmax(dim=1)
            sbs_lp = log_p_pool[:, sbs_idx]
            best_lp = log_p_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1)
            low_conf = (best_lp - sbs_lp) < gate_gamma
            pred = torch.where(low_conf, torch.full_like(best_idx, sbs_idx), best_idx)
        else:
            pred = log_p_pool.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_log_p.append(log_p_pool.cpu().numpy())
        all_costs.append(b["costs"].cpu().numpy())

    pred = np.concatenate(all_pred)
    log_p = np.concatenate(all_log_p)                              # (N, K_p)
    costs = np.concatenate(all_costs)                              # (N, K_p)
    N = costs.shape[0]

    # Oracle ranking (best first)
    true_rank = np.argsort(costs, axis=1)
    best_idx = true_rank[:, 0]
    # Per-instance rank of the SELECTOR'S PICK within the oracle ordering (0 = best).
    # This matches the 观测指标/观测指标.md definition of top-k accuracy.
    inv_rank = np.argsort(true_rank, axis=1)                       # (N, K_p)
    pick_rank = inv_rank[np.arange(N), pred]                       # (N,)
    top1 = float((pick_rank == 0).mean())
    top2 = float((pick_rank < min(2, K_p)).mean()) if K_p >= 2 else top1
    top3 = float((pick_rank < min(3, K_p)).mean()) if K_p >= 3 else top2

    sel_costs = costs[np.arange(N), pred]
    mean_cost = float(sel_costs.mean())

    # Per-method top1 winrate (= fraction of instances where method m is argmin) and mean_cost
    method_top1 = {}
    method_mean = {}
    for k, name in enumerate(pool_names):
        method_top1[name] = float((best_idx == k).mean())
        method_mean[name] = float(costs[:, k].mean())

    # Compare selector against each method on top1 (winrate) and mean cost
    beat_top1 = [n for n in pool_names if top1 > method_top1[n] + 1e-6]
    lose_top1 = [n for n in pool_names if top1 < method_top1[n] - 1e-6]
    beat_mc   = [n for n in pool_names if mean_cost < method_mean[n] - 1e-6]
    lose_mc   = [n for n in pool_names if mean_cost > method_mean[n] + 1e-6]

    # Arm distribution: selector picks per method (frequency)
    arm_dist = {n: float((pred == k).mean()) for k, n in enumerate(pool_names)}

    sbs_name = pool_names[sbs_idx]
    sbs_cost = method_mean[sbs_name]
    vbs_mean = float(costs.min(axis=1).mean())

    return dict(
        problem=problem, split=split, N=N, K_p=K_p, pool_names=pool_names,
        sbs_name=sbs_name, sbs_cost=sbs_cost, vbs_mean=vbs_mean,
        top1=top1, top2=top2, top3=top3,
        mean_cost=mean_cost,
        vs_sbs_pct=(mean_cost - sbs_cost) / (abs(sbs_cost) + 1e-9) * 100,
        vbs_gap_closed_pct=((sbs_cost - mean_cost) / (sbs_cost - vbs_mean + 1e-9)) * 100 if sbs_cost != vbs_mean else 0.0,
        method_top1=method_top1, method_mean=method_mean,
        beat_top1=beat_top1, lose_top1=lose_top1,
        beat_mean_cost=beat_mc, lose_mean_cost=lose_mc,
        arm_distribution=arm_dist,
    )


def fig_top1_vs_methods(per_p: Dict[str, dict], out: Path):
    problems = [p for p in PROBLEMS if p in per_p]
    P = len(problems)
    fig, ax = plt.subplots(figsize=(max(14, P * 0.9), 6))
    all_methods = sorted({m for p in problems for m in per_p[p]["pool_names"]})
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(all_methods)))
    mcolor = dict(zip(all_methods, palette))
    width = 0.85 / (len(all_methods) + 1)
    for i, m in enumerate(all_methods):
        xs, ys = [], []
        for j, p in enumerate(problems):
            if m in per_p[p]["method_top1"]:
                xs.append(j + (i - len(all_methods) / 2) * width)
                ys.append(per_p[p]["method_top1"][m])
        ax.bar(xs, ys, width=width, color=mcolor[m], label=m, alpha=0.8)
    # Plot selector on top as black crosses
    sel_xs = np.arange(P) + (len(all_methods) / 2 + 0.5) * width
    sel_ys = [per_p[p]["top1"] for p in problems]
    ax.bar(sel_xs, sel_ys, width=width, color="black", edgecolor="red", label="selector", linewidth=1.5)
    ax.set_xticks(np.arange(P))
    ax.set_xticklabels(problems, rotation=45, ha="right")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Selector vs each single method — top-1 accuracy per problem")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_mean_cost_vs_methods(per_p: Dict[str, dict], out: Path):
    problems = [p for p in PROBLEMS if p in per_p]
    P = len(problems)
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    axes = axes.flatten()
    for j, p in enumerate(problems):
        ax = axes[j]
        methods = per_p[p]["pool_names"]
        vals = [per_p[p]["method_mean"][m] for m in methods]
        colors = ["tab:blue"] * len(methods)
        sbs = per_p[p]["sbs_name"]
        if sbs in methods:
            colors[methods.index(sbs)] = "tab:green"
        xs = np.arange(len(methods))
        ax.bar(xs, vals, color=colors, alpha=0.75)
        sel_val = per_p[p]["mean_cost"]
        ax.axhline(sel_val, color="red", linewidth=2, label=f"selector={sel_val:.3f}")
        vbs = per_p[p]["vbs_mean"]
        ax.axhline(vbs, color="black", linestyle="--", linewidth=1, label=f"VBS={vbs:.3f}")
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{p} (SBS={sbs})", fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, axis="y", alpha=0.3)
    for k in range(len(problems), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Mean cost — selector vs each single method (SBS in green, VBS dashed, selector red)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def fig_arm_distribution(per_p: Dict[str, dict], out: Path):
    problems = [p for p in PROBLEMS if p in per_p]
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    axes = axes.flatten()
    for j, p in enumerate(problems):
        ax = axes[j]
        methods = per_p[p]["pool_names"]
        dist = per_p[p]["arm_distribution"]
        xs = np.arange(len(methods))
        vals = [dist[m] for m in methods]
        colors = ["tab:orange"] * len(methods)
        sbs = per_p[p]["sbs_name"]
        if sbs in methods:
            colors[methods.index(sbs)] = "tab:green"
        ax.bar(xs, vals, color=colors, alpha=0.75)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{p} top1={per_p[p]['top1']:.2f}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    for k in range(len(problems), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Selector's arm-pick distribution per problem (SBS in green)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def fig_loss_curve(train_log_path: Path, out: Path):
    """Parse train.log and plot loss EMA vs step."""
    if not train_log_path.exists():
        return
    steps, losses = [], []
    for ln in train_log_path.read_text().splitlines():
        if not ln.startswith("ep") or "loss=" not in ln or "ema=" not in ln:
            continue
        try:
            step = int(ln.split("step")[1].split()[0])
            ema = float(ln.split("ema=")[1].split()[0])
            steps.append(step); losses.append(ema)
        except Exception:
            pass
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="tab:blue", linewidth=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("loss EMA")
    ax.set_title(f"Training loss EMA — {train_log_path.parent.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--gate", default=None, help="Path to gate/gate.json with per-problem gamma")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(args.audit).read_text())

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model_args = ckpt.get("args", {})
    has_bias = "problem_solver_bias" in ckpt["model"]
    model = UnifiedSelector(
        d=model_args.get("d", 128), depth=model_args.get("depth", 4),
        dropout=model_args.get("dropout", 0.1),
        use_mvrp_factorized=not model_args.get("no_fact", False),
        use_problem_solver_bias=has_bias,
    ).to(args.device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[load] missing={missing} unexpected={unexpected}")
    model.eval()

    gate_cfg = None
    if args.gate:
        gate_cfg = json.loads(Path(args.gate).read_text())

    per_p: Dict[str, dict] = {}
    for p in PROBLEMS:
        g = gate_cfg.get(p, {}).get("gamma") if gate_cfg else None
        per_p[p] = evaluate_problem(model, p, args.split, args.device, audit, gate_gamma=g)
        r = per_p[p]
        print(f"  {p:>10}: top1={r['top1']:.3f} mc={r['mean_cost']:.4f} (sbs={r['sbs_cost']:.4f}) "
              f"vs_sbs={r['vs_sbs_pct']:+.2f}% beats_top1={len(r['beat_top1'])}/{r['K_p']} "
              f"beats_mc={len(r['beat_mean_cost'])}/{r['K_p']}")
    macro = dict(
        macro_top1=float(np.mean([r["top1"] for r in per_p.values()])),
        macro_top2=float(np.mean([r["top2"] for r in per_p.values()])),
        macro_top3=float(np.mean([r["top3"] for r in per_p.values()])),
        macro_vs_sbs_pct=float(np.mean([r["vs_sbs_pct"] for r in per_p.values()])),
        macro_vbs_gap_closed_pct=float(np.mean([r["vbs_gap_closed_pct"] for r in per_p.values()])),
        n_problems_beat_sbs=int(sum(1 for r in per_p.values() if r["vs_sbs_pct"] < 0)),
        n_problems_match_sbs=int(sum(1 for r in per_p.values() if abs(r["vs_sbs_pct"]) < 0.02)),
    )
    (out / f"analysis_{args.split}.json").write_text(
        json.dumps({"per_problem": per_p, "macro": macro}, indent=2))
    print(f"\nMacro [{args.split}]: top1={macro['macro_top1']:.4f} top2={macro['macro_top2']:.4f} "
          f"top3={macro['macro_top3']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% "
          f"vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}% "
          f"beat={macro['n_problems_beat_sbs']}/18 match={macro['n_problems_match_sbs']}/18")

    # Figures
    fig_top1_vs_methods(per_p, out / f"top1_vs_single_methods_{args.split}.png")
    fig_mean_cost_vs_methods(per_p, out / f"mean_cost_vs_single_methods_{args.split}.png")
    fig_arm_distribution(per_p, out / f"arm_distribution_{args.split}.png")
    # Loss curve if train.log lives next to ckpt
    tl = Path(args.ckpt).parent / "train.log"
    fig_loss_curve(tl, out / "loss_curve.png")
    print(f"Figures + analysis JSON saved to {out}")


if __name__ == "__main__":
    main()
