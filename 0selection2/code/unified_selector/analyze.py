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

from .registry import PROBLEMS, P2I, POOLS, problem_to_pool_mask
from .data import UnifiedProblemDataset, collate_single_problem, augment_xy_by_8_fold
from .model import UnifiedSelector, remap_legacy_state_dict, shortlist_from_support


def apply_meta_only(batch, meta_only: bool):
    if not meta_only:
        return batch
    if batch["kind"] == "coord":
        batch["node"] = torch.zeros_like(batch["node"])
    elif batch["kind"] == "matrix":
        batch["matrix"] = torch.zeros_like(batch["matrix"])
    return batch


@torch.no_grad()
def evaluate_problem(model, problem: str, split: str, device, audit: dict,
                     gate_gamma: Optional[float] = None,
                     meta_only: bool = False,
                     tta: int = 1,
                     use_support_head: bool = False,
                     support_threshold: float = 0.5,
                     support_topk: int = 3) -> dict:
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    collate_fn=collate_single_problem)
    pool_names = list(POOLS[problem])
    K_p = ds.K_p
    if len(pool_names) != K_p:
        pool_names = pool_names[:K_p]

    # The legacy audit file may belong to an older label space.  For analysis on
    # the current dataset, derive SBS from the current split labels instead of
    # trusting historical pool metadata.
    gate_sbs_idx = None
    if gate_gamma is not None:
        ref_costs = []
        for i in range(ds.base_N):
            ref_costs.append(np.asarray(ds.labels[str(i)]["cost"][:K_p], dtype=np.float32))
        if ref_costs:
            ref_costs = np.stack(ref_costs, axis=0)
            gate_sbs_idx = int(ref_costs.mean(axis=0).argmin())

    all_costs = []
    all_pred = []        # selector's chosen index within pool
    all_score = []
    all_shortlist = []
    for b in dl:
        for k, v in b.items():
            if torch.is_tensor(v):
                b[k] = v.to(device)
        b = apply_meta_only(b, meta_only)
        if tta > 1 and b["kind"] == "coord":
            logits = None
            support_logits = None
            n_tta = min(max(1, tta), 8)
            for aug_idx in range(n_tta):
                aug_b = dict(b)
                aug_node = b["node"].clone()
                aug_node[:, :, 0:2] = augment_xy_by_8_fold(aug_node[:, :, 0:2], aug_idx)
                aug_b["node"] = aug_node
                if use_support_head:
                    cur, cur_support = model(aug_b, return_support=True)
                else:
                    cur = model(aug_b)
                    cur_support = None
                logits = cur if logits is None else (logits + cur)
                if cur_support is not None:
                    support_logits = cur_support if support_logits is None else (support_logits + cur_support)
            logits = logits / float(n_tta)
            if support_logits is not None:
                support_logits = support_logits / float(n_tta)
        else:
            if use_support_head:
                logits, support_logits = model(b, return_support=True)
            else:
                logits = model(b)                                      # (B, M_global)
                support_logits = None
        logits_pool = logits[:, b["pool_ids"]]                         # (B, K_p)
        if support_logits is not None:
            support_pool = support_logits[:, b["pool_ids"]] if support_logits.dim() == 2 else support_logits[:, :, b["pool_ids"]]
        else:
            support_pool = None
        effective_logits_pool, shortlist, _ = shortlist_from_support(
            logits_pool,
            support_pool,
            threshold=support_threshold,
            topk=support_topk,
        )
        log_p_pool = F.log_softmax(effective_logits_pool, dim=1)
        if gate_gamma is not None and gate_sbs_idx is not None:
            # Alternative: margin-based gate — if best beats SBS by <gamma (log-prob), pick SBS.
            best_idx = log_p_pool.argmax(dim=1)
            sbs_lp = log_p_pool[:, gate_sbs_idx]
            best_lp = log_p_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1)
            low_conf = (best_lp - sbs_lp) < gate_gamma
            pred = torch.where(low_conf, torch.full_like(best_idx, gate_sbs_idx), best_idx)
        else:
            pred = log_p_pool.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_score.append(effective_logits_pool.cpu().numpy())
        if shortlist is not None:
            all_shortlist.append(shortlist.cpu().numpy())
        all_costs.append(b["costs"].cpu().numpy())

    pred = np.concatenate(all_pred)
    score = np.concatenate(all_score)                              # (N, K_p)
    costs = np.concatenate(all_costs)                              # (N, K_p)
    shortlist = np.concatenate(all_shortlist) if all_shortlist else None
    N = costs.shape[0]

    # Oracle ranking (best first)
    true_rank = np.argsort(costs, axis=1)
    best_idx = true_rank[:, 0]
    top1 = float((pred == best_idx).mean())
    pred_rank = np.argsort(-score, axis=1)
    top2 = float(np.any(pred[:, None] == true_rank[:, :min(2, K_p)], axis=1).mean())
    top3 = float(np.any(pred[:, None] == true_rank[:, :min(3, K_p)], axis=1).mean())

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
    final_arm_coverage = float(np.mean([v > 0 for v in arm_dist.values()]))
    pick_entropy = float(-(np.array(list(arm_dist.values())) * np.log(np.array(list(arm_dist.values())) + 1e-12)).sum() / np.log(max(2, K_p)))
    zero_pick_count = int(sum(v <= 0 for v in arm_dist.values()))
    hidden_winners = [n for n in pool_names if method_top1[n] > 0 and arm_dist[n] <= 0]
    hidden_winner_count = len(hidden_winners)
    hidden_winner_mass = float(sum(method_top1[n] for n in hidden_winners))
    oracle_support_mass_recall = float(1.0 - hidden_winner_mass)

    if shortlist is not None:
        support_top1_recall = float(shortlist[np.arange(N), best_idx].mean())
        top3_idx = true_rank[:, :min(3, K_p)]
        support_top3_recall = float(np.take_along_axis(shortlist.astype(np.float32), top3_idx, axis=1).max(axis=1).mean())
        support_dist = {n: float(shortlist[:, k].mean()) for k, n in enumerate(pool_names)}
        support_arm_coverage = float(np.mean([v > 0 for v in support_dist.values()]))
        support_entropy = float(-(np.array(list(support_dist.values())) * np.log(np.array(list(support_dist.values())) + 1e-12)).sum() / np.log(max(2, K_p)))
        support_hidden_winners = [n for n in pool_names if method_top1[n] > 0 and support_dist[n] <= 0]
        support_hidden_winner_count = len(support_hidden_winners)
        support_hidden_winner_mass = float(sum(method_top1[n] for n in support_hidden_winners))
        support_oracle_mass_recall = float(1.0 - support_hidden_winner_mass)
    else:
        support_top1_recall = None
        support_top3_recall = None
        support_dist = None
        support_arm_coverage = None
        support_entropy = None
        support_hidden_winner_count = None
        support_hidden_winner_mass = None
        support_oracle_mass_recall = None

    sbs_idx = int(np.argmin([method_mean[n] for n in pool_names]))
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
        final_arm_coverage=final_arm_coverage,
        pick_entropy=pick_entropy,
        zero_pick_count=zero_pick_count,
        hidden_winners=hidden_winners,
        hidden_winner_count=hidden_winner_count,
        hidden_winner_mass=hidden_winner_mass,
        oracle_support_mass_recall=oracle_support_mass_recall,
        support_distribution=support_dist,
        support_top1_recall=support_top1_recall,
        support_top3_recall=support_top3_recall,
        support_arm_coverage=support_arm_coverage,
        support_entropy=support_entropy,
        support_hidden_winner_count=support_hidden_winner_count,
        support_hidden_winner_mass=support_hidden_winner_mass,
        support_oracle_mass_recall=support_oracle_mass_recall,
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
    ap.add_argument("--split", choices=["val", "test", "TSPLIB", "CVRPLIB"], default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--gate", default=None, help="Path to gate/gate.json with per-problem gamma")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tta", type=int, default=1, help="Average logits over up to 8 coord augmentations at eval time.")
    ap.add_argument("--problems", nargs="*", default=None,
                    help="Optional subset of problems to analyze, e.g. --problems TSP or --problems CVRP.")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(args.audit).read_text())

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model_args = ckpt.get("args", {})
    has_bias = "problem_solver_bias" in ckpt["model"]
    meta_only = bool(model_args.get("meta_only", False))
    model = UnifiedSelector(
        d=model_args.get("d", 128), depth=model_args.get("depth", 4),
        dropout=model_args.get("dropout", 0.1),
        use_mvrp_factorized=not model_args.get("no_fact", False),
        use_problem_solver_bias=has_bias,
        use_problem_film=bool(model_args.get("problem_film", False)),
        use_size_feature=bool(model_args.get("size_feature", False)),
        rich_pool=bool(model_args.get("rich_pool", False)),
        use_global_stats=bool(model_args.get("global_stats", False)),
        use_manual_features=bool(model_args.get("manual_features", False)),
        use_constraint_experts=bool(model_args.get("constraint_experts", False)),
        use_problem_residual_head=bool(model_args.get("problem_residual_head", False)),
        use_problem_adapter=bool(model_args.get("problem_adapter", False)),
        use_support_head=bool(model_args.get("support_head", False)),
        support_hidden=int(model_args.get("support_hidden", 128)),
        support_generators=int(model_args.get("support_generators", 1)),
        adapter_hidden=int(model_args.get("adapter_hidden", 64)),
        coord_hier_pool=bool(model_args.get("coord_hier_pool", False)),
        coord_downsample_ratio=float(model_args.get("coord_downsample_ratio", 0.8)),
        deep_encoder_overhaul=bool(model_args.get("deep_encoder_overhaul", False)),
        encoder_rezero=bool(model_args.get("encoder_rezero", False)),
        encoder_constraint_experts=bool(model_args.get("encoder_constraint_experts", False)),
        encoder_constraint_hidden=int(model_args.get("encoder_constraint_hidden", 128)),
    ).to(args.device)
    missing, unexpected = model.load_state_dict(remap_legacy_state_dict(ckpt["model"]), strict=False)
    if missing or unexpected:
        print(f"[load] missing={missing} unexpected={unexpected}")
    model.eval()

    gate_cfg = None
    if args.gate:
        gate_cfg = json.loads(Path(args.gate).read_text())
        if "gammas" in gate_cfg:
            gate_cfg = gate_cfg["gammas"]

    if args.problems:
        problems = args.problems
    elif args.split == "TSPLIB":
        problems = ["TSP"]
    elif args.split == "CVRPLIB":
        problems = ["CVRP"]
    else:
        problems = PROBLEMS

    per_p: Dict[str, dict] = {}
    use_support_head = bool(model_args.get("support_head", False))
    support_threshold = float(model_args.get("support_threshold", 0.5))
    support_topk = int(model_args.get("support_topk", 3))
    for p in problems:
        g = gate_cfg.get(p, {}).get("gamma") if gate_cfg else None
        per_p[p] = evaluate_problem(
            model, p, args.split, args.device, audit,
            gate_gamma=g, meta_only=meta_only, tta=args.tta,
            use_support_head=use_support_head,
            support_threshold=support_threshold,
            support_topk=support_topk,
        )
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
        macro_final_arm_coverage=float(np.mean([r["final_arm_coverage"] for r in per_p.values()])),
        macro_pick_entropy=float(np.mean([r["pick_entropy"] for r in per_p.values()])),
        macro_zero_pick_count=float(np.mean([r["zero_pick_count"] for r in per_p.values()])),
        macro_hidden_winner_count=float(np.mean([r["hidden_winner_count"] for r in per_p.values()])),
        macro_hidden_winner_mass=float(np.mean([r["hidden_winner_mass"] for r in per_p.values()])),
        macro_oracle_support_mass_recall=float(np.mean([r["oracle_support_mass_recall"] for r in per_p.values()])),
    )
    if use_support_head:
        macro["macro_support_top1_recall"] = float(np.mean([r["support_top1_recall"] for r in per_p.values()]))
        macro["macro_support_top3_recall"] = float(np.mean([r["support_top3_recall"] for r in per_p.values()]))
        macro["macro_support_arm_coverage"] = float(np.mean([r["support_arm_coverage"] for r in per_p.values()]))
        macro["macro_support_entropy"] = float(np.mean([r["support_entropy"] for r in per_p.values()]))
        macro["macro_support_hidden_winner_count"] = float(np.mean([r["support_hidden_winner_count"] for r in per_p.values()]))
        macro["macro_support_hidden_winner_mass"] = float(np.mean([r["support_hidden_winner_mass"] for r in per_p.values()]))
        macro["macro_support_oracle_mass_recall"] = float(np.mean([r["support_oracle_mass_recall"] for r in per_p.values()]))
    (out / f"analysis_{args.split}.json").write_text(
        json.dumps({"per_problem": per_p, "macro": macro}, indent=2))
    print(f"\nMacro [{args.split}]: top1={macro['macro_top1']:.4f} top2={macro['macro_top2']:.4f} "
          f"top3={macro['macro_top3']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% "
          f"vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}% "
          f"beat={macro['n_problems_beat_sbs']}/{len(per_p)} match={macro['n_problems_match_sbs']}/{len(per_p)}")

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
