"""SBS-gating calibration + switch diagnostics for a trained selector.

Usage:
  python -m code.unified_selector.gate --ckpt <path_to_best.pt> --gate-grid 40 --device cuda:0
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, P2I
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector


@torch.no_grad()
def gather_all(model, problem, split, device, audit):
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=16, collate_fn=collate_single_problem, num_workers=0, shuffle=False)
    all_logits_pool = []
    all_costs = []
    sbs_idx = audit[problem]["sbs_pool_idx"]
    for b in dl:
        for k,v in b.items():
            if torch.is_tensor(v): b[k] = v.to(device)
        logits = model(b)  # (B, M)
        log_p = F.log_softmax(logits, dim=1)
        pool = b["pool_ids"]
        all_logits_pool.append(log_p[:, pool].cpu())
        all_costs.append(b["costs"].cpu())
    return torch.cat(all_logits_pool, dim=0), torch.cat(all_costs, dim=0), sbs_idx


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0):
    if len(values) == 0 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(values[idx].mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    se = float(np.std(means, ddof=1))
    return {"mean": float(np.mean(values)), "se": se, "ci95": [float(lo), float(hi)]}


def selector_scores(log_p_pool: torch.Tensor, sbs_idx: int, score_type: str, challenger_only: bool = False):
    if challenger_only:
        masked = log_p_pool.clone()
        masked[:, sbs_idx] = float("-inf")
        best_idx = masked.argmax(dim=1)
    else:
        best_idx = log_p_pool.argmax(dim=1)
    best_lp = log_p_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    sbs_lp = log_p_pool[:, sbs_idx]
    if score_type == "margin":
        score = best_lp - sbs_lp
    elif score_type == "prob_gap":
        probs = log_p_pool.exp()
        best_p = probs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        sbs_p = probs[:, sbs_idx]
        score = best_p - sbs_p
    elif score_type == "margin_over_entropy":
        probs = log_p_pool.exp()
        ent = -(probs * log_p_pool).sum(dim=1).clamp_min(1e-6)
        score = (best_lp - sbs_lp) / ent
    else:
        raise ValueError(f"unknown score_type={score_type}")
    return best_idx, score


def apply_gate(log_p_pool, costs, sbs_idx: int, gamma: float, score_type: str,
               enable_switch: bool = True, challenger_only: bool = False):
    best_idx, score = selector_scores(log_p_pool, sbs_idx, score_type, challenger_only=challenger_only)
    best_cost = costs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    sbs_cost = costs[:, sbs_idx]
    use_best = (score >= gamma) if enable_switch else torch.zeros_like(best_idx, dtype=torch.bool)
    sel_cost = torch.where(use_best, best_cost, sbs_cost)
    return {
        "best_idx": best_idx,
        "score": score,
        "best_cost": best_cost,
        "sbs_cost": sbs_cost,
        "use_best": use_best,
        "sel_cost": sel_cost,
    }


def calibrate_one(log_p_pool, costs, sbs_idx: int, score_type: str, gate_grid_n: int,
                  fallback_bootstrap: int = 0, fallback_se_mult: float = 1.0,
                  fallback_precision_floor: float | None = None,
                  challenger_only: bool = False):
    applied = apply_gate(
        log_p_pool, costs, sbs_idx, gamma=-1e9, score_type=score_type,
        enable_switch=True, challenger_only=challenger_only,
    )
    score = applied["score"].numpy()
    sbs_cost = applied["sbs_cost"].numpy()
    best_cost = applied["best_cost"].numpy()
    grid = np.quantile(score, np.linspace(0.0, 1.0, gate_grid_n))
    grid = np.unique(np.concatenate(([score.min() - 1e-6], grid, [score.max() + 1e-6])))
    best = None
    for g in grid:
        use_best = score >= float(g)
        sel_cost = np.where(use_best, best_cost, sbs_cost)
        mean_cost = float(sel_cost.mean())
        if best is None or mean_cost < best["mean_cost"]:
            best = {"gamma": float(g), "mean_cost": mean_cost, "switch_rate": float(use_best.mean())}
    sel_cost = np.where(score >= best["gamma"], best_cost, sbs_cost)
    use_best = score >= best["gamma"]
    improvement = sbs_cost - sel_cost
    boot = bootstrap_ci(improvement.astype(np.float64), n_boot=fallback_bootstrap, seed=0) if fallback_bootstrap > 0 else None
    good_switch = use_best & (best_cost < sbs_cost)
    bad_switch = use_best & (best_cost >= sbs_cost)
    switch_precision = float(good_switch.mean() / max(use_best.mean(), 1e-12)) if use_best.any() else 0.0
    benefit_rel = (((sbs_cost[good_switch] - best_cost[good_switch]) / (np.abs(sbs_cost[good_switch]) + 1e-9)) * 100.0) if good_switch.any() else np.array([], dtype=np.float64)
    harm_rel = (((best_cost[bad_switch] - sbs_cost[bad_switch]) / (np.abs(sbs_cost[bad_switch]) + 1e-9)) * 100.0) if bad_switch.any() else np.array([], dtype=np.float64)
    enable_switch = True
    if boot is not None and boot["mean"] <= fallback_se_mult * boot["se"]:
        enable_switch = False
    if fallback_precision_floor is not None and switch_precision < fallback_precision_floor:
        enable_switch = False
    return {
        "score_type": score_type,
        "gamma": best["gamma"],
        "train_calib_mean_cost": best["mean_cost"],
        "train_calib_switch_rate": best["switch_rate"],
        "enable_switch": enable_switch,
        "calib_improvement_mean": float(improvement.mean()),
        "calib_improvement_bootstrap": boot,
        "calib_switch_precision": switch_precision,
        "calib_avg_rel_benefit_if_correct_switch_pct": float(benefit_rel.mean()) if benefit_rel.size else 0.0,
        "calib_avg_rel_harm_if_wrong_switch_pct": float(harm_rel.mean()) if harm_rel.size else 0.0,
    }


def calibrate_gating(model, device, audit, calib_split: str = "train", calib_fraction: float = 0.2,
                     gate_grid_n: int = 40, score_type: str = "margin",
                     fallback_bootstrap: int = 0, fallback_se_mult: float = 1.0,
                     fallback_precision_floor: float | None = None,
                     challenger_only: bool = False):
    """Calibrate a conservative SBS-switch policy from an existing checkpoint."""
    gammas = {}
    score_types = ["margin", "prob_gap", "margin_over_entropy"] if score_type == "auto" else [score_type]
    for p in PROBLEMS:
        log_p_pool, costs, sbs_idx = gather_all(model, p, calib_split, device, audit)
        n = min(max(1, int(calib_fraction * len(costs))), len(costs))
        if n < len(costs):
            gen = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(costs), generator=gen)[:n]
            log_p_pool = log_p_pool[idx]
            costs = costs[idx]
        best_cfg = None
        for st in score_types:
            cfg = calibrate_one(log_p_pool, costs, sbs_idx, st, gate_grid_n,
                                fallback_bootstrap=fallback_bootstrap,
                                fallback_se_mult=fallback_se_mult,
                                fallback_precision_floor=fallback_precision_floor,
                                challenger_only=challenger_only)
            if best_cfg is None or cfg["train_calib_mean_cost"] < best_cfg["train_calib_mean_cost"]:
                best_cfg = cfg
        gammas[p] = best_cfg
    return gammas


def evaluate_gated(model, device, audit, gammas, eval_split: str = "val", bootstrap: int = 0,
                   challenger_only: bool = False):
    per_p = {}
    raw_eval = {}
    for p in PROBLEMS:
        log_p_pool, costs, sbs_idx = gather_all(model, p, eval_split, device, audit)
        cfg = gammas[p]
        applied = apply_gate(
            log_p_pool, costs, sbs_idx, cfg["gamma"], cfg["score_type"],
            enable_switch=cfg["enable_switch"], challenger_only=challenger_only,
        )
        best_cost = applied["best_cost"]
        sbs_cost = applied["sbs_cost"]
        use_best = applied["use_best"]
        sel_cost = applied["sel_cost"]
        switch_rate = float(use_best.float().mean())
        switch_precision = float((best_cost[use_best] < sbs_cost[use_best]).float().mean()) if use_best.any() else 0.0
        vbs = costs.min(dim=1).values
        vbs_beats = (sbs_cost - vbs) / (sbs_cost.abs() + 1e-9) > 0.002
        switch_recall = float(use_best[vbs_beats].float().mean()) if vbs_beats.any() else 0.0
        good_switch = use_best & (best_cost < sbs_cost)
        bad_switch = use_best & (best_cost >= sbs_cost)
        benefit = float((sbs_cost[good_switch] - best_cost[good_switch]).mean()) if good_switch.any() else 0.0
        harm = float((best_cost[bad_switch] - sbs_cost[bad_switch]).mean()) if bad_switch.any() else 0.0
        benefit_rel = float((((sbs_cost[good_switch] - best_cost[good_switch]) / (sbs_cost[good_switch].abs() + 1e-9)) * 100.0).mean()) if good_switch.any() else 0.0
        harm_rel = float((((best_cost[bad_switch] - sbs_cost[bad_switch]) / (sbs_cost[bad_switch].abs() + 1e-9)) * 100.0).mean()) if bad_switch.any() else 0.0

        split_ref = audit[p][eval_split]
        sbs_mean = split_ref["sbs_mean"]
        vbs_mean = split_ref["vbs_mean"]
        delta = ((sel_cost.numpy() - sbs_cost.numpy()) / (np.abs(sbs_cost.numpy()) + 1e-9)) * 100.0
        delta_boot = bootstrap_ci(delta.astype(np.float64), n_boot=bootstrap, seed=123) if bootstrap > 0 else None
        per_p[p] = {
            "mean_cost_gated": float(sel_cost.mean()),
            "mean_cost_ungated": float(best_cost.mean()),
            "mean_cost_sbs": float(sbs_cost.mean()),
            "gamma": cfg["gamma"],
            "score_type": cfg["score_type"],
            "enable_switch": cfg["enable_switch"],
            "switch_rate": switch_rate,
            "switch_precision": switch_precision,
            "switch_recall": switch_recall,
            "avg_benefit_if_correct_switch": benefit,
            "avg_harm_if_wrong_switch": harm,
            "avg_rel_benefit_if_correct_switch_pct": benefit_rel,
            "avg_rel_harm_if_wrong_switch_pct": harm_rel,
            "vs_sbs_pct_gated": (sel_cost.mean().item() - sbs_mean)/(abs(sbs_mean)+1e-9)*100,
            "vs_sbs_pct_ungated": (best_cost.mean().item() - sbs_mean)/(abs(sbs_mean)+1e-9)*100,
            "vbs_gap_closed_pct": ((sbs_mean - sel_cost.mean().item())/(sbs_mean - vbs_mean + 1e-9))*100,
            "vs_sbs_pct_gated_bootstrap": delta_boot,
        }
        raw_eval[p] = {"sel_cost": sel_cost.numpy(), "sbs_cost": sbs_cost.numpy()}
    macro_vs_sbs_gated = sum(v["vs_sbs_pct_gated"] for v in per_p.values())/len(per_p)
    macro_vs_sbs_ungated = sum(v["vs_sbs_pct_ungated"] for v in per_p.values())/len(per_p)
    macro = {"macro_vs_sbs_gated": macro_vs_sbs_gated, "macro_vs_sbs_ungated": macro_vs_sbs_ungated}
    if bootstrap > 0:
        rng = np.random.default_rng(999)
        macro_boot = []
        for _ in range(bootstrap):
            vals = []
            for p in PROBLEMS:
                arr = raw_eval[p]
                n = len(arr["sel_cost"])
                idx = rng.integers(0, n, size=n)
                sel_m = float(arr["sel_cost"][idx].mean())
                sbs_m = float(arr["sbs_cost"][idx].mean())
                vals.append((sel_m - sbs_m) / (abs(sbs_m) + 1e-9) * 100.0)
            macro_boot.append(float(np.mean(vals)))
        lo, hi = np.percentile(macro_boot, [2.5, 97.5])
        macro["macro_vs_sbs_gated_bootstrap"] = {
            "mean": macro_vs_sbs_gated,
            "ci95": [float(lo), float(hi)],
            "se": float(np.std(macro_boot, ddof=1)),
        }
    return per_p, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-grid", type=int, default=40)
    ap.add_argument("--calib-split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--calib-fraction", type=float, default=0.2)
    ap.add_argument("--score", choices=["margin", "prob_gap", "margin_over_entropy", "auto"], default="margin")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--fallback-bootstrap", type=int, default=0)
    ap.add_argument("--fallback-se-mult", type=float, default=1.0)
    ap.add_argument("--fallback-precision-floor", type=float, default=None)
    ap.add_argument("--challenger-only", action="store_true",
                    help="Gate the best non-SBS challenger against SBS instead of the overall argmax.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    has_bias = "problem_solver_bias" in ckpt["model"]
    model = UnifiedSelector(d=cfg.get("d",128), depth=cfg.get("depth",4),
                            dropout=cfg.get("dropout",0.1),
                            use_mvrp_factorized=not cfg.get("no_fact", False),
                            use_problem_solver_bias=has_bias,
                            use_problem_film=cfg.get("problem_film", False),
                            use_size_feature=cfg.get("size_feature", False),
                            rich_pool=cfg.get("rich_pool", False),
                            use_global_stats=cfg.get("global_stats", False),
                            use_manual_features=cfg.get("manual_features", False),
                            use_constraint_experts=cfg.get("constraint_experts", False),
                            use_problem_residual_head=cfg.get("problem_residual_head", False),
                            use_problem_adapter=cfg.get("problem_adapter", False),
                            adapter_hidden=cfg.get("adapter_hidden", 64),
                            coord_hier_pool=cfg.get("coord_hier_pool", False),
                            coord_downsample_ratio=cfg.get("coord_downsample_ratio", 0.8)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print(f"[gate] calibrating gammas on {args.calib_split} subset...")
    gammas = calibrate_gating(
        model, device, audit,
        calib_split=args.calib_split,
        calib_fraction=args.calib_fraction,
        gate_grid_n=args.gate_grid,
        score_type=args.score,
        fallback_bootstrap=args.fallback_bootstrap,
        fallback_se_mult=args.fallback_se_mult,
        fallback_precision_floor=args.fallback_precision_floor,
        challenger_only=args.challenger_only,
    )
    print(f"[gate] evaluating gated selector on {args.eval_split}...")
    per_p, macro = evaluate_gated(
        model, device, audit, gammas,
        eval_split=args.eval_split, bootstrap=args.bootstrap,
        challenger_only=args.challenger_only,
    )
    print(f"[gate] macro_vs_sbs UNGATED={macro['macro_vs_sbs_ungated']:+.3f}%   GATED={macro['macro_vs_sbs_gated']:+.3f}%")
    if "macro_vs_sbs_gated_bootstrap" in macro:
        ci = macro["macro_vs_sbs_gated_bootstrap"]["ci95"]
        print(f"[gate] macro_vs_sbs gated 95% CI = [{ci[0]:+.3f}%, {ci[1]:+.3f}%]")
    for p in PROBLEMS:
        r = per_p[p]
        print(f"  {p:>10}: score={r['score_type']} gamma={r['gamma']:+.3f} enable={int(r['enable_switch'])} switch={r['switch_rate']:.2f} "
              f"prec={r['switch_precision']:.2f} rec={r['switch_recall']:.2f} "
              f"vs_sbs gated={r['vs_sbs_pct_gated']:+.3f}% (ungated={r['vs_sbs_pct_ungated']:+.3f}%)")
    out = args.out or (Path(args.ckpt).parent / "gate_eval.json")
    Path(out).write_text(json.dumps({
        "config": {
            "calib_split": args.calib_split,
            "eval_split": args.eval_split,
            "calib_fraction": args.calib_fraction,
            "score": args.score,
            "gate_grid": args.gate_grid,
            "bootstrap": args.bootstrap,
            "fallback_bootstrap": args.fallback_bootstrap,
            "fallback_se_mult": args.fallback_se_mult,
            "challenger_only": args.challenger_only,
        },
        "gammas": gammas,
        "per_problem": per_p,
        "macro": macro,
    }, indent=2))
    print(f"[gate] saved {out}")

if __name__ == "__main__":
    main()
