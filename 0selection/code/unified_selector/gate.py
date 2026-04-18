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


def calibrate_gating(model, device, audit, train_fraction_for_gate: float = 0.2, gate_grid_n: int = 40):
    """For each problem, find per-problem gating margin gamma that minimizes train mean cost
    when using `argmax if margin > gamma else SBS`. Return dict problem -> gamma."""
    gammas = {}
    # Use a subset of train for calibration; eval on val.
    for p in PROBLEMS:
        log_p_pool, costs, sbs_idx = gather_all(model, p, "train", device, audit)
        n = min(int(train_fraction_for_gate * len(costs)), 2000)
        log_p_pool = log_p_pool[:n]
        costs = costs[:n]
        best_idx = log_p_pool.argmax(dim=1)
        best_cost = costs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        sbs_cost = costs[:, sbs_idx]
        # margin on pool log-probs (argmax vs SBS)
        margin = log_p_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1) - log_p_pool[:, sbs_idx]
        quantiles = np.linspace(-0.01, 1.0, gate_grid_n)
        grid = np.quantile(margin.numpy(), np.clip(quantiles, 0.0, 1.0))
        best_cost_mean = np.inf; best_gamma = 0.0
        for g in grid:
            pred_cost = torch.where(margin >= float(g), best_cost, sbs_cost).mean().item()
            if pred_cost < best_cost_mean:
                best_cost_mean = pred_cost
                best_gamma = float(g)
        gammas[p] = {"gamma": best_gamma, "train_calib_mean_cost": best_cost_mean}
    return gammas


def evaluate_gated(model, device, audit, gammas):
    per_p = {}
    for p in PROBLEMS:
        log_p_pool, costs, sbs_idx = gather_all(model, p, "val", device, audit)
        best_idx = log_p_pool.argmax(dim=1)
        best_cost = costs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        sbs_cost = costs[:, sbs_idx]
        margin = log_p_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1) - log_p_pool[:, sbs_idx]
        g = gammas[p]["gamma"]
        # Gated prediction
        use_best = (margin >= g)
        sel_cost = torch.where(use_best, best_cost, sbs_cost)
        # Diagnostics
        switch_rate = float(use_best.float().mean())
        switch_precision = float((best_cost[use_best] < sbs_cost[use_best]).float().mean()) if use_best.any() else 0.0
        vbs = costs.min(dim=1).values
        # Recall: of instances where vbs<sbs by >0.2%, how many did we switch?
        vbs_beats = (sbs_cost - vbs) / (sbs_cost.abs() + 1e-9) > 0.002
        switch_recall = float(use_best[vbs_beats].float().mean()) if vbs_beats.any() else 0.0

        sbs_mean = audit[p]["val"]["sbs_mean"]; vbs_mean = audit[p]["val"]["vbs_mean"]
        per_p[p] = {
            "mean_cost_gated": float(sel_cost.mean()),
            "mean_cost_ungated": float(best_cost.mean()),
            "mean_cost_sbs": float(sbs_cost.mean()),
            "gamma": g,
            "switch_rate": switch_rate,
            "switch_precision": switch_precision,
            "switch_recall": switch_recall,
            "vs_sbs_pct_gated": (sel_cost.mean().item() - sbs_mean)/(abs(sbs_mean)+1e-9)*100,
            "vs_sbs_pct_ungated": (best_cost.mean().item() - sbs_mean)/(abs(sbs_mean)+1e-9)*100,
            "vbs_gap_closed_pct": ((sbs_mean - sel_cost.mean().item())/(sbs_mean - vbs_mean + 1e-9))*100,
        }
    macro_vs_sbs_gated = sum(v["vs_sbs_pct_gated"] for v in per_p.values())/len(per_p)
    macro_vs_sbs_ungated = sum(v["vs_sbs_pct_ungated"] for v in per_p.values())/len(per_p)
    return per_p, {"macro_vs_sbs_gated": macro_vs_sbs_gated, "macro_vs_sbs_ungated": macro_vs_sbs_ungated}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-grid", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    model = UnifiedSelector(d=cfg.get("d",128), depth=cfg.get("depth",4),
                            dropout=cfg.get("dropout",0.1),
                            use_mvrp_factorized=not cfg.get("no_fact", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("[gate] calibrating gammas on train subset...")
    gammas = calibrate_gating(model, device, audit, train_fraction_for_gate=0.2, gate_grid_n=args.gate_grid)
    print("[gate] evaluating gated selector on val...")
    per_p, macro = evaluate_gated(model, device, audit, gammas)
    print(f"[gate] macro_vs_sbs UNGATED={macro['macro_vs_sbs_ungated']:+.3f}%   GATED={macro['macro_vs_sbs_gated']:+.3f}%")
    for p in PROBLEMS:
        r = per_p[p]; g = gammas[p]
        print(f"  {p:>10}: gamma={r['gamma']:+.3f} switch={r['switch_rate']:.2f} "
              f"prec={r['switch_precision']:.2f} rec={r['switch_recall']:.2f} "
              f"vs_sbs gated={r['vs_sbs_pct_gated']:+.3f}% (ungated={r['vs_sbs_pct_ungated']:+.3f}%)")
    out = args.out or (Path(args.ckpt).parent / "gate_eval.json")
    Path(out).write_text(json.dumps({"gammas": gammas, "per_problem": per_p, "macro": macro}, indent=2))
    print(f"[gate] saved {out}")

if __name__ == "__main__":
    main()
