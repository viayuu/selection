"""Multi-seed aggregation + bootstrap CI + frozen-gate test evaluation for the unified selector.

Reads a list of checkpoint directories (each with best.pt + optional gate_eval.json), runs
analyze on val+test for each, applies the frozen (val-calibrated) per-problem γ to the test split,
computes mean + 95% bootstrap CI across seeds for macro_vs_sbs_pct (val/test, gated/ungated).

Usage:
    python -m code.unified_selector.multiseed \
        --ckpts code/unified_selector/runs/R4_soft_bias/best.pt \
                code/unified_selector/runs/R4_soft_bias_seed1/best.pt \
                code/unified_selector/runs/R4_soft_bias_seed2/best.pt \
                code/unified_selector/runs/R4_soft_bias_seed3/best.pt \
                code/unified_selector/runs/R4_soft_bias_seed4/best.pt \
        --out code/unified_selector/runs/multiseed_R4
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, P2I
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, _migrate_state_dict
from .gate import calibrate_gating


@torch.no_grad()
def gated_eval_test_frozen(model, device, audit, gammas, split: str = "test"):
    per_p = {}
    for p in PROBLEMS:
        ds = UnifiedProblemDataset(p, split)
        dl = DataLoader(ds, batch_size=32, collate_fn=collate_single_problem, num_workers=0)
        costs_all = []; best_cost_all = []; sbs_cost_all = []; margin_all = []
        sbs_idx = audit[p]["sbs_pool_idx"]
        for b in dl:
            for k, v in b.items():
                if torch.is_tensor(v): b[k] = v.to(device)
            logits = model(b)
            log_p = F.log_softmax(logits, dim=1)
            log_p_pool = log_p[:, b["pool_ids"]]
            best_idx = log_p_pool.argmax(dim=1)
            best_cost = b["costs"].gather(1, best_idx.unsqueeze(1)).squeeze(1)
            sbs_cost = b["costs"][:, sbs_idx]
            margin = log_p_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1) - log_p_pool[:, sbs_idx]
            costs_all.append(b["costs"].cpu())
            best_cost_all.append(best_cost.cpu())
            sbs_cost_all.append(sbs_cost.cpu())
            margin_all.append(margin.cpu())
        best_cost_all = torch.cat(best_cost_all).numpy()
        sbs_cost_all = torch.cat(sbs_cost_all).numpy()
        margin_all = torch.cat(margin_all).numpy()
        g = gammas[p]["gamma"]
        use_best = margin_all >= g
        sel_cost = np.where(use_best, best_cost_all, sbs_cost_all)
        sbs_mean = audit[p][split]["sbs_mean"]
        vs_sbs = (sel_cost.mean() - sbs_mean) / (abs(sbs_mean) + 1e-9) * 100
        vs_sbs_ungated = (best_cost_all.mean() - sbs_mean) / (abs(sbs_mean) + 1e-9) * 100
        per_p[p] = dict(
            sel_cost=sel_cost.tolist(),
            sbs_cost=sbs_cost_all.tolist(),
            best_cost=best_cost_all.tolist(),
            gamma=g, switch_rate=float(use_best.mean()),
            vs_sbs_gated_pct=float(vs_sbs),
            vs_sbs_ungated_pct=float(vs_sbs_ungated),
            mean_cost_gated=float(sel_cost.mean()),
            mean_cost_ungated=float(best_cost_all.mean()),
            mean_cost_sbs=float(sbs_cost_all.mean()),
        )
    return per_p


def bootstrap_ci(values, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(0)
    arr = np.array(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = arr[rng.integers(0, len(arr), len(arr))].mean()
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    return float(arr.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(args.audit).read_text())

    per_seed = []
    # For each ckpt: calibrate γ on val (via gate.py), then evaluate gated on test with frozen γ.
    for ck_path in args.ckpts:
        ck_path = Path(ck_path)
        print(f"\n=== {ck_path} ===")
        ckpt = torch.load(ck_path, map_location=args.device)
        cfg = ckpt.get("args", {})
        has_bias = "problem_solver_bias" in ckpt["model"]
        model = UnifiedSelector(d=cfg.get("d", 128), depth=cfg.get("depth", 4),
                                dropout=cfg.get("dropout", 0.1),
                                use_mvrp_factorized=not cfg.get("no_fact", False),
                                use_problem_solver_bias=has_bias,
                                use_film=cfg.get("use_film", False),
                                use_arm_attn=cfg.get("use_arm_attn", False),
                                use_coe=cfg.get("use_coe", False),
                                arm_attn_heads=cfg.get("arm_attn_heads", 4),
                                coe_experts=cfg.get("coe_experts", None)).to(args.device)
        model.load_state_dict(_migrate_state_dict(ckpt["model"]), strict=False)
        model.eval()

        # Calibrate γ on VAL (true val-calibrated protocol; matches the paper claim).
        print("  calibrating γ on val (split='val', all samples)")
        gammas = calibrate_gating(model, args.device, audit, gate_grid_n=40, split="val")

        # Evaluate on val (gated, for completeness)
        val_per_p = gated_eval_test_frozen(model, args.device, audit, gammas, split="val")
        test_per_p = gated_eval_test_frozen(model, args.device, audit, gammas, split="test")

        macro_val_gated = float(np.mean([r["vs_sbs_gated_pct"] for r in val_per_p.values()]))
        macro_val_ungated = float(np.mean([r["vs_sbs_ungated_pct"] for r in val_per_p.values()]))
        macro_test_gated = float(np.mean([r["vs_sbs_gated_pct"] for r in test_per_p.values()]))
        macro_test_ungated = float(np.mean([r["vs_sbs_ungated_pct"] for r in test_per_p.values()]))

        print(f"  val_ungated={macro_val_ungated:+.3f}%  val_gated={macro_val_gated:+.3f}%")
        print(f"  test_ungated={macro_test_ungated:+.3f}%  test_gated={macro_test_gated:+.3f}%  (γ frozen from val)")
        per_seed.append(dict(
            ckpt=str(ck_path),
            gammas=gammas,
            val_per_problem=val_per_p,
            test_per_problem=test_per_p,
            macro_val_ungated=macro_val_ungated,
            macro_val_gated=macro_val_gated,
            macro_test_ungated=macro_test_ungated,
            macro_test_gated=macro_test_gated,
        ))

    # Aggregate across seeds
    ug_vals = [s["macro_val_ungated"] for s in per_seed]
    g_vals = [s["macro_val_gated"] for s in per_seed]
    ug_tests = [s["macro_test_ungated"] for s in per_seed]
    g_tests = [s["macro_test_gated"] for s in per_seed]
    summary = dict(
        n_seeds=len(per_seed),
        val_ungated_mean_lo_hi=bootstrap_ci(ug_vals),
        val_gated_mean_lo_hi=bootstrap_ci(g_vals),
        test_ungated_mean_lo_hi=bootstrap_ci(ug_tests),
        test_gated_mean_lo_hi=bootstrap_ci(g_tests),
    )
    print("\n=== AGGREGATE ===")
    for k, v in summary.items():
        if isinstance(v, tuple):
            mean, lo, hi = v
            print(f"  {k}: {mean:+.3f}% [95% CI: {lo:+.3f}, {hi:+.3f}]")
        else:
            print(f"  {k}: {v}")

    (out / "multiseed_report.json").write_text(json.dumps({
        "per_seed": per_seed, "summary": summary}, indent=2, default=str))
    print(f"\nSaved {out}/multiseed_report.json")


if __name__ == "__main__":
    main()
