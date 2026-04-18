"""R0 — split audit: compute SBS/VBS/normalization constants per problem.

Run: `python -m code.unified_selector.audit` from repo root.
Writes: code/unified_selector/runs/audit.json and prints a summary table.
"""
from __future__ import annotations
import json, pickle, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "code"))
from unified_selector.registry import PROBLEMS, POOLS, GLOBAL_SOLVERS, S2I, M_GLOBAL

def load_costs(problem: str, split: str):
    """Return (N, K_p) ndarray of cost[i][k], K_p = pool size for this problem."""
    d = DATA / f"{problem}{split}"
    with open(d / "raw_label.pkl", "rb") as f:
        lbl = pickle.load(f)
    n = len(lbl)
    k = len(lbl["0"]["cost"])
    costs = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        costs[i] = lbl[str(i)]["cost"][:k]
    return costs

def audit_one(problem: str):
    tr = load_costs(problem, "train")
    va = load_costs(problem, "val")
    te = load_costs(problem, "test")
    pool = POOLS[problem]
    # SBS = argmin of mean cost on val
    val_means = va.mean(axis=0)
    sbs_idx = int(np.argmin(val_means))  # position within pool
    sbs_solver = pool[sbs_idx]

    # Per-split metrics
    def stats(costs):
        vbs = np.min(costs, axis=1)
        sbs_c = costs[:, sbs_idx]
        return {
            "N": int(costs.shape[0]),
            "mean_each": costs.mean(axis=0).tolist(),
            "vbs_mean": float(vbs.mean()),
            "sbs_mean": float(sbs_c.mean()),
            "oracle_sbs_gap_pct": float((sbs_c.mean() - vbs.mean()) / (abs(vbs.mean()) + 1e-9) * 100),
            "sbs_win_rate": float((costs.argmin(axis=1) == sbs_idx).mean()),
            "winner_hist": np.bincount(costs.argmin(axis=1), minlength=len(pool)).tolist(),
        }

    return {
        "problem": problem,
        "pool": pool,
        "pool_global_ids": [S2I[s] for s in pool],
        "sbs_solver": sbs_solver,
        "sbs_pool_idx": sbs_idx,
        "train": stats(tr),
        "val":   stats(va),
        "test":  stats(te),
    }

def main():
    out = {}
    for p in PROBLEMS:
        print(f"... auditing {p}")
        out[p] = audit_one(p)
    outdir = Path(__file__).parent / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "audit.json").write_text(json.dumps(out, indent=2))
    # Summary
    print("\n=== R0 Audit Summary ===")
    print(f"{'problem':>12} {'K':>3} {'SBS':>12} {'val_sbs':>9} {'val_vbs':>9} {'gap%':>7} {'winrate':>8}")
    for p in PROBLEMS:
        r = out[p]
        v = r["val"]
        print(f"{p:>12} {len(r['pool']):>3} {r['sbs_solver']:>12} {v['sbs_mean']:>9.4f} {v['vbs_mean']:>9.4f} {v['oracle_sbs_gap_pct']:>6.2f}% {v['sbs_win_rate']:>7.2%}")
    print(f"\nSaved to {outdir/'audit.json'}")

if __name__ == "__main__":
    main()
