"""Rebuild runs/audit.json from data/*/raw_label.pkl.

Schema per problem:
  pool: list[str]                 # solver names in pool order
  pool_global_ids: list[int]      # GLOBAL_SOLVERS indices
  sbs_pool_idx: int               # argmin_k mean(cost[:, k]) on train split
  train/val/test:
    sbs_mean: float               # mean cost using single best solver (fixed on train)
    vbs_mean: float               # mean of per-instance min cost across pool
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np

from .registry import PROBLEMS, POOLS, S2I

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent / "runs" / "audit.json"


def load_cost_matrix(problem: str, split: str) -> np.ndarray:
    d = DATA_ROOT / f"{problem}{split}"
    with open(d / "raw_label.pkl", "rb") as f:
        raw = pickle.load(f)
    K_p = len(POOLS[problem])
    costs = np.stack([np.asarray(raw[str(i)]["cost"])[:K_p] for i in range(len(raw))])
    return costs  # (N, K_p)


def main():
    audit = {}
    for p in PROBLEMS:
        pool = POOLS[p]
        pool_global_ids = [S2I[s] for s in pool]
        train_costs = load_cost_matrix(p, "train")
        val_costs = load_cost_matrix(p, "val")
        test_costs = load_cost_matrix(p, "test")
        sbs_pool_idx = int(train_costs.mean(axis=0).argmin())
        entry = {
            "pool": pool,
            "pool_global_ids": pool_global_ids,
            "sbs_pool_idx": sbs_pool_idx,
        }
        for name, c in [("train", train_costs), ("val", val_costs), ("test", test_costs)]:
            entry[name] = {
                "sbs_mean": float(c[:, sbs_pool_idx].mean()),
                "vbs_mean": float(c.min(axis=1).mean()),
                "n": int(c.shape[0]),
            }
        audit[p] = entry
        print(f"[{p}] pool={pool} sbs={pool[sbs_pool_idx]} "
              f"val sbs={entry['val']['sbs_mean']:.4f} vbs={entry['val']['vbs_mean']:.4f} "
              f"test sbs={entry['test']['sbs_mean']:.4f} vbs={entry['test']['vbs_mean']:.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nWrote {OUT} ({len(audit)} problems)")


if __name__ == "__main__":
    main()
