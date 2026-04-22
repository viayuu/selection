"""Dump (P, M_GLOBAL, behavior_dim) behavior features + log_prior from train raw_labels.

Outputs two files in --out-dir:
  behavior_emb.npz       key 'behavior' shape (P, M_GLOBAL, 3)
                           channel 0: oracle-hit rate per (problem, global-arm)
                           channel 1: mean normalized rank (0 = best) per (p, arm)
                           channel 2: mean regret (cost - best)/best per (p, arm)
                          Unmapped arms (not in the per-problem pool) stay at 0.
  balanced_softmax_prior.npz  key 'log_prior' shape (P, M_GLOBAL)
                                log(hit_rate + eps), used in logit-adjustment CE.

All features are computed on the TRAIN split only so they remain a legitimate
prior at val/test time.
"""
import argparse, pickle
from pathlib import Path
import numpy as np

from code.unified_selector.registry import PROBLEMS, M_GLOBAL, problem_to_pool_mask

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def compute_problem_stats(problem: str) -> dict:
    pool_order, _ = problem_to_pool_mask(problem)      # list[int] of length K_p, global arm ids
    pool_order = [int(x) for x in pool_order]
    K_p = len(pool_order)
    d = DATA_ROOT / f"{problem}train"
    # Map POOLS order to raw-cost column indices via results/*.txt filenames.
    results_dir = d / "results"
    if results_dir.exists():
        from code.unified_selector.registry import POOLS
        solver_files = sorted(f.name for f in results_dir.iterdir() if f.name.startswith("result_") and f.name.endswith(".txt"))
        solver_names = [f[len("result_"):-len(".txt")] for f in solver_files]
        cost_gather_idx = [solver_names.index(s) for s in POOLS[problem]]
    else:
        cost_gather_idx = list(range(K_p))
    with open(d / "raw_label.pkl", "rb") as f:
        raw = pickle.load(f)
    N = len(raw)
    hit = np.zeros(K_p, dtype=np.float64)
    rank_sum = np.zeros(K_p, dtype=np.float64)
    regret_sum = np.zeros(K_p, dtype=np.float64)
    for k in range(N):
        cost = np.asarray([raw[str(k)]["cost"][j] for j in cost_gather_idx], dtype=np.float64)
        best = cost.min()
        order = np.argsort(cost)                         # index = rank 0..K_p-1, value = pool_local
        ranks = np.empty(K_p, dtype=np.float64)
        ranks[order] = np.arange(K_p, dtype=np.float64)
        oracle_local = int(order[0])
        hit[oracle_local] += 1.0
        rank_sum += ranks
        regret_sum += (cost - best) / max(best, 1e-9)
    hit_rate = hit / max(N, 1)
    mean_rank = rank_sum / max(N, 1) / max(K_p - 1, 1)   # 0 = always best, 1 = always worst
    mean_regret = regret_sum / max(N, 1)
    return {
        "pool_order": pool_order, "K_p": K_p, "N": N,
        "hit_rate": hit_rate, "mean_rank": mean_rank, "mean_regret": mean_regret,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="code/unified_selector/artifacts")
    ap.add_argument("--eps", type=float, default=1e-4, help="Log-prior smoothing floor.")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    P = len(PROBLEMS)
    behavior = np.zeros((P, M_GLOBAL, 3), dtype=np.float32)
    log_prior = np.full((P, M_GLOBAL), np.log(args.eps), dtype=np.float32)
    for pi, p in enumerate(PROBLEMS):
        stats = compute_problem_stats(p)
        pool = stats["pool_order"]
        # Map per-(pool-local) stats into the (M_GLOBAL,) slot indexed by global arm id.
        for k, g in enumerate(pool):
            behavior[pi, g, 0] = stats["hit_rate"][k]
            behavior[pi, g, 1] = stats["mean_rank"][k]
            behavior[pi, g, 2] = stats["mean_regret"][k]
            log_prior[pi, g] = float(np.log(max(stats["hit_rate"][k], args.eps)))
        print(f"[{p}] N={stats['N']} K_p={stats['K_p']} hit_rate={stats['hit_rate']}")
    np.savez_compressed(out / "behavior_emb.npz", behavior=behavior)
    np.savez_compressed(out / "balanced_softmax_prior.npz", log_prior=log_prior)
    print(f"\nSaved {out/'behavior_emb.npz'} shape {behavior.shape}")
    print(f"Saved {out/'balanced_softmax_prior.npz'} shape {log_prior.shape}")


if __name__ == "__main__":
    main()
