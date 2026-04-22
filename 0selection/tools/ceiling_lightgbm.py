"""Feature-ceiling LightGBM benchmark (reviewer: pre-conditions before chasing 80%).

Extracts R40D pooled instance embedding + cbits + (optional) retrieval prior + behavior emb
as frozen features, fits 18 per-problem LightGBM multiclass classifiers on train, reports
val top1 + test top1. Answers: "assuming encoder is frozen at R40D quality and labels are
what they are, how high can a shallow learner go?"

If LightGBM can only reach ~0.55-0.60 macro_top1, then: encoder depth is NOT the bottleneck,
label noise / feature insufficiency is. If LightGBM reaches 0.70+, NSS encoder is
underfitting — more capacity / longer training is worth trying.

Usage:
  python -m tools.ceiling_lightgbm \
      --ckpt code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/best_top1.pt \
      --artifacts-dir code/unified_selector/artifacts \
      --out tools/reports/ceiling_lightgbm.md
"""
import argparse, json, pickle
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import (
    PROBLEMS, M_GLOBAL, P2I, problem_to_pool_mask, constraint_bits, K_CBITS,
)
from code.unified_selector.data import UnifiedProblemDataset, DATA_ROOT

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("lightgbm is not installed in this env. pip install lightgbm")


def load_model(ckpt_path: str, device: torch.device):
    from code.unified_selector.model import UnifiedSelector
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = UnifiedSelector(
        d=args.get("d", 128), depth=args.get("depth", 4), dropout=0.0,
        use_mvrp_factorized=not args.get("no_fact", False),
        use_problem_solver_bias=not args.get("no_problem_solver_bias", False),
        use_film=args.get("use_film", False),
        use_arm_attn=args.get("use_arm_attn", False),
        use_coe=args.get("use_coe", False),
        arm_attn_heads=args.get("arm_attn_heads", 4),
        coe_experts=args.get("coe_experts", None),
        encoder_type=args.get("encoder_type", "standard"),
        rezero=args.get("rezero", False),
        block_num=args.get("block_num", 2),
        encoder_layer_num=args.get("encoder_layer_num", 2),
        heads=args.get("heads", 4),
        downsample_ratio=args.get("downsample_ratio", 0.8),
        local_head=args.get("local_head", False),
        residual_adapter=args.get("residual_adapter", False),
    ).to(device)
    sd = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] ckpt={ckpt_path}; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


@torch.no_grad()
def extract_pooled_h(model, problem: str, split: str, device: torch.device) -> np.ndarray:
    from code.unified_selector.data import collate_single_problem
    ds = UnifiedProblemDataset(problem, split, aug_8fold=False)
    N = len(ds)
    hs = []
    batch_size = 32
    for i in range(0, N, batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, N))]
        b = collate_single_problem(batch)
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        h = model.encode_instance(b)
        if isinstance(h, tuple):
            h = h[0]
        hs.append(h.cpu().numpy())
    return np.concatenate(hs, axis=0)   # (N, d)


def build_features(problem: str, pooled_h: np.ndarray, behavior: np.ndarray) -> np.ndarray:
    """For each instance, feature = pooled_h || cbits || per-arm behavior-diffs (hit_rate[arm] - hit_rate_mean).

    Returns (N, d + K_CBITS + K_p*3) float32 matrix.  Labels built separately.
    """
    N, d = pooled_h.shape
    pool_order, _ = problem_to_pool_mask(problem)
    pool_order = np.asarray([int(x) for x in pool_order], dtype=np.int64)
    K_p = len(pool_order)
    cbits = np.asarray(constraint_bits(problem), dtype=np.float32)    # (K_CBITS,)
    pi = P2I[problem]
    # behavior[pi, g, c] -> grab per-arm slots
    arm_feats = behavior[pi][pool_order]                              # (K_p, 3)
    arm_feats_flat = arm_feats.flatten()                              # (K_p*3,)
    inst_feats = np.concatenate([cbits, arm_feats_flat])              # (K_CBITS + K_p*3,)
    inst_tiled = np.broadcast_to(inst_feats[None, :], (N, inst_feats.shape[0]))
    X = np.concatenate([pooled_h.astype(np.float32), inst_tiled.astype(np.float32)], axis=1)
    return X


def load_labels(problem: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (oracle_local_idx, costs_matrix) for all instances of (problem, split)."""
    d = DATA_ROOT / f"{problem}{split}"
    with open(d / "raw_label.pkl", "rb") as f:
        raw = pickle.load(f)
    N = len(raw)
    pool_order, _ = problem_to_pool_mask(problem)
    K_p = len(pool_order)
    # Align raw cost (sorted solver file order) -> POOLS order
    results_dir = d / "results"
    if results_dir.exists():
        from code.unified_selector.registry import POOLS
        solver_files = sorted(f.name for f in results_dir.iterdir() if f.name.startswith("result_") and f.name.endswith(".txt"))
        solver_names = [f[len("result_"):-len(".txt")] for f in solver_files]
        cost_gather_idx = [solver_names.index(s) for s in POOLS[problem]]
    else:
        cost_gather_idx = list(range(K_p))
    costs = np.zeros((N, K_p), dtype=np.float32)
    oracle = np.zeros(N, dtype=np.int64)
    for k in range(N):
        c = np.asarray([raw[str(k)]["cost"][j] for j in cost_gather_idx], dtype=np.float32)
        costs[k] = c
        oracle[k] = int(c.argmin())
    return oracle, costs


def train_lightgbm_per_problem(problem, X_tr, y_tr, X_va, y_va, X_te, y_te,
                                params: dict, num_rounds: int = 200) -> dict:
    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=False)
    n_class = int(max(y_tr.max(), y_va.max(), y_te.max())) + 1
    p = dict(params)
    p["num_class"] = n_class
    model = lgb.train(p, dtr, num_boost_round=num_rounds, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(20, verbose=False)])
    pred_va = model.predict(X_va).argmax(axis=1)
    pred_te = model.predict(X_te).argmax(axis=1)
    return {
        "val_top1": float((pred_va == y_va).mean()),
        "test_top1": float((pred_te == y_te).mean()),
        "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
        "n_class": n_class, "best_iter": model.best_iteration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--artifacts-dir", default="code/unified_selector/artifacts")
    ap.add_argument("--out", default="tools/reports/ceiling_lightgbm.md")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--num-rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    beh = np.load(Path(args.artifacts_dir) / "behavior_emb.npz")["behavior"]
    params = {
        "objective": "multiclass", "metric": "multi_logloss",
        "learning_rate": args.lr, "num_leaves": args.leaves,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "verbosity": -1, "num_threads": 8, "seed": 0,
    }
    per_problem = {}
    for p in PROBLEMS:
        print(f"\n[{p}] extracting features...")
        h_tr = extract_pooled_h(model, p, "train", device)
        h_va = extract_pooled_h(model, p, "val", device)
        h_te = extract_pooled_h(model, p, "test", device)
        X_tr = build_features(p, h_tr, beh)
        X_va = build_features(p, h_va, beh)
        X_te = build_features(p, h_te, beh)
        y_tr, _ = load_labels(p, "train")
        y_va, _ = load_labels(p, "val")
        y_te, _ = load_labels(p, "test")
        print(f"[{p}] X shapes tr/va/te: {X_tr.shape} / {X_va.shape} / {X_te.shape}, "
              f"classes={int(y_tr.max()) + 1}")
        r = train_lightgbm_per_problem(p, X_tr, y_tr, X_va, y_va, X_te, y_te,
                                        params, num_rounds=args.num_rounds)
        per_problem[p] = r
        print(f"[{p}] val={r['val_top1']:.4f} test={r['test_top1']:.4f} "
              f"best_iter={r['best_iter']}")

    val_macro = float(np.mean([per_problem[p]["val_top1"] for p in PROBLEMS]))
    test_macro = float(np.mean([per_problem[p]["test_top1"] for p in PROBLEMS]))

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# LightGBM feature-ceiling with {args.ckpt} as frozen encoder\n",
        f"- val macro top1 = **{val_macro:.4f}**",
        f"- test macro top1 = **{test_macro:.4f}**",
        f"- R40D test macro = 0.5226 / R18 test macro = 0.5166 (baselines)",
        f"\n## Per-problem",
        "| Problem | val | test | n_class | best_iter |",
        "|:---|---:|---:|---:|---:|",
    ]
    for p in PROBLEMS:
        r = per_problem[p]
        lines.append(f"| {p} | {r['val_top1']:.4f} | {r['test_top1']:.4f} | {r['n_class']} | {r['best_iter']} |")
    out_path.write_text("\n".join(lines) + "\n")
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"val_macro": val_macro, "test_macro": test_macro,
                   "per_problem": per_problem, "args": vars(args)}, f, indent=2)
    print(f"\n=> macro val/test top1 = {val_macro:.4f} / {test_macro:.4f}")
    print(f"=> saved {out_path}")


if __name__ == "__main__":
    main()
