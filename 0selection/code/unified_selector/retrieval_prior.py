"""Plan I — Instance-level retrieval prior (oracle-pro recommendation).

Rationale: for each test instance, find its k nearest neighbors in the training set
(by encoder's pooled embedding), and use those neighbors' oracle-arm distribution
as a soft prior. Add this as a bonus logit to the base model's predictions.

This is especially expected to help in the zero-pick-with-oracle-wins mass:
if a rare-arm is the oracle for a specific kind of instance, similar training
instances will reveal that, even though R18's supervised argmax never picks it.

Usage:
  python -m code.unified_selector.retrieval_prior \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R25_retrieval \
    --split test --k 32 --weight-grid 0.0 0.1 0.2 0.5 1.0 2.0 4.0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, POOLS
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, _migrate_state_dict


def load_base(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("args", {})
    has_bias = "problem_solver_bias" in ck["model"]
    m = UnifiedSelector(
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
    m.load_state_dict(_migrate_state_dict(ck["model"]), strict=False)
    m.eval()
    return m


@torch.no_grad()
def encode_instances(model, split, device):
    """Returns per-problem: pooled embeddings (N, d), oracle indices (N,), costs (N, K_p), logits_pool (N, K_p)."""
    per_p = {}
    for p in PROBLEMS:
        ds = UnifiedProblemDataset(p, split)
        dl = DataLoader(ds, batch_size=32, collate_fn=collate_single_problem, num_workers=0)
        pooled_all = []; costs_all = []; logits_all = []
        for b in dl:
            for k, v in b.items():
                if torch.is_tensor(v):
                    b[k] = v.to(device)
            # Forward through encoder to get pooled embedding `g`
            if b["kind"] == "coord":
                g = model.coord_enc(b["node"], b["node_mask"])
            else:
                g = model.matrix_enc(b["matrix"], b["node_mask"])
            # Also compute logits_pool (for the blend)
            logits_pool = model(b)[:, b["pool_ids"]]
            pooled_all.append(g.cpu().numpy())
            costs_all.append(b["costs"].cpu().numpy())
            logits_all.append(logits_pool.cpu().numpy())
        pooled = np.concatenate(pooled_all, axis=0)      # (N, d)
        costs = np.concatenate(costs_all, axis=0)        # (N, K_p)
        logits = np.concatenate(logits_all, axis=0)      # (N, K_p)
        per_p[p] = dict(pooled=pooled, costs=costs, logits=logits,
                        oracle=costs.argmin(axis=1))
    return per_p


def compute_retrieval_prior(train_p, test_p, k=32, eps=1e-6, temperature=0.1):
    """For each test instance per problem, find k-NN in training (cosine sim on pooled emb),
    aggregate their oracle-arm distribution.

    Returns: per-problem (N_test, K_p) prior distribution.
    """
    priors = {}
    for p in PROBLEMS:
        t = train_p[p]; q = test_p[p]
        # Normalize
        t_emb = t["pooled"] / (np.linalg.norm(t["pooled"], axis=1, keepdims=True) + eps)
        q_emb = q["pooled"] / (np.linalg.norm(q["pooled"], axis=1, keepdims=True) + eps)
        sim = q_emb @ t_emb.T              # (N_test, N_train)
        # Top-k neighbors
        topk_idx = np.argpartition(-sim, k, axis=1)[:, :k]    # (N_test, k)
        # Their oracle arms
        t_oracle = t["oracle"]                                 # (N_train,)
        K_p = q["costs"].shape[1]
        prior = np.zeros((q_emb.shape[0], K_p), dtype=np.float32)
        for i in range(q_emb.shape[0]):
            neigh_oracles = t_oracle[topk_idx[i]]
            # Softmax-weighted by similarity
            sims = sim[i, topk_idx[i]]                          # (k,)
            w = np.exp(sims / temperature)
            w /= w.sum() + eps
            for j, arm in enumerate(neigh_oracles):
                prior[i, arm] += w[j]
        priors[p] = prior
    return priors


def eval_with_prior(test_p, priors, alpha_grid):
    """Blend: logit_blend = alpha * base_logit + (1-alpha) * log(prior + eps)."""
    results = {a: {"per_p": {}} for a in alpha_grid}
    picks_per_alpha = {a: {} for a in alpha_grid}  # For bootstrap
    for p in PROBLEMS:
        q = test_p[p]
        costs = q["costs"]
        oracle = q["oracle"]
        sbs_idx = int(costs.mean(axis=0).argmin())
        sbs_mean = costs[:, sbs_idx].mean()
        for alpha in alpha_grid:
            # Blend logit + log(prior)
            log_prior = np.log(priors[p] + 1e-6)  # (N, K_p)
            base_logits = q["logits"]
            # Normalize scales (each in 0-mean form)
            base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
            prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
            blend = alpha * base_norm + (1 - alpha) * prior_norm
            pred = blend.argmax(axis=1)
            top1 = float((pred == oracle).mean())
            mean_c = float(costs[np.arange(costs.shape[0]), pred].mean())
            vs_sbs = 100 * (mean_c - sbs_mean) / (abs(sbs_mean) + 1e-9)
            results[alpha]["per_p"][p] = dict(top1=top1, mean_cost=mean_c, vs_sbs_pct=vs_sbs,
                                              N=int(costs.shape[0]), K_p=int(costs.shape[1]))
            picks_per_alpha[alpha][p] = dict(picks=pred.astype(np.int32), costs=costs,
                                              oracle=oracle.astype(np.int32))
    for a in alpha_grid:
        pp = results[a]["per_p"]
        results[a]["macro_top1"] = float(np.mean([v["top1"] for v in pp.values()]))
        results[a]["macro_vs_sbs_pct"] = float(np.mean([v["vs_sbs_pct"] for v in pp.values()]))
    return results, picks_per_alpha


def bootstrap_vs_base(picks_per_alpha, ref_alpha=1.0, n_boot=5000, seed=0):
    """Bootstrap macro top1 delta of each α vs ref_alpha (pure base)."""
    rng = np.random.default_rng(seed)
    ref = picks_per_alpha[ref_alpha]
    bootstrap_out = {}
    for a, pp in picks_per_alpha.items():
        if a == ref_alpha:
            continue
        boot_top1 = np.zeros(n_boot)
        boot_cost = np.zeros(n_boot)
        for b in range(n_boot):
            per_p_top1 = []
            per_p_cost = []
            for p in PROBLEMS:
                A, B = pp[p], ref[p]
                N = A["picks"].shape[0]
                idx = rng.integers(0, N, N)
                oracle_idx = A["oracle"][idx]
                top1_a = (A["picks"][idx] == oracle_idx).mean()
                top1_b = (B["picks"][idx] == oracle_idx).mean()
                cost_a = A["costs"][idx, A["picks"][idx]].mean()
                cost_b = B["costs"][idx, B["picks"][idx]].mean()
                per_p_top1.append(top1_a - top1_b)
                per_p_cost.append(100 * (cost_a - cost_b) / (abs(cost_b) + 1e-9))
            boot_top1[b] = np.mean(per_p_top1)
            boot_cost[b] = np.mean(per_p_cost)
        bootstrap_out[a] = dict(
            top1_mean=float(boot_top1.mean()),
            top1_lo=float(np.quantile(boot_top1, 0.025)),
            top1_hi=float(np.quantile(boot_top1, 0.975)),
            cost_mean=float(boot_cost.mean()),
            cost_lo=float(np.quantile(boot_cost, 0.025)),
            cost_hi=float(np.quantile(boot_cost, 0.975)),
        )
    return bootstrap_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--alpha-grid", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="Softmax temperature on similarities for neighbor weighting.")
    ap.add_argument("--n-boot", type=int, default=5000, help="Bootstrap samples for α sig test.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[retrieval] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[retrieval] encoding train set")
    train_p = encode_instances(base_model, "train", device)
    print(f"[retrieval] encoding {args.split} set")
    test_p = encode_instances(base_model, args.split, device)

    print(f"[retrieval] computing k={args.k}-NN retrieval priors (τ={args.temperature})")
    priors = compute_retrieval_prior(train_p, test_p, k=args.k, temperature=args.temperature)

    print(f"[retrieval] α-sweep eval")
    res, picks_per_alpha = eval_with_prior(test_p, priors, args.alpha_grid)

    # Bootstrap each α vs α=1.0 (pure base) if in grid
    boot = {}
    if 1.0 in args.alpha_grid:
        print(f"[retrieval] bootstrapping α-sweep vs α=1.0, n_boot={args.n_boot}")
        boot = bootstrap_vs_base(picks_per_alpha, ref_alpha=1.0, n_boot=args.n_boot)

    lines = [f"# Retrieval-prior α-sweep ({args.split})\n",
             f"Base: {args.base_ckpt}, k={args.k} neighbors, τ={args.temperature}.\n",
             "| α (base weight) | macro top1 | Δtop1 [95% CI] | macro vs_sbs | Δcost% [95% CI] |",
             "|---:|---:|---:|---:|---:|"]
    for a in args.alpha_grid:
        if a == 1.0 or a not in boot:
            boot_str = "—"
            cost_str = "—"
        else:
            b = boot[a]
            sig_t = "*" if (b["top1_lo"] > 0 or b["top1_hi"] < 0) else ""
            sig_c = "*" if (b["cost_lo"] > 0 or b["cost_hi"] < 0) else ""
            boot_str = f"{b['top1_mean']:+.4f} [{b['top1_lo']:+.4f}, {b['top1_hi']:+.4f}]{sig_t}"
            cost_str = f"{b['cost_mean']:+.4f}% [{b['cost_lo']:+.4f}, {b['cost_hi']:+.4f}]{sig_c}"
        lines.append(f"| {a:.2f} | {res[a]['macro_top1']:.4f} | {boot_str} | {res[a]['macro_vs_sbs_pct']:+.4f}% | {cost_str} |")
    (out / f"retrieval_{args.split}.md").write_text("\n".join(lines))
    (out / f"retrieval_{args.split}.json").write_text(json.dumps(res, indent=2, default=float))
    (out / f"retrieval_{args.split}_bootstrap.json").write_text(json.dumps(boot, indent=2, default=float))
    # Save per-instance picks for cross-method blending
    np.savez(out / f"retrieval_{args.split}_picks.npz",
             **{f"{p}_{a:.2f}": picks_per_alpha[a][p]["picks"]
                for a in args.alpha_grid for p in PROBLEMS})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
