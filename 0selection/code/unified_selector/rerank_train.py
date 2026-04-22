"""R42S — KEEP_SBS + full-pool reranker (Plan C).

Anchors the selector on the per-problem SBS arm and learns when to
*override* with a different pool arm. Output action space per instance:
    {KEEP_SBS}  ∪  {all K_p pool arms}

Training target (per reviewer §4):
    target = KEEP_SBS if oracle == sbs OR regret_gap_pct < keep_tol_pct
           = oracle arm otherwise

Input features per arm:
    - R18 per-arm logit + base rank / margin to SBS
    - retrieval oracle-vote prior (k-NN, cross-fit on train)
    - retrieval cost prior (same; low=good)
    - learned solver embedding (d=16)
    - arm behavioural stats from train audit (oracle-frequency, mean regret)

Instance features:
    - R18 pooled encoder output (d=128)
    - learned problem embedding (d=16)
    - cbits (K=5)
    - base_entropy, base_margin(top1 - top2)

Loss (per reviewer §4):
    CE(scores, target) + 0.5 · pairwise_cost_loss + 0.3 · KL_to_base_softmax

Inference (val-locked threshold[p]):
    if keep_score - max(arm_scores) >= threshold[p]:
        pick = sbs_pool_idx
    else:
        pick = arm_scores.argmax()

Usage example:
  python -m code.unified_selector.rerank_train \
    --r18-ckpt code/unified_selector/runs/R18_alltail/best_top1.pt \
    --audit code/unified_selector/runs/audit.json \
    --epochs 8 --batch-size 256 --lr 1e-4 \
    --retrieval-k 32 --retrieval-tau 0.1 \
    --keep-ce-weight 1.0 --pairwise-cost-weight 0.5 --kl-weight 0.3 \
    --sample-mix 0.4,0.25,0.25,0.10 \
    --seed 0 --device cuda:1 \
    --save-dir code/unified_selector/runs/R42S_keepSBS_fullpool_rerank_seed0
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import PROBLEMS, POOLS, S2I, M_GLOBAL, K_CBITS
from .model import UnifiedSelector, _migrate_state_dict
from .retrieval_prior import load_base, encode_instances, compute_retrieval_prior
from .retrieval_cost import compute_cost_prior


# ---------------------------------------------------------------------------
#  Feature construction
# ---------------------------------------------------------------------------

def cross_fit_priors(enc_dict, k, temperature, n_folds, seed=0):
    """Vote + cost priors with exclude-self folds for train."""
    vote = {}
    cost = {}
    rng = np.random.default_rng(seed)
    for p in PROBLEMS:
        d = enc_dict[p]
        pooled = d["pooled"]
        costs = d["costs"]
        oracle = d["oracle"]
        N, K_p = costs.shape
        eps = 1e-6
        emb = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + eps)
        v_prior = np.zeros((N, K_p), dtype=np.float32)
        c_prior = np.zeros((N, K_p), dtype=np.float32)
        perm = rng.permutation(N)
        folds = np.array_split(perm, n_folds)
        for fi, fold_idx in enumerate(folds):
            t_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
            t_emb = emb[t_idx]
            q_emb = emb[fold_idx]
            sim = q_emb @ t_emb.T
            topk_idx = np.argpartition(-sim, min(k, sim.shape[1] - 1), axis=1)[:, :k]
            t_oracle = oracle[t_idx]
            t_costs = costs[t_idx]
            # rank-normalized cost (R42S uses rank so problem scale doesn't dominate)
            t_rank = np.argsort(np.argsort(t_costs, axis=1), axis=1).astype(np.float32)
            for i, qi in enumerate(fold_idx):
                idx = topk_idx[i]
                sims = sim[i, idx]
                w = np.exp(sims / temperature)
                w /= w.sum() + eps
                # vote prior
                for j, arm in enumerate(t_oracle[idx]):
                    v_prior[qi, arm] += w[j]
                # cost prior (lower = better)
                c_prior[qi] = (w[:, None] * t_rank[idx]).sum(axis=0)
        vote[p] = v_prior
        cost[p] = c_prior
    return vote, cost


def compute_arm_behavior_stats(train_enc):
    """Per-problem per-arm oracle-pick frequency + mean regret (if picked)."""
    stats = {}
    for p in PROBLEMS:
        d = train_enc[p]
        costs = d["costs"]
        oracle = d["oracle"]
        K_p = costs.shape[1]
        freq = np.zeros(K_p, dtype=np.float32)
        for a in oracle:
            freq[a] += 1
        freq /= max(1, len(oracle))
        # Mean regret if picked = E[(cost_k - oracle_cost) / |oracle_cost|]
        oracle_cost = costs[np.arange(len(oracle)), oracle][:, None]
        per_arm_regret = ((costs - oracle_cost) / (np.abs(oracle_cost) + 1e-9)).mean(axis=0)
        stats[p] = {"oracle_freq": freq, "mean_regret": per_arm_regret.astype(np.float32)}
    return stats


def build_inst_features(enc_dict):
    """Per-instance scalar features derived from base logits."""
    out = {}
    for p in PROBLEMS:
        logits = enc_dict[p]["logits"]
        base_norm = logits - logits.mean(axis=1, keepdims=True)
        sorted_logits = np.sort(base_norm, axis=1)[:, ::-1]
        K_p = logits.shape[1]
        base_margin = (sorted_logits[:, 0] - sorted_logits[:, 1]
                       if K_p >= 2 else np.zeros(logits.shape[0]))
        soft = np.exp(base_norm - base_norm.max(axis=1, keepdims=True))
        soft = soft / soft.sum(axis=1, keepdims=True)
        base_entropy = -(soft * np.log(soft + 1e-9)).sum(axis=1)
        out[p] = np.stack([base_margin, base_entropy], axis=1).astype(np.float32)  # (N, 2)
    return out


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class RerankHead(nn.Module):
    """Arm-level scorer + instance-level KEEP scorer."""

    def __init__(self, d_arm_feats: int, d_inst_feats: int, d_solver_emb: int = 16,
                 d_problem_emb: int = 16, d_pool_h: int = 128,
                 d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_inst_feats = d_inst_feats
        self.solver_emb = nn.Embedding(M_GLOBAL, d_solver_emb)
        self.problem_emb = nn.Embedding(len(PROBLEMS), d_problem_emb)
        arm_in = d_arm_feats + d_solver_emb + d_problem_emb + d_inst_feats + d_pool_h + K_CBITS
        keep_in = d_problem_emb + d_inst_feats + d_pool_h + K_CBITS
        self.arm_mlp = nn.Sequential(
            nn.Linear(arm_in, 2 * d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        self.keep_mlp = nn.Sequential(
            nn.Linear(keep_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, arm_feats, pool_global_ids, pid, inst_feats, pool_h, cbits):
        """
        arm_feats:   (B, K_p, d_arm_feats)
        pool_global_ids: (K_p,) long  — global solver ids for solver_emb lookup
        pid:          (B,) long
        inst_feats:   (B, d_inst_feats)
        pool_h:       (B, d_pool_h) — R18 pooled h
        cbits:        (B, K_cbits)
        Returns (arm_scores (B, K_p), keep_score (B,))
        """
        B, K_p, _ = arm_feats.shape
        solver = self.solver_emb(pool_global_ids).unsqueeze(0).expand(B, -1, -1)  # (B, K_p, d_s)
        pe = self.problem_emb(pid)                                                # (B, d_p)
        pe_exp = pe.unsqueeze(1).expand(B, K_p, -1)
        inst_exp = inst_feats.unsqueeze(1).expand(B, K_p, -1)
        pool_h_exp = pool_h.unsqueeze(1).expand(B, K_p, -1)
        cbits_exp = cbits.unsqueeze(1).expand(B, K_p, -1)
        arm_in = torch.cat([arm_feats, solver, pe_exp, inst_exp, pool_h_exp, cbits_exp], dim=-1)
        arm_scores = self.arm_mlp(arm_in).squeeze(-1)                            # (B, K_p)
        keep_in = torch.cat([pe, inst_feats, pool_h, cbits], dim=-1)             # (B, keep_in)
        keep_score = self.keep_mlp(keep_in).squeeze(-1)                          # (B,)
        return arm_scores, keep_score


# ---------------------------------------------------------------------------
#  Per-problem tensor dict + feature assembly
# ---------------------------------------------------------------------------

def assemble_tensors(enc_dict, vote, cost, inst_feats, arm_stats, audit, device,
                     pid_map):
    """Pack everything we need for training/inference into per-problem dicts of tensors."""
    out = {}
    for p in PROBLEMS:
        d = enc_dict[p]
        logits = d["logits"].astype(np.float32)
        costs = d["costs"].astype(np.float32)
        oracle = d["oracle"].astype(np.int64)
        pooled = d["pooled"].astype(np.float32)
        sbs_idx = int(audit[p]["sbs_pool_idx"])
        pool_global_ids = np.array(audit[p]["pool_global_ids"], dtype=np.int64)
        N, K_p = costs.shape
        base_rank = np.argsort(np.argsort(-logits, axis=1), axis=1).astype(np.float32) / max(K_p - 1, 1)
        base_norm = logits - logits.mean(axis=1, keepdims=True)
        base_margin_to_sbs = base_norm - base_norm[:, sbs_idx:sbs_idx + 1]  # (N, K_p)
        vote_p = vote[p]
        cost_p = cost[p]
        v_norm = vote_p / (vote_p.sum(axis=1, keepdims=True) + 1e-6)
        c_norm = cost_p / (cost_p.std(axis=1, keepdims=True) + 1e-6)
        c_norm = c_norm - c_norm.mean(axis=1, keepdims=True)
        # Broadcast arm-wise static features
        freq = arm_stats[p]["oracle_freq"][None, :].repeat(N, axis=0)
        regret = arm_stats[p]["mean_regret"][None, :].repeat(N, axis=0)
        arm_feat_np = np.stack([
            base_norm,
            base_rank,
            base_margin_to_sbs,
            v_norm,
            c_norm,
            freq,
            regret,
        ], axis=-1).astype(np.float32)  # (N, K_p, 7)
        # Target action: 0 = KEEP_SBS, else oracle arm + 1
        oracle_cost = costs[np.arange(N), oracle]
        sbs_cost = costs[:, sbs_idx]
        regret_gap_pct = (sbs_cost - oracle_cost) / (np.abs(sbs_cost) + 1e-9)
        # KEEP when oracle==sbs or the regret gap is tiny
        keep_mask = (oracle == sbs_idx) | (regret_gap_pct < 0.001)  # <0.1%
        target = np.where(keep_mask, 0, oracle + 1).astype(np.int64)
        out[p] = dict(
            arm_feats=torch.from_numpy(arm_feat_np).float().to(device),
            pool_global_ids=torch.from_numpy(pool_global_ids).long().to(device),
            pid=torch.full((N,), pid_map[p], dtype=torch.long, device=device),
            inst_feats=torch.from_numpy(inst_feats[p]).float().to(device),
            pool_h=torch.from_numpy(pooled).float().to(device),
            cbits=torch.zeros(N, K_CBITS, device=device),  # filled below from cbits registry
            costs=torch.from_numpy(costs).float().to(device),
            oracle=torch.from_numpy(oracle).long().to(device),
            target=torch.from_numpy(target).long().to(device),
            base_logits=torch.from_numpy(logits).float().to(device),
            sbs_idx=sbs_idx,
            K_p=K_p,
            N=N,
            keep_mask=torch.from_numpy(keep_mask.astype(np.float32)).float().to(device),
        )
    # Fill cbits by looking up the data layer once (0 for TSP/ATSP, else populated)
    from .registry import constraint_bits
    for p in PROBLEMS:
        cbits_vec = torch.tensor(constraint_bits(p), dtype=torch.float32, device=device)
        out[p]["cbits"] = cbits_vec.unsqueeze(0).expand(out[p]["N"], -1).contiguous()
    return out


# ---------------------------------------------------------------------------
#  Loss helpers
# ---------------------------------------------------------------------------

def pairwise_cost_loss(arm_scores, costs, gap_min: float = 0.0):
    """Gap-weighted pairwise hinge so score_i > score_j when cost_i < cost_j.

    costs[b, k] = tour cost (lower is better).
    For pair (i, j):
        diff_cost  = cost_i - cost_j      ; NEGATIVE when i better
        diff_score = score_i - score_j    ; we want POSITIVE when i better
        gap        = (cost_j - cost_i) / best_cost_in_instance  ; POSITIVE when i better
    Include a pair in the loss iff gap > gap_min (skip near-ties).
    Weight each included pair by gap (reviewer's 'gap-weighted pairwise').
    Minimize softplus(-(score_i - score_j)) -> drives score_i to exceed score_j.
    """
    B, K_p = arm_scores.shape
    diff_cost = costs.unsqueeze(2) - costs.unsqueeze(1)     # (B, K_p, K_p): pos means i worse
    diff_score = arm_scores.unsqueeze(2) - arm_scores.unsqueeze(1)
    best_cost = costs.min(dim=1, keepdim=True).values.clamp_min(1e-9)  # (B, 1)
    gap = (-diff_cost) / best_cost.unsqueeze(2)             # (B, K_p, K_p): pos when i better
    mask = (gap > gap_min).float()
    weight = gap.clamp_min(0.0) * mask
    pair = F.softplus(-diff_score)
    denom = weight.sum().clamp_min(1.0)
    return (pair * weight).sum() / denom


def kl_to_base_softmax(arm_scores, base_logits, tau=2.0):
    """KL(student || teacher) over pool only, teacher = softmax(base/tau)."""
    t = torch.softmax(base_logits / tau, dim=1)
    s_logp = torch.log_softmax(arm_scores / tau, dim=1)
    return -(t * s_logp).sum(dim=1).mean() * (tau ** 2)


# ---------------------------------------------------------------------------
#  Sampling (reviewer §4)
# ---------------------------------------------------------------------------

def sample_indices(d, n_samples, mix, rng):
    """Sample indices from a single-problem tensor dict with the 4-way mix:
         rand / base_correct / base_wrong_oracle_in_pool / zp_or_high_regret
    All categories "oracle_in_pool" by construction (dataset always has oracle in K_p).
    zp / high-regret: instances whose oracle != any base pick arm OR sbs-vs-oracle-gap > 5%.
    """
    N = d["N"]
    K_p = d["K_p"]
    base_pick = d["base_logits"].argmax(dim=1).cpu().numpy()
    oracle = d["oracle"].cpu().numpy()
    costs = d["costs"].cpu().numpy()
    sbs_idx = d["sbs_idx"]
    base_correct = base_pick == oracle
    base_wrong = ~base_correct
    # zp_or_high_regret: base's support is what it picks; oracle-arm not in that support -> zp
    base_support = set(np.unique(base_pick).tolist())
    zp_mask = ~np.isin(oracle, list(base_support))
    oracle_cost = costs[np.arange(N), oracle]
    sbs_cost = costs[:, sbs_idx]
    gap = (sbs_cost - oracle_cost) / (np.abs(sbs_cost) + 1e-9)
    high_regret = gap > 0.05
    hard_mask = zp_mask | high_regret
    buckets = {
        "rand": np.arange(N),
        "bc": np.where(base_correct)[0],
        "bw": np.where(base_wrong)[0],
        "hard": np.where(hard_mask)[0],
    }
    fracs = {"rand": mix[0], "bc": mix[1], "bw": mix[2], "hard": mix[3]}
    out = []
    for k, frac in fracs.items():
        n_k = int(round(n_samples * frac))
        if n_k == 0 or len(buckets[k]) == 0:
            continue
        chosen = rng.choice(buckets[k], size=n_k, replace=len(buckets[k]) < n_k)
        out.append(chosen)
    if not out:
        return np.arange(min(n_samples, N))
    out = np.concatenate(out)
    if len(out) > n_samples:
        out = rng.choice(out, size=n_samples, replace=False)
    return out


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_all(model, data_t):
    """Returns per-problem (keep_scores, arm_scores) tensors."""
    out = {}
    for p in PROBLEMS:
        d = data_t[p]
        arm_sc, keep_sc = model(d["arm_feats"], d["pool_global_ids"], d["pid"],
                                 d["inst_feats"], d["pool_h"], d["cbits"])
        out[p] = {"arm_scores": arm_sc.cpu(), "keep_score": keep_sc.cpu()}
    return out


FAMILY_OF = {
    "TSP": "TSP", "ATSP": "ATSP", "CVRP": "CVRP",
    **{p: "MVRP" for p in PROBLEMS[3:]},
}


def fit_thresholds(val_scores, val_data, grid, scope: str = "problem"):
    """Find θ that maximizes val top1.  scope: 'problem' (18 θ) or 'family' (4 θ) or 'global' (1 θ).

    Returns dict {problem_name -> {'theta': float, 'val_top1': float}} so apply_thresholds is uniform.
    In 'family' mode, all problems within a family share the same θ (chosen to maximize the family's
    pooled-instance top1, which averages down the per-problem θ noise).
    """
    thresholds = {}
    if scope == "problem":
        for p in PROBLEMS:
            arm_sc = val_scores[p]["arm_scores"].numpy()
            keep_sc = val_scores[p]["keep_score"].numpy()
            oracle = val_data[p]["oracle"].cpu().numpy()
            sbs_idx = val_data[p]["sbs_idx"]
            best = None
            for theta in grid:
                margin = keep_sc - arm_sc.max(axis=1)
                keep = margin >= theta
                pick = np.where(keep, sbs_idx, arm_sc.argmax(axis=1))
                top1 = float((pick == oracle).mean())
                if best is None or top1 > best[0]:
                    best = (top1, float(theta))
            thresholds[p] = {"theta": best[1], "val_top1": best[0]}
        return thresholds

    # 'family' or 'global': group problems together and share θ
    if scope == "global":
        groups = {"ALL": list(PROBLEMS)}
    else:  # family
        groups = {}
        for p in PROBLEMS:
            groups.setdefault(FAMILY_OF[p], []).append(p)

    for gname, members in groups.items():
        best = None
        for theta in grid:
            hits, total = 0, 0
            for p in members:
                arm_sc = val_scores[p]["arm_scores"].numpy()
                keep_sc = val_scores[p]["keep_score"].numpy()
                oracle = val_data[p]["oracle"].cpu().numpy()
                sbs_idx = val_data[p]["sbs_idx"]
                margin = keep_sc - arm_sc.max(axis=1)
                keep = margin >= theta
                pick = np.where(keep, sbs_idx, arm_sc.argmax(axis=1))
                hits += int((pick == oracle).sum())
                total += len(oracle)
            top1 = hits / max(total, 1)
            if best is None or top1 > best[0]:
                best = (top1, float(theta))
        # Backfill per-problem val_top1 under the chosen θ
        for p in members:
            arm_sc = val_scores[p]["arm_scores"].numpy()
            keep_sc = val_scores[p]["keep_score"].numpy()
            oracle = val_data[p]["oracle"].cpu().numpy()
            sbs_idx = val_data[p]["sbs_idx"]
            margin = keep_sc - arm_sc.max(axis=1)
            keep = margin >= best[1]
            pick = np.where(keep, sbs_idx, arm_sc.argmax(axis=1))
            p_top1 = float((pick == oracle).mean())
            thresholds[p] = {"theta": best[1], "val_top1": p_top1, "family": gname}
    return thresholds


def apply_thresholds(scores, data_t, thresholds):
    picks = {}
    for p in PROBLEMS:
        arm_sc = scores[p]["arm_scores"].numpy()
        keep_sc = scores[p]["keep_score"].numpy()
        sbs_idx = data_t[p]["sbs_idx"]
        theta = thresholds[p]["theta"]
        margin = keep_sc - arm_sc.max(axis=1)
        keep = margin >= theta
        pick = np.where(keep, sbs_idx, arm_sc.argmax(axis=1))
        picks[p] = pick.astype(np.int64)
    return picks


def bootstrap_vs_sbs(picks, data_t, n_boot=10000, seed=0):
    """Paired bootstrap of macro top1 Δ: method - SBS-constant."""
    rng = np.random.RandomState(seed)
    Ns = {p: data_t[p]["N"] for p in PROBLEMS}
    deltas = np.zeros(n_boot, dtype=np.float32)
    method_correct = {p: (picks[p] == data_t[p]["oracle"].cpu().numpy()).astype(np.float32)
                      for p in PROBLEMS}
    sbs_correct = {p: (np.full(Ns[p], data_t[p]["sbs_idx"]) == data_t[p]["oracle"].cpu().numpy()).astype(np.float32)
                   for p in PROBLEMS}
    for b in range(n_boot):
        m = 0.0; s = 0.0
        for p in PROBLEMS:
            idx = rng.randint(0, Ns[p], size=Ns[p])
            m += method_correct[p][idx].mean()
            s += sbs_correct[p][idx].mean()
        deltas[b] = (m - s) / len(PROBLEMS)
    return {
        "mean": float(deltas.mean()),
        "lo": float(np.percentile(deltas, 2.5)),
        "hi": float(np.percentile(deltas, 97.5)),
        "p_gt_0": float((deltas > 0).mean()),
        "significant": bool(np.percentile(deltas, 2.5) > 0),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r18-ckpt", required=True)
    ap.add_argument("--audit", default="code/unified_selector/runs/audit.json")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    # Retrieval
    ap.add_argument("--retrieval-k", type=int, default=32)
    ap.add_argument("--retrieval-tau", type=float, default=0.1)
    ap.add_argument("--n-folds", type=int, default=5)
    # Model
    ap.add_argument("--rerank-d", type=int, default=128, help="Hidden dim of scorer MLPs.")
    ap.add_argument("--rerank-dropout", type=float, default=0.1)
    # Training
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--keep-ce-weight", type=float, default=1.0)
    ap.add_argument("--pairwise-cost-weight", type=float, default=0.5)
    ap.add_argument("--kl-weight", type=float, default=0.3)
    ap.add_argument("--kl-tau", type=float, default=2.0)
    ap.add_argument("--sample-mix", type=str, default="0.4,0.25,0.25,0.10",
                    help="rand, base_correct, base_wrong, hard (zp/high-regret) fractions.")
    # Threshold grid
    ap.add_argument("--threshold-grid", type=str, default="-0.5,0.5,21")
    ap.add_argument("--threshold-scope", choices=["problem", "family", "global"], default="problem",
                    help="Per-problem (18 θ), per-family (4 θ: TSP/ATSP/CVRP/MVRP), or single global θ.")
    ap.add_argument("--pairwise-gap-min", type=float, default=0.0,
                    help="Minimum gap (pct of best cost) for a pair to enter pairwise loss. 0.001 = 0.1%% filter.")
    ap.add_argument("--n-boot", type=int, default=10000)
    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="selector")
    ap.add_argument("--wandb-entity", default="yjkds-southern-university-of-science-technology")
    ap.add_argument("--wandb-tag", default="R42S")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    audit = json.loads(Path(args.audit).read_text())
    pid_map = {p: i for i, p in enumerate(PROBLEMS)}
    mix = tuple(float(x) for x in args.sample_mix.split(","))
    assert len(mix) == 4, "--sample-mix must have 4 fractions"

    # 1. Load R18 and encode all splits
    print(f"[r42s] loading R18 base {args.r18_ckpt}")
    base_model = load_base(args.r18_ckpt, device)
    print(f"[r42s] encoding train / val / test")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    # 2. Retrieval priors
    print(f"[r42s] cross-fit train priors (k={args.retrieval_k}, tau={args.retrieval_tau})")
    train_vote, train_cost = cross_fit_priors(train_enc, args.retrieval_k, args.retrieval_tau,
                                               args.n_folds, seed=args.seed)
    print(f"[r42s] val/test priors from full train")
    val_vote = compute_retrieval_prior(train_enc, val_enc, k=args.retrieval_k,
                                        temperature=args.retrieval_tau)
    val_cost = compute_cost_prior(train_enc, val_enc, k=args.retrieval_k,
                                   temperature=args.retrieval_tau, normalize="rank")
    test_vote = compute_retrieval_prior(train_enc, test_enc, k=args.retrieval_k,
                                         temperature=args.retrieval_tau)
    test_cost = compute_cost_prior(train_enc, test_enc, k=args.retrieval_k,
                                    temperature=args.retrieval_tau, normalize="rank")

    # 3. Arm behavioural stats (from train)
    arm_stats = compute_arm_behavior_stats(train_enc)
    inst_train = build_inst_features(train_enc)
    inst_val = build_inst_features(val_enc)
    inst_test = build_inst_features(test_enc)

    # 4. Assemble per-problem tensor dicts
    train_t = assemble_tensors(train_enc, train_vote, train_cost, inst_train, arm_stats,
                                audit, device, pid_map)
    val_t = assemble_tensors(val_enc, val_vote, val_cost, inst_val, arm_stats,
                              audit, device, pid_map)
    test_t = assemble_tensors(test_enc, test_vote, test_cost, inst_test, arm_stats,
                               audit, device, pid_map)
    d_arm_feats = train_t[PROBLEMS[0]]["arm_feats"].shape[-1]
    d_inst_feats = train_t[PROBLEMS[0]]["inst_feats"].shape[-1]
    d_pool_h = train_t[PROBLEMS[0]]["pool_h"].shape[-1]

    # 5. Model
    model = RerankHead(d_arm_feats=d_arm_feats, d_inst_feats=d_inst_feats,
                        d_pool_h=d_pool_h, d_hidden=args.rerank_d,
                        dropout=args.rerank_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                              betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print(f"[r42s] model params = {sum(p.numel() for p in model.parameters()):,}")

    # W&B
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=save_dir.name, tags=[args.wandb_tag], config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            args.wandb = False

    # 6. Train loop
    rng = np.random.default_rng(args.seed)
    best_val_macro = -1.0
    best_thresholds = None
    best_state = None
    step = 0
    t0 = time.time()
    per_problem_batch = max(1, args.batch_size // len(PROBLEMS))
    for ep in range(args.epochs):
        model.train()
        for sp in range(args.steps_per_epoch):
            opt.zero_grad()
            total_loss = 0.0
            per_problem_loss = {}
            for p in PROBLEMS:
                d = train_t[p]
                idx = sample_indices(d, per_problem_batch, mix, rng)
                idx_t = torch.from_numpy(idx).long().to(device)
                arm_feats = d["arm_feats"].index_select(0, idx_t)
                pid = d["pid"].index_select(0, idx_t)
                inst_feats = d["inst_feats"].index_select(0, idx_t)
                pool_h = d["pool_h"].index_select(0, idx_t)
                cbits = d["cbits"].index_select(0, idx_t)
                costs = d["costs"].index_select(0, idx_t)
                target = d["target"].index_select(0, idx_t)
                base_logits_pool = d["base_logits"].index_select(0, idx_t)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    arm_sc, keep_sc = model(arm_feats, d["pool_global_ids"], pid,
                                             inst_feats, pool_h, cbits)
                    scores = torch.cat([keep_sc.unsqueeze(1), arm_sc], dim=1)  # (B, K_p+1)
                    ce = F.cross_entropy(scores, target)
                    pw = pairwise_cost_loss(arm_sc, costs, gap_min=args.pairwise_gap_min)
                    kl = kl_to_base_softmax(arm_sc, base_logits_pool, tau=args.kl_tau)
                    loss_p = (args.keep_ce_weight * ce
                              + args.pairwise_cost_weight * pw
                              + args.kl_weight * kl) / len(PROBLEMS)
                scaler.scale(loss_p).backward()
                per_problem_loss[p] = float(ce.item() + pw.item() + kl.item())
                total_loss += float(loss_p.item()) * len(PROBLEMS)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            step += 1
            if step % 20 == 0:
                mean_loss = total_loss / len(PROBLEMS)
                print(f"ep{ep} step{step} loss={mean_loss:.4f}", flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": mean_loss, "step": step})
        # Eval on val
        model.eval()
        val_scores = infer_all(model, val_t)
        grid_lo, grid_hi, grid_n = args.threshold_grid.split(",")
        grid = np.linspace(float(grid_lo), float(grid_hi), int(grid_n))
        thresholds = fit_thresholds(val_scores, val_t, grid, scope=args.threshold_scope)
        val_macro = float(np.mean([thresholds[p]["val_top1"] for p in PROBLEMS]))
        print(f"[ep{ep}] val_macro_top1 = {val_macro:.4f}")
        if args.wandb:
            import wandb
            wandb.log({"val/macro_top1": val_macro, "epoch": ep})
            for p in PROBLEMS:
                wandb.log({f"val_{p}/top1": thresholds[p]["val_top1"],
                           f"val_{p}/theta": thresholds[p]["theta"], "epoch": ep})
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_thresholds = thresholds
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": {k: v for k, v in best_state.items()},
                        "args": vars(args),
                        "thresholds": best_thresholds,
                        "val_macro_top1": best_val_macro,
                        "d_arm_feats": d_arm_feats, "d_inst_feats": d_inst_feats,
                        "d_pool_h": d_pool_h},
                       save_dir / "best.pt")
            print(f"  -> saved best (val={best_val_macro:.4f})")

    # 7. Test eval with best thresholds
    print(f"[r42s] loading best (val={best_val_macro:.4f}); running test")
    model.load_state_dict(best_state)
    model.eval()
    test_scores = infer_all(model, test_t)
    picks = apply_thresholds(test_scores, test_t, best_thresholds)
    per_problem = {}
    for p in PROBLEMS:
        o = test_t[p]["oracle"].cpu().numpy()
        sbs = test_t[p]["sbs_idx"]
        method_top1 = float((picks[p] == o).mean())
        sbs_top1 = float((np.full(len(o), sbs) == o).mean())
        per_problem[p] = {
            "method_top1": method_top1,
            "sbs_top1": sbs_top1,
            "delta": method_top1 - sbs_top1,
            "theta": best_thresholds[p]["theta"],
            "n": int(len(o)),
        }
    macro_method = float(np.mean([v["method_top1"] for v in per_problem.values()]))
    macro_sbs = float(np.mean([v["sbs_top1"] for v in per_problem.values()]))
    boot = bootstrap_vs_sbs(picks, test_t, n_boot=args.n_boot, seed=args.seed)
    elapsed = time.time() - t0

    summary = dict(
        elapsed_sec=elapsed,
        best_val_macro_top1=best_val_macro,
        thresholds=best_thresholds,
        test_macro_method=macro_method,
        test_macro_sbs=macro_sbs,
        test_macro_delta=macro_method - macro_sbs,
        bootstrap=boot,
        per_problem=per_problem,
    )
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    lines = [
        f"# R42S KEEP_SBS + full-pool reranker — {save_dir.name}",
        f"",
        f"- Best val macro_top1 = **{best_val_macro:.4f}**",
        f"- Test: SBS {macro_sbs:.4f} → method {macro_method:.4f} "
        f"(Δ = **{macro_method - macro_sbs:+.4f}**, 95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}], "
        f"p(Δ>0) = {boot['p_gt_0']:.3f}, sig = {boot['significant']})",
        f"",
        "## Per-problem",
        "| Problem | SBS top1 | Method top1 | Δ | θ |",
        "|:---|---:|---:|---:|---:|",
    ]
    for p in PROBLEMS:
        v = per_problem[p]
        lines.append(
            f"| {p} | {v['sbs_top1']:.3f} | {v['method_top1']:.3f} | {v['delta']:+.3f} | {v['theta']:+.3f} |"
        )
    (save_dir / "report.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
