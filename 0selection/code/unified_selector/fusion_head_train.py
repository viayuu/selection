"""R38 — Train-time retrieval-aware fusion head.

Per oracle-pro R2:
Train a small supervised model that receives base logits + retrieval priors + features
as inputs, and outputs per-arm probabilities. Trained with cross-fitted retrieval
priors on TRAIN (split into folds, retrieve from other folds only).

Critical anti-leakage: retrieval priors for training examples must be computed
from OTHER training examples (not themselves). Otherwise model sees optimistic
retrieval signals.

Validation-locked pre-registration: one model choice on val, one test eval.

Usage:
  python -m code.unified_selector.fusion_head_train \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R38_fusion \
    --k 32 --temperature 0.1 --n-folds 5 \
    --device cuda:0
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import PROBLEMS
from .retrieval_prior import (
    load_base, encode_instances, compute_retrieval_prior, bootstrap_vs_base,
)
from .retrieval_cost import compute_cost_prior


def margin(x):
    s = np.sort(x, axis=1)[:, ::-1]
    return s[:, 0] - s[:, 1]


def cross_fit_retrieval(enc_dict, k=32, temperature=0.1, n_folds=5, eps=1e-6):
    """For each problem and each training instance, compute k-NN retrieval prior
    using ONLY the other (n_folds-1)/n_folds of training data.

    This is out-of-fold retrieval — prevents the model from seeing its own
    oracle as a neighbor.
    """
    priors_out = {}
    for p in PROBLEMS:
        d = enc_dict[p]
        pooled = d["pooled"]     # (N, d)
        N = pooled.shape[0]
        K_p = d["costs"].shape[1]
        # Shuffle indices for random folds
        rng = np.random.default_rng(0)
        perm = rng.permutation(N)
        folds = np.array_split(perm, n_folds)
        prior = np.zeros((N, K_p), dtype=np.float32)
        emb_norm = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + eps)
        for fi, fold_idx in enumerate(folds):
            # train_idx = everything NOT in this fold
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
            t_emb = emb_norm[train_idx]       # (Nt, d)
            q_emb = emb_norm[fold_idx]        # (Nq, d)
            sim = q_emb @ t_emb.T              # (Nq, Nt)
            topk_idx = np.argpartition(-sim, min(k, sim.shape[1]-1), axis=1)[:, :k]
            t_oracle = d["oracle"][train_idx]  # (Nt,)
            for i, qi in enumerate(fold_idx):
                neigh_oracles = t_oracle[topk_idx[i]]
                sims = sim[i, topk_idx[i]]
                w = np.exp(sims / temperature)
                w /= w.sum() + eps
                for j, arm in enumerate(neigh_oracles):
                    prior[qi, arm] += w[j]
        priors_out[p] = prior
    return priors_out


def build_features(enc_dict, priors_vote, priors_cost):
    """Per-instance feature vector incorporating base + retrieval signals.

    Returns per-problem: X (N, F), base_logits (N, K_p), priors_vote, priors_cost,
    costs (N, K_p), oracle (N,).
    """
    out = {}
    for p in PROBLEMS:
        d = enc_dict[p]
        logits = d["logits"]
        prior_v = priors_vote[p]
        prior_c = priors_cost[p]
        K_p = logits.shape[1]
        # Instance-level features (same for all arms)
        base_norm = logits - logits.mean(axis=1, keepdims=True)
        sorted_base = np.sort(base_norm, axis=1)[:, ::-1]
        base_margin = sorted_base[:, 0] - sorted_base[:, 1] if sorted_base.shape[1] >= 2 else np.zeros(sorted_base.shape[0])
        base_softmax = np.exp(base_norm - base_norm.max(axis=1, keepdims=True))
        base_softmax = base_softmax / base_softmax.sum(axis=1, keepdims=True)
        base_entropy = -(base_softmax * np.log(base_softmax + 1e-9)).sum(axis=1)
        pv_p = prior_v / (prior_v.sum(axis=1, keepdims=True) + 1e-6)
        pv_entropy = -(pv_p * np.log(pv_p + 1e-9)).sum(axis=1)
        base_top1 = logits.argmax(axis=1)
        prior_top1 = prior_v.argmax(axis=1)
        agree_vote = (base_top1 == prior_top1).astype(np.float32)
        # cost_prior: low = good, so argmin is top1
        cost_top1 = prior_c.argmin(axis=1)
        agree_cost = (base_top1 == cost_top1).astype(np.float32)
        agree_both = ((base_top1 == prior_top1) & (base_top1 == cost_top1)).astype(np.float32)
        vote_cost_agree = (prior_top1 == cost_top1).astype(np.float32)
        pv_max = pv_p.max(axis=1)
        # Per-instance features
        inst_feats = np.stack([base_margin, base_entropy, pv_entropy,
                                agree_vote, agree_cost, agree_both,
                                vote_cost_agree, pv_max], axis=1).astype(np.float32)
        out[p] = dict(
            base_logits=logits,
            prior_vote=prior_v,
            prior_cost=prior_c,
            costs=d["costs"],
            oracle=d["oracle"],
            inst_feats=inst_feats,
        )
    return out


class FusionHead(nn.Module):
    """Small head: per-arm score = W·[base_logit, prior_vote, -prior_cost, arm_rank_base, arm_rank_vote]
    + MLP(inst_feats, problem_emb) contribution.

    Output shape: (B, K_p) final logits.
    """
    def __init__(self, n_problems, d_inst_feats=8, d_arm_feats=6,
                 d_hidden=64, dropout=0.2):
        super().__init__()
        self.problem_emb = nn.Embedding(n_problems, 16)
        # Per-arm MLP
        self.arm_mlp = nn.Sequential(
            nn.Linear(d_arm_feats + 16 + d_inst_feats, d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        # Blend weight between base logit and new score (start close to pure base)
        self.base_weight = nn.Parameter(torch.tensor(2.0))  # sigmoid starts at 0.88

    def forward(self, base_logits, prior_vote, prior_cost, inst_feats, problem_ids):
        """
        base_logits: (B, K_p)
        prior_vote: (B, K_p) — probability-like
        prior_cost: (B, K_p) — cost-like (low is good)
        inst_feats: (B, d_inst)
        problem_ids: (B,) long

        Returns: (B, K_p) final logits.
        """
        B, K = base_logits.shape
        base_norm = base_logits - base_logits.mean(dim=1, keepdim=True)
        pv_norm = prior_vote / (prior_vote.sum(dim=1, keepdim=True) + 1e-6)
        pv_log = torch.log(pv_norm + 1e-6)
        pv_log = pv_log - pv_log.mean(dim=1, keepdim=True)
        # Normalize cost prior (negate so higher is better)
        pc_sign = -(prior_cost - prior_cost.mean(dim=1, keepdim=True))
        pc_sign = pc_sign / (pc_sign.std(dim=1, keepdim=True) + 1e-6)
        # Per-arm rank features
        rank_base = base_logits.argsort(dim=1).argsort(dim=1).float() / max(K - 1, 1)
        rank_vote = prior_vote.argsort(dim=1).argsort(dim=1).float() / max(K - 1, 1)
        # Per-arm feature stack
        arm_feats = torch.stack([base_norm, pv_log, pc_sign, rank_base, rank_vote,
                                  pv_norm], dim=-1)  # (B, K, 6)
        # Broadcast instance features
        inst_bcast = inst_feats.unsqueeze(1).expand(-1, K, -1)
        pe = self.problem_emb(problem_ids).unsqueeze(1).expand(-1, K, -1)
        mlp_in = torch.cat([arm_feats, inst_bcast, pe], dim=-1)
        delta = self.arm_mlp(mlp_in).squeeze(-1)                # (B, K)
        # Final = sigmoid(w)*base_logits + (1-sigmoid(w))*delta
        w = torch.sigmoid(self.base_weight)
        final = w * base_logits + (1 - w) * delta
        return final, w.item()


def prepare_tensors(fused_dict, device, problem_id_map):
    out = {}
    for p in PROBLEMS:
        d = fused_dict[p]
        out[p] = dict(
            base_logits=torch.from_numpy(d["base_logits"]).float().to(device),
            prior_vote=torch.from_numpy(d["prior_vote"]).float().to(device),
            prior_cost=torch.from_numpy(d["prior_cost"]).float().to(device),
            costs=torch.from_numpy(d["costs"]).float().to(device),
            oracle=torch.from_numpy(d["oracle"]).long().to(device),
            inst_feats=torch.from_numpy(d["inst_feats"]).float().to(device),
            pid=torch.full((d["base_logits"].shape[0],), problem_id_map[p], dtype=torch.long, device=device),
        )
    return out


def eval_fusion(fusion, data_t):
    total_loss = 0.0
    metrics = {}
    for p in PROBLEMS:
        d = data_t[p]
        final, w = fusion(d["base_logits"], d["prior_vote"], d["prior_cost"],
                           d["inst_feats"], d["pid"])
        log_p = F.log_softmax(final, dim=1)
        ce = -log_p.gather(1, d["oracle"].unsqueeze(1)).squeeze(1).mean()
        # Risk (optional)
        soft_p = F.softmax(final / 0.2, dim=1)
        sbs_idx = int(d["costs"].mean(dim=0).argmin())
        sbs_cost = d["costs"][:, sbs_idx]
        risk = ((soft_p * d["costs"]).sum(dim=1) - sbs_cost) / (sbs_cost.abs() + 1e-6)
        risk = risk.clamp(-0.05, 0.05).mean()
        loss_p = ce + 0.3 * risk
        total_loss = total_loss + loss_p
        with torch.no_grad():
            pick = final.argmax(dim=1)
            top1 = float((pick == d["oracle"]).float().mean())
            metrics[p] = dict(top1=top1, base_weight=w)
    return total_loss / len(PROBLEMS), float(np.mean([m["top1"] for m in metrics.values()])), metrics


def get_picks(fusion, data_t):
    picks = {}
    for p in PROBLEMS:
        d = data_t[p]
        final, _ = fusion(d["base_logits"], d["prior_vote"], d["prior_cost"],
                           d["inst_feats"], d["pid"])
        pick = final.argmax(dim=1)
        picks[p] = dict(picks=pick.cpu().numpy().astype(np.int32),
                         costs=d["costs"].cpu().numpy(),
                         oracle=d["oracle"].cpu().numpy().astype(np.int32))
    return picks


def get_base_picks(data_t):
    picks = {}
    for p in PROBLEMS:
        d = data_t[p]
        pick = d["base_logits"].argmax(dim=1)
        picks[p] = dict(picks=pick.cpu().numpy().astype(np.int32),
                         costs=d["costs"].cpu().numpy(),
                         oracle=d["oracle"].cpu().numpy().astype(np.int32))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)

    print(f"[fusion] loading base")
    base_model = load_base(args.base_ckpt, device)
    print(f"[fusion] encoding all splits")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    # CROSS-FITTED TRAIN priors: retrieve for each fold from the other folds
    print(f"[fusion] cross-fitting k={args.k} train priors (n_folds={args.n_folds})")
    train_priors_vote = cross_fit_retrieval(train_enc, k=args.k, temperature=args.temperature,
                                              n_folds=args.n_folds)
    # Cost priors: cross-fit similarly
    # For simplicity, reuse standard retrieval but with exclude-self fold
    # (we'll compute below using our own cross-fit helper)
    print(f"[fusion] cross-fitting train cost priors")
    def cross_fit_cost_retrieval(enc_dict, k, temperature, n_folds, normalize="rank", eps=1e-6):
        priors_out = {}
        for p in PROBLEMS:
            d = enc_dict[p]
            pooled = d["pooled"]
            N = pooled.shape[0]
            K_p = d["costs"].shape[1]
            rng = np.random.default_rng(1)
            perm = rng.permutation(N)
            folds = np.array_split(perm, n_folds)
            prior = np.zeros((N, K_p), dtype=np.float32)
            emb_norm = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + eps)
            for fi, fold_idx in enumerate(folds):
                train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
                t_emb = emb_norm[train_idx]
                q_emb = emb_norm[fold_idx]
                sim = q_emb @ t_emb.T
                topk_idx = np.argpartition(-sim, min(k, sim.shape[1]-1), axis=1)[:, :k]
                t_costs = d["costs"][train_idx]
                if normalize == "rank":
                    t_costs_n = np.argsort(np.argsort(t_costs, axis=1), axis=1).astype(np.float32)
                else:
                    t_costs_n = t_costs
                for i, qi in enumerate(fold_idx):
                    sims = sim[i, topk_idx[i]]
                    w = np.exp(sims / temperature)
                    w /= w.sum() + eps
                    prior[qi] = (w[:, None] * t_costs_n[topk_idx[i]]).sum(axis=0)
            priors_out[p] = prior
        return priors_out
    train_priors_cost = cross_fit_cost_retrieval(train_enc, args.k, args.temperature, args.n_folds)

    # For val/test, retrieve from full train
    print(f"[fusion] val/test priors from full train")
    val_priors_vote = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    val_priors_cost = compute_cost_prior(train_enc, val_enc, k=args.k, temperature=args.temperature,
                                          normalize="rank")
    test_priors_vote = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)
    test_priors_cost = compute_cost_prior(train_enc, test_enc, k=args.k, temperature=args.temperature,
                                           normalize="rank")

    train_feats = build_features(train_enc, train_priors_vote, train_priors_cost)
    val_feats = build_features(val_enc, val_priors_vote, val_priors_cost)
    test_feats = build_features(test_enc, test_priors_vote, test_priors_cost)

    problem_id_map = {p: i for i, p in enumerate(PROBLEMS)}
    train_t = prepare_tensors(train_feats, device, problem_id_map)
    val_t = prepare_tensors(val_feats, device, problem_id_map)
    test_t = prepare_tensors(test_feats, device, problem_id_map)

    fusion = FusionHead(n_problems=len(PROBLEMS), d_inst_feats=8,
                         d_hidden=args.d_hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"[fusion] params={sum(p.numel() for p in fusion.parameters())}")

    best_val_top1 = -1.0
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        fusion.train()
        loss, train_top1, train_m = eval_fusion(fusion, train_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
        opt.step()
        fusion.eval()
        with torch.no_grad():
            _, val_top1, _ = eval_fusion(fusion, val_t)
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_state = {k: v.detach().clone().cpu() for k, v in fusion.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            w = list(train_m.values())[0]["base_weight"]
            print(f"ep{ep} loss={float(loss):.4f} train_top1={train_top1:.4f} val_top1={val_top1:.4f} (best={best_val_top1:.4f}) w={w:.3f}")

    fusion.load_state_dict(best_state); fusion.eval()
    with torch.no_grad():
        _, test_top1, test_m = eval_fusion(fusion, test_t)
        fused_picks = get_picks(fusion, test_t)
    base_picks = get_base_picks(test_t)
    boot = bootstrap_vs_base({1.0: base_picks, 0.0: fused_picks},
                              ref_alpha=1.0, n_boot=args.n_boot)[0.0]

    # Flip counts
    wr = rw = 0
    for p in PROBLEMS:
        a = fused_picks[p]["picks"]; b = base_picks[p]["picks"]; o = fused_picks[p]["oracle"]
        changed = (a != b); bc = (b == o); nc = (a == o)
        wr += int((changed & ~bc & nc).sum())
        rw += int((changed & bc & ~nc).sum())

    # Zero-pick rescue on the TEST zp subset
    zp_base_correct = 0; zp_fused_correct = 0; total_zp = 0
    for p in PROBLEMS:
        K_p = fused_picks[p]["costs"].shape[1]
        unique_base = set(base_picks[p]["picks"].tolist())
        zp_arms = [k for k in range(K_p) if k not in unique_base]
        oracle = fused_picks[p]["oracle"]
        zp_mask = np.isin(oracle, zp_arms)
        n = int(zp_mask.sum()); total_zp += n
        zp_base_correct += int((base_picks[p]["picks"][zp_mask] == oracle[zp_mask]).sum())
        zp_fused_correct += int((fused_picks[p]["picks"][zp_mask] == oracle[zp_mask]).sum())

    base_top1 = float(np.mean([(base_picks[p]["picks"] == base_picks[p]["oracle"]).mean() for p in PROBLEMS]))
    fused_top1 = float(np.mean([(fused_picks[p]["picks"] == fused_picks[p]["oracle"]).mean() for p in PROBLEMS]))

    lines = [f"# Fusion head (R38)\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}, n_folds={args.n_folds}",
             f"Epochs: {args.epochs}, lr: {args.lr}, dropout: {args.dropout}\n",
             f"Best val top1: {best_val_top1:.4f}",
             f"Test top1: base {base_top1:.4f} → fusion {fused_top1:.4f} (Δ={fused_top1-base_top1:+.4f})",
             f"",
             f"## Bootstrap (paired, stratified per problem, n={args.n_boot})",
             f"- Δtop1: **{boot['top1_mean']:+.4f}** [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}]",
             f"- Δcost%: {boot['cost_mean']:+.4f}% [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}]",
             f"- **SIG** (2-sided 95%): {'YES' if boot['top1_lo'] > 0 else 'NO'}",
             f"",
             f"## Flip counts (test)",
             f"- wrong→right: {wr}",
             f"- right→wrong: {rw}",
             f"- wr/rw ratio: {wr/max(rw,1):.3f}",
             f"",
             f"## Zero-pick subset rescue",
             f"- zp mass: {total_zp} ({100*total_zp/18000:.1f}%)",
             f"- base zp top1: {zp_base_correct/max(total_zp,1):.4f}",
             f"- fused zp top1: {zp_fused_correct/max(total_zp,1):.4f}",
             f"- Δzp: {(zp_fused_correct - zp_base_correct)/max(total_zp,1):+.4f}"]
    (out / "fusion.md").write_text("\n".join(lines))
    (out / "fusion.json").write_text(json.dumps(dict(
        best_val=best_val_top1, test_top1=fused_top1, base_top1=base_top1,
        bootstrap=boot, flips=dict(wr=wr, rw=rw),
        zp=dict(total=total_zp, base_correct=zp_base_correct, fused_correct=zp_fused_correct),
        args=vars(args),
    ), indent=2, default=float))
    torch.save(best_state, out / "fusion.pt")
    print("\n".join(lines))
    print(f"[elapsed] {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
