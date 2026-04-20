"""Plan H — Correction head trained on top of R18 using retrieval prior as input.

Instead of test-time blending with fixed α, we train a small head that takes
(pooled_embedding, base_logits, retrieval_prior_probs, features) and outputs
per-arm correction deltas. Final logit = base_logit + correction.

Training: cross-entropy + soft risk loss on oracle.
Advantage: the correction head LEARNS when to use retrieval vs base, instead
of assuming a fixed blend.

Usage:
  python -m code.unified_selector.correction_train \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R29_correction \
    --k 32 --temperature 0.1 --epochs 50 \
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


class CorrectionHead(nn.Module):
    """Per-arm correction logit from (pooled_emb, prior, base_logit) features.

    For each instance:
      - Input: pooled_emb (d,), and per-arm (base_logit, prior_prob, rank_base, rank_prior)
      - Output: per-arm correction delta
      - Final logit = base_logit + delta

    Architecture: shared MLP on instance-level context, plus per-arm MLP
    taking (arm_features, instance_context) → correction.
    """
    def __init__(self, d_pooled=128, d_ctx=64, d_hidden=128, n_arm_feats=6, dropout=0.1):
        super().__init__()
        self.ctx_mlp = nn.Sequential(
            nn.Linear(d_pooled, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_ctx), nn.GELU(),
        )
        self.arm_mlp = nn.Sequential(
            nn.Linear(n_arm_feats + d_ctx, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        # Scale delta so it doesn't immediately dominate
        self.delta_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, pooled, base_logits, prior):
        """
        pooled: (B, d)
        base_logits: (B, K_p)
        prior: (B, K_p) probability distribution

        Returns: (B, K_p) correction delta.
        """
        B, K = base_logits.shape
        ctx = self.ctx_mlp(pooled)                             # (B, d_ctx)
        ctx_bcast = ctx.unsqueeze(1).expand(-1, K, -1)          # (B, K, d_ctx)
        # Per-arm features
        base_norm = base_logits - base_logits.mean(dim=1, keepdim=True)
        prior_eps = prior + 1e-6
        prior_log = torch.log(prior_eps) - torch.log(prior_eps).mean(dim=1, keepdim=True)
        # Rank features
        rank_base = base_logits.argsort(dim=1).argsort(dim=1).float() / max(K - 1, 1)
        rank_prior = prior.argsort(dim=1).argsort(dim=1).float() / max(K - 1, 1)
        # Margin per arm
        top1 = base_norm.max(dim=1, keepdim=True).values
        gap_base = top1 - base_norm                              # (B, K)
        arm_feats = torch.stack([base_norm, prior_log, rank_base, rank_prior,
                                  gap_base, prior_eps], dim=-1)  # (B, K, 6)
        combined = torch.cat([arm_feats, ctx_bcast], dim=-1)      # (B, K, 6+d_ctx)
        delta = self.arm_mlp(combined).squeeze(-1)                # (B, K)
        return delta * self.delta_scale


def prepare_tensors(enc_dict, priors, device):
    out = {}
    for p in PROBLEMS:
        d = enc_dict[p]
        pooled = torch.from_numpy(d["pooled"]).float().to(device)
        costs = torch.from_numpy(d["costs"]).float().to(device)
        logits = torch.from_numpy(d["logits"]).float().to(device)
        oracle = torch.from_numpy(d["oracle"]).long().to(device)
        prior = torch.from_numpy(priors[p]).float().to(device)
        # Normalize prior
        prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-6)
        out[p] = dict(pooled=pooled, costs=costs, logits=logits, oracle=oracle, prior=prior)
    return out


def eval_metrics(head, data_t, risk_cap=0.05):
    """Compute loss + top1 over all problems."""
    total_loss = 0.0
    metrics = {}
    for p in PROBLEMS:
        d = data_t[p]
        delta = head(d["pooled"], d["logits"], d["prior"])
        final_logits = d["logits"] + delta
        # Soft CE on oracle
        log_p = F.log_softmax(final_logits, dim=1)
        ce = -log_p.gather(1, d["oracle"].unsqueeze(1)).squeeze(1)
        # Soft risk (cost-weighted)
        soft_p = F.softmax(final_logits / 0.2, dim=1)
        # Cost normalized per instance
        sbs_mean = d["costs"].mean(dim=0)  # K_p
        sbs_idx = sbs_mean.argmin()
        sbs_cost = d["costs"][:, sbs_idx]
        risk = ((soft_p * d["costs"]).sum(dim=1) - sbs_cost) / (sbs_cost.abs() + 1e-6)
        risk = risk.clamp(-risk_cap, risk_cap)
        loss_p = ce.mean() + 0.5 * risk.mean()
        total_loss = total_loss + loss_p
        with torch.no_grad():
            pick = final_logits.argmax(dim=1)
            top1 = float((pick == d["oracle"]).float().mean())
            mean_cost = float(d["costs"].gather(1, pick.unsqueeze(1)).mean())
            metrics[p] = dict(top1=top1, mean_cost=mean_cost,
                              delta_mean=float(delta.abs().mean()))
    macro_top1 = float(np.mean([m["top1"] for m in metrics.values()]))
    return total_loss / len(PROBLEMS), macro_top1, metrics


def get_picks_tensors(head, data_t):
    picks = {}
    for p in PROBLEMS:
        d = data_t[p]
        delta = head(d["pooled"], d["logits"], d["prior"])
        final_logits = d["logits"] + delta
        pick = final_logits.argmax(dim=1)
        picks[p] = dict(picks=pick.cpu().numpy().astype(np.int32),
                         costs=d["costs"].cpu().numpy(),
                         oracle=d["oracle"].cpu().numpy().astype(np.int32))
    return picks


def get_base_picks(data_t):
    picks = {}
    for p in PROBLEMS:
        d = data_t[p]
        pick = d["logits"].argmax(dim=1)
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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--d-ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)

    print(f"[correction] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[correction] encoding all splits")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    print(f"[correction] computing priors (k={args.k})")
    train_priors = compute_retrieval_prior(train_enc, train_enc, k=args.k, temperature=args.temperature)
    val_priors = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    test_priors = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)

    train_t = prepare_tensors(train_enc, train_priors, device)
    val_t = prepare_tensors(val_enc, val_priors, device)
    test_t = prepare_tensors(test_enc, test_priors, device)

    d_pooled = train_t[PROBLEMS[0]]["pooled"].shape[1]
    head = CorrectionHead(d_pooled=d_pooled, d_hidden=args.d_hidden, d_ctx=args.d_ctx).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"[correction] params={sum(p.numel() for p in head.parameters())}")

    best_val = -1.0
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        loss, train_top1, _ = eval_metrics(head, train_t)
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            _, val_top1, _ = eval_metrics(head, val_t)
        if val_top1 > best_val:
            best_val = val_top1
            best_state = {k: v.detach().clone().cpu() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} train_loss={float(loss):.4f} train_top1={train_top1:.4f} val_top1={val_top1:.4f} (best={best_val:.4f})")

    # Test
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        _, test_top1, test_metrics = eval_metrics(head, test_t)
        corrected_picks = get_picks_tensors(head, test_t)
    base_picks = get_base_picks(test_t)
    boot = bootstrap_vs_base({1.0: base_picks, 0.0: corrected_picks},
                              ref_alpha=1.0, n_boot=args.n_boot)[0.0]

    lines = [f"# Correction head ({Path(args.base_ckpt).name}, k={args.k})\n",
             f"Epochs: {args.epochs}, lr: {args.lr}, d_hidden: {args.d_hidden}\n",
             f"Best val macro_top1: **{best_val:.4f}**",
             f"Test macro_top1: **{test_top1:.4f}**",
             "",
             "| metric | value | 95% CI |",
             "|:---|---:|---:|",
             f"| Δtop1 vs base | {boot['top1_mean']:+.4f} | [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}] |",
             f"| Δcost% vs base | {boot['cost_mean']:+.4f}% | [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}] |",
             "",
             "## Per-problem",
             "| Problem | top1 | mean |delta| |",
             "|:---|---:|---:|"]
    for p, m in test_metrics.items():
        lines.append(f"| {p} | {m['top1']:.4f} | {m['delta_mean']:.3f} |")
    (out / "correction_test.md").write_text("\n".join(lines))
    (out / "correction_test.json").write_text(json.dumps(dict(
        best_val=best_val, test=test_top1,
        bootstrap=boot, test_metrics=test_metrics,
        args=vars(args),
    ), indent=2, default=float))
    torch.save(best_state, out / "head.pt")
    print("\n".join(lines))
    print(f"[elapsed] {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
