"""Plan G — Learned per-instance gate combining base logits with retrieval prior.

For each instance, a small MLP predicts α ∈ [0,1] from
(pooled_embedding, base_margin, prior_confidence).
Final pick = argmax of (α · base_norm + (1-α) · prior_norm).

Training: minimize expected cost via differentiable softmax over blend.
Loss = softmax_temp(blend) · cost (smaller is better), with entropy regularization.

Advantage over fixed-α retrieval_prior: different instances may benefit from
different blend weights (high base margin → trust base, low margin → trust prior).

Usage:
  python -m code.unified_selector.gated_blend_train \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R28_gated \
    --k 32 --temperature 0.1 --epochs 10 \
    --device cuda:0
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import PROBLEMS
from .retrieval_prior import (
    load_base, encode_instances, compute_retrieval_prior, bootstrap_vs_base,
)


class GatePredictor(nn.Module):
    """Predicts per-instance α ∈ [0,1]."""
    def __init__(self, d_in=128, d_hidden=128, extra_features=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + extra_features, d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, g, extras):
        """g: (B, d), extras: (B, E). Returns α ∈ [0,1] shape (B,)."""
        x = torch.cat([g, extras], dim=-1)
        logit = self.net(x).squeeze(-1)
        return torch.sigmoid(logit)


def prepare_tensors(enc_dict, priors, device):
    """Concat per-problem (N, d), (N, K_p) tensors, return as torch tensors per problem."""
    out = {}
    for p in PROBLEMS:
        d = enc_dict[p]
        pooled = torch.from_numpy(d["pooled"]).float().to(device)  # (N, d)
        costs = torch.from_numpy(d["costs"]).float().to(device)    # (N, K_p)
        logits = torch.from_numpy(d["logits"]).float().to(device)  # (N, K_p)
        oracle = torch.from_numpy(d["oracle"]).long().to(device)
        prior = torch.from_numpy(priors[p]).float().to(device)     # (N, K_p)
        out[p] = dict(pooled=pooled, costs=costs, logits=logits, oracle=oracle, prior=prior)
    return out


def extras_features(logits, prior):
    """Per-instance features for the gate predictor.

    Returns (N, 4):
      - base_top1_logit (normalized)
      - base_top1-top2 margin
      - prior_top1_prob (entropy-based)
      - prior_entropy
    """
    eps = 1e-6
    base_norm = logits - logits.mean(dim=1, keepdim=True)
    top2 = torch.topk(base_norm, k=min(2, base_norm.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        margin = torch.zeros(base_norm.shape[0], device=base_norm.device)
    else:
        margin = top2[:, 0] - top2[:, 1]
    base_top1 = top2[:, 0]
    prior_p = prior / (prior.sum(dim=1, keepdim=True) + eps)
    prior_top1 = prior_p.max(dim=1).values
    prior_ent = -(prior_p * torch.log(prior_p + eps)).sum(dim=1)
    return torch.stack([base_top1, margin, prior_top1, prior_ent], dim=1)


def compute_loss_and_metrics(gate, enc_t, temp=1.0, entropy_weight=0.0):
    """For each problem: compute expected cost via softmax over blend."""
    total_loss = 0.0
    per_p_metrics = {}
    n_prob = 0
    for p in PROBLEMS:
        d = enc_t[p]
        N, K_p = d["logits"].shape
        log_prior = torch.log(d["prior"] + 1e-6)
        base_norm = d["logits"] - d["logits"].mean(dim=1, keepdim=True)
        prior_norm = log_prior - log_prior.mean(dim=1, keepdim=True)
        feats = extras_features(d["logits"], d["prior"])
        alpha = gate(d["pooled"], feats).unsqueeze(1)          # (N, 1)
        blend = alpha * base_norm + (1 - alpha) * prior_norm    # (N, K_p)
        # Soft pick
        p_soft = F.softmax(blend / temp, dim=1)
        # Expected cost
        exp_cost = (p_soft * d["costs"]).sum(dim=1)             # (N,)
        loss_p = exp_cost.mean()
        if entropy_weight > 0:
            ent = -(p_soft * torch.log(p_soft + 1e-9)).sum(dim=1).mean()
            loss_p = loss_p - entropy_weight * ent
        total_loss += loss_p
        n_prob += 1
        # Metrics
        with torch.no_grad():
            pick = blend.argmax(dim=1)
            top1 = (pick == d["oracle"]).float().mean()
            mean_c = d["costs"].gather(1, pick.unsqueeze(1)).squeeze(1).mean()
            per_p_metrics[p] = dict(top1=float(top1), mean_cost=float(mean_c),
                                     alpha_mean=float(alpha.mean()))
    return total_loss / max(n_prob, 1), per_p_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--soft-temp", type=float, default=0.5,
                    help="Temp for differentiable expected cost loss.")
    ap.add_argument("--entropy-weight", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)

    print(f"[gated] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)
    print(f"[gated] encoding train+val+test")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)
    print(f"[gated] computing retrieval priors for train/val/test")
    train_priors = compute_retrieval_prior(train_enc, train_enc, k=args.k,
                                            temperature=args.temperature)
    val_priors = compute_retrieval_prior(train_enc, val_enc, k=args.k,
                                          temperature=args.temperature)
    test_priors = compute_retrieval_prior(train_enc, test_enc, k=args.k,
                                           temperature=args.temperature)

    train_t = prepare_tensors(train_enc, train_priors, device)
    val_t = prepare_tensors(val_enc, val_priors, device)
    test_t = prepare_tensors(test_enc, test_priors, device)

    d_in = train_t[PROBLEMS[0]]["pooled"].shape[1]
    gate = GatePredictor(d_in=d_in, d_hidden=args.d_hidden).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_top1 = -1.0
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        gate.train()
        loss, _ = compute_loss_and_metrics(gate, train_t, temp=args.soft_temp,
                                            entropy_weight=args.entropy_weight)
        opt.zero_grad(); loss.backward(); opt.step()
        # Eval on val
        gate.eval()
        with torch.no_grad():
            _, val_m = compute_loss_and_metrics(gate, val_t, temp=args.soft_temp)
            macro_val = float(np.mean([m["top1"] for m in val_m.values()]))
            mean_alpha_val = float(np.mean([m["alpha_mean"] for m in val_m.values()]))
        if macro_val > best_val_top1:
            best_val_top1 = macro_val
            best_state = {k: v.detach().clone().cpu() for k, v in gate.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} train_loss={float(loss):.5f} val_top1={macro_val:.4f} mean_α={mean_alpha_val:.3f}")

    # Test with best val state
    gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        _, test_m = compute_loss_and_metrics(gate, test_t, temp=args.soft_temp)
        macro_test = float(np.mean([m["top1"] for m in test_m.values()]))
        mean_alpha_test = float(np.mean([m["alpha_mean"] for m in test_m.values()]))
    print(f"[done] best_val={best_val_top1:.4f}, test={macro_test:.4f}, mean_α_test={mean_alpha_test:.3f}")

    # Bootstrap test picks vs pure base
    base_picks = {p: dict(picks=test_t[p]["logits"].argmax(dim=1).cpu().numpy().astype(np.int32),
                           costs=test_t[p]["costs"].cpu().numpy(),
                           oracle=test_t[p]["oracle"].cpu().numpy().astype(np.int32)) for p in PROBLEMS}
    # Compute gated picks on test
    gated_picks = {}
    for p in PROBLEMS:
        d = test_t[p]
        log_prior = torch.log(d["prior"] + 1e-6)
        base_norm = d["logits"] - d["logits"].mean(dim=1, keepdim=True)
        prior_norm = log_prior - log_prior.mean(dim=1, keepdim=True)
        feats = extras_features(d["logits"], d["prior"])
        alpha = gate(d["pooled"], feats).unsqueeze(1).detach()
        blend = alpha * base_norm + (1 - alpha) * prior_norm
        gated_picks[p] = dict(picks=blend.argmax(dim=1).cpu().numpy().astype(np.int32),
                               costs=d["costs"].cpu().numpy(),
                               oracle=d["oracle"].cpu().numpy().astype(np.int32))
    picks_for_boot = {1.0: base_picks, 0.0: gated_picks}
    boot = bootstrap_vs_base(picks_for_boot, ref_alpha=1.0, n_boot=args.n_boot)[0.0]
    print(f"[boot] Δtop1 {boot['top1_mean']:+.4f} [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}], "
          f"Δcost {boot['cost_mean']:+.4f}% [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}]")

    lines = [f"# Gated-blend ({Path(args.base_ckpt).name}, k={args.k}, τ={args.temperature})\n",
             f"Epochs: {args.epochs}, lr: {args.lr}, soft_temp: {args.soft_temp}\n",
             f"Best val macro_top1: {best_val_top1:.4f}",
             f"Test macro_top1: {macro_test:.4f}",
             f"Mean α (test): {mean_alpha_test:.3f}",
             "",
             "| metric | value | 95% CI |",
             "|:---|---:|---:|",
             f"| Δtop1 vs base | {boot['top1_mean']:+.4f} | [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}] |",
             f"| Δcost% vs base | {boot['cost_mean']:+.4f}% | [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}] |"]
    (out / "gated_test.md").write_text("\n".join(lines))
    (out / "gated_test.json").write_text(json.dumps(dict(
        best_val=best_val_top1, test=macro_test,
        alpha_test=mean_alpha_test,
        bootstrap=boot, args=vars(args),
    ), indent=2, default=float))
    torch.save(best_state, out / "gate.pt")
    print("\n".join(lines))
    print(f"[elapsed] {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
