"""R33: Binary loss-aware gate.

Per oracle-pro R1.1:
Train a BINARY classifier `g(x) = 1[retrieval blend should override base]`,
using VALIDATION labels:
  - For each val instance, compute base top1 and retrieval-blended top1 for a fixed menu
    α ∈ {0.3, 0.4, 0.5}.
  - Label = +1 if blend changes wrong→right. Label = 0 if blend changes right→wrong.
  - (Unchanged cases are ignored or down-weighted.)

Features:
  - base top1-top2 margin
  - entropy of base softmax
  - entropy of retrieval prior
  - base top1 == retrieval top1 (bool)
  - retrieval support on base top1 (prior prob mass on base's choice)
  - retrieval support on base top2
  - retrieval support on base top3
  - whether retrieval top1 is a val-blind-spot arm (val pick freq < 1%)
  - retrieval entropy
  - neighbor concentration (top1 prior mass)
  - per-problem one-hot (or per-problem embedding)

Threshold t selected on val F1 or val Δtop1. Apply at test.

Usage:
  python -m code.unified_selector.binary_gate_train \
    --base-ckpt code/unified_selector/runs/R18_alltail/best.pt \
    --out code/unified_selector/runs/R33_binary_gate \
    --k 32 --temperature 0.1 --epochs 100 \
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


def compute_blind_spot_arms(val_enc, threshold=0.01):
    """For each problem, find arms whose val pick-frequency is < threshold."""
    bs = {}
    for p in PROBLEMS:
        logits = val_enc[p]["logits"]
        picks = logits.argmax(axis=1)
        K_p = logits.shape[1]
        freq = np.bincount(picks, minlength=K_p) / len(picks)
        bs[p] = np.where(freq < threshold)[0].tolist()
    return bs


def instance_features(base_logits, prior, blind_spot_arms_of_problem):
    """Per-instance feature vector (N, F)."""
    eps = 1e-6
    base_norm = base_logits - base_logits.mean(axis=1, keepdims=True)
    sorted_base = np.sort(base_norm, axis=1)[:, ::-1]
    base_margin = sorted_base[:, 0] - sorted_base[:, 1] if sorted_base.shape[1] >= 2 else np.zeros(sorted_base.shape[0])
    base_softmax = np.exp(base_norm - base_norm.max(axis=1, keepdims=True))
    base_softmax = base_softmax / (base_softmax.sum(axis=1, keepdims=True) + eps)
    base_entropy = -(base_softmax * np.log(base_softmax + eps)).sum(axis=1)
    prior_p = prior / (prior.sum(axis=1, keepdims=True) + eps)
    prior_entropy = -(prior_p * np.log(prior_p + eps)).sum(axis=1)
    base_top1 = base_logits.argmax(axis=1)
    prior_top1 = prior.argmax(axis=1)
    agree = (base_top1 == prior_top1).astype(np.float32)
    # Prior support on base top1
    N = base_logits.shape[0]
    idx = np.arange(N)
    prior_on_base_top1 = prior[idx, base_top1]
    # Prior support on base top2 (if exists)
    base_argsort = np.argsort(-base_logits, axis=1)
    top2 = base_argsort[:, min(1, base_logits.shape[1] - 1)]
    prior_on_base_top2 = prior[idx, top2]
    top3 = base_argsort[:, min(2, base_logits.shape[1] - 1)]
    prior_on_base_top3 = prior[idx, top3]
    # Is retrieval top1 a blind-spot arm?
    prior_top1_is_bs = np.isin(prior_top1, blind_spot_arms_of_problem).astype(np.float32)
    # Neighbor concentration = max prior
    prior_max = prior_p.max(axis=1)
    # Base-top1 frequency in the VAL set of the same problem — proxy for "how common is base's choice"
    feats = np.stack([
        base_margin,
        base_entropy,
        prior_entropy,
        agree,
        prior_on_base_top1,
        prior_on_base_top2,
        prior_on_base_top3,
        prior_top1_is_bs,
        prior_max,
    ], axis=1).astype(np.float32)
    return feats


def build_training_labels(enc, priors, alphas, blind_spots):
    """For each instance, find the best-α blended-pick across the menu, determine if
    blended-pick beats base-pick on that instance.

    Returns:
      features: (N_total, F)
      labels: (N_total,) float in {0.0, 0.5, 1.0}:
        1.0 if blend (at any α in menu) would change wrong→right
        0.0 if blend always changes right→wrong
        0.5 if unchanged (weights to 0 in training)
      weights: (N_total,) down-weight unchanged cases
      problem_ids: (N_total,)
    """
    all_feats = []; all_labels = []; all_weights = []; all_p = []
    for pi, p in enumerate(PROBLEMS):
        d = enc[p]
        logits = d["logits"]  # (N, K_p)
        costs = d["costs"]
        oracle = d["oracle"]
        prior = priors[p]
        # Compute base pick & correctness
        base_pick = logits.argmax(axis=1)
        base_correct = (base_pick == oracle)
        # For each α, compute blend pick
        base_norm = logits - logits.mean(axis=1, keepdims=True)
        log_prior = np.log(prior + 1e-6)
        prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
        # Instance label: 1 if ANY α gives wrong→right; 0 if ALL α give right→wrong
        N = logits.shape[0]
        best_blend_correct = np.zeros(N, dtype=bool)
        any_changes = np.zeros(N, dtype=bool)
        any_worsens = np.zeros(N, dtype=bool)
        any_improves = np.zeros(N, dtype=bool)
        for a in alphas:
            blend = a * base_norm + (1 - a) * prior_norm
            blend_pick = blend.argmax(axis=1)
            changed = blend_pick != base_pick
            blend_correct = (blend_pick == oracle)
            improved = changed & (~base_correct) & blend_correct     # wrong→right
            worsened = changed & base_correct & (~blend_correct)      # right→wrong
            any_changes = any_changes | changed
            any_improves = any_improves | improved
            any_worsens = any_worsens | worsened
            best_blend_correct = best_blend_correct | blend_correct
        # Labels
        label = np.full(N, 0.5, dtype=np.float32)
        weight = np.full(N, 0.5, dtype=np.float32)
        label[any_improves] = 1.0; weight[any_improves] = 1.0
        label[any_worsens & ~any_improves] = 0.0; weight[any_worsens & ~any_improves] = 1.0
        # Unchanged / tied
        weight[~any_changes] = 0.1
        feats = instance_features(logits, prior, np.array(blind_spots[p], dtype=int))
        # Add per-problem id (one-hot)
        onehot = np.zeros((N, len(PROBLEMS)), dtype=np.float32)
        onehot[:, pi] = 1.0
        feats = np.concatenate([feats, onehot], axis=1)
        all_feats.append(feats)
        all_labels.append(label)
        all_weights.append(weight)
        all_p.append(np.full(N, pi, dtype=np.int64))
    return (np.concatenate(all_feats, axis=0),
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_weights, axis=0),
            np.concatenate(all_p, axis=0))


class BinaryGate(nn.Module):
    def __init__(self, d_in, d_hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def apply_gate_on_split(enc, priors, blind_spots, gate, device,
                         alpha, threshold):
    """For each instance, predict gate prob; if > threshold, apply blend with alpha.
    Returns per-problem picks and stats."""
    picks_per_p = {}
    flips = dict(wr=0, rw=0, unchanged=0)
    for pi, p in enumerate(PROBLEMS):
        d = enc[p]
        logits = d["logits"]
        costs = d["costs"]
        oracle = d["oracle"]
        prior = priors[p]
        feats_np = instance_features(logits, prior, np.array(blind_spots[p], dtype=int))
        onehot = np.zeros((feats_np.shape[0], len(PROBLEMS)), dtype=np.float32)
        onehot[:, pi] = 1.0
        feats_np = np.concatenate([feats_np, onehot], axis=1)
        feats = torch.from_numpy(feats_np).to(device)
        with torch.no_grad():
            logit = gate(feats).cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logit))
        gate_on = prob > threshold
        # Blend pick
        base_norm = logits - logits.mean(axis=1, keepdims=True)
        log_prior = np.log(prior + 1e-6)
        prior_norm = log_prior - log_prior.mean(axis=1, keepdims=True)
        blend = alpha * base_norm + (1 - alpha) * prior_norm
        blend_pick = blend.argmax(axis=1)
        base_pick = logits.argmax(axis=1)
        picks = np.where(gate_on, blend_pick, base_pick)
        # Count flips for diagnostics
        changed = (picks != base_pick)
        base_correct = (base_pick == oracle)
        new_correct = (picks == oracle)
        flips["wr"] += int((changed & (~base_correct) & new_correct).sum())
        flips["rw"] += int((changed & base_correct & (~new_correct)).sum())
        flips["unchanged"] += int((~changed).sum())
        picks_per_p[p] = dict(picks=picks.astype(np.int32), costs=costs,
                              oracle=oracle.astype(np.int32))
    return picks_per_p, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--alpha-menu", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    ap.add_argument("--alpha-apply", type=float, default=0.4)
    ap.add_argument("--bs-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)

    print(f"[binary-gate] loading base {args.base_ckpt}")
    base_model = load_base(args.base_ckpt, device)

    print(f"[binary-gate] encoding splits")
    train_enc = encode_instances(base_model, "train", device)
    val_enc = encode_instances(base_model, "val", device)
    test_enc = encode_instances(base_model, "test", device)

    print(f"[binary-gate] computing priors k={args.k}")
    val_priors = compute_retrieval_prior(train_enc, val_enc, k=args.k, temperature=args.temperature)
    test_priors = compute_retrieval_prior(train_enc, test_enc, k=args.k, temperature=args.temperature)

    print(f"[binary-gate] finding val blind-spot arms (threshold={args.bs_threshold})")
    blind_spots = compute_blind_spot_arms(val_enc, threshold=args.bs_threshold)
    for p, bs in blind_spots.items():
        print(f"  {p}: |blind-spot|={len(bs)} arms")

    print(f"[binary-gate] building val training labels")
    val_feats, val_labels, val_weights, val_pids = build_training_labels(
        val_enc, val_priors, args.alpha_menu, blind_spots)
    print(f"  {len(val_feats)} val instances, label_pos_frac={val_labels.mean():.3f}")

    # Train binary gate on val (yes, on val — this is the point: supervise from val outcomes)
    d_in = val_feats.shape[1]
    gate = BinaryGate(d_in=d_in, d_hidden=args.d_hidden).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.wd)
    X = torch.from_numpy(val_feats).to(device)
    y = torch.from_numpy(val_labels).to(device)
    w = torch.from_numpy(val_weights).to(device)

    best_val_f1 = -1.0
    best_state = None
    for ep in range(args.epochs):
        gate.train()
        logit = gate(X)
        loss = F.binary_cross_entropy_with_logits(logit, y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
        # Pred
        with torch.no_grad():
            prob = torch.sigmoid(logit).cpu().numpy()
        # F1 at threshold 0.5 on weighted instances
        y_np = val_labels
        w_np = val_weights
        mask = (w_np > 0.3)
        tp = ((prob > 0.5) & (y_np > 0.5) & mask).sum()
        fp = ((prob > 0.5) & (y_np <= 0.5) & mask).sum()
        fn = ((prob <= 0.5) & (y_np > 0.5) & mask).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.detach().clone().cpu() for k, v in gate.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep{ep} loss={float(loss):.4f} F1={f1:.4f} prec={prec:.4f} rec={rec:.4f}")

    gate.load_state_dict(best_state); gate.eval()

    # Select threshold on val — look at val Δtop1 over a threshold grid
    print(f"[binary-gate] selecting threshold on val")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_t = 0.5; best_val_delta = -1.0; best_val_cover = 0.0
    base_val_picks = {p: dict(picks=val_enc[p]["logits"].argmax(axis=1).astype(np.int32),
                              costs=val_enc[p]["costs"], oracle=val_enc[p]["oracle"].astype(np.int32))
                       for p in PROBLEMS}
    val_base_top1 = float(np.mean([(base_val_picks[p]["picks"] == base_val_picks[p]["oracle"]).mean()
                                    for p in PROBLEMS]))
    for t in thresholds:
        val_picks, flips = apply_gate_on_split(val_enc, val_priors, blind_spots,
                                                gate, device, args.alpha_apply, t)
        val_top1 = float(np.mean([(val_picks[p]["picks"] == val_picks[p]["oracle"]).mean()
                                   for p in PROBLEMS]))
        delta = val_top1 - val_base_top1
        N = sum((val_picks[p]["picks"] != base_val_picks[p]["picks"]).sum() for p in PROBLEMS)
        cover = N / 18000
        print(f"  t={t:.1f}: val_top1={val_top1:.4f} Δ={delta:+.4f} cover={cover:.3f} wr={flips['wr']} rw={flips['rw']}")
        if delta > best_val_delta:
            best_val_delta = delta; best_t = t; best_val_cover = cover
    print(f"[binary-gate] val-best t={best_t}, Δ_val={best_val_delta:+.4f}")

    # Apply on test
    test_picks, test_flips = apply_gate_on_split(test_enc, test_priors, blind_spots,
                                                  gate, device, args.alpha_apply, best_t)
    base_test_picks = {p: dict(picks=test_enc[p]["logits"].argmax(axis=1).astype(np.int32),
                                costs=test_enc[p]["costs"], oracle=test_enc[p]["oracle"].astype(np.int32))
                        for p in PROBLEMS}
    test_base_top1 = float(np.mean([(base_test_picks[p]["picks"] == base_test_picks[p]["oracle"]).mean()
                                     for p in PROBLEMS]))
    test_top1 = float(np.mean([(test_picks[p]["picks"] == test_picks[p]["oracle"]).mean()
                                for p in PROBLEMS]))
    # Bootstrap
    boot = bootstrap_vs_base({1.0: base_test_picks, 0.0: test_picks},
                              ref_alpha=1.0, n_boot=args.n_boot)[0.0]
    lines = [f"# Binary learned gate (R33)\n",
             f"Base: {args.base_ckpt}, k={args.k}, τ={args.temperature}, alpha_apply={args.alpha_apply}",
             f"Val-selected threshold: {best_t}",
             f"Val: top1 {val_base_top1:.4f} → {val_base_top1 + best_val_delta:.4f} (Δ={best_val_delta:+.4f})",
             f"Test: top1 {test_base_top1:.4f} → {test_top1:.4f} (Δ={test_top1 - test_base_top1:+.4f})",
             "",
             f"Test wrong→right: {test_flips['wr']}, right→wrong: {test_flips['rw']}, unchanged: {test_flips['unchanged']}",
             f"Test fraction modified: {(test_flips['wr'] + test_flips['rw']) / 18000:.3f} (including some wrong→wrong)",
             "",
             f"Bootstrap Δtop1: {boot['top1_mean']:+.4f} [{boot['top1_lo']:+.4f}, {boot['top1_hi']:+.4f}]",
             f"Bootstrap Δcost%: {boot['cost_mean']:+.4f}% [{boot['cost_lo']:+.4f}, {boot['cost_hi']:+.4f}]",
             f"**SIGNIFICANT** at 95% two-sided: {'YES' if boot['top1_lo'] > 0 else 'NO'}",]
    (out / "binary_gate.md").write_text("\n".join(lines))
    (out / "binary_gate.json").write_text(json.dumps(dict(
        val_base_top1=val_base_top1, val_delta=best_val_delta, best_t=best_t,
        test_base_top1=test_base_top1, test_top1=test_top1,
        bootstrap=boot, test_flips=test_flips,
        args=vars(args), blind_spots={p: bs for p, bs in blind_spots.items()},
    ), indent=2, default=float))
    torch.save(best_state, out / "gate.pt")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
