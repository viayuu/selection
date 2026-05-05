"""Train the unified selector across 18 problems.

Usage:
    python -m code.unified_selector.train --epochs 20 --batch-per-problem 8 \
           --lr 3e-4 --tau 0.02 --dropout 0.1 --problem-dropout 0.25 \
           --save-dir code/unified_selector/runs/R1_seed0 --seed 0 --wandb

Training loop: each gradient step = one single-problem microbatch of
BATCH_PER_PROBLEM instances.  An optional interleaved schedule is available
to reduce cross-problem forgetting during long runs.
"""
from __future__ import annotations
import argparse, json, math, pickle, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .registry import PROBLEMS, P2I
from .data import UnifiedProblemDataset, collate_single_problem, DATA_ROOT
from .model import (
    UnifiedSelector,
    remap_legacy_state_dict,
    regret_soft_targets,
    masked_listwise_ce,
    masked_ranking_loss,
    gap_regression_rank_loss,
    winner_margin_loss,
    predict_cost,
    combined_cost_loss,
    soft_sbs_risk_loss,
    shortlist_from_support,
    support_asymmetric_bce_loss,
)


# Oversampling weights (fix 3) — more weight on under-performing problems.
PROBLEM_WEIGHT = {
    "CVRP": 4.0, "VRPB": 3.0, "OVRPB": 3.0, "OVRP": 2.0,
    "VRPBL": 1.5, "VRPBTW": 1.5, "OVRPBL": 1.5, "OVRPBTW": 1.5,
    "VRPBLTW": 1.5, "OVRPBLTW": 1.5,
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def configure_torch_runtime():
    """Runtime-only speed knobs. Keeps model logic unchanged."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def apply_meta_only(batch, meta_only: bool):
    """Zero instance features for train/eval when running the metadata-only baseline."""
    if not meta_only:
        return batch
    if batch["kind"] == "coord":
        batch["node"] = torch.zeros_like(batch["node"])
    elif batch["kind"] == "matrix":
        batch["matrix"] = torch.zeros_like(batch["matrix"])
    return batch


def problem_weight(problem: str, oversample: bool) -> float:
    return float(PROBLEM_WEIGHT.get(problem, 1.0)) if oversample else 1.0


def build_hard_case_weights(ds: UnifiedProblemDataset, sbs_pool_idx: int,
                            alpha: float, beta: float, gamma: float,
                            ambiguity_eps: float) -> torch.Tensor | None:
    if alpha <= 0 and beta <= 0 and gamma <= 0:
        return None
    weights = []
    for i in range(ds.base_N):
        costs = torch.tensor(ds.labels[str(i)]["cost"][:ds.K_p], dtype=torch.float32)
        best_cost, best_idx = costs.min(dim=0)
        sbs_cost = costs[sbs_pool_idx]
        non_sbs = float(best_idx.item() != sbs_pool_idx)
        switch_gain = float(((sbs_cost - best_cost) / (sbs_cost.abs() + 1e-9)).clamp_min(0).item())
        if costs.numel() > 1:
            top2 = costs.topk(k=min(2, costs.numel()), largest=False).values
            ambiguity = float((((top2[1] - top2[0]) / (top2[0].abs() + 1e-9)) < ambiguity_eps).item())
        else:
            ambiguity = 0.0
        weights.append(1.0 + alpha * non_sbs + beta * switch_gain + gamma * ambiguity)
    weights = torch.tensor(weights, dtype=torch.double)
    if ds.coord_augment > 1:
        weights = weights.repeat(ds.coord_augment)
    return weights


def build_oracle_support_stats(ds: UnifiedProblemDataset, smoothing: float = 0.01,
                               class_balance_power: float = 0.5):
    counts = torch.zeros(ds.K_p, dtype=torch.float32)
    for i in range(ds.base_N):
        costs = torch.tensor(ds.labels[str(i)]["cost"][:ds.K_p], dtype=torch.float32)
        winner = int(costs.argmin().item())
        counts[winner] += 1.0
    prior = (counts + smoothing)
    prior = prior / prior.sum().clamp_min(1e-8)
    weights = (counts + smoothing).pow(-class_balance_power)
    weights = weights / weights.mean().clamp_min(1e-8)
    return {
        "winner_counts": counts,
        "winner_prior": prior,
        "winner_weights": weights,
    }


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def dataloader_kwargs(num_workers: int):
    kwargs = {}
    if num_workers > 0:
        kwargs.update(
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    return kwargs


def evaluate(model, problems_val, batch_per_problem: int, num_workers: int, device, tau: float,
             audit_vbs_sbs: dict, meta_only: bool = False,
             use_support_head: bool = False,
             support_threshold: float = 0.5,
             support_topk: int = 3):
    model.eval()
    per_p = {}
    with torch.no_grad():
        for p in problems_val:
            ds = UnifiedProblemDataset(p, "val", coord_augment=0)
            loader = DataLoader(
                ds, batch_size=batch_per_problem, shuffle=False, num_workers=num_workers,
                collate_fn=collate_single_problem, drop_last=False,
                persistent_workers=(num_workers > 0),
            )
            top1 = top2 = top3 = n_tot = 0
            cost_sum = 0.0
            support_top1 = support_top3 = 0.0
            support_arm_seen = torch.zeros(ds.K_p, dtype=torch.bool)
            for batch in loader:
                batch = to_device(batch, device)
                batch = apply_meta_only(batch, meta_only)
                if use_support_head:
                    logits, support_logits = model(batch, return_support=True)
                else:
                    logits = model(batch)
                    support_logits = None
                log_p = torch.log_softmax(logits, dim=1)
                pool_ids = batch["pool_ids"]
                log_p_pool = log_p[:, pool_ids]  # (B, K_p)
                if support_logits is not None:
                    support_pool = support_logits[:, pool_ids] if support_logits.dim() == 2 else support_logits[:, :, pool_ids]
                else:
                    support_pool = None
                effective_log_p_pool, shortlist, _ = shortlist_from_support(
                    log_p_pool,
                    support_pool,
                    threshold=support_threshold,
                    topk=support_topk,
                )
                # Ranking over pool for top-k accuracy vs. true oracle ranking
                costs = batch["costs"]
                true_rank = torch.argsort(costs, dim=1)  # best first
                pred_rank = torch.argsort(-effective_log_p_pool, dim=1)  # best first
                best_true = true_rank[:, 0]
                top1 += (pred_rank[:, 0] == best_true).sum().item()
                top2_k = min(2, pred_rank.shape[1])
                top3_k = min(3, pred_rank.shape[1])
                top2 += (pred_rank[:, :top2_k] == best_true.unsqueeze(1)).any(dim=1).sum().item()
                top3 += (pred_rank[:, :top3_k] == best_true.unsqueeze(1)).any(dim=1).sum().item()
                if shortlist is not None:
                    support_top1 += shortlist.gather(1, best_true.unsqueeze(1)).float().sum().item()
                    top3_true = true_rank[:, :top3_k]
                    support_top3 += shortlist.gather(1, top3_true).any(dim=1).float().sum().item()
                    support_arm_seen |= shortlist.any(dim=0).cpu()
                # Selected cost
                sel = pred_rank[:, 0]
                sel_cost = costs.gather(1, sel.unsqueeze(1)).squeeze(1)
                cost_sum += sel_cost.sum().item()
                n_tot += costs.shape[0]
            sbs = audit_vbs_sbs[p]["val"]["sbs_mean"]
            vbs = audit_vbs_sbs[p]["val"]["vbs_mean"]
            mean_cost = cost_sum / max(1, n_tot)
            per_p[p] = {
                "top1": top1 / n_tot, "top2": top2 / n_tot, "top3": top3 / n_tot,
                "mean_cost": mean_cost, "sbs": sbs, "vbs": vbs,
                "vs_sbs_pct": (mean_cost - sbs) / (abs(sbs) + 1e-9) * 100,
                "vbs_gap_closed_pct": ((sbs - mean_cost) / (sbs - vbs + 1e-9)) * 100 if sbs != vbs else 0.0,
                "support_top1_recall": support_top1 / n_tot if use_support_head else None,
                "support_top3_recall": support_top3 / n_tot if use_support_head else None,
                "support_arm_coverage": float(support_arm_seen.float().mean().item()) if use_support_head else None,
                "n": n_tot,
            }
    # Macro
    macro_top1 = sum(v["top1"] for v in per_p.values()) / len(per_p)
    macro_vs_sbs = sum(v["vs_sbs_pct"] for v in per_p.values()) / len(per_p)
    macro_vbs_closed = sum(v["vbs_gap_closed_pct"] for v in per_p.values()) / len(per_p)
    model.train()
    return per_p, {"macro_top1": macro_top1, "macro_vs_sbs_pct": macro_vs_sbs, "macro_vbs_gap_closed_pct": macro_vbs_closed}


def build_oracle_support_stats_from_labels(problem: str, overfit: int, smoothing: float = 0.01,
                                           class_balance_power: float = 0.5):
    d = DATA_ROOT / f"{problem}train"
    with open(d / "raw_label.pkl", "rb") as f:
        raw = pickle.load(f)
    keys = sorted(raw.keys(), key=lambda x: int(x))
    if overfit > 0:
        keys = keys[:overfit]
    first = raw[keys[0]]
    k_p = len(first["cost"])
    counts = torch.zeros(k_p, dtype=torch.float32)
    for k in keys:
        costs = torch.tensor(raw[k]["cost"][:k_p], dtype=torch.float32)
        counts[int(costs.argmin().item())] += 1.0
    prior = (counts + smoothing)
    prior = prior / prior.sum().clamp_min(1e-8)
    weights = (counts + smoothing).pow(-class_balance_power)
    weights = weights / weights.mean().clamp_min(1e-8)
    return {
        "winner_counts": counts,
        "winner_prior": prior,
        "winner_weights": weights,
    }


def chunked_problem_order(problems, group_size: int):
    probs = list(problems)
    if group_size <= 0 or group_size >= len(probs):
        return [probs]
    random.shuffle(probs)
    return [probs[i:i + group_size] for i in range(0, len(probs), group_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-per-problem", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1500)
    ap.add_argument("--cosine-total", type=int, default=40000)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--problem-dropout", type=float, default=0.25)
    ap.add_argument("--no-fact", action="store_true")
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--init-ckpt", type=str, default=None,
                    help="Optional checkpoint to initialize model weights from.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-out", type=str, nargs="*", default=[],
                    help="MVRP bitvector-based problems to hold out (for zero-shot exp).")
    ap.add_argument("--audit", type=str, default="code/unified_selector/runs/audit.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="selector")
    ap.add_argument("--wandb-entity", type=str, default="yjkds-southern-university-of-science-technology")
    ap.add_argument("--wandb-tag", type=str, default="R1")
    ap.add_argument("--eval-every-epoch", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coord-augment", type=int, default=0,
                    help="Coordinate augmentation multiplier for train split. Supports 0/1/8.")
    ap.add_argument("--overfit", type=int, default=0, help="If >0, use first N instances of each problem and train for more epochs.")
    ap.add_argument("--meta-only", action="store_true", help="Zero-out instance features (R2 baseline).")
    ap.add_argument("--loss", choices=["soft", "rank", "combined", "soft_risk", "gap_rank"], default="soft")
    ap.add_argument("--rank-topk", type=int, default=3)
    ap.add_argument("--gap-reference", choices=["sbs", "best"], default="sbs")
    ap.add_argument("--gap-cap", type=float, default=0.25)
    ap.add_argument("--huber-delta", type=float, default=0.05)
    ap.add_argument("--reg-weight", type=float, default=1.0)
    ap.add_argument("--pairwise-weight", type=float, default=1.0)
    ap.add_argument("--pair-sample-topk", type=int, default=0)
    ap.add_argument("--risk-weight", type=float, default=2.0)
    ap.add_argument("--risk-cap", type=float, default=0.05)
    ap.add_argument("--soft-ce-weight", type=float, default=0.5)
    ap.add_argument("--sbs-bias-init", type=float, default=0.0,
                    help="If >0, initialize the SBS solver bias for each problem to this positive value.")
    ap.add_argument("--switch-hinge-weight", type=float, default=0.0)
    ap.add_argument("--switch-gap", type=float, default=0.001)
    ap.add_argument("--switch-margin", type=float, default=0.10)
    ap.add_argument("--no-problem-solver-bias", action="store_true")
    ap.add_argument("--problem-film", action="store_true",
                    help="Apply problem-conditioned FiLM after shared instance encoding.")
    ap.add_argument("--size-feature", action="store_true",
                    help="Add a small learned projection of instance size features.")
    ap.add_argument("--rich-pool", action="store_true",
                    help="Use mean/max/std/depot pooling for coordinate problems.")
    ap.add_argument("--global-stats", action="store_true",
                    help="Add explicit deterministic global instance statistics.")
    ap.add_argument("--manual-features", action="store_true",
                    help="Add a richer hand-crafted instance feature branch.")
    ap.add_argument("--constraint-experts", action="store_true",
                    help="Add simple constraint-conditioned expert residual heads.")
    ap.add_argument("--problem-residual-head", action="store_true",
                    help="Add a lightweight per-problem residual scorer on top of the shared head.")
    ap.add_argument("--problem-adapter", action="store_true",
                    help="Add a stronger problem-conditioned adapter after the shared backbone.")
    ap.add_argument("--adapter-hidden", type=int, default=64)
    ap.add_argument("--coord-hier-pool", action="store_true",
                    help="Use NSS-style hierarchical pooling in the coordinate encoder.")
    ap.add_argument("--coord-downsample-ratio", type=float, default=0.8,
                    help="Downsample ratio for hierarchical coordinate pooling.")
    ap.add_argument("--deep-encoder-overhaul", action="store_true",
                    help="Use hierarchical MBM/ReZero encoder upgrades inspired by URS + CoEKS.")
    ap.add_argument("--encoder-rezero", action="store_true",
                    help="Enable ReZero residual scaling in the deep encoder overhaul.")
    ap.add_argument("--encoder-constraint-experts", action="store_true",
                    help="Apply shallow constraint-expert FFN residuals inside the encoder.")
    ap.add_argument("--encoder-constraint-hidden", type=int, default=128)
    ap.add_argument("--oversample", action="store_true", help="Apply PROBLEM_WEIGHT oversampling.")
    ap.add_argument("--hard-case-alpha", type=float, default=0.0,
                    help="Extra sampling mass for instances where the per-instance oracle is not the SBS solver.")
    ap.add_argument("--hard-case-beta", type=float, default=0.0,
                    help="Extra sampling mass proportional to normalized switch gain over SBS.")
    ap.add_argument("--hard-case-gamma", type=float, default=0.0,
                    help="Extra sampling mass for ambiguous near-tie instances.")
    ap.add_argument("--ambiguity-eps", type=float, default=0.002,
                    help="Relative top1-top2 gap threshold used to flag ambiguous instances.")
    ap.add_argument("--winner-ce-weight", type=float, default=0.0,
                    help="Auxiliary weighted CE on the oracle-best solver to improve rare-arm recall.")
    ap.add_argument("--winner-balance-power", type=float, default=0.5,
                    help="Inverse-frequency exponent for oracle-winner class balancing.")
    ap.add_argument("--winner-margin-weight", type=float, default=0.0,
                    help="Margin loss weight that pushes the oracle-best solver above the strongest competitor.")
    ap.add_argument("--winner-margin-base", type=float, default=0.05,
                    help="Base required logit margin between oracle-best solver and strongest competitor.")
    ap.add_argument("--winner-margin-gap-scale", type=float, default=2.0,
                    help="Additional margin scaling from oracle best-vs-runner-up relative cost gap.")
    ap.add_argument("--support-match-weight", type=float, default=0.0,
                    help="KL penalty that matches the batch-average predicted arm mass to the empirical oracle prior.")
    ap.add_argument("--support-prior-smoothing", type=float, default=0.01,
                    help="Smoothing added to oracle winner priors for support matching / class-balanced CE.")
    ap.add_argument("--support-head", action="store_true",
                    help="Train a separate shortlist head and use it at eval time before final ranking.")
    ap.add_argument("--support-hidden", type=int, default=128)
    ap.add_argument("--support-generators", type=int, default=1,
                    help="Number of shortlist generators. >1 enables union-of-generators shortlist retrieval.")
    ap.add_argument("--support-loss-weight", type=float, default=0.0,
                    help="Weight of the shortlist supervision loss.")
    ap.add_argument("--support-eps", type=float, default=0.01,
                    help="Relative regret threshold used to mark shortlist positives.")
    ap.add_argument("--support-topk", type=int, default=3,
                    help="Always include oracle top-k solvers in the shortlist target / fallback shortlist.")
    ap.add_argument("--support-threshold", type=float, default=0.5,
                    help="Sigmoid threshold used by the shortlist head at eval / inference time.")
    ap.add_argument("--support-focal-gamma-neg", type=float, default=4.0,
                    help="Asymmetric focal exponent on shortlist negatives.")
    ap.add_argument("--support-pos-weight", type=float, default=2.0,
                    help="Positive reweighting for shortlist supervision.")
    ap.add_argument("--support-diversity-weight", type=float, default=0.0,
                    help="Penalty on overlapping negative shortlist mass across multiple generators.")
    ap.add_argument("--support-budget-weight", type=float, default=0.0,
                    help="Penalty that encourages each shortlist generator to keep a sparse support budget.")
    ap.add_argument("--support-target-mode", choices=["legacy", "rank_partition"], default="legacy",
                    help="How multi-generator shortlist heads are supervised.")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Freeze the shared instance backbone and only train solver/head-specific modules.")
    ap.add_argument("--schedule", choices=["interleaved", "sequential", "grouped_interleaved"], default="sequential",
                    help="Problem scheduling within each epoch. Interleaved is experimental and may reduce forgetting.")
    ap.add_argument("--interleaved-group-size", type=int, default=0,
                    help="If >0, only keep this many problems resident at once under grouped_interleaved.")
    ap.add_argument("--max-batches-per-problem", type=int, default=0,
                    help="If >0, cap the number of train batches drawn per problem in each epoch. Useful for quick pilots.")
    args = ap.parse_args()

    set_seed(args.seed)
    configure_torch_runtime()
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    problems_train = [p for p in PROBLEMS if p not in args.hold_out]
    problems_val_all = PROBLEMS
    print(f"[train] train problems: {problems_train}")
    print(f"[train] held-out: {args.hold_out}")

    # Load audit for SBS/VBS
    audit = json.loads(Path(args.audit).read_text())

    # Train datasets: stream per-problem. Interleaved schedule keeps one iterator per
    # problem to avoid end-of-epoch forgetting when training a pooled selector.
    def _stream_train_loader(problem):
        ds = UnifiedProblemDataset(problem, "train", coord_augment=args.coord_augment)
        if args.overfit > 0:
            ds.instances = ds.instances[:args.overfit]
            ds.labels = {str(i): ds.labels[str(i)] for i in range(args.overfit)}
            ds.base_N = args.overfit
            ds.N = ds.base_N * (ds.coord_augment if ds.coord_augment > 1 else 1)
        sampler = None
        weights = build_hard_case_weights(
            ds,
            sbs_pool_idx=audit[problem]["sbs_pool_idx"],
            alpha=args.hard_case_alpha,
            beta=args.hard_case_beta,
            gamma=args.hard_case_gamma,
            ambiguity_eps=args.ambiguity_eps,
        )
        if weights is not None:
            sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(
            ds,
            batch_size=args.batch_per_problem,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_single_problem,
            drop_last=True,
            **dataloader_kwargs(args.num_workers),
        )

    oracle_support = {}
    if args.winner_ce_weight > 0 or args.winner_margin_weight > 0 or args.support_match_weight > 0 or (args.support_head and args.support_loss_weight > 0):
        for p in problems_train:
            oracle_support[p] = build_oracle_support_stats_from_labels(
                p,
                overfit=args.overfit,
                smoothing=args.support_prior_smoothing,
                class_balance_power=args.winner_balance_power,
            )

    model = UnifiedSelector(d=args.d, depth=args.depth, dropout=args.dropout,
                            use_mvrp_factorized=not args.no_fact,
                            use_problem_solver_bias=not args.no_problem_solver_bias,
                            use_problem_film=args.problem_film,
                            use_size_feature=args.size_feature,
                            rich_pool=args.rich_pool,
                            use_global_stats=args.global_stats,
                            use_manual_features=args.manual_features,
                            use_constraint_experts=args.constraint_experts,
                            use_problem_residual_head=args.problem_residual_head,
                            use_problem_adapter=args.problem_adapter,
                            use_support_head=args.support_head,
                            support_hidden=args.support_hidden,
                            support_generators=args.support_generators,
                            adapter_hidden=args.adapter_hidden,
                            coord_hier_pool=args.coord_hier_pool,
                            coord_downsample_ratio=args.coord_downsample_ratio,
                            deep_encoder_overhaul=args.deep_encoder_overhaul,
                            encoder_rezero=args.encoder_rezero,
                            encoder_constraint_experts=args.encoder_constraint_experts,
                            encoder_constraint_hidden=args.encoder_constraint_hidden).to(device)
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(remap_legacy_state_dict(init["model"]), strict=False)
        print(f"[train] initialized from {args.init_ckpt}")
        if missing:
            print(f"[train] missing keys: {missing}")
        if unexpected:
            print(f"[train] unexpected keys: {unexpected}")
    if args.freeze_backbone:
        trainable_prefixes = (
            "solver_emb",
            "head",
            "support_head",
            "support_heads",
            "support_generator_solver_bias",
            "support_problem_heads",
            "support_constraint_expert_heads",
            "problem_heads",
            "constraint_expert_heads",
            "problem_adapters",
            "adapter_ln",
            "problem_solver_bias",
            "fact_head",
            "W_c",
            "W_d",
        )
        for name, param in model.named_parameters():
            param.requires_grad = any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in trainable_prefixes
            )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[train] freeze_backbone enabled: trainable_params={n_train}/{n_total}")
    model.set_prob_dropout(args.problem_dropout)
    if (not args.no_problem_solver_bias) and args.sbs_bias_init > 0:
        with torch.no_grad():
            for problem in PROBLEMS:
                pid = P2I[problem]
                sbs_pool_idx = audit[problem]["sbs_pool_idx"]
                sbs_global = audit[problem]["pool_global_ids"][sbs_pool_idx]
                model.problem_solver_bias[pid, sbs_global] = args.sbs_bias_init

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.cosine_total - args.warmup_steps)
        t = min(1.0, t)
        import math
        return 3e-5 + (args.lr - 3e-5) * 0.5 * (1 + math.cos(math.pi * t))

    # WandB
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=Path(args.save_dir).name,
                       tags=[args.wandb_tag], config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            args.wandb = False

    step = 0
    best_macro_vs_sbs = 1e9
    loss_ema = None
    t0 = time.time()
    for epoch in range(args.epochs):
        def train_one_batch(problem, batch):
            nonlocal step, loss_ema
            batch = to_device(batch, device)
            batch = apply_meta_only(batch, args.meta_only)
            if args.support_head:
                logits, support_logits = model(batch, return_support=True)
            else:
                logits = model(batch)
                support_logits = None
            logits_pool = logits[:, batch["pool_ids"]]
            aux_winner = logits.new_tensor(0.0)
            aux_margin = logits.new_tensor(0.0)
            aux_support = logits.new_tensor(0.0)
            aux_shortlist = logits.new_tensor(0.0)
            if args.winner_ce_weight > 0 or args.support_match_weight > 0:
                winner = batch["costs"].argmin(dim=1)
                support = oracle_support[problem]
                if args.winner_ce_weight > 0:
                    class_w = support["winner_weights"].to(device)
                    aux_winner = F.cross_entropy(logits_pool, winner, weight=class_w)
                if args.support_match_weight > 0:
                    mean_pred = torch.softmax(logits_pool, dim=1).mean(dim=0).clamp_min(1e-8)
                    prior = support["winner_prior"].to(device)
                    aux_support = F.kl_div(mean_pred.log(), prior, reduction="batchmean")
            if args.winner_margin_weight > 0:
                class_w = oracle_support[problem]["winner_weights"].to(device) if problem in oracle_support else None
                aux_margin = winner_margin_loss(
                    logits,
                    batch["pool_ids"],
                    batch["costs"],
                    base_margin=args.winner_margin_base,
                    gap_scale=args.winner_margin_gap_scale,
                    class_weight=class_w,
                )
            if args.loss == "combined":
                sbs_idx = audit[problem]["sbs_pool_idx"]
                loss = combined_cost_loss(logits, batch["pool_ids"], batch["costs"],
                                          sbs_pool_idx=sbs_idx, tau=0.07)
            elif args.loss == "soft_risk":
                sbs_idx = audit[problem]["sbs_pool_idx"]
                loss = soft_sbs_risk_loss(
                    logits, batch["pool_ids"], batch["costs"],
                    sbs_pool_idx=sbs_idx, tau=args.tau,
                    w_ce=args.soft_ce_weight, w_risk=args.risk_weight,
                    risk_cap=args.risk_cap,
                    w_switch_hinge=args.switch_hinge_weight,
                    switch_gap=args.switch_gap,
                    switch_margin=args.switch_margin,
                )
            elif args.loss == "gap_rank":
                sbs_idx = audit[problem]["sbs_pool_idx"]
                loss = gap_regression_rank_loss(
                    logits, batch["pool_ids"], batch["costs"],
                    sbs_pool_idx=sbs_idx,
                    gap_reference=args.gap_reference,
                    gap_cap=args.gap_cap,
                    huber_delta=args.huber_delta,
                    w_reg=args.reg_weight,
                    w_pair=args.pairwise_weight,
                    pair_sample_topk=args.pair_sample_topk,
                )
            elif args.loss == "rank":
                loss = masked_ranking_loss(
                    logits, batch["pool_ids"], batch["costs"],
                    rank_topk=args.rank_topk,
                )
            else:
                soft = regret_soft_targets(batch["costs"], tau=args.tau)
                loss = masked_listwise_ce(logits, batch["pool_ids"], soft)
            if args.support_head and args.support_loss_weight > 0:
                solver_pos_weight = oracle_support[problem]["winner_weights"].to(device) if problem in oracle_support else None
                support_pool = support_logits[:, batch["pool_ids"]] if support_logits.dim() == 2 else support_logits[:, :, batch["pool_ids"]]
                aux_shortlist = support_asymmetric_bce_loss(
                    support_pool,
                    batch["costs"],
                    eps=args.support_eps,
                    topk=args.support_topk,
                    gamma_neg=args.support_focal_gamma_neg,
                    pos_weight=args.support_pos_weight,
                    solver_pos_weight=solver_pos_weight,
                    diversity_weight=args.support_diversity_weight,
                    budget_weight=args.support_budget_weight,
                    target_mode=args.support_target_mode,
                )
            loss = (
                loss
                + args.winner_ce_weight * aux_winner
                + args.winner_margin_weight * aux_margin
                + args.support_match_weight * aux_support
                + args.support_loss_weight * aux_shortlist
            )
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
            step += 1
            if step % args.log_every == 0:
                msg = f"ep{epoch} step{step} prob={problem} lr={lr_at(step):.2e} loss={loss.item():.4f} ema={loss_ema:.4f}"
                print(msg, flush=True)
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/loss_ema": loss_ema, "train/lr": lr_at(step), "step": step})

        if epoch == 0:
            train_loaders0 = {p: _stream_train_loader(p) for p in problems_train}
            target_batches0 = {
                p: max(1, math.ceil(len(train_loaders0[p]) * problem_weight(p, args.oversample)))
                for p in problems_train
            }
            if args.max_batches_per_problem > 0:
                target_batches0 = {p: min(v, args.max_batches_per_problem) for p, v in target_batches0.items()}
            plan = ", ".join(f"{p}:{target_batches0[p]}" for p in problems_train)
            print(f"[train] epoch schedule={args.schedule} target_batches={plan}")
            del train_loaders0

        if args.schedule == "sequential":
            epoch_groups = [random.sample(problems_train, len(problems_train))]
        elif args.schedule == "grouped_interleaved":
            group_size = args.interleaved_group_size or max(1, min(6, len(problems_train)))
            epoch_groups = chunked_problem_order(problems_train, group_size)
        else:
            epoch_groups = [problems_train.copy()]

        for group in epoch_groups:
            train_loaders = {p: _stream_train_loader(p) for p in group}
            target_batches = {
                p: max(1, math.ceil(len(train_loaders[p]) * problem_weight(p, args.oversample)))
                for p in group
            }
            if args.max_batches_per_problem > 0:
                target_batches = {p: min(v, args.max_batches_per_problem) for p, v in target_batches.items()}
            seen_batches = {p: 0 for p in group}
            iterators = {p: iter(train_loaders[p]) for p in group}

            if args.schedule == "sequential":
                epoch_order = group.copy()
                for p in epoch_order:
                    while seen_batches[p] < target_batches[p]:
                        try:
                            batch = next(iterators[p])
                        except StopIteration:
                            iterators[p] = iter(train_loaders[p])
                            batch = next(iterators[p])
                        train_one_batch(p, batch)
                        seen_batches[p] += 1
            else:
                while True:
                    active = [p for p in group if seen_batches[p] < target_batches[p]]
                    if not active:
                        break
                    random.shuffle(active)
                    for p in active:
                        if seen_batches[p] >= target_batches[p]:
                            continue
                        try:
                            batch = next(iterators[p])
                        except StopIteration:
                            iterators[p] = iter(train_loaders[p])
                            batch = next(iterators[p])
                        train_one_batch(p, batch)
                        seen_batches[p] += 1

            del iterators
            del train_loaders

        if (epoch + 1) % args.eval_every_epoch == 0 or epoch == args.epochs - 1:
            per_p, macro = evaluate(
                model, problems_val_all, args.batch_per_problem, args.num_workers,
                device, args.tau, audit, meta_only=args.meta_only,
                use_support_head=args.support_head,
                support_threshold=args.support_threshold,
                support_topk=args.support_topk,
            )
            print(f"[eval epoch {epoch}] macro_top1={macro['macro_top1']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}%")
            for p in PROBLEMS:
                r = per_p[p]
                mark = ""
                if p in args.hold_out: mark = " [HELD-OUT]"
                print(f"   {p:>10}: top1={r['top1']:.3f} mean_cost={r['mean_cost']:.4f} (sbs={r['sbs']:.4f}) vs_sbs={r['vs_sbs_pct']:+.2f}%{mark}")
            # Save per-epoch
            (save_dir / f"eval_epoch{epoch}.json").write_text(json.dumps({"per_problem": per_p, "macro": macro}, indent=2))
            if args.wandb:
                import wandb
                wandb.log({"val/macro_top1": macro["macro_top1"], "val/macro_vs_sbs_pct": macro["macro_vs_sbs_pct"], "val/macro_vbs_closed": macro["macro_vbs_gap_closed_pct"], "epoch": epoch})
                for p, r in per_p.items():
                    wandb.log({f"val_{p}/top1": r["top1"], f"val_{p}/mean_cost": r["mean_cost"], f"val_{p}/vs_sbs_pct": r["vs_sbs_pct"], "epoch": epoch})
            if macro["macro_vs_sbs_pct"] < best_macro_vs_sbs:
                best_macro_vs_sbs = macro["macro_vs_sbs_pct"]
                torch.save({"model": model.state_dict(), "args": vars(args), "macro": macro}, save_dir / "best.pt")
                print(f"   -> saved best checkpoint (macro_vs_sbs={best_macro_vs_sbs:+.3f}%)")

    elapsed = time.time() - t0
    print(f"[done] {args.epochs} epochs in {elapsed/60:.1f} min, best macro_vs_sbs={best_macro_vs_sbs:+.3f}%")
    (save_dir / "summary.json").write_text(json.dumps({"elapsed_sec": elapsed, "best_macro_vs_sbs_pct": best_macro_vs_sbs}, indent=2))


if __name__ == "__main__":
    main()
