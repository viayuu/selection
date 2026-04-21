"""Train the unified selector across 18 problems.

Usage:
    python -m code.unified_selector.train --epochs 20 --batch-per-problem 8 \
           --lr 3e-4 --tau 0.02 --dropout 0.1 --problem-dropout 0.25 \
           --save-dir code/unified_selector/runs/R1_seed0 --seed 0 --wandb

Training loop: each gradient step = one single-problem microbatch of
BATCH_PER_PROBLEM instances, cycling through all 18 problems in a fixed
shuffle order per epoch (= balanced per-problem exposure, oracle-recommended).
"""
from __future__ import annotations
import argparse, os, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .registry import PROBLEMS, P2I, M_GLOBAL as M_GLOBAL_CHECK
from .data import UnifiedProblemDataset, collate_single_problem
from .model import UnifiedSelector, regret_soft_targets, masked_listwise_ce, predict_cost, combined_cost_loss, _migrate_state_dict, plackett_luce_loss, diversity_entropy_penalty, gap_regression_rank_loss, winner_margin_loss
from .specialist_train import SpecialistSelector


# Oversampling weights (fix 3) — more weight on under-performing problems.
PROBLEM_WEIGHT = {
    "CVRP": 4.0, "VRPB": 3.0, "OVRPB": 3.0, "OVRP": 2.0,
    "VRPBL": 1.5, "VRPBTW": 1.5, "OVRPBL": 1.5, "OVRPBTW": 1.5,
    "VRPBLTW": 1.5, "OVRPBLTW": 1.5,
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_loaders(split: str, problems, batch_per_problem: int, shuffle: bool, num_workers: int = 2,
                  aug_8fold: bool = False):
    loaders = {}
    for p in problems:
        ds = UnifiedProblemDataset(p, split, aug_8fold=aug_8fold)
        loaders[p] = DataLoader(
            ds, batch_size=batch_per_problem, shuffle=shuffle, num_workers=num_workers,
            collate_fn=collate_single_problem, drop_last=shuffle, persistent_workers=(num_workers>0),
        )
    return loaders


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def compute_loss_for_problem(model, batch, p, args, audit, teachers, base_model=None):
    """Forward + all loss terms for one problem batch. Returns scalar loss.

    This extracts the loss body that used to live inline in the block training
    loop so round_robin / task_accum schedulers can share it. The caller is
    responsible for autocast / opt.zero_grad / backward / step.
    """
    logits = model(batch)
    pool_ids = batch["pool_ids"]
    if args.loss == "combined":
        sbs_idx = audit[p]["sbs_pool_idx"]
        loss = combined_cost_loss(logits, pool_ids, batch["costs"],
                                   sbs_pool_idx=sbs_idx, tau=0.07)
    elif args.loss == "gap_rank":
        sbs_idx = audit[p]["sbs_pool_idx"]
        loss = gap_regression_rank_loss(
            logits, pool_ids, batch["costs"],
            sbs_pool_idx=sbs_idx, gap_reference=args.gap_reference,
            gap_cap=args.gap_cap, huber_delta=args.huber_delta,
            w_reg=args.reg_weight, w_pair=args.pairwise_weight,
            pair_sample_topk=args.pair_sample_topk,
        )
    else:
        soft = regret_soft_targets(batch["costs"], tau=args.tau)
        oracle_idx = batch["costs"].argmin(dim=1) if (args.focal_gamma > 0 or args.hard_upweight != 1.0) else None
        loss = masked_listwise_ce(logits, pool_ids, soft,
                                   focal_gamma=args.focal_gamma,
                                   hard_upweight=args.hard_upweight,
                                   oracle_idx_pool=oracle_idx)
    if args.winner_margin_weight > 0:
        loss = loss + args.winner_margin_weight * winner_margin_loss(
            logits, pool_ids, batch["costs"],
            base_margin=args.winner_margin_base,
            gap_scale=args.winner_margin_gap_scale,
        )
    if args.plackett_luce:
        logits_pool = logits[:, pool_ids]
        loss_pl = plackett_luce_loss(logits_pool, batch["costs"], top_k=args.pl_topk)
        if args.pl_only:
            loss = args.pl_weight * loss_pl
        else:
            loss = loss + args.pl_weight * loss_pl
    if args.diversity_weight > 0:
        logits_pool = logits[:, pool_ids]
        loss = loss + args.diversity_weight * diversity_entropy_penalty(logits_pool)
    # Distillation (R18-era teachers)
    if teachers is not None and p in teachers:
        with torch.no_grad():
            t_logits_pool_list = []
            for t_model, kind in teachers[p]:
                if kind == "specialist":
                    t_logits_pool = t_model(batch)
                else:
                    t_logits_pool = t_model(batch)[:, pool_ids]
                t_logits_pool_list.append(t_logits_pool)
            t_probs = torch.stack([torch.softmax(l / args.distill_tau, dim=1)
                                    for l in t_logits_pool_list], dim=0).mean(0)
        s_logp = torch.log_softmax(logits[:, pool_ids] / args.distill_tau, dim=1)
        kl = -(t_probs * s_logp).sum(dim=1).mean() * (args.distill_tau ** 2)
        loss = loss + args.distill_weight * kl
    # Base-KL trust region (R41B): soft regularization toward frozen base model
    if args.base_kl_weight > 0 and base_model is not None:
        with torch.no_grad():
            base_logits = base_model(batch)
        tau = args.base_kl_tau
        base_p = torch.softmax(base_logits[:, pool_ids] / tau, dim=1)
        new_logp = torch.log_softmax(logits[:, pool_ids] / tau, dim=1)
        per_sample_kl = -(base_p * new_logp).sum(dim=1) * (tau ** 2)
        with torch.no_grad():
            base_pick = base_p.argmax(dim=1)
            true_rank = torch.argsort(batch["costs"], dim=1)
            inv_rank = torch.argsort(true_rank, dim=1)
            base_pick_rank = inv_rank.gather(1, base_pick.unsqueeze(1)).squeeze(1)
            is_base_correct = (base_pick_rank == 0).float()
        kl_w = (args.base_kl_weight_correct * is_base_correct
                + args.base_kl_weight_wrong * (1.0 - is_base_correct))
        loss = loss + args.base_kl_weight * (kl_w * per_sample_kl).mean()
    return loss


def _encoder_grad_vector(model):
    """Concatenate flattened gradients of encoder parameters for grad-conflict probe.
    Returns None if no encoder grad is populated.
    """
    pieces = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if ("coord_enc" in name) or ("matrix_enc" in name):
            pieces.append(p.grad.detach().flatten())
    if not pieces:
        return None
    return torch.cat(pieces)


def evaluate(model, loaders_val, device, tau: float, audit_vbs_sbs: dict):
    """Per-problem top1/top2/top3 per the spec in 观测指标/观测指标.md:
    「selector 选择了 top-k 方法的实例数量 / 总实例数量」 →
    fraction of instances where the SELECTOR'S PICK lies in the oracle top-k set.
    """
    model.eval()
    per_p = {}
    with torch.no_grad():
        for p, loader in loaders_val.items():
            top1 = top2 = top3 = n_tot = 0
            cost_sum = 0.0
            for batch in loader:
                batch = to_device(batch, device)
                logits = model(batch)  # (B, M)
                log_p = torch.log_softmax(logits, dim=1)
                pool_ids = batch["pool_ids"]
                log_p_pool = log_p[:, pool_ids]  # (B, K_p)
                costs = batch["costs"]
                K_p = costs.shape[1]
                # Selector's pick (pool-index)
                pred = log_p_pool.argmax(dim=1)                                   # (B,)
                # Oracle rank of each pool slot (0 = best).  ranks[i, s] = position of solver s in oracle order.
                true_rank = torch.argsort(costs, dim=1)                           # (B, K_p)
                inv_rank = torch.argsort(true_rank, dim=1)                        # rank of each pool slot
                pick_rank = inv_rank.gather(1, pred.unsqueeze(1)).squeeze(1)      # (B,) rank of pick, 0=best
                top1 += (pick_rank == 0).sum().item()
                top2 += (pick_rank < min(2, K_p)).sum().item()
                top3 += (pick_rank < min(3, K_p)).sum().item()
                # Selected cost = cost at selector's pick
                sel_cost = costs.gather(1, pred.unsqueeze(1)).squeeze(1)
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
                "n": n_tot,
            }
    # Macro
    macro_top1 = sum(v["top1"] for v in per_p.values()) / len(per_p)
    macro_vs_sbs = sum(v["vs_sbs_pct"] for v in per_p.values()) / len(per_p)
    macro_vbs_closed = sum(v["vbs_gap_closed_pct"] for v in per_p.values()) / len(per_p)
    model.train()
    return per_p, {"macro_top1": macro_top1, "macro_vs_sbs_pct": macro_vs_sbs, "macro_vbs_gap_closed_pct": macro_vbs_closed}


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
    ap.add_argument("--heads", type=int, default=4, help="Attention heads per encoder layer.")
    # R40 NSS-style encoder flags
    ap.add_argument("--encoder-type", choices=["standard", "nss_hierarchical"],
                    default="standard", help="Base encoder architecture.")
    ap.add_argument("--rezero", action="store_true",
                    help="Use ReZero residuals (intrinsic for nss_hierarchical).")
    ap.add_argument("--block-num", type=int, default=2,
                    help="Number of hierarchical blocks (NSS encoder only).")
    ap.add_argument("--encoder-layer-num", type=int, default=2,
                    help="Attention layers per hierarchical block.")
    ap.add_argument("--downsample-ratio", type=float, default=0.8,
                    help="Fraction of tokens kept after each hierarchical block.")
    ap.add_argument("--local-head", action="store_true",
                    help="Add per-problem local MLP head blended with global head.")
    ap.add_argument("--freeze-except-local-head", action="store_true",
                    help="Freeze all params except local_head (for R41-style warm-start from R18).")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Freeze only encoder / matrix_encoder; keep head + local_head + solver_emb trainable.")
    # Early-stopping flags (R40 plan)
    ap.add_argument("--early-stop-cold-epoch", type=int, default=10,
                    help="T1: at/after this epoch, if val top1 < --early-stop-cold-floor, stop.")
    ap.add_argument("--early-stop-cold-floor", type=float, default=0.45,
                    help="T1 floor; val top1 below this at cold-epoch = aborted recipe.")
    ap.add_argument("--early-stop-drop", type=float, default=0.02,
                    help="T2: abort if val top1 drops by more than this from its running max.")
    ap.add_argument("--early-stop-plateau", type=int, default=10,
                    help="T3 patience: epochs with <=0.002 gain vs best before aborting.")
    ap.add_argument("--early-stop-wall-hours", type=float, default=12.0,
                    help="T4: hard wall-clock limit in hours; training stops after this.")
    # R40D/E/R41B task scheduler + trust-region flags
    ap.add_argument("--task-schedule", choices=["block", "round_robin", "task_accum"],
                    default="block",
                    help="Training loop: 'block' = legacy, full loader per problem then next "
                         "(R18/R40 default). 'round_robin' = one mini-batch per problem per step. "
                         "'task_accum' = accumulate 18-problem losses then single opt.step.")
    ap.add_argument("--grad-probe", action="store_true",
                    help="Record encoder-gradient cosine between TSP vs {CVRP,VRPBTW,OVRPBLTW}.")
    ap.add_argument("--grad-probe-every", type=int, default=500)
    ap.add_argument("--residual-adapter", action="store_true",
                    help="Residual-adapter mode for LocalProblemHead: delta zero-init, "
                         "per-problem α init σ≈0.02, logits = base + (α·0.2)·delta. "
                         "Falls back to legacy convex blend when off.")
    ap.add_argument("--base-kl-weight", type=float, default=0.0,
                    help="Trust-region KL to frozen base model. 0 = off.")
    ap.add_argument("--base-kl-ckpt", type=str, default=None,
                    help="Path to frozen base ckpt for trust-region KL. Defaults to --resume-from.")
    ap.add_argument("--base-kl-tau", type=float, default=2.0)
    ap.add_argument("--base-kl-weight-correct", type=float, default=1.0,
                    help="Sample weight for base-KL on instances where base was correct (oracle).")
    ap.add_argument("--base-kl-weight-wrong", type=float, default=0.2,
                    help="Sample weight for base-KL on instances where base was wrong.")
    ap.add_argument("--hard-weight-zero-pick", type=float, default=1.0,
                    help="(reserved for future use; currently unused) multiplier for instances "
                         "whose oracle arm is not in base-support.")
    ap.add_argument("--hard-weight-base-wrong", type=float, default=1.0,
                    help="(reserved) multiplier for instances where base is wrong.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--problem-dropout", type=float, default=0.25)
    ap.add_argument("--no-fact", action="store_true")
    ap.add_argument("--save-dir", type=str, required=True)
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
    ap.add_argument("--overfit", type=int, default=0, help="If >0, use first N instances of each problem and train for more epochs.")
    ap.add_argument("--meta-only", action="store_true", help="Zero-out instance features (R2 baseline).")
    ap.add_argument("--loss", choices=["soft", "combined", "gap_rank"], default="soft")
    ap.add_argument("--no-problem-solver-bias", action="store_true")
    ap.add_argument("--oversample", action="store_true", help="Apply PROBLEM_WEIGHT oversampling.")
    ap.add_argument("--oversample-families", type=str, nargs="*", default=[],
                    help="Problem families to oversample by --oversample-weight (overrides PROBLEM_WEIGHT when set).")
    ap.add_argument("--oversample-weight", type=int, default=2,
                    help="Integer pass count for --oversample-families (default 2).")
    ap.add_argument("--use-film", action="store_true", help="Enable FiLM conditioning from problem embedding on the head MLP.")
    ap.add_argument("--distill-from", type=str, nargs="*", default=[],
                    help="Optional specialist ckpt paths for distillation (1 per problem, or multiple teacher seeds per problem — see --teacher-map).")
    ap.add_argument("--teacher-map", type=str, default=None,
                    help="JSON file mapping 'problem -> [ckpt1, ckpt2, ...]' for distillation teachers.")
    ap.add_argument("--distill-weight", type=float, default=0.5,
                    help="Weight of distillation KL term (vs regret-soft CE).")
    ap.add_argument("--distill-tau", type=float, default=1.0,
                    help="Softmax temperature for distillation targets.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="If >0, maintain EMA of parameters and evaluate EMA model.")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Start training from this checkpoint (used for tail fine-tuning).")
    ap.add_argument("--only-problems", type=str, nargs="*", default=[],
                    help="Restrict training to this subset of problems (for tail fine-tune).")
    ap.add_argument("--min-lr", type=float, default=3e-5,
                    help="Lower LR bound for cosine schedule.")
    ap.add_argument("--aug-8fold", action="store_true",
                    help="Apply 8-fold D4 coord augmentation to train split (NSS-style). Skips ATSP.")
    ap.add_argument("--plackett-luce", action="store_true",
                    help="Use Plackett-Luce listwise ranking loss instead of regret-soft CE.")
    ap.add_argument("--pl-weight", type=float, default=1.0,
                    help="Weight of PL loss (added to soft-CE; use --pl-only to replace).")
    ap.add_argument("--pl-only", action="store_true",
                    help="Use ONLY Plackett-Luce loss (no soft-CE).")
    ap.add_argument("--pl-topk", type=int, default=None,
                    help="PL stages to unroll (default=K_p=full).")
    ap.add_argument("--diversity-weight", type=float, default=0.0,
                    help="Penalize batch arm-collapse (entropy of pooled p).")
    ap.add_argument("--amp", action="store_true",
                    help="Enable torch.cuda.amp mixed precision (faster + lower VRAM).")
    ap.add_argument("--train-batch-multiplier", type=int, default=1,
                    help="Concatenate N mini-batches per step (larger effective batch; helps stability with PL).")
    # R12 architecture flags
    ap.add_argument("--use-arm-attn", action="store_true",
                    help="Enable arm-conditioned cross-attention head (each arm attends over node tokens).")
    ap.add_argument("--arm-attn-heads", type=int, default=4,
                    help="Number of attention heads in arm-attention head.")
    ap.add_argument("--use-coe", action="store_true",
                    help="Enable CoE constraint-expert FFN head (one FFN per constraint bit).")
    ap.add_argument("--coe-experts", type=int, default=None,
                    help="Number of CoE experts (default: K_CBITS+1 = 6).")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="Focal-loss γ weight for soft CE. 0=off, 2=standard focal.")
    ap.add_argument("--hard-upweight", type=float, default=1.0,
                    help="Multiplier for samples where argmax(model) != oracle. 1.0=off, 2.0=double weight on hard.")
    # Gap-rank loss (ported from 0selection2) — regression on normalized cost gaps + pairwise logistic.
    ap.add_argument("--gap-reference", choices=["sbs", "best"], default="sbs",
                    help="Reference cost for gap regression (sbs=per-problem-SBS, best=per-instance-oracle).")
    ap.add_argument("--gap-cap", type=float, default=0.25, help="Clamp for normalized gap magnitude.")
    ap.add_argument("--huber-delta", type=float, default=0.05)
    ap.add_argument("--reg-weight", type=float, default=1.0, help="Weight on gap-regression Huber loss.")
    ap.add_argument("--pairwise-weight", type=float, default=1.0, help="Weight on pairwise logistic loss.")
    ap.add_argument("--pair-sample-topk", type=int, default=0,
                    help="If >0, restrict pairwise loss to pairs involving top-K best gaps (0=all pairs).")
    ap.add_argument("--winner-margin-weight", type=float, default=0.0,
                    help=">0 enables cost-aware winner margin loss on top of --loss.")
    ap.add_argument("--winner-margin-base", type=float, default=0.05)
    ap.add_argument("--winner-margin-gap-scale", type=float, default=2.0)
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    problems_train = [p for p in PROBLEMS if p not in args.hold_out]
    problems_val_all = PROBLEMS
    print(f"[train] train problems: {problems_train}")
    print(f"[train] held-out: {args.hold_out}")

    # Load audit for SBS/VBS
    audit = json.loads(Path(args.audit).read_text())

    # Val loaders: load once, keep cached (1k each is manageable).
    val_loaders   = build_loaders("val",   problems_val_all, args.batch_per_problem, shuffle=False, num_workers=args.num_workers)
    # Train datasets: stream per-problem to keep RSS low.
    def _stream_train_loader(problem):
        ds = UnifiedProblemDataset(problem, "train", aug_8fold=args.aug_8fold)
        if args.overfit > 0:
            ds.instances = ds.instances[:args.overfit]
            ds.labels = {str(i): ds.labels[str(i)] for i in range(args.overfit)}
            ds.N = args.overfit
        return DataLoader(ds, batch_size=args.batch_per_problem, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_single_problem, drop_last=True)

    model = UnifiedSelector(d=args.d, depth=args.depth, dropout=args.dropout,
                            use_mvrp_factorized=not args.no_fact,
                            use_problem_solver_bias=not args.no_problem_solver_bias,
                            use_film=args.use_film,
                            use_arm_attn=args.use_arm_attn,
                            use_coe=args.use_coe,
                            arm_attn_heads=args.arm_attn_heads,
                            coe_experts=args.coe_experts,
                            encoder_type=args.encoder_type,
                            rezero=args.rezero,
                            block_num=args.block_num,
                            encoder_layer_num=args.encoder_layer_num,
                            heads=args.heads,
                            downsample_ratio=args.downsample_ratio,
                            local_head=args.local_head,
                            residual_adapter=args.residual_adapter).to(device)
    model.set_prob_dropout(args.problem_dropout)

    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        sd = _migrate_state_dict(ck["model"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] loaded {args.resume_from} (missing={len(missing)} unexpected={len(unexpected)})")

    # Frozen base model for trust-region KL (R41B). Loaded only when --base-kl-weight > 0.
    base_model = None
    if args.base_kl_weight > 0:
        base_path = args.base_kl_ckpt or args.resume_from
        if base_path is None:
            raise ValueError("--base-kl-weight > 0 requires --base-kl-ckpt or --resume-from")
        bck = torch.load(base_path, map_location=device, weights_only=False)
        bcfg = bck.get("args", {})
        base_model = UnifiedSelector(
            d=bcfg.get("d", args.d), depth=bcfg.get("depth", args.depth),
            dropout=bcfg.get("dropout", 0.1),
            use_mvrp_factorized=not bcfg.get("no_fact", False),
            use_problem_solver_bias="problem_solver_bias" in bck["model"],
            use_film=bcfg.get("use_film", False),
            use_arm_attn=bcfg.get("use_arm_attn", False),
            use_coe=bcfg.get("use_coe", False),
            arm_attn_heads=bcfg.get("arm_attn_heads", 4),
            coe_experts=bcfg.get("coe_experts", None),
            encoder_type=bcfg.get("encoder_type", "standard"),
            rezero=bcfg.get("rezero", False),
            block_num=bcfg.get("block_num", 2),
            encoder_layer_num=bcfg.get("encoder_layer_num", 2),
            heads=bcfg.get("heads", 4),
            downsample_ratio=bcfg.get("downsample_ratio", 0.8),
            local_head=False,
        ).to(device)
        base_model.load_state_dict(_migrate_state_dict(bck["model"]), strict=False)
        base_model.eval()
        for p_ in base_model.parameters():
            p_.requires_grad = False
        print(f"[base-kl] loaded frozen base {base_path} (weight={args.base_kl_weight}, tau={args.base_kl_tau})")

    if args.freeze_except_local_head:
        n_frozen = n_train = 0
        for name, p in model.named_parameters():
            if "local_head" in name:
                p.requires_grad = True
                n_train += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"[freeze] {n_frozen:,} params frozen, {n_train:,} params trainable (local_head only)")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    elif args.freeze_encoder:
        n_frozen = n_train = 0
        for name, p in model.named_parameters():
            if "encoder" in name:
                p.requires_grad = False
                n_frozen += p.numel()
            else:
                p.requires_grad = True
                n_train += p.numel()
        print(f"[freeze] {n_frozen:,} params frozen (encoder only), {n_train:,} params trainable (heads + solver_emb + local_head)")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    # Mixed-precision scaler — only enabled when --amp is set.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # EMA shadow parameters (optional)
    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    # Distillation teachers (optional). Each teacher is either a UnifiedSelector-style ckpt
    # (18-solver table, index by pool_ids) or a SpecialistSelector-style ckpt (K_p-solver table,
    # direct pool-level output).  We auto-detect based on the shape of `solver_emb.weight`.
    teachers = None
    if args.teacher_map:
        tm = json.loads(Path(args.teacher_map).read_text())
        teachers = {}
        for prob, paths in tm.items():
            teachers[prob] = []
            K_p = None
            for pth in paths:
                tck = torch.load(pth, map_location=device, weights_only=False)
                sd = tck["model"]
                emb_shape = sd.get("solver_emb.weight").shape  # (K, d)
                K_t, d_t = emb_shape
                is_specialist = (K_t != M_GLOBAL_CHECK)
                if is_specialist:
                    from .registry import POOLS as _POOLS
                    K_p_ref = len(_POOLS[prob])
                    assert K_t == K_p_ref, f"specialist pool size {K_t} != registry pool size {K_p_ref} for {prob}"
                    t_model = SpecialistSelector(K_p=K_t, d=d_t, depth=tck.get("args", {}).get("depth", 4),
                                                  dropout=0.0, is_matrix=(prob == "ATSP")).to(device)
                    t_model.load_state_dict(sd, strict=False)
                else:
                    tcfg = tck.get("args", {})
                    t_model = UnifiedSelector(d=tcfg.get("d", 128), depth=tcfg.get("depth", 4),
                                              dropout=0.0,
                                              use_mvrp_factorized=not tcfg.get("no_fact", False),
                                              use_problem_solver_bias="problem_solver_bias" in sd,
                                              use_film=tcfg.get("use_film", False)).to(device)
                    t_model.load_state_dict(_migrate_state_dict(sd), strict=False)
                t_model.eval()
                for p_ in t_model.parameters():
                    p_.requires_grad = False
                teachers[prob].append((t_model, "specialist" if is_specialist else "unified"))
        print(f"[distill] loaded teachers for {len(teachers)} problems "
              f"(specialist={sum(1 for v in teachers.values() for _, kind in v if kind=='specialist')}, "
              f"unified={sum(1 for v in teachers.values() for _, kind in v if kind=='unified')})")

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.cosine_total - args.warmup_steps)
        t = min(1.0, t)
        import math
        return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * t))

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
    best_macro_top1 = -1.0
    best_joint = 1e9  # joint = macro_vs_sbs_pct - 100 * macro_top1 (smaller is better)
    loss_ema = None
    t0 = time.time()
    # Early-stopping state (R40 plan)
    top1_history: list[float] = []         # one entry per eval
    stop_reason: str | None = None
    # Task-scheduler diagnostics (R40D/R40E plan §1.c/1.d)
    last_seen_step = {p: 0 for p in PROBLEMS}
    grad_vecs: dict[str, torch.Tensor] = {}   # encoder-grad vectors keyed by problem
    grad_cos_history: list[dict] = []
    probe_targets = ["CVRP", "VRPBTW", "OVRPBLTW"]

    def _prepare_batch(batch):
        batch = to_device(batch, device)
        if args.meta_only and batch["kind"] == "coord":
            batch["node"] = torch.zeros_like(batch["node"])
        elif args.meta_only and batch["kind"] == "matrix":
            batch["matrix"] = torch.zeros_like(batch["matrix"])
        return batch

    def _forward_loss(batch, p):
        """One forward pass under autocast producing the scalar loss."""
        with torch.cuda.amp.autocast(enabled=args.amp):
            return compute_loss_for_problem(model, batch, p, args, audit, teachers, base_model)

    def _apply_single_step(batch, p, step_counter, loss_ema_val):
        """Legacy / round-robin single optimizer step for one (batch, problem)."""
        loss = _forward_loss(batch, p)
        for g in opt.param_groups:
            g["lr"] = lr_at(step_counter)
        opt.zero_grad()
        scaler.scale(loss).backward()
        # Encoder-grad probe (cosine between task gradients).
        # Capture BEFORE unscale_; scaling is the same for all tasks so cosine is invariant.
        if args.grad_probe:
            gvec = _encoder_grad_vector(model)
            if gvec is not None:
                grad_vecs[p] = gvec.detach().clone()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        # EMA update
        if ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in ema_state:
                        ema_state[k].mul_(args.ema_decay).add_(v.detach(), alpha=(1.0 - args.ema_decay))
        loss_val = float(loss.item())
        loss_ema_val = loss_val if loss_ema_val is None else 0.98 * loss_ema_val + 0.02 * loss_val
        new_step = step_counter + 1
        last_seen_step[p] = new_step
        # Periodic logging + grad-cosine probe.
        if new_step % args.log_every == 0:
            msg = f"ep{epoch} step{new_step} prob={p} lr={lr_at(new_step):.2e} loss={loss_val:.4f} ema={loss_ema_val:.4f}"
            print(msg, flush=True)
            if args.wandb:
                import wandb
                wandb.log({"train/loss": loss_val, "train/loss_ema": loss_ema_val,
                           "train/lr": lr_at(new_step), "step": new_step,
                           f"train_per_problem/{p}": loss_val})
        if args.grad_probe and (new_step % args.grad_probe_every == 0):
            cos_row = {"step": new_step}
            gtsp = grad_vecs.get("TSP")
            if gtsp is not None:
                gtsp_f = gtsp.float()
                for other in probe_targets:
                    go = grad_vecs.get(other)
                    if go is None:
                        continue
                    cos = float(F.cosine_similarity(
                        gtsp_f.unsqueeze(0), go.float().unsqueeze(0), dim=1).item())
                    cos_row[f"TSP_{other}"] = cos
                    if args.wandb:
                        import wandb
                        wandb.log({f"grad_cos/TSP_{other}": cos, "step": new_step})
            if len(cos_row) > 1:
                grad_cos_history.append(cos_row)
        return new_step, loss_ema_val

    for epoch in range(args.epochs):
        epoch_problems = [p for p in problems_train if (not args.only_problems or p in args.only_problems)]
        epoch_order = epoch_problems.copy(); random.shuffle(epoch_order)

        if args.task_schedule == "block":
            # Legacy behaviour — train one problem's loader to completion then move on.
            for p in epoch_order:
                n_passes = int(PROBLEM_WEIGHT.get(p, 1.0)) if args.oversample else 1
                if args.oversample_families and p in args.oversample_families:
                    n_passes = max(n_passes, args.oversample_weight)
                for _pass in range(n_passes):
                    loader = _stream_train_loader(p)
                    for batch in loader:
                        batch = _prepare_batch(batch)
                        step, loss_ema = _apply_single_step(batch, p, step, loss_ema)
                    del loader

        elif args.task_schedule == "round_robin":
            # R40D / R41B — one mini-batch per problem per global step, fresh shuffle each cycle.
            loaders = {p: _stream_train_loader(p) for p in epoch_problems}
            iters = {p: iter(loaders[p]) for p in epoch_problems}
            steps_per_epoch = min(len(loaders[p]) for p in epoch_problems)
            for cycle in range(steps_per_epoch):
                order = list(epoch_problems)
                random.shuffle(order)
                for p in order:
                    try:
                        batch = next(iters[p])
                    except StopIteration:
                        iters[p] = iter(loaders[p])
                        batch = next(iters[p])
                    batch = _prepare_batch(batch)
                    step, loss_ema = _apply_single_step(batch, p, step, loss_ema)
            for ld in loaders.values():
                del ld

        elif args.task_schedule == "task_accum":
            # R40E — accumulate 18 per-problem losses before a single opt.step().
            loaders = {p: _stream_train_loader(p) for p in epoch_problems}
            iters = {p: iter(loaders[p]) for p in epoch_problems}
            steps_per_epoch = min(len(loaders[p]) for p in epoch_problems)
            denom = float(len(epoch_problems))
            for cycle in range(steps_per_epoch):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad()
                losses_this_step = []
                order = list(epoch_problems)
                random.shuffle(order)
                for p in order:
                    try:
                        batch = next(iters[p])
                    except StopIteration:
                        iters[p] = iter(loaders[p])
                        batch = next(iters[p])
                    batch = _prepare_batch(batch)
                    loss_p = _forward_loss(batch, p) / denom
                    scaler.scale(loss_p).backward()
                    losses_this_step.append((p, float(loss_p.item()) * denom))
                    last_seen_step[p] = step + 1
                if args.grad_probe:
                    gvec = _encoder_grad_vector(model)
                    if gvec is not None:
                        grad_vecs["__accum__"] = gvec.detach().clone()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if ema_state is not None:
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            if k in ema_state:
                                ema_state[k].mul_(args.ema_decay).add_(v.detach(), alpha=(1.0 - args.ema_decay))
                mean_loss = float(np.mean([l for _, l in losses_this_step]))
                loss_ema = mean_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * mean_loss
                step += 1
                if step % args.log_every == 0:
                    print(f"ep{epoch} step{step} accum mean_loss={mean_loss:.4f} ema={loss_ema:.4f} "
                          f"lr={lr_at(step):.2e}", flush=True)
                    if args.wandb:
                        import wandb
                        wandb.log({"train/loss": mean_loss, "train/loss_ema": loss_ema,
                                   "train/lr": lr_at(step), "step": step,
                                   **{f"train_per_problem/{p}": l for p, l in losses_this_step}})
            for ld in loaders.values():
                del ld
        else:
            raise ValueError(f"unknown --task-schedule {args.task_schedule}")

        if (epoch + 1) % args.eval_every_epoch == 0 or epoch == args.epochs - 1:
            per_p, macro = evaluate(model, val_loaders, device, args.tau, audit)
            print(f"[eval epoch {epoch}] macro_top1={macro['macro_top1']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}%")
            for p in PROBLEMS:
                r = per_p[p]
                mark = ""
                if p in args.hold_out: mark = " [HELD-OUT]"
                print(f"   {p:>10}: top1={r['top1']:.3f} mean_cost={r['mean_cost']:.4f} (sbs={r['sbs']:.4f}) vs_sbs={r['vs_sbs_pct']:+.2f}%{mark}")
            # Save per-epoch
            eval_payload = {"per_problem": per_p, "macro": macro}
            eval_payload["last_seen"] = {
                p: {"last_step": last_seen_step.get(p, 0),
                    "steps_since_last_seen": step - last_seen_step.get(p, 0)}
                for p in PROBLEMS
            }
            if args.grad_probe and grad_cos_history:
                eval_payload["grad_cos_recent"] = grad_cos_history[-20:]
            (save_dir / f"eval_epoch{epoch}.json").write_text(json.dumps(eval_payload, indent=2))
            if args.wandb:
                import wandb
                wandb.log({"val/macro_top1": macro["macro_top1"], "val/macro_vs_sbs_pct": macro["macro_vs_sbs_pct"], "val/macro_vbs_closed": macro["macro_vbs_gap_closed_pct"], "epoch": epoch})
                for p, r in per_p.items():
                    wandb.log({f"val_{p}/top1": r["top1"], f"val_{p}/mean_cost": r["mean_cost"], f"val_{p}/vs_sbs_pct": r["vs_sbs_pct"], "epoch": epoch})
            if macro["macro_vs_sbs_pct"] < best_macro_vs_sbs:
                best_macro_vs_sbs = macro["macro_vs_sbs_pct"]
                torch.save({"model": model.state_dict(), "args": vars(args), "macro": macro}, save_dir / "best.pt")
                print(f"   -> saved best checkpoint vs_sbs (macro_vs_sbs={best_macro_vs_sbs:+.3f}%)")
            if macro["macro_top1"] > best_macro_top1:
                best_macro_top1 = macro["macro_top1"]
                torch.save({"model": model.state_dict(), "args": vars(args), "macro": macro}, save_dir / "best_top1.pt")
                print(f"   -> saved best_top1 (macro_top1={best_macro_top1:.4f})")
            joint = macro["macro_vs_sbs_pct"] - 100.0 * macro["macro_top1"]
            if joint < best_joint:
                best_joint = joint
                torch.save({"model": model.state_dict(), "args": vars(args), "macro": macro}, save_dir / "best_joint.pt")
                print(f"   -> saved best_joint (score={best_joint:+.3f})")
            # === Early stopping checks (R40 plan) ===
            top1_history.append(macro["macro_top1"])
            cur_top1 = macro["macro_top1"]
            wall_h = (time.time() - t0) / 3600.0
            # T1: cold-start failure
            if epoch >= args.early_stop_cold_epoch and cur_top1 < args.early_stop_cold_floor:
                stop_reason = (f"T1 cold-start failure: val_top1={cur_top1:.4f} < "
                               f"{args.early_stop_cold_floor:.4f} at epoch {epoch}")
            # T2: divergence (current drop from historical max)
            elif best_macro_top1 - cur_top1 > args.early_stop_drop:
                stop_reason = (f"T2 divergence: val_top1 dropped {best_macro_top1-cur_top1:+.4f} "
                               f"from best {best_macro_top1:.4f} to {cur_top1:.4f} at epoch {epoch}")
            # T3: plateau (max over last N epochs minus max before that ≤ 0.002)
            elif len(top1_history) >= 2 * args.early_stop_plateau:
                recent = max(top1_history[-args.early_stop_plateau:])
                older = max(top1_history[-2 * args.early_stop_plateau:-args.early_stop_plateau])
                if recent - older < 0.002:
                    stop_reason = (f"T3 plateau: recent max {recent:.4f} vs prior max "
                                   f"{older:.4f} over {args.early_stop_plateau}-epoch windows at epoch {epoch}")
            # T4: wall clock
            elif wall_h > args.early_stop_wall_hours:
                stop_reason = f"T4 wall clock: {wall_h:.2f} h exceeded {args.early_stop_wall_hours} h at epoch {epoch}"
            if stop_reason is not None:
                print(f"[early-stop] {stop_reason}")
                (save_dir / "stopped_reason.txt").write_text(
                    f"stopped_at_epoch={epoch}\nbest_top1={best_macro_top1:.4f}\n"
                    f"reason={stop_reason}\ntop1_history={top1_history}\n")
                break
        # allow break out of outer loop when early-stopped mid-eval
        if stop_reason is not None:
            break

    elapsed = time.time() - t0
    print(f"[done] {args.epochs} epochs in {elapsed/60:.1f} min, best macro_vs_sbs={best_macro_vs_sbs:+.3f}%")
    (save_dir / "summary.json").write_text(json.dumps({
        "elapsed_sec": elapsed,
        "best_macro_vs_sbs_pct": best_macro_vs_sbs,
        "best_macro_top1": best_macro_top1,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "epochs_run": len(top1_history),
        "top1_history": top1_history,
        "task_schedule": args.task_schedule,
        "last_seen_step": last_seen_step,
        "grad_cos_history": grad_cos_history,
    }, indent=2))


if __name__ == "__main__":
    main()
