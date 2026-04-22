from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import UnifiedProblemDataset, collate_single_problem
from .eval_log import format_eval_block
from .model import load_partial_state_dict
from .registry import PROBLEMS, M_GLOBAL
from .rerank_train import (
    set_seed,
    to_device,
    weighted_mean,
    parse_float_grid,
    build_base_model,
    load_or_build_problem_stats,
    gather_problem_stats,
    build_candidate_set,
    rerank_loss,
    apply_rerank,
    summarize_macro,
    evaluate_base_all,
    top1_safe_score,
)


def subsample_instance_tokens(tokens: torch.Tensor, mask: torch.Tensor, max_tokens: int,
                              keep_first: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.shape[1] <= max_tokens:
        return tokens, mask
    keep_tokens = []
    keep_masks = []
    for i in range(tokens.shape[0]):
        n = int(mask[i].sum().item())
        if n <= max_tokens:
            cur_tok = tokens[i, :n]
            cur_mask = torch.ones(cur_tok.shape[0], dtype=torch.bool, device=tokens.device)
            if cur_tok.shape[0] < max_tokens:
                pad = max_tokens - cur_tok.shape[0]
                cur_tok = F.pad(cur_tok, (0, 0, 0, pad))
                cur_mask = F.pad(cur_mask, (0, pad), value=False)
            keep_tokens.append(cur_tok)
            keep_masks.append(cur_mask)
            continue
        if keep_first:
            base = [0]
            remain = max_tokens - 1
            if remain > 0:
                idx = torch.linspace(1, n - 1, steps=remain, device=tokens.device).round().long()
                idx = idx.unique()
                while idx.numel() < remain:
                    tail = torch.arange(1, n, device=tokens.device)
                    idx = torch.unique(torch.cat([idx, tail[: remain - idx.numel()]]))
                sel = torch.cat([torch.tensor(base, device=tokens.device), idx[:remain]])
            else:
                sel = torch.tensor(base, device=tokens.device)
        else:
            sel = torch.linspace(0, n - 1, steps=max_tokens, device=tokens.device).round().long().unique()
            while sel.numel() < max_tokens:
                tail = torch.arange(0, n, device=tokens.device)
                sel = torch.unique(torch.cat([sel, tail[: max_tokens - sel.numel()]]))
        sel = sel[:max_tokens]
        cur_tok = tokens[i].index_select(0, sel)
        cur_mask = torch.ones(cur_tok.shape[0], dtype=torch.bool, device=tokens.device)
        if cur_tok.shape[0] < max_tokens:
            pad = max_tokens - cur_tok.shape[0]
            cur_tok = F.pad(cur_tok, (0, 0, 0, pad))
            cur_mask = F.pad(cur_mask, (0, pad), value=False)
        keep_tokens.append(cur_tok)
        keep_masks.append(cur_mask)
    return torch.stack(keep_tokens, dim=0), torch.stack(keep_masks, dim=0)


def load_or_build_solver_profiles(base_model, device: str, problem_stats: dict,
                                  cache_path: Path | None, profile_batch_size: int = 256):
    if cache_path is not None and cache_path.exists():
        print(f"[profile-cache] loading {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")

    profiles = {}
    d = base_model.d
    for problem in PROBLEMS:
        print(f"[profile-cache] building profiles for {problem}", flush=True)
        ds = UnifiedProblemDataset(problem, "train")
        dl = DataLoader(ds, batch_size=profile_batch_size, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
        sum_win = torch.zeros(M_GLOBAL, d)
        cnt_win = torch.zeros(M_GLOBAL)
        sum_run = torch.zeros(M_GLOBAL, d)
        cnt_run = torch.zeros(M_GLOBAL)
        sum_fail = torch.zeros(M_GLOBAL, d)
        cnt_fail = torch.zeros(M_GLOBAL)
        for batch in dl:
            batch = to_device(batch, device)
            with torch.no_grad():
                h = base_model.encode_instance(batch).detach().cpu()
            costs = batch["costs"].detach().cpu()
            rank = costs.argsort(dim=1)
            pool_ids = batch["pool_ids"].detach().cpu()
            topk = min(3, costs.shape[1])
            for local_idx in range(costs.shape[1]):
                gid = int(pool_ids[local_idx].item())
                win_mask = rank[:, 0] == local_idx
                run_mask = rank[:, 1] == local_idx if costs.shape[1] > 1 else torch.zeros_like(win_mask)
                top_mask = (rank[:, :topk] == local_idx).any(dim=1)
                fail_mask = ~top_mask
                if win_mask.any():
                    sum_win[gid] += h[win_mask].sum(dim=0)
                    cnt_win[gid] += float(win_mask.sum().item())
                if run_mask.any():
                    sum_run[gid] += h[run_mask].sum(dim=0)
                    cnt_run[gid] += float(run_mask.sum().item())
                if fail_mask.any():
                    sum_fail[gid] += h[fail_mask].sum(dim=0)
                    cnt_fail[gid] += float(fail_mask.sum().item())

        cur = problem_stats[problem]
        profile = torch.zeros(M_GLOBAL, 3 * d + 8)
        for local_idx, gid in enumerate(ds.pool_order):
            gid = int(gid)
            win_mean = sum_win[gid] / cnt_win[gid].clamp_min(1.0)
            run_mean = sum_run[gid] / cnt_run[gid].clamp_min(1.0)
            fail_mean = sum_fail[gid] / cnt_fail[gid].clamp_min(1.0)
            stats = torch.tensor([
                cur["oracle_win_prior"][local_idx],
                cur["pick_rate"][local_idx],
                cur["hidden_winner_flag"][local_idx],
                cur["mean_gap_vs_sbs"][local_idx],
                cur["rank_hist"][0][local_idx],
                cur["rank_hist"][1][local_idx],
                cur["rank_hist"][2][local_idx],
                float(cnt_fail[gid].item()) / max(1.0, float(ds.base_N)),
            ], dtype=torch.float32)
            profile[gid] = torch.cat([win_mean, run_mean, fail_mean, stats], dim=0)
        profiles[problem] = profile

    payload = {
        "profile_dim": int(next(iter(profiles.values())).shape[-1]),
        "profiles": profiles,
    }
    if cache_path is not None:
        torch.save(payload, cache_path)
        print(f"[profile-cache] wrote {cache_path}", flush=True)
    return payload


def load_or_build_problem_sample_cache(base_model, device: str, problem_stats: dict,
                                       cache_path: Path | None, hard_gain_pct: float,
                                       rare_prior_thr: float, sample_batch_size: int = 256):
    if cache_path is not None and cache_path.exists():
        print(f"[sample-cache] loading {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu")

    payload = {}
    for problem in PROBLEMS:
        print(f"[sample-cache] building sample stats for {problem}", flush=True)
        ds = UnifiedProblemDataset(problem, "train")
        dl = DataLoader(ds, batch_size=sample_batch_size, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
        cur = problem_stats[problem]
        prior = torch.tensor(cur["oracle_win_prior"], dtype=torch.float32)
        hidden = torch.tensor(cur["hidden_winner_flag"], dtype=torch.float32) > 0
        base_wrong_all = []
        switch_gain_all = []
        winner_all = []
        rare_all = []
        hard_all = []
        for batch in dl:
            batch = to_device(batch, device)
            with torch.no_grad():
                logits = base_model(batch)
                logits_pool = logits[:, batch["pool_ids"]]
                base_pred = logits_pool.argmax(dim=1).cpu()
            costs = batch["costs"].detach().cpu()
            best = costs.argmin(dim=1)
            base_wrong = base_pred != best
            base_cost = costs.gather(1, base_pred.unsqueeze(1)).squeeze(1)
            oracle_cost = costs.gather(1, best.unsqueeze(1)).squeeze(1)
            switch_gain_pct = ((base_cost - oracle_cost).clamp_min(0.0) / (base_cost.abs() + 1e-9)) * 100.0
            rare = hidden[best] | ((prior[best] > 0) & (prior[best] <= rare_prior_thr))
            hard = base_wrong & (switch_gain_pct >= hard_gain_pct)
            base_wrong_all.append(base_wrong)
            switch_gain_all.append(switch_gain_pct)
            winner_all.append(best)
            rare_all.append(rare)
            hard_all.append(hard)
        payload[problem] = {
            "base_wrong": torch.cat(base_wrong_all).to(torch.bool),
            "switch_gain_pct": torch.cat(switch_gain_all).to(torch.float32),
            "winner_local": torch.cat(winner_all).to(torch.long),
            "rare_mask": torch.cat(rare_all).to(torch.bool),
            "hard_mask": torch.cat(hard_all).to(torch.bool),
        }
    full = {
        "hard_gain_pct": hard_gain_pct,
        "rare_prior_thr": rare_prior_thr,
        "problems": payload,
    }
    if cache_path is not None:
        torch.save(full, cache_path)
        print(f"[sample-cache] wrote {cache_path}", flush=True)
    return full


def build_replay_sampler(problem: str, base_N: int, coord_augment: int, sample_cache: dict,
                         rare_frac: float, hard_frac: float, num_samples: int):
    cur = sample_cache["problems"][problem]
    rare = cur["rare_mask"].float()
    hard = cur["hard_mask"].float()
    normal = (1.0 - torch.clamp(rare + hard, max=1.0)).float()
    w = torch.zeros(base_N, dtype=torch.float32)
    normal_frac = max(0.0, 1.0 - rare_frac - hard_frac)
    if normal.sum() > 0 and normal_frac > 0:
        w += normal * (normal_frac / normal.sum().item())
    if rare.sum() > 0 and rare_frac > 0:
        w += rare * (rare_frac / rare.sum().item())
    if hard.sum() > 0 and hard_frac > 0:
        w += hard * (hard_frac / hard.sum().item())
    w = w / w.mean().clamp_min(1e-6)
    if coord_augment > 1:
        w = w.repeat(coord_augment)
    return WeightedRandomSampler(weights=w, num_samples=num_samples, replacement=True)


class ResidualCrossBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.cross_scale_raw = nn.Parameter(torch.tensor(-7.0))
        self.ffn_scale_raw = nn.Parameter(torch.tensor(-7.0))

    def forward(self, q: torch.Tensor, inst: torch.Tensor, inst_mask: torch.Tensor):
        cross, _ = self.cross_attn(self.norm1(q), inst, inst, key_padding_mask=~inst_mask, need_weights=False)
        q = q + F.softplus(self.cross_scale_raw) * cross
        q = q + F.softplus(self.ffn_scale_raw) * self.ffn(self.norm2(q))
        return q


class SolverContextResidualReranker(nn.Module):
    def __init__(self, cand_dim: int, inst_dim: int, solver_dim: int, profile_dim: int,
                 cond_dim: int, d_model: int = 128, heads: int = 4, cross_layers: int = 1,
                 self_layers: int = 2, dropout: float = 0.1, max_topk: int = 10,
                 delta_init_scale: float = 1e-4):
        super().__init__()
        self.cand_in = nn.Sequential(nn.Linear(cand_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.inst_in = nn.Sequential(nn.Linear(inst_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.solver_profile = nn.Sequential(
            nn.Linear(solver_dim + profile_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.inst_cond_proj = nn.Linear(d_model, d_model)
        self.cond_film = nn.Linear(d_model, 2 * d_model)
        self.query_ln = nn.LayerNorm(d_model)
        self.inst_ln = nn.LayerNorm(d_model)
        self.pos_emb = nn.Embedding(max_topk, d_model)
        self.cross_blocks = nn.ModuleList([ResidualCrossBlock(d_model, heads, dropout) for _ in range(cross_layers)])
        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=2 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(self_layers)
        ])
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.gap_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.delta_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(max(delta_init_scale, 1e-6))), dtype=torch.float32))
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, cand_feats: torch.Tensor, inst_tokens: torch.Tensor, inst_mask: torch.Tensor,
                solver_embs: torch.Tensor, solver_profiles: torch.Tensor, cond_vec: torch.Tensor):
        bsz, k, _ = cand_feats.shape
        pos = torch.arange(k, device=cand_feats.device).unsqueeze(0).expand(bsz, k)
        cand = self.cand_in(cand_feats) + self.pos_emb(pos)
        inst = self.inst_in(inst_tokens)
        solver_ctx = self.solver_profile(torch.cat([solver_embs, solver_profiles], dim=-1))
        cond = self.cond_proj(cond_vec)
        film_gain, film_bias = torch.tanh(self.cond_film(cond)).chunk(2, dim=-1)
        q = self.query_ln(cand + solver_ctx + cond.unsqueeze(1))
        q = q * (1.0 + 0.10 * film_gain.unsqueeze(1)) + 0.10 * film_bias.unsqueeze(1)
        inst = self.inst_ln(inst + self.inst_cond_proj(cond).unsqueeze(1))
        for block in self.cross_blocks:
            q = block(q, inst, inst_mask)
        for block in self.self_blocks:
            q = block(q)
        delta = F.softplus(self.delta_scale_raw) * torch.tanh(self.delta_head(q).squeeze(-1))
        gap = self.gap_head(q).squeeze(-1)
        return delta, gap


def build_context_meta(base_model, batch, problem_stats: dict, solver_profiles: dict, audit: dict,
                       topk: int, instance_max_tokens: int, base_topk: int | None = None,
                       support_topk: int = 0, full_pool: bool = False,
                       support_feature_mode: str = "per_generator"):
    meta = build_candidate_set(
        base_model,
        batch,
        problem_stats=problem_stats,
        audit=audit,
        topk=topk,
        base_topk=base_topk,
        support_topk=support_topk,
        full_pool=full_pool,
        support_feature_mode=support_feature_mode,
    )
    with torch.no_grad():
        _, inst_tokens, inst_mask = base_model.encode_instance_with_tokens(batch)
    inst_tokens, inst_mask = subsample_instance_tokens(
        inst_tokens,
        inst_mask,
        max_tokens=instance_max_tokens,
        keep_first=(batch["kind"] == "coord"),
    )
    bsz = meta["token_feats"].shape[0]
    pool_ids = batch["pool_ids"].unsqueeze(0).expand(bsz, -1)
    cand_solver_ids = torch.gather(pool_ids, 1, meta["top_idx"])
    solver_embs = base_model.solver_emb(cand_solver_ids)
    profile_table = solver_profiles[meta["problem"]].to(meta["token_feats"].device)
    solver_profiles_tok = profile_table.unsqueeze(0).expand(bsz, -1, -1)
    solver_profiles_tok = torch.gather(
        solver_profiles_tok,
        1,
        cand_solver_ids.unsqueeze(-1).expand(-1, -1, profile_table.shape[-1]),
    )
    pid = torch.full((bsz,), batch["problem_id"], dtype=torch.long, device=meta["token_feats"].device)
    cond_vec = torch.cat([
        base_model.prob_emb(pid),
        batch["cbits"],
        base_model.compute_global_stats(batch),
        base_model.compute_manual_features(batch),
    ], dim=1)
    meta["inst_tokens"] = inst_tokens
    meta["inst_mask"] = inst_mask
    meta["cand_solver_ids"] = cand_solver_ids
    meta["solver_embs"] = solver_embs
    meta["solver_profiles"] = solver_profiles_tok
    meta["cond_vec"] = cond_vec
    return meta


@torch.no_grad()
def evaluate_problem(base_model, reranker, problem: str, split: str, device: str, audit: dict,
                     problem_stats: dict, solver_profiles: dict, topk: int, instance_max_tokens: int,
                     blend_alpha: float = 1.0, base_gap_max: float = float("inf"),
                     rerank_margin_min: float = 0.0, base_topk: int | None = None,
                     support_topk: int = 0, full_pool: bool = False,
                     support_feature_mode: str = "per_generator") -> dict:
    ds = UnifiedProblemDataset(problem, split)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_single_problem)
    top1 = top2 = top3 = 0
    cost_sum = 0.0
    total = 0
    applied = 0
    fixable_recovered = 0
    fixable_total = 0
    safe_harm = 0
    safe_total = 0
    decisive_hit = {0.1: [0, 0], 0.5: [0, 0], 1.0: [0, 0]}
    oracle_rank_sum = 0.0
    oracle_rank_le2 = 0
    oracle_rank_le3 = 0
    oracle_support_mass = 0.0
    picks = []

    for batch in dl:
        batch = to_device(batch, device)
        meta = build_context_meta(
            base_model,
            batch,
            problem_stats=problem_stats,
            solver_profiles=solver_profiles,
            audit=audit,
            topk=topk,
            instance_max_tokens=instance_max_tokens,
            base_topk=base_topk,
            support_topk=support_topk,
            full_pool=full_pool,
            support_feature_mode=support_feature_mode,
        )
        delta, _ = reranker(
            meta["token_feats"],
            meta["inst_tokens"],
            meta["inst_mask"],
            meta["solver_embs"],
            meta["solver_profiles"],
            meta["cond_vec"],
        )
        final_pool, gate, _ = apply_rerank(
            meta,
            delta,
            blend_alpha=blend_alpha,
            base_gap_max=base_gap_max,
            rerank_margin_min=rerank_margin_min,
        )
        pred = final_pool.argmax(dim=1)
        best = batch["costs"].argmin(dim=1)
        pred_rank = final_pool.argsort(dim=1, descending=True)
        top1 += (pred == best).sum().item()
        top2 += (pred_rank[:, :min(2, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        top3 += (pred_rank[:, :min(3, pred_rank.shape[1])] == best.unsqueeze(1)).any(dim=1).sum().item()
        sel_cost = batch["costs"].gather(1, pred.unsqueeze(1)).squeeze(1)
        cost_sum += sel_cost.sum().item()
        total += batch["costs"].shape[0]
        applied += int(gate.sum().item())

        fixable = meta["base_wrong"]
        safe = ~fixable
        if fixable.any():
            fixable_recovered += (pred[fixable] == best[fixable]).sum().item()
            fixable_total += int(fixable.sum().item())
        if safe.any():
            safe_harm += (pred[safe] != best[safe]).sum().item()
            safe_total += int(safe.sum().item())
        switch_gain_pct = meta["switch_gain_pct"]
        for thr in decisive_hit:
            mask = fixable & (switch_gain_pct > thr)
            if mask.any():
                decisive_hit[thr][0] += int(mask.sum().item())
                decisive_hit[thr][1] += int((pred[mask] == best[mask]).sum().item())
        rank_pos = (pred_rank == best.unsqueeze(1)).float().argmax(dim=1) + 1
        oracle_rank_sum += rank_pos.float().sum().item()
        oracle_rank_le2 += int((rank_pos <= 2).sum().item())
        oracle_rank_le3 += int((rank_pos <= 3).sum().item())
        oracle_support_mass += float(meta["oracle_in_topk"].float().sum().item())
        picks.append(pred.cpu().numpy())

    pred = np.concatenate(picks)
    pool_names = audit[problem]["pool"]
    ds_costs = [ds.labels[str(i)]["cost"][:ds.K_p] for i in range(ds.base_N)]
    costs_np = np.asarray(ds_costs, dtype=np.float32)
    best_idx_np = costs_np.argmin(axis=1)
    method_top1 = {name: float((best_idx_np == k).mean()) for k, name in enumerate(pool_names)}
    method_mean = {name: float(costs_np[:, k].mean()) for k, name in enumerate(pool_names)}
    arm_dist = {name: float((pred == k).mean()) for k, name in enumerate(pool_names)}
    final_arm_coverage = float(np.mean([v > 0 for v in arm_dist.values()]))
    pick_entropy = float(-(np.array(list(arm_dist.values())) * np.log(np.array(list(arm_dist.values())) + 1e-12)).sum() / np.log(max(2, len(pool_names))))
    zero_pick_count = int(sum(v <= 0 for v in arm_dist.values()))
    hidden_winners = [name for name in pool_names if method_top1[name] > 0 and arm_dist[name] <= 0]
    hidden_winner_count = len(hidden_winners)
    hidden_winner_mass = float(sum(method_top1[name] for name in hidden_winners))
    mean_cost = cost_sum / max(1, total)
    sbs_cost = audit[problem][split]["sbs_mean"]
    vbs_mean = audit[problem][split]["vbs_mean"]
    return {
        "top1": top1 / total,
        "top2": top2 / total,
        "top3": top3 / total,
        "mean_cost": mean_cost,
        "sbs_cost": sbs_cost,
        "vbs_mean": vbs_mean,
        "vs_sbs_pct": (mean_cost - sbs_cost) / (abs(sbs_cost) + 1e-9) * 100,
        "vbs_gap_closed_pct": ((sbs_cost - mean_cost) / (sbs_cost - vbs_mean + 1e-9)) * 100 if sbs_cost != vbs_mean else 0.0,
        "rerank_apply_rate": applied / total,
        "fixable_recovery_rate": fixable_recovered / fixable_total if fixable_total > 0 else 0.0,
        "safe_harm_rate": safe_harm / safe_total if safe_total > 0 else 0.0,
        "n_fixable": fixable_total,
        "n_safe": safe_total,
        "oracle_rank_mean": oracle_rank_sum / total,
        "oracle_rank_top2_rate": oracle_rank_le2 / total,
        "oracle_rank_top3_rate": oracle_rank_le3 / total,
        "top1_decisive_gap_0.1": decisive_hit[0.1][1] / decisive_hit[0.1][0] if decisive_hit[0.1][0] > 0 else 0.0,
        "top1_decisive_gap_0.5": decisive_hit[0.5][1] / decisive_hit[0.5][0] if decisive_hit[0.5][0] > 0 else 0.0,
        "top1_decisive_gap_1.0": decisive_hit[1.0][1] / decisive_hit[1.0][0] if decisive_hit[1.0][0] > 0 else 0.0,
        "arm_distribution": arm_dist,
        "final_arm_coverage": final_arm_coverage,
        "pick_entropy": pick_entropy,
        "zero_pick_count": zero_pick_count,
        "hidden_winner_count": hidden_winner_count,
        "hidden_winner_mass": hidden_winner_mass,
        "oracle_support_mass_recall": oracle_support_mass / total,
        "method_top1": method_top1,
        "method_mean": method_mean,
    }


@torch.no_grad()
def evaluate_all(base_model, reranker, split: str, device: str, audit: dict, problem_stats: dict,
                 solver_profiles: dict, topk: int, instance_max_tokens: int,
                 blend_alpha: float = 1.0, base_gap_max: float = float("inf"),
                 rerank_margin_min: float = 0.0, base_topk: int | None = None,
                 support_topk: int = 0, full_pool: bool = False,
                 support_feature_mode: str = "per_generator"):
    per = {}
    for problem in PROBLEMS:
        per[problem] = evaluate_problem(
            base_model, reranker, problem, split, device, audit, problem_stats, solver_profiles,
            topk=topk, instance_max_tokens=instance_max_tokens, blend_alpha=blend_alpha,
            base_gap_max=base_gap_max, rerank_margin_min=rerank_margin_min, base_topk=base_topk,
            support_topk=support_topk, full_pool=full_pool, support_feature_mode=support_feature_mode,
        )
    return {"per_problem": per, "macro": summarize_macro(per)}


@torch.no_grad()
def sweep_gate(base_model, reranker, device: str, audit: dict, problem_stats: dict, solver_profiles: dict,
               topk: int, instance_max_tokens: int, blend_alphas: list[float], base_gap_grid: list[float],
               rerank_margin_grid: list[float], base_topk: int | None = None, support_topk: int = 0,
               full_pool: bool = False, support_feature_mode: str = "per_generator"):
    best = None
    for alpha in blend_alphas:
        for gap in base_gap_grid:
            for rerank_margin in rerank_margin_grid:
                out = evaluate_all(
                    base_model, reranker, "val", device, audit, problem_stats, solver_profiles,
                    topk=topk, instance_max_tokens=instance_max_tokens, blend_alpha=alpha,
                    base_gap_max=gap, rerank_margin_min=rerank_margin, base_topk=base_topk,
                    support_topk=support_topk, full_pool=full_pool, support_feature_mode=support_feature_mode,
                )
                macro = out["macro"]
                key = (-macro["macro_top1"], macro["macro_vs_sbs_pct"], macro["macro_safe_harm_rate"], -macro["macro_fixable_recovery_rate"])
                if best is None or key < best["key"]:
                    best = {"key": key, "config": {"blend_alpha": alpha, "base_gap_max": gap, "rerank_margin_min": rerank_margin}, "val": out}
    assert best is not None
    return best


def evaluate_checkpoint(tag: str, save_dir: Path, base_model, reranker, device: str, audit: dict,
                        problem_stats: dict, solver_profiles: dict, args):
    best_gate = sweep_gate(
        base_model, reranker, device, audit, problem_stats, solver_profiles,
        topk=args.rerank_topk, instance_max_tokens=args.instance_max_tokens,
        blend_alphas=parse_float_grid(args.blend_alpha_grid, [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]),
        base_gap_grid=parse_float_grid(args.base_gap_grid, [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, float("inf")]),
        rerank_margin_grid=parse_float_grid(args.rerank_margin_grid, [0.0, 0.01, 0.02, 0.05]),
        base_topk=args.base_topk, support_topk=args.support_candidate_topk,
        full_pool=args.full_pool, support_feature_mode=args.support_feature_mode,
    )
    (save_dir / f"best_gate_{tag}.json").write_text(json.dumps(best_gate, indent=2))
    for split in ["val", "test"]:
        out = evaluate_all(
            base_model, reranker, split, device, audit, problem_stats, solver_profiles,
            topk=args.rerank_topk, instance_max_tokens=args.instance_max_tokens,
            blend_alpha=best_gate["config"]["blend_alpha"],
            base_gap_max=best_gate["config"]["base_gap_max"],
            rerank_margin_min=best_gate["config"]["rerank_margin_min"],
            base_topk=args.base_topk, support_topk=args.support_candidate_topk,
            full_pool=args.full_pool, support_feature_mode=args.support_feature_mode,
        )
        print(
            format_eval_block(
                f"[{split} {tag}]",
                out["macro"],
                out["per_problem"],
                problem_order=PROBLEMS,
            ),
            flush=True,
        )
        (save_dir / f"analysis_{split}_{tag}.json").write_text(json.dumps(out, indent=2))
        if tag == "best_top1_safe":
            (save_dir / f"analysis_{split}.json").write_text(json.dumps(out, indent=2))


def balanced_aux_loss(final_scores: torch.Tensor, winner_idx: torch.Tensor, prior_tok: torch.Tensor,
                      example_weight: torch.Tensor | None, ce_mask: torch.Tensor,
                      bal_lambda: float) -> torch.Tensor:
    if not ce_mask.any():
        return final_scores.new_tensor(0.0)
    adjusted = final_scores + bal_lambda * prior_tok.clamp_min(1e-6).log()
    raw = F.cross_entropy(adjusted, winner_idx, reduction="none")
    weight = example_weight[ce_mask] if example_weight is not None else None
    return weighted_mean(raw[ce_mask], weight)


def rare_head_pair_loss(final_scores: torch.Tensor, winner_idx: torch.Tensor, base_pred_idx: torch.Tensor,
                        rare_focus: torch.Tensor, example_weight: torch.Tensor | None) -> torch.Tensor:
    if not rare_focus.any():
        return final_scores.new_tensor(0.0)
    rows = torch.arange(final_scores.shape[0], device=final_scores.device)
    winner_score = final_scores[rows, winner_idx]
    base_score = final_scores[rows, base_pred_idx]
    raw = F.softplus(-(winner_score - base_score))
    weight = example_weight[rare_focus] if example_weight is not None else None
    return weighted_mean(raw[rare_focus], weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-per-problem", type=int, default=256)
    ap.add_argument("--instance-max-tokens", type=int, default=64)
    ap.add_argument("--profile-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rerank-topk", type=int, default=10)
    ap.add_argument("--rerank-d-model", type=int, default=128)
    ap.add_argument("--rerank-heads", type=int, default=4)
    ap.add_argument("--rerank-layers", type=int, default=2)
    ap.add_argument("--rerank-cross-layers", type=int, default=1)
    ap.add_argument("--rerank-pair-weight", type=float, default=0.30)
    ap.add_argument("--gap-weight", type=float, default=0.20)
    ap.add_argument("--rerank-ce-weight", type=float, default=0.18)
    ap.add_argument("--anchor-weight", type=float, default=0.50)
    ap.add_argument("--anchor-mode", choices=["kl", "delta"], default="kl")
    ap.add_argument("--base-topk", type=int, default=2)
    ap.add_argument("--support-candidate-topk", type=int, default=10)
    ap.add_argument("--rare-winner-weight", type=float, default=1.0)
    ap.add_argument("--hidden-winner-weight", type=float, default=1.5)
    ap.add_argument("--switch-gain-weight", type=float, default=0.20)
    ap.add_argument("--coord-augment", type=int, default=0)
    ap.add_argument("--ce-fixable-only", action="store_true")
    ap.add_argument("--pair-min-gain-pct", type=float, default=0.10)
    ap.add_argument("--ce-min-gain-pct", type=float, default=0.05)
    ap.add_argument("--decisive-gain-pct", type=float, default=0.20)
    ap.add_argument("--decisive-upweight", type=float, default=1.5)
    ap.add_argument("--delta-init-scale", type=float, default=1e-4)
    ap.add_argument("--blend-alpha-grid", default="0.05,0.10,0.15,0.25,0.40,0.60")
    ap.add_argument("--base-gap-grid", default="0.01,0.02,0.03,0.05,0.08,0.12,inf")
    ap.add_argument("--rerank-margin-grid", default="0.0,0.01,0.02,0.05")
    ap.add_argument("--max-batches-per-problem", type=int, default=0)
    ap.add_argument("--cost-slack", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full-pool", action="store_true")
    ap.add_argument("--support-feature-mode", choices=["none", "max", "per_generator"], default="per_generator")
    ap.add_argument("--prior-cache", type=str, default="")
    ap.add_argument("--profile-cache", type=str, default="")
    ap.add_argument("--sample-cache", type=str, default="")
    ap.add_argument("--rare-prior-thr", type=float, default=0.08)
    ap.add_argument("--hard-replay-gain-pct", type=float, default=0.20)
    ap.add_argument("--rare-replay-frac", type=float, default=0.20)
    ap.add_argument("--hard-replay-frac", type=float, default=0.10)
    ap.add_argument("--balanced-ce-weight", type=float, default=0.0)
    ap.add_argument("--balanced-ce-lambda", type=float, default=0.25)
    ap.add_argument("--rare-head-pair-weight", type=float, default=0.0)
    ap.add_argument("--init-reranker-ckpt", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    base_model, _ = build_base_model(args.base_ckpt, args.device)
    audit = json.loads(Path("code/unified_selector/runs/audit.json").read_text())
    prior_cache = Path(args.prior_cache) if args.prior_cache else save_dir / "problem_solver_stats.json"
    problem_stats = load_or_build_problem_stats(base_model, audit, args.device, prior_cache)
    profile_cache = Path(args.profile_cache) if args.profile_cache else save_dir / "solver_profiles.pt"
    profile_payload = load_or_build_solver_profiles(
        base_model, args.device, problem_stats, profile_cache, profile_batch_size=args.profile_batch_size
    )
    solver_profiles = profile_payload["profiles"]
    sample_cache_path = Path(args.sample_cache) if args.sample_cache else save_dir / "problem_sample_cache.pt"
    sample_cache = load_or_build_problem_sample_cache(
        base_model,
        args.device,
        problem_stats,
        sample_cache_path,
        hard_gain_pct=args.hard_replay_gain_pct,
        rare_prior_thr=args.rare_prior_thr,
        sample_batch_size=args.profile_batch_size,
    )
    base_val = evaluate_base_all(base_model, "val", args.device, audit)
    (save_dir / "base_val_metrics.json").write_text(json.dumps(base_val, indent=2))

    sample_ds = UnifiedProblemDataset(PROBLEMS[0], "train")
    sample_batch = collate_single_problem([sample_ds[0], sample_ds[1]])
    sample_batch = to_device(sample_batch, args.device)
    sample_meta = build_context_meta(
        base_model, sample_batch, problem_stats, solver_profiles, audit,
        topk=args.rerank_topk, instance_max_tokens=args.instance_max_tokens,
        base_topk=args.base_topk, support_topk=args.support_candidate_topk,
        full_pool=args.full_pool, support_feature_mode=args.support_feature_mode,
    )
    cond_dim = sample_meta["cond_vec"].shape[-1]
    reranker = SolverContextResidualReranker(
        cand_dim=sample_meta["token_feats"].shape[-1],
        inst_dim=sample_meta["inst_tokens"].shape[-1],
        solver_dim=sample_meta["solver_embs"].shape[-1],
        profile_dim=sample_meta["solver_profiles"].shape[-1],
        cond_dim=cond_dim,
        d_model=args.rerank_d_model,
        heads=args.rerank_heads,
        cross_layers=args.rerank_cross_layers,
        self_layers=args.rerank_layers,
        max_topk=max(args.rerank_topk, 2),
        delta_init_scale=args.delta_init_scale,
    ).to(args.device)
    if args.init_reranker_ckpt:
        init_ckpt = torch.load(args.init_reranker_ckpt, map_location=args.device)
        init_state = init_ckpt.get("reranker", init_ckpt)
        missing, unexpected, skipped, remapped = load_partial_state_dict(reranker, init_state)
        print(
            f"[init reranker] loaded {args.init_reranker_ckpt} missing={missing} unexpected={unexpected} "
            f"skipped={skipped} remapped={remapped}",
            flush=True,
        )
    opt = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=args.wd)

    winner_weights = {}
    hidden_weights = {}
    for problem in PROBLEMS:
        cur = gather_problem_stats(problem, problem_stats, args.device)
        w = (cur["oracle_win_prior"] + 0.01).pow(-0.5)
        winner_weights[problem] = w / w.mean().clamp_min(1e-6)
        hidden_weights[problem] = 1.0 + args.hidden_winner_weight * cur["hidden_winner_flag"]

    best_top1_safe = -1e18
    best_cost = 1e18
    step = 0

    for epoch in range(args.epochs):
        for problem in PROBLEMS:
            ds = UnifiedProblemDataset(problem, "train", coord_augment=args.coord_augment)
            sampler = build_replay_sampler(
                problem=problem,
                base_N=ds.base_N,
                coord_augment=(ds.coord_augment if ds.coord_augment > 1 else 1),
                sample_cache=sample_cache,
                rare_frac=args.rare_replay_frac,
                hard_frac=args.hard_replay_frac,
                num_samples=max(ds.N, args.batch_per_problem),
            )
            dl = DataLoader(
                ds,
                batch_size=args.batch_per_problem,
                shuffle=False,
                sampler=sampler,
                num_workers=0,
                collate_fn=collate_single_problem,
                drop_last=True,
            )
            for i, batch in enumerate(dl):
                if args.max_batches_per_problem > 0 and i >= args.max_batches_per_problem:
                    break
                batch = to_device(batch, args.device)
                with torch.no_grad():
                    meta = build_context_meta(
                        base_model, batch, problem_stats, solver_profiles, audit,
                        topk=args.rerank_topk, instance_max_tokens=args.instance_max_tokens,
                        base_topk=args.base_topk, support_topk=args.support_candidate_topk,
                        full_pool=args.full_pool, support_feature_mode=args.support_feature_mode,
                    )
                delta, gap_pred = reranker(
                    meta["token_feats"], meta["inst_tokens"], meta["inst_mask"],
                    meta["solver_embs"], meta["solver_profiles"], meta["cond_vec"],
                )
                base_wrong = meta["base_wrong"] if args.full_pool else meta["base_wrong"] & meta["oracle_in_topk"]
                pair_focus = base_wrong & (meta["switch_gain_pct"] >= args.pair_min_gain_pct)
                if args.ce_fixable_only:
                    ce_focus = base_wrong.clone()
                else:
                    ce_focus = torch.ones_like(base_wrong, dtype=torch.bool)
                if args.ce_min_gain_pct > 0:
                    ce_focus = ce_focus & (meta["switch_gain_pct"] >= args.ce_min_gain_pct)
                safe = ~base_wrong
                winner = batch["costs"].argmin(dim=1)
                ex_weight = winner_weights[problem][winner]
                ex_weight = 1.0 + args.rare_winner_weight * (ex_weight - 1.0)
                ex_weight = ex_weight * hidden_weights[problem][winner]
                if args.switch_gain_weight > 0:
                    gain_scale = meta["switch_gain_pct"] / meta["switch_gain_pct"].median().clamp_min(1e-6)
                    ex_weight = ex_weight * (1.0 + args.switch_gain_weight * gain_scale * base_wrong.float())
                if args.decisive_upweight > 0:
                    decisive = base_wrong & (meta["switch_gain_pct"] >= args.decisive_gain_pct)
                    ex_weight = ex_weight * (1.0 + args.decisive_upweight * decisive.float())
                loss, parts = rerank_loss(
                    delta, gap_pred, meta["cand_base"], meta["cand_gap_target"], meta["cand_costs"],
                    example_weight=ex_weight, pair_mask=pair_focus, ce_mask=ce_focus, safe_mask=safe,
                    pair_weight=args.rerank_pair_weight, gap_weight=args.gap_weight,
                    ce_weight=args.rerank_ce_weight, anchor_weight=args.anchor_weight, anchor_mode=args.anchor_mode,
                )
                final_scores = meta["cand_base"] + delta
                if args.balanced_ce_weight > 0:
                    loss_bal = balanced_aux_loss(
                        final_scores=final_scores,
                        winner_idx=meta["cand_costs"].argmin(dim=1),
                        prior_tok=meta["cand_oracle_prior"],
                        example_weight=ex_weight,
                        ce_mask=ce_focus,
                        bal_lambda=args.balanced_ce_lambda,
                    )
                    loss = loss + args.balanced_ce_weight * loss_bal
                    parts["bal_ce"] = float(loss_bal.item())
                else:
                    parts["bal_ce"] = 0.0
                if args.rare_head_pair_weight > 0:
                    winner_idx = meta["cand_costs"].argmin(dim=1)
                    winner_hidden = meta["cand_hidden_flag"].gather(1, winner_idx.unsqueeze(1)).squeeze(1) > 0
                    winner_prior = meta["cand_oracle_prior"].gather(1, winner_idx.unsqueeze(1)).squeeze(1)
                    rare_focus = base_wrong & (winner_hidden | ((winner_prior > 0) & (winner_prior <= args.rare_prior_thr)))
                    loss_rare_head = rare_head_pair_loss(
                        final_scores=final_scores,
                        winner_idx=winner_idx,
                        base_pred_idx=meta["base_pred_cand_pos"],
                        rare_focus=rare_focus,
                        example_weight=ex_weight,
                    )
                    loss = loss + args.rare_head_pair_weight * loss_rare_head
                    parts["rare_head"] = float(loss_rare_head.item())
                else:
                    parts["rare_head"] = 0.0
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    dscale = float(F.softplus(reranker.delta_scale_raw).item())
                    print(
                        f"ep{epoch} step{step} prob={problem} loss={loss.item():.4f} "
                        f"pair={parts['pair']:.4f} gap={parts['gap']:.4f} ce={parts['ce']:.4f} "
                        f"anchor={parts['anchor']:.4f} bal_ce={parts['bal_ce']:.4f} "
                        f"rare_head={parts['rare_head']:.4f} dscale={dscale:.6f}",
                        flush=True,
                    )

        val_out = evaluate_all(
            base_model, reranker, "val", args.device, audit, problem_stats, solver_profiles,
            topk=args.rerank_topk, instance_max_tokens=args.instance_max_tokens,
            base_topk=args.base_topk, support_topk=args.support_candidate_topk,
            full_pool=args.full_pool, support_feature_mode=args.support_feature_mode,
        )
        macro = val_out["macro"]
        score = top1_safe_score(
            macro,
            baseline_vs_sbs=base_val["macro"]["macro_vs_sbs_pct"],
            baseline_hidden_mass=base_val["macro"]["macro_hidden_winner_mass"],
            cost_slack=args.cost_slack,
        )
        print(
            format_eval_block(
                f"[eval epoch {epoch}]",
                macro,
                val_out["per_problem"],
                problem_order=PROBLEMS,
                extra_macro_fields=[("score", f"{score:.6f}")],
            ),
            flush=True,
        )
        payload = {
            "reranker": reranker.state_dict(),
            "base_ckpt": args.base_ckpt,
            "args": vars(args),
            "val_macro": macro,
            "top1_safe_score": score,
        }
        if macro["macro_vs_sbs_pct"] < best_cost:
            best_cost = macro["macro_vs_sbs_pct"]
            torch.save(payload, save_dir / "best_cost.pt")
            print(f"  -> saved best_cost (macro_vs_sbs={best_cost:+.3f}%)", flush=True)
        if score > best_top1_safe:
            best_top1_safe = score
            torch.save(payload, save_dir / "best_top1_safe.pt")
            torch.save(payload, save_dir / "best.pt")
            print(f"  -> saved best_top1_safe (score={best_top1_safe:.6f})", flush=True)

    for tag in ["best_top1_safe", "best_cost"]:
        ckpt_path = save_dir / f"{tag}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=args.device)
        reranker.load_state_dict(ckpt["reranker"])
        evaluate_checkpoint(tag, save_dir, base_model, reranker, args.device, audit, problem_stats, solver_profiles, args)


if __name__ == "__main__":
    main()
