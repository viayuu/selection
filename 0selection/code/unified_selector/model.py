"""Unified selector model.

Implements the 10-line forward pass from RESEARCH_REVIEW.md §1:
    1) Adapter (coord-GAT or matrix-stats for ATSP)
    2) Pooled global feature g_b
    3) h_b = LN(W_g g_b + E_prob + W_v cbits + W_q coord_dist)
    4) per-solver pair feature [h, e_s, h*e_s, |h-e_s|]
    5) MLP head -> logits
    6) MVRP factorized add-on (gated by is_mvrp)
    7) masked softmax over M_global solvers (via additive -inf for unavailable)
    8-10) regret-soft labels + listwise CE
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import PROBLEMS, P2I, M_GLOBAL, K_CBITS, D_COORD


class CoordEncoder(nn.Module):
    """Two-layer GAT-ish encoder over padded node sequence (includes depot/feature heads)."""

    def __init__(self, d_in_max: int = 8, d: int = 128, depth: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_in_max = d_in_max
        self.d = d
        self.in_proj = nn.Linear(d_in_max, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2*d, dropout=dropout,
                                       batch_first=True, norm_first=True, activation="gelu")
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d)

    def forward(self, node, node_mask):
        # node: (B, N, d_in) — features may have d_in < d_in_max; pad right with zeros
        B, N, d_in = node.shape
        if d_in < self.d_in_max:
            pad = torch.zeros(B, N, self.d_in_max - d_in, device=node.device, dtype=node.dtype)
            node = torch.cat([node, pad], dim=-1)
        elif d_in > self.d_in_max:
            node = node[..., :self.d_in_max]
        x = self.in_proj(node)
        # key_padding_mask: True where PAD
        kpm = ~node_mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=kpm)
        x = self.out_norm(x)
        # Masked mean-pool
        m = node_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))  # (B, d)
        return pooled


class MatrixEncoder(nn.Module):
    """Simple ATSP matrix encoder: summary features + MLP.  Pilot-level; MatNet-proper can replace later."""

    def __init__(self, d: int = 128):
        super().__init__()
        # Stats: mean, std, max, min, skew, symmetry_ratio, row_entropy_mean, col_entropy_mean, TI-violation-rate, diag_max
        self.proj = nn.Sequential(
            nn.Linear(10, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.LayerNorm(d),
        )

    def forward(self, mat, node_mask):
        # mat: (B, N, N), node_mask: (B, N). Use only valid n subrange.
        B, N, _ = mat.shape
        feats = []
        for i in range(B):
            n = int(node_mask[i].sum().item())
            m = mat[i, :n, :n].clamp(min=0, max=1e3)
            off = m[~torch.eye(n, dtype=torch.bool, device=m.device)]
            mean = off.mean(); std = off.std().clamp_min(1e-6)
            mx = off.max(); mn = off.min()
            skew = ((off - mean) ** 3).mean() / (std ** 3 + 1e-9)
            sym = (m - m.t()).abs().mean() / (mean + 1e-6)
            row_std = m.std(dim=1).mean()
            col_std = m.std(dim=0).mean()
            # Triangle inequality violation rate (approx by sampling)
            k = min(256, n*n)
            idx = torch.randint(0, n, (k, 3), device=m.device)
            a, b, c = idx[:,0], idx[:,1], idx[:,2]
            ti = (m[a,b] > (m[a,c] + m[c,b])).float().mean()
            diag_max = m.diag().abs().max()
            feats.append(torch.stack([mean, std, mx, mn, skew, sym, row_std, col_std, ti, diag_max]))
        x = torch.stack(feats)  # (B, 10)
        # Normalize roughly
        x = x / (x.abs().mean(dim=0, keepdim=True) + 1e-6)
        return self.proj(x)


class UnifiedSelector(nn.Module):
    def __init__(self, d: int = 128, depth: int = 4, dropout: float = 0.1,
                 d_coord_in: int = 8, head_hidden: int = 256,
                 use_mvrp_factorized: bool = True,
                 use_problem_solver_bias: bool = True):
        super().__init__()
        self.d = d
        self.coord_enc = CoordEncoder(d_in_max=d_coord_in, d=d, depth=depth, dropout=dropout)
        self.matrix_enc = MatrixEncoder(d=d)
        # Metadata
        self.prob_emb = nn.Embedding(len(PROBLEMS), d)
        self.cbits_proj = nn.Linear(K_CBITS, d, bias=False)
        self.coord_dist_emb = nn.Embedding(D_COORD, d)
        self.metadata_ln = nn.LayerNorm(d)
        # Solver embeddings (global vocabulary)
        self.solver_emb = nn.Embedding(M_GLOBAL, d)
        # Head MLP on pair features
        self.head = nn.Sequential(
            nn.Linear(4*d, head_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        # Problem × solver learned bias table (init 0) — captures per-problem solver priors.
        self.use_problem_solver_bias = use_problem_solver_bias
        if use_problem_solver_bias:
            self.problem_solver_bias = nn.Parameter(torch.zeros(len(PROBLEMS), M_GLOBAL))
        self.use_fact = use_mvrp_factorized
        if use_mvrp_factorized:
            self.W_c = nn.Linear(K_CBITS, d, bias=False)
            self.W_d = nn.Embedding(D_COORD, d)
            self.fact_head = nn.Sequential(
                nn.Linear(3*d, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
        # Problem-ID dropout for anti-leakage ablation
        self.prob_dropout_p = 0.0  # set externally

    def set_prob_dropout(self, p: float):
        self.prob_dropout_p = float(p)

    def encode_instance(self, batch):
        if batch["kind"] == "coord":
            g = self.coord_enc(batch["node"], batch["node_mask"])   # (B, d)
        else:
            g = self.matrix_enc(batch["matrix"], batch["node_mask"])
        B = g.shape[0]
        pid = torch.full((B,), batch["problem_id"], dtype=torch.long, device=g.device)
        prob_vec = self.prob_emb(pid)
        if self.training and self.prob_dropout_p > 0:
            drop = (torch.rand(B, device=g.device) < self.prob_dropout_p).float().unsqueeze(-1)
            prob_vec = prob_vec * (1 - drop)
        cbits_vec = self.cbits_proj(batch["cbits"])
        cd_vec = self.coord_dist_emb(batch["coord_dist"])
        h = self.metadata_ln(g + prob_vec + cbits_vec + cd_vec)
        return h

    def forward(self, batch):
        """Compute logits over all M_GLOBAL solvers (unavailable get -inf via mask)."""
        h = self.encode_instance(batch)  # (B, d)
        B = h.shape[0]
        M = M_GLOBAL
        # Solver embeddings
        e = self.solver_emb.weight  # (M, d)
        # Pair features: (B, M, 4d)
        h_exp = h.unsqueeze(1).expand(B, M, self.d)
        e_exp = e.unsqueeze(0).expand(B, M, self.d)
        z = torch.cat([h_exp, e_exp, h_exp * e_exp, (h_exp - e_exp).abs()], dim=-1)
        logits = self.head(z).squeeze(-1)  # (B, M)
        # Per-problem per-solver learnable bias
        if self.use_problem_solver_bias:
            pid = batch["problem_id"]
            logits = logits + self.problem_solver_bias[pid]  # (M,) broadcast over B
        # MVRP factorization add-on (applied only for MVRP problems)
        if self.use_fact and batch["problem_id"] >= 3:
            v = batch["cbits"]                # (B, K_CBITS)
            cd = batch["coord_dist"]          # (B,)
            v_e = self.W_c(v)                 # (B, d)
            d_e = self.W_d(cd)                # (B, d)
            v_s = v_e.unsqueeze(1) * e_exp    # (B, M, d)
            d_s = d_e.unsqueeze(1) * e_exp    # (B, M, d)
            vd_s = v_s * d_e.unsqueeze(1)     # (B, M, d)  interaction
            z_fact = torch.cat([v_s, d_s, vd_s], dim=-1)  # (B, M, 3d)
            logits = logits + self.fact_head(z_fact).squeeze(-1)
        # Apply availability mask: unavailable solvers -> -inf
        mask = batch["mask"]   # (B, M) {0,1}
        logits = logits + torch.log(mask.clamp_min(1e-30))
        return logits


def regret_soft_targets(costs: torch.Tensor, tau: float, eps: float = 1e-6, tie_tol: float = 1e-3,
                        tie_floor: bool = True) -> torch.Tensor:
    """Given costs (B, K_p), return soft labels over the SAME K_p slots.
    p*(s) propto exp(-r_s / tau); r_s = (c_s - c*)/(|c*|+eps)."""
    c_star = costs.min(dim=1, keepdim=True).values  # (B,1)
    r = (costs - c_star) / (c_star.abs() + eps)
    if tie_floor:
        r = torch.where(r < tie_tol, torch.zeros_like(r), r)
    logits = -r / tau
    return torch.softmax(logits, dim=1)


def masked_listwise_ce(logits_global: torch.Tensor, pool_ids: torch.Tensor,
                       soft_targets_pool: torch.Tensor) -> torch.Tensor:
    """logits_global: (B, M_global) already masked via +log(mask).
    pool_ids: (K_p,) global ids for this problem.
    soft_targets_pool: (B, K_p)
    Return scalar loss = -sum_k y_k log softmax_s_valid(ℓ)[pool_ids[k]].
    """
    # log_softmax over all M (unavailable are -inf so zero prob)
    log_p = F.log_softmax(logits_global, dim=1)  # (B, M)
    # Gather at pool_ids
    log_p_pool = log_p[:, pool_ids]              # (B, K_p)
    loss = -(soft_targets_pool * log_p_pool).sum(dim=1).mean()
    return loss


def predict_cost(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Return the COST achieved by argmax-over-valid selection. (B,)"""
    log_p = F.log_softmax(logits_global, dim=1)
    log_p_pool = log_p[:, pool_ids]
    pred = log_p_pool.argmax(dim=1)  # index within pool
    return costs.gather(1, pred.unsqueeze(1)).squeeze(1), pred


def combined_cost_loss(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor,
                       sbs_pool_idx: int, tau: float = 0.07, w_ce: float = 0.5,
                       w_regret: float = 1.0, w_hinge: float = 0.2,
                       hinge_gap: float = 0.002, hinge_margin: float = 0.20,
                       r_cap: float = 0.20) -> torch.Tensor:
    """CE + regret + switch-hinge combined loss. Operates on POOL slice.
    logits_global: (B, M) with -inf on unavailable via +log(mask).
    pool_ids: (K_p,). costs: (B, K_p). sbs_pool_idx: int.
    """
    K_p = pool_ids.shape[0]
    c_best = costs.min(dim=1, keepdim=True).values          # (B,1)
    sbs_cost = costs[:, sbs_pool_idx]                        # (B,)
    # Normalized regret r ∈ [0, r_cap/r_cap]
    r = ((costs - c_best) / (c_best.abs() + 1e-9)).clamp(min=0, max=r_cap) / r_cap
    q = torch.softmax(-r / tau, dim=1)                       # (B, K_p) soft targets
    # Pool logits and log-prob
    logits_pool = logits_global[:, pool_ids]                 # (B, K_p)
    log_p = F.log_softmax(logits_pool, dim=1)
    p = torch.softmax(logits_pool, dim=1)
    loss_ce = -(q * log_p).sum(dim=1).mean()
    loss_regret = (p * r).sum(dim=1).mean()
    # Switch hinge: when VBS beats SBS by hinge_gap, push best logit above SBS logit by hinge_margin
    b = costs.argmin(dim=1)
    gap_vs_sbs = ((sbs_cost - costs.gather(1, b.unsqueeze(1)).squeeze(1)) /
                  (sbs_cost.abs() + 1e-9)).clamp_min(0)
    switch_mask = gap_vs_sbs > hinge_gap
    if switch_mask.any():
        best_logit = logits_pool.gather(1, b.unsqueeze(1)).squeeze(1)
        sbs_logit = logits_pool[:, sbs_pool_idx]
        hinge = torch.relu(hinge_margin - (best_logit - sbs_logit))
        loss_hinge = (gap_vs_sbs[switch_mask] * hinge[switch_mask]).mean()
    else:
        loss_hinge = torch.tensor(0.0, device=logits_global.device)
    return w_ce * loss_ce + w_regret * loss_regret + w_hinge * loss_hinge
