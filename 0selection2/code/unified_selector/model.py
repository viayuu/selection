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

from .registry import PROBLEMS, P2I, M_GLOBAL, K_CBITS, D_COORD, K_PROBLEM_DESC


def remap_legacy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upgrade older checkpoint key names to the current module layout."""
    upgraded = dict(state)
    legacy_prefix_map = {
        "support_head.": "support_heads.0.",
    }
    for old_prefix, new_prefix in legacy_prefix_map.items():
        for key in list(upgraded.keys()):
            if key.startswith(old_prefix):
                upgraded[new_prefix + key[len(old_prefix):]] = upgraded.pop(key)
    return upgraded


class MixedBiasEncoderBlock(nn.Module):
    """Three-branch mixed-bias attention with optional ReZero residual scaling."""

    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1,
                 condition_dim: int = K_CBITS + D_COORD + 2,
                 use_rezero: bool = False):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d={d} must be divisible by heads={heads}")
        self.d = d
        self.heads = heads
        self.dh = d // heads
        self.use_rezero = use_rezero
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(3 * d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)
        self.alpha_net = nn.Sequential(
            nn.Linear(condition_dim, max(16, d // 2)),
            nn.GELU(),
            nn.Linear(max(16, d // 2), 3 * heads),
        )
        if use_rezero:
            self.rezero_attn = nn.Parameter(torch.zeros(1))
            self.rezero_ffn = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("rezero_attn", torch.ones(1), persistent=False)
            self.register_buffer("rezero_ffn", torch.ones(1), persistent=False)

    def _attend(self, x: torch.Tensor, mask: torch.Tensor,
                biases: list[torch.Tensor], cond_vec: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = x.shape
        x = self.norm1(x)
        q = self.q_proj(x).view(bsz, n_nodes, self.heads, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_nodes, self.heads, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_nodes, self.heads, self.dh).transpose(1, 2)
        base = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        alpha = F.softplus(self.alpha_net(cond_vec)).view(bsz, 3, self.heads, 1, 1) + 1e-3
        key_mask = (~mask).unsqueeze(1).unsqueeze(2)
        query_mask = mask.unsqueeze(-1).to(x.dtype)
        outs = []
        for idx, bias in enumerate(biases):
            bias = bias.unsqueeze(1).to(base.dtype)
            score = base - alpha[:, idx] * bias
            score = score.masked_fill(key_mask, float("-inf"))
            attn = torch.softmax(score, dim=-1)
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, n_nodes, self.d)
            outs.append(out * query_mask)
        return self.out_proj(torch.cat(outs, dim=-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                biases: list[torch.Tensor], cond_vec: torch.Tensor) -> torch.Tensor:
        attn_out = self._attend(x, mask, biases, cond_vec)
        x = x + self.rezero_attn * attn_out
        ff_out = self.ffn_drop(self.ffn(self.norm2(x)))
        x = x + self.rezero_ffn * ff_out
        return x


class CoordEncoder(nn.Module):
    """Two-layer GAT-ish encoder over padded node sequence (includes depot/feature heads)."""

    def __init__(self, d_in_max: int = 8, d: int = 128, depth: int = 4, heads: int = 4,
                 dropout: float = 0.1, rich_pool: bool = False,
                 hier_pool: bool = False, downsample_ratio: float = 0.8,
                 deep_mbm: bool = False, use_rezero: bool = False):
        super().__init__()
        self.d_in_max = d_in_max
        self.d = d
        self.rich_pool = rich_pool
        self.hier_pool = hier_pool
        self.deep_mbm = deep_mbm
        self.in_proj = nn.Linear(d_in_max, d)
        if hier_pool:
            n_blocks = 2
            base = max(1, depth // n_blocks)
            rem = max(0, depth - base * n_blocks)
            block_depths = [base + (1 if i < rem else 0) for i in range(n_blocks)]
            self.blocks = nn.ModuleList([
                CoordHierPoolBlock(
                    d=d, heads=heads, depth=max(1, bd), dropout=dropout,
                    downsample_ratio=downsample_ratio,
                    deep_mbm=deep_mbm,
                    use_rezero=use_rezero,
                )
                for bd in block_depths
            ])
            self.final_pool_proj = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        else:
            if deep_mbm:
                self.layers = nn.ModuleList([
                    MixedBiasEncoderBlock(d=d, heads=heads, dropout=dropout, use_rezero=use_rezero)
                    for _ in range(depth)
                ])
            else:
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2*d, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
                    for _ in range(depth)
                ])
        self.out_norm = nn.LayerNorm(d)
        if rich_pool:
            self.pool_proj = nn.Sequential(
                nn.Linear(4 * d, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )

    @staticmethod
    def build_condition_vec(node: torch.Tensor) -> torch.Tensor:
        bsz = node.shape[0]
        device = node.device
        dtype = node.dtype
        cond = torch.zeros(bsz, K_CBITS + D_COORD + 2, device=device, dtype=dtype)
        if node.shape[-1] > 4:
            active = (node[:, 1:, 4].abs().amax(dim=1) > 0).to(dtype)
            cond[:, 3] = active
        if node.shape[-1] > 7:
            active = (node[:, 1:, 6:8].abs().amax(dim=(1, 2)) > 0).to(dtype)
            cond[:, 4] = active
        has_neg = (node[:, 1:, 2] < 0).any(dim=1).to(dtype) if node.shape[-1] > 2 else torch.zeros(bsz, device=device, dtype=dtype)
        cond[:, 2] = has_neg
        cond[:, K_CBITS + D_COORD] = 1.0
        cond[:, K_CBITS + D_COORD + 1] = 0.0
        return cond

    @staticmethod
    def build_biases(node: torch.Tensor, node_mask: torch.Tensor) -> list[torch.Tensor]:
        xy = node[..., :2]
        dist = torch.cdist(xy, xy, p=2)
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        dist = dist / dist.masked_select(valid).amax().clamp_min(1e-6)
        if node.shape[-1] > 2:
            demand = node[..., 2]
            demand_gap = (demand.unsqueeze(2) - demand.unsqueeze(1)).abs()
            demand_gap = demand_gap / demand_gap.masked_select(valid).amax().clamp_min(1e-6)
        else:
            demand_gap = torch.zeros_like(dist)
        n_nodes = node.shape[1]
        masked = dist.masked_fill(~valid, float("inf"))
        k = min(4, max(1, n_nodes - 1))
        knn = masked.topk(k, dim=-1, largest=False).indices
        relation = torch.ones_like(dist)
        relation.scatter_(2, knn, 0.0)
        relation = torch.minimum(relation, relation.transpose(1, 2))
        relation = relation.masked_fill(~valid, 1.0)
        eye = torch.eye(n_nodes, device=node.device, dtype=torch.bool).unsqueeze(0)
        relation = relation.masked_fill(eye, 0.0)
        return [dist, demand_gap, relation]

    def _prepare_node(self, node: torch.Tensor) -> torch.Tensor:
        B, N, d_in = node.shape
        if d_in < self.d_in_max:
            pad = torch.zeros(B, N, self.d_in_max - d_in, device=node.device, dtype=node.dtype)
            node = torch.cat([node, pad], dim=-1)
        elif d_in > self.d_in_max:
            node = node[..., :self.d_in_max]
        return node

    def forward_tokens(self, node, node_mask):
        node = self._prepare_node(node)
        B = node.shape[0]
        x = self.in_proj(node)
        if self.hier_pool:
            graph_terms = []
            cur_x, cur_mask = x, node_mask
            cur_idx = torch.arange(node.shape[1], device=node.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            cond_vec = self.build_condition_vec(node)
            biases = self.build_biases(node, node_mask) if self.deep_mbm else None
            for block in self.blocks:
                graph_emb, cur_x, cur_mask, cur_idx = block(
                    cur_x, cur_mask, raw_node=node, cond_vec=cond_vec, full_biases=biases,
                    node_indices=cur_idx,
                )
                graph_terms.append(graph_emb)
            x = self.out_norm(cur_x)
            m = cur_mask.unsqueeze(-1).float()
            mean = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            neg_inf = torch.full_like(x, float("-inf"))
            maxv = torch.where(cur_mask.unsqueeze(-1), x, neg_inf).max(dim=1).values
            pooled = self.final_pool_proj(torch.cat([mean, maxv], dim=-1))
            graph_terms.append(pooled)
            pooled = torch.stack(graph_terms, dim=0).mean(dim=0)
            return pooled, x, cur_mask
        # key_padding_mask: True where PAD
        kpm = ~node_mask
        if self.deep_mbm:
            cond_vec = self.build_condition_vec(node)
            biases = self.build_biases(node, node_mask)
            for layer in self.layers:
                x = layer(x, node_mask, biases, cond_vec)
        else:
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=kpm)
        x = self.out_norm(x)
        m = node_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))  # (B, d)
        if self.rich_pool:
            # Rich instance summary: mean + max + std + depot token.
            neg_inf = torch.full_like(x, float("-inf"))
            x_max = torch.where(node_mask.unsqueeze(-1), x, neg_inf).max(dim=1).values
            centered = (x - pooled.unsqueeze(1)) * m
            x_std = torch.sqrt((centered.pow(2).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).clamp_min(1e-8))
            depot = x[:, 0]
            pooled = self.pool_proj(torch.cat([pooled, x_max, x_std, depot], dim=-1))
        return pooled, x, node_mask

    def forward(self, node, node_mask):
        pooled, _, _ = self.forward_tokens(node, node_mask)
        return pooled


class CoordHierPoolBlock(nn.Module):
    def __init__(self, d: int, heads: int, depth: int, dropout: float, downsample_ratio: float,
                 deep_mbm: bool = False, use_rezero: bool = False):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.deep_mbm = deep_mbm
        if deep_mbm:
            self.layers = nn.ModuleList([
                MixedBiasEncoderBlock(d=d, heads=heads, dropout=dropout, use_rezero=use_rezero)
                for _ in range(depth)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=heads, dim_feedforward=2 * d,
                    dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
                )
                for _ in range(depth)
            ])
        self.score = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.full_like(x, float("-inf"))
        return torch.where(mask.unsqueeze(-1), x, neg_inf).max(dim=1).values

    def downsample(self, x: torch.Tensor, mask: torch.Tensor, node_indices: torch.Tensor | None = None):
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        kept_x = []
        kept_mask = []
        kept_idx = []
        max_keep = 0
        for i in range(x.shape[0]):
            n_valid = int(mask[i].sum().item())
            keep = min(n_valid, max(1, int(math.ceil(n_valid * self.downsample_ratio))))
            top = scores[i, :n_valid].topk(keep, largest=True).indices.sort().values
            sel = x[i].index_select(0, top)
            kept_x.append(sel)
            kept_mask.append(torch.ones(keep, dtype=torch.bool, device=x.device))
            if node_indices is None:
                sel_idx = top
            else:
                sel_idx = node_indices[i].index_select(0, top)
            kept_idx.append(sel_idx)
            max_keep = max(max_keep, keep)
        padded_x = []
        padded_mask = []
        padded_idx = []
        for sel_x, sel_m, sel_idx in zip(kept_x, kept_mask, kept_idx):
            pad = max_keep - sel_x.shape[0]
            if pad > 0:
                sel_x = F.pad(sel_x, (0, 0, 0, pad))
                sel_m = F.pad(sel_m, (0, pad), value=False)
                sel_idx = F.pad(sel_idx, (0, pad), value=0)
            padded_x.append(sel_x.unsqueeze(0))
            padded_mask.append(sel_m.unsqueeze(0))
            padded_idx.append(sel_idx.unsqueeze(0))
        return torch.cat(padded_x, dim=0), torch.cat(padded_mask, dim=0), torch.cat(padded_idx, dim=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                raw_node: torch.Tensor | None = None,
                cond_vec: torch.Tensor | None = None,
                full_biases: list[torch.Tensor] | None = None,
                node_indices: torch.Tensor | None = None):
        kpm = ~mask
        if self.deep_mbm:
            cur_biases = []
            assert full_biases is not None and cond_vec is not None
            for bias in full_biases:
                gathered = []
                for i in range(x.shape[0]):
                    idx = node_indices[i]
                    sub = bias[i].index_select(0, idx).index_select(1, idx)
                    gathered.append(sub.unsqueeze(0))
                cur_biases.append(torch.cat(gathered, dim=0))
            for layer in self.layers:
                x = layer(x, mask, cur_biases, cond_vec)
        else:
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=kpm)
        graph_emb = self.readout(torch.cat([self.masked_mean(x, mask), self.masked_max(x, mask)], dim=-1))
        pooled_x, pooled_mask, pooled_idx = self.downsample(x, mask, node_indices=node_indices)
        return graph_emb, pooled_x, pooled_mask, pooled_idx


class MatrixEncoder(nn.Module):
    """Simple ATSP matrix encoder: summary features + MLP.  Pilot-level; MatNet-proper can replace later."""

    def __init__(self, d: int = 128, depth: int = 2, heads: int = 4,
                 dropout: float = 0.1, deep_mbm: bool = False, use_rezero: bool = False):
        super().__init__()
        self.deep_mbm = deep_mbm
        # Stats: mean, std, max, min, skew, symmetry_ratio, row_entropy_mean, col_entropy_mean, TI-violation-rate, diag_max
        if deep_mbm:
            self.in_proj = nn.Sequential(
                nn.Linear(10, d), nn.GELU(),
                nn.Linear(d, d), nn.GELU(),
            )
            self.layers = nn.ModuleList([
                MixedBiasEncoderBlock(d=d, heads=heads, dropout=dropout, use_rezero=use_rezero)
                for _ in range(depth)
            ])
            self.out_proj = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(10, d), nn.GELU(),
                nn.Linear(d, d), nn.GELU(),
                nn.LayerNorm(d),
            )

    def build_node_features(self, mat, node_mask):
        B, N, _ = mat.shape
        feats = mat.new_zeros(B, N, 10)
        mask = node_mask.bool()
        for i in range(B):
            n = int(mask[i].sum().item())
            m = mat[i, :n, :n].clamp(min=0, max=1e3)
            eye = torch.eye(n, dtype=torch.bool, device=m.device)
            masked = m.masked_fill(eye, float("inf"))
            row_mean = m.mean(dim=1)
            col_mean = m.mean(dim=0)
            row_std = m.std(dim=1, unbiased=False)
            col_std = m.std(dim=0, unbiased=False)
            near_out = masked.min(dim=1).values if n > 1 else m.new_zeros(n)
            near_in = masked.min(dim=0).values if n > 1 else m.new_zeros(n)
            asym = (m - m.t()).abs().mean(dim=1)
            diag = m.diag()
            feats_i = torch.stack([
                row_mean,
                col_mean,
                row_std,
                col_std,
                near_out,
                near_in,
                asym,
                diag,
                row_mean - col_mean,
                near_out - near_in,
            ], dim=-1)
            feats[i, :n] = feats_i
        return feats, mask

    def forward_tokens(self, mat, node_mask):
        B, N, _ = mat.shape
        if self.deep_mbm:
            feats, mask = self.build_node_features(mat, node_mask)
            cond_vec = []
            bias0 = mat.new_ones(B, N, N)
            bias1 = mat.new_ones(B, N, N)
            bias2 = mat.new_ones(B, N, N)
            for i in range(B):
                n = int(mask[i].sum().item())
                cur = mat[i, :n, :n].clamp(min=0, max=1e3)
                cond_i = torch.zeros(K_CBITS + D_COORD + 2, device=mat.device, dtype=mat.dtype)
                cond_i[K_CBITS + D_COORD] = 0.0
                cond_i[K_CBITS + D_COORD + 1] = 1.0
                cond_vec.append(cond_i)
                vmax = cur.max().clamp_min(1e-6)
                bias0[i, :n, :n] = cur / vmax
                bias1[i, :n, :n] = cur.t() / vmax
                asym = (cur - cur.t()).abs()
                bias2[i, :n, :n] = asym / asym.max().clamp_min(1e-6)
                eye = torch.eye(n, dtype=torch.bool, device=mat.device)
                bias0[i, :n, :n].masked_fill_(eye, 0.0)
                bias1[i, :n, :n].masked_fill_(eye, 0.0)
                bias2[i, :n, :n].masked_fill_(eye, 0.0)
            cond = torch.stack(cond_vec, dim=0)
            x = self.in_proj(feats)
            for layer in self.layers:
                x = layer(x, mask, [bias0, bias1, bias2], cond)
            neg_inf = torch.full_like(x, float("-inf"))
            mean = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.unsqueeze(-1).float().sum(dim=1).clamp_min(1.0)
            maxv = torch.where(mask.unsqueeze(-1), x, neg_inf).max(dim=1).values
            pooled = self.out_proj(torch.cat([mean, maxv], dim=-1))
            return pooled, x, mask

        feats, mask = self.build_node_features(mat, node_mask)
        scale = feats.abs().mean(dim=(0, 1), keepdim=True) + 1e-6
        x = self.proj(feats / scale)
        pooled = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.unsqueeze(-1).float().sum(dim=1).clamp_min(1.0)
        return pooled, x, mask

    def forward(self, mat, node_mask):
        # mat: (B, N, N), node_mask: (B, N). Use only valid n subrange.
        B, N, _ = mat.shape
        if self.deep_mbm:
            node_feats = []
            cond_vec = []
            for i in range(B):
                n = int(node_mask[i].sum().item())
                m = mat[i, :n, :n].clamp(min=0, max=1e3)
                eye = torch.eye(n, dtype=torch.bool, device=m.device)
                masked = m.masked_fill(eye, float("inf"))
                row_mean = m.mean(dim=1)
                col_mean = m.mean(dim=0)
                row_std = m.std(dim=1, unbiased=False)
                col_std = m.std(dim=0, unbiased=False)
                near_out = masked.min(dim=1).values if n > 1 else m.new_zeros(n)
                near_in = masked.min(dim=0).values if n > 1 else m.new_zeros(n)
                asym = (m - m.t()).abs().mean(dim=1)
                diag = m.diag()
                feats_i = torch.stack([
                    row_mean,
                    col_mean,
                    row_std,
                    col_std,
                    near_out,
                    near_in,
                    asym,
                    diag,
                    row_mean - col_mean,
                    near_out - near_in,
                ], dim=-1)
                node_feats.append(feats_i)
                cond_i = torch.zeros(K_CBITS + D_COORD + 2, device=m.device, dtype=m.dtype)
                cond_i[K_CBITS + D_COORD] = 0.0
                cond_i[K_CBITS + D_COORD + 1] = 1.0
                cond_vec.append(cond_i)
            max_n = max(f.shape[0] for f in node_feats)
            x = mat.new_zeros(B, max_n, 10)
            mask = torch.zeros(B, max_n, dtype=torch.bool, device=mat.device)
            bias0 = mat.new_ones(B, max_n, max_n)
            bias1 = mat.new_ones(B, max_n, max_n)
            bias2 = mat.new_ones(B, max_n, max_n)
            for i, feats_i in enumerate(node_feats):
                n = feats_i.shape[0]
                x[i, :n] = feats_i
                mask[i, :n] = True
                cur = mat[i, :n, :n].clamp(min=0, max=1e3)
                vmax = cur.max().clamp_min(1e-6)
                bias0[i, :n, :n] = cur / vmax
                bias1[i, :n, :n] = cur.t() / vmax
                bias2[i, :n, :n] = (cur - cur.t()).abs() / (cur - cur.t()).abs().max().clamp_min(1e-6)
                eye = torch.eye(n, dtype=torch.bool, device=mat.device)
                bias0[i, :n, :n].masked_fill_(eye, 0.0)
                bias1[i, :n, :n].masked_fill_(eye, 0.0)
                bias2[i, :n, :n].masked_fill_(eye, 0.0)
            cond = torch.stack(cond_vec, dim=0)
            x = self.in_proj(x)
            for layer in self.layers:
                x = layer(x, mask, [bias0, bias1, bias2], cond)
            neg_inf = torch.full_like(x, float("-inf"))
            mean = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.unsqueeze(-1).float().sum(dim=1).clamp_min(1.0)
            maxv = torch.where(mask.unsqueeze(-1), x, neg_inf).max(dim=1).values
            return self.out_proj(torch.cat([mean, maxv], dim=-1))
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
                 use_problem_solver_bias: bool = True,
                 use_problem_film: bool = False,
                 use_size_feature: bool = False,
                 rich_pool: bool = False,
                 use_global_stats: bool = False,
                 use_manual_features: bool = False,
                 use_constraint_experts: bool = False,
                 use_problem_residual_head: bool = False,
                 use_problem_adapter: bool = False,
                 use_support_head: bool = False,
                 support_hidden: int = 128,
                 support_generators: int = 1,
                 adapter_hidden: int = 64,
                 coord_hier_pool: bool = False,
                 coord_downsample_ratio: float = 0.8,
                 deep_encoder_overhaul: bool = False,
                 encoder_rezero: bool = False,
                 encoder_constraint_experts: bool = False,
                 encoder_constraint_hidden: int = 128,
                 use_problem_descriptor: bool = False,
                 use_descriptor_solver_bias: bool = False):
        super().__init__()
        self.d = d
        self.coord_enc = CoordEncoder(
            d_in_max=d_coord_in, d=d, depth=depth, dropout=dropout,
            rich_pool=rich_pool, hier_pool=coord_hier_pool,
            downsample_ratio=coord_downsample_ratio,
            deep_mbm=deep_encoder_overhaul,
            use_rezero=encoder_rezero,
        )
        self.matrix_enc = MatrixEncoder(
            d=d, depth=max(2, depth // 2), heads=4, dropout=dropout,
            deep_mbm=deep_encoder_overhaul, use_rezero=encoder_rezero,
        )
        self.deep_encoder_overhaul = deep_encoder_overhaul
        # Metadata.  The descriptor path is the zero-shot-friendly alternative
        # to discrete problem IDs: new tasks can populate the descriptor slots
        # without adding a new embedding row.
        self.use_problem_descriptor = use_problem_descriptor
        self.prob_emb = nn.Embedding(len(PROBLEMS), d)
        if use_problem_descriptor:
            self.problem_desc_proj = nn.Sequential(
                nn.Linear(K_PROBLEM_DESC, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        self.cbits_proj = nn.Linear(K_CBITS, d, bias=False)
        self.coord_dist_emb = nn.Embedding(D_COORD, d)
        self.use_size_feature = use_size_feature
        if use_size_feature:
            self.size_proj = nn.Sequential(
                nn.Linear(3, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        self.use_global_stats = use_global_stats
        if use_global_stats:
            self.stats_proj = nn.Sequential(
                nn.Linear(16, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        self.use_manual_features = use_manual_features
        if use_manual_features:
            self.manual_proj = nn.Sequential(
                nn.Linear(32, d),
                nn.GELU(),
                nn.LayerNorm(d),
            )
        self.metadata_ln = nn.LayerNorm(d)
        self.use_problem_film = use_problem_film
        if use_problem_film:
            if use_problem_descriptor:
                self.problem_film = nn.Linear(K_PROBLEM_DESC, 2 * d)
                nn.init.zeros_(self.problem_film.weight)
                nn.init.zeros_(self.problem_film.bias)
            else:
                self.problem_film_gain = nn.Embedding(len(PROBLEMS), d)
                self.problem_film_shift = nn.Embedding(len(PROBLEMS), d)
                nn.init.zeros_(self.problem_film_gain.weight)
                nn.init.zeros_(self.problem_film_shift.weight)
            self.film_ln = nn.LayerNorm(d)
        self.use_problem_adapter = use_problem_adapter
        if use_problem_adapter:
            if use_problem_descriptor:
                self.problem_desc_adapter = nn.Sequential(
                    nn.Linear(2 * d, adapter_hidden),
                    nn.GELU(),
                    nn.Linear(adapter_hidden, d),
                )
            else:
                self.problem_adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d, adapter_hidden),
                        nn.GELU(),
                        nn.Linear(adapter_hidden, d),
                    )
                    for _ in range(len(PROBLEMS))
                ])
            self.adapter_ln = nn.LayerNorm(d)
        self.encoder_constraint_experts = encoder_constraint_experts
        if encoder_constraint_experts:
            self.encoder_constraint_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, encoder_constraint_hidden),
                    nn.GELU(),
                    nn.Linear(encoder_constraint_hidden, d),
                )
                for _ in range(8)
            ])
            self.encoder_constraint_ln = nn.LayerNorm(d)
        # Solver embeddings (global vocabulary)
        self.solver_emb = nn.Embedding(M_GLOBAL, d)
        # Head MLP on pair features
        self.head = nn.Sequential(
            nn.Linear(4*d, head_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.use_support_head = use_support_head
        self.support_generators = max(1, int(support_generators))
        if use_support_head:
            self.support_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4 * d, support_hidden), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(support_hidden, 1),
                )
                for _ in range(self.support_generators)
            ])
            if self.support_generators > 1:
                # Give each proposal head a lightweight solver-specific prior so heads can
                # specialize on different solver families instead of collapsing together.
                self.support_generator_solver_bias = nn.Parameter(
                    torch.zeros(self.support_generators, M_GLOBAL)
                )
        self.use_problem_residual_head = use_problem_residual_head
        if use_problem_residual_head:
            self.problem_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4*d, head_hidden), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(len(PROBLEMS))
            ])
            if use_support_head:
                self.support_problem_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(4 * d, support_hidden), nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(support_hidden, 1),
                    )
                    for _ in range(len(PROBLEMS))
                ])
        self.use_constraint_experts = use_constraint_experts
        if use_constraint_experts:
            self.constraint_expert_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4*d, head_hidden), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(8)
            ])
            if use_support_head:
                self.support_constraint_expert_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(4 * d, support_hidden), nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(support_hidden, 1),
                    )
                    for _ in range(8)
                ])
        # Problem × solver learned bias table (init 0) — captures per-problem solver priors.
        self.use_problem_solver_bias = use_problem_solver_bias
        if use_problem_solver_bias:
            self.problem_solver_bias = nn.Parameter(torch.zeros(len(PROBLEMS), M_GLOBAL))
        self.use_descriptor_solver_bias = bool(use_problem_descriptor and use_descriptor_solver_bias)
        if self.use_descriptor_solver_bias:
            self.desc_solver_bias = nn.Sequential(
                nn.Linear(K_PROBLEM_DESC, d),
                nn.GELU(),
                nn.Linear(d, M_GLOBAL),
            )
            nn.init.zeros_(self.desc_solver_bias[-1].weight)
            nn.init.zeros_(self.desc_solver_bias[-1].bias)
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

    def constraint_weight_vector(self, batch, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        bsz = batch["cbits"].shape[0]
        return torch.cat([
            torch.ones(bsz, 1, device=device, dtype=dtype),
            torch.full((bsz, 1), 1.0 if batch["kind"] == "coord" else 0.0, device=device, dtype=dtype),
            torch.full((bsz, 1), 1.0 if batch["kind"] == "matrix" else 0.0, device=device, dtype=dtype),
            batch["cbits"].to(device=device, dtype=dtype),
        ], dim=1)

    def encode_instance(self, batch):
        if batch["kind"] == "coord":
            g = self.coord_enc(batch["node"], batch["node_mask"])   # (B, d)
        else:
            g = self.matrix_enc(batch["matrix"], batch["node_mask"])
        B = g.shape[0]
        pid = torch.full((B,), batch["problem_id"], dtype=torch.long, device=g.device)
        desc_vec = None
        if self.use_problem_descriptor:
            desc = batch["problem_desc"].to(device=g.device, dtype=g.dtype)
            desc_vec = self.problem_desc_proj(desc)
            prob_vec = desc_vec
        else:
            prob_vec = self.prob_emb(pid)
            if self.training and self.prob_dropout_p > 0:
                drop = (torch.rand(B, device=g.device) < self.prob_dropout_p).float().unsqueeze(-1)
                prob_vec = prob_vec * (1 - drop)
        cbits_vec = self.cbits_proj(batch["cbits"])
        cd_vec = self.coord_dist_emb(batch["coord_dist"])
        h = g + prob_vec + cbits_vec + cd_vec
        if self.use_size_feature:
            n = batch["n"].float().unsqueeze(-1)
            size_feats = torch.cat([
                torch.log1p(n) / 5.0,
                n / 256.0,
                (n >= 100).float(),
            ], dim=-1)
            h = h + self.size_proj(size_feats)
        if self.use_global_stats:
            stats = self.compute_global_stats(batch)
            h = h + self.stats_proj(stats)
        if self.use_manual_features:
            h = h + self.manual_proj(self.compute_manual_features(batch))
        h = self.metadata_ln(h)
        if self.use_problem_film:
            if self.use_problem_descriptor:
                gain, shift = self.problem_film(batch["problem_desc"].to(device=h.device, dtype=h.dtype)).chunk(2, dim=-1)
            else:
                gain = self.problem_film_gain(pid)
                shift = self.problem_film_shift(pid)
            h = self.film_ln(h * (1.0 + gain) + shift)
        if self.use_problem_adapter:
            if self.use_problem_descriptor:
                if desc_vec is None:
                    desc_vec = self.problem_desc_proj(batch["problem_desc"].to(device=h.device, dtype=h.dtype))
                h = self.adapter_ln(h + self.problem_desc_adapter(torch.cat([h, desc_vec], dim=-1)))
            else:
                h = self.adapter_ln(h + self.problem_adapters[batch["problem_id"]](h))
        if self.encoder_constraint_experts:
            expert_weights = self.constraint_weight_vector(batch, dtype=h.dtype, device=h.device)
            expert_delta = torch.stack([ffn(h) for ffn in self.encoder_constraint_ffns], dim=1)
            mixed = (expert_delta * expert_weights.unsqueeze(-1)).sum(dim=1) / expert_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            h = self.encoder_constraint_ln(h + mixed)
        return h

    def encode_instance_with_tokens(self, batch):
        if batch["kind"] == "coord":
            _, token_x, token_mask = self.coord_enc.forward_tokens(batch["node"], batch["node_mask"])
        else:
            _, token_x, token_mask = self.matrix_enc.forward_tokens(batch["matrix"], batch["node_mask"])
        h = self.encode_instance(batch)
        return h, token_x, token_mask

    def compute_global_stats(self, batch):
        if batch["kind"] == "coord":
            return self.compute_coord_stats(batch["node"], batch["node_mask"])
        return self.compute_matrix_stats(batch["matrix"], batch["node_mask"])

    def compute_coord_stats(self, node, node_mask):
        feats = []
        B = node.shape[0]
        for i in range(B):
            n = int(node_mask[i].sum().item())
            cur = node[i, :n]
            depot = cur[0, :2]
            cust = cur[1:] if n > 1 else cur[:1]
            xy = cust[:, :2]
            rel = xy - depot.unsqueeze(0)
            dist = torch.norm(rel, dim=1) if cust.numel() else torch.zeros(1, device=node.device)
            x_std = xy[:, 0].std(unbiased=False) if xy.shape[0] > 1 else torch.tensor(0.0, device=node.device)
            y_std = xy[:, 1].std(unbiased=False) if xy.shape[0] > 1 else torch.tensor(0.0, device=node.device)

            if cur.shape[1] > 2:
                dem = cust[:, 2]
                dem_mean = dem.mean()
                dem_std = dem.std(unbiased=False)
                dem_min = dem.min()
                dem_max = dem.max()
                neg_frac = (dem < 0).float().mean()
                pos_frac = (dem > 0).float().mean()
            else:
                dem_mean = dem_std = dem_min = dem_max = neg_frac = pos_frac = torch.tensor(0.0, device=node.device)

            route_mean = cust[:, 4].mean() if cur.shape[1] > 4 else torch.tensor(0.0, device=node.device)
            if cur.shape[1] > 7:
                tw_width = (cust[:, 7] - cust[:, 6]).clamp_min(0)
                tw_mean = tw_width.mean()
                tw_std = tw_width.std(unbiased=False)
            else:
                tw_mean = tw_std = torch.tensor(0.0, device=node.device)

            feats.append(torch.stack([
                torch.log1p(torch.tensor(float(n), device=node.device)) / 5.0,
                torch.tensor(float(n), device=node.device) / 256.0,
                dist.mean(),
                dist.std(unbiased=False),
                dist.max(),
                x_std,
                y_std,
                dem_mean,
                dem_std,
                dem_min,
                dem_max,
                neg_frac,
                pos_frac,
                route_mean,
                tw_mean,
                tw_std,
            ]))
        return torch.stack(feats, dim=0)

    def compute_matrix_stats(self, mat, node_mask):
        feats = []
        B = mat.shape[0]
        for i in range(B):
            n = int(node_mask[i].sum().item())
            cur = mat[i, :n, :n]
            eye = torch.eye(n, dtype=torch.bool, device=mat.device)
            off = cur[~eye]
            row_mean = cur.mean(dim=1)
            col_mean = cur.mean(dim=0)
            masked = cur.masked_fill(eye, float("inf"))
            nearest_out = masked.min(dim=1).values
            nearest_in = masked.min(dim=0).values
            feats.append(torch.stack([
                torch.log1p(torch.tensor(float(n), device=mat.device)) / 5.0,
                torch.tensor(float(n), device=mat.device) / 256.0,
                off.mean(),
                off.std(unbiased=False),
                off.min(),
                off.max(),
                (cur - cur.t()).abs().mean(),
                row_mean.mean(),
                row_mean.std(unbiased=False),
                col_mean.mean(),
                col_mean.std(unbiased=False),
                nearest_out.mean(),
                nearest_out.std(unbiased=False),
                nearest_in.mean(),
                nearest_in.std(unbiased=False),
                cur.diag().abs().max(),
            ]))
        return torch.stack(feats, dim=0)

    @staticmethod
    def _quantiles(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.numel() == 0:
            z = values.new_tensor(0.0)
            return z, z, z
        q = torch.tensor([0.25, 0.5, 0.75], device=values.device, dtype=values.dtype)
        out = torch.quantile(values, q)
        return out[0], out[1], out[2]

    @staticmethod
    def _subsample_points(x: torch.Tensor, max_points: int = 32) -> torch.Tensor:
        if x.shape[0] <= max_points:
            return x
        idx = torch.linspace(0, x.shape[0] - 1, steps=max_points, device=x.device)
        idx = idx.round().long().unique()
        return x.index_select(0, idx)

    def compute_manual_features(self, batch):
        if batch["kind"] == "coord":
            return self.compute_coord_manual_features(batch["node"], batch["node_mask"])
        return self.compute_matrix_manual_features(batch["matrix"], batch["node_mask"])

    def compute_coord_manual_features(self, node, node_mask):
        feats = []
        for i in range(node.shape[0]):
            n = int(node_mask[i].sum().item())
            cur = node[i, :n]
            zero = cur.new_tensor(0.0)
            has_explicit_depot = cur.shape[1] > 3 and bool((cur[:, 3] > 0.5).any().item())
            if has_explicit_depot:
                depot_idx = int((cur[:, 3] > 0.5).nonzero(as_tuple=False)[0].item())
                depot = cur[depot_idx, :2]
                keep = torch.ones(n, dtype=torch.bool, device=cur.device)
                keep[depot_idx] = False
                cust = cur[keep]
            else:
                depot = cur[:, :2].mean(dim=0)
                cust = cur
            if cust.shape[0] == 0:
                cust = cur[:1]
            xy = cust[:, :2]
            xy_s = self._subsample_points(xy, max_points=32)
            pair = torch.cdist(xy_s, xy_s, p=2)
            eye = torch.eye(xy_s.shape[0], dtype=torch.bool, device=xy_s.device)
            off = pair[~eye] if xy_s.shape[0] > 1 else xy_s.new_zeros(1)
            q25, q50, q75 = self._quantiles(off)
            masked = pair.masked_fill(eye, float("inf"))
            nn = masked.min(dim=1).values if xy_s.shape[0] > 1 else xy_s.new_zeros(1)
            depot_dist = torch.norm(xy_s - depot.unsqueeze(0), dim=1)
            dq25, dq50, _ = self._quantiles(depot_dist)
            centroid = xy_s.mean(dim=0)
            radius = torch.norm(xy_s - centroid.unsqueeze(0), dim=1)
            depot_to_centroid = torch.norm(depot - centroid)

            if cust.shape[1] > 2:
                dem = cust[:, 2]
                dem_mean = dem.mean()
                dem_std = dem.std(unbiased=False)
                dem_abs = dem.abs().mean()
                pos_frac = (dem > 0).float().mean()
                neg_frac = (dem < 0).float().mean()
                pos_sum = dem.clamp_min(0).sum()
                neg_sum = (-dem.clamp_max(0)).sum()
            else:
                dem_mean = dem_std = dem_abs = pos_frac = neg_frac = pos_sum = neg_sum = zero

            route_mean = cust[:, 4].mean() if cust.shape[1] > 4 else zero
            service_mean = cust[:, 5].mean() if cust.shape[1] > 5 else zero
            if cust.shape[1] > 7:
                tw_start = cust[:, 6]
                tw_end = cust[:, 7]
                tw_width = (tw_end - tw_start).clamp_min(0)
                tw_width_mean = tw_width.mean()
                tw_width_std = tw_width.std(unbiased=False)
                tw_start_mean = tw_start.mean()
                tw_end_mean = tw_end.mean()
            else:
                tw_width_mean = tw_width_std = tw_start_mean = tw_end_mean = zero

            feats.append(torch.stack([
                torch.log1p(cur.new_tensor(float(n))) / 5.0,
                cur.new_tensor(float(n)) / 256.0,
                off.mean(),
                off.std(unbiased=False),
                q25,
                q50,
                q75,
                off.max(),
                nn.mean(),
                nn.std(unbiased=False),
                nn.min(),
                depot_dist.mean(),
                depot_dist.std(unbiased=False),
                dq25,
                dq50,
                radius.mean(),
                radius.std(unbiased=False),
                xy[:, 0].std(unbiased=False) if xy.shape[0] > 1 else zero,
                xy[:, 1].std(unbiased=False) if xy.shape[0] > 1 else zero,
                dem_mean,
                dem_std,
                dem_abs,
                pos_frac,
                neg_frac,
                pos_sum,
                neg_sum,
                route_mean,
                tw_width_mean,
                tw_width_std,
                tw_start_mean,
                tw_end_mean,
                service_mean + depot_to_centroid,
            ]))
        return torch.stack(feats, dim=0)

    def compute_matrix_manual_features(self, mat, node_mask):
        feats = []
        for i in range(mat.shape[0]):
            n = int(node_mask[i].sum().item())
            cur = mat[i, :n, :n]
            zero = cur.new_tensor(0.0)
            eye = torch.eye(n, dtype=torch.bool, device=cur.device)
            off = cur[~eye] if n > 1 else cur.new_zeros(1)
            q25, q50, q75 = self._quantiles(off)
            masked = cur.masked_fill(eye, float("inf"))
            nearest_out = masked.min(dim=1).values if n > 1 else cur.new_zeros(1)
            nearest_in = masked.min(dim=0).values if n > 1 else cur.new_zeros(1)
            asym = (cur - cur.t()).abs()
            row_mean = cur.mean(dim=1)
            col_mean = cur.mean(dim=0)
            row_std = cur.std(dim=1, unbiased=False)
            col_std = cur.std(dim=0, unbiased=False)
            row_min = masked.min(dim=1).values if n > 1 else cur.new_zeros(1)
            row_max = masked.masked_fill(torch.isinf(masked), zero).max(dim=1).values if n > 1 else cur.new_zeros(1)
            col_min = masked.min(dim=0).values if n > 1 else cur.new_zeros(1)
            col_max = masked.masked_fill(torch.isinf(masked), zero).max(dim=0).values if n > 1 else cur.new_zeros(1)
            if off.numel() > 1:
                mean = off.mean()
                std = off.std(unbiased=False).clamp_min(1e-6)
                skew = ((off - mean) ** 3).mean() / (std ** 3 + 1e-9)
            else:
                mean = off.mean()
                std = zero
                skew = zero
            k = min(256, max(1, n * n))
            idx = torch.randint(0, n, (k, 3), device=cur.device) if n > 1 else torch.zeros(1, 3, dtype=torch.long, device=cur.device)
            a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
            tri_vio = (cur[a, b] > (cur[a, c] + cur[c, b])).float().mean() if n > 1 else zero
            feats.append(torch.stack([
                torch.log1p(cur.new_tensor(float(n))) / 5.0,
                cur.new_tensor(float(n)) / 256.0,
                mean,
                std,
                q25,
                q50,
                q75,
                off.max(),
                off.min(),
                nearest_out.mean(),
                nearest_out.std(unbiased=False),
                nearest_in.mean(),
                nearest_in.std(unbiased=False),
                asym.mean(),
                asym.max(),
                row_mean.std(unbiased=False),
                col_mean.std(unbiased=False),
                row_min.mean(),
                row_max.mean(),
                col_min.mean(),
                col_max.mean(),
                cur.diag().abs().max(),
                tri_vio,
                skew,
                row_mean.mean(),
                col_mean.mean(),
                row_std.mean(),
                col_std.mean(),
                (nearest_out - nearest_in).abs().mean(),
                std / (mean.abs() + 1e-9),
                asym.mean() / (mean.abs() + 1e-9),
                cur.diag().mean(),
            ]))
        return torch.stack(feats, dim=0)

    def compute_pair_features(self, batch):
        h = self.encode_instance(batch)  # (B, d)
        B = h.shape[0]
        M = M_GLOBAL
        # Solver embeddings
        e = self.solver_emb.weight  # (M, d)
        # Pair features: (B, M, 4d)
        h_exp = h.unsqueeze(1).expand(B, M, self.d)
        e_exp = e.unsqueeze(0).expand(B, M, self.d)
        z = torch.cat([h_exp, e_exp, h_exp * e_exp, (h_exp - e_exp).abs()], dim=-1)
        return z, e_exp

    def forward(self, batch, return_support: bool = False):
        """Compute logits over all M_GLOBAL solvers (unavailable get -inf via mask)."""
        z, e_exp = self.compute_pair_features(batch)
        logits = self.head(z).squeeze(-1)  # (B, M)
        B = logits.shape[0]
        dtype = logits.dtype
        device = logits.device
        support_logits = None
        if self.use_support_head:
            support_logits = torch.stack(
                [head(z).squeeze(-1) for head in self.support_heads],
                dim=1,
            )  # (B, G, M)
        if self.use_problem_residual_head:
            logits = logits + self.problem_heads[batch["problem_id"]](z).squeeze(-1)
            if support_logits is not None:
                support_logits = support_logits + self.support_problem_heads[batch["problem_id"]](z).squeeze(-1).unsqueeze(1)
        if self.use_constraint_experts:
            expert_weights = self.constraint_weight_vector(batch, dtype=dtype, device=device)
            expert_logits = torch.stack([
                head(z).squeeze(-1) for head in self.constraint_expert_heads
            ], dim=1)  # (B, 8, M)
            logits = logits + (expert_logits * expert_weights.unsqueeze(-1)).sum(dim=1) / expert_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            if support_logits is not None:
                support_expert_logits = torch.stack([
                    head(z).squeeze(-1) for head in self.support_constraint_expert_heads
                ], dim=1)  # (B, 8, M)
                support_residual = (support_expert_logits * expert_weights.unsqueeze(-1)).sum(dim=1) / expert_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                support_logits = support_logits + support_residual.unsqueeze(1)
        # Per-problem per-solver learnable bias
        if self.use_problem_solver_bias:
            pid = batch["problem_id"]
            logits = logits + self.problem_solver_bias[pid]  # (M,) broadcast over B
        if self.use_descriptor_solver_bias:
            logits = logits + self.desc_solver_bias(batch["problem_desc"].to(device=device, dtype=dtype))
        # MVRP factorization add-on (applied only for MVRP problems)
        if self.use_fact and (self.use_problem_descriptor or batch["problem_id"] >= 3):
            v = batch["cbits"]                # (B, K_CBITS)
            cd = batch["coord_dist"]          # (B,)
            v_e = self.W_c(v)                 # (B, d)
            d_e = self.W_d(cd)                # (B, d)
            v_s = v_e.unsqueeze(1) * e_exp    # (B, M, d)
            d_s = d_e.unsqueeze(1) * e_exp    # (B, M, d)
            vd_s = v_s * d_e.unsqueeze(1)     # (B, M, d)  interaction
            z_fact = torch.cat([v_s, d_s, vd_s], dim=-1)  # (B, M, 3d)
            fact_logits = self.fact_head(z_fact).squeeze(-1)
            if self.use_problem_descriptor:
                desc = batch["problem_desc"].to(device=device, dtype=dtype)
                # Apply the factorized MVRP-style add-on only when extra routing
                # constraints beyond plain CVRP are present (O/B/BP/L/TW/MD).
                gate = ((v[:, 1:].sum(dim=1) + desc[:, 3] + desc[:, 6]) > 0).to(dtype).unsqueeze(-1)
                logits = logits + gate * fact_logits
            else:
                logits = logits + fact_logits
        # Apply availability mask: unavailable solvers -> -inf
        mask = batch["mask"]   # (B, M) {0,1}
        logits = logits + torch.log(mask.clamp_min(1e-30))
        if support_logits is not None:
            if self.support_generators > 1:
                support_logits = support_logits + self.support_generator_solver_bias.unsqueeze(0)
            support_logits = support_logits + torch.log(mask.clamp_min(1e-30)).unsqueeze(1)
        if return_support:
            if support_logits is not None and self.support_generators == 1:
                support_logits = support_logits.squeeze(1)
            return logits, support_logits
        return logits


def shortlist_from_support(logits_pool: torch.Tensor,
                           support_logits_pool: torch.Tensor | None,
                           threshold: float = 0.5,
                           topk: int = 3) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply an optional shortlist mask derived from support logits before final ranking."""
    if support_logits_pool is None:
        return logits_pool, None, None
    if support_logits_pool.dim() == 2:
        support_prob_all = torch.sigmoid(support_logits_pool).unsqueeze(1)
    elif support_logits_pool.dim() == 3:
        support_prob_all = torch.sigmoid(support_logits_pool)
    else:
        raise ValueError(f"support_logits_pool must be 2D or 3D, got shape={tuple(support_logits_pool.shape)}")
    shortlist_all = support_prob_all >= threshold
    if topk > 0:
        kk = min(max(1, topk), support_prob_all.shape[-1])
        top_idx = support_prob_all.topk(kk, dim=2).indices
        top_mask = torch.zeros_like(shortlist_all)
        top_mask.scatter_(2, top_idx, True)
        shortlist_all = shortlist_all | top_mask
    shortlist = shortlist_all.any(dim=1)
    support_prob = support_prob_all.max(dim=1).values
    empty = ~shortlist.any(dim=1)
    if empty.any():
        best = support_prob.argmax(dim=1, keepdim=True)
        shortlist = shortlist.clone()
        shortlist[empty] = False
        shortlist.scatter_(1, best, True)
    masked_logits = logits_pool.masked_fill(~shortlist, float("-inf"))
    return masked_logits, shortlist, support_prob


def support_targets_from_costs(costs: torch.Tensor, eps: float = 0.01, topk: int = 3) -> torch.Tensor:
    """Build shortlist targets from near-best relative regret and/or oracle top-k membership."""
    best = costs.min(dim=1, keepdim=True).values
    rel_gap = (costs - best) / (best.abs() + 1e-9)
    target = rel_gap <= eps
    if topk > 0:
        kk = min(max(1, topk), costs.shape[1])
        top_idx = costs.topk(kk, dim=1, largest=False).indices
        top_mask = torch.zeros_like(target)
        top_mask.scatter_(1, top_idx, True)
        target = target | top_mask
    return target.to(costs.dtype)


def support_asymmetric_bce_loss(support_logits_pool: torch.Tensor, costs: torch.Tensor,
                                eps: float = 0.01, topk: int = 3,
                                gamma_neg: float = 4.0, pos_weight: float = 2.0,
                                solver_pos_weight: torch.Tensor | None = None,
                                diversity_weight: float = 0.0,
                                budget_weight: float = 0.0,
                                target_mode: str = "legacy") -> torch.Tensor:
    """Asymmetric BCE for shortlist supervision, emphasizing recall on rare positive solvers."""
    if support_logits_pool.dim() == 2:
        support_logits_pool = support_logits_pool.unsqueeze(1)
    if support_logits_pool.dim() != 3:
        raise ValueError(f"support_logits_pool must be 2D or 3D, got shape={tuple(support_logits_pool.shape)}")

    target_base = support_targets_from_costs(costs, eps=eps, topk=topk)
    if solver_pos_weight is not None:
        solver_pos_weight = solver_pos_weight.to(costs.device, costs.dtype).view(1, -1)

    losses = []
    n_heads = support_logits_pool.shape[1]
    k_pool = costs.shape[1]
    topk_eff = min(max(1, topk), k_pool)
    top_idx = costs.topk(topk_eff, dim=1, largest=False).indices
    top_mask = torch.zeros_like(target_base)
    top_mask.scatter_(1, top_idx, 1.0)
    for g in range(n_heads):
        if target_mode == "rank_partition" and n_heads > 1:
            if g < topk_eff:
                cur_target = torch.zeros_like(target_base)
                cur_target.scatter_(1, top_idx[:, g:g+1], 1.0)
                cur_pos_scale = 1.5 + 0.25 * float(g == 0)
            elif g == topk_eff:
                cur_target = (target_base - top_mask).clamp_min(0.0)
                cur_pos_scale = 1.0
            else:
                cur_target = torch.zeros_like(target_base)
                cur_pos_scale = 1.0
            target_ratio = cur_target.mean(dim=1)
        else:
            if g == 0:
                cur_target = target_base
                cur_pos_scale = 1.0
                target_ratio = costs.new_full((costs.shape[0],), min(max(1, topk), 2) / max(1, k_pool))
            elif g == 1:
                cur_target = support_targets_from_costs(costs, eps=max(0.0, eps * 0.5), topk=max(1, topk - 1))
                cur_pos_scale = 1.5
                target_ratio = costs.new_full((costs.shape[0],), 1.0 / max(1, k_pool))
            else:
                cur_target = support_targets_from_costs(costs, eps=0.0, topk=1)
                cur_pos_scale = 2.0
                target_ratio = costs.new_full((costs.shape[0],), 1.0 / max(1, k_pool))
        cur_logits = support_logits_pool[:, g]
        prob = torch.sigmoid(cur_logits)
        bce = F.binary_cross_entropy_with_logits(cur_logits, cur_target, reduction="none")
        pos_w = torch.full_like(cur_target, pos_weight * cur_pos_scale)
        if solver_pos_weight is not None:
            pos_w = pos_w * solver_pos_weight
        neg_w = prob.pow(gamma_neg)
        weight = torch.where(cur_target > 0.5, pos_w, neg_w)
        cur_loss = (bce * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        if budget_weight > 0:
            cur_loss = cur_loss + budget_weight * (prob.mean(dim=1) - target_ratio).pow(2)
        losses.append(cur_loss)

    loss = torch.stack(losses, dim=1).mean()
    if n_heads > 1 and diversity_weight > 0:
        probs = torch.sigmoid(support_logits_pool)
        if target_mode == "rank_partition":
            overlap_mask = torch.ones_like(target_base)
        else:
            overlap_mask = (1.0 - target_base)
        overlap = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                overlap.append((probs[:, i] * probs[:, j] * overlap_mask).mean())
        if overlap:
            loss = loss + diversity_weight * torch.stack(overlap).mean()
    return loss


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


def masked_ranking_loss(logits_global: torch.Tensor, pool_ids: torch.Tensor,
                        costs: torch.Tensor, rank_topk: int = 3) -> torch.Tensor:
    """NSS-style iterative ranking loss over the available solver pool."""
    logits_pool = logits_global[:, pool_ids]
    k_pool = logits_pool.shape[1]
    topk = min(max(1, rank_topk), k_pool)
    loss = 0.0
    for i in range(topk):
        cur_k = k_pool - i
        cur_cost, ind = costs.topk(cur_k, dim=1, largest=True)
        cur_label = cur_cost.min(dim=1).indices
        cur_logits = torch.take_along_dim(logits_pool, ind, dim=1)
        loss = loss + F.nll_loss(F.log_softmax(cur_logits, dim=1), cur_label)
    return loss / topk


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


def soft_sbs_risk_loss(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor,
                       sbs_pool_idx: int, tau: float = 0.02, w_ce: float = 0.5,
                       w_risk: float = 2.0, risk_cap: float = 0.05,
                       w_switch_hinge: float = 0.0, switch_gap: float = 0.001,
                       switch_margin: float = 0.10) -> torch.Tensor:
    """Soft targets plus expected normalized delta-vs-SBS under the predicted policy.

    This directly penalizes unsafe switching while retaining listwise supervision.
    Positive delta-vs-SBS means higher cost than SBS (bad); negative is better than SBS.
    """
    logits_pool = logits_global[:, pool_ids]                 # (B, K_p)
    log_p = F.log_softmax(logits_pool, dim=1)
    p = torch.softmax(logits_pool, dim=1)

    soft = regret_soft_targets(costs, tau=tau)
    loss_ce = -(soft * log_p).sum(dim=1).mean()

    sbs = costs[:, sbs_pool_idx:sbs_pool_idx + 1]
    delta_vs_sbs = (costs - sbs) / (sbs.abs() + 1e-9)
    delta_vs_sbs = delta_vs_sbs.clamp(min=-risk_cap, max=risk_cap) / risk_cap
    loss_risk = (p * delta_vs_sbs).sum(dim=1).mean()

    best_idx = costs.argmin(dim=1)
    best_cost = costs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    sbs_cost = sbs.squeeze(1)
    gap_vs_sbs = ((sbs_cost - best_cost) / (sbs_cost.abs() + 1e-9)).clamp_min(0)
    switch_mask = gap_vs_sbs > switch_gap
    if w_switch_hinge > 0 and switch_mask.any():
        best_logit = logits_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        sbs_logit = logits_pool[:, sbs_pool_idx]
        hinge = torch.relu(switch_margin - (best_logit - sbs_logit))
        loss_switch = (gap_vs_sbs[switch_mask] * hinge[switch_mask]).mean()
    else:
        loss_switch = torch.tensor(0.0, device=logits_global.device)

    return w_ce * loss_ce + w_risk * loss_risk + w_switch_hinge * loss_switch


def gap_regression_rank_loss(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor,
                             sbs_pool_idx: int, gap_reference: str = "sbs",
                             gap_cap: float = 0.25, huber_delta: float = 0.05,
                             w_reg: float = 1.0, w_pair: float = 1.0,
                             pair_sample_topk: int = 0) -> torch.Tensor:
    """Regress normalized cost gaps and preserve solver ordering with pairwise logistic loss."""
    logits_pool = logits_global[:, pool_ids]
    if gap_reference == "sbs":
        ref = costs[:, sbs_pool_idx:sbs_pool_idx + 1]
    elif gap_reference == "best":
        ref = costs.min(dim=1, keepdim=True).values
    else:
        raise ValueError(f"unknown gap_reference={gap_reference}")

    target_gap = (costs - ref) / (ref.abs() + 1e-9)
    target_gap = target_gap.clamp(min=-gap_cap, max=gap_cap)
    pred_gap = -logits_pool
    loss_reg = F.huber_loss(pred_gap, target_gap, delta=huber_delta, reduction="none").mean(dim=1)

    gap_ji = target_gap.unsqueeze(1) - target_gap.unsqueeze(2)   # gap_j - gap_i
    score_ij = logits_pool.unsqueeze(2) - logits_pool.unsqueeze(1)  # score_i - score_j
    pair_sign = torch.sign(gap_ji)
    pair_weight = gap_ji.abs()
    pair_mask = pair_sign != 0
    k_pool = logits_pool.shape[1]
    tri = torch.triu(torch.ones(k_pool, k_pool, dtype=torch.bool, device=logits_pool.device), diagonal=1)
    pair_mask = pair_mask & tri.unsqueeze(0)
    if pair_sample_topk > 0 and pair_sample_topk < k_pool:
        focus = torch.zeros_like(target_gap, dtype=torch.bool)
        top_idx = target_gap.topk(pair_sample_topk, dim=1, largest=False).indices
        focus.scatter_(1, top_idx, True)
        pair_mask = pair_mask & (focus.unsqueeze(1) | focus.unsqueeze(2))
    if pair_mask.any():
        pair_loss = F.softplus(-pair_sign * score_ij)
        pair_loss = (pair_loss * pair_weight).masked_select(pair_mask).sum() / pair_weight.masked_select(pair_mask).sum().clamp_min(1e-6)
    else:
        pair_loss = logits_global.new_tensor(0.0)
    return w_reg * loss_reg.mean() + w_pair * pair_loss


def winner_margin_loss(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor,
                       base_margin: float = 0.05, gap_scale: float = 2.0,
                       class_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Push the oracle-best solver above the strongest competitor by a cost-aware margin."""
    logits_pool = logits_global[:, pool_ids]
    best_idx = costs.argmin(dim=1)
    sorted_costs, _ = costs.sort(dim=1)
    if costs.shape[1] > 1:
        gap12 = ((sorted_costs[:, 1] - sorted_costs[:, 0]) / (sorted_costs[:, 0].abs() + 1e-9)).clamp_min(0)
    else:
        gap12 = torch.zeros(costs.shape[0], device=costs.device, dtype=costs.dtype)
    target_margin = base_margin + gap_scale * gap12
    best_logit = logits_pool.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    comp_logits = logits_pool.clone()
    comp_logits.scatter_(1, best_idx.unsqueeze(1), float("-inf"))
    competitor = comp_logits.max(dim=1).values
    loss = F.relu(target_margin - (best_logit - competitor))
    if class_weight is not None:
        w = class_weight.to(costs.device, costs.dtype).gather(0, best_idx)
        loss = loss * w
    return loss.mean()
