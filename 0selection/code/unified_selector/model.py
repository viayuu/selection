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

from .registry import PROBLEMS, P2I, M_GLOBAL, K_CBITS, D_COORD, POOLS, S2I


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

    def forward(self, node, node_mask, return_tokens: bool = False):
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
        if return_tokens:
            return pooled, x   # (B, d), (B, N, d)
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

    def forward(self, mat, node_mask, return_tokens: bool = False):
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
            # Triangle inequality violation rate: deterministic grid in eval, random subsample in train.
            k = min(256, n*n)
            if self.training:
                idx = torch.randint(0, n, (k, 3), device=m.device)
            else:
                # Deterministic cyclic triplets — reproducible across reruns of the same checkpoint.
                ar = torch.arange(k, device=m.device)
                a = ar % n
                b = (ar // n) % n
                c = (ar // (n * n) + ar % 7) % n
                idx = torch.stack([a, b, c], dim=1)
            a, b, c = idx[:,0], idx[:,1], idx[:,2]
            ti = (m[a,b] > (m[a,c] + m[c,b])).float().mean()
            diag_max = m.diag().abs().max()
            feats.append(torch.stack([mean, std, mx, mn, skew, sym, row_std, col_std, ti, diag_max]))
        x = torch.stack(feats)  # (B, 10)
        # Normalize roughly
        x = x / (x.abs().mean(dim=0, keepdim=True) + 1e-6)
        pooled = self.proj(x)
        if return_tokens:
            # For ATSP there is no node token sequence; expose pooled as a length-1 "token" so
            # arm-attention still has something to attend over.  Mask is all-True.
            tokens = pooled.unsqueeze(1)                        # (B, 1, d)
            mask = torch.ones(B, 1, dtype=torch.bool, device=pooled.device)
            return pooled, tokens, mask
        return pooled


def _migrate_state_dict(sd: dict) -> dict:
    """Map old `self.head = Sequential(Linear, GELU, Dropout, Linear)` keys
    ('head.0.*', 'head.3.*') onto the new split head
    ('head_linear1.*', 'head_linear2.*').  Idempotent.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("head.0."):
            out["head_linear1." + k[len("head.0."):]] = v
        elif k.startswith("head.3."):
            out["head_linear2." + k[len("head.3."):]] = v
        else:
            out[k] = v
    return out


class ArmAttentionHead(nn.Module):
    """Per-arm cross-attention over node tokens.

    Each solver emb becomes a query; node tokens become keys/values.  The attended output is
    combined with the arm-specific pair features (h, e_s, h*e_s, |h-e_s|) through a fusion MLP.
    Rationale: the pure dot-product (h · e_s) head cannot distinguish instances that share a
    pooled summary but differ node-wise — a known bottleneck in NSS (Table 1) where top1 stays
    near 70% even with specialized solvers because the selector sees only g_b.  Giving each arm
    its own attention query over node tokens is a standard strong head (URS §3.2 arm-query
    attention; CoE decoder-side expert attention).
    """

    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.post_ln = nn.LayerNorm(d)
        self.fuse = nn.Sequential(
            nn.Linear(5 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.GELU(),
        )

    def forward(self, node_tokens, node_mask, solver_emb, h_pooled):
        """node_tokens: (B, N, d), node_mask: (B, N) bool True=valid, solver_emb: (M, d),
        h_pooled: (B, d).  Returns fused per-arm representation (B, M, d)."""
        B, N, d = node_tokens.shape
        M = solver_emb.shape[0]
        # Query: expand solver_emb to (B, M, d)
        q = self.q_proj(solver_emb).unsqueeze(0).expand(B, M, d).contiguous()
        k = self.k_proj(node_tokens)
        v = self.v_proj(node_tokens)
        kpm = ~node_mask  # True at PAD
        attn_out, _ = self.attn(q, k, v, key_padding_mask=kpm, need_weights=False)   # (B, M, d)
        attn_out = self.post_ln(attn_out)
        # Fuse with pair features (h, e_s, h*e_s, |h-e_s|, attn_out)
        e_exp = solver_emb.unsqueeze(0).expand(B, M, d)
        h_exp = h_pooled.unsqueeze(1).expand(B, M, d)
        z = torch.cat([h_exp, e_exp, h_exp * e_exp, (h_exp - e_exp).abs(), attn_out], dim=-1)
        return self.fuse(z)


class CoEExpertHead(nn.Module):
    """Constraint-of-Experts FFN head (CoEKS-style).

    Maintain K_CBITS+1 parallel FFN experts (one per constraint bit + a shared "always-on").
    Per-instance gating uses the cbits vector to softly weight expert outputs:
        g(cbits) = softmax(W_g · [cbits; 1])        # (K_CBITS+1,)
    Final pair logit = sum_k g_k · expert_k(z).

    Targets MVRP arm collapse: different constraint signatures get routed to different experts
    so the selector no longer defaults to "always RELD_MTL" for every L/TW/BL/BTW variant.
    """

    def __init__(self, d_in: int, head_hidden: int, dropout: float, num_experts: int = None):
        super().__init__()
        if num_experts is None:
            num_experts = K_CBITS + 1  # 5 constraint bits + 1 shared
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, head_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            ) for _ in range(num_experts)
        ])
        # Gating: concat cbits + [1] (always-on bias)
        self.gate = nn.Sequential(
            nn.Linear(K_CBITS + 1, 2 * num_experts), nn.GELU(),
            nn.Linear(2 * num_experts, num_experts),
        )

    def forward(self, z, cbits):
        """z: (B, M, d_in), cbits: (B, K_CBITS) float.  Returns (B, M) logits."""
        B = z.shape[0]
        gate_in = torch.cat([cbits, torch.ones(B, 1, device=z.device, dtype=cbits.dtype)], dim=-1)
        gate_logits = self.gate(gate_in)                       # (B, num_experts)
        gate = torch.softmax(gate_logits, dim=-1)              # (B, num_experts)
        # Expert outputs: each produces (B, M, 1) → stack to (B, num_experts, M)
        outs = torch.stack([expert(z).squeeze(-1) for expert in self.experts], dim=1)  # (B, E, M)
        # Gate-weighted sum over experts
        logits = (gate.unsqueeze(-1) * outs).sum(dim=1)        # (B, M)
        return logits


class ReZeroAddNorm(nn.Module):
    """input + alpha * residual with learnable scalar alpha init to 0.

    Port of NSS Add_And_Normalization_Module(norm="rezero"). Used as the
    residual-combiner inside ReZeroEncoderLayer so deep stacks start near identity.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, residual):
        return x + self.alpha * residual


class ReZeroEncoderLayer(nn.Module):
    """Transformer encoder layer with ReZero residuals (no LayerNorm)."""

    def __init__(self, d: int, heads: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=heads,
                                          dropout=dropout, batch_first=True)
        self.rz_attn = ReZeroAddNorm()
        self.ff = nn.Sequential(
            nn.Linear(d, ff_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_hidden, d),
        )
        self.rz_ff = ReZeroAddNorm()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = self.rz_attn(x, self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.rz_ff(x, self.drop(ff_out))
        return x


class HierarchicalBlock(nn.Module):
    """One NSS-style block: encoder_layer_num ReZero attention layers,
    then masked mean+max pool → graph embedding, then top-k downsample.
    """

    def __init__(self, d: int, heads: int, ff_hidden: int, encoder_layer_num: int,
                 dropout: float, downsample_ratio: float):
        super().__init__()
        self.layers = nn.ModuleList([
            ReZeroEncoderLayer(d, heads, ff_hidden, dropout)
            for _ in range(encoder_layer_num)
        ])
        self.score_layer = ReZeroEncoderLayer(d, heads, ff_hidden, dropout)
        self.score_head = nn.Linear(d, 1)
        self.act = nn.Tanh()
        self.nonlinear = nn.GELU()
        self.downsample_ratio = downsample_ratio

    def forward(self, x, mask):
        """x: (B, N, d)  mask: (B, N) bool (True=valid).
        Returns (graph_emb (B,2d), new_x (B,K,d), new_mask (B,K) bool)."""
        kpm = ~mask
        for layer in self.layers:
            x = layer(x, key_padding_mask=kpm)

        m = mask.unsqueeze(-1).float()
        mean_e = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_e = x_masked.max(dim=1).values
        graph_emb = self.nonlinear(torch.cat([mean_e, max_e], dim=-1))

        score_x = self.score_layer(x, key_padding_mask=kpm)
        scores = self.act(self.score_head(score_x)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        B, N, d = x.shape
        lengths = mask.sum(dim=1)
        keeps = (lengths.float() * self.downsample_ratio).long().clamp_min(1)
        K = int(keeps.max().item())
        top_scores, top_idx = scores.topk(K, dim=1)
        new_x = x.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, d))
        new_mask = torch.arange(K, device=x.device).unsqueeze(0) < keeps.unsqueeze(1)
        # gate with scores — but zero out padded positions so -inf doesn't propagate to next attention
        gate = top_scores.masked_fill(~new_mask, 0.0).unsqueeze(-1)
        new_x = new_x + gate
        # hard-zero padded rows so any residual NaN/-inf in their features is wiped before next block
        new_x = new_x * new_mask.unsqueeze(-1).to(new_x.dtype)
        return graph_emb, new_x, new_mask


class HierarchicalCoordEncoder(nn.Module):
    """NSS-style hierarchical pooling encoder for the unified 18-problem setting.
    Signature matches CoordEncoder so UnifiedSelector can swap via encoder_type.
    return_tokens=True yields 3-tuple (pooled, tokens, tokens_mask).
    """

    def __init__(self, d_in_max: int = 8, d: int = 128, block_num: int = 2,
                 encoder_layer_num: int = 2, heads: int = 8,
                 downsample_ratio: float = 0.8, dropout: float = 0.1):
        super().__init__()
        self.d_in_max = d_in_max
        self.d = d
        self.block_num = block_num
        self.in_proj = nn.Linear(d_in_max, d)
        ff_hidden = 2 * d
        self.blocks = nn.ModuleList([
            HierarchicalBlock(d=d, heads=heads, ff_hidden=ff_hidden,
                              encoder_layer_num=encoder_layer_num, dropout=dropout,
                              downsample_ratio=downsample_ratio)
            for _ in range(block_num)
        ])
        self.out_proj = nn.Linear(2 * d, d)
        self.nonlinear = nn.GELU()

    def forward(self, node, node_mask, return_tokens: bool = False):
        B, N, d_in = node.shape
        if d_in < self.d_in_max:
            pad = torch.zeros(B, N, self.d_in_max - d_in, device=node.device, dtype=node.dtype)
            node = torch.cat([node, pad], dim=-1)
        elif d_in > self.d_in_max:
            node = node[..., :self.d_in_max]
        x = self.in_proj(node)
        mask = node_mask

        graph_emb_h = None
        for block in self.blocks:
            graph_emb, x, mask = block(x, mask)
            graph_emb_h = graph_emb if graph_emb_h is None else graph_emb_h + graph_emb

        m = mask.unsqueeze(-1).float()
        mean_e = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_e = x_masked.max(dim=1).values
        final_emb = self.nonlinear(torch.cat([mean_e, max_e], dim=-1))
        graph_emb_h = graph_emb_h + final_emb if graph_emb_h is not None else final_emb

        pooled = self.out_proj(graph_emb_h)
        if return_tokens:
            return pooled, x, mask
        return pooled


class LocalProblemHead(nn.Module):
    """Per-problem MLP head. 18 independent 2-layer MLPs, one per problem.
    Outputs (B, M_GLOBAL) with logits filled at the pool-ids positions.
    """

    def __init__(self, d: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pool_lens = [len(POOLS[p]) for p in PROBLEMS]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, K_p),
            )
            for K_p in self.pool_lens
        ])
        for i, p in enumerate(PROBLEMS):
            ids = torch.tensor([S2I[s] for s in POOLS[p]], dtype=torch.long)
            self.register_buffer(f"pool_ids_{i}", ids, persistent=False)

    def forward(self, h: torch.Tensor, problem_id: int) -> torch.Tensor:
        B = h.shape[0]
        local = self.heads[problem_id](h)
        pool_ids = getattr(self, f"pool_ids_{problem_id}")
        out = h.new_zeros(B, M_GLOBAL)
        out[:, pool_ids] = local
        return out


class UnifiedSelector(nn.Module):
    def __init__(self, d: int = 128, depth: int = 4, dropout: float = 0.1,
                 d_coord_in: int = 8, head_hidden: int = 256,
                 use_mvrp_factorized: bool = True,
                 use_problem_solver_bias: bool = True,
                 use_film: bool = False,
                 use_arm_attn: bool = False,
                 use_coe: bool = False,
                 arm_attn_heads: int = 4,
                 coe_experts: int = None,
                 encoder_type: str = "standard",
                 rezero: bool = False,
                 block_num: int = 2,
                 encoder_layer_num: int = 2,
                 heads: int = 4,
                 downsample_ratio: float = 0.8,
                 local_head: bool = False):
        super().__init__()
        self.d = d
        self.head_hidden = head_hidden
        self.encoder_type = encoder_type
        if encoder_type == "nss_hierarchical":
            # ReZero is intrinsic to the NSS hierarchical encoder; the `rezero` flag
            # is accepted here for CLI symmetry but the encoder always uses ReZero.
            self.coord_enc = HierarchicalCoordEncoder(
                d_in_max=d_coord_in, d=d, block_num=block_num,
                encoder_layer_num=encoder_layer_num, heads=heads,
                downsample_ratio=downsample_ratio, dropout=dropout,
            )
        else:
            self.coord_enc = CoordEncoder(d_in_max=d_coord_in, d=d, depth=depth,
                                          heads=heads, dropout=dropout)
        self.matrix_enc = MatrixEncoder(d=d)
        # Metadata
        self.prob_emb = nn.Embedding(len(PROBLEMS), d)
        self.cbits_proj = nn.Linear(K_CBITS, d, bias=False)
        self.coord_dist_emb = nn.Embedding(D_COORD, d)
        self.metadata_ln = nn.LayerNorm(d)
        # Solver embeddings (global vocabulary)
        self.solver_emb = nn.Embedding(M_GLOBAL, d)
        # Default pair-feature head (always exists, used when use_arm_attn=False).
        # Split into two linears + activation so FiLM can inject between.
        self.head_linear1 = nn.Linear(4*d, head_hidden)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(dropout)
        self.head_linear2 = nn.Linear(head_hidden, 1)
        # FiLM modulation from problem embedding (optional).
        self.use_film = use_film
        if use_film:
            self.film_mlp = nn.Sequential(
                nn.Linear(d, 2*head_hidden),
            )
            # Zero-init so initial forward is identity-like (gamma ≈ 0, beta = 0 → 1+gamma multiplier ≈ 1).
            nn.init.zeros_(self.film_mlp[0].weight)
            nn.init.zeros_(self.film_mlp[0].bias)
        # Arm-conditioned cross-attention head (optional): replaces the default
        # dot-product head with a richer per-arm cross-attention over node tokens.
        self.use_arm_attn = use_arm_attn
        if use_arm_attn:
            self.arm_attn = ArmAttentionHead(d=d, heads=arm_attn_heads, dropout=dropout)
            # Linear from arm-attn output (d) → scalar logit (used when CoE is disabled).
            self.arm_attn_linear = nn.Linear(d, 1)
        # CoE constraint-expert head (optional): replaces head_linear2 with gated
        # per-constraint expert FFN.  When enabled together with use_arm_attn, we
        # feed arm-attention output (B, M, d) to CoE; otherwise we feed pair features (B, M, 4d).
        self.use_coe = use_coe
        if use_coe:
            d_in = d if use_arm_attn else 4 * d
            self.coe = CoEExpertHead(d_in=d_in, head_hidden=head_hidden,
                                     dropout=dropout, num_experts=coe_experts)
        # Problem × solver learned bias table (init 0) — captures per-problem solver priors.
        self.use_problem_solver_bias = use_problem_solver_bias
        if use_problem_solver_bias:
            self.problem_solver_bias = nn.Parameter(torch.zeros(len(PROBLEMS), M_GLOBAL))
        # Per-problem local head (R40C): each problem gets its own K_p-way MLP.
        # Blended with global head via a learnable sigmoid gate (init 0.5).
        self.use_local_head = local_head
        if local_head:
            self.local_head = LocalProblemHead(d=d, hidden=head_hidden, dropout=dropout)
            self.local_head_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
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

    def encode_instance(self, batch, return_tokens: bool = False):
        tokens = None
        tok_mask = None
        if batch["kind"] == "coord":
            if return_tokens:
                out = self.coord_enc(batch["node"], batch["node_mask"], return_tokens=True)
                if self.encoder_type == "nss_hierarchical":
                    # Hierarchical encoder downsamples nodes → returns (pooled, tokens, mask)
                    g, tokens, tok_mask = out
                else:
                    g, tokens = out
                    tok_mask = batch["node_mask"]
            else:
                g = self.coord_enc(batch["node"], batch["node_mask"])
        else:
            if return_tokens:
                g, tokens, tok_mask = self.matrix_enc(batch["matrix"], batch["node_mask"],
                                                      return_tokens=True)
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
        if return_tokens:
            return h, tokens, tok_mask
        return h

    def forward(self, batch):
        """Compute logits over all M_GLOBAL solvers (unavailable get -inf via mask)."""
        need_tokens = self.use_arm_attn
        if need_tokens:
            h, tokens, tok_mask = self.encode_instance(batch, return_tokens=True)
        else:
            h = self.encode_instance(batch)
            tokens = tok_mask = None
        B = h.shape[0]
        M = M_GLOBAL
        # Solver embeddings
        e = self.solver_emb.weight  # (M, d)
        # e_exp is always needed by the MVRP factorization add-on below, even when arm-attn
        # short-circuits the legacy pair-feature path.
        e_exp = e.unsqueeze(0).expand(B, M, self.d)
        # Per-arm feature construction.
        if self.use_arm_attn:
            # z_arm: (B, M, d) — fused cross-attention + pair features
            z_arm = self.arm_attn(tokens, tok_mask, e, h)
            if self.use_coe:
                logits = self.coe(z_arm, batch["cbits"])
            else:
                # Dedicated d → 1 projection on the fused representation
                logits = self.arm_attn_linear(self.head_drop(z_arm)).squeeze(-1)
        else:
            # Legacy 4d pair features
            h_exp = h.unsqueeze(1).expand(B, M, self.d)
            z = torch.cat([h_exp, e_exp, h_exp * e_exp, (h_exp - e_exp).abs()], dim=-1)
            if self.use_coe:
                logits = self.coe(z, batch["cbits"])
            else:
                # Head with optional FiLM conditioning from problem embedding.
                hx = self.head_linear1(z)        # (B, M, head_hidden)
                hx = self.head_act(hx)
                if self.use_film:
                    pid = torch.full((B,), batch["problem_id"], dtype=torch.long, device=h.device)
                    prob_vec = self.prob_emb(pid)           # (B, d)
                    film = self.film_mlp(prob_vec)          # (B, 2*head_hidden)
                    gamma, beta = film.chunk(2, dim=-1)     # each (B, head_hidden)
                    hx = hx * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # broadcast over M
                hx = self.head_drop(hx)
                logits = self.head_linear2(hx).squeeze(-1)  # (B, M)
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
        # Per-problem local head blend (R40C): λ·global + (1−λ)·local_scattered
        if self.use_local_head:
            pid = batch["problem_id"]
            local_logits = self.local_head(h, pid)                       # (B, M_GLOBAL)
            lam = torch.sigmoid(self.local_head_logit)
            logits = lam * logits + (1.0 - lam) * local_logits
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
                       soft_targets_pool: torch.Tensor,
                       focal_gamma: float = 0.0,
                       hard_upweight: float = 1.0,
                       oracle_idx_pool: torch.Tensor = None) -> torch.Tensor:
    """logits_global: (B, M_global) already masked via +log(mask).
    pool_ids: (K_p,) global ids for this problem.
    soft_targets_pool: (B, K_p)
    Return scalar loss = -sum_k y_k log softmax_s_valid(ℓ)[pool_ids[k]].

    If focal_gamma > 0, weight each sample's loss by (1 - p_oracle)^focal_gamma where
    p_oracle = softmax(logits_pool)[oracle_idx_pool].  This focuses training on cases
    the model currently gets wrong (focal-style hard-example emphasis).

    If hard_upweight != 1.0, upweight samples where the model's argmax disagrees with
    the oracle.  Similar intent but less smooth than focal; use one or the other.
    """
    # log_softmax over all M (unavailable are -inf so zero prob)
    log_p = F.log_softmax(logits_global, dim=1)  # (B, M)
    # Gather at pool_ids
    log_p_pool = log_p[:, pool_ids]              # (B, K_p)
    per_sample = -(soft_targets_pool * log_p_pool).sum(dim=1)    # (B,)

    # Optional focal / hard-mining reweighting (only if oracle_idx_pool provided).
    if (focal_gamma > 0 or hard_upweight != 1.0) and oracle_idx_pool is not None:
        p_pool = log_p_pool.exp()                              # (B, K_p)
        p_oracle = p_pool.gather(1, oracle_idx_pool.unsqueeze(1)).squeeze(1)  # (B,)
        if focal_gamma > 0:
            w_focal = (1.0 - p_oracle.clamp(0.0, 1.0)) ** focal_gamma
            per_sample = per_sample * w_focal
        if hard_upweight != 1.0:
            pred_arg = p_pool.argmax(dim=1)
            hard = (pred_arg != oracle_idx_pool).float()
            per_sample = per_sample * (1.0 + (hard_upweight - 1.0) * hard)
    return per_sample.mean()


def predict_cost(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Return the COST achieved by argmax-over-valid selection. (B,)"""
    log_p = F.log_softmax(logits_global, dim=1)
    log_p_pool = log_p[:, pool_ids]
    pred = log_p_pool.argmax(dim=1)  # index within pool
    return costs.gather(1, pred.unsqueeze(1)).squeeze(1), pred


def plackett_luce_loss(logits_pool: torch.Tensor, costs: torch.Tensor,
                        top_k: int = None) -> torch.Tensor:
    """Listwise Plackett-Luce NLL (NSS loss.py:7-19).
    logits_pool: (B, K_p) pool-restricted logits.  costs: (B, K_p).
    Iterates top_k stages; at each, log-softmax over still-alive arms, NLL on current best.
    """
    B, K = logits_pool.shape
    if top_k is None or top_k > K:
        top_k = K
    order = torch.argsort(costs, dim=1)                 # (B, K)  best-first by true cost
    alive = torch.ones_like(logits_pool, dtype=torch.bool)
    loss = logits_pool.new_zeros(())
    for step in range(top_k):
        log_p = torch.where(alive, logits_pool, torch.full_like(logits_pool, float("-inf")))
        log_p = F.log_softmax(log_p, dim=1)
        pick = order[:, step]                           # (B,)
        loss = loss + (-log_p.gather(1, pick.unsqueeze(1)).squeeze(1)).mean()
        alive = alive.scatter(1, pick.unsqueeze(1), False)
    return loss / top_k


def diversity_entropy_penalty(logits_pool: torch.Tensor) -> torch.Tensor:
    """Penalize batch-level arm-distribution collapse by encouraging non-zero entropy on the
    argmax histogram (soft version).  Used with small weight to counter "always pick arm 0".
    """
    p = torch.softmax(logits_pool, dim=1)               # (B, K)
    p_mean = p.mean(dim=0)                              # (K,) — average prob mass per arm
    ent = -(p_mean * (p_mean.clamp_min(1e-9)).log()).sum()
    logK = torch.log(torch.tensor(float(logits_pool.shape[1]), device=p.device))
    return (logK - ent) / logK.clamp_min(1e-9)          # in [0,1]; 0 when uniform


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


def gap_regression_rank_loss(logits_global: torch.Tensor, pool_ids: torch.Tensor, costs: torch.Tensor,
                             sbs_pool_idx: int, gap_reference: str = "sbs",
                             gap_cap: float = 0.25, huber_delta: float = 0.05,
                             w_reg: float = 1.0, w_pair: float = 1.0,
                             pair_sample_topk: int = 0) -> torch.Tensor:
    """Regress normalized cost gaps and preserve solver ordering with pairwise logistic loss.

    Ported from 0selection2 code/unified_selector/model.py:1262.
    - pred_gap = -logit; target_gap = (cost - ref)/|ref| clamped to [-gap_cap, gap_cap]
    - Huber regression on predicted gap
    - Pairwise softplus loss weighted by the absolute gap difference
    """
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
                       base_margin: float = 0.05, gap_scale: float = 2.0) -> torch.Tensor:
    """Push the oracle-best solver above the strongest competitor by a cost-aware margin.

    target_margin = base_margin + gap_scale * (second_best_gap / |best_cost|)
    Encourages larger logit gap when the oracle winner is clearly better.  Ported from 0selection2.
    """
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
    return torch.relu(target_margin - (best_logit - competitor)).mean()
