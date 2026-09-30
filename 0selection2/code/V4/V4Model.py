import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.unified_selector.registry import (
    D_COORD,
    K_CBITS,
    K_PROBLEM_DESC,
    M_GLOBAL,
    PROBLEMS,
)

from .solver_features import SOLVER_FEATURE_DIM, build_solver_features, get_solver_feature_spec


def _reshape_by_heads(x, head_num):
    batch, token, _ = x.size()
    qkv_dim = x.size(2) // head_num
    return x.reshape(batch, token, head_num, qkv_dim).transpose(1, 2)


def _mask_value(x):
    return torch.finfo(x.dtype).min


def _masked_mean(x, mask, dim=1):
    w = mask.float()
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


def _masked_max(x, mask, dim=1):
    neg = torch.finfo(x.dtype).min
    w = mask
    while w.dim() < x.dim():
        w = w.unsqueeze(-1)
    return x.masked_fill(~w, neg).max(dim=dim).values


class AddAndNorm(nn.Module):
    def __init__(self, embedding_dim, rezero=False):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.rezero = nn.Parameter(torch.zeros(1)) if rezero else None

    def forward(self, x, sub):
        if self.rezero is not None:
            sub = self.rezero * sub
        return self.norm(x + sub)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, dropout, sdpa=False):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.sdpa = sdpa
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.out = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_input, k_input, v_input, key_mask=None, attn_bias=None):
        q = _reshape_by_heads(self.Wq(q_input), self.head_num)
        k = _reshape_by_heads(self.Wk(k_input), self.head_num)
        v = _reshape_by_heads(self.Wv(v_input), self.head_num)
        if self.sdpa:
            bias = attn_bias.to(q.dtype) if attn_bias is not None else None
            if key_mask is not None:
                if bias is None:
                    bias = torch.zeros_like(key_mask[:, None, None, :], dtype=q.dtype)
                bias = bias.masked_fill(~key_mask[:, None, None, :], _mask_value(q))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).reshape(q_input.size(0), q_input.size(1), self.head_num * self.qkv_dim)
            return self.out(out)
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        if attn_bias is not None:
            score = score + attn_bias
        if key_mask is not None:
            score = score.masked_fill(~key_mask[:, None, None, :], _mask_value(score))
        weight = self.dropout(F.softmax(score, dim=-1))
        out = torch.matmul(weight, v)
        out = out.transpose(1, 2).reshape(q_input.size(0), q_input.size(1), self.head_num * self.qkv_dim)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.attn = MultiHeadAttention(d, params["head_num"], params["qkv_dim"], params["dropout"], params.get("sdpa", False))
        self.add_norm_1 = AddAndNorm(d, rezero=params.get("rezero", False))
        self.ffn = FeedForward(d, params["ff_hidden_dim"], params["dropout"])
        self.add_norm_2 = AddAndNorm(d, rezero=params.get("rezero", False))

    def forward(self, x, mask=None, attn_bias=None):
        x = self.add_norm_1(x, self.attn(x, x, x, key_mask=mask, attn_bias=attn_bias))
        return self.add_norm_2(x, self.ffn(x))


class ConditionEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.problem_emb = nn.Embedding(len(PROBLEMS), d)
        self.coord_dist_emb = nn.Embedding(D_COORD, d)
        self.kind_emb = nn.Embedding(2, d)
        self.cbits_proj = nn.Linear(K_CBITS, d, bias=False)
        self.desc_proj = nn.Linear(K_PROBLEM_DESC, d, bias=False)
        self.stats_proj = nn.Linear(params["stats_dim"], d)
        self.n_proj = nn.Linear(1, d)
        self.norm = nn.LayerNorm(d)
        self.use_coord_dist = not params.get("ignore_coord_dist", False)

    def forward(self, batch, stats, kind_id):
        batch_size = stats.size(0)
        device = stats.device
        pid = torch.full((batch_size,), int(batch["problem_id"]), dtype=torch.long, device=device)
        n_norm = batch["n"].float().to(device).log1p().unsqueeze(-1) / 6.0
        kind = torch.full((batch_size,), kind_id, dtype=torch.long, device=device)
        cond = (
            self.problem_emb(pid)
            + (self.coord_dist_emb(batch["coord_dist"]) if self.use_coord_dist else 0.0)
            + self.kind_emb(kind)
            + self.cbits_proj(batch["cbits"])
            + self.desc_proj(batch["problem_desc"])
            + self.stats_proj(stats)
            + self.n_proj(n_norm)
        )
        return self.norm(cond), pid


class InstanceEncoder(nn.Module):
    """NSS-style compact instance encoder with mean/max summaries and relation bias."""

    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.params = params
        self.coord_proj = nn.Linear(8, d)
        self.matrix_proj = nn.Linear(8, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.stats_token_proj = nn.Linear(params["stats_dim"], d)
        self.condition_encoder = ConditionEncoder(**params)
        self.layers = nn.ModuleList([EncoderLayer(**params) for _ in range(params["encoder_layer_num"])])
        self.coord_bias_scale = nn.Parameter(torch.ones(params["head_num"]))
        self.matrix_bias_scale = nn.Parameter(torch.ones(params["head_num"]))
        self.node_only = params.get("node_only", False)
        if not self.node_only:
            self.summary_proj = nn.Sequential(
                nn.Linear(3 * d, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
            self.norm = nn.LayerNorm(d)

    def _pad_node(self, node):
        if node.size(-1) < 8:
            pad = torch.zeros(node.size(0), node.size(1), 8 - node.size(-1), device=node.device, dtype=node.dtype)
            node = torch.cat([node, pad], dim=-1)
        return node[:, :, :8]

    def _coord_stats(self, node, mask):
        node = self._pad_node(node)
        xy = node[:, :, :2]
        dem = node[:, :, 2:3]
        route = node[:, :, 4:5]
        tw_width = (node[:, :, 7:8] - node[:, :, 6:7]).clamp_min(0.0)
        xy_mean = _masked_mean(xy, mask)
        xy_std = torch.sqrt(_masked_mean((xy - xy_mean[:, None, :]).pow(2), mask).clamp_min(1.0e-9))
        dem_mean = _masked_mean(dem, mask)
        dem_std = torch.sqrt(_masked_mean((dem - dem_mean[:, None, :]).pow(2), mask).clamp_min(1.0e-9))
        neg_frac = _masked_mean((dem < 0).float(), mask)
        route_mean = _masked_mean(route, mask)
        tw_mean = _masked_mean(tw_width, mask)
        tw_std = torch.sqrt(_masked_mean((tw_width - tw_mean[:, None, :]).pow(2), mask).clamp_min(1.0e-9))
        n_norm = mask.float().sum(dim=1, keepdim=True).log1p() / 6.0
        zeros = torch.zeros_like(n_norm)
        return torch.cat([xy_mean, xy_std, dem_mean, dem_std, neg_frac, route_mean, tw_mean, tw_std, n_norm, zeros], dim=-1)

    def _matrix_stats(self, mat, mask):
        valid = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(mat.size(1), dtype=torch.bool, device=mat.device)[None, :, :]
        valid = valid & ~eye
        vals = mat.masked_fill(~valid, 0)
        denom = valid.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = vals.sum(dim=(1, 2), keepdim=True) / denom
        std = torch.sqrt((((mat - mean) * valid.float()).pow(2).sum(dim=(1, 2), keepdim=True) / denom).clamp_min(1.0e-9))
        asym = (mat - mat.transpose(1, 2)).abs().masked_fill(~valid, 0)
        asym_mean = asym.sum(dim=(1, 2), keepdim=True) / denom
        row_mean = vals.sum(dim=2) / valid.float().sum(dim=2).clamp_min(1.0)
        col_mean = vals.sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)
        row_stat = _masked_mean(row_mean.unsqueeze(-1), mask)
        col_stat = _masked_mean(col_mean.unsqueeze(-1), mask)
        n_norm = mask.float().sum(dim=1, keepdim=True).log1p() / 6.0
        zeros = torch.zeros(mat.size(0), 6, device=mat.device, dtype=mat.dtype)
        return torch.cat([mean.flatten(1), std.flatten(1), asym_mean.flatten(1), row_stat, col_stat, n_norm, zeros], dim=-1)

    def _coord_bias(self, node, mask, special_count):
        xy = node[:, :, :2]
        dist = torch.cdist(xy, xy, p=2)
        valid = mask[:, :, None] & mask[:, None, :]
        denom = (dist * valid.float()).sum(dim=(1, 2), keepdim=True) / valid.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        node_bias = (-dist / denom.clamp_min(1.0e-6)).masked_fill(~valid, 0)
        batch, n = node_bias.size(0), node_bias.size(1)
        total = n + special_count
        bias = torch.zeros(batch, self.params["head_num"], total, total, device=node.device, dtype=node.dtype)
        scale = F.softplus(self.coord_bias_scale)[None, :, None, None]
        bias[:, :, special_count:, special_count:] = node_bias[:, None, :, :] * scale
        return bias

    def _matrix_features_and_bias(self, mat, mask, special_count):
        batch, n = mat.size(0), mat.size(1)
        eye = torch.eye(n, dtype=torch.bool, device=mat.device)[None, :, :]
        valid = mask[:, :, None] & mask[:, None, :]
        off_valid = valid & ~eye

        mat_zero = mat.masked_fill(~off_valid, 0.0)
        mat_inf = mat.masked_fill(~off_valid, float("inf"))
        mat_ninf = mat.masked_fill(~off_valid, float("-inf"))
        row_cnt = off_valid.float().sum(dim=2).clamp_min(1.0)
        col_cnt = off_valid.float().sum(dim=1).clamp_min(1.0)

        row_mean = mat_zero.sum(dim=2) / row_cnt
        col_mean = mat_zero.sum(dim=1) / col_cnt
        row_min = mat_inf.min(dim=2).values
        col_min = mat_inf.min(dim=1).values
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        col_min = torch.where(torch.isfinite(col_min), col_min, torch.zeros_like(col_min))
        row_max = mat_ninf.max(dim=2).values
        col_max = mat_ninf.max(dim=1).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        col_max = torch.where(torch.isfinite(col_max), col_max, torch.zeros_like(col_max))
        asym = (mat - mat.transpose(1, 2)).abs().masked_fill(~off_valid, 0.0).sum(dim=2) / row_cnt
        diag = mat.diagonal(dim1=1, dim2=2)
        feat = torch.stack([row_mean, col_mean, row_min, col_min, row_max, col_max, asym, diag], dim=-1)

        denom = mat_zero.abs().sum(dim=(1, 2), keepdim=True) / off_valid.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        node_bias = (-mat / denom.clamp_min(1.0e-6)).masked_fill(~off_valid, 0)
        total = n + special_count
        bias = torch.zeros(batch, self.params["head_num"], total, total, device=mat.device, dtype=mat.dtype)
        scale = F.softplus(self.matrix_bias_scale)[None, :, None, None]
        bias[:, :, special_count:, special_count:] = node_bias[:, None, :, :] * scale
        return feat, bias

    def forward(self, batch):
        special_count = 3
        if batch["kind"] == "coord":
            node = self._pad_node(batch["node"])
            node_mask = batch["node_mask"]
            stats = self._coord_stats(node, node_mask)
            cond, pid = self.condition_encoder(batch, stats, kind_id=0)
            node_token = self.coord_proj(node)
            bias = self._coord_bias(node, node_mask, special_count)
        else:
            node_mask = batch["node_mask"]
            stats = self._matrix_stats(batch["matrix"], node_mask)
            cond, pid = self.condition_encoder(batch, stats, kind_id=1)
            feat, bias = self._matrix_features_and_bias(batch["matrix"], node_mask, special_count)
            node_token = self.matrix_proj(feat)

        batch_size = node_token.size(0)
        cls = self.cls_token.expand(batch_size, 1, -1)
        cond_tok = cond[:, None, :]
        stats_tok = self.stats_token_proj(stats)[:, None, :]
        token = torch.cat([cls, cond_tok, stats_tok, node_token], dim=1)
        special_mask = torch.ones(batch_size, special_count, dtype=torch.bool, device=node_token.device)
        mask = torch.cat([special_mask, node_mask], dim=1)

        for layer in self.layers:
            token = layer(token, mask=mask, attn_bias=bias)

        node_out = token[:, special_count:, :]
        if self.node_only:
            return node_out, node_mask
        mean = _masked_mean(node_out, node_mask)
        maxv = _masked_max(node_out, node_mask)
        h_cls = self.norm(token[:, 0, :] + self.summary_proj(torch.cat([token[:, 0, :], mean, maxv], dim=-1)))
        return token, mask, h_cls, cond, pid


class SolverMemoryEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.solver_emb = nn.Embedding(M_GLOBAL, d)
        self.problem_emb = nn.Embedding(len(PROBLEMS), d)
        self.arm_emb = nn.Embedding(len(PROBLEMS) * M_GLOBAL, d)
        self.ffn = nn.Sequential(
            nn.LayerNorm(4 * d),
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(2 * d, d),
        )
        self.norm = nn.LayerNorm(d)
        self.solver_feature_weight = float(params.get("solver_feature_weight", 0.3))
        self.register_buffer("solver_features", None)
        self.feature_mlp = None
        spec = params.get("solver_feature_spec")
        if spec is not None:
            self.solver_features = build_solver_features(spec)
            hidden = params.get("solver_feature_hidden", 128)
            self.feature_mlp = nn.Sequential(
                nn.Linear(SOLVER_FEATURE_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, d),
                nn.LayerNorm(d),
            )

    def forward(self, solver_ids, pid, cond):
        batch_size = cond.size(0)
        solver_ids = solver_ids.to(cond.device)
        solver = self.solver_emb(solver_ids)
        if self.feature_mlp is not None and self.solver_feature_weight != 0.0:
            solver = solver + self.solver_feature_weight * self.feature_mlp(self.solver_features[solver_ids])
        solver = solver[None, :, :].expand(batch_size, -1, -1)
        problem = self.problem_emb(pid)[:, None, :].expand_as(solver)
        arm_idx = pid[:, None] * M_GLOBAL + solver_ids[None, :]
        arm = self.arm_emb(arm_idx)
        cond_b = cond[:, None, :].expand_as(solver)
        z = torch.cat([solver, problem, arm, cond_b], dim=-1)
        return self.norm(solver + arm + self.ffn(z))


class ProblemToSolverSelector(nn.Module):
    """R31c: multi-query problem-to-solver attention + solver-set Transformer."""

    def __init__(self, **params):
        super().__init__()
        self.params = params
        d = params["embedding_dim"]
        r = params["query_num"]
        self.instance_encoder = InstanceEncoder(**params)
        self.solver_encoder = SolverMemoryEncoder(**params)
        self.query_seed = nn.Parameter(torch.randn(r, d) * 0.02)
        self.cls_to_queries = nn.Linear(d, r * d)
        self.cond_to_queries = nn.Linear(d, r * d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.query_gate = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, r),
        )
        solver_feat_dim = 4 * d + 2
        self.solver_token_proj = nn.Sequential(
            nn.LayerNorm(solver_feat_dim),
            nn.Linear(solver_feat_dim, d),
            nn.GELU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(d, d),
        )
        self.set_layers = nn.ModuleList([EncoderLayer(**params) for _ in range(params["solver_set_layer_num"])])
        self.utility_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, params["head_hidden_dim"]),
            nn.GELU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["head_hidden_dim"], 1),
        )
        self.gap_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, params["head_hidden_dim"]),
            nn.GELU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["head_hidden_dim"], 1),
        )
        self.use_support_branch = bool(params.get("support_branch", False))
        self.support_generators = int(params.get("support_generators", 0))
        self.support_score_weight = float(params.get("support_score_weight", 0.0))
        self.support_score_mode = str(params.get("support_score_mode", "g0"))
        self.support_g0_as_main = bool(params.get("support_g0_as_main", False))
        self.support_feature_to_token = bool(params.get("support_feature_to_token", True))
        self.support_main_utility_weight = float(params.get("support_main_utility_weight", 0.0))
        self.support_main_pre_weight = float(params.get("support_main_pre_weight", 0.0))
        self.support_main_gap_weight = float(params.get("support_main_gap_weight", 0.0))
        if self.use_support_branch:
            self.support_generators = max(1, self.support_generators)
            support_hidden = int(params.get("support_hidden", 128))
            self.support_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, support_hidden),
                        nn.GELU(),
                        nn.Dropout(params["dropout"]),
                        nn.Linear(support_hidden, 1),
                    )
                    for _ in range(self.support_generators)
                ]
            )
            if self.support_generators > 1:
                self.support_generator_solver_bias = nn.Parameter(torch.zeros(self.support_generators, M_GLOBAL))
            support_feat_dim = self.support_generators + 2
            self.support_feature_proj = nn.Sequential(
                nn.LayerNorm(support_feat_dim),
                nn.Linear(support_feat_dim, d),
                nn.GELU(),
                nn.Dropout(params["dropout"]),
                nn.Linear(d, d),
            )
            self.support_token_norm = nn.LayerNorm(d)

    def forward(self, batch):
        assert "problem_desc" in batch, "batch missing problem_desc; update data/collate first"
        assert batch["costs"].shape[1] == batch["pool_ids"].numel(), (
            f"cost/pool mismatch: costs={tuple(batch['costs'].shape)} pool={tuple(batch['pool_ids'].shape)}"
        )
        _, _, h_cls, cond, pid = self.instance_encoder(batch)
        solver_mask = torch.ones(
            h_cls.size(0), batch["pool_ids"].numel(), dtype=torch.bool, device=h_cls.device
        )
        memory = self.solver_encoder(batch["pool_ids"], pid, cond)
        key = self.k_proj(memory)
        value = self.v_proj(memory)

        query = self.query_seed[None, :, :].expand(h_cls.size(0), -1, -1)
        query = query + self.cls_to_queries(h_cls).view(h_cls.size(0), self.params["query_num"], -1)
        query = query + self.cond_to_queries(cond).view(h_cls.size(0), self.params["query_num"], -1)
        query = self.q_proj(query)

        query_logits = torch.einsum("brd,bsd->brs", query, key) / math.sqrt(self.params["embedding_dim"])
        query_logits = query_logits.masked_fill(~solver_mask[:, None, :], _mask_value(query_logits))
        query_gate = F.softmax(self.query_gate(torch.cat([h_cls, cond], dim=-1)), dim=-1)
        pre_score = (query_gate[:, :, None] * query_logits).sum(dim=1)
        pre_score = pre_score.masked_fill(~solver_mask, _mask_value(pre_score))
        pre_prob = F.softmax(pre_score, dim=-1)

        context = torch.einsum("bs,bsd->bd", pre_prob, value)
        feat = torch.cat(
            [
                memory,
                h_cls[:, None, :].expand_as(memory),
                cond[:, None, :].expand_as(memory),
                context[:, None, :].expand_as(memory),
                pre_score.unsqueeze(-1),
                pre_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        solver_token = self.solver_token_proj(feat)
        for layer in self.set_layers:
            solver_token = layer(solver_token, mask=solver_mask)

        support_logits = None
        support_prob = None
        support_score = None
        if self.use_support_branch:
            support_logits = torch.stack([head(solver_token).squeeze(-1) for head in self.support_heads], dim=1)
            if self.support_generators > 1:
                bias = self.support_generator_solver_bias[:, batch["pool_ids"].to(solver_token.device)]
                support_logits = support_logits + bias[None, :, :]
            support_logits = support_logits.masked_fill(~solver_mask[:, None, :], _mask_value(support_logits))
            support_prob = torch.sigmoid(support_logits)
            support_max = support_prob.max(dim=1).values
            if self.support_feature_to_token:
                support_mean = support_prob.mean(dim=1)
                support_feat = torch.cat(
                    [
                        support_max.unsqueeze(-1),
                        support_mean.unsqueeze(-1),
                        support_prob.transpose(1, 2),
                    ],
                    dim=-1,
                )
                solver_token = self.support_token_norm(solver_token + self.support_feature_proj(support_feat))
            if self.support_score_mode == "max":
                support_score = torch.logit(support_max.clamp(1.0e-4, 1.0 - 1.0e-4))
            else:
                support_score = support_logits[:, 0]

        utility = self.utility_head(solver_token).squeeze(-1)
        pred_gap = self.gap_head(solver_token).squeeze(-1)
        base_logits = (
            utility
            + self.params["pre_score_weight"] * pre_score
            - self.params["gap_score_weight"] * pred_gap
        )
        if support_score is not None and self.support_g0_as_main:
            logits = (
                support_score
                + self.support_main_utility_weight * utility
                + self.support_main_pre_weight * pre_score
                - self.support_main_gap_weight * pred_gap
            )
        else:
            logits = base_logits
            if support_score is not None and self.support_score_weight != 0.0:
                logits = logits + self.support_score_weight * support_score
        logits = logits.masked_fill(~solver_mask, _mask_value(logits))
        return {
            "logits": logits,
            "base_logits": base_logits.masked_fill(~solver_mask, _mask_value(base_logits)),
            "utility": utility.masked_fill(~solver_mask, _mask_value(utility)),
            "pre_score": pre_score,
            "pre_prob": pre_prob,
            "pred_gap": pred_gap,
            "query_logits": query_logits,
            "query_gate": query_gate,
            "support_logits": support_logits,
            "support_score": support_score,
        }


def make_selector(params):
    if params.get("architecture", "legacy") == "dual_stream":
        from .dual_stream import DualStreamSelector
        return DualStreamSelector(**params)
    return ProblemToSolverSelector(**params)


def score_components(out):
    scores = {"final": out["logits"]}
    for name, key in (("base", "base_logits"), ("pre", "pre_score"), ("utility", "utility")):
        if key in out:
            scores[name] = out[key]
    if "pred_gap" in out:
        scores["gap"] = -out["pred_gap"]
    if out.get("support_logits") is not None:
        support = out["support_logits"]
        scores["support_g0"] = support[:, 0] if support.dim() == 3 else support
    return scores


def get_default_model_params():
    return dict(
        embedding_dim=128,
        head_num=4,
        qkv_dim=32,
        ff_hidden_dim=512,
        head_hidden_dim=256,
        encoder_layer_num=4,
        solver_set_layer_num=2,
        query_num=4,
        dropout=0.1,
        stats_dim=12,
        rezero=False,
        gap_score_weight=0.25,
        pre_score_weight=0.25,
        support_branch=False,
        support_generators=0,
        support_hidden=128,
        support_score_weight=0.0,
        support_score_mode="g0",
        support_g0_as_main=False,
        support_feature_to_token=True,
        support_main_utility_weight=0.0,
        support_main_pre_weight=0.0,
        support_main_gap_weight=0.0,
        solver_feature_spec=get_solver_feature_spec(),
        solver_feature_weight=0.3,
        solver_feature_hidden=128,
        sdpa=False,
    )
