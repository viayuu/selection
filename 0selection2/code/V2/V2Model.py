import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.unified_selector.registry import D_COORD, K_CBITS, M_GLOBAL, PROBLEMS


def reshape_by_heads(qkv, head_num):
    batch, token, _ = qkv.size()
    qkv_dim = qkv.size(2) // head_num
    return qkv.reshape(batch, token, head_num, qkv_dim).transpose(1, 2)


class AddAndNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sub):
        return self.norm(x + sub)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, q_input, k_input, v_input, key_mask=None, attn_bias=None):
        head_num = self.head_num
        q = reshape_by_heads(self.Wq(q_input), head_num)
        k = reshape_by_heads(self.Wk(k_input), head_num)
        v = reshape_by_heads(self.Wv(v_input), head_num)

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        if attn_bias is not None:
            score = score + attn_bias
        if key_mask is not None:
            score = score.masked_fill(~key_mask[:, None, None, :], -1.0e9)
        weights = F.softmax(score, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(q_input.size(0), q_input.size(1), head_num * self.qkv_dim)
        return self.multi_head_combine(out)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            head_num=model_params["head_num"],
            qkv_dim=model_params["qkv_dim"],
        )
        self.add_norm_1 = AddAndNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_norm_2 = AddAndNorm(embedding_dim)

    def forward(self, x, mask=None, attn_bias=None):
        out = self.attention(x, x, x, key_mask=mask, attn_bias=attn_bias)
        x = self.add_norm_1(x, out)
        out = self.feed_forward(x)
        x = self.add_norm_2(x, out)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            head_num=model_params["head_num"],
            qkv_dim=model_params["qkv_dim"],
        )
        self.add_norm_1 = AddAndNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_norm_2 = AddAndNorm(embedding_dim)

    def forward(self, solver_token, inst_token, inst_mask):
        out = self.attention(solver_token, inst_token, inst_token, key_mask=inst_mask)
        solver_token = self.add_norm_1(solver_token, out)
        out = self.feed_forward(solver_token)
        solver_token = self.add_norm_2(solver_token, out)
        return solver_token


class CoordEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        encoder_layer_num = model_params["encoder_layer_num"]
        self.input_proj = nn.Linear(8, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.bias_scale = nn.Parameter(torch.ones(model_params["head_num"]))

    def forward(self, node, node_mask):
        if node.size(-1) < 8:
            pad = torch.zeros(node.size(0), node.size(1), 8 - node.size(-1), device=node.device, dtype=node.dtype)
            node = torch.cat([node, pad], dim=-1)
        node = node[:, :, :8]
        x = self.input_proj(node)
        xy = node[:, :, :2]
        dist = torch.cdist(xy, xy, p=2)
        denom = dist.masked_fill(~node_mask[:, :, None], 0).mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        bias = -dist / denom
        bias = bias[:, None, :, :] * F.softplus(self.bias_scale)[None, :, None, None]
        for layer in self.layers:
            x = layer(x, mask=node_mask, attn_bias=bias)
        return x, node_mask


class MatrixEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        encoder_layer_num = model_params["encoder_layer_num"]
        self.input_proj = nn.Linear(8, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.bias_scale = nn.Parameter(torch.ones(model_params["head_num"]))

    def forward(self, mat, node_mask):
        eye = torch.eye(mat.size(1), dtype=torch.bool, device=mat.device)[None, :, :]
        masked = mat.masked_fill(eye, float("inf"))
        masked_zero = mat.masked_fill(eye, 0)
        row_mean = masked_zero.mean(dim=2)
        col_mean = masked_zero.mean(dim=1)
        row_min = masked.min(dim=2).values
        col_min = masked.min(dim=1).values
        row_max = masked_zero.max(dim=2).values
        col_max = masked_zero.max(dim=1).values
        asym = (mat - mat.transpose(1, 2)).abs().mean(dim=2)
        diag = mat.diagonal(dim1=1, dim2=2)
        feat = torch.stack([row_mean, col_mean, row_min, col_min, row_max, col_max, asym, diag], dim=-1)
        x = self.input_proj(feat)

        valid_pair = node_mask[:, :, None] & node_mask[:, None, :]
        finite_cost = mat.masked_fill(~valid_pair, 0)
        denom = finite_cost.abs().sum(dim=(1, 2), keepdim=True) / valid_pair.float().sum(dim=(1, 2), keepdim=True).clamp_min(1)
        bias = -mat / denom.clamp_min(1.0e-6)
        bias = bias.masked_fill(~valid_pair, 0)
        bias = bias[:, None, :, :] * F.softplus(self.bias_scale)[None, :, None, None]
        for layer in self.layers:
            x = layer(x, mask=node_mask, attn_bias=bias)
        return x, node_mask


class InstanceEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.coord_encoder = CoordEncoder(**model_params)
        self.matrix_encoder = MatrixEncoder(**model_params)
        self.problem_embedding = nn.Embedding(len(PROBLEMS), embedding_dim)
        self.constraint_proj = nn.Linear(K_CBITS, embedding_dim, bias=False)
        self.coord_dist_embedding = nn.Embedding(D_COORD, embedding_dim)
        self.global_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, batch):
        if batch["kind"] == "coord":
            token, mask = self.coord_encoder(batch["node"], batch["node_mask"])
        else:
            token, mask = self.matrix_encoder(batch["matrix"], batch["node_mask"])

        batch_size = token.size(0)
        pid = torch.full((batch_size,), batch["problem_id"], dtype=torch.long, device=token.device)
        cond = (
            self.problem_embedding(pid)
            + self.constraint_proj(batch["cbits"])
            + self.coord_dist_embedding(batch["coord_dist"])
        )
        token = token + cond[:, None, :]
        global_token = self.global_token.expand(batch_size, 1, -1) + cond[:, None, :]
        token = torch.cat([global_token, token], dim=1)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=token.device), mask], dim=1)
        return self.norm(token), mask, cond


class SolverEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.embedding = nn.Embedding(M_GLOBAL, embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, solver_ids, cond):
        # solver_ids.shape: (solver,)
        solver = self.embedding(solver_ids)[None, :, :].expand(cond.size(0), -1, -1)
        solver = solver + cond[:, None, :]
        return self.norm(solver + self.ffn(solver))


class SolverConditionedSelector(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        self.instance_encoder = InstanceEncoder(**model_params)
        self.solver_encoder = SolverEncoder(**model_params)
        self.cross_layers = nn.ModuleList([CrossAttentionLayer(**model_params) for _ in range(model_params["cross_layer_num"])])
        self.set_layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(model_params["solver_set_layer_num"])])
        self.utility_head = nn.Sequential(
            nn.Linear(3 * embedding_dim, model_params["head_hidden_dim"]),
            nn.GELU(),
            nn.Linear(model_params["head_hidden_dim"], 1),
        )
        self.gap_head = nn.Sequential(
            nn.Linear(3 * embedding_dim, model_params["head_hidden_dim"]),
            nn.GELU(),
            nn.Linear(model_params["head_hidden_dim"], 1),
        )

    def forward(self, batch):
        inst_token, inst_mask, cond = self.instance_encoder(batch)
        solver_token = self.solver_encoder(batch["pool_ids"], cond)
        for layer in self.cross_layers:
            solver_token = layer(solver_token, inst_token, inst_mask)
        for layer in self.set_layers:
            solver_token = layer(solver_token)

        global_token = inst_token[:, :1, :].expand(-1, solver_token.size(1), -1)
        z = torch.cat([solver_token, global_token, solver_token * global_token], dim=-1)
        utility = self.utility_head(z).squeeze(-1)
        gap = self.gap_head(z).squeeze(-1)
        score = utility - self.model_params.get("gap_score_weight", 0.0) * gap
        return score, gap


def get_default_model_params():
    return dict(
        embedding_dim=128,
        head_num=4,
        qkv_dim=32,
        ff_hidden_dim=256,
        head_hidden_dim=256,
        encoder_layer_num=3,
        cross_layer_num=2,
        solver_set_layer_num=1,
        gap_score_weight=0.0,
    )
