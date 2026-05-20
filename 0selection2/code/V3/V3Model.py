import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.unified_selector.registry import D_COORD, K_CBITS, M_GLOBAL, PROBLEMS


def reshape_by_heads(x, head_num):
    batch, token, _ = x.size()
    qkv_dim = x.size(2) // head_num
    return x.reshape(batch, token, head_num, qkv_dim).transpose(1, 2)


class AddAndNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, y):
        return self.norm(x + y)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.out = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, q_input, k_input, v_input, key_mask=None, attn_bias=None):
        q = reshape_by_heads(self.Wq(q_input), self.head_num)
        k = reshape_by_heads(self.Wk(k_input), self.head_num)
        v = reshape_by_heads(self.Wv(v_input), self.head_num)
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
        if attn_bias is not None:
            score = score + attn_bias
        if key_mask is not None:
            score = score.masked_fill(~key_mask[:, None, None, :], -1.0e9)
        weight = F.softmax(score, dim=-1)
        out = torch.matmul(weight, v)
        out = out.transpose(1, 2).reshape(q_input.size(0), q_input.size(1), self.head_num * self.qkv_dim)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.attn = MultiHeadAttention(d, params["head_num"], params["qkv_dim"])
        self.add_norm_1 = AddAndNorm(d)
        self.ffn = FeedForward(d, params["ff_hidden_dim"])
        self.add_norm_2 = AddAndNorm(d)

    def forward(self, x, mask=None, attn_bias=None):
        x = self.add_norm_1(x, self.attn(x, x, x, key_mask=mask, attn_bias=attn_bias))
        return self.add_norm_2(x, self.ffn(x))


class CoordEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.input_proj = nn.Linear(8, d)
        self.layers = nn.ModuleList([EncoderLayer(**params) for _ in range(params["encoder_layer_num"])])
        self.bias_scale = nn.Parameter(torch.ones(params["head_num"]))

    def forward(self, node, node_mask):
        if node.size(-1) < 8:
            pad = torch.zeros(node.size(0), node.size(1), 8 - node.size(-1), device=node.device, dtype=node.dtype)
            node = torch.cat([node, pad], dim=-1)
        node = node[:, :, :8]
        x = self.input_proj(node)
        xy = node[:, :, :2]
        dist = torch.cdist(xy, xy, p=2)
        denom = dist.masked_fill(~node_mask[:, :, None], 0).sum(dim=(1, 2), keepdim=True)
        denom = denom / node_mask.float().sum(dim=1, keepdim=True).pow(2).clamp_min(1.0)[:, None]
        bias = -dist / denom.clamp_min(1.0e-6)
        bias = bias[:, None, :, :] * F.softplus(self.bias_scale)[None, :, None, None]
        for layer in self.layers:
            x = layer(x, mask=node_mask, attn_bias=bias)
        return x, node_mask


class MatrixEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.input_proj = nn.Linear(8, d)
        self.layers = nn.ModuleList([EncoderLayer(**params) for _ in range(params["encoder_layer_num"])])
        self.bias_scale = nn.Parameter(torch.ones(params["head_num"]))

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
        x = self.input_proj(torch.stack([row_mean, col_mean, row_min, col_min, row_max, col_max, asym, diag], dim=-1))

        valid = node_mask[:, :, None] & node_mask[:, None, :]
        denom = mat.masked_fill(~valid, 0).abs().sum(dim=(1, 2), keepdim=True)
        denom = denom / valid.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        bias = -mat / denom.clamp_min(1.0e-6)
        bias = bias.masked_fill(~valid, 0)
        bias = bias[:, None, :, :] * F.softplus(self.bias_scale)[None, :, None, None]
        for layer in self.layers:
            x = layer(x, mask=node_mask, attn_bias=bias)
        return x, node_mask


class InstanceQueryEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.coord_encoder = CoordEncoder(**params)
        self.matrix_encoder = MatrixEncoder(**params)
        self.problem_embedding = nn.Embedding(len(PROBLEMS), d)
        self.constraint_proj = nn.Linear(K_CBITS, d, bias=False)
        self.coord_dist_embedding = nn.Embedding(D_COORD, d)
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, batch):
        if batch["kind"] == "coord":
            token, mask = self.coord_encoder(batch["node"], batch["node_mask"])
        else:
            token, mask = self.matrix_encoder(batch["matrix"], batch["node_mask"])
        batch_size = token.size(0)
        pid = torch.full((batch_size,), batch["problem_id"], dtype=torch.long, device=token.device)
        cond = self.problem_embedding(pid) + self.constraint_proj(batch["cbits"]) + self.coord_dist_embedding(batch["coord_dist"])
        token = token + cond[:, None, :]
        pooled = (token * mask[:, :, None].float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        query = self.query_mlp(torch.cat([pooled, cond], dim=-1))
        return query


class SolverKeyEncoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d = params["embedding_dim"]
        self.embedding = nn.Embedding(M_GLOBAL, d)
        self.key_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, solver_ids):
        solver = self.embedding(solver_ids)
        return self.key_mlp(solver)


class AttentionSelectorV3(nn.Module):
    """Naive V3: instance query attends over solver keys; attention logits are solver scores."""

    def __init__(self, **params):
        super().__init__()
        self.params = params
        d = params["embedding_dim"]
        self.instance_encoder = InstanceQueryEncoder(**params)
        self.solver_encoder = SolverKeyEncoder(**params)
        self.query_proj = nn.Linear(d, d, bias=False)
        self.key_proj = nn.Linear(d, d, bias=False)
        self.logit_scale_raw = nn.Parameter(torch.tensor(float(params.get("init_logit_scale", 1.0))))

    def forward(self, batch):
        query = self.query_proj(self.instance_encoder(batch))
        key = self.key_proj(self.solver_encoder(batch["pool_ids"]))
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        scale = F.softplus(self.logit_scale_raw) + 1.0e-4
        logits = scale * torch.matmul(query[:, None, :], key.t()[None, :, :]).squeeze(1)
        prob = F.softmax(logits, dim=-1)
        return logits, prob


def get_default_model_params():
    return dict(
        embedding_dim=128,
        head_num=4,
        qkv_dim=32,
        ff_hidden_dim=256,
        encoder_layer_num=3,
        init_logit_scale=1.0,
    )

