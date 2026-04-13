from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Naive_Encoder(nn.Module):
    """
    A local copy of the paper selector encoder (non-hierarchical).

    This is intentionally placed under `my/` so the RL prototype does not import
    the repo-root `model.py` directly.
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        node_dim = 2

        if model_params["problem_type"] == "TSP":
            self.embedding = nn.Linear(node_dim, embedding_dim)
        elif model_params["problem_type"] == "CVRP":
            self.embedding = nn.Linear(node_dim + 1, embedding_dim)
        else:
            raise ValueError(f"Unsupported problem_type: {model_params['problem_type']}")

        encoder_layer_num = self.model_params["encoder_layer_num"] * self.model_params["block_num"]
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, points, mask):
        # points shape: (batch, n, node_dim)
        # mask shape:   (batch, n) with 0 for valid and -inf for padded
        embs = self.embedding(points)

        for layer in self.layers:
            embs = layer(embs, mask=mask)

        emb_mask = torch.where(mask == float("-inf"), 0, 1)
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        graph_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]
        return graph_emb


class Encoder_h(nn.Module):
    """
    A local copy of the paper selector encoder (hierarchical pooling).
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        node_dim = 2

        if model_params["problem_type"] == "TSP":
            self.embedding = nn.Linear(node_dim, embedding_dim)
        elif model_params["problem_type"] == "CVRP":
            self.embedding = nn.Linear(node_dim + 1, embedding_dim)
        else:
            raise ValueError(f"Unsupported problem_type: {model_params['problem_type']}")

        self.blocks = nn.ModuleList([Encoder_block_h(**model_params) for _ in range(model_params["block_num"])])
        self.nonlinear = nn.GELU()

    def masked_mean(self, embs, mask):
        emb_mask = torch.where(mask == float("-inf"), 0, 1)
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]
        return mean_emb

    def masked_max(self, embs, mask):
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]
        return max_emb

    def forward(self, data, mask):
        out = self.embedding(data)

        i = 0
        for block in self.blocks:
            _graph_emb, out, mask = block(out, mask, i)
            i += 1

        mean_emb = self.masked_mean(out, mask)
        max_emb = self.masked_max(out, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        return graph_emb


class Encoder_block_h(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        encoder_layer_num = self.model_params["encoder_layer_num"]
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.layer_score = EncoderLayer(**model_params)
        self.p = nn.Linear(model_params["embedding_dim"], 1)
        self.modulate = nn.Linear(1, model_params["embedding_dim"])
        self.act = nn.Tanh()
        self.nonlinear = nn.GELU()

    def padding_concate(self, tensors: list[torch.Tensor]):
        device = tensors[0].device
        dtype = tensors[0].dtype

        lengths = [int(t.shape[0]) for t in tensors]
        max_len = max(lengths) if lengths else 0

        mask = torch.zeros((len(tensors), max_len), device=device, dtype=dtype)
        padded_list = []
        for i, t in enumerate(tensors):
            pad_len = max_len - lengths[i]
            if pad_len > 0:
                t = F.pad(t, (0, 0, 0, pad_len))
                mask[i, lengths[i] : max_len] = float("-inf")
            padded_list.append(t[None, :, :])

        embs = torch.cat(padded_list, dim=0) if padded_list else tensors[0].new_zeros((0, 0, 0))
        return embs, mask

    def masked_mean(self, embs, mask):
        emb_mask = torch.where(mask == float("-inf"), 0, 1)
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]
        return mean_emb

    def masked_max(self, embs, mask):
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]
        return max_emb

    def forward(self, embs, mask, _i):
        for layer in self.layers:
            embs = layer(embs, mask)

        mean_emb = self.masked_mean(embs, mask)
        max_emb = self.masked_max(embs, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))

        # Downsampling
        score_embs = self.layer_score(embs, mask)
        scores = self.act(self.p(score_embs)).squeeze(-1)
        scores = scores + mask

        selected_embs_list: list[torch.Tensor] = []
        ratio = float(self.model_params["downsample_ratio"])
        for b in range(embs.shape[0]):
            num_valid = int((mask[b] == 0).sum().item())
            k = max(1, int(num_valid * ratio))
            score, ind = scores[b].topk(k, dim=-1, largest=True)
            selected_emb = embs[b].take_along_dim(ind[:, None].expand(-1, embs.shape[-1]), dim=0)
            selected_emb = selected_emb + score[:, None]
            selected_embs_list.append(selected_emb)

        selected_embs, selected_mask = self.padding_concate(selected_embs_list)
        return graph_emb, selected_embs, selected_mask


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, mask=None, edges=None, kv=None):
        head_num = self.model_params["head_num"]
        if kv is None:
            kv = input1
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(kv), head_num=head_num)
        v = reshape_by_heads(self.Wv(kv), head_num=head_num)

        out_concat = multi_head_attention(q, k, v, rank2_ninf_mask=mask)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(-1)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / math.sqrt(float(key_dim))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    assert not score_scaled.isinf().all(dim=-1).any(), "All the valid nodes are filtered! Check the pooling operation."

    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1, input2):
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2
            transposed = added.transpose(1, 2)
            normalized = self.norm(transposed)
            back_trans = normalized.transpose(1, 2)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2
        else:
            back_trans = input1 + input2
        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))

