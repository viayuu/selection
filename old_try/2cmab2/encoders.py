from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 说明
# ============================================================
# 本文件前半部分（Naive_Encoder / Encoder_h / EncoderLayer 等）
# 是直接从 1two_gate/paper_encoder.py 复制过来的 NSS 风格编码器实现。
#
# 这样做的原因是：
# 1. 用户明确要求“复用 NSS encoder 时尽量直接复制，不要自己重写”
# 2. 这部分是参考论文结构的本体，直接复制可以最大限度减少实现偏差
#
# 在复制代码的基础上，本文件后半部分新增了：
# - build_default_encoder_kwargs
# - DualNSSEncoder
# - HybridContextEncoder
#
# 它们负责把原本“单问题的 NSS 编码器”适配到当前
# “TSP + CVRP 共享 bandit 框架”的使用场景中。
# ============================================================


class Naive_Encoder(nn.Module):
    """
    NSS 论文中的非层次化编码器本地副本。

    这里基本保持原实现，不主动改逻辑。
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
    NSS 论文中的层次化编码器本地副本。

    它的核心特点是：
    - 先做多层注意力编码
    - 再做层次化下采样
    - 最后把 mean pooling 和 max pooling 结果拼接起来

    输出维度是 2 * embedding_dim。
    在当前默认配置下，embedding_dim=128，因此输出是 256 维。
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


def build_default_encoder_kwargs(problem_type: str) -> dict:
    """
    为 NSS 编码器准备一组默认超参数。

    这些参数直接对齐 NSS 代码里的默认配置：
    - embedding_dim=128
    - block_num=2
    - encoder_layer_num=2
    - head_num=8
    - qkv_dim=16
    - ff_hidden_dim=512
    - downsample_ratio=0.8
    - norm=rezero
    """
    return {
        "problem_type": problem_type,
        "embedding_dim": 128,
        "encoder_layer_num": 2,
        "block_num": 2,
        "head_num": 8,
        "qkv_dim": 16,
        "ff_hidden_dim": 512,
        "downsample_ratio": 0.8,
        "norm": "rezero",
    }


class DualNSSEncoder(nn.Module):
    """
    共享框架下的双编码器包装：
    - TSP 走 2 维输入 encoder
    - CVRP 走 3 维输入 encoder

    为什么不能只用一个 Encoder_h？
    因为 NSS 原始设计里：
    - TSP 输入维度是 2（坐标）
    - CVRP 输入维度是 3（坐标 + demand）

    两者第一层线性映射就不同，所以最稳妥的做法是保留两个编码器实例。
    """

    def __init__(self) -> None:
        super().__init__()
        self.tsp_encoder = Encoder_h(**build_default_encoder_kwargs("TSP"))
        self.cvrp_encoder = Encoder_h(**build_default_encoder_kwargs("CVRP"))

    def encode_problem(self, batch_nodes: torch.Tensor, problem: str, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        编码同一种 problem 的一个 batch。

        输出是：
        - 256 维图嵌入
        - 再拼一个 scale
        - 最终得到 257 维
        """
        problem = problem.lower()
        batch_nodes = batch_nodes.to(torch.float32)
        if mask is None:
            mask = torch.zeros(batch_nodes.size(0), batch_nodes.size(1), device=batch_nodes.device, dtype=batch_nodes.dtype)
        else:
            mask = mask.to(device=batch_nodes.device, dtype=batch_nodes.dtype)
        if problem == "tsp":
            # TSP: 只取前 2 列坐标。
            # 如果 batch 内图规模不同，mask 会把 padding 节点标成 -inf，
            # 编码器内部的注意力与 pooling 都会自动跳过这些 padding 位置。
            graph_emb = self.tsp_encoder(batch_nodes[:, :, :2], mask)
        elif problem == "cvrp":
            # CVRP: 取 [x, y, demand]，同样支持 padding mask。
            graph_emb = self.cvrp_encoder(batch_nodes[:, :, :3], mask)
        else:
            raise ValueError(f"未知问题类型: {problem}")

        # 按 NSS 的思路，编码器后面额外拼一个规模特征 scale
        scale = (mask == 0).sum(dim=1, keepdim=True).to(dtype=batch_nodes.dtype)
        return torch.cat((graph_emb, scale), dim=1)


class HybridContextEncoder(nn.Module):
    """
    最终上下文特征 = NSS 图编码(257) + 手工特征(11) + 问题 one-hot(2) = 270 维

    这个类就是当前共享模型里的“统一上下文表示器”。
    它把三类信息拼到一起：
    1. 图结构表示：NSS encoder
    2. 可解释的统计特征：manual features
    3. 问题身份：problem one-hot
    """

    output_dim = 270

    def __init__(self) -> None:
        super().__init__()
        self.graph_encoder = DualNSSEncoder()

    def freeze_graph_encoder(self) -> None:
        """冻结底层图编码器，常用于先跑一个更稳的轻量版本。"""
        for param in self.graph_encoder.parameters():
            param.requires_grad = False

    def encode_samples(self, samples: Sequence) -> torch.Tensor:
        """
        把多个样本编码成统一张量。

        返回 shape:
        [batch_size, 270]

        这里要注意：
        - TSP 和 CVRP 不能直接拼成一个原始 batch 送给同一个 encoder
        - 所以会先按 problem 分组编码，再按原顺序拼回去
        """
        if len(samples) == 0:
            return torch.zeros(0, self.output_dim, dtype=torch.float32)

        device = next(self.parameters()).device
        graph_parts: list[torch.Tensor | None] = [None] * len(samples)

        for problem in ("tsp", "cvrp"):
            idxs = [idx for idx, sample in enumerate(samples) if sample.problem == problem]
            if not idxs:
                continue
            # NSS 数据里同一 problem 下的实例规模可能不同，不能直接 stack。
            # 这里先手动 padding，再配一个 -inf mask 给 NSS encoder。
            raw_nodes = [samples[idx].nodes.to(device=device, dtype=torch.float32) for idx in idxs]
            max_length = max(int(nodes.size(0)) for nodes in raw_nodes)
            feature_dim = int(raw_nodes[0].size(1))
            batch_nodes = torch.zeros(len(raw_nodes), max_length, feature_dim, device=device, dtype=torch.float32)
            mask = torch.full((len(raw_nodes), max_length), float("-inf"), device=device, dtype=torch.float32)
            for pos, nodes in enumerate(raw_nodes):
                length = int(nodes.size(0))
                batch_nodes[pos, :length, :] = nodes
                mask[pos, :length] = 0.0
            encoded = self.graph_encoder.encode_problem(batch_nodes, problem=problem, mask=mask)
            for pos, idx in enumerate(idxs):
                graph_parts[idx] = encoded[pos]

        # graph_emb: [B, 257]
        graph_emb = torch.stack([part for part in graph_parts if part is not None], dim=0)
        # manual_raw: [B, 11]
        manual_raw = torch.as_tensor(np.stack([sample.manual_raw for sample in samples], axis=0), dtype=torch.float32, device=device)
        # onehot: [B, 2]
        onehot = torch.as_tensor(np.stack([sample.problem_onehot for sample in samples], axis=0), dtype=torch.float32, device=device)
        return torch.cat((graph_emb, manual_raw, onehot), dim=1)
