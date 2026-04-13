import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_size = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_size, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank2_ninf_mask: Optional[torch.Tensor] = None,
    rank3_ninf_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_size, head_num, n, input_s
        )
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_size, head_num, n, input_s
        )

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_size, n, head_num * key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.add = (
            True
            if ("norm_loc" in model_params.keys() and model_params["norm_loc"] == "norm_last")
            else False
        )
        norm = model_params.get("norm", "none")
        if norm == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif norm == "batch_no_track":
            self.norm = nn.BatchNorm1d(
                embedding_dim, affine=True, track_running_stats=False
            )
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif norm == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif norm == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        else:
            self.norm = None

    def forward(
        self, input1: Optional[torch.Tensor] = None, input2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            normalized = self.norm(transposed)
            back_trans = normalized.transpose(1, 2)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(input1)))


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
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        head_num = self.model_params["head_num"]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        if self.model_params["norm_loc"] == "norm_last":
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3


class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        encoder_layer_num = self.model_params["encoder_layer_num"]

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

    def forward(self, depot_xy: torch.Tensor, node_xy_demand_tw: torch.Tensor) -> torch.Tensor:
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand_tw)
        out = torch.cat((embedded_depot, embedded_node), dim=1)

        for layer in self.layers:
            out = layer(out)

        return out


class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params["embedding_dim"]
        head_num = self.model_params["head_num"]
        qkv_dim = self.model_params["qkv_dim"]
        self.ffidt = bool(self.model_params.get("ffidt", False))

        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        if self.ffidt:
            self.attr_mapping = nn.Linear(4, embedding_dim, bias=False)
            self.feed_forward = FeedForward(**model_params)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes: torch.Tensor):
        head_num = self.model_params["head_num"]
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(
        self, encoded_last_node: torch.Tensor, attr: torch.Tensor, ninf_mask: torch.Tensor
    ) -> torch.Tensor:
        head_num = self.model_params["head_num"]

        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)

        mh_atten_out = self.multi_head_combine(out_concat)

        if self.ffidt:
            mh_atten_out = mh_atten_out + encoded_last_node + self.attr_mapping(attr.clone())
            mh_atten_out = self.feed_forward(mh_atten_out) + mh_atten_out

        score = torch.matmul(mh_atten_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params["sqrt_embedding_dim"]
        logit_clipping = self.model_params["logit_clipping"]

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs
