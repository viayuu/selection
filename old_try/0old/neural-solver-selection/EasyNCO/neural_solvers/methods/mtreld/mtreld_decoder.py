import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict

from EasyNCO.neural_solvers.backbones.Transformer.attr_component import (
    FeedForward,
    multi_head_attention,
    reshape_by_heads,
)
from EasyNCO.neural_solvers.methods.mvmoe.moe_layer import MoE


class MTReLDDecoder(nn.Module):
    """
    Multi-task ReLD decoder (ported from compare ReLD changes on top of mvmoe-style decoder):
    - Optional `ffidt`: apply IDT + FF after attention combine.
    - When `ffidt=True`, disable decoder MoE combine / hierarchical gating and force dense combine.
    """

    def __init__(
        self,
        embedding_dim: int,
        ff_hidden_dim: int,
        head_num: int,
        qkv_dim: int,
        sqrt_embedding_dim: float,
        logit_clipping: float,
        MoE_param: Optional[Dict] = None,
    ):
        super().__init__()
        MoE_param = MoE_param or {}

        self.sqrt_embedding_dim = sqrt_embedding_dim
        self.logit_clipping = logit_clipping
        self.head_num = head_num

        self.light = bool(MoE_param.get("light", False))
        self.ffidt = bool(MoE_param.get("ffidt", False))

        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.hierarchical_gating = False
        self.multi_head_combine_dense = nn.Linear(head_num * qkv_dim, embedding_dim)

        use_decoder_moe = (
            (MoE_param.get("num_experts", 1) > 1)
            and ("Dec" in MoE_param.get("expert_loc", []))
            and (not self.ffidt)
        )
        if use_decoder_moe:
            self.multi_head_combine_moe = MoE(
                input_size=head_num * qkv_dim,
                output_size=embedding_dim,
                num_experts=MoE_param["num_experts"],
                k=MoE_param["topk"],
                T=1.0,
                noisy_gating=True,
                routing_level=MoE_param["routing_level"],
                routing_method=MoE_param["routing_method"],
                moe_model="Linear",
            )
            if self.light:
                self.hierarchical_gating = True
                self.dense_or_moe = nn.Linear(head_num * qkv_dim, 2, bias=False)

        if self.ffidt:
            self.attr_mapping = nn.Linear(4, embedding_dim, bias=False)
            self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes: torch.Tensor):
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(
        self,
        encoded_last_node: torch.Tensor,
        attr: torch.Tensor,
        ninf_mask: torch.Tensor,
        T: float = 1.0,
        step: int = 2,
    ):
        moe_loss = 0

        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=self.head_num)
        out_concat = multi_head_attention(q_last, self.k, self.v, mask=ninf_mask)

        if self.hierarchical_gating and not self.ffidt:
            if step == 2:
                self.probs = F.softmax(
                    self.dense_or_moe(out_concat.mean(0).mean(0).unsqueeze(0)) / T,
                    dim=-1,
                )
            selected = self.probs.multinomial(1).squeeze(0)
            if selected.item() == 1:
                mh_atten_out, moe_loss = self.multi_head_combine_moe(out_concat)
            else:
                mh_atten_out = self.multi_head_combine_dense(out_concat)
            mh_atten_out = mh_atten_out * self.probs.squeeze(0)[selected]
        else:
            mh_atten_out = self.multi_head_combine_dense(out_concat)

        if self.ffidt:
            mh_atten_out = (
                mh_atten_out
                + encoded_last_node
                + self.attr_mapping(attr)
            )
            mh_atten_out = self.feed_forward(mh_atten_out) + mh_atten_out

        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / self.sqrt_embedding_dim
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)

        return probs, moe_loss
