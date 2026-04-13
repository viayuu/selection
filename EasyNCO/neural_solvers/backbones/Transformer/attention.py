import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from EasyNCO.utils.utils import getLogger
from EasyNCO.neural_solvers.backbones import (
    reshape_by_heads,
    multi_head_attention,
    FeedForward,
    SkipConnection,
    Normalization,
)

logger = getLogger(__name__)


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer with normalization(optional) and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_dim: int = 16,
        bias: bool = False,  # bias for Wq
        bias_k: bool = None, # bias for Wk, if None, use bias for Wk
        bias_v: bool = None, # bias for Wv, if None, use bias for Wv
        bias_combine: bool = True,  # bias for multi_head_combine
        multi_head_combine_used : bool = True,
    ):
        super().__init__()
        self.multi_head_combine_used = multi_head_combine_used

        if bias_k is None:
            bias_k = bias
        if bias_v is None:
            bias_v = bias

        self.Wq = nn.Linear(embed_dim, num_heads * qkv_dim, bias=bias)
        self.Wk = nn.Linear(embed_dim, num_heads * qkv_dim, bias=bias_k)
        self.Wv = nn.Linear(embed_dim, num_heads * qkv_dim, bias=bias_v)
        if multi_head_combine_used:
            self.multi_head_combine = nn.Linear(
                num_heads * qkv_dim, embed_dim, bias=bias_combine
            )

        self.num_heads = num_heads

    def forward(self, q_input, kv_input=None, mask=None, score_dim=None, sharp=False,weights=None):
        # x.shape: (batch, problem, embedding)
        if kv_input is None:
            kv_input = q_input

        if weights is None:
            q = reshape_by_heads(self.Wq(q_input), head_num=self.num_heads)
            k = reshape_by_heads(self.Wk(kv_input), head_num=self.num_heads)
            v = reshape_by_heads(self.Wv(kv_input), head_num=self.num_heads)
            # q,k,v.shape: (batch, head_num, pomo/node_num, key_dim) or (batch, pomo, head_num, node_num, key_dim)

            out = multi_head_attention(q, k, v, mask, score_dim=score_dim, sharp=sharp)
            if self.multi_head_combine_used:
                out = self.multi_head_combine(out)
            # out.shape: (batch, pomo, embed_dim) or (batch, pomo, node_num_q, embed_dim)

        else:
            q = reshape_by_heads(F.linear(q_input, weights['Wq_weight'], bias=None),
                                 head_num=self.num_heads)
            k = reshape_by_heads(F.linear(kv_input, weights['Wk_weight'], bias=None),
                                 head_num=self.num_heads)
            v = reshape_by_heads(F.linear(kv_input, weights['Wv_weight'], bias=None),
                                 head_num=self.num_heads)
            out = multi_head_attention(q, k, v, mask, score_dim=score_dim, sharp=sharp)
            out = F.linear(out, weights['Wc_weight'], weights['Wc_bias'])

        return out


class TransformerNet(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization(optional), feed-forward layer. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        bias: whether to use bias in the linear layers
    """

    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        qkv_dim: int = 16,
        embed_dim: int = 128,
        normalization: str = "batch",  # "batch", "instance", None
        feedforward_hidden: int = 512,
        bias: bool = False,  # bias for Wq
        bias_k: bool = None,  # bias for Wk, if None, use bias for Wk
        bias_v: bool = None,  # bias for Wv, if None, use bias for Wv
        bias_combine: bool = True,  # bias for multi_head_combine
    ):
        super().__init__()
        self.num_layers = num_layers
        self.MHA_layers = nn.ModuleList(
            [
                SkipConnection(
                    MultiHeadAttentionLayer(embed_dim, num_heads, qkv_dim,
                                            bias=bias, bias_k=bias_k, bias_v=bias_v,
                                            bias_combine=bias_combine)
                )
                for _ in range(num_layers)
            ]
        )
        self.norm1 = nn.ModuleList(
            [Normalization(embed_dim, normalization) for _ in range(num_layers)]
        )
        self.FF_layers = nn.ModuleList(
            [
                SkipConnection(FeedForward(embed_dim, feedforward_hidden))
                for _ in range(num_layers)
            ]
        )
        self.norm2 = nn.ModuleList(
            [Normalization(embed_dim, normalization) for _ in range(num_layers)]
        )

    def forward(self, input: torch.Tensor, weights=None) -> torch.Tensor:
        """Forward pass of the encoder
        Args:
            input: [batch_size, graph_size, embed_dim] initial embeddings to process
        """
        out = input
        for i in range(self.num_layers):
            if weights is None:
                out = self.MHA_layers[i](out)
                out = self.norm1[i](out)
                out = self.FF_layers[i](out)
                out = self.norm2[i](out)
            else:
                Norm_weights1 = {
                    'weight': weights['encoder.net.norm1.{}.normalizer.weight'.format(i)],
                    'bias': weights['encoder.net.norm1.{}.normalizer.bias'.format(i)],
                }
                Norm_weights2 = {
                    'weight': weights['encoder.net.norm2.{}.normalizer.weight'.format(i)],
                    'bias': weights['encoder.net.norm2.{}.normalizer.bias'.format(i)],
                }
                Mha_weights = {
                    'Wq_weight': weights['encoder.net.MHA_layers.{}.0.Wq.weight'.format(i)],
                    'Wk_weight': weights['encoder.net.MHA_layers.{}.0.Wk.weight'.format(i)],
                    'Wv_weight': weights['encoder.net.MHA_layers.{}.0.Wv.weight'.format(i)],
                    'Wc_weight': weights['encoder.net.MHA_layers.{}.0.multi_head_combine.weight'.format(i)],
                    'Wc_bias': weights['encoder.net.MHA_layers.{}.0.multi_head_combine.bias'.format(i)]
                }
                FF_weights = {
                    'weight1': weights['encoder.net.FF_layers.{}.0.W1.weight'.format(i)],
                    'bias1': weights['encoder.net.FF_layers.{}.0.W1.bias'.format(i)],
                    'weight2': weights['encoder.net.FF_layers.{}.0.W2.weight'.format(i)],
                    'bias2': weights['encoder.net.FF_layers.{}.0.W2.bias'.format(i)]
                }
                out = self.MHA_layers[i](out, weights=Mha_weights)
                out = self.norm1[i](out, weights=Norm_weights1)
                out = self.FF_layers[i](out, weights=FF_weights)
                out = self.norm2[i](out, weights=Norm_weights2)

        return out


class Compatibility(nn.Module):
    """
    The compatibility function used in the RL-based NCO models, such as Attention Model and POMO Model.

    And the compatibility function is used to calculate the compatibility score between the current state and the remaining nodes.
    """

    def __init__(self, embed_dim, n_heads, qkv_dim, key_dim, am_mode=True, **kwargs):
        super().__init__()

        # self.n_heads = n_heads
        self.embed_dim = embed_dim
        # self.val_dim = val_dim
        self.key_dim = key_dim
        self.am_mode = am_mode
        """
        Regarding the q_out and single_key, there are some differences between Attention Model and POMO Model:
        In Attention Model, the linear projection is applied to the q_out and single_key.
        However, the linear projection is not applied to the q_out and single_key in POMO Model.
        """
        self.W_query = None
        self.W_key = None
        self.sub_glop = kwargs.get("sub_glop")
        if am_mode:
            self.W_query = nn.Linear(self.embed_dim, n_heads * qkv_dim, bias=False)
            self.W_key = nn.Linear(self.embed_dim, n_heads * qkv_dim, bias=False)

    def forward(
        self, q, encoded_nodes, mask=None, logit_clipping=10, penalty=None, **kwargs
    ):
        """
        :param q: queries (batch_size, n_query, input_dim) or (batch_size, pomo, n_query, input_dim)
        :param encoded_nodes: data (batch_size, graph_size, input_dim) or (batch_size, pomo, graph_size, input_dim)
        :param logit_clipping: used for control the logit value
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if self.am_mode:
            sing_q = self.W_query(q)
            sing_head_k = self.W_key(encoded_nodes).transpose(-2, -1)
        else:
            sing_q = q
            if self.sub_glop:
                sing_head_k=kwargs.get("sing_head_k").transpose(1, 2)
            else:
                sing_head_k = encoded_nodes.transpose(-1, -2)
        # sing_q.shape: (batch_size, n_query, dim) or (batch, pomo, n_query, dim)
        # sing_head_k.shape:(batch_size, dim, graph_size) or (batch, pomo, input_dim, graph_size)

        score = torch.matmul(sing_q, sing_head_k)
        # shape: (batch, pomo, n_query, graph_size) or (batch, pomo, n_query, graph_size)
        score_scaled = score / math.sqrt(self.key_dim)
        if penalty is not None:
            score_scaled += penalty
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        if mask is not None:
            if q.dim() == 3:
                score_masked = score_clipped + mask
            elif q.dim() == 4:
                score_masked = score_clipped + mask[:, :, None, :].expand(
                    -1, -1, sing_q.size(2), -1
                )
            else:
                raise RuntimeError(
                    f"Invalid dimension of q: {q.dim()}. It should be either 3 or 4."
                )
        else:
            score_masked = score_clipped
        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, n_query, graph_size) or (batch, pomo, n_query, graph_size)
        assert not torch.isnan(
            probs
        ).any(), "probs has nan, but it should not have any nans."

        if q.dim() == 4:
            probs = probs.squeeze(dim=-2)
        return probs
