import torch.nn as nn
from EasyNCO.neural_solvers.backbones.Transformer.attr_component import (
    Normalization,
    FeedForward,
)
import torch
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class TSPICAMEncoder(nn.Module):
    def __init__(self, embedding_dim=128, ff_hidden_dim=512, encoder_layer_num=12, **kwargs):
        super().__init__()

        self.udc_flag = kwargs.get("udc_flag", False)
        self.embedding = nn.Linear(2, embedding_dim)
        if self.udc_flag:
            self.embedding_out = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim=embedding_dim, ff_hidden_dim=ff_hidden_dim, **kwargs)
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(self, data, dist, log_scale):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)
        if self.udc_flag:
            embedded_input[:, -1, :] = self.embedding_out(embedded_input[:, -1, :].clone())
            embedded_input[:, 0, :] = self.embedding_out(embedded_input[:, 0, :].clone())

        out = embedded_input
        negative_scale_dist = -1 * log_scale * dist
        for layer in self.layers:
            out = layer(out, negative_scale_dist)

        return out

class CVRPICAMEncoder(nn.Module):

    def __init__(self, embedding_dim=128, ff_hidden_dim=512, encoder_layer_num=6, **kwargs):
        super().__init__()
        self.udc_flag = kwargs.get("udc_flag", False)
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        if self.udc_flag:
            self.embedding_out = nn.Linear(embedding_dim, embedding_dim)
            self.embedding_in = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim=embedding_dim, ff_hidden_dim=ff_hidden_dim)
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(self, depot_xy, node_xy_demand, dist, log_scale, flag=None):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        if self.udc_flag:
            id = ((1 - flag) * embedded_node.size(1))[:, None, None].expand(
                -1, -1, embedded_node.size(-1)
            )
            # shape: (batch, problem, embedding)
            embedded_node[:, 0, :] = self.embedding_in(embedded_node[:, 0, :].clone())

            out = torch.cat((embedded_depot, embedded_node), dim=1)
            last = self.embedding_out(out.gather(1, id).clone())
            out = out.scatter(1, id, last)
        else:
            out = torch.cat((embedded_depot, embedded_node), dim=1)
            # shape: (batch, problem+1, embedding)
        # shape: (batch, problem+1, embedding)
        negative_dist = -1 * dist * log_scale  # -1 * dist represents negative distance

        for layer in self.layers:
            out = layer(out, negative_dist)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        ff_hidden_dim=512,
        **kwargs  # Additional parameters for flexibility
    ):
        super().__init__()
        self.udc_flag = kwargs.get("udc_flag", False)
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.addAndNormalization1 = Normalization(
            embed_dim=embedding_dim, normalization="instance"
        )
        self.feedForward = FeedForward(
            embedding_dim=embedding_dim, ff_hidden_dim=ff_hidden_dim
        )
        self.addAndNormalization2 = Normalization(
            embed_dim=embedding_dim, normalization="instance"
        )

        self.AFT_dist_alpha_1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def forward(self, input1, negative_scale_dist):
        # input.shape: (batch, problem, embedding_dim)
        # dist.shape: (batch, problem, problem)
        # scale.shape: (1,)

        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)
        # shape: (batch, problem, embedding_dim)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_dist_bias_scale = self.AFT_dist_alpha_1 * negative_scale_dist
        # shape: (batch, problem, problem)
        AAFM_OUT = adaptation_attention_free_module(q, k, v, alpha_dist_bias_scale, udc_flag=self.udc_flag)
        # shape: (batch, problem, embedding)

        out1 = self.addAndNormalization1(input1 + AAFM_OUT)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1 + out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


def adaptation_attention_free_module(q, k, v, adaptation_bias, ninf_mask=None, udc_flag=False):
    """
    The core code of Adaptation Attention Free Module.

    Inspired by the paper: An Attention Free Transformer
    (url:  https://arxiv.org/pdf/2105.14103.pdf)

    Args:
        q: query, shape: (batch, n, embedding_dim)
        k: key, shape: (batch, m, embedding_dim)
        v: value, shape: (batch, m, embedding_dim)
        adaptation_bias: - alpha * log_scale * dist, shape: (batch, n, m)
        ninf_mask: shape: (batch, n, m)

    Return:
        out: shape: (batch, n, embedding_dim)

    Note:
    To prevent potential value overflows caused by exponential operations, we use "torch.nan_to_num" to solve it.
    For more details, please refer to the official document:
    https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html
    """

    sigmoid_q = torch.sigmoid(q)
    # shape: (batch, n, embedding_dim)

    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask

    bias = torch.exp(adaptation_bias) @ torch.mul(torch.exp(k), v)
    # shape: (batch, n, embedding_dim)
    if udc_flag:
        weighted = torch.nan_to_num_(bias) / (
            torch.nan_to_num_(torch.exp(adaptation_bias) @ torch.exp(k)) + 1e-20
        )
    else:
        a_k = torch.exp(adaptation_bias) @ torch.exp(k)

        weighted = bias / a_k
        if torch.isinf(bias).any() or torch.isinf(a_k).any():
            weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(a_k)
        if torch.isnan(weighted).any():
            torch.nan_to_num_(weighted)
        # shape: (batch, n, embedding_dim)

    out = torch.mul(sigmoid_q, weighted)
    # shape: (batch, n, embedding_dim)

    return out
