from typing import Tuple, Optional
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch
import numpy as np
import math

from EasyNCO.neural_solvers.utils import GraphMeanEmbedding
from EasyNCO.neural_solvers.backbones import TransformerNet,Compatibility


class MLP(nn.Module):
    def __init__(self,
                 input_dim=128,
                 feed_forward_dim=64,
                 embedding_dim=64,
                 output_dim=1
                 ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.ReLU = nn.ReLU(inplace=True)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result


class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = torch.matmul(Q, K.transpose(2, 3))

        return self.norm_factor * compatibility


class DACT_Decoder(nn.Module):
    def __init__(
            self,
            num_heads: int = 4,
            input_dim: int = 64,
            embed_dim: int = 64,
            qkv_dim: int = 16,
            val_dim: int = None,
            key_dim: int = None
    ):
        super(DACT_Decoder, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        # for MHC sublayer (NFE aspect)
        self.compater_node = MultiHeadCompat(num_heads,
                                             embed_dim,
                                             embed_dim,
                                             embed_dim,
                                             key_dim
                                             )

        # for MHC sublayer (PFE aspect)
        self.compater_pos = MultiHeadCompat(num_heads,
                                             embed_dim,
                                             embed_dim,
                                             embed_dim,
                                             key_dim
                                             )

        # for Max-Pooling sublayer
        self.project_graph_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_graph_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # for feed-forward aggregation (FFA)sublayer
        self.value_head = MLP(self.num_heads * 2, 32, 32, 1)

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_em, pos_em):

        batch_size, graph_size, dim = h_em.size()

        # Max-Pooling sublayer
        h_node_refined = self.project_node_node(h_em) + self.project_graph_node(h_em.max(1)[0])[:, None, :].expand(
            batch_size, graph_size, dim)
        h_pos_refined = self.project_node_pos(pos_em) + self.project_graph_pos(pos_em.max(1)[0])[:, None, :].expand(
            batch_size, graph_size, dim)

        # MHC sublayer
        compatibility = torch.zeros((batch_size, graph_size, graph_size, self.num_heads * 2),
                                    device=h_node_refined.device)
        compatibility[:, :, :, :self.num_heads] = self.compater_pos(h_pos_refined).permute(1, 2, 3, 0)
        compatibility[:, :, :, self.num_heads:] = self.compater_node(h_node_refined).permute(1, 2, 3, 0)

        # FFA sublater
        return self.value_head(compatibility).squeeze(-1)

class DACTCritic_Decoder(nn.Module):
    def __init__(
            self,
            env_name: str = "tsp_improve",  # tsp or cvrp
            input_dim: int = 128,
            embed_dim: int = 128,
    ):
        super(DACTCritic_Decoder, self).__init__()

        self.hidden_dim = embed_dim
        self.embedding_dim = embed_dim

        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.MLP = MLP(input_dim, embed_dim)

    def forward(self, h_em):
        # mean Pooling
        mean_pooling = h_em.mean(1)
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        fusion = node_feature + graph_feature.expand_as(node_feature)

        # pass through value_head, get estimated values
        value = self.MLP(fusion.mean(1))

        return value