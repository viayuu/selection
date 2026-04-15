from typing import Tuple, Optional
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch
import numpy as np


from EasyNCO.neural_solvers.utils import GraphMeanEmbedding
from EasyNCO.neural_solvers.backbones import TransformerNet, positional_encoding_init
from EasyNCO.neural_solvers.methods.lih.policy import tsp_embedding,cvrp_embedding


class Critic_network(nn.Module):
    def __init__(
        self,
        env_name: str = "tsp",  # tsp or cvrp
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 128,
        num_encoder_layers: int = 3,
        normalization: str = "batch", # norm can be None, batch, or instance
        feedforward_hidden: int = 512,
        net: nn.Module = None,
    ):
        super(Critic_network, self).__init__()

        self.env_name = env_name
        self.max_timescale = 10000.0
        self.min_timescale = 1.0
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.num_encoder_layers = num_encoder_layers

        # cvrp: x_t-1 , x_t , x_t+1 , demand 组成一个点，共7维
        if self.env_name == 'tsp':
            node_dim = 2
        elif self.env_name == 'cvrp':
            node_dim = 7

        self.init_embed = nn.Linear(node_dim, embed_dim)

        self.net = (TransformerNet(
                    num_encoder_layers,
                    num_heads,
                    qkv_dim,
                    embed_dim,
                    normalization,
                    feedforward_hidden,
                ) if net is None
                else net
        )
        self.GraphMeanEmbedding = GraphMeanEmbedding(embed_dim)

        self.project_node = nn.Linear(embed_dim, embed_dim, bias=False)

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, feedforward_hidden),
            nn.ReLU(),
            nn.Linear(feedforward_hidden, 1)
        )

    def forward(self, td: TensorDict, mask: Optional[Tensor] = None,**kwargs):
        already_embed = kwargs.get("already_embed")
        if already_embed:
            input_info, pos_enc = td['input_info'] , td['pos_enc']
        else:
            input_info, pos_enc = self.get_input_and_pe(td)

        out = self.net(self.init_embed(input_info) + pos_enc)

        fixed_context = self.GraphMeanEmbedding(out)

        node_feature = self.project_node(out)

        fusion = node_feature + fixed_context.expand_as(node_feature)
        #   返回 reward (batch,1)
        return self.value_head(fusion.mean(dim=1))


    def get_input_and_pe(self,td):
        if self.env_name == "tsp":
            input_info, pos_enc = tsp_embedding(td['locs'], td['solution'])
        elif self.env_name == "cvrp":
            input_info, pos_enc = cvrp_embedding(td['locs'], td['solution'])

        return input_info, pos_enc



