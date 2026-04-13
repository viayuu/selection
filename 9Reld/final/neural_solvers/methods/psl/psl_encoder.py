from typing import Tuple, Optional
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch.nn.functional as F
import torch

from EasyNCO.neural_solvers.utils import initial_embedding
from EasyNCO.neural_solvers.backbones import TransformerNet
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class PSLEncoder(nn.Module):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
    """

    def __init__(
        self,
        env_name: str = "motsp",  # tsp or cvrp (currently only tsp and cvrp are supported)
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        qkv_dim: int = 16,
        num_layers: int = 6,
        normalization: str = "batch", # norm can be None, batch, or instance
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        bias: bool = False,  # bias for Wq
        bias_k: bool = None,  # bias for Wk, if None, use bias for Wk
        bias_v: bool = None,  # bias for Wv, if None, use bias for Wv
        bias_combine: bool = True,  # bias for multi_head_combine
        num_target: int = 2
    ):
        super(PSLEncoder, self).__init__()

        self.env_name = env_name
        if env_name == "motsp":

            self.initial_embedding = InitEmbedding(embed_dim=embed_dim, num_target=num_target)
        elif env_name == "mocvrp":
            self.initial_embedding = CVRPInitEmbedding(embed_dim=embed_dim, node_dim=3)
        elif env_name == "mokp":
            self.initial_embedding = KPInitEmbedding(embed_dim=embed_dim)

        self.net = (TransformerNet(
                    num_layers,
                    num_heads,
                    qkv_dim,
                    embed_dim,
                    normalization,
                    feedforward_hidden,
                    bias=bias,
                    bias_k=bias_k,
                    bias_v=bias_v,
                    bias_combine=bias_combine,
                ) if net is None
                else net
        )

    def forward(self, td: TensorDict, mask: Optional[Tensor] = None, weights = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.
        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention
            weights: use existed weights for meta-learning
        Returns:
            out: Latent representation of the input
            init_out: Initial embedding of the input
        """
        # Transfer to embedding space
        if weights is None:
            init_out = self.initial_embedding(td)

            # Process embedding
            out = self.net(init_out)
        else:
            init_out = self.initial_embedding(td, weights=weights)

            # Process embedding
            out = self.net(init_out, weights=weights)

        # Return latent representation and initial embedding
        return out, init_out

class InitEmbedding(nn.Module):

    def __init__(self, embed_dim, linear_bias=True,num_target=2):
        super(InitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim * num_target, embed_dim, linear_bias)

    def forward(self, td, weights=None):
        if weights is None:
            out = self.init_embed(td["locs"])
        else:
            out = F.linear(td["locs"], weights['encoder.initial_embedding.init_embed.weight'],
                           weights['encoder.initial_embedding.init_embed.bias'])
        # shape: (batch, N, embed_dim)
        return out

class CVRPInitEmbedding(nn.Module):
    """
    Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(CVRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, demand
        self.init_embed_customers = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td, weights=None):

        depot, customers = td["locs"][:, :1, :-1], td["locs"][:, 1:, :]
        if weights is None:
            depot_embedding = self.init_embed_depot(depot)
        else:
            depot_embedding = F.linear(depot, weights['encoder.initial_embedding.init_embed_depot.weight'],
                                       weights['encoder.initial_embedding.init_embed_depot.bias'])
        # shape: (batch, 1, embed_dim)

        #customers_demands = torch.cat((customers, td["demand"][..., None]), -1)
        customers_demands = customers
        # shape: (batch, N, 3)
        if weights is None:
            node_embeddings = self.init_embed_customers(customers_demands)
        else:
            node_embeddings = F.linear(customers_demands,
                                       weights['encoder.initial_embedding.init_embed_customers.weight'],
                                       weights['encoder.initial_embedding.init_embed_customers.bias'])
        # shape: (batch, N, embed_dim)
        out = torch.cat((depot_embedding, node_embeddings), dim=-2)
        # shape: (batch, N+1, embed_dim)

        return out

class KPInitEmbedding(nn.Module):
    """
    Initial embedding for the Knapsack Problem (KP).
    Embed the following node features to the embedding space:
        - items: weight, and values of the items
    """
    def __init__(self, embed_dim, linear_bias=True):
        super(KPInitEmbedding, self).__init__()
        node_dim = 3  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["items"])
        # shape: (batch, N, embed_dim)
        return out