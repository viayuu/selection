import torch
import torch.nn as nn

from EasyNCO.neural_solvers.backbones import TransformerNet
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class LEHDEncoder(nn.Module):
    def __init__(
        self,
        problem_type: str = 'tsp',
        node_dim: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        feedforward_hidden: int = 512,
        normalization: str = None,
        bias: bool = False
    ):
        super().__init__()
        if problem_type == 'tsp':
            self.embedding = nn.Linear(node_dim, embed_dim, bias=True, dtype=torch.float32)
        elif problem_type == 'cvrp':
            self.embedding = nn.Linear(node_dim + 1, embed_dim, bias=True, dtype=torch.float32)
        self.MHA_embed = TransformerNet(1, num_heads, qkv_dim, embed_dim, normalization, feedforward_hidden, bias)

    def forward(self, data):
        embed_data = self.MHA_embed(self.embedding(data))
        return embed_data