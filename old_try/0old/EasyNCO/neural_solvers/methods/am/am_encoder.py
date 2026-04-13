from typing import Tuple, Optional
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from EasyNCO.neural_solvers.utils import initial_embedding
from EasyNCO.neural_solvers.backbones import TransformerNet
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class AttentionModelEncoder(nn.Module):
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
        env_name: str = "tsp",  # tsp or cvrp (currently only tsp and cvrp are supported)
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        qkv_dim: int = 16,
        num_layers: int = 3,
        normalization: str = "batch", # norm can be None, batch, or instance
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        bias: bool = False,  # bias for Wq
        bias_k: bool = None,  # bias for Wk, if None, use bias for Wk
        bias_v: bool = None,  # bias for Wv, if None, use bias for Wv
        bias_combine: bool = True,  # bias for multi_head_combine
    ):
        super(AttentionModelEncoder, self).__init__()

        self.env_name = env_name

        self.initial_embedding = (
            initial_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

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