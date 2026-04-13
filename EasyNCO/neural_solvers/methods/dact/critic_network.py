import torch.nn as nn

from EasyNCO.neural_solvers.utils import GraphMeanEmbedding
from EasyNCO.neural_solvers.backbones import TransformerNet
from EasyNCO.neural_solvers.methods import DACTCritic_Encoder,DACTCritic_Decoder



class Critic_network(nn.Module):
    def __init__(
            self,
            env_name: str = "tsp",  # tsp or cvrp
            embed_dim: int = 128,
            num_heads: int = 8,
            qkv_dim: int = 128,
            num_encoder_layers: int = 3,
            normalization: str = "batch",  # norm can be None, batch, or instance
            feedforward_hidden: int = 512,
    ):
        super(Critic_network, self).__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.normalization = normalization

        self.encoder = DACTCritic_Encoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            num_encoder_layers=num_encoder_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        self.value_head = DACTCritic_Decoder(
            input_dim=self.embed_dim * 2,
            embed_dim=self.embed_dim * 2,
        )

    def forward(self,input):
        h_em = self.encoder(input)

        baseline_value = self.value_head(h_em)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()


