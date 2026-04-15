import torch.nn as nn
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods import DPNEncoder, DPNDecoder
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class DPNPolicy(nn.Module):

    def __init__(
        self,
        env_name: str = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,  # in the original code, num_heads is val_dim = embed_dim // n_heads
        num_encoder_layers: int = 6,
        feedforward_hidden: int = 512,
        tanh_clipping: float = 50.0,  # the clipping in the original code is 50.0, instead of logit_clipping with 10.0
    ):
        super().__init__()
        self.encoder = DPNEncoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
        )
        self.decoder = DPNDecoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            tanh_clipping=tanh_clipping,
        )
        self.decoder_strategy = None
        self.env_name = env_name

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):
        if self.env_name == "mtsp" or self.env_name == "mpdp":
            embeddings, _ = self.encoder(td)
            self.decoder.set_fixed(embeddings)
        elif self.env_name == "fmdvrp" or self.env_name == "mdvrp":
            agent_embeddings, embeddings = self.encoder(td)
            self.decoder.set_fixed(embeddings, agent_embeddings, td)

    def forward(self, td: TensorDict) -> TensorDict:
        selected, prob = self.decoder(td, decode_type=self.decoder_strategy)

        td.set("action", selected)
        td.set("prob", prob)

        return td
