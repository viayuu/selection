import torch.nn as nn
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods.invit.invit_decoder import INVITDecoder
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class INVITPolicy(nn.Module):
    def __init__(
        self,
        # node_dim: int = 2,
        env_name: str = "tsp",
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        feedforward_dim: int = 512,
        action_embed_layer_num: int = 2,
        state_embed_layer_num: int = 2,
        decoder_layer_num: int = 3,
        normalization: str = "layer",
        first_placeholder: bool = False,
        state_k: list = [35, 50, 65], # directly from the INViT code
        action_k: int = 15,
        logit_clipping: float = 10
    ):
        super().__init__()

        node_dim = 2 if env_name == 'tsp' else 3

        self.decoder = INVITDecoder(
                            node_dim=node_dim,
                            env_name=env_name,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            qkv_dim=qkv_dim,
                            feedforward_dim=feedforward_dim,
                            action_embed_layer_num=action_embed_layer_num,
                            state_embed_layer_num=state_embed_layer_num,
                            state_num=3,
                            decoder_layer_num=decoder_layer_num,
                            normalization=normalization,
                            first_placeholder=first_placeholder,
                            state_k=state_k,
                            action_k=action_k,
                            logit_clipping=logit_clipping
        )
        self.decoder_strategy = None
        self.problems = None

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):
        self.problems = td['locs']

    def forward(self, td: TensorDict, first_mode: str = 'random') -> TensorDict:

        selected, prob = self.decoder(self.problems, self.decoder_strategy, td, first_mode=first_mode)

        td.set("action", selected)
        td.set("prob", prob)

        return td