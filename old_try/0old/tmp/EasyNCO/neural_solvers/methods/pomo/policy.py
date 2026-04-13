from typing import Literal
import torch.nn as nn
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods import AttentionModelEncoder, AttentionModelDecoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class POMOPolicy(nn.Module):
    def __init__(self,
                 env_name: str = "tsp",
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 qkv_dim: int = 16,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = None,  # It is None in POMO and POMO_based models
                 normalization: str = "instance",
                 feedforward_hidden: int = 512,
                 logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
                 use_graph_mean: bool = False,  # It is False in POMO and POMO_based models
                 am_mode: bool = True,
                 first_placeholder: bool = True,  # only used in TSP of Attention Model
                 first_mode: Literal["random","placeholder"] = "random",  # only used in TSP of Attention Model
                 ):
        super().__init__()
        self.encoder = AttentionModelEncoder(
                        env_name=env_name,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        num_layers=num_encoder_layers,
                        normalization=normalization,
                        feedforward_hidden=feedforward_hidden,
        )
        self.decoder = AttentionModelDecoder(
                        env_name=env_name,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        logit_clipping=logit_clipping,
                        use_graph_mean=use_graph_mean,
                        am_mode=am_mode,
                        first_placeholder=first_placeholder
        )
        self.decoder_strategy = None
        self.use_graph_mean = use_graph_mean
        self.first_mode = first_mode

    def set_decoder_strategy(self, strategy: str="sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):
        encoded_nodes, _ = self.encoder(td)
        self.decoder.set_kv(encoded_nodes)
        if self.use_graph_mean:
            self.decoder.set_graph_mean(encoded_nodes)

    def forward(self, td: TensorDict) -> TensorDict:

        pomo_selected, probs = self.decoder(td, first_mode=self.first_mode)

        if pomo_selected is not None:
            selected = pomo_selected
            prob = probs
        else:
            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        td.set("action", selected)
        td.set("prob", prob)

        return td

