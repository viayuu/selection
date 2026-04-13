import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Optional

from EasyNCO.neural_solvers.methods.mtreld_mtl.mtl_model import MTL_Encoder, MTL_Decoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import _get_encoding


class MTReLDMTLPolicy(nn.Module):
    """
    ReLD-MTL (multi-task MTLModel) port for EasyNCO.

    Reference: `final/compare/reld/models/MTLModel.py` (MTL encoder/decoder + optional `ffidt`).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        ff_hidden_dim: int = 512,
        encoder_layer_num: int = 6,
        num_heads: int = 8,
        qkv_dim: int = 16,
        logit_clipping: float = 10.0,
        normalization: str = "none",
        norm_loc: str = "norm_last",
        ffidt: bool = True,
    ):
        super().__init__()

        self.aux_loss = 0
        self.decoder_strategy: Optional[str] = None
        self.sqrt_embedding_dim = embedding_dim ** 0.5

        model_params = {
            "embedding_dim": embedding_dim,
            "ff_hidden_dim": ff_hidden_dim,
            "encoder_layer_num": encoder_layer_num,
            "head_num": num_heads,
            "qkv_dim": qkv_dim,
            "sqrt_embedding_dim": self.sqrt_embedding_dim,
            "logit_clipping": logit_clipping,
            "norm": normalization,
            "norm_loc": norm_loc,
            "ffidt": ffidt,
        }

        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        self.encoded_nodes = None

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, reset_state: TensorDict):
        depot_node_xy = reset_state["depot_node_xy"]
        depot_node_demand = reset_state["depot_node_demand"]
        depot_node_tw_start = reset_state["depot_node_tw_start"]
        depot_node_tw_end = reset_state["depot_node_tw_end"]

        node_xy_demand_tw = torch.cat(
            (
                depot_node_xy[:, 1:, :],
                depot_node_demand[:, 1:, :],
                depot_node_tw_start[:, 1:, None],
                depot_node_tw_end[:, 1:, None],
            ),
            dim=2,
        )
        depot_xy = depot_node_xy[:, [0], :]

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state: TensorDict, first_mode: str = "random") -> TensorDict:
        batch_size = state.batch_size[0]
        pomo_size = state.batch_size[1]
        device = state["next"]["selected_count"].device

        if (state["next"]["selected_count"] == 0).all():
            selected = torch.zeros(
                size=(batch_size, pomo_size), dtype=torch.long, device=device
            )
            prob = torch.ones(size=(batch_size, pomo_size), device=device)
        elif (state["next"]["selected_count"] == 1).all():
            selected = state["start_node"]
            prob = torch.ones(size=(batch_size, pomo_size), device=device)
        else:
            encoded_last_node = _get_encoding(
                self.encoded_nodes, state["current_node"].unsqueeze(-1)
            ).squeeze(2)
            attr = torch.cat(
                (
                    state["load"][:, :, None],
                    state["current_time"][:, :, None],
                    state["length"][:, :, None],
                    state["open"][:, :, None],
                ),
                dim=2,
            )

            probs = self.decoder(
                encoded_last_node, attr, ninf_mask=state["next"]["ninf_mask"]
            )
            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        state.set("action", selected)
        state.set("prob", prob)
        return state

