import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Optional, Dict

from EasyNCO.neural_solvers.methods.mtreld.mtreld_decoder import MTReLDDecoder
from EasyNCO.neural_solvers.methods.mvmoe.mvmoe_encoder import MTMoEEncoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import _get_encoding


class MTReLDPolicy(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        ff_hidden_dim: int,
        encoder_layer_num: int,
        num_heads: int = 8,
        qkv_dim: int = 16,
        logit_clipping: float = 10.0,
        normalization: str = "batch",
        norm_loc: str = "norm_last",
        MoE_param: Optional[Dict] = None,
    ):
        super().__init__()

        MoE_param = MoE_param or {}

        self.aux_loss = 0
        self.light = bool(MoE_param.get("light", False))
        if self.light:
            self.T = 1.0

        self.sqrt_embedding_dim = embedding_dim ** 0.5
        self.encoder = MTMoEEncoder(
            embedding_dim,
            ff_hidden_dim,
            encoder_layer_num,
            num_heads,
            qkv_dim,
            normalization,
            norm_loc,
            MoE_param,
        )
        self.decoder = MTReLDDecoder(
            embedding_dim,
            ff_hidden_dim,
            num_heads,
            qkv_dim,
            self.sqrt_embedding_dim,
            logit_clipping,
            MoE_param,
        )
        self.encoded_nodes = None
        self.decoder_strategy = None

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

        self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = moe_loss
        self.decoder.set_kv(self.encoded_nodes)

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def forward(self, state: TensorDict, first_mode: str = "random") -> TensorDict:
        batch_size = state.batch_size[0]
        pomo_size = state.batch_size[1]

        if (state["next"]["selected_count"] == 0).all():
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
        elif (state["next"]["selected_count"] == 1).all():
            selected = state["start_node"]
            prob = torch.ones(size=(batch_size, pomo_size))
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

            if self.light:
                probs, moe_loss = self.decoder(
                    encoded_last_node,
                    attr,
                    ninf_mask=state["next"]["ninf_mask"],
                    T=self.T,
                    step=state["next"]["selected_count"][0, 0],
                )
            else:
                probs, moe_loss = self.decoder(
                    encoded_last_node,
                    attr,
                    ninf_mask=state["next"]["ninf_mask"],
                )
            self.aux_loss += moe_loss

            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        state.set("action", selected)
        state.set("prob", prob)
        return state
