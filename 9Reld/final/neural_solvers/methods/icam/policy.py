import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from EasyNCO.neural_solvers.methods import (
    TSPICAMEncoder,
    TSPICAMDecoder,
    CVRPICAMEncoder,
    CVRPICAMDecoder,
)
from EasyNCO.utils.utils import _get_encoding
from EasyNCO.neural_solvers.utils import get_post_search_strategy


class ICAMPolicy:
    def __init__(self, env_name: str, **kwargs):
        if env_name == "tsp":
            self.model = TSPICAMPolicy(**kwargs)
        elif env_name == "cvrp":
            self.model = CVRPICAMPolicy(**kwargs)
        else:
            raise NotImplementedError(f"Environment {env_name} is not implemented.")
    def _get_model(self):
        return self.model

class TSPICAMPolicy(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        sqrt_embedding_dim=128**0.5,
        num_encoder_layers: int = 12,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
        **kwargs  # Additional parameters for flexibility
    ):
        super().__init__()
        self.encoder = TSPICAMEncoder(
            embedding_dim=embed_dim,
            ff_hidden_dim=feedforward_hidden,
            encoder_layer_num=num_encoder_layers,
        )
        self.decoder = TSPICAMDecoder(
            embedding_dim=embed_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
        )

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):

        self.dist = (td["locs"][:, :, None, :] - td["locs"][:, None, :, :]).norm(
            p=2, dim=-1
        )  # shape: (batch, problem, problem)
        self.log_scale = math.log2(td["locs"].size(1))

        self.encoded_nodes = self.encoder(td["locs"], self.dist, self.log_scale)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, td: TensorDict) -> TensorDict:

        batch_size = td["action"].size(0)
        pomo_size = td["action"].size(1)

        if (td["action"] == -1).all():
            self.batch_idx = torch.arange(batch_size)[:, None].expand(
                batch_size, pomo_size
            )
            self.pomo_idx = torch.arange(pomo_size)[None, :].expand(
                batch_size, pomo_size
            )
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(
                self.encoded_nodes,
                selected.unsqueeze(2), 
            ).squeeze(2)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            current_node = td["action"][:, :, None, None].expand(
                batch_size, pomo_size, 1, self.dist.size(-1)
            )
            cur_dist = (
                self.dist[:, None, :, :]
                .expand(batch_size, pomo_size, self.dist.size(-1), self.dist.size(-1))
                .gather(2, current_node)
                .squeeze(2)
            )
            encoded_last_node = _get_encoding(
                self.encoded_nodes, td["action"].unsqueeze(2)
            ).squeeze(2)
            # shape: (batch, pomo, embedding)
            if self.decoder.q_first.shape != encoded_last_node.shape:
               # for RRC
                encoded_first_node = _get_encoding(
                    self.encoded_nodes,
                    td["first_node"].unsqueeze(2),
                ).squeeze(2)
                self.decoder.set_q1(encoded_first_node)
            probs = self.decoder(
                encoded_last_node,
                cur_dist,
                self.log_scale,
                ninf_mask=td["next"]["ninf_mask"],
            )
            # shape: (batch, pomo, problem)

            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        td.set("action", selected)
        if prob != None:
            td.set("prob", prob)

        return td


class CVRPICAMPolicy(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        sqrt_embedding_dim=128**0.5,
        num_encoder_layers: int = 12,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
        **kwargs  # Additional parameters for flexibility
    ):
        super().__init__()
        self.encoder = CVRPICAMEncoder(
            embedding_dim=embed_dim,
            ff_hidden_dim=feedforward_hidden,
            encoder_layer_num=num_encoder_layers,
        )
        self.decoder = CVRPICAMDecoder(
            embedding_dim=embed_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
        )
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):
        depot_xy = td["locs"][:, :1, :2]
        # shape: (batch, 1, 2)
        node_xy = td["locs"][:, 1:, :2]
        # shape: (batch, problem, 2)
        node_demand = td["locs"][:, 1:, -1]
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        self.dist = torch.cdist(
            td["locs"][:, :, :2],
            td["locs"][:, :, :2],
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )

        self.log_scale = math.log2(
            td["locs"].size(1) - 1
        )  # it is a scalar and used for influence of distance
        self.encoded_nodes = self.encoder(
            depot_xy, node_xy_demand, self.dist, self.log_scale
        )
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, td: TensorDict) -> TensorDict:
        batch_size = td["action"].size(0)
        pomo_size = td["action"].size(1)

        if (td["action"] == -1).all():  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif td["next"]["selected_count"][0][0] == 1 and pomo_size > 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(
                batch_size, pomo_size
            )
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            current_node = td["action"][:, :, None, None].expand(
                batch_size, pomo_size, 1, self.dist.size(-1)
            )
            encoded_last_node = _get_encoding(
                self.encoded_nodes, td["action"].unsqueeze(2)
            ).squeeze(2)
            # shape: (batch, pomo, embedding)
            cur_dist = (
                self.dist[:, None, :, :]
                .expand(
                    batch_size,
                    pomo_size,
                    self.dist.size(-1),
                    self.dist.size(-1),
                )
                .gather(2, current_node)
                .squeeze(2)
            )
            probs = self.decoder(
                encoded_last_node,
                td["load"],
                cur_dist,
                self.log_scale,
                ninf_mask=td["next"]["ninf_mask"],
            )
            # shape: (batch, pomo, problem+1)
            assert not torch.isnan(
                probs
            ).any(), "probs has nan, but it should not have any nans."

            if self.training or self.decoder_strategy == "sampling":
                # Check if sampling went OK, can go wrong due to bug on GPU
                # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
                # to fix pytorch.multinomial bug on selecting 0 probability elements
                while True:
                    selected = (
                        probs.reshape(batch_size * pomo_size, -1)
                        .multinomial(1)
                        .squeeze(dim=1)
                        .reshape(batch_size, pomo_size)
                    )
                    # shape: (batch, pomo)
                    prob = torch.gather(
                        probs, dim=-1, index=selected.unsqueeze(-1)
                    ).squeeze(-1)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            elif self.decoder_strategy == "greedy":
                selected = probs.argmax(dim=-1)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.
            else:
                raise NotImplementedError(
                    f"eval_type: {self.model_params['eval_type']} is not implemented!"
                )

        td.set("action", selected)
        if prob != None:
            td.set("prob", prob)

        return td
