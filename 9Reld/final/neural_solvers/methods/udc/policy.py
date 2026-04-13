import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods import (
    TSPICAMEncoder,
    TSPICAMDecoder,
    CVRPICAMEncoder,
    CVRPICAMDecoder,
)
from EasyNCO.neural_solvers.backbones.GNN.partition_net import glop_partition_net
from EasyNCO.utils.utils import _get_encoding
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class UDCPolicy():
    def __init__(self, env_name: str, **kwargs):
        super().__init__()
        if env_name == "tsp":
            self.model = TSPUDCPolicy(env_name=env_name, **kwargs)
        elif env_name == "cvrp":
            self.model = CVRPUDCPolicy(env_name=env_name, **kwargs)
        else:
            raise NotImplementedError(f"Not implemented for {env_name}")
        self.model_t = self.model.model_t
        self.model_p = self.model.model_p
    def _get_model(self):
        return self.model


class TSPUDCPolicy(nn.Module):

    def __init__(
        self,
        env_name: str = "tsp",
        embed_dim: int = 128,
        sqrt_embedding_dim=128**0.5,
        num_heads: int = 8,
        qkv_dim: int = 16,
        num_encoder_layers: int = 6,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
        units: int = 64,
        feats: int = 2,
        edge_feats: int = 2,
        depth: int = 12,
        k_sparse: int = 100,
    ):
        super().__init__()
        self.model_p = glop_partition_net(
            units=units,
            feats=feats,
            k_sparse=k_sparse,
            edge_feats=edge_feats,
            depth=depth,
            udc_flag="tsp",
        )
        self.model_t = TSPModel(
            embed_dim=embed_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            num_encoder_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            logit_clipping=logit_clipping,
        )

    def parameters(self, recurse: bool = True):
        return self.model_p.parameters(recurse=recurse)


class CVRPUDCPolicy(nn.Module):

    def __init__(
        self,
        env_name: str = "cvrp",
        embed_dim: int = 128,
        sqrt_embedding_dim=128**0.5,
        num_heads: int = 8,
        qkv_dim: int = 16,
        num_encoder_layers: int = 6,
        feedforward_hidden: int = 512,
        logit_clipping: float = 50,  # clipping value for logits, used in Compatibility
        units: int = 64,
        feats: int = 2,
        edge_feats: int = 2,
        depth: int = 12,
        k_sparse: int = 100,
    ):
        super().__init__()
        self.model_p = glop_partition_net(
            units=units,
            feats=feats,
            k_sparse=k_sparse,
            edge_feats=edge_feats,
            depth=depth,
            udc_flag="cvrp",
        )
        self.model_t = CVRPModel(
            embed_dim=embed_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            num_encoder_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            logit_clipping=logit_clipping,
        )

    def parameters(self, recurse: bool = True):
        return self.model_p.parameters(recurse=recurse)


class TSPModel(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        sqrt_embedding_dim=128**0.5,
        num_encoder_layers: int = 6,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
    ):
        super().__init__()
        self.encoder = TSPICAMEncoder(
            embedding_dim=embed_dim,
            ff_hidden_dim=feedforward_hidden,
            encoder_layer_num=num_encoder_layers,
            udc_flag=True,
        )
        self.decoder = TSPICAMDecoder(
            embedding_dim=embed_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
            udc_flag=True,
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
            selected = torch.cat(
                (
                    torch.zeros(pomo_size // 2)[None, :].expand(
                        batch_size, pomo_size // 2
                    ),
                    self.dist.size(-1)
                    - torch.ones(pomo_size // 2)[None, :].expand(
                        batch_size, pomo_size // 2
                    ),
                ),
                dim=-1,
            ).long()
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(
                self.encoded_nodes,
                torch.cat(
                    (
                        self.dist.size(-1)
                        - torch.ones(pomo_size // 2)[None, :].expand(
                            batch_size, pomo_size // 2
                        ),
                        torch.zeros(pomo_size // 2)[None, :].expand(
                            batch_size, pomo_size // 2
                        ),
                    ),
                    dim=-1,
                )
                .unsqueeze(2)
                .long(),
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


class CVRPModel(nn.Module):

    def __init__(
        self,
        embed_dim: int = 128,
        sqrt_embedding_dim=128**0.5,
        num_encoder_layers: int = 6,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
    ):
        super().__init__()
        self.encoder = CVRPICAMEncoder(
            embedding_dim=embed_dim,
            ff_hidden_dim=feedforward_hidden,
            encoder_layer_num=num_encoder_layers,
            udc_flag=True,
        )
        self.decoder = CVRPICAMDecoder(
            embedding_dim=embed_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
            udc_flag=True,
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
            depot_xy, node_xy_demand, self.dist, self.log_scale, self.flag_return
        )
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, td: TensorDict, first_mode: str = "random") -> TensorDict:
        batch_size = td["action"].size(0)
        pomo_size = td["action"].size(1)

        if (td["action"] == -1).all():  # First Move, depot
            self.batch_idx = torch.arange(batch_size)[:, None].expand(
                batch_size, pomo_size
            )
            self.pomo_idx = torch.arange(pomo_size)[None, :].expand(
                batch_size, pomo_size
            )
            selected = (
                torch.ones(pomo_size)[None, :].expand(batch_size, pomo_size).long()
            )
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = _get_encoding(
                self.encoded_nodes,
                (
                    (
                        self.dist.size(-1)
                        - torch.ones(pomo_size)[None, :]
                        .expand(batch_size, pomo_size)
                        .long()
                    )
                    * (1 - self.flag_return)[:, None]
                )
                .unsqueeze(2)
                .long(),
            ).squeeze(2)
            self.decoder.set_q1(encoded_first_node)
            encoded_depot = _get_encoding(
                self.encoded_nodes,
                torch.zeros(pomo_size)[None, :]
                .expand(batch_size, pomo_size)
                .unsqueeze(2)
                .long(),
            ).squeeze(2)
            self.decoder.set_q2(encoded_depot)

        else:
            current_node = td["action"][:, :, None, None].expand(
                batch_size, pomo_size, 1, self.dist.size(-1)
            )
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
            encoded_last_node = _get_encoding(
                self.encoded_nodes, td["action"].unsqueeze(2)
            ).squeeze(2)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(
                encoded_last_node,
                td["load"],
                cur_dist,
                self.log_scale,
                td["next"]["ninf_mask"],
                self.left,
            )
            # shape: (batch, pomo, problem+1)

            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        td.set("action", selected)
        if prob != None:
            td.set("prob", prob)

        return td
