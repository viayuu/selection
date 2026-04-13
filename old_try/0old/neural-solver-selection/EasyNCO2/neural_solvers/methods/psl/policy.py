from typing import Literal
import torch
import torch.nn as nn
from tensordict import TensorDict

from EasyNCO.neural_solvers.methods.psl.psl_encoder import PSLEncoder
from EasyNCO.neural_solvers.methods.psl.psl_decoder import PSLDecoder
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class PSLPolicy(nn.Module):
    def __init__(self,
                 env_name: str = "motsp",
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
                 pref: list = None,
                 ):
        super().__init__()
        num_target = len(pref)
        hyper_input_dim = len(pref)
        self.encoder = PSLEncoder(
                        env_name=env_name,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        num_layers=num_encoder_layers,
                        normalization=normalization,
                        feedforward_hidden=feedforward_hidden,
                        num_target=num_target
        )
        self.decoder = PSLDecoder(
                        env_name=env_name,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        logit_clipping=logit_clipping,
                        use_graph_mean=use_graph_mean,
                        am_mode=am_mode,
                        first_placeholder=first_placeholder,
                        hyper_input_dim=hyper_input_dim
        )
        self.decoder_strategy = None
        self.use_graph_mean = use_graph_mean
        self.first_mode = first_mode
        self.env_name = env_name

    def set_decoder_strategy(self, strategy: str="sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):
        self.encoded_nodes, _ = self.encoder(td)
        self.decoder.set_kv(self.encoded_nodes)
        if self.use_graph_mean:
            self.decoder.set_graph_mean(self.encoded_nodes)

    def forward(self, td: TensorDict) -> TensorDict:

        batch_size = td["action"].size(0)
        pomo_size = td["action"].size(1)

        if self.env_name == 'motsp':
            if (td['action'] + 1).sum() == 0:
                selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))

                encoded_first_node = _get_encoding(self.encoded_nodes, selected)
                # shape: (batch, pomo, embedding)
                self.decoder.set_q1(encoded_first_node)

                td.set("action", selected)
                td.set("prob", prob)

            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, td['action'])
                # shape: (batch, pomo, embedding)
                selected,probs = self.decoder(encoded_last_node,td, first_mode=self.first_mode)
                # shape: (batch, pomo, problem)

                if self.training:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
                    POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                    prob = probs[BATCH_IDX, POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    #     .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                else:
                    selected = probs.argmax(dim=2)

                    # shape: (batch, pomo)
                    BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
                    POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                    prob = probs[BATCH_IDX, POMO_IDX, selected].reshape(batch_size, pomo_size)

        elif self.env_name == 'mocvrp':
            if torch.all(td['next']['selected_count'] == 0):  # First Move, depot
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
                prob = torch.ones(size=(batch_size, pomo_size))

            elif torch.all(td['next']['selected_count'] == 1):  # Second Move, POMO
                selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))

            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, td['action'])
                # shape: (batch, pomo, embedding)
                selected,probs = self.decoder(encoded_last_node,td, first_mode=self.first_mode)
                # shape: (batch, pomo, problem+1)

                if self.training:
                    while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                        with torch.no_grad():
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
                        POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                        prob = probs[BATCH_IDX, POMO_IDX, selected].reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        if (prob != 0).all():
                            break

                else:
                    selected = probs.argmax(dim=2)
                    # shape: (batch, pomo)
                    # shape: (batch, pomo)
                    BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
                    POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                    prob = probs[BATCH_IDX, POMO_IDX, selected].reshape(batch_size, pomo_size)

        elif self.env_name == 'mokp':
            encoded_last_node = self.encoded_nodes #_get_encoding(self.encoded_nodes, td['action'])
            # shape: (batch, pomo, embedding)
            selected,probs = self.decoder(encoded_last_node,td, first_mode=self.first_mode)
            # shape: (batch, pomo, problem)

            if self.training:
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
                POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = probs[BATCH_IDX, POMO_IDX, selected].reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
                POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = probs[BATCH_IDX, POMO_IDX, selected].reshape(batch_size, pomo_size)

        td.set("action", selected)
        td.set("prob", prob)

        return td

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes