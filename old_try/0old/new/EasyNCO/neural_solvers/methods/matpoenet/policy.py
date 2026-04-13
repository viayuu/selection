from EasyNCO.neural_solvers.utils.post_search import get_post_search_strategy
from EasyNCO.neural_solvers.methods import MatNetEncoder
from EasyNCO.neural_solvers.methods import ATSPDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from EasyNCO.utils.utils import _get_encoding



class MatPOENetPolicy(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 16,
        qkv_dim: int = 16,
        num_encoder_layers: int = 5,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10.0,
        one_hot_seed_cnt: int = 20,  # must be >= node_cnt
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = (1 / 2) ** (1 / 2),
        ms_layer2_init: float = (1 / 16) ** (1 / 2),
        pos_embedding_dim: int = 512,
    ):
        super().__init__()
        self.encoder = MatNetEncoder(
            encoder_layer_num=num_encoder_layers,
            embedding_dim=embedding_dim,
            head_num=num_heads,
            qkv_dim=qkv_dim,
            ms_hidden_dim=ms_hidden_dim,
            ms_layer1_init=ms_layer1_init,
            ms_layer2_init=ms_layer2_init,
            ff_hidden_dim=feedforward_hidden,
        )
        self.decoder = ATSPDecoder(
            head_num=num_heads,
            embedding_dim=embedding_dim,
            qkv_dim=qkv_dim,
            logit_clipping=logit_clipping,
        )
        self.decoder_strategy = None
        self.encoded_row = None
        self.encoded_col = None
        self.embedding_dim = embedding_dim
        self.seed_cnt = one_hot_seed_cnt

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):
        problems = td["problems"]
        # problems.shape: (batch, node, node)
        batch_size = problems.size(0)
        node_cnt = problems.size(1)

        
        col_emb = td["pos_emb"]
        # shape: (batch, node, embedding)
        row_emb = torch.zeros_like(col_emb)
        # emb.shape: (batch, node, embedding)


        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        # encoded_nodes.shape: (batch, node, embedding)

        self.decoder.set_kv(self.encoded_col)


    def forward(self, td: TensorDict) -> TensorDict:
        batch_size = td["batch_idx"].size(0)
        pomo_size = td["batch_idx"].size(1)

        if torch.equal(td["current_node"], -torch.ones(batch_size, pomo_size)):
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            # encoded_rows_mean = self.encoded_row.mean(dim=1, keepdim=True)
            # encoded_cols_mean = self.encoded_col.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            encoded_first_row = _get_encoding(
                self.encoded_row, selected[:, :, None]
            ).squeeze()
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = _get_encoding(
                self.encoded_row, td["current_node"][:, :, None]
            ).squeeze()
            # shape: (batch, pomo, embedding)
            all_job_probs = self.decoder(encoded_current_row, ninf_mask=td["ninf_mask"])
            # shape: (batch, pomo, job)

            selected, prob = get_post_search_strategy(
                self.decoder_strategy, all_job_probs
            )
        td.set("action", selected)
        td.set("prob", prob)

        return td





