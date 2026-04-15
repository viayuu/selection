from EasyNCO.neural_solvers.utils.post_search import get_post_search_strategy
from EasyNCO.neural_solvers.methods import MatNetEncoder
from EasyNCO.neural_solvers.methods import ATSPDecoder, FFSPDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from EasyNCO.utils.utils import _get_encoding


class MatNetPolicy:
    def __init__(self, env_name: str, **kwargs):
        if env_name == "atsp":
            self.model = ATSPPolicy(**kwargs)
        elif env_name == "ffsp":
            self.model = FFSPPolicy(**kwargs)
        else:
            raise NotImplementedError(f"Environment {env_name} is not implemented.")
    def _get_model(self):
        return self.model

class MatNetGLOPPolicy(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 16,
        qkv_dim: int = 16,
        num_encoder_layers: int = 5,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10.0,
        # one_hot_seed_cnt: int = 20,  # must be >= node_cnt #! delete
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = (1 / 2) ** (1 / 2),
        ms_layer2_init: float = (1 / 16) ** (1 / 2),
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
        # self.seed_cnt = one_hot_seed_cnt  #! delete

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):

        problems = td["locs"]
        # problems.shape: (batch, node, node)

        batch_size = problems.size(0)
        node_cnt = problems.size(1)

        row_emb = torch.zeros(size=(batch_size, node_cnt, self.embedding_dim))
        # emb.shape: (batch, node, embedding)
        col_emb = torch.zeros(size=(batch_size, node_cnt, self.embedding_dim))
        # shape: (batch, node, embedding)

        rand = torch.rand(batch_size, node_cnt)  #! self.seed_cnt = node_cnt
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :node_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, node_cnt)
        n_idx = torch.arange(node_cnt)[None, :].expand(batch_size, node_cnt)
        col_emb[b_idx, n_idx, rand_idx] = 1
        # shape: (batch, node, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        # encoded_nodes.shape: (batch, node, embedding)

        self.decoder.set_kv(self.encoded_col)

    def forward(self, state: TensorDict):

        if (state["action"] == 0).all():
            encoded_last_row = _get_encoding(
                self.encoded_row, state["last_node"][:, :, None]
            )
            encoded_last_row = encoded_last_row.squeeze()
            if encoded_last_row.dim() == 2:
                encoded_last_row = encoded_last_row.unsqueeze(1)
            self.decoder.set_q1(encoded_last_row)  # batch_size,pomo,embedding_dim

        encoded_current_row = _get_encoding(
            self.encoded_row, state["action"][:, :, None]
        )
        encoded_current_row = encoded_current_row.squeeze()
        if encoded_current_row.dim() == 2:
            encoded_current_row = encoded_current_row.unsqueeze(1)
        # shape: (batch, pomo, embedding)
        all_job_probs = self.decoder(
            encoded_current_row, ninf_mask=state["next"]["ninf_mask"]
        )
        # shape: (batch, pomo, job)
        selected, prob = get_post_search_strategy(self.decoder_strategy, all_job_probs)

        state.set("action", selected)
        state.set("prob", prob)
        return state


class ATSPPolicy(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 16,
        qkv_dim: int = 16,
        num_encoder_layers: int = 5,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10.0,
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = (1 / 2) ** (1 / 2),
        ms_layer2_init: float = (1 / 16) ** (1 / 2),
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

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):
        problems = td["problems"]
        # problems.shape: (batch, node, node)
        batch_size = problems.size(0)
        node_cnt = problems.size(1)

        row_emb = torch.zeros(size=(batch_size, node_cnt, self.embedding_dim))
        # emb.shape: (batch, node, embedding)
        col_emb = torch.zeros(size=(batch_size, node_cnt, self.embedding_dim))
        # shape: (batch, node, embedding)

        self.seed_cnt = node_cnt

        rand = torch.rand(batch_size, self.seed_cnt)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :node_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, node_cnt)
        n_idx = torch.arange(node_cnt)[None, :].expand(batch_size, node_cnt)
        col_emb[b_idx, n_idx, rand_idx] = 1
        # shape: (batch, node, embedding)

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


class FFSPPolicy(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 16,
        qkv_dim: int = 16,
        num_encoder_layers: int = 5,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10.0,
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = (1 / 2) ** (1 / 2),
        ms_layer2_init: float = (1 / 16) ** (1 / 2),
        machine_cnt_list: list = [4, 4, 4],
    ):
        super().__init__()
        self.machine_cnt_list = machine_cnt_list
        self.stage_models = nn.ModuleList(
            [
                OneStageModel(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    qkv_dim=qkv_dim,
                    num_encoder_layers=num_encoder_layers,
                    feedforward_hidden=feedforward_hidden,
                    logit_clipping=logit_clipping,
                    ms_hidden_dim=ms_hidden_dim,
                    ms_layer1_init=ms_layer1_init,
                    ms_layer2_init=ms_layer2_init,
                    machine_cnt_list=machine_cnt_list,
                )
                for stage_idx in range(len(machine_cnt_list))
            ]
        )

    def set_decoder_strategy(self, strategy: str = "sampling"):
        for model in self.stage_models:
            model.decoder_strategy = strategy

    def pre_forward(self, td: TensorDict):
        for stage_idx in range(len(self.machine_cnt_list)):
            problems = td["problems"][
                :,
                :,
                sum(self.machine_cnt_list[:stage_idx]) : sum(
                    self.machine_cnt_list[: stage_idx + 1]
                ),
            ]
            model = self.stage_models[stage_idx]
            model.pre_forward(problems)

    def forward(self, td: TensorDict):
        batch_size = td["batch_idx"].size(0)
        pomo_size = td["batch_idx"].size(1)

        action_stack = torch.empty(
            size=(batch_size, pomo_size, len(self.machine_cnt_list)), dtype=torch.long
        )
        prob_stack = torch.empty(
            size=(batch_size, pomo_size, len(self.machine_cnt_list))
        )

        for stage_idx in range(len(self.machine_cnt_list)):
            model = self.stage_models[stage_idx]
            action, prob = model(td)

            action_stack[:, :, stage_idx] = action
            prob_stack[:, :, stage_idx] = prob

        gathering_index = td["stage_idx"][:, :, None]
        # shape: (batch, pomo, 1)
        action = action_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        prob = prob_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        td.set("action", action)
        td.set("prob", prob)

        return td


class OneStageModel(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 16,
        qkv_dim: int = 16,
        num_encoder_layers: int = 5,
        feedforward_hidden: int = 512,
        logit_clipping: float = 10.0,
        ms_hidden_dim: int = 16,
        ms_layer1_init: float = (1 / 2) ** (1 / 2),
        ms_layer2_init: float = (1 / 16) ** (1 / 2),
        machine_cnt_list: list = [4, 4, 4],
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
        self.decoder = FFSPDecoder(
            embedding_dim=embedding_dim,
            head_num=num_heads,
            qkv_dim=qkv_dim,
            logit_clipping=logit_clipping,
        )
        self.decoder_strategy = None

        self.machine_cnt_list = machine_cnt_list
        self.embedding_dim = embedding_dim
        self.encoded_col = None
        # shape: (batch, machine_cnt, embedding)
        self.encoded_row = None
        # shape: (batch, job_cnt, embedding)

    def pre_forward(self, problems):
        # problems.shape: (batch, job_cnt, machine_cnt)
        batch_size = problems.size(0)
        job_cnt = problems.size(1)
        machine_cnt = problems.size(2)

        row_emb = torch.zeros(size=(batch_size, job_cnt, self.embedding_dim))
        # shape: (batch, job_cnt, embedding)
        col_emb = torch.zeros(size=(batch_size, machine_cnt, self.embedding_dim))
        # shape: (batch, machine_cnt, embedding)

        seed_cnt = job_cnt
        # torch.manual_seed(0)
        rand = torch.rand(batch_size, seed_cnt)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :machine_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, machine_cnt)
        m_idx = torch.arange(machine_cnt)[None, :].expand(batch_size, machine_cnt)
        col_emb[b_idx, m_idx, rand_idx] = 1
        # shape: (batch, machine_cnt, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        # encoded_row.shape: (batch, job_cnt, embedding)
        # encoded_col.shape: (batch, machine_cnt, embedding)

        self.decoder.set_kv(self.encoded_row)

    def forward(self, td):
        batch_size = td["batch_idx"].size(0)
        pomo_size = td["batch_idx"].size(1)

        encoded_current_machine = _get_encoding(
            self.encoded_col, td["stage_machine_idx"][:, :, None]
        ).squeeze()
        # shape: (batch, pomo, embedding)
        all_job_probs = self.decoder(
            encoded_current_machine, ninf_mask=td["job_ninf_mask"]
        )
        # shape: (batch, pomo, job)

        job_selected, job_prob = get_post_search_strategy(
            self.decoder_strategy, all_job_probs
        )

        return job_selected, job_prob
