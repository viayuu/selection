import torch
import torch.nn as nn
from tensordict import TensorDict
from EasyNCO.neural_solvers.methods import LEHDEncoder, LEHDDecoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class LEHDPolicy(nn.Module):
    def __init__(
        self,
        phase: str = 'train',
        env_name: str = 'tsp',
        layer_num: int = 6,
        node_dim: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        feedforward_hidden: int = 512,
        normalization: str = None,
        multihead_bias: bool = False,
        first_mode: str = 'random',
    ):
        super().__init__()
        self.phase = phase
        self.problem_type = env_name
        self.encoder = LEHDEncoder(
                                    problem_type=env_name,
                                    node_dim=node_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    qkv_dim=qkv_dim,
                                    feedforward_hidden=feedforward_hidden,
                                    normalization=normalization,
                                    bias=multihead_bias
                                    )
        self.decoder = LEHDDecoder(
                                    problem_type=env_name,
                                    num_layers=layer_num,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    qkv_dim=qkv_dim,
                                    feedforward_hidden=feedforward_hidden,
                                    normalization=normalization,
                                    bias=multihead_bias
                                    )

        # only used in test
        self.embed_data = None
        self.decoder_strategy = None
        self.first_mode = first_mode

    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):
        if self.problem_type == 'tsp':
            self.problem_size = td['locs'].shape[1]
        else:
            self.problem_size = td['locs'].shape[1] - 1
        if self.phase == 'test':
            self.embed_data = self.encoder(td['locs']).unsqueeze(1)
        else:
            pass

    def forward(self, td:TensorDict):
        if self.phase == 'train':
            _, probs = self.decoder(self.encoder(td['locs']), td, self.phase)
            selected, prob = self.select_next_node(probs, td)

        elif self.phase == 'test':
            if 'partial_length' not in td.keys():
                td['partial_length'] = torch.zeros((td.batch_size[0],td.batch_size[1]), dtype=torch.int) + self.problem_size
            selected, probs = self.decoder(self.embed_data, td, self.phase, self.first_mode)
            if selected is not None:
                if self.problem_type == 'cvrp':
                    selected = torch.cat((selected[:, :, None], torch.ones((td.batch_size[0], td.batch_size[1], 1), dtype=selected.dtype)), dim=-1)
                prob = probs
            else:
                selected, prob = self.select_next_node(probs, td)
        else:
            raise RuntimeError(f"Invalid learning phase: {self.phase}.")

        td.set("action", selected)
        td.set("prob", prob)

        return td

    def select_next_node(self, probs, td):
        '''
        The method is used to select the next node and its corresponding probability.
        probs.shape: (batch, node_num), if CVRP uses flag in its tours, then probs.shape: (batch, node_num*2)
        '''
        if self.phase == 'train':
            next_selected = td['next_select']
            cur_problem_size = td['partial_length'][0, 0]

            if self.problem_type == 'tsp':
                selected = next_selected
                prob = probs.gather(dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)

            elif self.problem_type == 'cvrp':
                selected = next_selected
                selected_node = selected[:, :, 0] - 1
                selected_flag = selected[:, :, 1]
                # shape (batch, pomo, selected_count)
                is_via_depot = (selected_flag == 1)
                selected_node_copy = selected_node.clone()
                selected_node_copy[is_via_depot] += cur_problem_size
                prob = probs.gather(dim=-1, index=selected_node_copy.unsqueeze(-1)).squeeze(-1)

        elif self.phase == 'test':
            # problem_size = td['partial_length'][0, 0] if 'partial_length' in td.keys() else 0
            new_selected, prob = get_post_search_strategy(self.decoder_strategy, probs)
            selected = new_selected
            if self.problem_type == 'cvrp':
                new_selected += 1
                at_depot_index = new_selected > self.problem_size
                new_selected[at_depot_index] -= self.problem_size
                flag = torch.zeros_like(new_selected, dtype=torch.int32)
                flag[at_depot_index] = 1
                selected = torch.cat((new_selected[:, :, None], flag[:, :, None]), dim=-1) # shape (batch, pomo, 2)
        else:
            raise RuntimeError(f"Invalid learning phase: {self.phase}.")

        return selected, prob