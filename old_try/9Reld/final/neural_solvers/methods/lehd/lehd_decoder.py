import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from EasyNCO.neural_solvers.backbones import TransformerNet
from EasyNCO.utils.utils import getLogger
from EasyNCO.utils.utils import _get_encoding
from EasyNCO.neural_solvers.utils import special_selected

logger = getLogger(__name__)

class LEHDDecoder(nn.Module):
    def __init__(
        self,
        problem_type: str = 'tsp',
        num_layers: int = 6,
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        feedforward_hidden: int = 512,
        normalization: str = None,
        bias: bool = False,
        first_placeholder: bool = False,  # only used in TSP
    ):
        super().__init__()
        self.problem_type = problem_type
        self.embed_dim = embed_dim
        self.first_placeholder = first_placeholder
        if problem_type == 'tsp':
            self.destination_project = nn.Linear(embed_dim, embed_dim, bias=True)
            self.start_project = nn.Linear(embed_dim, embed_dim, bias=True)
            self.final = nn.Linear(embed_dim, 1, bias=True)
        elif problem_type == 'cvrp':
            self.destination_project = nn.Linear(embed_dim + 1, embed_dim, bias=True)
            self.start_project = nn.Linear(embed_dim + 1, embed_dim, bias=True)
            self.final = nn.Linear(embed_dim, 2, bias=True)
        else:
            raise RuntimeError(f"Invalid problem type: {self.problem_type}.")

        self.decode_layers = TransformerNet(num_layers, num_heads, qkv_dim, embed_dim, normalization, feedforward_hidden, bias)

    def set_problem_size(self, problem_size: int):
        self.problem_size = problem_size


    def forward(self, embed_data, td: TensorDict, phase: str, first_mode: str = 'random'):
        # embed_data.shape (batch, pomo, problem, embed_dim)
        if phase == 'test' and td['next']['selected_node_list'].size(2) == 0:
            if self.problem_type != 'cvrp' or (self.problem_type == 'cvrp' and td.batch_size[1] > 1):
                selected, probs = special_selected(self.problem_type,
                                                   {
                                                       "td": td,
                                                       "first_mode": first_mode,
                                                       "first_placeholder": self.first_placeholder,
                                                   })
            else:
                selected, probs = None, None

                if 'remain_capacity' not in td.keys():
                    td['remain_capacity'] = td['vehicle_capacity'].squeeze(-1)
        else:
            selected, probs = None, None

        if probs is None:
            selected_list = td['next']['selected_node_list'].clone()

            if self.problem_type == 'tsp':
                destination_embed = _get_encoding(embed_data, selected_list[:, :, [0]]) #shape (batch, pomo, 1, embed_dim)
                start_embed = _get_encoding(embed_data, selected_list[:, :, [-1]]) #shape (batch, pomo, 1, embed_dim)
                available_embed = self.get_available_embed_node(embed_data, td) #shape (batch, pomo, N, embed_dim)
                destination_projection = self.destination_project(destination_embed)
                start_projection = self.start_project(start_embed)

            elif self.problem_type == 'cvrp':
                remain_capacity = td['remain_capacity'][:, :, None, None] #shape (batch, pomo, 1, 1)
                destination_embed = embed_data[:, :, [0], :].expand(-1, td.batch_size[1], -1, -1)
                if selected_list.shape[2] == 0:
                    start_embed = destination_embed
                else:
                    start_embed = _get_encoding(embed_data, selected_list[:, :, [-1], 0])
                available_embed = self.get_available_embed_node(embed_data[:, :, 1:, :], td)
                destination_projection = self.destination_project(torch.cat((destination_embed, remain_capacity), dim=-1))
                start_projection = self.start_project(torch.cat((start_embed, remain_capacity), dim=-1))

            else:
                raise RuntimeError(f"Invalid problem type: {self.problem_type}.")

            out_1 = torch.cat((destination_projection, available_embed, start_projection), dim=-2)
            out_2 = self.decode_layers(out_1)

            out = self.final(out_2).squeeze(-1)
            probs = self.calculate_probs(out, td)

            selected = None
        return selected, probs

    def get_available_embed_node(self, data, td):
        '''
        Thd method is used to fetch embed data of available nodes.
        '''
        selected_list = td['next']['selected_node_list']
        batch_size = td.batch_size[0]
        pomo_size = td.batch_size[1]
        cur_problem_size = td['partial_length'][0, 0]
        selected_count = td['next']['selected_count'][0, 0]
        if self.problem_type == 'tsp':
            cur_selected_node_list = selected_list #shape (batch, pomo, selected_count)
        elif self.problem_type == 'cvrp':
            cur_selected_node_list = selected_list[:, :, :, 0] - 1 #shape (batch, pomo, selected_count)
        else:
            raise RuntimeError(f"Invalid problem type: {self.problem_type}.")

        new_selected_node_list = torch.arange(cur_problem_size)[None, None, :].expand(batch_size, pomo_size, -1).clone()
        # shape: (batch, pomo, node_num)
        new_data_len = cur_problem_size - selected_count
        new_selected_node_list.scatter_(dim=-1, index=cur_selected_node_list, value=-2)
        unselect_node_list = new_selected_node_list[torch.gt(new_selected_node_list, -1)].view(batch_size, pomo_size, new_data_len)
        # shape (batch, pomo, unselected_node_num)
        new_data = _get_encoding(data, unselect_node_list)
        # shape (batch, pomo, unselected_node_num, embed_dim)
        if new_data.shape[2] == 0:
            new_data = data

        return new_data

    def calculate_probs(self, out, td):
        selected_list = td['next']['selected_node_list']
        batch_size = td.batch_size[0]
        pomo_size = td.batch_size[1]
        cur_problem_size = td['partial_length'][0,0]
        selected_count = td['next']['selected_count'][0,0]

        if self.problem_type == 'tsp':
            cur_selected_node_list = selected_list
            out[:, :, [0, -1]] = out[:, :, [0, -1]] + float('-inf')  # shape: (batch, pomo, reminding_nodes_number + 2)
            probs = F.softmax(out, dim=-1)
            probs_available = probs[:, :, 1:-1].clone()
            new_probs = torch.zeros(batch_size, pomo_size, cur_problem_size)
            index_small = torch.le(probs_available, 1e-5)
            probs_available[index_small] = probs_available[index_small] + torch.tensor(1e-7, dtype=probs_available[index_small].dtype)  # prevent the probability from being too small
            cur_selected_node_index = cur_selected_node_list

        elif self.problem_type == 'cvrp':
            cur_selected_node_list = selected_list[:, :, :, 0] - 1
            out[:, :, [0, -1], :] = out[:, :, [0, -1], :] + float('-inf')  # shape: (batch, pomo, reminding_nodes_number + 2, 2)
            out = torch.cat((out[:, :, :, 0], out[:, :, :, 1]), dim=-1) # shape: (batch, pomo, 2*(reminding_nodes_number + 2))
            available_node_num = cur_problem_size - selected_count
            probs = F.softmax(out, dim=-1)
            probs_available = torch.cat((probs[:, :, 1:available_node_num + 1], probs[:, :, -1 - available_node_num:-1]), dim=-1).clone()
            new_probs = torch.zeros(batch_size, pomo_size, 2 * cur_problem_size)
            index_small = torch.le(probs_available, 1e-5)
            probs_available[index_small] = probs_available[index_small] + torch.tensor(1e-7, dtype=probs_available[index_small].dtype)  # prevent the probability from being too small
            cur_selected_node_index = torch.cat((cur_selected_node_list, cur_problem_size+cur_selected_node_list), dim=-1)

        else:
            raise RuntimeError(f"Invalid problem type: {self.problem_type}.")

        # new_probs.scatter_(dim=-1, index=cur_selected_node_index, value=float('-inf'))
        new_probs.scatter_(dim=-1, index=cur_selected_node_index, value=-2)
        index = torch.gt(new_probs, -1).view(batch_size, pomo_size, -1)
        new_probs[index] = probs_available.ravel()

        return new_probs