import torch
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List
import math
from torch.utils.data import DataLoader,Dataset


import argparse

from EasyNCO.exact_solvers.params_setting import get_options


def update_buffer(memory, trajectory):
    """this buffer is for tranining step, which saves (states,rewards, dones, actions,...) """
    current_items = list(map(list, zip(*trajectory))) 
    current_items = [torch.cat(item, dim=0) for item in current_items]
    memory[:] = current_items
    num_steps = len(memory[1])
    experience_reward = memory[1].mean().item()

    return num_steps, experience_reward



def node_distance(left, right):
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)



class FragmentBuffer:
    """This is segment path for lower solver"""
    def __init__(self, max_len: int, frag_len: int, node_dim: int = 2) -> None:
        self.max_len = max_len
        self.frag_len = frag_len
        self.node_dim = node_dim
        self.frag_buffer = torch.empty((max_len, frag_len, node_dim))
        self.if_full = False
        self.now_len = 0
        self.next_idx = 0

    def update_buffer(self, fragments: torch.Tensor) -> None:
        size = fragments.shape[0]
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            self.frag_buffer[self.next_idx : self.max_len] = fragments[:self.max_len-self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len
            self.frag_buffer[0:next_idx] = fragments[-next_idx:]
        else:
            self.frag_buffer[self.next_idx : next_idx] = fragments
        self.next_idx = next_idx
        self.update_now_len()

    def update_now_len(self) -> None:
        self.now_len = self.max_len if self.if_full else self.next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = np.random.randint(self.now_len - 1, size=batch_size)
        return self.frag_buffer[indices]



# ============================================================================
# lower level model
# ============================================================================



DISTANCE_SCALE = 1000
class RLSolver():
    def __init__(self, model:type , sample_size: int = 200) -> None:

        # initiate low_level_model in policy
        self.low_level_model = model
        self._sample_size = max(sample_size, 2)


    def solve(self,
              data: torch.Tensor,
              fragment: torch.Tensor,
              frag_buffer: FragmentBuffer):
        
        node_pos = torch.gather(input=data, index=fragment[..., None].expand(-1, -1, data.shape[-1]), dim=1)
        device = data.device
        B = node_pos.shape[0]  #batch
        N = node_pos.shape[1]  #problem_size
        x = node_pos.to(device=device) 

        #normalization
        x1_min = x[..., 0].min(dim=-1, keepdim=True)[0]
        x2_min = x[..., 1].min(dim=-1, keepdim=True)[0]
        x1_max = x[..., 0].max(dim=-1, keepdim=True)[0]
        x2_max = x[..., 1].max(dim=-1, keepdim=True)[0]
        s = 0.9 / torch.maximum(x1_max - x1_min, x2_max - x2_min)
        x_new = torch.empty_like(x)
        x_new[..., 0] = s * (x[..., 0] - x1_min) + 0.05
        x_new[..., 1] = s * (x[..., 1] - x2_min) + 0.05 
        frag_buffer.update_buffer(x_new.cpu())
        source_nodes = torch.tensor([[0]], device=device).expand(B, 1)
        target_nodes = torch.tensor([[N - 1]], device=device).expand(B, 1)

        # lower_level model solve the problem
        lengths, paths = self.low_level_model(x_new.float(),
                                            source_nodes=source_nodes,
                                            target_nodes=target_nodes,
                                            val_type="x8Aug_2Traj",
                                            group_size=self._sample_size,)
 
        lengths = (lengths / s.squeeze()).detach().cpu().numpy()
        out_paths = []
        paths = paths.detach().cpu().numpy()
        fragment = fragment.cpu().numpy()
        if B == 1:
            paths = [paths]
        for i, path in enumerate(paths):
            # make sure path start with `0` and end with `N-1`
            zero_idx = np.nonzero(path == 0)[0][0]
            path = np.roll(path, -zero_idx)

            if path[-1] != N - 1:
                path = np.roll(path, -1)
                path = np.flip(path)

            new_path = [fragment[i][j] for j in path]
            out_paths.append(new_path)

        return out_paths, lengths
    


class LKHSolver():
    def solve(self, data: np.ndarray, fragment: np.ndarray, runs=2):

        node_pos = np.take_along_axis(data, fragment[..., None], axis=1)

        dist_matrice = []
        for fragment_pos in node_pos:
            dist_matrix = cdist(fragment_pos, fragment_pos)
            # make sure the souce node is the cloest node for target node
            dist_matrix[-1][1:] = 0.9999  
            dist_matrice.append(dist_matrix)
            # dist_matrice(List):(batch, [num_node, num_node])
            
        # list to tensor
        dist_matrice = torch.tensor(np.array(dist_matrice), dtype=torch.float32)

        par_args=get_options()
        # parser = argparse.ArgumentParser()
        # opts = parser.parse_args(args)
        par_args.solver = "lkh"
        par_args.ptype = "atsp"
        par_args.RUNS = 2
        par_args.MAX_TRIALS = node_pos.shape[1]  # 动态设置
        par_args.int_matrix_scale = 10000
        par_args.cpus = 4
        par_args.TRACE_LEVEL = 1
        par_args.SEED = 1
        par_args.CANDIDATE_SET_TYPE = "ALPHA"
        par_args.save_as_txt = True
        par_args.delete = False
        par_args.solver_mode = "multi"
        
        
        # par_args.RUNS = 2
        # par_args.MAX_TRIALS = node_pos.shape[1]
        # par_args.ptype = "ATSP"
        # par_args.int_matrix_scale = 10000

        data_list = []
        #ATSP_dataset
        atsp_dataloader = DataLoader(ATSP_Dataset(dist_matrice),
                                    batch_size=dist_matrice.shape[0])
        for batch_idx, batch in enumerate(atsp_dataloader):
            for instance in batch:
                node_matrix = instance
                data_list.append(node_matrix)

        # sovler from EasyNCO/exact_sovler
        from EasyNCO.exact_solvers.exact_solver_main import exact_solver_main
        results = exact_solver_main(par_args, data_list=data_list)

        
        paths, lengths = [], []
        
        for res in results:
            route, dist = res 
            # the route for 
            route = [r - 1 for r in route]
            paths.append(route)
            lengths.append(dist)
            
        lengths = np.array(lengths)
        out_paths = []
        for i, path in enumerate(paths):
            assert path[0] == 0
            new_path = [fragment[i][j] for j in path]
            out_paths.append(new_path)
        
        return out_paths, lengths



class GreedySolver():
    """Optional solver"""

    def solve(self, x: np.ndarray, fragment: List[int]):
        if len(fragment) == 0:
            return []
        fragment_pos = x.take(fragment, axis=0)
        frag_len = len(fragment) - 1
        dist = cdist(fragment_pos, fragment_pos)
        dist_search = dist.copy()
        np.fill_diagonal(dist_search, np.inf)
        dist_search[:, frag_len] = np.inf
        dist_search[:, 0] = np.inf
        greedy_tour = [0]
        city_last = 0
        length = 0.0

        while len(greedy_tour) < frag_len:
            city_next = dist_search[city_last].argmin()
            length += dist[city_last, city_next]
            greedy_tour.append(city_next)
            dist_search[:, city_next] = np.inf
            city_last = city_next
        greedy_tour.append(frag_len)
        length += dist[city_last, frag_len]
        new_path = [fragment[i] for i in greedy_tour]

        return new_path, length


class ATSP_Dataset(Dataset):

    def __init__(self, tensor):
        self.tensor = tensor.to("cpu")

    def __len__(self):
        return self.tensor.size(0)  # batch_size

    def __getitem__(self, idx):
        return self.tensor[idx] 