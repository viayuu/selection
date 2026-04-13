import torch
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
# from EasyNCO.data.data_utils import augment_xy_data_by_8_fold, run_aug
from EasyNCO.data import ATSPGenerator


# for MatPOENet
from EasyNCO.neural_solvers.methods.matpoenet.utils import *
import numpy as np

logger = getLogger(__name__)


class SATEnv(EnvBase):

    def __init__(
        self,
        pomo_size: int = 20,
        device: str = "cpu",
        seed: int = 0,
        problem_size: int = 20,
        **kwargs,
    ):
        super().__init__(device=device)
        self.env_name = "sat"
        self.node_cnt = problem_size
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device
        self.aug_type = None

        self.seed_value = seed
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        self._set_seed(seed=self.seed_value)
        self.aug_factor = kwargs.get("aug_factor")
        
        
        self.pos_embedding_dim = kwargs.get("pos_embedding_dim")
        self.init_solver = kwargs.get("init_solver")
        
        self.method_name = kwargs.get("method_name")

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def load_problems(self, dataset, batch_size: int = 64):
        
        self.problems = dataset
        # shape: (batch, node, node)
        
        self.env_batch_size = batch_size
        self.pomo_size = self.problems.size(1)


        self.batch_size = torch.Size([batch_size, self.pomo_size])
        if self.aug_factor != 1:
            self.batch_size = torch.Size(
                [self.env_batch_size * self.aug_factor, self.pomo_size]
            )
            self.env_batch_size = self.env_batch_size * self.aug_factor
            self.problems = self.problems.repeat(self.aug_factor, 1, 1)
            
            # initial tours
        if self.method_name=="matpoenet":
            
            problems  = self.problems
            batch_size = self.problems.size(0)
            self.node_cnt = self.problems.size(1)
            
            solver = BaseSolver(n=self.node_cnt)
            if self.init_solver == None:
                pass
            elif self.init_solver == "nn":
                f_solver = solver.solve_nearest_neighbor
            elif self.init_solver == "lkh":
                f_solver = solver.solve_lkh
            elif self.init_solver == "fi":
                f_solver = solver.solve_farthest_insertion
            else:
                raise NotImplementedError("Solver {} is not implemented.".format(self.init_solver))
            if self.init_solver:
                for i in range(batch_size):
                    f_solver(problems[i])
                    init_tour = np.array(solver.get_path(), dtype=np.int64)
                    problems[i] = problems[i][init_tour, :][:, init_tour]
                    
            self.problems = problems
            
        self.BATCH_IDX = torch.arange(self.env_batch_size)[:, None].expand(
            self.env_batch_size, self.pomo_size
        )
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.env_batch_size, self.pomo_size
        )
        assert self.env_batch_size == self.problems.size(0), (
            "batch_size and the first dimension of problems should be the same. "
            + f"Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}"
        )
        
        if self.method_name == "matpoenet":
            self.pos_emb = make_positional_encoding_cosh_recur(self.pos_embedding_dim, self.node_cnt, scaler=100)
            self.pos_emb = self.pos_emb.unsqueeze(0).repeat(self.env_batch_size, 1, 1)

    def _reset(self, td: TensorDict, batch_size=None) -> TensorDict:
        
        self.output_spec=None  # Solve the inconsistent batch size
        self.pomo_size = self.node_cnt
        
        self.selected_count = torch.zeros(
            (self.env_batch_size, self.pomo_size), dtype=torch.long
        )
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.empty(
            (self.env_batch_size, self.pomo_size, 0), dtype=torch.long
        )
        # shape: (batch, pomo, 0~)
        self.ninf_mask = torch.zeros(
            (self.env_batch_size, self.pomo_size, self.node_cnt)
        )
        # shape: (batch, pomo, problem)
        # for done
        self.dummy_flag_bool = torch.zeros(
            (self.env_batch_size, self.pomo_size), dtype=torch.bool
        )
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = (
            torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long) - 1
        )
        
        if self.method_name == "matpoenet":
            return TensorDict(
            {
                "problems": self.problems,
                "pos_emb": self.pos_emb,
            },
            batch_size=torch.Size([self.env_batch_size]),
        )

        return TensorDict(
            {
                "problems": self.problems
            },
            batch_size=torch.Size([self.env_batch_size]),
        )

    def pre_step(self) -> TensorDict:
            
            
        return TensorDict(
            {
                "ninf_mask": self.ninf_mask,
                "batch_idx": self.BATCH_IDX,
                "pomo_idx": self.POMO_IDX,
                "current_node": self.dummy_flag_long,
                "reward": self.dummy_flag_long,
                "done": self.dummy_flag_bool,
            },
            batch_size=self.batch_size,
        )
    def _step(self, td: TensorDict) -> TensorDict:
        # node_idx.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = td["action"]
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )
        # shape: (batch, pomo, 0~node)

        # update step state
        ninf_mask = td["ninf_mask"]
        ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float("-inf")
        # shape: (batch, pomo, node)

        # returning values
        done = self.selected_count == self.node_cnt
        
        if done.all():
            reward = (
                -self._get_total_distance()
            )  # Note the MINUS Sign ==> We MAXIMIZE reward
            # shape: (batch, pomo)
        else:
            
            reward = self.dummy_flag_long
        next_state = {
            "ninf_mask": self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
            "reward": reward,
            "done": done,
        }
        return TensorDict(
            {
                "next": next_state,
                "ninf_mask": ninf_mask,
                "batch_idx": self.BATCH_IDX,
                "pomo_idx": self.POMO_IDX,
                "reward": reward,
                "done": done,
                "current_node": self.current_node,
            },
            batch_size=self.batch_size,
        )

    def _get_total_distance(self):

        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(
            self.env_batch_size, self.pomo_size, self.node_cnt
        )
        # shape: (batch, pomo, node)

        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state["rng"])