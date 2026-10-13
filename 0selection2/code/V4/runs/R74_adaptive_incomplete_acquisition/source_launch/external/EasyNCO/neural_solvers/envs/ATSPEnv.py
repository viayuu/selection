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

class ASHPPEnv(EnvBase):

    def __init__(self,
                 problem_size: int=20,
                 pomo_size: int = 1,  # multi_start trajectory, in AM-based models, it is 1 by default
                 device: str = 'cpu',
                 seed: int = 2024,
                 **kwargs):
        super().__init__(device=device)

        self.env_name = 'ashpp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device
        self.aug_type = kwargs.get('aug_type')
        self.aug_factor = kwargs.get('aug_factor',1)
        self.seed_value = seed
        self.saved_problem = None
        self.saved_index = None
        self.FLAG__use_saved_problems = False
        self.problems = None
        self.selected_count = None
        self.current_node = None
        self.ninf_mask = None
        self.selected_node_list = None
        #self.first_node = None
        self.dummy_flag_bool = None
        self.dummy_flag_long = None
        self._set_seed(seed=seed)



    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def load_problems(self, dataset, batch_size: int = 64, aug_factor: int = 1, algorithm_type="am", avg_type="mix",
                      aug_all=False):
        self.problems = dataset
        # shape: (batch, node, node)
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])
        assert self.env_batch_size == self.problems.size(0), (
            "batch_size and the first dimension of problems should be the same. "
            + f"Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}"
        )
    def _reset(self,td: TensorDict,batch_size=None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)

        self.selected_count += 2 # Add starting and terminating ndoes

        self.current_node = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
        self.last_node= torch.ones((self.env_batch_size, self.pomo_size), dtype=torch.long)* (self.problem_size - 1)
        self.first_node=torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.selected_node_list = self.current_node[:, :, None]
        # self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)
        # CREATE STEP STATE
        self.ninf_mask = torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size))
        self.ninf_mask[:,:,0]=float("-inf")
        self.ninf_mask[:,:,-1]=float("-inf")

        # shape: (batch, pomo, problem)

        # for done
        self.dummy_flag_bool = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.bool)
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long) - 1

        return TensorDict({
            'action': self.current_node,  # 'action': 'int',  # shape: (batch, pomo)
            'locs': self.problems,
            'ninf_mask': self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
            "reward": self.dummy_flag_long,
            "done": self.dummy_flag_bool,
        }, batch_size=torch.Size([self.env_batch_size])
        )
    def pre_step(self) -> TensorDict:
        next_state = {
            'ninf_mask': self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
        }
        out = TensorDict({
            'action': self.current_node,  # 'action': 'int',  # shape: (batch, pomo)
            "first_node": self.dummy_flag_long,  # "first_node": "int",  # shape: (batch, pomo)
            "next": next_state,
            "last_node":self.last_node,
            "reward": self.dummy_flag_long,
            "done": self.dummy_flag_bool,
        }, batch_size=self.batch_size
        )

        return out

    def _step(self, td: TensorDict) -> TensorDict:
        self.selected_count += 1
        self.current_node = td["action"]
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)


        self.ninf_mask.scatter_(dim=-1, index=self.current_node.unsqueeze(-1), value=float('-inf'))
        # shape: (batch, pomo, problem)

        # returning values
        done = (self.selected_count == self.problem_size)
        done_all = done.all()

        if done_all:
            # judge whether solution is valid.
            assert (self.ninf_mask == float('-inf')).all(), \
                'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'
            self.current_node = torch.ones((self.env_batch_size, self.pomo_size), dtype=torch.long) * (self.problem_size - 1)
            self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = self.dummy_flag_long

        next_state = {
            'ninf_mask': self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
            "reward": reward,
            "done": done,
            }

        out = TensorDict({
            'action': self.current_node,  # 'action': 'int',  # shape: (batch, pomo)
            "first_node": self.first_node,  # "first_node": "int",  # shape: (batch, pomo)
            "next": next_state,
            "last_node": self.last_node,
            "reward": reward,
            "done": done,
        }, batch_size=self.batch_size
        )


        return out


    def _get_travel_distance(self):

        node_from = self.selected_node_list[:,:,:-1]
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)[:, :, :-1]
        batch_index  = torch.arange(self.env_batch_size).repeat(self.pomo_size, 1).transpose(0, 1)[:,:,None].expand(self.env_batch_size, self.pomo_size, self.problem_size-1)

        # shape: (batch, pomo, node)

        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)


        return total_distance



    def use_saved_problems_test(self, filename):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=self.device)
        self.saved_node_xy = loaded_dict['problems']
        self.saved_index = 0

    def generate_eval_instances(self, num_instances: int, batch_size: int) -> DataLoader:
        '''
        Note that this function is necessary for the AM-based models.
        If you are using the AM-based models, you should implement this function according to specific CO problems.
        The function generates the evaluation instances for the CO problem.
        It is used to evaluate the trained model using greedyRollout baseline in AM-based models.
        :param num_instances:
        :param batch_size:
        :return: DataLoader
        '''
        eval_dataset = ATSPGenerator(num_instances, self.problem_size, batch_size, self.device)
        return eval_dataset




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




class ATSPEnv(EnvBase):

    def __init__(
        self,
        pomo_size: int = 20,
        device: str = "cpu",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(device=device)
        self.env_name = "atsp"

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
        self.problem_size = self.problems.shape[1]
        self.node_cnt = self.problems.shape[1]
        
        self.env_batch_size = batch_size

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
                "problems": self.problems,
            },
            batch_size=torch.Size([self.env_batch_size]),
        )

    def pre_step(self) -> TensorDict:
        
        if self.method_name == "matpoenet":
            
            return TensorDict(
            {
                "ninf_mask": self.ninf_mask,
                "batch_idx": self.BATCH_IDX,
                "pomo_idx": self.POMO_IDX,
                "current_node": self.dummy_flag_long,
                "reward": self.dummy_flag_long,
                "done": self.dummy_flag_bool,
                "pos_emb": self.pos_emb,
            },
            batch_size=self.batch_size,
        )
            
        else:
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