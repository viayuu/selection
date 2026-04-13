from dataclasses import dataclass
import torch
import os
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import augment_xy_data_by_64_fold_2obj,augment_xy_data_by_n_fold_3obj
from EasyNCO.data import MOTSPGenerator
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec

logger = getLogger(__name__)


class MOTSPEnv(EnvBase):
    # batch_locked = False
    def __init__(self,
                 problem_size: int,
                 pomo_size: int = 1,  # multi_start trajectory, in AM-based models, it is 1 by default
                 device: str = 'cpu',
                 seed: int = 2024,
                 **kwargs):
        super().__init__(device=device)

        self.env_name = 'motsp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device

        self.problems = None

        self.selected_count = None
        self.current_node = None
        self.ninf_mask = None
        self.selected_node_list = None
        # self.first_node = None
        self.dummy_flag_bool = None
        self.dummy_flag_long = None

        self.seed_value = seed

        self._set_seed(seed=seed)
        self.method_name = kwargs.get('method_name')

        self.aug_type = kwargs.get('aug_type', None)
        self.aug_factor = kwargs.get('aug_factor',1)
        self.mix_prop = kwargs.get('mix_prop')
        self.aug_flag = self.aug_type is not None and self.aug_factor > 1
        self.distribution = kwargs.get("distribution","uniform")
        # PSL
        self.pref = kwargs.get('pref')
        self.num_target = len(self.pref) if self.pref is not None else 1
        self.batch_size = torch.Size([1])
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.params = {
            "num_target": self.num_target
        }
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1, self.pomo_size, 2)
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def generate_eval_instances(self, num_instances: int, batch_size: int, data_path: str = None) -> DataLoader:
        '''
        Note that this function is necessary for the AM-based models.
        If you are using the AM-based models, you should implement this function according to specific CO problems.
        The function generates the evaluation instances for the CO problem.
        It is used to evaluate the trained model using greedyRollout baseline in AM-based models.
        :param num_instances:
        :param batch_size:
        :return: DataLoader
        '''
        eval_dataset = MOTSPGenerator(num_instances if self.aug_type is None else num_instances//self.aug_factor,
                                    self.problem_size, batch_size, self.device, data_path,distribution=self.distribution,**self.params)
        return eval_dataset

    def load_problems(self, dataset, batch_size: int = 64):
        self.problems = dataset
        # problems.shape: (batch, problem, 2)
        self.raw_problems = dataset
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        if self.aug_flag is True:
            self.batch_size = torch.Size([self.env_batch_size * self.aug_factor, self.pomo_size])
            self.env_batch_size = self.env_batch_size * self.aug_factor
            if self.num_target == 2 and self.aug_factor == 64:
                self.problems = augment_xy_data_by_64_fold_2obj(problems=self.problems)
            elif self.num_target == 3:
                self.problems = augment_xy_data_by_n_fold_3obj(problems=self.problems, aug_factor=self.aug_factor)

        assert self.env_batch_size == self.problems.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}')

        #MOEAD

        self.BATCH_IDX = torch.arange(self.env_batch_size)[:, None].expand(self.env_batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.env_batch_size, self.pomo_size)


    def _reset(self,td: TensorDict,batch_size=None) -> TensorDict:
        self.output_spec=None  # Solve the inconsistent batch size

        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)
        # CREATE STEP STATE
        self.ninf_mask = torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)
        # for done
        self.dummy_flag_bool = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.bool)
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size,self.num_target), dtype=torch.long) - 1
        self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)

        return TensorDict({
            'action': self.dummy_flag_long,  # 'action': 'int',  # shape: (batch, pomo)
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
            'action': self.dummy_flag_long,  # 'action': 'int',  # shape: (batch, pomo)
            "first_node": self.dummy_flag_long,  # "first_node": "int",  # shape: (batch, pomo)
            "next": next_state,
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

        # record the first node in the trajectory
        if self.selected_count[0, 0] == 1:
            self.first_node = self.current_node

        self.ninf_mask.scatter_(dim=-1, index=self.current_node.unsqueeze(-1), value=float('-inf'))
        # shape: (batch, pomo, problem)

        # returning values
        done = (self.selected_count == self.problem_size)
        done_all = done.all()

        if done_all:
            # judge whether solution is valid.
            assert (self.ninf_mask == float('-inf')).all(), \
                'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'

            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = self.dummy_flag_long

        # print(self.batch_size)
        # reward = reward.view(-1, self.pomo_size, self.num_target)
        reward = reward.view(-1, self.pomo_size, self.num_target)
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(reward.shape)
        )
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
            "reward": reward,
            "done": done,
        }, batch_size=self.batch_size
        )

        return out

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

    def _get_travel_distance(
        self,
        problems=None,
        selected_node_list=None,
        batch_size=None,
        pomo_size=None,
        open_flag=False
    ):
        if problems is None:
            problems = self.problems
        if selected_node_list is None:
            selected_node_list = self.selected_node_list
        if batch_size is None:
            batch_size = self.env_batch_size
        if pomo_size is None:
            pomo_size = self.pomo_size
        gathering_index = selected_node_list.unsqueeze(3).expand(batch_size, -1, self.problem_size, 2*self.num_target)
        # shape: (batch, pomo, problem, 2*num_target)
        seq_expanded = problems[:, None, :, :].expand(
                batch_size, pomo_size, self.problem_size, 2*self.num_target
            )

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2*num_target)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths_obj1 = ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
        segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:] - rolled_seq[:, :, :, 2:]) ** 2).sum(3).sqrt()
        if self.num_target == 3:
            segment_lengths_obj1 = ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
            segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:4] - rolled_seq[:, :, :, 2:4]) ** 2).sum(3).sqrt()
            segment_lengths_obj3 = ((ordered_seq[:, :, :, 4:] - rolled_seq[:, :, :, 4:]) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        if open_flag:
            travel_distances_obj1 = segment_lengths_obj1[:, :, :-1].sum(2)
            travel_distances_obj2 = segment_lengths_obj2[:, :, :-1].sum(2)
            if self.num_target == 3:
                travel_distances_obj1 = segment_lengths_obj1.sum(2)
                travel_distances_obj2 = segment_lengths_obj2.sum(2)
                travel_distances_obj3 = segment_lengths_obj3.sum(2)
        else:
            travel_distances_obj1 = segment_lengths_obj1.sum(2)
            travel_distances_obj2 = segment_lengths_obj2.sum(2)
            if self.num_target == 3:
                travel_distances_obj1 = segment_lengths_obj1.sum(2)
                travel_distances_obj2 = segment_lengths_obj2.sum(2)
                travel_distances_obj3 = segment_lengths_obj3.sum(2)

        # shape: (batch, pomo)

        travel_distances_vec = torch.stack([travel_distances_obj1, travel_distances_obj2], axis=2)
        if self.num_target == 3:
            travel_distances_vec = torch.stack([travel_distances_obj1, travel_distances_obj2, travel_distances_obj3],
                                               axis=2)

        return travel_distances_vec



