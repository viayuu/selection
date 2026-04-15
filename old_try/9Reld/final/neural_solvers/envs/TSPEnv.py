import torch
import os
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import *
from EasyNCO.data import TSPGenerator


logger = getLogger(__name__)

class TSPEnv(EnvBase):
    # batch_locked = False
    def __init__(self,
                 problem_size: int,
                 pomo_size: int = 1,  # multi_start trajectory, in AM-based models, it is 1 by default
                 device: str = 'cpu',
                 seed: int = 2024,
                 **kwargs):
        super().__init__(device=device)

        self.env_name = 'tsp'
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
        self.distribution=kwargs.get("distribution","uniform")
        # ELG
        self.dist = None

        # INVIT
        self.raw_problems = None
        # sub_glop
        # glop
        self.sub_glop=kwargs.get('sub_glop')
        # udc
        self.sample_size = kwargs.get('sample_size', None)
        self.sub_problem_size = kwargs.get('sub_problem_size', None)
        # RRC
        self.solution = []

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
        eval_dataset = TSPGenerator(num_instances if self.aug_type is None else num_instances//self.aug_factor,
                                    self.problem_size, batch_size, self.device, data_path,distribution=self.distribution)
        return eval_dataset

    def load_problems(self, dataset, batch_size: int = 64):
        self.problems = dataset
        # problems.shape: (batch, problem, 2)
        if self.method_name == 'udc':
            self.problem_size = self.problems.size(1)
        self.raw_problems = dataset
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        if self.aug_flag is True:
            self.pomo_size = min(self.pomo_size, 5000) # due to memory issue, we limit the pomo size to 5000 when adopting augmentation
            self.batch_size = torch.Size([self.env_batch_size * self.aug_factor, self.pomo_size])
            self.env_batch_size = self.env_batch_size * self.aug_factor
            aug_operator = {'pomo_aug': augment_pomo,
                            'rotate': augment_rotate,
                            'reflect': augment_reflect,
                            'mix': augment_mix,
                            'noise': augment_noise}
            if self.aug_type not in aug_operator:
                raise NotImplementedError(f'Augment type {self.aug_type} is not supported.')
            self.problems = aug_operator[self.aug_type](problems=self.problems, aug_factor=self.aug_factor, mix_prop=self.mix_prop)
            if self.aug_type != 'pomo_aug':
                self.problems[:self.batch_size[0]//self.aug_factor] = self.raw_problems

        assert self.env_batch_size == self.problems.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}')
        # ELG
        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p = 2, dim = -1).to(self.device)
        # (batch, problem, problem)

    def _reset(self,td: TensorDict,batch_size=None) -> TensorDict:
        self.output_spec=None  # Solve the inconsistent batch size

        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)
        # CREATE STEP STATE
        self.ninf_mask = torch.zeros((self.env_batch_size, self.pomo_size, self.problems.shape[1]))
        # shape: (batch, pomo, problem)
        # for done
        self.dummy_flag_bool = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.bool)
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long) - 1
        self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
        if self.method_name == 'udc':
            self.ninf_mask[:, : self.pomo_size // 2, -1] = float("-inf")
            self.ninf_mask[:, self.pomo_size // 2 :, 0] = float("-inf")
            self.selected_count += 1
            self.first_node = self.dummy_flag_long

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
        if self.method_name == 'elg' and self.current_node is not None:
            local_feature = self._get_local_feature()

            out = TensorDict({
                'action': self.dummy_flag_long,  # 'action': 'int',  # shape: (batch, pomo)
                "first_node": self.dummy_flag_long,  # "first_node": "int",  # shape: (batch, pomo)
                "next": next_state,
                'local_feature' : local_feature,
                "reward": self.dummy_flag_long,
                "done": self.dummy_flag_bool,
            }, batch_size=self.batch_size
            )
        else:
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
        done = (self.selected_count == self.problems.
                shape[1])
        done_all = done.all()

        if done_all:
            # judge whether solution is valid.
            assert (self.ninf_mask == float('-inf')).all(), \
                'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'
            if self.method_name == 'udc':
                self.selected_node_list = torch.cat(
                    (
                        self.selected_node_list,
                        torch.cat(
                            (
                                self.problem_size
                                - torch.ones(self.pomo_size // 2)[None, :].expand(
                                    self.env_batch_size, self.pomo_size // 2
                                ),
                                torch.zeros(self.pomo_size // 2)[None, :].expand(
                                    self.env_batch_size, self.pomo_size // 2
                                ),
                            ),
                            dim=-1,
                        ).long()[:, :, None],
                    ),
                    dim=2,
                )
                reward = -self._get_travel_distance(selected_node_list = self.selected_node_list,
                                                    problems = self.problems,
                                                    open_flag=True)
            else:
                reward = -self._get_travel_distance(selected_node_list = self.selected_node_list,
                                                    problems = self.problems)  # note the minus sign!
        else:
            reward = self.dummy_flag_long

        next_state = {
            'ninf_mask': self.ninf_mask,
            "selected_count": self.selected_count,
            "selected_node_list": self.selected_node_list,
            "reward": reward,
            "done": done,
            }

        if self.method_name == 'elg' and self.current_node is not None:
            local_feature = self._get_local_feature()
            out = TensorDict({
                'action': self.current_node,  # 'action': 'int',  # shape: (batch, pomo)
                "first_node": self.first_node,  # "first_node": "int",  # shape: (batch, pomo)
                "next": next_state,
                'local_feature': local_feature,
                "reward": reward,
                "done": done,
            }, batch_size=self.batch_size
            )
        elif self.method_name == 'lehd':
            out = TensorDict({
                'action': self.current_node,  # 'action': 'int',  # shape: (batch, pomo)
                "first_node": self.first_node,  # "first_node": "int",  # shape: (batch, pomo)
                "next": next_state,
                "partial_length": torch.zeros(self.batch_size, dtype=torch.int) + self.problems.shape[1],
                "reward": reward,
                "done": done,
            }, batch_size=self.batch_size
            )

        else:
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
        # gathering_index = selected_node_list.unsqueeze(3).expand(batch_size, -1, self.problem_size, 2)

        problem_size = problems.shape[1]

        gathering_index = selected_node_list.unsqueeze(3).expand(batch_size, -1, problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        if self.aug_flag is True:
            seq_expanded = self.raw_problems[:, None, :, :].repeat(self.aug_factor, pomo_size, 1, 1)
        else:
            seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        if open_flag:
            travel_distances = segment_lengths[:, :, :-1].sum(2)
        else:
            travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def _get_local_feature(self):
        if self.current_node == None:
            return None, None, None

        current_node = self.current_node[:, :, None, None].expand(self.env_batch_size, self.pomo_size, 1, self.problem_size)
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.env_batch_size, self.pomo_size, self.problem_size, self.problem_size),
                                        current_node, dim = 2).squeeze(2)
        # shape: (batch, pomo, problem)
        expanded_xy = self.problems[:, None, :, :].expand(self.env_batch_size, self.pomo_size, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        relative_xy = expanded_xy - torch.take_along_dim(expanded_xy, self.current_node[:, :, None, None].expand(
            self.env_batch_size, self.pomo_size, 1, 2
        ), dim = 2)
        # shape: (batch, pomo, problem, 2)

        relative_x = relative_xy[:, :, :, 0]
        relative_y = relative_xy[:, :, :, 1]

        cur_theta = torch.atan2(relative_y, relative_x)
        # shape: (batch, pomo, problem)

        local_feature = {
            'cur_dist' : cur_dist,
            'cur_theta' : cur_theta,
            'relative_xy' : relative_xy
        }

        return local_feature








