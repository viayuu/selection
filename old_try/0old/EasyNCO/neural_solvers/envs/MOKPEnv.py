import torch
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import *
from EasyNCO.data import MOKPGenerator
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec


logger = getLogger(__name__)

class MOKPEnv(EnvBase):
    def __init__(self,
                 problem_size : int,
                 pomo_size : int = 1,
                 device : str = 'cpu',
                 seed : int = 2024,
                 **kwargs):
        super().__init__(device = device)
        self.env_name = 'mokp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device

        self.item_data = None
        self.items_and_a_dummy = None

        self.current_node = None
        self.selected_count = None
        self.selected_node_list = None
        self.ninf_mask_w_dummy = None
        self.ninf_mask = None
        self.fit_ninf_mask = None
        self.dummy_flag_long = None

        self.accumulated_value = None
        self.capacity = None
        self.max_capacity = None
        self.first_node = None

        self.FLAG__use_saved_problems = True
        self.saved_problems = None
        self.saved_index = 0
        self.optimal = 0

        self.finished = None
        self.problems_values = None
        self.problems_weights = None

        self.seed_value = seed
        self._set_seed(seed = seed)

        # PSL
        self.aug_factor = kwargs.get('aug_factor', 1)
        self.pref = kwargs.get('pref')
        self.num_target = len(self.pref) if self.pref is not None else 1

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)

        self.rng = rng

    def use_saved_problems_test(self, filename, capacity):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location = self.device)
        self.saved_item_data = loaded_dict['problems']
        # shape: (batch, problem, 2)
        self.optimal = loaded_dict['optimal']
        self.max_capacity = torch.ones((self.saved_problems.size(0), self.pomo_size)) * capacity
        # shape: (batch, pomo)
        self.saved_index = 0


    def generate_eval_instances(self, num_instances : int, batch_size : int) -> DataLoader:
        '''
        Note that this function is necessary for the AM-based models.
        If you are using the AM-based models, you should implement this function according to specific CO problems.
        The function generates the evaluation instances for the CO problem.
        It is used to evaluate the trained model using greedyRollout baseline in AM-based models.
        :param num_instances:
        :param batch_size:
        :return: DataLoader
        '''
        eval_dataset = MOKPGenerator(num_instances, self.problem_size, batch_size, self.device)

        return eval_dataset

    def load_problems(self, dataset, batch_size : int = 64, aug_factor : int = 1):

        self.problems = dataset
        # problems.shape: (batch, problem, 2)
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        # Set capacity
        if self.problem_size == 50:
            cap = 12.5
        elif self.problem_size == 100:
            cap = 25
        elif self.problem_size == 200:
            cap = 25
        else:
            raise NotImplementedError(
                f'problem size: {self.problem_size} is not supported, expected values are 50, 100, or 200.')

        self.capacity = torch.ones((self.env_batch_size, self.pomo_size)) * cap

        if aug_factor > 1:

                raise NotImplementedError('augmentation is not supported for kp')

        assert self.env_batch_size == self.problems.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.problems.size(0)}, got: {self.env_batch_size}')


    def _reset(self, td : TensorDict, batch_size = None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.items_and_a_dummy = torch.zeros((self.env_batch_size, self.problem_size + 1, 3))
        self.items_and_a_dummy[:, : self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, : self.problem_size, :]
        # shape: (batch, problem, 2)

        self.current_node = None
        self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.long)
        self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype = torch.long)
        # shape: (batch, pomo, 0~problem)
        self.ninf_mask_w_dummy = torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem + 1)
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, : self.problem_size]
        # shape: (batch, pomo, problem)

        self.accumulated_value = torch.zeros((self.env_batch_size, self.pomo_size))
        # shape: (batch, pomo)

        self.fit_ninf_mask = None
        self.finished = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.bool)
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.long)
        # shape: (batch, pomo)
        self.dummy_flag_long_ = torch.zeros((self.env_batch_size, self.pomo_size, self.num_target), dtype = torch.long)
        self.accumulated_value_obj1 = torch.zeros((self.env_batch_size, self.pomo_size))
        self.accumulated_value_obj2 = torch.zeros((self.env_batch_size, self.pomo_size))

        return TensorDict({
            'action' : self.dummy_flag_long, # action " int, # shape: (batch, pomo)
            'items' : self.problems,
            'capacity' : self.capacity,
            'ninf_mask' : self.ninf_mask,
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
            'reward' : self.dummy_flag_long_,
            'done' : self.finished
        }, batch_size = torch.Size([self.env_batch_size])
        )

    def pre_step(self) -> TensorDict:

        next_state = {
            'ninf_mask' : torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size)),
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
        }

        out = TensorDict({
            'action' : self.dummy_flag_long,
            'capacity' : self.capacity,
            # 'first_node' : self.dummy_flag_long,
            'next' : next_state,
            'reward' : self.dummy_flag_long_,
            'done' : self.finished,
        }, batch_size = self.batch_size
        )

        return out

    def _step(self, td : TensorDict) -> TensorDict:
        self.selected_count += 1

        self.current_node = td['action']

        # selected = self.current_node.clamp(0, self.problem_size - 1)
        selected = self.current_node

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim = 2)
        # shape: (batch, pomo, 0~problem)

        items_mat = self.items_and_a_dummy[:, None, :, :].expand((
            self.batch_size[0], self.pomo_size, self.problem_size + 1, 3
        ))
        gathering_index = selected[:, :, None, None].expand(self.batch_size[0], self.pomo_size, 1, 3)
        selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo, 2) weight and value

        # index 0: weight, index 1: value
        self.accumulated_value_obj1 += selected_item[:, :, 1]
        self.accumulated_value_obj2 += selected_item[:, :, 2]
        self.capacity -= selected_item[:, :, 0]

        assert (self.capacity >= 0).all(), "The knapsack is overloaded"

        self.ninf_mask_w_dummy.scatter_(dim = -1, index = self.current_node.unsqueeze(-1), value = float('-inf'))
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]

        # whether there exists a node's weight larger than the capacity of the knapsack, if so, set it to -inf
        unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) < 0
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = float('-inf')
        # shape: (batch, pomo, problem)

        self.finished = (self.fit_ninf_mask == float('-inf')).all(dim = 2)

        self.fit_ninf_mask[self.finished[:, :, None].expand(self.env_batch_size, self.pomo_size, self.problem_size)] = 0
        # do not mask finished episode

        done = self.finished.all()
        if done:
            assert (self.ninf_mask == float('-inf')).all, \
                'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'
            reward = torch.stack([self.accumulated_value_obj1,self.accumulated_value_obj2],axis = 2)
            # shape: (batch, pomo)
        else:
            reward = self.dummy_flag_long_

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(reward.shape)
        )

        next_state = {
            'ninf_mask' : self.fit_ninf_mask,
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
            'reward' : reward,
            'done' : self.finished
        }

        out = TensorDict({
            'action' : self.current_node,
            'capacity' : self.capacity,
            # 'first_node' : self.first_node,
            'next' : next_state,
            'reward' : reward,
            'done' : self.finished,
        }, batch_size = self.batch_size
        )

        return out

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
            the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state['rng'] = state['rng'].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state['rng'])

