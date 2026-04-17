import torch
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import *
from EasyNCO.data import CVRPGenerator
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec

logger = getLogger(__name__)

class MOCVRPEnv(EnvBase):
    def __init__(self,
                 problem_size :int,
                 pomo_size : int = 1, # multi_start trajectory, in AM-based models, it is 1 by default
                 device : str = 'cpu',
                 seed : int = 2024,
                 **kwargs):
        super().__init__(device = device)
        self.env_name = 'mocvrp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device
        self.depot_node_xy = None

        self.problems = None
        self.capacity = None

        # Dynamic
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.at_the_depot = None
        self.load = None
        self.ninf_mask = None
        self.visited_ninf_flag = None
        self.finished = None

        self.dummy_flag_bool = None
        self.dummy_flag_long = None

        self.seed_value = seed
        self._set_seed(seed = seed)

        self.method_name = kwargs.get('method_name')

        self.aug_type = kwargs.get('aug_type')
        self.aug_factor = kwargs.get('aug_factor',1)
        self.mix_prop = kwargs.get('mix_prop')
        self.aug_flag = self.aug_type is not None and self.aug_factor > 1

        # LEHD
        self.use_flag_to_indicate_depot = kwargs.get('use_flag_to_indicate_depot')

        # INVIT
        self.raw_depot_node_xy = None # shape (batch, 1 + problem, 2)

        # UDC
        self.sample_size = kwargs.get("sample_size", None)
        self.sub_problem_size = kwargs.get("sub_problem_size", None)
        self.flag = None

        # PSL
        self.pref = kwargs.get('pref')
        self.num_target = len(self.pref) if self.pref is not None else 1

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def generate_eval_instances(self, num_instances : int, batch_size : int, data_path: str = None) -> DataLoader:
        '''
        Note that this function is necessary for the AM-based models.
        If you are using the AM-based models, you should implement this function according to specific CO problems.
        The function generates the evaluation instances for the CO problem.
        It is used to evaluate the trained model using greedyRollout baseline in AM-based models.
        :param num_instances:
        :param batch_size:
        :return: DataLoader
        '''
        eval_dataset = CVRPGenerator(num_instances if self.aug_type is None else num_instances//self.aug_factor,
                                    self.problem_size, batch_size, self.device, data_path)

        return eval_dataset

    def load_problems(self, dataset, batch_size : int = 64):

        self.problems = dataset
        # problem.shape: (batch, 1 + problem, 3)

        self.raw_depot_node_xy = dataset[:, :, :2]
        # shape (batch, 1 + problem, 2)
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        if self.aug_flag is True:
            self.batch_size = torch.Size([self.env_batch_size * self.aug_factor, self.pomo_size])
            self.env_batch_size = self.env_batch_size * self.aug_factor
            aug_operator = {'pomo_aug': augment_pomo,
                            'rotate': augment_rotate,
                            'reflect': augment_reflect,
                            'mix': augment_mix,
                            'noise': augment_noise}
            if self.aug_type not in aug_operator:
                raise NotImplementedError(f'Augment type {self.aug_type} is not supported.')
            self.depot_node_xy = aug_operator[self.aug_type](problems=self.problems, aug_factor=self.aug_factor, mix_prop=self.mix_prop)
            self.depot_node_demand = self.problems[:, :, 2].repeat(self.aug_factor, 1) # shape: (batch, 1 + problem)
            self.problems = torch.cat((self.depot_node_xy, self.depot_node_demand[:, :, None]), dim=-1) # shape (batch, 1 + problem, 3)
            if self.aug_type != 'pomo_aug':
                self.problems[:self.batch_size[0]//self.aug_factor, :, :2] = self.raw_depot_node_xy
                self.depot_node_xy[:self.batch_size[0]//self.aug_factor] = self.raw_depot_node_xy
        else:
            self.depot_node_xy = self.problems[:, :, :2] # shape: (batch, 1 + problem, 2)
            self.depot_node_demand = self.problems[:, :, 2] # shape: (batch, 1 + problem)

        assert self.env_batch_size == self.depot_node_xy.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.depot_node_xy.size(0)}, got: {self.env_batch_size}')

    def _reset(self, td : TensorDict, batch_size = None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.current_node = None
        # shape: (batch, pomo)
        if self.use_flag_to_indicate_depot:
            self.selected_count = torch.ones((self.env_batch_size, self.pomo_size), dtype=torch.long)
            self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0, 2), dtype=torch.long)
            # shape: (batch, pomo, 0~, 2)
        else:
            self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
            self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype = torch.long)
            # shape: (batch, pomo, 0~)

        self.ninf_mask = torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem + 1)

        self.at_the_depot = torch.ones(size = (self.env_batch_size, self.pomo_size), dtype = torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size = (self.env_batch_size, self.pomo_size))
        self.capacity = torch.ones(size = (self.env_batch_size, self.pomo_size, 1))
        # self.capacity = self.load.unsqueeze(2)
        self.used_capacity = torch.zeros(size = (self.env_batch_size, self.pomo_size, 1))
        self.visited_ninf_flag = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem + 1)

        # for done
        self.dummy_flag_bool = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.bool)
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.long) - 1
        self.finished = torch.zeros(size = (self.env_batch_size, self.pomo_size), dtype = torch.bool)
        # shape: (batch, pomo)


        return TensorDict({
            'action' : self.dummy_flag_long,  # 'action': 'int',  # shape: (batch, pomo)
            'locs' : self.problems,
            'ninf_mask' : self.ninf_mask,
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
            'vehicle_capacity' : self.capacity,
            'used_capacity' : self.used_capacity,
            'reward' : self.dummy_flag_long, #*
            'done' : self.dummy_flag_bool,
        }, batch_size = torch.Size([self.env_batch_size])
        )

    def pre_step(self) -> TensorDict:

        next_state = {
            'ninf_mask' : self.ninf_mask,
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
        }
        out = TensorDict({
            'action' : self.dummy_flag_long,
            'first_node' : self.dummy_flag_long,
            'vehicle_capacity' : self.capacity,
            'used_capacity' : self.used_capacity,
            'next' : next_state,
            'reward' : self.dummy_flag_long,
            'done' : self.dummy_flag_bool,
        }, batch_size = self.batch_size
        )

        return out

    def _step(self, td : TensorDict,**kwargs) -> TensorDict:

        self.selected_count += 1
        self.current_node = td['action']
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim = 2)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = (self.current_node == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.env_batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem + 1)
        gathering_index = self.current_node[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim = 2, index = gathering_index).squeeze(dim = 2)
        # shape: (batch, pomo)
        # load handling
        self.load -= selected_demand
        self.used_capacity += selected_demand.unsqueeze(2)
        self.load[self.at_the_depot] = 1 # refill loaded at the depot
        self.used_capacity[self.at_the_depot] = 0 # clear used_capacity at the depot

        # mask handling
        self.visited_ninf_flag.scatter_(dim = -1, index = self.current_node.unsqueeze(-1), value = float('-inf'))
        # shape: (batch, pomo, problem + 1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0 # depot is considered unvisited, unless you are at the depot
        self.ninf_mask = self.visited_ninf_flag.clone()

        round_error_epsilon = 0.00001

        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim = 2)
        # shape: (batch, pomo)

        self.finished = self.finished + newly_finished

        # do not mask depot for finished episode
        self.ninf_mask[:, :, 0][self.finished] = 0

        done_all = self.finished.all()

        if done_all:
            # whether solution is valid
            assert (self.visited_ninf_flag == float('-inf')).all(), \
                    'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'
            reward = -self._get_travel_distance()
        else:
            reward = self.dummy_flag_long
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(reward.shape)
        )
        next_state = {
            'ninf_mask' : self.ninf_mask,
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
            "reward": reward,
            "done": self.finished,
        }

        out = TensorDict({
            'action' : self.current_node,
            'next' : next_state,
            'vehicle_capacity' : self.capacity,
            'used_capacity' : self.used_capacity,
            'reward' : reward,
            'done' : self.finished,
            'load' : self.load,
        }, batch_size = self.batch_size
        )

        return out

    def __getstate__(self):
        """
        Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state['rng'] = state['rng'].get_state()

        return state

    def __setstate__(self, state):
        """
        Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state['rng'])

    def _get_travel_distance(self):

        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        # obj1: travel_distances
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)

        # obj2: makespans
        not_idx = (gathering_index[:, :, :, 0] > 0)
        cum_lengths = torch.cumsum(segment_lengths, dim=2)

        cum_lengths[not_idx] = 0
        sorted_cum_lengths, _ = cum_lengths.sort(axis=2)

        rolled_sorted_cum_lengths = sorted_cum_lengths.roll(dims=2, shifts=1)
        diff_mat = sorted_cum_lengths - rolled_sorted_cum_lengths
        diff_mat[diff_mat < 0] = 0

        makespans, _ = torch.max(diff_mat, dim=2)

        objs = torch.stack([travel_distances, makespans], axis=2)

        return objs


    def _get_travel_distance_use_flag(self):
        tour = self.selected_node_list[:, :, :, 0].clone() # shape (batch, pomo, problem)
        flag = self.selected_node_list[:, :, :, 1].clone() # shape (batch, pomo, problem)
        problem = self.depot_node_xy[:, None, :, :].clone().expand(-1, self.pomo_size, -1, -1)
        index_not_at_depot = torch.le(flag, 0.5)
        index_at_depot = torch.gt(flag, 0.5)

        flag[index_not_at_depot] = tour[index_not_at_depot]
        flag[index_at_depot] = 0
        roll_tour = tour.roll(dims=-1, shifts=1)

        tour_gather_index = tour.unsqueeze(-1).expand(-1, -1, -1, 2)
        tour_loc = problem.gather(dim=2, index=tour_gather_index)

        roll_gather_index = roll_tour.unsqueeze(-1).expand(-1, -1, -1, 2)
        roll_tour_loc = problem.gather(dim=2, index=roll_gather_index)

        flag_gather_index = flag.unsqueeze(-1).expand(-1, -1, -1, 2)
        flag_loc = problem.gather(dim=2, index=flag_gather_index)

        tour_lengths = ((tour_loc - flag_loc) ** 2)
        flag[:, :, 0] = 0
        roll_lengths = ((roll_tour_loc - flag_loc) ** 2)
        distance = (tour_lengths.sum(-1).sqrt() + roll_lengths.sum(-1).sqrt()).sum(-1)

        return distance





