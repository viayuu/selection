import torch
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.utils.utils import getLogger
from EasyNCO.data.data_utils import *
from EasyNCO.data import CVRPGenerator
from EasyNCO.neural_solvers.methods.lih.utils import lih_reset


logger = getLogger(__name__)

class CVRPEnv(EnvBase):
    def __init__(self,
                 problem_size :int,
                 pomo_size : int = 1, # multi_start trajectory, in AM-based models, it is 1 by default
                 device : str = 'cpu',
                 seed : int = 2024,
                 **kwargs):
        super().__init__(device = device)
        self.env_name = 'cvrp'
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
        self.rrc = False

        # INVIT
        self.raw_depot_node_xy = None # shape (batch, 1 + problem, 2)

        # UDC
        self.sample_size = kwargs.get("sample_size", None)
        self.sub_problem_size = kwargs.get("sub_problem_size", None)
        self.flag = None

        # ELG
        self.dist = None

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
        if self.method_name == "udc":
            self.problem_size = self.problems.size(1) - 1
            if self.flag is not None:
                self.demand_last = self.problems[:, -1, 2].clone()
                self.demand_last[self.flag] = 0
        self.raw_depot_node_xy = dataset[:, :, :2]
        # shape (batch, 1 + problem, 2)
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        if self.aug_flag is True:
            self.pomo_size = min(self.pomo_size, 5000)  # due to memory issue, we limit the pomo size to 5000 when adopting augmentation
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

        # ELG
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)

    def _reset(self, td : TensorDict, batch_size = None) -> TensorDict:
        self.output_spec = None  # Solve the inconsistent batch size
        self.current_node = None
        # shape: (batch, pomo)
        if self.use_flag_to_indicate_depot:
            self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
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
        if self.method_name != "udc" and not self.rrc:
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
        if self.method_name == "udc":
            self.last_mask = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size + 1))
            self.last_mask[:, :, -1][
                (self.flag == 0)[:, None].expand(-1, self.pomo_size)
            ] = float("-inf")
            self.solution_list = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size + 1), dtype = torch.long)
            self.solution_flag = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size + 1), dtype = torch.long)
            self.udc_solution_list = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size), dtype = torch.long)
            self.udc_solution_flag = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size), dtype = torch.long)
            self.node_count = -1 * torch.ones(size = (self.env_batch_size, self.pomo_size, 1), dtype = torch.long)

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

    def _step(self, td : TensorDict, **kwargs) -> TensorDict:
        if self.use_flag_to_indicate_depot:
            # if self.selected_node_list.size(-2) != 0:
            #     self.selected_count += 1
            self.selected_count += 1
            self.current_node = td['action']
            # shape: (batch, pomo, 2)
            if self.selected_node_list.size(-2) == 0 and not self.rrc:
                self.current_node[:, :, 1] = 1
            demand_list = self.depot_node_demand[:, None, :].expand(self.env_batch_size, self.pomo_size, -1)
            # shape: (batch, pomo, problem + 1)
            gathering_index = self.current_node[:, :, [0]]
            # shape: (batch, pomo, 1)
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            # shape: (batch, pomo)
            round_error_epsilon = 0.00001
            demand_too_large = self.load + round_error_epsilon < selected_demand
            # shape: (batch, pomo)
            # the flags of first node and the nodes whose demand is too large is 1
            self.at_the_depot = (self.current_node[:, :, 1] == 1) | (demand_too_large == True)
            self.current_node[:, :, 1][self.at_the_depot] = 1
            self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None, :]), dim=-2)
            self.load[self.at_the_depot] = 1 # refill loaded at the depot
            self.used_capacity[self.at_the_depot[:, :, None]] = 0  # clear used_capacity at the depot
            self.load -= selected_demand
            assert (self.load >= - round_error_epsilon).all()
            assert (self.load <= 1).all()
            self.used_capacity += selected_demand[:, :, None]

            self.finished = (self.selected_count == self.problems.shape[1] - 1)
            done_all = self.finished.all()
            if done_all:
                # reward = -self._get_travel_distance_2(self.selected_node_list.squeeze(1), self.problems[:, :, :2])
                reward = -self._get_travel_distance_use_flag(self.selected_node_list, self.problems[:, :, :2])
            else:
                reward = self.dummy_flag_long

            next_state = {
                'selected_count': self.selected_count,
                'selected_node_list': self.selected_node_list,
                'reward': reward,
                'done': self.finished
            }
            out = TensorDict({
                'action': self.current_node,
                'next': next_state,
                'remain_capacity': self.load,
                'partial_length': torch.zeros(size=self.batch_size, dtype=torch.int) + self.problems.shape[1] - 1,
                'reward': reward,
                'done': self.finished
            }, batch_size=self.batch_size
            )

        else:
            round_error_epsilon = 0.00001
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
            assert (self.load >= - round_error_epsilon).all()
            self.used_capacity += selected_demand.unsqueeze(2)
            self.load[self.at_the_depot] = 1 # refill loaded at the depot
            self.used_capacity[self.at_the_depot] = 0 # clear used_capacity at the depot

            # mask handling
            self.visited_ninf_flag.scatter_(dim = -1, index = self.current_node.unsqueeze(-1), value = float('-inf'))
            # shape: (batch, pomo, problem + 1)
            self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0 # depot is considered unvisited, unless you are at the depot
            self.ninf_mask = self.visited_ninf_flag.clone()

            if self.method_name == "udc":
                if (self.selected_count > 1).all():
                    self.solution_flag = self.solution_flag.scatter_add(
                        dim=-1,
                        index=self.node_count,
                        src=(td["action"][:, :, None] == 0).long(),
                    )
                self.node_count[td["action"][:, :, None] != 0] += 1
                self.solution_list = self.solution_list.scatter_add(
                    dim=-1, index=self.node_count, src=td["action"][:, :, None]
                )

                condition_mask = (
                    (self.visited_ninf_flag[:, :, 1:] == float("-inf")).sum(-1)
                    < self.problem_size - 1
                ) | (
                    1 - self.load + self.demand_last[:, None]
                    > self.left + round_error_epsilon
                )
                self.ninf_mask[
                    condition_mask[:, :, None].expand_as(self.ninf_mask)
                ] += self.last_mask[
                    condition_mask[:, :, None].expand_as(self.ninf_mask)
                ]


            demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
            # shape: (batch, pomo, problem+1)
            self.ninf_mask[demand_too_large] = float('-inf')
            # shape: (batch, pomo, problem+1)

            newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim = 2)
            # shape: (batch, pomo)
            if self.method_name == "udc":
                newly_finished = (self.visited_ninf_flag[:, :, 1:] == float('-inf')).all(dim=2) & (1 - self.load < self.left + round_error_epsilon) & ~self.finished
                # shape: (batch, pomo)
                self.udc_solution_list[newly_finished] = self.solution_list[:, :, :-1][
                    newly_finished
                ]
                self.udc_solution_flag[newly_finished] = self.solution_flag[:, :, :-1][newly_finished]
            self.finished = self.finished + newly_finished

            # do not mask depot for finished episode
            self.ninf_mask[:, :, 0][self.finished] = 0

            done_all = self.finished.all()

            if done_all:
                if self.method_name == "udc":
                    new_solution = torch.cat(
                        (
                            self.udc_solution_list.unsqueeze(-1),
                            self.udc_solution_flag.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                    reward = -self.cal_open_length(
                        self.depot_node_xy,
                        new_solution[:, :, :, 0],
                        new_solution[:, :, :, 1],
                    )
                else:
                    # whether solution is valid
                    assert (self.visited_ninf_flag == float('-inf')).all(), \
                            'The selected nodes should be masked with -inf. It means that the certain solutions are not valid.'
                reward = -self._get_travel_distance()
            else:
                reward = self.dummy_flag_long

            next_state = {
                'ninf_mask' : self.ninf_mask,
                'selected_count' : self.selected_count,
                'selected_node_list' : self.selected_node_list,
                "reward": reward,
                "done": self.finished,
            }

            if self.method_name == 'elg' and self.current_node is not None:
                local_feature = self._get_local_feature()
                out = TensorDict({
                    'action' : self.current_node,
                    'next' : next_state,
                    'vehicle_capacity' : self.capacity,
                    'used_capacity' : self.used_capacity,
                    'local_feature': local_feature,
                    'reward' : reward,
                    'done' : self.finished,
                    'load' : self.load,
                }, batch_size = self.batch_size
                )

            else:
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
        selected_node_list = self.selected_node_list
        gathering_index = selected_node_list.unsqueeze(3).expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        if self.aug_flag is True:
            all_xy = self.raw_depot_node_xy[:, None, :, :].repeat(self.aug_factor, self.pomo_size, 1, 1)
        else:
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem + 1, 2)

        ordered_seq = all_xy.gather(dim = 2, index = gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims = 2, shifts = -1)
        segment_length = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_length.sum(2)
        # shape(batch, pomo)

        return travel_distances

    def _get_travel_distance_use_flag(self, selected_node_list = None, depot_node_xy = None):
        if selected_node_list == None:
            selected_node_list = self.selected_node_list
        if depot_node_xy == None:
            depot_node_xy = self.problems[:, :, : 2]


        tour = selected_node_list[:, :, :, 0].clone() # shape (batch, pomo, problem)
        flag = selected_node_list[:, :, :, 1].clone() # shape (batch, pomo, problem)
        problem = depot_node_xy[:, None, :, :].clone().expand(-1, self.pomo_size, -1, -1)
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

    def validate_solution_legal(self, problem, solution):

        round_error_epsilon = 0.00001
        problem_size = solution.shape[1]
        capacity = 1

        coor = problem[:, :, [0, 1]]
        demand = problem[:, :, 2]

        order_node = solution[:, :, 0].clone()
        order_flag = solution[:, :, 1].clone()

        if_begin_flag_legal = (order_flag[:,0]!=1).any()

        # 0.
        if if_begin_flag_legal:
            assert False, 'e1: wrong begin_flag_legal!'

        # 1. Determine whether each index of the solution node list is unique
        uniques = torch.unique(order_node[0])
        if len(uniques) != problem.shape[1] - 1:
            assert False, 'e2: wrong node list!'


        # 2. Find the demand for each sub tour and determine whether it exceeds capacity

        batch_size = solution.shape[0]


        visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

        all_subtour_num = torch.sum(visit_depot_num)

        fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)

        start_from_depot = fake_solution.nonzero()

        start_from_depot_1 = start_from_depot[:, 1]

        start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)

        sub_tours_length = start_from_depot_2 - start_from_depot_1

        max_subtour_length = torch.max(sub_tours_length)

        # 2。
        start_from_depot2 = solution[:, :, 1].nonzero()
        start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

        repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
        double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)

        x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             >= start_from_depot2[:, 1][:, None]
        x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             <= start_from_depot3[:, 1][:, None]

        x3 = (x1 * x2).long()

        sub_tourss = double_repeat_solution_node * x3

        x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) \
             < (start_from_depot2[:, 1][:, None] + max_subtour_length)

        x5 = x1 * x4

        sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

        ########################----------
        ########################----------

        demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

        index = torch.arange(sub_tours_padding.shape[0])[:, None].repeat(1, sub_tours_padding.shape[1])
        sub_tours_demands = demands[index, sub_tours_padding].sum(dim=1)
        if_legal = (sub_tours_demands > capacity + round_error_epsilon)

        if if_legal.any():
            assert False, 'e3: wrong capacity!'

        return

    def _get_travel_distance_2(self, solution_, problems_,):

        problems = problems_[:, :, [0, 1]].clone()
        order_node = solution_[:, :, 0].clone()
        order_flag = solution_[:, :, 1].clone()
        travel_distances = self.cal_length(problems, order_node, order_flag)

        travel_distances = travel_distances.unsqueeze(1)
        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)

        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)

        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length

    def cal_open_length(self, problems, order_node, order_flag, total_flag=False, last_flag=False):
        if total_flag:
            order_node_ = order_node[None, :, :].clone()
            order_flag_ = order_flag[None, :, :].clone()
        else:
            order_node_ = order_node.clone()
            order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(
            order_node_.size(0), order_node_.size(1), -1
        )
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)
        if last_flag:
            travel_distances = segment_lengths.sum(2)
        else:
            travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def _get_local_feature(self):
        if self.current_node is None:
            return None, None, None, None
        current_node = self.current_node[:, :, None, None].expand(self.env_batch_size, self.pomo_size, 1,
                                                                  self.problem_size + 1)
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.env_batch_size, self.pomo_size, self.problem_size + 1, self.problem_size + 1),
                                        current_node, dim=2).squeeze(2)
        expanded_xy = self.depot_node_xy[:, None, :, :].expand(self.env_batch_size, self.pomo_size, self.problem_size + 1, 2)
        relative_xy = expanded_xy - torch.take_along_dim(expanded_xy, self.current_node[:, :, None, None].expand(
            self.env_batch_size, self.pomo_size, 1, 2), dim=2)

        relative_x = relative_xy[:, :, :, 0]
        relative_y = relative_xy[:, :, :, 1]

        cur_theta = torch.atan2(relative_y, relative_x)
        demand_list = self.depot_node_demand[:, None, :].expand(self.env_batch_size, self.pomo_size, -1)
        norm_demand = demand_list / self.load[:, :, None]

        local_feature = {
            'cur_dist' : cur_dist,
            'cur_theta' : cur_theta,
            'relative_xy' : relative_xy,
            'norm_demand' : norm_demand,
        }
        return local_feature