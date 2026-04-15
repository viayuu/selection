import numpy as np
import torch
from tensordict import TensorDict
from EasyNCO.utils.utils import _get_encoding

'''
This file contains classes needed to train partial solution models.
It is used to construct partial solution dataset and update the needed data when training.
For example, when training LEHD-TSP, the model needs destination node, last node and available nodes.
The file is used to adapt Lightning module.

In partial updator, we will handle data of all partial problems. 
Specifically speaking, if we plan to train 300 episodes in one epoch and the batch size is 100, then we have 3 batches in one epoch.
Suppose that the lengths of these 3 batches of partial problems are 13, 15, 17,  We will sample the sub-problems with corresponding lengths
and concatenate them together in the dimension of node lengths and store them in the updator.(Variables prefixed with 'all_')
During training, we fetch the data of the corresponding batch and put it in the variable prefixed with 'cur_batch_'
'''
def partial_construction_updater(problem_type: str, config: dict):
    """
    Get partial solution construction tool, it is typically used to update the corresponding variables.

    Args:
        problem_type (str): Type of problem, it can be 'tsp' or 'cvrp'
    """
    partial_construction_updater_registry = {
        'tsp': TSPPartialUpdater,
        'cvrp': CVRPPartialUpdater
    }

    if problem_type not in partial_construction_updater_registry.keys():
        raise ValueError(
            f"Unknown environment name '{problem_type}'. Available partial construction updaters: {partial_construction_updater_registry.keys()}"
        )

    return partial_construction_updater_registry[problem_type](**config)

class TSPPartialUpdater:
    def __init__(
        self,
        problem_size: int = 100,
        batch_size: int = 512,
        train_episodes: int = 12800,
        device: str = 'cpu'
    ):
        self.problem_size = problem_size
        self.batch_size = torch.Size([batch_size, 1]) # 1 is pomo_size, used to align dimension
        self.train_episodes = train_episodes
        self.node_num = None  # Record the length of partial solutions in each batch.
        self.batch_flag = -1  # Indicate which batch is in training
        self.device = device

        self.all_partial_nodes = None  # Record all partial solution nodes of the current epoch. shape (batch, pomo, all_partial_num, 2)
        self.all_partial_tours = None  # Record all partial solutions of the current epoch. shape (batch, pomo, all_partial)

        self.cur_batch_nodes = None  # Record partial solution nodes of the current batch. shape (batch, pomo, partial_num, 2)
        self.cur_batch_tours = None  # Record partial solutions of the current batch. shape (batch, pomo, partial_num)
        self.selected_count = None # shape (batch, pomo)
        self.cur_batch_done = True  # Indicate whether current batch is done.

        self.cur_batch_selected_list = None # shape (batch, pomo, current_step)
        self.cur_batch_prob_list = None # shape (batch, pomo, current_step)

        self.finished_nodes = 0  # It is used to count all trained nodes, eventually used to get data of next batch.

    def load_problem_one_epoch(self, dataset):
        '''
        The function is used to construct partial problem dataset.
        Returns:
            loader_data is just a list used as a counter adapted to logger.
        '''
        self.reset_one_epoch_partial()
        i = 0
        loader_data = []
        shuffle_data(dataset)
        for node_num in self.node_num:
            temp_node, temp_tour = sample('tsp', dataset.data, self.problem_size, node_num, self.batch_size, i)

            self.all_partial_nodes = torch.cat((self.all_partial_nodes, temp_node), dim=-2)
            self.all_partial_tours = torch.cat((self.all_partial_tours, temp_tour), dim=-1)
            loader_data += list(range(node_num - 2))
            i += 1
        return loader_data

    def pre_set(self):
        '''
        It is used to initialize the corresponding variables when needed.
        For example, when training LEHD model, we need to get the start node and destination node before training a batch.
        '''
        if self.cur_batch_done:
            self.reset_one_batch_partial()

        if self.selected_count[0,0] < 2:
            selected = self.cur_batch_tours[:, :, -1]  # destination node
            prob = torch.ones(self.batch_size[0], self.batch_size[1])
            self.update(selected, prob)

            selected = self.cur_batch_tours[:, :, 0]  # starting node
            prob = torch.ones(self.batch_size[0], self.batch_size[1])
            self.update(selected, prob)

    def reset_one_epoch_partial(self):
        '''
        For partial solution model, the data for training one epoch is different, so we need to reset corresponding variables.
        '''
        self.node_num = torch.randint(low=4, high=self.problem_size + 1, size=(int(self.train_episodes / self.batch_size[0]),))
        self.batch_flag = -1
        self.all_partial_nodes = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0, 2))
        self.all_partial_tours = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0), dtype=torch.long)

        self.cur_batch_nodes = None
        self.cur_batch_tours = None
        self.selected_count = None
        self.cur_batch_done = True

        self.cur_batch_selected_list = None
        self.cur_batch_prob_list = None

        self.finished_nodes = 0

    def reset_one_batch_partial(self):
        '''
        For partial solution model, the data for training one batch is different, so we need to reset corresponding variables (such as the length of partial solution).
        '''
        self.selected_count = torch.zeros(size=self.batch_size, dtype=torch.int)
        self.batch_flag += 1
        self.cur_batch_prob_list = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0))
        self.cur_batch_done = False

        self.cur_batch_selected_list = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0), dtype=torch.long)

        self.cur_batch_nodes = self.all_partial_nodes[:, :, self.finished_nodes:self.finished_nodes + self.node_num[self.batch_flag], :]
        self.cur_batch_tours = self.all_partial_tours[:, :, self.finished_nodes:self.finished_nodes + self.node_num[self.batch_flag]]
        self.finished_nodes += self.node_num[self.batch_flag]

    def update(self, selected, prob):
        '''
        Update some variables after finding next node.
        selected.shape: (batch, 1)
        prob.shape: (batch, 1)
        '''
        self.selected_count += 1
        self.cur_batch_selected_list = torch.cat((self.cur_batch_selected_list, selected[:, :, None]), dim=-1)  # shape: (batch, pomo, selected_count)
        self.cur_batch_prob_list = torch.cat((self.cur_batch_prob_list, prob[:, :, None]), dim=-1)
        self.cur_batch_done = (self.selected_count == self.node_num[self.batch_flag]).all()

    def get_current_td(self):
        next_state = {"selected_count":self.selected_count,
                      "selected_node_list":self.cur_batch_selected_list}
        out = TensorDict({
            "next_select": self.cur_batch_tours[:, :, self.selected_count[0, 0]-1],
            "locs": self.cur_batch_nodes,
            "next": next_state,
            "partial_length": torch.zeros(size=self.batch_size, dtype=torch.int) + self.node_num[self.batch_flag]
        },batch_size=self.batch_size
        )
        return out

class CVRPPartialUpdater:
    def __init__(
        self,
        problem_size: int = 100,
        batch_size: int = 512,
        train_episodes: int = 12800,
        device: str = 'cpu'
    ):
        self.problem_size = problem_size
        self.batch_size = torch.Size([batch_size, 1]) # 1 is pomo_size, used to align dimension
        self.train_episodes = train_episodes
        self.node_num = None # Record the length of partial solutions in each batch.
        self.batch_flag = -1 # Indicate which batch is in training
        self.device = device

        self.all_partial_node_xy_demand = None # Record all partial solution nodes and their demand of the current epoch.
        self.all_partial_tour_flag = None # Record all partial solutions of the current epoch.
        self.all_partial_remain_capacity = None # Record all remain capacity before training a batch. shape (batch, pomo, batch_num)

        self.cur_batch_node_xy_demand = None # Record partial solution nodes of the current batch. shape (batch, pomo, 1 + partial_length, 3)
        self.cur_batch_tour_flag = None # Record partial solutions of the current batch. shape (batch, pomo, partial_length, 3)
        self.cur_batch_remain_capacity = None # Record remain capacity of the current batch. shape (batch, pomo)
        self.selected_count = None
        self.cur_batch_done = True # Indicate whether current batch is done.

        self.cur_batch_selected_list = None
        self.cur_batch_prob_list = None

        self.finished_nodes = 0 # It is used to count all trained nodes, eventually used to get data of next batch.

    def load_problem_one_epoch(self, dataset):
        '''
        The function is used to construct partial problem dataset.
        Returns:
            loader_data is just a list used as a counter adapted to logger.
        '''
        self.reset_one_epoch_partial()
        i = 0
        loader_data = []
        shuffle_data(dataset)
        for node_num in self.node_num:
            temp_node_xy_demand, temp_tour_flag, temp_remain_capacity = sample('cvrp', dataset.data, self.problem_size, node_num, self.batch_size, i)

            self.all_partial_node_xy_demand = torch.cat((self.all_partial_node_xy_demand, temp_node_xy_demand), dim=-2)
            self.all_partial_tour_flag = torch.cat((self.all_partial_tour_flag, temp_tour_flag), dim=-2)
            self.all_partial_remain_capacity = torch.cat((self.all_partial_remain_capacity, temp_remain_capacity), dim=-1)
            loader_data += list(range(node_num-1))
            i += 1
        return loader_data

    def pre_set(self):
        '''
        It is used to initialize the corresponding variables when needed.
        For example, when training LEHD model, we need to get the start node and destination node before training a batch.
        '''
        if self.cur_batch_done:
            self.reset_one_batch_partial()

        if self.selected_count[0, 0] == 0:
            selected = self.cur_batch_tour_flag[:, :, 0, :]
            prob = torch.ones(self.batch_size[0], self.batch_size[1])
            self.update(selected, prob)

    def reset_one_epoch_partial(self):
        '''
        For partial solution model, the data for training one epoch is different, so we need to reset corresponding variables.
        '''
        self.node_num = torch.randint(low=4, high=self.problem_size + 1, size=(int(self.train_episodes / self.batch_size[0]),))
        self.batch_flag = -1
        self.all_partial_node_xy_demand = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0, 3))
        self.all_partial_tour_flag = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0, 2), dtype=torch.long)
        self.all_partial_remain_capacity = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0))

        self.cur_batch_node_xy_demand = None
        self.cur_batch_tour_flag = None
        self.cur_batch_remain_capacity = None
        self.selected_count = None
        self.cur_batch_done = True

        self.cur_batch_selected_list = None
        self.cur_batch_prob_list = None

        self.finished_nodes = 0

    def reset_one_batch_partial(self):
        '''
        For partial solution model, the data for training one batch is different, so we need to reset corresponding variables (such as the length of partial solution).
        '''
        self.selected_count = torch.zeros(size=self.batch_size, dtype=torch.int)
        self.batch_flag += 1
        self.cur_batch_prob_list = torch.zeros(size=(self.batch_size[0], self.batch_size[1], 0))
        self.cur_batch_selected_list = torch.zeros((self.batch_size[0], self.batch_size[1], 0, 2), dtype=torch.long)
        self.cur_batch_done = False

        self.cur_batch_node_xy_demand = self.all_partial_node_xy_demand[:, :, self.finished_nodes+self.batch_flag:self.finished_nodes+self.batch_flag+self.node_num[self.batch_flag]+1, :] #shape (batch, pomo, 1 + partial_length, 3)
        self.cur_batch_tour_flag = self.all_partial_tour_flag[:, :, self.finished_nodes:self.finished_nodes + self.node_num[self.batch_flag], :] #shape (batch, pomo, partial_length, 2)
        self.cur_batch_remain_capacity = self.all_partial_remain_capacity[:, :, self.batch_flag] #shape (batch, pomo)
        self.finished_nodes += self.node_num[self.batch_flag]

    def update(self, selected, prob):
        '''
        Update some variables after finding next node.
        selected.shape: (batch, pomo, 2)
        prob.shape: (batch, pomo)
        '''
        self.selected_count += 1
        gather_index = selected[:, :, [0]]

        ### Update capacity
        # First, the vehicle returns that to depot will refill the capacity.
        self.cur_batch_remain_capacity[selected[:, :, 1] == 1] = 1 # shape (batch, pomo)

        # Second, subtract the demand of the currently visited node from remain capacity
        demand = self.cur_batch_node_xy_demand[...,2].gather(index=gather_index, dim=2).squeeze(-1) # shape (batch, pomo)
        self.cur_batch_remain_capacity -= demand

        ### Update
        self.cur_batch_selected_list = torch.cat((self.cur_batch_selected_list, selected[:, :, None, :]), dim=-2)
        self.cur_batch_prob_list = torch.cat((self.cur_batch_prob_list, prob[:, :, None]), dim=-1)
        self.cur_batch_done = (self.selected_count == self.node_num[self.batch_flag]).all()

    def get_current_td(self):
        next_state = {"selected_count": self.selected_count,
                      "selected_node_list": self.cur_batch_selected_list}
        out = TensorDict({
            "next_select": self.cur_batch_tour_flag[:, :, self.selected_count[0, 0]],
            "locs": self.cur_batch_node_xy_demand,
            "next": next_state,
            "remain_capacity": self.cur_batch_remain_capacity,
            "partial_length": torch.zeros(size=self.batch_size, dtype=torch.int) + self.node_num[self.batch_flag]
        }, batch_size=self.batch_size
        )
        return out

def shuffle_data(dataset):
    '''
    It is used to shuffle data of one epoch before constructing partial solution dataset.
    '''
    shuffle_index = torch.randperm(dataset.num_sample).long()
    for key, value in dataset.data.items():
        dataset.data[key] = value[shuffle_index]

def VRP_segment_inverse_and_shuffle(solution):
    '''
    Inverse and shuffle VRP segments in one episode. Typically used in SL paradigm.
    '''
    clockwise_or_not: object = torch.rand(1)[0]
    batch_size = solution.shape[0]
    problem_size = solution.shape[1]
    if clockwise_or_not >= 0.5:
        solution = torch.flip(solution, dims=[1])
        index = torch.arange(solution.shape[1]).roll(shifts=1)
        solution[:, :, 1] = solution[:, index, 1]

    # 1.
    # find the number of subtours in each instance.
    # the total number of subpaths in all instances:     all_subtour_num，
    # The longest length in a subpath among all instances:  max_subtour_length
    visit_depot_num = torch.sum(solution[:, :, 1], dim=1)
    all_subtour_num = torch.sum(visit_depot_num)
    fake_solution = torch.cat((solution[:, :, 1], torch.ones(batch_size)[:, None]), dim=1)
    start_from_depot = fake_solution.nonzero()
    start_from_depot_1 = start_from_depot[:, 1]
    start_from_depot_2 = torch.roll(start_from_depot_1, shifts=-1)
    sub_tours_length = start_from_depot_2 - start_from_depot_1
    max_subtour_length = torch.max(sub_tours_length)

    # 2。
    # For each subpath, take it out separately, pandding 0 to lengt    h max_subtour_length
    # For each instance, padding 0 to max_subtour_num number of subpaths
    # Put all subpaths of all instances into the same array
    start_from_depot2 = solution[:, :, 1].nonzero()
    start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()
    repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)
    double_repeat_solution_node = repeat_solutions_node.repeat(1, 2)
    x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) >= start_from_depot2[:, 1][:, None]
    x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) <= start_from_depot3[:, 1][:, None]
    x3 = (x1 * x2).long()
    sub_tourss = double_repeat_solution_node * x3
    x4 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node), 1) < (start_from_depot2[:, 1][:, None] + max_subtour_length)
    x5 = x1 * x4
    sub_tours_padding = sub_tourss[x5].reshape(all_subtour_num, max_subtour_length)

    # 3.
    # For each row, a random number of [0,100] is generated, greater than 50 is positive and less than 50 is inverse
    clockwise_or_not = torch.rand(len(sub_tours_padding))
    clockwise_or_not_bool = clockwise_or_not.le(0.5)

    # 4.
    # For each row, randomly flip
    sub_tours_padding[clockwise_or_not_bool] = torch.flip(sub_tours_padding[clockwise_or_not_bool], dims=[1])

    # 5.
    # Map the subtours to the original solution matrix dimension
    sub_tourss_back = sub_tourss
    sub_tourss_back[x5] = sub_tours_padding.ravel()
    solution_node_flip = sub_tourss_back[sub_tourss_back.gt(0.1)].reshape(batch_size, problem_size)
    solution = torch.cat((solution_node_flip.unsqueeze(2), solution[:, :, 1].unsqueeze(2)), dim=2)

    return solution

def sample(problem_type, data, problem_size, partial_length, Batch_Size, batch_index):
    '''
    It is used to sample partial solution and its corresponding nodes of a batch.
    Args:
        problem_type: Type of problem.
        data: A dict contains nodes of problem and its tour.
            nodes.shape:(batch, problem_size, feature_dim),
            if tour.shape(batch, pomo, problem_size)
        problem_size: Size of problem.
        partial_length: Length of partial problem.
        Batch_Size: torch.Size([batch, pomo])
        batch_index: Indicate which batch is being sampled.

    Returns:
        if problem is tsp:
            new_data: sampled nodes.
            new_solution_rank: sampled tours.
        if problem is cvrp:
            new_node_xy_demand: sampled nodes with their demands.
            new_tour_flag: sampled tours with their flags
            raw_node_xy_demand[:, [0], :]: depot
            remain_capacity: For partial solution, it starts without full capacity.
    '''
    batch_size = Batch_Size[0]
    pomo_size = Batch_Size[1]
    if problem_type == 'tsp':
        raw_node = data['node'][batch_size * batch_index: batch_size * (batch_index + 1)][:, None, :, :].expand(-1, pomo_size, -1, -1)
        raw_tour = data['label'][batch_size * batch_index: batch_size * (batch_index + 1)][:, None, :].expand(-1, pomo_size, -1)
        node_dim = raw_node.shape[-1]
        first_node_index = torch.randint(low=0, high=problem_size, size=[pomo_size], dtype=torch.long)  # shape (pomo)
        # -----------------------------
        # new_tour
        # -----------------------------
        double_solution = torch.cat([raw_tour, raw_tour], dim=-1)
        batch_expand = torch.arange(batch_size)[:, None, None].expand(-1, pomo_size, partial_length)
        pomo_expand = torch.arange(pomo_size)[None, :, None].expand(batch_size, -1, partial_length)
        partial_expand = torch.arange(partial_length)[None, None, :].expand(batch_size, pomo_size,-1) + first_node_index[None, :,None].expand(batch_size, -1,partial_length)
        new_solution = double_solution[batch_expand, pomo_expand, partial_expand]
        new_solution_ascending, rank = torch.sort(new_solution, dim=-1, descending=False)  # 升序
        new_solution_rank = torch.sort(rank, dim=-1, descending=False)[1]  # 升序
        # -----------------------------
        # new_node
        # -----------------------------
        batch_expand = torch.arange(batch_size)[:, None, None, None].expand(-1, pomo_size, partial_length, node_dim)  # shape: (batch, pomo, partial_length, node_dim)
        pomo_expand = torch.arange(pomo_size)[None, :, None, None].expand(batch_size, -1, partial_length, node_dim) # shape: (batch, pomo, partial_length, node_dim)
        partial_expand = new_solution_ascending[:, :, :, None].expand(-1, -1, -1, node_dim)
        node_dim_expand = torch.arange(node_dim)[None, None, None, :].expand(batch_size, pomo_size, partial_length, -1)
        new_data = raw_node[batch_expand, pomo_expand, partial_expand, node_dim_expand]

        return new_data, new_solution_rank

    elif problem_type == 'cvrp':
        raw_node_xy_demand = data['node_xy_demand'][batch_size * batch_index: batch_size * (batch_index + 1)][:, None, :, :].expand(-1, pomo_size, -1, -1)
        raw_tour_flag = data['label'][batch_size * batch_index: batch_size * (batch_index + 1)]

        # data augment
        raw_tour_flag = VRP_segment_inverse_and_shuffle(raw_tour_flag)[:, None, :, :].expand(-1, pomo_size, -1, -1)

        ### Calculate the partial solution
        # First, choose the finished node of the partial solution, the finished node must end with depot
        start_from_depot = raw_tour_flag[:, :, :, 1].nonzero()
        end_with_depot = start_from_depot
        end_with_depot[:, 2] = end_with_depot[:, 2] - 1
        end_with_depot[end_with_depot.le(-0.5)] = problem_size - 1

        visit_depot_num = torch.sum(raw_tour_flag[:, :, :, 1], dim=2, dtype=torch.float32).reshape(-1).unsqueeze(0) #shape (1, batch * pomo)
        select_end_with_depot_node_index = torch.floor(torch.mul(torch.rand((1, batch_size*pomo_size)), visit_depot_num)).squeeze().long()
        temp_index = torch.mm(visit_depot_num, torch.tensor(np.triu(np.ones((batch_size*pomo_size, batch_size*pomo_size)), k=1),dtype=torch.float32)).squeeze().long()
        select_end_with_depot_node_index += temp_index
        select_end_with_depot_node = end_with_depot[select_end_with_depot_node_index, 2] #shape (batch * pomo)

        # Second, calculate the partial solution end with the finished node chosen last step
        double_tour_flag = torch.cat((raw_tour_flag, raw_tour_flag), dim=-2)
        select_end_with_depot_node += problem_size
        start_point = (select_end_with_depot_node - partial_length + 1).reshape(batch_size, pomo_size) #shape (batch, pomo)
        index_of_new_tour_flag = (start_point[:, :, None] + torch.arange(partial_length)[None, None, :].expand(batch_size, pomo_size, -1))[:, :, :, None].expand(-1, -1, -1, 2) #shape (batch, pomo, partial, 2)
        new_tour_flag = double_tour_flag.gather(dim=2, index=index_of_new_tour_flag)

        # Third, calculate the capacity of the first point
        start_index = index_of_new_tour_flag[:, :, 0, 0] #shape (batch, pomo)
        temp = torch.arange(problem_size * 2) <= start_index[:, :, None]
        tour_flag_before = double_tour_flag[:, :, :, 1] * temp
        start_with_depot_index_before = tour_flag_before.nonzero()
        visit_depot_num_before = torch.sum(tour_flag_before, dim=2, dtype=torch.float32).reshape(-1).unsqueeze(0) #shape (1, batch * pomo)
        select_end_with_depot_node_index_before = (visit_depot_num_before - 1).reshape(-1).long()
        temp_index = torch.mm(visit_depot_num_before, torch.tensor(np.triu(np.ones((batch_size*pomo_size, batch_size*pomo_size)), k=1),dtype=torch.float32)).squeeze().long()
        select_end_with_depot_node_index_before += temp_index
        node_index_previous_to_partial_tour = start_with_depot_index_before[select_end_with_depot_node_index_before, 2].reshape(batch_size, pomo_size) #shape (batch, pomo)

        segment_tour_previous_to_partial_tour = (torch.arange(problem_size * 2) < start_index[:, :, None]) * (torch.arange(problem_size * 2) >= node_index_previous_to_partial_tour[:, :, None])
        double_demand = raw_node_xy_demand[:, :, :, 2][torch.arange(batch_size)[:, None, None].expand(-1, pomo_size, problem_size * 2),
                                                       torch.arange(pomo_size)[None, :, None].expand(batch_size, -1, problem_size * 2),
                                                       double_tour_flag[:, :, :, 0]]
        satisfy_demand = (double_demand * segment_tour_previous_to_partial_tour).sum(2) # shape (batch, pomo)
        remain_capacity = (1 - satisfy_demand)[:, :, None] #shape (batch, pomo, 1)

        ### Update the partial solution's index
        sub_solution = new_tour_flag[:, :, :, 0]
        new_solution_ascending, rank = torch.sort(sub_solution, dim=-1, descending=False)  # 升序
        new_solution_rank = torch.sort(rank, dim=-1, descending=False)[1]  # 升序
        new_tour_flag[:, :, :, 0] = new_solution_rank + 1
        new_node_xy_demand = torch.cat((raw_node_xy_demand[:, :, [0], :], _get_encoding(raw_node_xy_demand, new_solution_ascending)), dim=-2)

        return new_node_xy_demand, new_tour_flag, remain_capacity