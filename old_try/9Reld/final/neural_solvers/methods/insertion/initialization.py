from typing import Any, Tuple, Literal
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from EasyNCO.neural_solvers.pipeline.initialization import Initialization
from EasyNCO.utils.utils import *

class INSERTIONInitialization(Initialization):
    """
    This class creates the initial solution using the insertion method.
    Currently supports only TSP and CVRP problems.
    If you want to initialize the solution using 'INSERTIONInitialization',
    please add the following parameter keys to {yours}_setting.yaml:
    - problem: ${problem}
    - insert_strategy: 'random'/'nearest'/'farthest'/'median'/'regret'(TSP only)
    - metric_strategy: 'cartesian'/'polar'
    - p: {int}       # Cartesian distance norm (optional)
    - K: {int}       # Regret order (optional)
    """
    def __init__(self, policy, **kwargs):
        super().__init__(policy)
        self.problem = kwargs.get("problem")
        self.insert_strategy = kwargs.get("insert_strategy")
        self.metric_strategy = kwargs.get("metric_strategy")
        self.p = kwargs.get("p", 2) # for 'metric_strategy' = 'cartesian'
        self.K = kwargs.get("K", 1) # for 'insert_strategy' = 'regret'

    def run(self,
            env: EnvBase,
            batch: int,
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        """
        This function is used to create initial solution for training.
        It simply loads the problems into the environment and plays an episode
        using the provided policy and strategy.
        args:
            - policy: The policy network to be used.
            - env: The environment to be used.
            - batch: The batch size.
            - strategy: The strategy to decode the output of the policy network.
            - mode: The mode of the initialization, can be 'train' or 'eval'.
        returns:
            - state_td: The final state of the environment.
            - out: The output of the policy network or the scores for evaluation.
        """
        batch_size, problem_size, problem_dim = batch.shape

        if self.problem == "tsp":
            distance_matrix_batch, metric_batch = self.compute_matrices(batch)
            device = distance_matrix_batch.device

            batch_tour = []
            batch_cost = []
            for i in range(batch_size):
                pomo_tour = []
                pomo_cost = []
                distance_matrix = distance_matrix_batch[i] # (problem_size, problem_size)
                metric = metric_batch[i] # (problem_size, problem_size)

                for j in range(env.pomo_size):
                    unvisited = torch.ones(problem_size, device=device, dtype=torch.bool)
                    # randomly select the first node
                    start_node = torch.randint(0, problem_size, (1,), device=device).item()
                    unvisited[start_node] = False
                    tour = [start_node, start_node]
                    total_distance = 0

                    while unvisited.any():
                        mask = unvisited.nonzero(as_tuple=True)[0]
                        # select which node to insert with different strategies

                        if self.insert_strategy == 'regret':
                            regret_values = []
                            regret_indexs = []
                            increases = []

                            for node in mask:
                                prev_cities = torch.tensor(tour[:-1], device=device)
                                next_cities = torch.tensor(tour[1:], device=device)
                                increase = (distance_matrix[prev_cities, node] + 
                                            distance_matrix[node, next_cities] - 
                                            distance_matrix[prev_cities, next_cities])
                                # shape: (tour, )
                                sorted_increases_value, sorted_increases_index = torch.sort(increase)
                                regret = sorted_increases_value[self.K % len(increase)] - sorted_increases_value[0]
                                increases.append(sorted_increases_value[0])
                                regret_values.append(regret)
                                regret_indexs.append(sorted_increases_index[0].item() + 1)

                            index = torch.argmax(torch.tensor(regret_values)).item()
                            current_to_insert = mask[index].item()
                            best_pos = regret_indexs[index]
                            increase_insert = increases[index]
                            total_distance += increase_insert.item()

                        else:
                            dists_to_tour = metric[mask][:, tour].min(dim=1).values
                            if self.insert_strategy == 'random':
                                random_idx = torch.randint(0, len(mask), (1,), device=device).item()
                                current_to_insert = mask[random_idx].item()

                            elif self.insert_strategy == 'nearest':
                                current_to_insert = mask[torch.argmin(dists_to_tour)].item()

                            elif self.insert_strategy == 'farthest':
                                current_to_insert = mask[torch.argmax(dists_to_tour)].item()

                            elif self.insert_strategy == 'median':
                                sorted_indices = torch.argsort(dists_to_tour)
                                median_index = sorted_indices[len(sorted_indices) // 2]
                                current_to_insert = mask[median_index].item()

                            prev_cities = torch.tensor(tour[:-1], device=device)
                            next_cities = torch.tensor(tour[1:], device=device)
                            increase = (distance_matrix[prev_cities, current_to_insert] + 
                                        distance_matrix[current_to_insert, next_cities] - 
                                        distance_matrix[prev_cities, next_cities])
                            # shape: (visited tour, )
                            best_pos = torch.argmin(increase).item() + 1
                            total_distance += increase[best_pos - 1].item()

                        tour.insert(best_pos, current_to_insert)
                        unvisited[current_to_insert] = False

                    # tour: (problem_size)
                    pomo_tour.append(tour[:-1])
                    pomo_cost.append(total_distance)

                # pomo_tour: (pomo_size, problem_size)
                batch_tour.append(pomo_tour)
                batch_cost.append(pomo_cost)

            solution = torch.tensor(batch_tour) # (batch_size, pomo_size, problem_size)
            cost = torch.tensor(batch_cost) # (batch_size, pomo_size)

            td = TensorDict({
                "data": batch ,
                "selected_node_list":solution,
                "reward":cost,
            }, batch_size=[batch_size])

            mean_cost=cost.mean()
            aug_cost=cost.min(dim=1)[0].mean()
            out = {
                "no_aug_score": mean_cost,
                "aug_score": aug_cost,
            }

        elif self.problem== "cvrp":

            coor = batch[:,:,:2] # (batch_size, problem_size, 2)
            demands = batch[:,:,2] # (batch_size, problem_size)
            vehicle_capacity = 1.0

            distance_matrix_batch, metric_batch = self.compute_matrices(coor)
            device = distance_matrix_batch.device

            batch_tour = []
            batch_cost = []
            for i in range(batch_size):
                pomo_tour = []
                pomo_cost = []
                distance_matrix = distance_matrix_batch[i] # (problem_size, problem_size)
                metric = metric_batch[i] # (problem_size, problem_size)
                demand = demands[i]

                for j in range(env.pomo_size):

                    unvisited = torch.ones(len(demand), device=device, dtype=torch.bool)  # (problem, ) bool
                    unvisited[0] = False
                    route = [0, 0]

                    mask = unvisited.nonzero(as_tuple=True)[0]

                    if self.insert_strategy == 'random':
                        random_idx = torch.randint(0, len(mask), (1,), device=device).item()
                        current_to_insert = mask[random_idx].item()

                    elif self.insert_strategy == 'nearest':
                        current_to_insert = mask[torch.argmin(metric[route[-1], mask])].item()

                    elif self.insert_strategy == 'farthest':
                        current_to_insert = mask[torch.argmax(metric[route[-1], mask])].item()

                    elif self.insert_strategy == 'median':
                        current_distances = metric[route[-1], mask]
                        sorted_indices = torch.argsort(current_distances)
                        median_index = sorted_indices[len(sorted_indices) // 2]
                        current_to_insert = mask[median_index].item()

                    route = [0, current_to_insert, 0]
                    unvisited[current_to_insert] = False

                    while unvisited.any():
                        mask = unvisited.nonzero(as_tuple=True)[0]

                        dists_to_route = metric[mask][:, route].min(dim=1).values
                        if self.insert_strategy == 'random':
                            random_idx = torch.randint(0, len(mask), (1,), device=device).item()
                            current_to_insert = mask[random_idx].item()

                        elif self.insert_strategy == 'nearest':
                            current_to_insert = mask[torch.argmin(dists_to_route)].item()

                        elif self.insert_strategy == 'farthest':
                            current_to_insert = mask[torch.argmax(dists_to_route)].item()

                        elif self.insert_strategy == 'median':
                            sorted_indices = torch.argsort(dists_to_route)
                            median_index = sorted_indices[len(sorted_indices) // 2]
                            current_to_insert = mask[median_index].item()

                        prev_cities = torch.tensor(route[:-1], device=device)
                        next_cities = torch.tensor(route[1:], device=device)
                        increase = (distance_matrix[prev_cities, current_to_insert] +
                                    distance_matrix[current_to_insert, next_cities] -
                                    distance_matrix[prev_cities, next_cities])

                        depot_insert_length = (distance_matrix[[0], current_to_insert] +
                                            distance_matrix[current_to_insert, [0]] -
                                            distance_matrix[[0], [0]])

                        abs_partial_solu_2 = torch.tensor(route)

                        abs_partial_solu_2 = tran_to_node_flag(abs_partial_solu_2.unsqueeze(0))
                        remaining_capacity = cal_remaining_capacity(
                            demand.unsqueeze(0), abs_partial_solu_2, capacity=vehicle_capacity)
                        remaining_capacity = remaining_capacity.squeeze(0)

                        increase = torch.cat((increase, depot_insert_length), dim=0)
                        # print(increase)
                        current_demand = demand[current_to_insert]
                        mask2 = remaining_capacity < current_demand

                        increase[mask2.eq(1)] = 10000000000

                        best_pos = torch.argmin(increase).item() + 1

                        if best_pos == len(increase):
                            route = route + [current_to_insert, 0]
                        else:
                            route.insert(best_pos, current_to_insert)

                        unvisited[current_to_insert] = False

                    pomo_tour.append(route)
                    cost = calculate_vrp_cost(route, distance_matrix)
                    pomo_cost.append(cost)

                batch_tour.append(pomo_tour)
                batch_cost.append(pomo_cost)

            # padding
            max_len = max(len(x) for sub_tour in batch_tour for x in sub_tour)
            padded = [x + [0]*(max_len - len(x)) for sub_tour in batch_tour for x in sub_tour]

            solution = torch.tensor(padded) # (batch_size, pomo_size, problem_size)
            cost = torch.tensor(batch_cost) # (batch_size, pomo_size)

            td = TensorDict({
                "data": batch ,
                "selected_node_list":solution,
                "reward":cost,
            }, batch_size=[batch_size])

            mean_cost=cost.mean()
            aug_cost=cost.min(dim=1)[0].mean()
            out = {
                "no_aug_score": mean_cost,
                "aug_score": aug_cost,
            }

        else:
            raise NotImplementedError(
                f"Insertion method is not supported for creating initial solution for {self.problem}!"
                )

        return td,out

    def compute_matrices(self, batch):
        dist = torch.cdist(batch, batch, p=2)
        if self.metric_strategy == 'cartesian':
            metric = torch.cdist(batch, batch, p=self.p)
        elif self.metric_strategy == 'polar':
            origins = batch[:, [0]]
            rel = batch - origins
            angles = torch.atan2(rel[:, :, 1], rel[:, :, 0])
            diff = torch.abs(angles[:, :, None] - angles[:, None, :])
            metric = torch.min(diff, 2 * torch.pi - diff)
        else:
            raise ValueError(f"Unknown metric strategy: {self.metric_strategy}")
        return dist, metric

def calculate_vrp_cost(route, distance_mat):
    total_cost = torch.tensor(0.0, device=distance_mat.device)

    route = torch.tensor(route, device=distance_mat.device, dtype=torch.long)
    indices = route[:-1]
    next_indices = route[1:]
    total_cost += torch.sum(distance_mat[indices, next_indices])
    return total_cost.item()

def node_flag_tran_to_(node_flag):
    '''
    :param node_list: [B, V, 2]
    :return: [B, V+n]
    '''

    batch_size = node_flag.shape[0]
    problem_size = node_flag.shape[1]
    node = node_flag[:, :, 0]
    flag = node_flag[:, :, 1]
    depot_num = flag.sum(1)

    max_length = torch.max(depot_num)

    store_1 = torch.ones(size=(batch_size, problem_size + max_length), dtype=torch.long)

    where_is_depot_0, where_is_depot_1 = torch.where(flag == 1)

    temp1 = torch.arange(max_length)[None, :].repeat(batch_size, 1)
    temp2 = temp1 < depot_num[:, None]
    temp3 = temp1[temp2]
    where_is_depot_1 = where_is_depot_1 + temp3

    store_1[where_is_depot_0, where_is_depot_1] = 0

    mask = torch.arange(problem_size + max_length)[None, :].repeat(batch_size, 1)
    nodesss = problem_size + depot_num
    mask2 = (mask < nodesss[:, None]).long()
    store_2 = store_1 * mask2

    store_2[store_2.gt(0.1)] = node.ravel()

    zeros = torch.zeros(size=(batch_size, 1), dtype=torch.long)

    result = torch.cat((store_2, zeros), dim=1)

    return result

def tran_to_node_flag(node_list):
    '''
    :param node_list: [B, V+n]
    :return: [B, V, 2]
    '''

    batch_size = node_list.shape[0]

    index_smaller_0_shift = torch.roll(torch.le(node_list, 0), shifts=1, dims=1).long()
    index_bigger_0 = torch.gt(node_list, 0).long()

    flag_index = index_smaller_0_shift * index_bigger_0

    save_index = torch.gt(node_list, 0.1)

    save_node = node_list[save_index].reshape(batch_size, -1)
    save_flag = flag_index[save_index].reshape(batch_size, -1)

    node_flag_1 = torch.cat((save_node.unsqueeze(2), save_flag.unsqueeze(2)), dim=2)

    return node_flag_1

def cal_remaining_capacity(demand, solution, capacity=30):
    # demand: (1, problem_size+1)
    # solution: (1, sub_length, 2)

    solution_size = solution.shape[1]

    order_flag = solution[:, :, 1].clone()

    batch_size = solution.shape[0]

    visit_depot_num = torch.sum(solution[:, :, 1], dim=1)

    start_from_depot2 = solution[:, :, 1].nonzero()

    start_from_depot3 = solution[:, :, 1].roll(shifts=-1, dims=1).nonzero()

    repeat_solutions_node = solution[:, :, 0].repeat_interleave(visit_depot_num, dim=0)

    double_repeat_solution_node = repeat_solutions_node #.repeat(1, 2)

    x1 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node),
                                                                            1) >= start_from_depot2[:, 1][:, None]


    x2 = torch.arange(double_repeat_solution_node.shape[1])[None, :].repeat(len(repeat_solutions_node),
                                                                            1) <= start_from_depot3[:, 1][:, None]


    x3 = (x1 * x2).long()



    sub_tourss = double_repeat_solution_node * x3
    demands = torch.repeat_interleave(demand, repeats=visit_depot_num, dim=0)

    demands_ = get_encoding(demands.unsqueeze(2),sub_tourss).squeeze(2)

    demands_total_per_subtour = demands_.sum(1).unsqueeze(1)

    demands_total_per_subtour = demands_total_per_subtour * x3


    demands_total_per_subtour_ = demands_total_per_subtour[demands_total_per_subtour.gt(0)].reshape(batch_size, solution_size)


    remaining_capacitys = capacity - demands_total_per_subtour_

    #######
    remaining_capacitys = remaining_capacitys * 10000

    # print('25 ---------- remaining_capacitys \n', remaining_capacitys)

    remaining_capacitys_flag = torch.cat((remaining_capacitys.unsqueeze(2), order_flag.unsqueeze(2)),dim=2).long()

    # print('26 ---------- remaining_capacitys_flag \n', remaining_capacitys_flag)

    remaining_capacitys_edges = node_flag_tran_to_(remaining_capacitys_flag)

    # print('27 ---------- remaining_capacitys_edges \n', remaining_capacitys_edges)

    remaining_capacitys_edges_shift = torch.roll(remaining_capacitys_edges,dims=1,shifts=-1)

    # print('28 ---------- remaining_capacitys_edges_shift \n', remaining_capacitys_edges_shift)

    index = remaining_capacitys_edges.eq(0)

    remaining_capacitys_edges[index] = remaining_capacitys_edges_shift[index]

    # print('30 ---------- remaining_capacitys_edges \n', remaining_capacitys_edges)


    ######
    remaining_capacitys_edges = remaining_capacitys_edges.float() / 10000

    # remaining_capacitys_edges[remaining_capacitys_edges.eq(0)] = capacity
    remaining_capacitys_edges[:,-1] = capacity
    # 等于0的话，”depot的demand=0“ 与 “该路径的remaining capacity” 混合起来了。

    return remaining_capacitys_edges

def get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes