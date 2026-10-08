from dataclasses import dataclass

import math
import torch

from ProblemDef import get_random_problems_acvrp


@dataclass
class Reset_State:
    problem_name: str = "acvrp"

    demand: torch.Tensor = None
    # shape: (batch, problem+1, 1)
    dist: torch.Tensor = None
    problems: torch.Tensor = None  #only include demand
    # shape: (batch, problem+1, 3)
    log_scale: float = None
    relation: torch.Tensor = None


@dataclass
class Step_State:
    batch_size: torch.Tensor = None
    pomo_size: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class ACVRPEnv:
    def __init__(self, ):

        # Const @INIT
        ####################################
        self.problem_size = None
        self.pomo_size = None

        self.FLAG__use_saved_problems = False
        self.saved_node_demand = None
        self.saved_index = None
        self.device = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.dist = None # shape: (batch, problem+1, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.round_error_epsilon = 0.00001 # for precision stability

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def input_saved_data(self,dist_matrix,node_demand,device):
        self.FLAG__use_saved_problems = True
        self.saved_dist_matrix = dist_matrix
        self.saved_node_demand = node_demand
        self.saved_index = 0
        self.device = device

    def load_problems(self, batch_size, problem_size, pomo_size=None, validation_data=None, aug_factor=1, device=None,
                      start=0,capacity=None, **kwargs):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device

        self.depot_num = 1
        if validation_data is not None:
            data = validation_data
            # xy 特殊处理
            if 'xy' in data:
                xy = data['xy'][start:start + self.batch_size].to(self.device)
            else:
                xy = torch.zeros(
                    batch_size, problem_size + self.depot_num, 2, device=self.device
                )

            # demand
            if 'demand' in data:
                demand = data['demand'][start:start + self.batch_size].to(self.device)
            else:
                demand = torch.zeros(
                    batch_size, problem_size + self.depot_num, device=self.device
                )

            # dist
            if 'dist' in data:
                dist = data['dist'][start:start + self.batch_size].to(self.device)
            else:
                dist = torch.cdist(xy, xy, p=2, compute_mode='donot_use_mm_for_euclid_dist')

            # prize
            if 'prize' in data:
                prize = data['prize'][start:start + self.batch_size].to(self.device)
            else:
                prize = torch.zeros(
                    batch_size, problem_size + self.depot_num, device=self.device
                )

            # penalty
            if 'penalty' in data:
                penalty = data['penalty'][start:start + self.batch_size].to(self.device)
            else:
                penalty = torch.zeros(
                    batch_size, problem_size + self.depot_num, device=self.device
                )

            # fake_prize
            if 'fake_prize' in data:
                fake_prize = data['fake_prize'][start:start + self.batch_size].to(self.device)
            else:
                fake_prize = torch.zeros(
                    batch_size, problem_size + self.depot_num, device=self.device
                )

            # service_time
            if 'service_time' in data:
                service_time = data['service_time'][start:start + self.batch_size].to(self.device)
            else:
                service_time = torch.zeros(
                    batch_size, problem_size + self.depot_num, device=self.device
                )

            # tw_start
            if 'tw_start' in data:
                tw_start = data['tw_start'][start:start + self.batch_size].to(self.device)
            else:
                tw_start = torch.zeros(
                    batch_size, problem_size + self.depot_num, device=self.device
                )

            # tw_end
            if 'tw_end' in data:
                tw_end = data['tw_end'][start:start + self.batch_size].to(self.device)
            else:
                tw_end = torch.full(
                    (batch_size, problem_size + self.depot_num), float('inf'), device=self.device
                )
            # route_limit
            if 'route_limit' in data:
                route_limit = data['route_limit'][start:start + self.batch_size].to(self.device)
            else:
                route_limit = torch.full(
                    (batch_size,), float('inf'), device=self.device
                )

            dist_matrix = dist

        else:
            if not self.FLAG__use_saved_problems:
                assert capacity is not None, "capacity must be given when generating random problems."
                dist_matrix,demand = get_random_problems_acvrp(batch_size, self.problem_size, capacity)
            else:
                dist_matrix = self.saved_dist_matrix[self.saved_index:self.saved_index + self.batch_size].to(self.device)
                demand = self.saved_node_demand[self.saved_index:self.saved_index + self.batch_size].to(self.device)
                self.saved_index += self.batch_size

        if aug_factor > 1:
            dist_matrix = dist_matrix.repeat(aug_factor, 1, 1)
            demand = demand.repeat(aug_factor, 1)
            self.batch_size = self.batch_size * aug_factor

        self.dist_matrix = dist_matrix
        self.depot_node_demand = demand
        self.reset_state.problems = demand

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        # Note that for "torch.cdist" function, compute_mode must be 'donot_use_mm_for_euclid_dist'.
        # For more details about this issue, please refer to the TSPEnv.py file.
        self.dist = self.dist_matrix
        self.reset_state.dist = self.dist
        # shape: (batch, problem+1, problem+1)
        self.reset_state.demand = self.depot_node_demand
        self.reset_state.log_scale = math.log2(self.problem_size)

        self.step_state.batch_size = self.batch_size
        self.step_state.pomo_size = self.pomo_size

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node.unsqueeze(-1)), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected.unsqueeze(-1)
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        assert (self.load >= -self.round_error_epsilon).all(), "load cannot be negative!"
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag.scatter_(dim=-1, index=gathering_index, value=float('-inf'))
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()

        demand_too_large = self.load[:, :, None] + self.round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_total_distance()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done

    def _get_total_distance(self):

        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        solution_length = node_from.shape[-1]
        batch_index = BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, solution_length)
        # shape: (batch, pomo, node)
        selected_cost = self.dist_matrix[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance

    def get_local_feature(self):
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(self.batch_size, self.pomo_size, self.problem_size+1)
        # shape: (batch, pomo, problem)
        cur_dist = self.dist_matrix.gather(dim=1, index=current_node)
        # shape: (batch, pomo, problem)

        return cur_dist
