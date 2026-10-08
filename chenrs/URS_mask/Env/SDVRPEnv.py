from dataclasses import dataclass
import math
import torch

from Env.CVRPEnv import CVRPEnv
from ProblemDef import get_random_problems_cvrp, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problem_name: str = "sdvrp"
    problems: torch.Tensor = None
    # shape: (batch, problem+1, 3)
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
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



class SDVRPEnv(CVRPEnv):
    def __init__(self, ):
        super().__init__()
        self.reset_state.problem_name = 'sdvrp'

    def load_problems(self, batch_size, problem_size,pomo_size=None,lib_data=None, validation_data=None, aug_factor=1, device=None,start=0,capacity=None,**kwargs):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device
        self.depot_num = 1
        if lib_data is not None:
            depot_xy = lib_data["depot_xy"].to(device)  # # shape: (1, 1, 2)
            node_xy = lib_data["node_xy"].to(device)  # shape: (1, problem, 2)
            node_demand = lib_data['node_demand'].to(device)  # not including the depot node
            self.original_depot_node_xy_lib = lib_data['original_depot_node_xy_lib'].to(device)  # shape: (1, problem+1, 2)
        elif validation_data is not None:
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
            depot_xy = xy[:, 0:1, :]
            # shape: (batch, 1, 2)
            node_xy = xy[:, 1:, :]
            # shape: (batch, problem, 2)
            node_demand = demand[:, 1:]
        else:
            if not self.FLAG__use_saved_problems:
                assert capacity is not None, "capacity must be given when generating random problems."
                depot_xy, node_xy, node_demand = get_random_problems_cvrp(batch_size, self.problem_size, capacity)
            else:
                depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
                node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
                node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
                self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError(f'The augmentation factor {aug_factor} is not implemented.')

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, 1, problem+1)

        depot_node_xy_demand = torch.cat((self.depot_node_xy, self.depot_node_demand.unsqueeze(-1)), dim=-1)
        # shape: (batch, problem+1, 3)

        # self.reset_state.depot_xy = depot_xy
        # self.reset_state.node_xy = node_xy
        # self.reset_state.node_demand = node_demand

        self.reset_state.problems = depot_node_xy_demand
        self.depot_node_demand = self.depot_node_demand[:,None,:].expand(self.batch_size, self.pomo_size, -1)

    def step(self, selected,lib_mode=False):
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

        demand_list = self.depot_node_demand
        # shape: (batch, pomo, problem+1)
        gathering_index = selected.unsqueeze(-1)
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        actual_selected_demand = torch.min(selected_demand,self.load)
        self.load -= actual_selected_demand
        self.depot_node_demand = self.depot_node_demand.scatter_add(-1, gathering_index, -actual_selected_demand.unsqueeze(-1))

        assert (self.load >= -self.round_error_epsilon).all(), "load cannot be negative!"
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        mask_index = (self.depot_node_demand == 0)
        self.visited_ninf_flag[mask_index] = float('-inf')

        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        # 如果满载了，除了depot全部mask
        full_index = (self.load <= self.round_error_epsilon).unsqueeze(-1).expand(-1, -1, self.problem_size)
        self.ninf_mask[:, :, 1:][full_index] = float('-inf')

        # 有的solution在访问完最后一个节点时，每个节点的demand就都被满足了，但此时因为不在depot，所以depot并没有被mask掉
        newly_finished = (self.visited_ninf_flag[...,1:] == float('-inf')).all(dim=2)
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

        print("#####################################")
        print(self.ninf_mask[0, 0])
        print(self.selected_node_list[0, 0])
        print("#####################################")

        # returning values
        done = self.selected_count >= self.depot_node_demand.size(-1) and not (self.depot_node_demand > 0).any()
        if done:
            assert self.finished.all(), "All episodes should be finished when done is True."
            #self.check_solution_validity()
            reward = -self._get_travel_distance(lib_mode)  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done




