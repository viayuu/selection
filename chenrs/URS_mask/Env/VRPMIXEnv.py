from dataclasses import dataclass
import torch
from ProblemDef import get_random_problems_vrp_mixed_mvmoe_version, augment_xy_data_by_8_fold
import math



@dataclass
class Reset_State:
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
    log_scale: float = None
    problems: torch.Tensor = None
    problem_name: str = None
    relation: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    batch_size: torch.Tensor = None
    pomo_size: torch.Tensor = None

    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class VRPMIXEnv:
    def __init__(self):

        # Const @INIT
        ####################################
        self.backhaul_ratio = 0.2
        self.problem_size = None
        self.pomo_size = None
        self.FLAG__use_saved_problems = False

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_service_time = None
        # shape: (batch, problem+1)
        self.depot_node_tw_start = None
        # shape: (batch, problem+1)
        self.depot_node_tw_end = None
        # shape: (batch, problem+1)
        self.speed = 1.0
        self.depot_start, self.depot_end = 0., 3.  # tw for depot [0, 3]

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
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.open = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, problem_size,pomo_size=None,lib_data=None, validation_data=None, aug_factor=1, device=None,capacity=None,**kwargs):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device

        problem_name = kwargs.get("problem_name")  #O,B,L,TW
        self.reset_state.problem_name = problem_name
        start = 0
        self.depot_num = 1
        if validation_data is not None:
            data = validation_data
            # xy
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

            depot_xy = xy[:,0:1,:2]
            # shape: (batch, 1, 2)
            node_xy = xy[:,1:,:2]
            # shape: (batch, problem, 2)
            node_demand = demand[:,1:]
            # shape: (batch, problem)
            service_time = torch.zeros(batch_size, problem_size).to(device)
            tw_start = torch.zeros(batch_size, problem_size).to(device)
            tw_end = (torch.ones(batch_size, problem_size)*3.0).to(device)
            route_limit = (torch.ones(batch_size, problem_size)*float("inf")).to(device) #但实际上在step过程中，如果没有对应约束，属性是不会用的
            if 'tw' in problem_name:
                service_time = validation_data["node_servicetime"][:self.batch_size].to(device)
                tw_start = validation_data["node_earlyTW"][:self.batch_size].to(device)
                tw_end = validation_data["node_lateTW"][:self.batch_size].to(device)
            if 'l' in problem_name:
                route_limit = validation_data["route_limit"] * torch.ones(batch_size)

        else:
            if not self.FLAG__use_saved_problems:
                assert capacity is not None, "capacity must be given when generating random problems."
                depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end = get_random_problems_vrp_mixed_mvmoe_version(
                    batch_size, self.problem_size, capacity, problem_name=problem_name)
            else:
                depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
                node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
                node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
                tw_start = self.saved_node_earlyTW[self.saved_index:self.saved_index + batch_size]
                tw_end = self.saved_node_lateTW[self.saved_index:self.saved_index + batch_size]
                service_time = self.saved_node_servicetime[self.saved_index:self.saved_index + batch_size]
                route_limit = self.saved_route_limit[self.saved_index:self.saved_index + batch_size]
                self.saved_index += batch_size

        route_limit = route_limit[:, None] if route_limit.dim() == 1 else route_limit

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                route_limit = route_limit.repeat(8, 1)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        # reset pomo_size
        if 'b' in problem_name:
            self.pomo_size = min(int(self.problem_size * (1 - self.backhaul_ratio)), self.pomo_size)
            self.START_NODE = torch.arange(start=1, end=self.problem_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
            self.START_NODE = self.START_NODE[node_demand > 0].reshape(self.batch_size, -1)[:, :self.pomo_size]

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_service_time = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_tw_start = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_start
        depot_tw_end = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_end
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.route_limit = route_limit
        # shape: (batch, 1)
        self.depot_node_service_time = torch.cat((depot_service_time, service_time), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_end = torch.cat((depot_tw_end, tw_end), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        if 'tw' in problem_name:
            depot_node_xy_demand_earlyTW_lateTW_serviceTime = torch.cat(
                (self.depot_node_xy, self.depot_node_demand.unsqueeze(-1),
                 self.depot_node_tw_start.unsqueeze(-1),
                 self.depot_node_tw_end.unsqueeze(-1), self.depot_node_service_time.unsqueeze(-1)), dim=-1)
            self.reset_state.problems = depot_node_xy_demand_earlyTW_lateTW_serviceTime
        else:
            depot_node_xy_demand = torch.cat((self.depot_node_xy, self.depot_node_demand.unsqueeze(-1)),dim=-1)
            self.reset_state.problems = depot_node_xy_demand


        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        if 'o' in problem_name:
            self.step_state.open = torch.ones(self.batch_size, self.pomo_size).to(self.device)
        else:
            self.step_state.open = torch.zeros(self.batch_size, self.pomo_size).to(self.device)
        self.step_state.START_NODE = self.START_NODE


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        self.dist = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2,
                                compute_mode='donot_use_mm_for_euclid_dist')
        self.reset_state.dist = self.dist
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
        self.step_state.time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        problem_name = self.reset_state.problem_name

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        current_coord = self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.length[self.at_the_depot] = 0  # reset the length of route at the depot
        self.current_coord = current_coord

        # Mask
        ####################################
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        # Only for VRPB: reset load to 0.
        # >> Old implementation
        #   a. if visit backhaul nodes in the first two POMO moves (i.e., depot -> backhaul, the route is mixed with backhauls and linehauls alternatively);
        #   b. if only backhaul nodes unserved, we relax the load to be 0 (i.e., the vehicle only visit backhauls nodes in the last few routes).
        # if self.selected_node_list.size(-1) == 1:  # POMO first move
        #     depot_backhaul = self.at_the_depot & (self.depot_node_demand[:, 1:self.pomo_size+1] < 0.)
        #     # shape: (batch, pomo)
        #     self.load[depot_backhaul] = 0.
        # else:
        # >> New implementation - Remove constraint a, the POMO start node should be a linehaul.
        if 'b' in problem_name:
            unvisited_demand = demand_list + self.visited_ninf_flag
            # shape: (batch, pomo, problem+1)
            linehauls_unserved = torch.where(unvisited_demand > 0., True, False)
            reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
            # shape: (batch, pomo)
            self.load[reset_index] = 0.

        # capacity constraint
        #   a. the remaining vehicle capacity >= the customer demands
        #   b. the remaining vehicle capacity <= the vehicle capacity (i.e., 1.0)
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        exceed_capacity = self.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
        self.ninf_mask[exceed_capacity] = float('-inf')

        # duration limit constraint
        if 'l' in problem_name:
            route_limit = self.route_limit[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size + 1)
            # shape: (batch, pomo, problem+1)
            # check route limit constraint: length + cur->next->depot <= route_limit
            route_too_large = self.length[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) > route_limit + round_error_epsilon
            route_too_large[:, :, 0] = False
            # shape: (batch, pomo, problem+1)
            self.ninf_mask[route_too_large] = float('-inf')
            # shape: (batch, pomo, problem+1)

        # time window constraint
        # Note: different from VRPTW, since no need to return to depot for OVRPBLTW
        #   current_time: the end time of serving the current node
        #   a. max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        if 'tw' in problem_name:
            self.current_time = torch.max(self.current_time + new_length / self.speed, self.depot_node_tw_start[torch.arange(self.batch_size)[:, None], selected]) + self.depot_node_service_time[torch.arange(self.batch_size)[:, None], selected]
            self.current_time[self.at_the_depot] = 0
            # shape: (batch, pomo)
            arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed, self.depot_node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
            out_of_tw = arrival_time > self.depot_node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
            # shape: (batch, pomo, problem+1)
            if 'o' in problem_name:
                out_of_tw[:, :, 0] = False
                self.ninf_mask[out_of_tw] = float('-inf')
            else:
                self.ninf_mask[out_of_tw] = float('-inf')
                fail_return_depot = arrival_time + self.depot_node_service_time[:, None, :].expand(-1, self.pomo_size,-1) + (self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :,:].expand(-1, self.pomo_size, -1,
                                                                                               -1)).norm(p=2,dim=-1) / self.speed > self.depot_end + round_error_epsilon
                # shape: (batch, pomo, problem+1)
                self.ninf_mask[fail_return_depot] = float('-inf')



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
        self.step_state.time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        print("#####################################")
        print(self.ninf_mask[0, 0])
        print(self.selected_node_list[0, 0])
        print("#####################################")
        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        problem_name = self.reset_state.problem_name

        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        if 'o' in problem_name:
            not_to_depot = self.selected_node_list.roll(dims=2, shifts=-1) != 0
            # shape: (batch, pomo, selected_list_length)
            travel_distances = (segment_lengths * not_to_depot).sum(2)
            # shape: (batch, pomo)
        else:
            travel_distances = segment_lengths.sum(2)

        return travel_distances

    def get_local_feature(self):
        # dist.shape: (batch, problem+1, problem+1)
        # current_node.shape: (batch, pomo)
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(-1, -1, self.problem_size + 1)
        # shape: (batch, pomo, problem+1)
        cur_dist = self.dist.gather(dim=1, index=current_node)
        # shape: (batch, pomo, problem+1)

        return cur_dist



