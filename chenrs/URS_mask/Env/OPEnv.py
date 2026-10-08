from dataclasses import dataclass
import torch
from ProblemDef import get_random_problems_op, augment_xy_data_by_8_fold
import math


@dataclass
class Reset_State:
    problem_name: str = "op"
    problems: torch.Tensor = None
    # shape: (batch, problem+1, 3)
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
    log_scale: float = None
    first: torch.Tensor = None
    last: torch.Tensor = None
    depot_tag: torch.Tensor = None
    relation: torch.Tensor = None

@dataclass
class Step_State:
    curr_length: torch.Tensor = None
    total_length: torch.Tensor = None

    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)




class OPEnv:
    def __init__(self):

        # Const @INIT
        ####################################
        self.problem_size = None
        self.pomo_size = None

        self.FLAG__use_saved_problems = False
        self.check_validity = False  # 是否要检查solution的可行性，默认为False

        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_prize = None
        self.saved_index = None
        self.device = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.problems = None

        self.depot_node_xy = None # shape: (batch, problem+1, 2)
        self.depot_node_prize = None # shape: (batch, problem+1)
        self.dist = None  # shape: (batch, problem+1, problem+1)

        self.max_length = 4.0  # when problem size is greater than 100, max length is 4.0, following GOAL and RL4CO

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)



        self.tour = None
        self.first = None
        self.last = None
        self.collected = None
        self.last_collected = None
        self.finished = None
        self.allow = None
        self.depot_flag = None
        self.load = None
        self.dist = None
        self.pomo_last = None
        self.pomo_first = None
        self.cur = None
        self.cur_last = None
        self.dist_last = None
        self.dist_last_and_depot = None
        self.depot_tag = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()



    def input_saved_data(self,depot_xy, node_xy, prize,device):
        self.FLAG__use_saved_problems = True
        self.saved_depot_xy = depot_xy
        self.saved_node_xy = node_xy
        self.saved_prize = prize
        self.saved_index = 0
        self.device = device


    def load_problems(self, batch_size, problem_size, pomo_size=None, lib_data=None, validation_data=None,
                      aug_factor=1, device=None, start=0,**kwargs):
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
            node_xy = xy[:,1:,:2]
            prize = prize[:,1:]
            # depot_xy = validation_data["depot_xy"][start:start+self.batch_size].to(device)
            # # shape: (batch, 1, 2)
            # node_xy = validation_data["node_xy"][start:start+self.batch_size].to(device)
            # # shape: (batch, problem, 2)
            # prize = validation_data["prize"][start:start+self.batch_size].to(device)
            # # shape: (batch, problem)
        else:
            if not self.FLAG__use_saved_problems:
                problems = get_random_problems_op(batch_size, self.problem_size)
                depot_xy = problems[:,0:1,:2]
                node_xy = problems[:,1:,:2]
                prize = problems[:,1:,2]
            else:
                depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
                node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
                prize = self.saved_prize[self.saved_index:self.saved_index + batch_size]
                self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                prize = prize.repeat(8, 1)
            else:
                raise NotImplementedError(f'The augmentation factor {aug_factor} is not implemented.')

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_prize = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_prize = torch.cat((depot_prize, prize), dim=1)
        # shape: (batch, problem+1)

        depot_node_xy_prize = torch.cat((self.depot_node_xy, self.depot_node_prize.unsqueeze(-1)), dim=-1)
        # shape: (batch, problem+1, 3)

        self.reset_state.problems = depot_node_xy_prize


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        self.load = torch.ones(size=(self.batch_size, self.pomo_size)) * self.max_length  # max_length<=4.0
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        # Note that for "torch.cdist" function, compute_mode must be 'donot_use_mm_for_euclid_dist'.
        # For more details about this issue, please refer to the TSPEnv.py file.
        self.dist = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2,
                                compute_mode='donot_use_mm_for_euclid_dist')
        self.reset_state.dist = self.dist
        # shape: (batch, problem+1, problem+1)
        self.reset_state.log_scale = math.log2(self.problem_size)

        self.step_state.batch_size = self.batch_size
        self.step_state.pomo_size = self.pomo_size

        reward = None
        done = False
        self.pomo_first = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)

        #self.last_selected_xy = self.reset_state.problems[:, :, :2][:, None, :, :].repeat(1, self.pomo_size, 1, 1).gather(2,self.pomo_first[:, :,None, None].expand(-1, -1, -1,2)).squeeze(2)
        self.last_selected_xy = self.reset_state.problems[:, [0], :2].expand(-1,self.pomo_size,-1)
        # shape: (batch, pomo, 2)
        # distances from each node to depot
        #self.depot_to_any_dict = self.reset_state.dist.gather(dim=1, index=self.pomo_first.unsqueeze(-1).expand(-1, -1, self.problem_size + 1))
        self.any_to_depot_dist = self.reset_state.dist[:,:,0][:,None,:].expand(-1,self.pomo_size,-1)
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
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        self.at_the_depot = (selected == 0)

        if self.selected_count != 1:
            #current_node_xy = self.reset_state.problems[:, :, :2][:, None, :, :].expand(-1, self.pomo_size, -1, -1).gather(2,selected[:, :, None,None].expand(-1, -1,-1,2)).squeeze(2)
            current_node_xy = self.reset_state.problems[:, :, :2].gather(1,selected[:, :,None].expand(-1,-1,2))
            # shape: (batch, pomo, 2)
            self.load -= (current_node_xy - self.last_selected_xy).norm(p=2, dim=-1)
            self.last_selected_xy = current_node_xy.clone()

            dist_cur = self.get_local_feature() #当前节点到每一个其余节点的距离
            dist_back = self.any_to_depot_dist.clone()#t+1回到起点的距离
            # shape: (batch, pomo, problem+1)
            #从第t个点到t+1点的距离 t+1回到depot要小于4
            self.step_state.ninf_mask[(self.load[:, :, None] - dist_cur - dist_back) < 0.0] = float('-inf')

            gathering_index = selected.unsqueeze(-1)
            self.step_state.ninf_mask.scatter_(dim=-1, index=gathering_index, value=float('-inf'))
            # depot is always allowed, although we hope it is not early selected.
            self.step_state.ninf_mask[:, :, 0] = 0

        # 将finished的其他action设置为-inf,只能在depot
        # once at the depot(except for the first step), all other actions are invalid and the tour is finished.
        self.finished = self.at_the_depot & (self.selected_count > 1)
        finished_extend = self.finished.unsqueeze(-1).expand(-1, -1, self.problem_size)
        self.step_state.ninf_mask[:, :, 1:][finished_extend] = float('-inf')


        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.finished = self.finished

        print("#####################################")
        print(self.ninf_mask[0, 0])
        print(self.selected_node_list[0, 0])
        print("#####################################")

        # returning values
        done = self.finished.all()
        if done:
            reward = self._get_open_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done


    def _get_open_travel_distance(self):
        if self.check_validity:
            self.check_solution_validity()
        solution = self.selected_node_list.clone()
        visited = torch.zeros((solution.size(0), solution.size(1), self.reset_state.problems.size(-2)))
        visited = visited.scatter(-1, solution, 1)
        prize = (visited * self.reset_state.problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)

        return prize

    def get_local_feature(self):
        # dist.shape: (batch, problem+1, problem+1)
        # current_node.shape: (batch, pomo)
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(-1, -1, self.problem_size + 1)
        # shape: (batch, pomo, problem+1)
        cur_dist = self.dist.gather(dim=1,index=current_node)
        # shape: (batch, pomo, problem+1)

        return cur_dist



    def _get_travel_distance(self,lib_mode):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        if not lib_mode:
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)
        else:
            assert self.original_depot_node_xy_lib.size(0) == 1, 'The original_node_xy_lib should be a single instance.'
            self.original_depot_node_xy_lib = self.original_depot_node_xy_lib.expand(self.batch_size, -1, -1)  # shape:(8,problem+1,2)
            all_xy = self.original_depot_node_xy_lib[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (8, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


    def check_solution_validity(self):
        #除了depot，每个点只能访问一次
        actions = self.selected_node_list.clone()
        sorted_actions = actions.sort(-1)[0]

        # Make sure each node visited once at most (except for depot)
        assert (
                (sorted_actions[..., 1:] == 0)
                | (sorted_actions[..., 1:] > sorted_actions[..., :-1])
        ).all(), "Duplicates"

        travel_distances = self._get_travel_distance(lib_mode=False) #- 1e-6

        # if ~(travel_distances[..., None] <= self.max_length + 1e-5).all():
        #     print(travel_distances.max())
        #     print((travel_distances[..., None] - self.max_length).max())

        assert (
                travel_distances[..., None]  <= self.max_length + 1e-5 # 1e-5 is a small tolerance to avoid numerical issues
        ).all(), "Max length exceeded by {}".format(
            (travel_distances[..., None] - self.max_length).max()
        )