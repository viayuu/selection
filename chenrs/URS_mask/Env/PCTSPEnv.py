
from dataclasses import dataclass
import torch

from ProblemDef import get_random_problems_pctsp, augment_xy_data_by_8_fold
import math


@dataclass
class Reset_State:
    problem_name: str = "pctsp"
    problems: torch.Tensor = None
    # shape: (batch, problem, 2)
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
    log_scale: float = None
    first: torch.Tensor = None
    last: torch.Tensor = None
    depot_tag: torch.Tensor =None
    relation: torch.Tensor = None



@dataclass
class Step_State:
    batch_size: torch.Tensor = None

    pomo_size: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    load = None


class PCTSPEnv:
    def __init__(self):

        # Const @INIT
        ####################################
        self.problem_size = None
        self.pomo_size = None
        self.FLAG__use_saved_problems = False
        self.check_validity = False #是否要检查solution的可行性，默认为False

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        self.tour = None
        self.first = None
        self.last = None
        self.collected = None
        self.last_collected = None
        self.finished = None
        self.allow = None
        self.depot_flag = None
        self.depot_tag = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)
        self.load = None



        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()



    def input_saved_data(self,depot_xy, node_xy, prize, penalty, device):
        self.FLAG__use_saved_problems = True
        self.saved_depot_xy = depot_xy
        self.saved_node_xy = node_xy
        self.saved_prize = prize
        self.saved_index = 0
        self.device = device
        self.saved_penalty = penalty

    def load_problems(self, batch_size, problem_size, pomo_size=None, lib_data=None, validation_data=None,
                      aug_factor=1, device=None, start=0,capacity=None,**kwargs):
        self.batch_size = batch_size
        self.problem_size = problem_size
        # self.fix_problem_size = kwargs.get("fix_problem_size",True)
        if self.problem_size == 100:
            self.fix_problem_size = True
        else:
            self.fix_problem_size = False
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device

        if validation_data is not None:
            depot_xy = validation_data["depot_xy"][start:start+self.batch_size].to(device)
            # shape: (batch, 1, 2)
            node_xy = validation_data["node_xy"][start:start+self.batch_size].to(device)
            # shape: (batch, problem, 2)
            prize = validation_data["prize"][start:start+self.batch_size].to(device)
            # shape: (batch, problem)
            penalty = validation_data["penalty"][start:start+self.batch_size].to(device)

        else:
            if not self.FLAG__use_saved_problems:
                problems = get_random_problems_pctsp(batch_size, self.problem_size,fix_problem_size=self.fix_problem_size)
                depot_xy = problems[:,0:1,:2]
                node_xy = problems[:,1:,:2]
                prize = problems[:,1:,2]
                penalty = problems[:,1:,3]
            else:
                depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
                node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
                prize = self.saved_prize[self.saved_index:self.saved_index + batch_size]
                penalty = self.saved_penalty[self.saved_index:self.saved_index + batch_size]
                self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                prize = prize.repeat(8, 1)
                penalty = penalty.repeat(8, 1)
            else:
                raise NotImplementedError(f'The augmentation factor {aug_factor} is not implemented.')

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_prize = torch.zeros(size=(self.batch_size, 1))
        depot_penalty = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_prize = torch.cat((depot_prize, prize), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_penalty = torch.cat((depot_penalty, penalty), dim=1)
        depot_node_xy_prize_penalty = torch.cat((self.depot_node_xy, self.depot_node_prize.unsqueeze(-1),self.depot_node_penalty.unsqueeze(-1)), dim=-1)
        # shape: (batch, problem+1, 4)

        self.reset_state.problems = depot_node_xy_prize_penalty



    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.load = torch.zeros(size=(self.batch_size, self.pomo_size))  #用于收集prize,prize满足条件便可以选择结束

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
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

        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.ninf_mask[:,:,0] = float('-inf')  #没收集到足够多的奖励不能回到depot

        reward = None
        done = False

        return self.reset_state, reward, done



    def _get_open_travel_distance(self):
        self.problems =self.reset_state.problems
        solution = self.selected_node_list.clone()
        if self.check_validity:
            self.check_solution_validity()
        visited = torch.ones((solution.size(0), solution.size(1), self.problems.size(-2)))
        visited = visited.scatter(-1, solution, 0)
        penalty = (visited * self.problems[:, :, -1][:, None, :].expand(-1, solution.size(1), -1)).sum(-1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :2].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)
        travel_distances = segment_lengths[:, :, :-1].sum(2)
        # shape: (batch, pomo)
        return travel_distances + penalty



    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
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

        #当前点加mask
        gathering_index = selected.unsqueeze(-1)
        self.step_state.ninf_mask.scatter_(dim=-1, index=gathering_index, value=float('-inf'))

        self.load += self.reset_state.problems[:, :, 2][:, None, :].expand(-1, self.pomo_size, -1).gather(-1, selected[:, :, None]).squeeze()
        # load这里不需要加误差判断,最后判断合法性那里加了
        allow = (self.load >= 1.)| (self.selected_count==self.problem_size+1)  #奖励收集齐全，或者是全部点都访问过，允许访问终点
        self.step_state.ninf_mask[:,:,0][allow] = 0

        # 将finished的其他action设置为-inf
        self.finished = self.at_the_depot & (self.selected_count > 1)
        finished_extend = self.finished.unsqueeze(-1).expand(-1, -1, self.problem_size)
        self.step_state.ninf_mask[:,:, 1:][finished_extend] = float('-inf')

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.finished = self.finished


        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_open_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done


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

    def check_solution_validity(self):
        """Check that the solution is valid, i.e. contains all nodes once at most, and either prize constraint is met or all nodes are visited"""

        # Check that tours are valid, i.e. contain 0 to n -1
        actions = self.selected_node_list.clone()
        sorted_actions = actions.sort(-1)[0]

        # Make sure each node visited once at most (except for depot)
        assert (
                (sorted_actions[..., 1:] == 0)
                | (sorted_actions[..., 1:] > sorted_actions[..., :-1])
        ).all(), "Duplicates"

        prize = self.problems[:, :, 2]
        prize_with_dummy = prize.unsqueeze(-1)
        solution_expanded = actions.unsqueeze(-1)
        prize_expand = prize_with_dummy.unsqueeze(1).expand(-1, actions.size(1), -1, -1)
        prize_selected = prize_expand.gather(2, solution_expanded)
        prize_sum = prize_selected.sum(dim=2).squeeze(-1)

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
                (prize_sum >= 1 - 1e-5)
                | (
                        sorted_actions.size(-1) - (sorted_actions == 0).int().sum(-1)
                        == (self.problems.size(-2) - 1)
                )  # no depot
        ).all(), "Total prize does not satisfy min total prize"
