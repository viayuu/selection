import os
import pickle

import torch
from typing import Optional

from torch.utils.data import DataLoader
from torchrl.envs import EnvBase
from tensordict import TensorDict

from EasyNCO.data.data_utils import *
from EasyNCO.data import MVRPGenerator


class MVRPEnv(EnvBase):
    def __init__(self,
                 problem_size :int,
                 pomo_size : int = 1, # multi_start trajectory, in AM-based models, it is 1 by default
                 device : str = 'cpu',
                 seed : int = 2024,
                 **kwargs):
        super().__init__(device = device)
        self.env_name = 'mtvrp'
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.tem_pomo_size=pomo_size#b会改变pomo_size,避免之后别的问题pomo_size错误
        self.device = device
        self.mode=kwargs.get("mode",1)
        self.train_problems=kwargs.get("train_problems",None)
        self.test_problem=kwargs.get('test_problem',"CVRP")
        #setting
        self.backhaul_ratio=0.2
        self.speed = 1.0
        # self.depot_start= None
        # self.depot_end = None
        # tw for depot [0, 3]or [0,4.6]

        #node_feat
        self.depot_node_xy = None
        self.depot_node_demand=None
        self.depot_node_tw_start=None
        self.depot_node_tw_end=None
        self.depot_node_service_time = None
        self.route_limit=None
        self.problem_type=None
        self.problems = None
        self.capacity = None
        self.o_flag = None
        self.tw_flag=None
        self.b_flag=None
        self.l_flag=None
        # Dynamic
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.at_the_depot = None
        self.ninf_mask = None
        self.visited_ninf_flag = None
        self.finished = None
        self.current_time = None
        self.load = None
        self.length = None
        self.current_coord = None
        # shape: (batch, pomo, 2)

        self.dummy_flag_bool = None
        self.dummy_flag_long = None
        self.seed_value = seed
        self._set_seed(seed = seed)

        self.aug_type = kwargs.get('aug_type')
        self.aug_factor = kwargs.get('aug_factor')
        self.mix_prop = kwargs.get('mix_prop')
        self.aug_flag = self.aug_type is not None  # Used in baseline to prevent data from augment.


    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def generate_eval_instances(self, num_instances : int, batch_size : int, data_path: str = None,env_name="CVRP") -> DataLoader:
        '''
        Note that this function is necessary for the AM-based models.
        If you are using the AM-based models, you should implement this function according to specific CO problems.
        The function generates the evaluation instances for the CO problem.
        It is used to evaluate the trained model using greedyRollout baseline in AM-based models.
        :param num_instances:
        :param batch_size:
        :return: DataLoader
        '''
        eval_dataset = MVRPGenerator(num_instances,
                                    self.problem_size, batch_size, self.device, data_path,mode=self.mode,load_env_name=env_name)

        return eval_dataset

    def load_problems(self, dataset, batch_size : int = 64):
        self.pomo_size=self.tem_pomo_size
        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size, self.pomo_size])

        self.depot_node_xy = dataset["depot_node_xy"].to(self.device)
        self.depot_node_demand=dataset["depot_node_demand"].to(self.device)
        self.route_limit=dataset["route_limit"].to(self.device)
        self.depot_node_tw_start=dataset["depot_node_tw_start"].to(self.device)
        self.depot_node_tw_end=dataset["depot_node_tw_end"].to(self.device)
        self.depot_node_service_time = dataset["depot_node_service_time"].to(self.device)
        self.problem_type=dataset["problem_type"]
        assert len(set(self.problem_type))==1


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
            self.depot_node_xy = aug_operator[self.aug_type](problems=self.depot_node_xy, aug_factor=self.aug_factor, mix_prop=self.mix_prop)
            self.depot_node_demand = self.depot_node_demand.repeat(self.aug_factor,1, 1) # shape: (batch, 1 + problem)
            self.depot_node_service_time = self.depot_node_service_time.repeat(self.aug_factor, 1)
            self.depot_node_tw_start = self.depot_node_tw_start.repeat(self.aug_factor, 1)
            self.depot_node_tw_end = self.depot_node_tw_end.repeat(self.aug_factor, 1)
            self.route_limit=self.route_limit.repeat(self.aug_factor)
            self.problem_type=self.problem_type* self.aug_factor

        self.o_flag = torch.tensor(
            ["O" in p for p in self.problem_type],  # 显式列表推导
            dtype=torch.bool,  # 明确指定bool类型
            device=self.device  # 保持设备一致性
        )
        self.b_flag = torch.tensor(
            ["B" in p for p in self.problem_type],  # 显式列表推导
            dtype=torch.bool,  # 明确指定bool类型
            device=self.device  # 保持设备一致性
        )
        self.l_flag = torch.tensor(
            ["L" in p for p in self.problem_type],  # 显式列表推导
            dtype=torch.bool,  # 明确指定bool类型
            device=self.device  # 保持设备一致性
        )
        self.tw_flag = torch.tensor(
            ["TW" in p for p in self.problem_type],  # 显式列表推导
            dtype=torch.bool,  # 明确指定bool类型
            device=self.device  # 保持设备一致性
        )
        self.prob_emb = torch.stack([self.o_flag,self.b_flag,self.l_flag,self.tw_flag],dim=1).to(torch.float32)#o,b,l,tw
        self.depot_end=self.depot_node_tw_end[:,0]
        self.START_NODE = torch.arange(start=1, end=self.problem_size + 1)[None, :].expand(self.env_batch_size, -1)

        if self.b_flag.all()==True and self.mode==1:
            self.pomo_size = min(int(self.problem_size * (1 - self.backhaul_ratio)), self.pomo_size)
            self.batch_size = torch.Size([self.env_batch_size, self.pomo_size])
            self.START_NODE = self.START_NODE[self.depot_node_demand[:,1:,0] > 0].reshape(self.env_batch_size, -1)[:, :self.pomo_size]


        assert self.env_batch_size == self.depot_node_xy.size(0), \
            ('batch_size and the first dimension of problems should be the same. ' +
             f'Expected batch_size: {self.depot_node_xy.size(0)}, got: {self.env_batch_size}')


    def _reset(self, td : TensorDict, batch_size = None) -> TensorDict:
        self.output_spec=None  #解决空间batch_size变化，发生形状不匹配的原因
        self.current_node = None
        # shape: (batch, pomo)

        self.selected_count = torch.zeros((self.env_batch_size, self.pomo_size), dtype=torch.long)
        self.selected_node_list = torch.zeros((self.env_batch_size, self.pomo_size, 0), dtype = torch.long)
            # shape: (batch, pomo, 0~)

        self.ninf_mask = torch.zeros((self.env_batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem + 1)
        self.at_the_depot = torch.ones(size = (self.env_batch_size, self.pomo_size), dtype = torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size = (self.env_batch_size, self.pomo_size))
        self.capacity = torch.ones(size = (self.env_batch_size, self.pomo_size, 1))
        #self.capacity = self.load.unsqueeze(2)
        self.used_capacity = torch.zeros(size = (self.env_batch_size, self.pomo_size, 1))
        self.visited_ninf_flag = torch.zeros(size = (self.env_batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem + 1)
        # for done
        self.dummy_flag_bool = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.bool)
        # for action, -1 represents the dummy action, which is used to select the first node
        self.dummy_flag_long = torch.zeros((self.env_batch_size, self.pomo_size), dtype = torch.long) - 1
        self.finished = torch.zeros(size = (self.env_batch_size, self.pomo_size), dtype = torch.bool)

        #### mutli_task
        self.current_time = torch.zeros(size=(self.env_batch_size, self.pomo_size)).to(self.device)
        self.length = torch.zeros(size=(self.env_batch_size, self.pomo_size)).to(self.device)
        self.current_coord = self.depot_node_xy[:, [0], :].repeat(1,self.pomo_size,1)  # depot


        # shape: (batch, pomo)

        return TensorDict({
            'action' : self.dummy_flag_long,  # 'action': 'int',  # shape: (batch, pomo)
            'ninf_mask' : self.ninf_mask,
            'selected_count' : self.selected_count,
            'selected_node_list' : self.selected_node_list,
            'vehicle_capacity' : self.capacity,
            'used_capacity' : self.used_capacity,
            'reward' : self.dummy_flag_long, #*
            'done' : self.dummy_flag_bool,
            'depot_node_xy':self.depot_node_xy,
            'depot_node_demand':self.depot_node_demand,
            'depot_node_service_time':self.depot_node_service_time,
            'depot_node_tw_start':self.depot_node_tw_start,
            'depot_node_tw_end':self.depot_node_tw_end,
            'prob_emb':self.prob_emb,
            'open':self.o_flag,
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
            'finished':self.finished,
            'load':self.load,
            'current_time':self.current_time,
            'length':self.length,
            'open':self.o_flag[:,None].repeat(1,self.pomo_size),
            'start_node': self.START_NODE,
            'current_coord':self.current_coord
        }, batch_size = self.batch_size
        )

        return out

    def _step(self, td : TensorDict) -> TensorDict:
        #update state
        self.selected_count += 1
        self.current_node = td['action']
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim = 2)
        # shape: (batch, pomo, 0~)
        self.at_the_depot = (self.current_node == 0)
        demand_list = self.depot_node_demand.squeeze(-1)[:, None, :].expand(self.env_batch_size, self.pomo_size, -1)
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
        # 已经访问节点的标志更新
        self.visited_ninf_flag.scatter_(dim = -1, index = self.current_node.unsqueeze(-1), value = float('-inf'))
        # shape: (batch, pomo, problem + 1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0 # depot is considered unvisited, unless you are at the depot
        unvisited_demand = demand_list + self.visited_ninf_flag

        # B约束的处理，load的额外更新
        if self.b_flag.any() and self.mode==1:
            linehauls_unserved = torch.where(unvisited_demand > 0., True, False)  # 需求大于0为True，需求小于0，或者已经访问的false
            reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))  # 如果存在一个需求大于0的，后面为0，
            self.load[reset_index] = 0.  # 将在仓库，且所有需求都小于0的负载设置为0，

        #length更新
        current_coord = self.depot_node_xy[torch.arange(self.env_batch_size)[:, None], self.current_node]
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        self.length = self.length + new_length
        self.length[self.at_the_depot] = 0  # reset the length of route at the depot
        self.current_coord = current_coord



        #不同约束的mask处理：
        self.ninf_mask = self.visited_ninf_flag.clone()

        #1，demand(C)约束
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # # shape: (batch, pomo, problem+1)

        #2，B容量上限约束
        if self.b_flag.any() and self.mode==1:
            exceed_capacity = self.load[:, :, None] - demand_list > 1.0 + round_error_epsilon #负载在那些负的节点加上后会超过车的容量1,无法去该负节点
            self.ninf_mask[exceed_capacity] = float('-inf')

        #3，L路径长度约束(需要考虑是否回仓库点(O))
        if self.l_flag.any():
            route_limit = self.route_limit[:, None, None].expand(self.env_batch_size, self.pomo_size, self.problem_size+1)
            # shape: (batch, pomo, problem+1)
            # check route limit constraint: length + cur->next->depot <= route_limit
            route_too_large = self.length[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) + \
                              ~self.o_flag[:,None,None]*((self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)) > route_limit + round_error_epsilon
            route_too_large[self.o_flag, :, 0] = False
            self.ninf_mask[route_too_large] = float('-inf')

        #4_a,tw时间窗口约束 max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        #时间更新
        if self.tw_flag.any():
            self.current_time = torch.max(self.current_time + new_length / self.speed, self.depot_node_tw_start[
                                torch.arange(self.env_batch_size)[:, None], self.current_node]) + \
                                self.depot_node_service_time[torch.arange(self.env_batch_size)[:, None], self.current_node]
            self.current_time[self.at_the_depot] = 0
            # shape: (batch, pomo)
            arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed, self.depot_node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
            out_of_tw = arrival_time > self.depot_node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
            # shape: (batch, pomo, problem+1)
            out_of_tw[self.o_flag, :, 0] = False
            self.ninf_mask[out_of_tw] = float('-inf')
        #4_b,返回仓库节点时间窗约束，O与否,非O需要考虑，且 and self.mode==1
            if (~self.o_flag.any()) and self.mode==1:
                fail_return_depot = arrival_time + self.depot_node_service_time[:, None, :].expand(-1, self.pomo_size, -1) \
                                    + (self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed \
                                    > self.depot_end[:,None,None] + round_error_epsilon
                # shape: (batch, pomo, problem+1)
                self.ninf_mask[fail_return_depot] = float('-inf')



        # time window constraint
        #   current_time: the end time of serving the current node
        #   a. max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        #   b. vehicle should return to the depot: max(current_time + travel_time, tw_start) + service_time + dist(node, depot)/speed <= self.depot_end

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
            # 'PROBLEM': self.env_name,
            'current_node': self.current_node,
            'current_coord': self.current_coord,
            'finished': self.finished,
            'load': self.load,
            'current_time': self.current_time,
            'length': self.length,
            'open': self.o_flag[:,None].repeat(1,self.pomo_size),
            'start_node': self.START_NODE,

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

        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem + 1, 2)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        if self.o_flag.all():
            not_to_depot = self.selected_node_list.roll(dims=2, shifts=-1) != 0

            segment_length = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # shape: (batch, pomo, selected_list_length)
            travel_distances = (segment_length * not_to_depot).sum(2)
        else:
            segment_length = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # shape: (batch, pomo, selected_list_length)

            travel_distances = segment_length.sum(2)
            # shape(batch, pomo)

        return travel_distances