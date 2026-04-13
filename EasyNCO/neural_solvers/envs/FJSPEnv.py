import torch
from torchrl.envs import EnvBase
from tensordict import TensorDict, TensorDictBase
from typing import Optional
from EasyNCO.neural_solvers.methods.drl_hgnn.utils import load_fjs,nums_detec,get_ope_num
import copy
import numpy as np
import re

def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(EnvBase):
    def __init__(self,
                 device: str = 'cpu',
                 seed: int = 2024,
                 **kwargs):
        super().__init__(device=device)

        self.num_job = kwargs.get('num_job', 10)
        self.num_machine = kwargs.get('num_machine', 10)
        self.time_low = kwargs.get("time_low", 1)
        self.time_high = kwargs.get("time_high", 20)
        self.num_sample = kwargs.get("num_sample", 1)
        self.test_strategy = kwargs.get("test_strategy", 'greedy')

        self.device = device
        self.seed_value = seed
        self.env_name = "fjsp"
        self.aug_type = None
        self.pomo_size = 1  # dummy
        self._set_seed(seed=self.seed_value)


    def load_problems(self, dataset, batch_size: int = 1):
        # just for drl_hgnn


        self.problems = dataset
        if self.test_strategy == 'sampling':
            batch_size = self.num_sample

        self.env_batch_size = batch_size
        self.batch_size = torch.Size([batch_size])
        self.problem_size = str(self.num_job.item()) + "×" + str(self.num_machine.item())


        unpadded_tensor = self.problems.squeeze(0)  # shape: (N, L)
        # 转为 list 并去除 padding
        original_list = []
        for row in unpadded_tensor:
            values = row.tolist()
            values = [v for v in values if v != -1]  # 去除 padding 值
            original_list.append(values)

        num_data = 8
        tensors = [[] for _ in range(num_data)]

        self.num_opes = get_ope_num(original_list)

        load_data = load_fjs(original_list, self.num_machine, self.num_opes)
        for j in range(num_data):
            tensors[j].append(load_data[j])
            tensors[j] = [tensor.clone() for _ in range(self.env_batch_size) for tensor in tensors[j]]

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_job), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_job), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_job), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.env_batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.env_batch_size)  # Current time of the environment
        self.N = torch.zeros(self.env_batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_job), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        # Generate raw feature vectors
        ope_feat_dim = 6
        ma_feat_dim = 3
        feat_opes_batch = torch.zeros(size=(self.env_batch_size, ope_feat_dim, self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.env_batch_size, ma_feat_dim, self.num_machine))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_job), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.env_batch_size, self.num_job), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_job), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.env_batch_size, self.num_job), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.env_batch_size, self.num_machine), dtype=torch.bool, fill_value=False)

        self.schedules_batch = torch.zeros(size=(self.env_batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]

        self.machines_batch = torch.zeros(size=(self.env_batch_size, self.num_machine, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.env_batch_size, self.num_machine))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = TensorDict({
            # dynamic
            "batch_idxes": self.batch_idxes,
            "feat_opes_batch": self.feat_opes_batch,
            "feat_mas_batch": self.feat_mas_batch,
            "proc_times_batch": self.proc_times_batch,
            "ope_ma_adj_batch": self.ope_ma_adj_batch,
            "mask_job_procing_batch": self.mask_job_procing_batch,
            "mask_job_finish_batch": self.mask_job_finish_batch,
            "mask_ma_procing_batch": self.mask_ma_procing_batch,
            "ope_step_batch": self.ope_step_batch,
            "time_batch": self.time,
            # static
            "opes_appertain_batch": self.opes_appertain_batch,
            "ope_pre_adj_batch": self.ope_pre_adj_batch,
            "ope_sub_adj_batch": self.ope_sub_adj_batch,
            "end_ope_biases_batch": self.end_ope_biases_batch,
            "nums_opes_batch": self.nums_opes,
        }, batch_size=[batch_size])





    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(self.seed_value)
        self.rng.set_state(state["rng"])

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.dummy_flag_action = torch.zeros((self.env_batch_size, 3), dtype=torch.int64)-1
        self.dummy_flag_reward = torch.zeros((self.env_batch_size), dtype=torch.int64)
        self.dummy_flag_bool = torch.zeros((self.env_batch_size), dtype=torch.bool)

        return TensorDict({
            'action': self.dummy_flag_action,  # 'action': 'int',  # shape: (batch, 3) operation,machine,job
            "reward": self.dummy_flag_reward,
            "done": self.dummy_flag_bool,
        }, batch_size=torch.Size([self.env_batch_size])
        )

    def _step(self, td: TensorDict) -> TensorDict:
        '''
        Environment transition function
        '''
        action = td['action'].t()

        opes = action[0, :]
        mas = action[1, :]
        jobs = action[2, :]
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.env_batch_size, self.num_machine), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() \
                         * un_scheduled  # estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                  self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                                              self.batch_idxes, :])

        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas),
                                                                       dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :],
                                                                          dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max
        self.makespan_batch = max

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Update the vector for uncompleted instances
        # mask_finish = (self.N + 1) <= self.nums_opes
        # if ~(mask_finish.all()):
        #     self.batch_idxes = torch.arange(self.env_batch_size)[mask_finish] #batch_size!=1时这里要怎么改

        # Update state of the environment
        self.state.update({
            "batch_idxes": self.batch_idxes,
            "feat_opes_batch": self.feat_opes_batch,
            "feat_mas_batch": self.feat_mas_batch,
            "proc_times_batch": self.proc_times_batch,
            "ope_ma_adj_batch": self.ope_ma_adj_batch,
            "mask_job_procing_batch": self.mask_job_procing_batch,
            "mask_job_finish_batch": self.mask_job_finish_batch,
            "mask_ma_procing_batch": self.mask_ma_procing_batch,
            "ope_step_batch": self.ope_step_batch,
            "time_batch": self.time,  # 注意 key 名是 time_batch，不是 time
        })

        next_state =  {
            "state": self.state,
            "reward": self.reward_batch,
            "done": self.done_batch,
        }

        out = TensorDict({
            "next": next_state,
            "reward": self.reward_batch,
            "done": self.done_batch,
        }, batch_size=self.batch_size
        )

        return out
        

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        
        
    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_machine)] for __ in range(self.env_batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.env_batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_machine):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.env_batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_job):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch