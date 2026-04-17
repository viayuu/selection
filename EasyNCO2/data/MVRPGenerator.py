import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

####################################
# DATA
####################################
def MVRPGenerator(data_size, problem_size, batch_size, device = 'cpu', path = None, demand_scaler=None,
                            partial_updater=None, train_problems=None, test_problem=None,**kwargs):
    if train_problems is None:
        train_problems = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"]
    mode=kwargs.get("mode",0)
    if path is None:
        dataset = random_vrp_generator(num_sample = data_size,
                                       num_nodes = problem_size,
                                       demand_scaler = demand_scaler,
                                       batch_size=batch_size,
                                       mode=mode,
                                       train_problems=train_problems,
                                       device = device,
                                       )
    else:
        dataset = customized_vrp_loader(num_sample = data_size,
                                        num_nodes = problem_size,
                                        mode=mode,
                                        device = device,
                                        test_problem=test_problem,
                                        path=path)
        if partial_updater is not None:
            dataset.data['to_DataLoader'] = partial_updater.load_problem_one_epoch(dataset)
            batch_size = 1
    data_loader = DataLoader(dataset = dataset,
                             batch_size = batch_size,
                             shuffle = False,
                             num_workers = 0,
                             collate_fn = None)

    return data_loader

class random_vrp_generator(Dataset):
    '''
    The random_vrp_generator is used to generate the random VRP data.
    The data is generated randomly with the size of [num_sample, num_nodes, demand, 3]
    By default, the data is generated with the uniform distribution.
    '''
    def __init__(self, num_sample, num_nodes, demand_scaler=None,batch_size=64,mode=0,train_problems=None,device = 'cpu',** kwargs):
        self.train_problem=train_problems
        self.current_env=train_problems[0]
        self.batch_size=batch_size
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.mode=mode
        self.device = device
        self.demand_scaler = demand_scaler
        self.limit= kwargs.get('aug_factor', 3.0)
        self.depot_start=kwargs.get("depot_start",torch.tensor(0.,device=self.device))
        if self.mode==0:
            self.depot_end=kwargs.get("depot_end",torch.tensor(4.6,device=self.device))
        elif self.mode==1:
            self.depot_end=kwargs.get("depot_end",torch.tensor(3.0,device=self.device))
        self.speed=kwargs.get("speed",1.)
        self.backhaul_ratio=kwargs.get("backhaul_ratio",0.2)
    def __getitem__(self, index):
        if index%self.batch_size==0:
            self.current_env=random.sample(self.train_problem, 1)[0]
        limit_flag, tw_flag, backhaul_flag = False, False, False
        if 'L' in self.current_env:
            limit_flag=True
        if "TW" in self.current_env:
            tw_flag=True
        if "B" in self.current_env:
            backhaul_flag=True


        depot_node_xy_data = torch.rand(size = (self.num_nodes+1, 2)).to(self.device)
        if self.demand_scaler is not None:
            demand_scaler = self.demand_scaler
        else:
            if self.num_nodes == 20:
                demand_scaler = 30
            elif self.num_nodes == 50:
                demand_scaler = 40
            elif self.num_nodes == 100:
                demand_scaler = 50
            elif self.num_nodes == 1000:
                demand_scaler = 200
            elif self.num_nodes == 2000:
                demand_scaler = 300
            elif self.num_nodes == 5000:
                demand_scaler = 300
            elif self.num_nodes == 7000:
                demand_scaler = 300
            else:
                raise ValueError(f"The default demand_scaler is not supported for the problem size of {self.num_nodes}, " +
                                 f"please specify the demand_scaler")
        depot_node_demand = torch.randint(1, 10, size = (self.num_nodes + 1,)) / float(demand_scaler)
        if backhaul_flag:
            backhauls_index = torch.randperm(self.num_nodes)[:int(
                self.num_nodes * self.backhaul_ratio)] + 1  # randomly select 20% customers as backhaul ones
            depot_node_demand[backhauls_index] = -1 * depot_node_demand[backhauls_index]
        depot_node_demand[0] = 0
        depot_node_demand = depot_node_demand.unsqueeze(1)
        route_limit = torch.tensor(0.0, device=self.device)  # 默认值
        depot_node_service_time = torch.zeros(self.num_nodes+1, device=self.device)  # 默认值
        depot_node_tw_start = torch.zeros(self.num_nodes+1, device=self.device)  # 默认值
        depot_node_tw_end = torch.zeros(self.num_nodes+1, device=self.device)  # 默认值

        if limit_flag:
            route_limit = torch.tensor(1.0, device=self.device) * self.limit
        if tw_flag:
            if self.mode==0:
                depot_node_service_time,depot_node_tw_start,depot_node_tw_end=get_tw_data_0(self.num_nodes,depot_node_xy_data,
                    speed=self.speed,depot_start=self.depot_start.unsqueeze(0),depot_end=self.depot_end.unsqueeze(0),device=self.device)
            elif self.mode==1:
                depot_node_service_time,depot_node_tw_start,depot_node_tw_end=get_tw_data_1(self.num_nodes,depot_node_xy_data,
                    speed=self.speed,depot_start=self.depot_start.unsqueeze(0),depot_end=self.depot_end.unsqueeze(0),device=self.device)

        return {    'depot_node_xy': depot_node_xy_data,
                    'depot_node_demand':depot_node_demand,
                    'route_limit':route_limit,
                    'depot_node_service_time': depot_node_service_time,
                    'depot_node_tw_start': depot_node_tw_start,
                    'depot_node_tw_end': depot_node_tw_end,
                    'problem_type':self.current_env
                }


    def __len__(self):
        return self.num_sample

class customized_vrp_loader(Dataset):
    '''
    The customized_vrp_loader is used to load the customized VRP data from the file.
    The file should be saved in the format of .pkl or .pt.
    And the data should be saved in the format of {'depot_xy': shape (sample_num, 1, 2),
                                                    'node_xy': shape (sample_num, problem, 2),
                                                    'node_demand': shape (sample_num, problem),
                                                    (if learning paradigm is sl, 'label' will be contained)}
    In the future, we will support the more general type of data.
    '''
    def __init__(self, num_sample, num_nodes, mode,device='cpu',test_problem="CVRP", path=None):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1] # maybe .pkl or .pt
        self.data = None
        self.depot_start=0
        if "TW" in test_problem:
            if mode==0:
                self.depot_end=4.6
            else:
                self.depot_end=3.0
        else:
            self.depot_end=0.0
        if self.file_type == '.pkl':
            import pickle
            # with open(self.path, 'rb') as f:
            #     self.data = torch.tensor(pickle.load(f)[:self.num_sample], dtype = torch.float32)

            with open(path, 'rb') as f:
                data = pickle.load(f)[:self.num_sample]
            if test_problem in ["CVRP","OVRP","VRPB","OVRPB"]:
                depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], \
                                                        [i[2] for i in data], [i[3] for i in data]

                depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(
                    node_demand), torch.Tensor(capacity)
                route_limit=torch.zeros(size=(self.num_sample,))
                depot_node_service_time=torch.zeros(size=(self.num_sample,node_xy.shape[1]+1))
                depot_node_tw_start=torch.zeros(size=(self.num_sample,node_xy.shape[1]+1))
                depot_node_tw_end=torch.zeros(size=(self.num_sample,node_xy.shape[1]+1))
            elif test_problem in ["VRPL","OVRPL","VRPBL","OVRPBL"]:
                depot_xy, node_xy, node_demand, capacity, route_limit = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data], [i[4] for i in data]
                depot_xy, node_xy, node_demand, capacity, route_limit = torch.Tensor(depot_xy), torch.Tensor(node_xy), \
                                            torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(route_limit)
                depot_node_service_time=torch.zeros(size=(self.num_sample,node_xy.shape[1]+1))
                depot_node_tw_start=torch.zeros(size=(self.num_sample,node_xy.shape[1]+1))
                depot_node_tw_end=torch.zeros(size=(self.num_sample,node_xy.shape[1]+1))
            elif test_problem in ["VRPTW","OVRPTW","VRPBTW","OVRPBTW"]:
                depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], \
                    [ i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in data]
                depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = torch.Tensor(depot_xy), torch.Tensor(node_xy),\
                    torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
                route_limit=torch.zeros(size=(self.num_sample,))
                depot_node_service_time=torch.cat((torch.zeros(depot_xy.shape[0],1,device=self.device),service_time),dim=1)
                depot_node_tw_start=torch.cat((torch.full((depot_xy.shape[0],1),self.depot_start),tw_start),dim=1)
                depot_node_tw_end=torch.cat((torch.full((depot_xy.shape[0],1),self.depot_end),tw_end),dim=1)
            elif test_problem in ["VRPLTW","OVRPLTW","VRPBLTW","OVRPBLTW"]:
                depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = [i[0] for i in data], \
                    [i[1] for i in data], [ i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in data], [ i[7] for i in data]
                depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end = torch.Tensor( depot_xy), torch.Tensor(node_xy), \
                    torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor( route_limit), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
                depot_node_service_time=torch.cat((torch.zeros(depot_xy.shape[0],1,device=self.device),service_time),dim=1)
                depot_node_tw_start=torch.cat((torch.full((depot_xy.shape[0],1),self.depot_start),tw_start),dim=1)
                depot_node_tw_end=torch.cat((torch.full((depot_xy.shape[0],1),self.depot_end),tw_end),dim=1)
            else:
                raise ValueError(f"Unsupported problem: {test_problem}")
            node_demand = node_demand / capacity.view(-1, 1)
            depot_node_demand = torch.cat((torch.zeros(size=(self.num_sample, 1)), node_demand),
                                          dim=1).unsqueeze(-1)  # shape: (batch, 1+num_nodes)
            depot_node_xy=torch.cat((depot_xy, node_xy),dim=1)
            prob_type=[test_problem for i in range(depot_xy.shape[0])]
            self.data={'depot_node_xy': depot_node_xy,
                       'depot_node_demand':depot_node_demand,
                       'route_limit': route_limit,
                       'depot_node_service_time': depot_node_service_time,
                         'depot_node_tw_start': depot_node_tw_start,
                         'depot_node_tw_end': depot_node_tw_end,
                         'problem_type':prob_type
             }
        elif self.file_type == '.pt':

            loaded_dict = torch.load(self.path, map_location=device)
            depot_xy = loaded_dict['depot_xy'][:self.num_sample]
            node_xy = loaded_dict['node_xy'][:self.num_sample]
            node_demand = loaded_dict['node_demand'][:self.num_sample]
            depot_node_demand = torch.cat((torch.zeros(size=(self.num_sample, 1)), node_demand),
                                          dim=1).unsqueeze(-1)  # shape: (batch, 1+num_nodes)
            depot_node_xy=torch.cat((depot_xy, node_xy),dim=1)
            route_limit = torch.zeros(self.num_sample, device=device)  # 默认无路线长度限制
            service_time = torch.zeros_like(node_demand, device=device)  # 默认服务时间为0
            tw_start = torch.zeros_like(node_demand, device=device)  # 默认时间窗开始时间为0
            tw_end = torch.full_like(node_demand, 0, device=device)  # 默认时间窗结束时间为1
            if test_problem in ["VRPL", "OVRPL", "VRPBL", "OVRPBL", "VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
                route_limit = loaded_dict['route_length_limit'][:self.num_sample][:, 1]

            if test_problem in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW", "VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
                tw_start = loaded_dict['node_earlyTW'][:self.num_sample]
                tw_end = loaded_dict['node_lateTW'][:self.num_sample]
                service_time = loaded_dict['node_serviceTime'][:self.num_sample]
            depot_node_service_time = torch.cat((torch.zeros(depot_xy.shape[0], 1, device=self.device), service_time),
                                                dim=1)
            depot_node_tw_start = torch.cat((torch.full((depot_xy.shape[0], 1), self.depot_start), tw_start), dim=1)
            depot_node_tw_end = torch.cat((torch.full((depot_xy.shape[0], 1), self.depot_end), tw_end), dim=1)
            prob_type=[test_problem for i in range(depot_xy.shape[0])]
            self.data={'depot_node_xy': depot_node_xy,
                       'depot_node_demand':depot_node_demand,
                       'route_limit': route_limit,
                       'depot_node_service_time': depot_node_service_time,
                         'depot_node_tw_start': depot_node_tw_start,
                         'depot_node_tw_end': depot_node_tw_end,
                         'problem_type':prob_type
             }
        else:
            raise NotImplementedError(f"The file type ({self.file_type}) is not supported in current version")


    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}
    def __len__(self):
        if isinstance(self.data, dict) and 'to_DataLoader' in self.data.keys():
            return len(self.data['problem'])

        else:
            return self.num_sample



def get_tw_data_0(num_nodes,depot_node_xy,speed,depot_start,depot_end,device):
    # time windows (vehicle speed = 1.):
    #   1. The setting of "MTL for Routing Problem with Zero-Shot Generalization".
    node_xy=depot_node_xy[1:]
    depot_xy=depot_node_xy[[0]]
    node_serviceTime = torch.rand(size=(num_nodes,),device=device) * 0.05 + 0.15
    # shape: (batch, problem)
    # range: (0.15,0.2) for T=4.6
    node_lengthTW = torch.rand(size=(num_nodes,),device=device) * 0.05 + 0.15
    # shape: (batch, problem)
    # range: (0.15,0.2) for T=4.6
    d0i = ((node_xy - depot_xy) ** 2).sum(1).sqrt()
    # shape: (batch, problem)
    ei = torch.rand(size=(num_nodes,),device=device).mul((torch.div(
        (4.6 * torch.ones(size=(num_nodes,),device=device) - node_serviceTime - node_lengthTW),
        d0i) - 1) - 1) + 1
    # shape: (batch, problem)
    # default velocity = 1.0
    node_earlyTW = ei.mul(d0i)
    # shape: (batch, problem)
    # default velocity = 1.0
    node_lateTW = node_earlyTW + node_lengthTW
    depot_node_service_time=torch.cat((torch.zeros(1,device=device),node_serviceTime),dim=0)
    depot_node_tw_start=torch.cat((depot_start,node_earlyTW),dim=0)
    depot_node_tw_end=torch.cat((depot_end,node_lateTW),dim=0)
    return depot_node_service_time,depot_node_tw_start,depot_node_tw_end

def get_tw_data_1(num_nodes,depot_node_xy,speed,depot_start,depot_end,device="cpu"):
    #   2. See "Learning to Delegate for Large-scale Vehicle Routing" in NeurIPS 2021.
    #   Note: this setting follows a similar procedure as in Solomon, and therefore is more realistic and harder.
    node_xy_data=depot_node_xy[1:]
    depot_xy=depot_node_xy[[0]]
    service_time = torch.ones(num_nodes,device=device) * 0.2
    travel_time = (node_xy_data - depot_xy).norm(p=2, dim=-1) / speed
    a, b = depot_start + travel_time, depot_end - travel_time - service_time  # a代表每个节点的最早开始时间，b代表每个窗口的最晚解释时间，这是个范围
    time_centers = (a - b) * torch.rand(num_nodes) + b  # 随机定义选取范围内的某个值作为每个点的中心
    time_half_width = (service_time / 2 - depot_end / 3) * torch.rand(num_nodes) + depot_end / 3
    tw_start = torch.clamp(time_centers - time_half_width, min=depot_start, max=depot_end)
    tw_end = torch.clamp(time_centers + time_half_width, min=depot_start, max=depot_end)
    # shape: (batch, problem)
    # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
    instance_invalid, round_error_epsilon = False, 0.00001
    total_time = torch.max(0 + (depot_xy - node_xy_data).norm(p=2, dim=-1) / speed,
                           tw_start) + service_time + (
                         node_xy_data - depot_xy).norm(p=2,
                                                       dim=-1) / speed > depot_end + round_error_epsilon
    # (batch, problem)
    instance_invalid = total_time.any()

    if instance_invalid:
        print(">> Invalid instances, Re-generating ...")
        return get_tw_data_1(num_nodes,depot_node_xy,speed=speed,depot_start=depot_start,depot_end=depot_end)
    depot_node_service_time=torch.cat((torch.zeros(1,device=device),service_time),dim=0)
    depot_node_tw_start=torch.cat((depot_start,tw_start),dim=0)
    depot_node_tw_end=torch.cat((depot_end,tw_end),dim=0)

    return depot_node_service_time,depot_node_tw_start,depot_node_tw_end