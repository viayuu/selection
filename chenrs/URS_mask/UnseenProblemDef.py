import torch
import numpy as np
import random
import os
from collections import namedtuple
import pickle

# from ProblemDef import get_random_problems_vrp_mixed_mvmoe_version


def get_random_problems_pdcvrptw(batch_size, problem_size,capacity,**kwargs):

    normalized = True
    speed = 1.0
    depot_start, depot_end = 0., 3.
    problem_name = kwargs.get("problem_name",None)

    depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

    demand_scaler = capacity

    #tw constraint
    service_time = torch.ones(batch_size, problem_size) * 0.2
    travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / speed
    a, b = depot_start + travel_time, depot_end - travel_time - service_time
    time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
    time_half_width = (service_time / 2 - depot_end / 3) * torch.rand(batch_size,problem_size) + depot_end / 3
    tw_start = torch.clamp(time_centers - time_half_width, min=depot_start, max=depot_end)
    tw_end = torch.clamp(time_centers + time_half_width, min=depot_start, max=depot_end)
    # shape: (batch, problem)

    # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
    instance_invalid, round_error_epsilon = False, 0.00001
    total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / speed, tw_start) + service_time + (
                node_xy - depot_xy).norm(p=2, dim=-1) / speed > depot_end + round_error_epsilon
    # (batch, problem)
    instance_invalid = total_time.any()

    if instance_invalid:
        print(">> Invalid instances, Re-generating ...")
        return get_random_problems_pdcvrptw(batch_size, problem_size,capacity)
    elif normalized:
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)

        return depot_xy, node_xy, node_demand, service_time, tw_start, tw_end


def get_random_problems_avrpmix(batch_size, problem_size, capacity, **kwargs):
    ################################
    # "tmat" type
    # Following MatNet: https://github.com/yd-kwon/MatNet/blob/main/ATSP/ATSProblemDef.py
    ################################
    problem_gen_params = {
        'int_min': 0,
        'int_max': 1000 * 1000,
        'scaler': 1000 * 1000
    }

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, problem_size + 1, problem_size + 1))
    # shape: (batch, node, node)
    problems[:, torch.arange(problem_size + 1), torch.arange(problem_size + 1)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    demand = torch.randint(1, 10, size=(batch_size, problem_size + 1))
    # shape: (batch, problem)

    node_demand = demand / float(capacity)

    node_demand[:, 0] = 0

    # for variants of vrp
    speed = 1.0
    depot_start, depot_end = 0., 1.
    backhaul_ratio = 0.2
    problem_name = kwargs.get("problem_name", None)

    # l constraints
    if 'l' in problem_name:
        route_limit = torch.ones(batch_size) * 0.5
    else:
        route_limit = torch.ones(batch_size) * float("inf")

    # b constraints
    if 'b' in problem_name:
        backhauls_index = torch.randperm(problem_size)[
                          :int(problem_size * backhaul_ratio)]  # randomly select 20% customers as backhaul ones
        node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]

    # tw constraints
    if 'tw' in problem_name:
        dist_matrix = scaled_problems
        travel_time_depot_to_node = dist_matrix[:, 0, 1:] / speed
        travel_time_node_to_depot = dist_matrix[:, 1:, 0] / speed

        service_time = torch.ones(batch_size, problem_size) * 0.2
        a, b = depot_start + travel_time_depot_to_node, depot_end - travel_time_node_to_depot - service_time
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (service_time / 2 - depot_end / 3) * torch.rand(batch_size, problem_size) + depot_end / 3
        tw_start = torch.clamp(time_centers - time_half_width, min=depot_start, max=depot_end)
        tw_end = torch.clamp(time_centers + time_half_width, min=depot_start, max=depot_end)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + travel_time_depot_to_node,
                               tw_start) + service_time + travel_time_node_to_depot > depot_end + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return get_random_problems_avrpmix(batch_size, problem_size, capacity, problem_name=problem_name)
    else:
        service_time = torch.zeros(batch_size, problem_size)
        tw_start = torch.zeros(batch_size, problem_size)
        tw_end = torch.ones(batch_size, problem_size) * float("inf")

    data_dict = {
        'dist': scaled_problems,
        'demand': node_demand,
        'capacity': capacity,
        'route_limit': route_limit,
        'service_time': service_time,
        'tw_start': tw_start,
        'tw_end': tw_end,
    }
    return data_dict


# TSPTW
TSPTW_SET = namedtuple("TSPTW_SET",
                       ["node_loc",  # Node locations 1
                        "node_tw",  # node time windows 5
                        "durations",  # service duration per node 6
                        "service_window",  # maximum of time units 7
                        "time_factor", "loc_factor"])


def gen_tw(size, graph_size, time_factor, dura_region, rnds):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    service_window = int(time_factor * 2)

    horizon = np.zeros((size, graph_size, 2))
    horizon[:] = [0, service_window]

    # sample earliest start times
    tw_start = rnds.randint(horizon[..., 0], horizon[..., 1] / 2)
    tw_start[:, 0] = 0

    # calculate latest start times b, which is
    # tw_start + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with tw_start to create the time windows
    epsilon = rnds.uniform(dura_region[0], dura_region[1], (tw_start.shape))
    duration = np.around(time_factor * epsilon)
    duration[:, 0] = service_window
    tw_end = np.minimum(tw_start + duration, horizon[..., 1]).astype(int)

    tw = np.concatenate([tw_start[..., None], tw_end[..., None]], axis=2).reshape(size, graph_size, 2)

    return tw


def generate_tsptw_data(size, graph_size, rnds=None, time_factor=100.0, loc_factor=100, tw_duration="5075"):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    rnds = np.random if rnds is None else rnds
    service_window = int(time_factor * 2)

    # sample locations
    nloc = rnds.uniform(size=(size, graph_size, 2)) * loc_factor  # node locations

    # tw duration
    dura_region = {
        "5075": [.5, .75],
        "1020": [.1, .2],
    }
    if isinstance(tw_duration, str):
        dura_region = dura_region[tw_duration]
    else:
        dura_region = tw_duration

    tw = gen_tw(size, graph_size, time_factor, dura_region, rnds)

    return TSPTW_SET(node_loc=nloc,  # tsptw100 = 1depot+99customer
                     node_tw=tw,  # batch,problem_size,2
                     durations=tw[..., 1] - tw[..., 0],
                     service_window=[service_window] * size,
                     time_factor=[time_factor] * size,
                     loc_factor=[loc_factor] * size, )


def get_random_problems_tsptw(batch_size, problem_size, coord_factor=100, max_tw_size=100, hardness='easy'):
    speed = 1

    if hardness == "hard":
        # Taken from DPDP (Kool et. al)
        # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
        # max_tw_size = 1000 if tw_type == "da_silva" else 100
        # max_tw_size = problem_size * 2 if tw_type == "da_silva" else 100
        """
        :param problem_size: number of cities
        :param grid_size (=1): x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
        :return: a feasible TSPTW instance randomly generated using the parameters
        """
        node_xy = torch.rand(size=(batch_size, problem_size, 2)) * coord_factor  # (batch, problem, 2)
        travel_time = torch.cdist(node_xy, node_xy, p=2,
                                  compute_mode='donot_use_mm_for_euclid_dist') / speed  # (batch, problem, problem)

        random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
        for i in range(batch_size):
            random_solution[i] = random_solution[i][torch.randperm(random_solution.size(1))]
        zeros = torch.zeros(size=(batch_size, 1)).long()
        random_solution = torch.cat([zeros, random_solution], dim=1)

        time_windows = torch.zeros((batch_size, problem_size, 2))
        time_windows[:, 0, :] = torch.tensor([0, 1000. * coord_factor]).repeat(batch_size, 1)

        total_dist = torch.zeros(batch_size)
        for i in range(1, problem_size):
            prev_city = random_solution[:, i - 1]
            cur_city = random_solution[:, i]

            cur_dist = travel_time[torch.arange(batch_size), prev_city, cur_city]

            # tw_lb_min = time_windows[torch.arange(batch_size), prev_city, 0] + cur_dist
            total_dist += cur_dist

            # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
            rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
            rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist

            time_windows[torch.arange(batch_size), cur_city, :] = torch.cat(
                [rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)

    elif hardness in ["easy", "medium"]:

        tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=problem_size * 55,
                                 tw_duration="5075" if hardness == "easy" else "1020")
        node_xy = torch.tensor(tw.node_loc).float()
        time_windows = torch.tensor(tw.node_tw)

    else:
        raise NotImplementedError

    service_time = torch.zeros(size=(batch_size, problem_size))
    # Don't store travel time since it takes up much
    return node_xy, service_time, time_windows[:, :, 0], time_windows[:, :, 1]


# USE SAVED DATA
def use_saved_problems_tsptw_pkl(filename, total_episodes, device, start=0, solution_name=None):
    with open(filename, 'rb') as f1:
        data = pickle.load(f1)[start:start + total_episodes]  # node_xy, service_time, tw_start, tw_end

    node_xy, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3]
                                                                                                                 for i
                                                                                                                 in
                                                                                                                 data]
    node_xy, service_time, tw_start, tw_end = torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(
        tw_start), torch.Tensor(tw_end)

    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:start + total_episodes]
            optimal_score = [i[0] for i in out_2]
            optimal_score_all = torch.tensor(optimal_score, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1

    dataset_dict = {
        'node_xy': node_xy,
        'service_time': service_time,
        'tw_start': tw_start,
        'tw_end': tw_end,
    }

    return dataset_dict, optimal_score


def use_saved_problems_mdcvrp_npz(filename, total_episodes, device, start=0, solution_name=None):
    data = np.load(filename)  # ['locs', 'demand_linehaul', 'vehicle_capacity', 'speed', 'num_depots']
    node_demands = torch.tensor(data['demand_linehaul'], device=device, dtype=torch.float32)[
                   start:start + total_episodes]  # 不包含起点,已经norm过
    node_coords = torch.tensor(data['locs'], device=device, dtype=torch.float32)[start:start + total_episodes]
    depot_xy = node_coords[:, :3, :]
    node_xy = node_coords[:, 3:, :]

    if solution_name is not None:
        solution = np.load(solution_name)  # ['actions','costs','time']
        tour_lens = torch.tensor(solution['costs'], device=device, dtype=torch.float32)[start:start + total_episodes]
        tour_lens *= -1  # 记录的是负的
        optimal_score = tour_lens.mean()
    else:
        optimal_score = 1.0

    xy = torch.cat((depot_xy, node_xy), dim=1)
    depot_demand = torch.zeros(size=(total_episodes, 3))
    demand = torch.cat((depot_demand, node_demands), dim=-1)
    data_dict = {
        'xy': xy,
        "demand": demand
    }

    return data_dict, optimal_score


def use_saved_problems_mdcvrptw_npz(filename, total_episodes, device, start=0, solution_name=None):
    data = np.load(filename)  # ['locs', 'demand_linehaul', 'vehicle_capacity', 'speed', 'num_depots','time_windows,service_time]
    node_demands = torch.tensor(data['demand_linehaul'], device=device, dtype=torch.float32)[
                   start:start + total_episodes]  # 不包含起点,已经norm过
    xy = torch.tensor(data['locs'], device=device, dtype=torch.float32)[start:start + total_episodes]
    time_windows = torch.tensor(data['time_windows'], device=device, dtype=torch.float32)[start:start + total_episodes]
    service_time = torch.tensor(data['service_time'], device=device, dtype=torch.float32)[start:start + total_episodes]

    if solution_name is not None:
        solution = np.load(solution_name)  # ['actions','costs','time']
        tour_lens = torch.tensor(solution['costs'], device=device, dtype=torch.float32)[start:start + total_episodes]
        tour_lens *= -1  # 记录的是负的
        optimal_score = tour_lens.mean()
    else:
        optimal_score = 1.0

    depot_demand = torch.zeros(size=(total_episodes, 3))
    demand = torch.cat((depot_demand, node_demands), dim=-1)
    data_dict = {
        'xy': xy,
        "demand": demand
    }

    return data_dict, optimal_score


def use_saved_problems_acvrp_pt(filename, total_episodes, device, start=0, solution_name=None):
    data = torch.load(filename, map_location=device)
    dist_matrix = data['node_matrix'][start:start + total_episodes].to(device)
    demand = data['demand'][start:start + total_episodes].to(device)
    optimal = data['result'][start:start + total_episodes].mean()

    data_dict = {
        "dist": dist_matrix,
        "demand": demand,
    }

    return data_dict, optimal

def use_saved_problems_hcvrp_pkl(filename, total_episodes, device, start=0, solution_name=None):
    with open(filename, 'rb') as f1:
        data = pickle.load(f1)

    depot, node_xy, demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]

    depot, node_xy, demand, capacity = torch.Tensor(depot), torch.Tensor(node_xy), torch.Tensor(demand), torch.Tensor(capacity)


    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:start + total_episodes]
            optimal_score = [i[0] for i in out_2]
            optimal_score_all = torch.tensor(optimal_score, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 8.89 #min-max 8.89 min-sum 124.61
    xy = torch.cat((depot.unsqueeze(1), node_xy), dim=1)
    depot_demand = torch.zeros(size=(total_episodes,1))
    demand = torch.cat((depot_demand, demand),dim=1)
    dataset_dict = {
        'xy': xy,
        'demand': demand,
        'capacity': capacity,
    }

    return dataset_dict, optimal_score


def use_saved_problems_vrpbp_npz(filename, total_episodes, device, start=0, solution_name=None,problem_name=None):
    data = np.load(filename)  # ['locs', 'demand_linehaul', 'vehicle_capacity', 'speed', 'num_depots']
    node_demands_line = torch.tensor(data['demand_linehaul'], device=device, dtype=torch.float32)[
                   start:start + total_episodes]  # 不包含起点,已经norm过
    node_demands_back = torch.tensor(data['demand_backhaul'], device=device, dtype=torch.float32)[
                   start:start + total_episodes] * (-1.)  # 不包含起点,已经norm过
    node_demands = node_demands_line + node_demands_back
    node_coords = torch.tensor(data['locs'], device=device, dtype=torch.float32)[start:start + total_episodes]


    if solution_name is not None:
        solution = np.load(solution_name)  # ['actions','costs','time']
        tour_lens = torch.tensor(solution['costs'], device=device, dtype=torch.float32)[start:start + total_episodes]
        tour_lens *= -1  # 记录的是负的
        optimal_score = tour_lens.mean()
    else:
        optimal_score = 1.0

    depot_demand = torch.zeros(size=(total_episodes, 1)).to(device)
    demand = torch.cat((depot_demand, node_demands), dim=-1)
    data_dict = {
        'xy': node_coords,
        "demand": demand
    }

    return data_dict, optimal_score

def get_random_problems_pdcvrp(batch_size, problem_size,capacity,depot_num=1):

    xy = torch.rand(size=(batch_size, depot_num+problem_size, 2))*5
    # shape: (batch, 1, 2)
    # xy = torch.distributions.uniform.Uniform(0.0, 5.0).sample((batch_size,depot_num+problem_size, 2)) 两种实现方式一样的

    demand = torch.randint(1, 10, size=(batch_size, problem_size//2))
    demand = torch.cat((demand,-demand),dim=-1)
    # shape: (batch, problem)
    depot_demand = torch.zeros(size=(batch_size,1))
    demand = torch.cat((depot_demand,demand),dim=-1)

    node_demand = demand / float(capacity)
    # shape: (batch, problem)

    data_dict = {
        "xy": xy,
        "demand": node_demand,
    }

    return data_dict

def use_saved_problems_pdcvrp_pt(filename, total_episodes, device, start=0, solution_name=None):
    data = torch.load(filename)

    if 'xy' in data:
        xy = data['xy'][start: start + total_episodes].to(device)
    else:
        dist = data['dist'][start: start + total_episodes].to(device)

    demand = data['demand'][start: start + total_episodes].to(device)
    if "route_limit" in data:
        route_limit = data['route_limit'][start: start + total_episodes].to(device)
    else:
        route_limit = torch.ones(total_episodes)*float('inf')

    if solution_name is not None:
        solution = torch.load(solution_name)
        optimal = solution.mean()
    else:
        optimal = 1
    if 'xy' in data:
        data_dict = {
            'xy':xy,
            'demand':demand,
            'route_limit': route_limit,
        }
    else:
        data_dict = {
            'dist': dist,
            'demand': demand,
            'route_limit': route_limit,
        }

    return data_dict,optimal



def use_saved_problems_amdcvrp_pt(filename, total_episodes, device, start=0, solution_name=None):
    data = torch.load(filename)

    dist = data['dist'][start: start + total_episodes].to(device)
    demand = data['demand'][start: start + total_episodes].to(device)

    data_dict = {
        'dist':dist,
        'demand':demand,
    }
    if solution_name is not None:
        solution = torch.load(solution_name)
        optimal = solution['cost'].mean()
    else:
        optimal = 1
    return data_dict,optimal



def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)





def get_random_problems_mdvrp_mixed(batch_size, problem_size,capacity,**kwargs):
    #follow mvmoe:https://github.com/RoyalSkye/Routing-MVMoE/blob/main/envs/OVRPBLTWEnv.py


    normalized = True   #demand有没有除以demand_scaler
    speed = 1.0
    depot_start, depot_end = 0., 3.
    backhaul_ratio = 0.2
    problem_name = kwargs.get("problem_name",None)
    depot_num = kwargs.get("depot_num",1)

    depot_xy = torch.rand(size=(batch_size, depot_num, 2))  # (batch, depot_num, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

    demand_scaler = capacity


    route_limit = torch.ones(batch_size) * 3.0

    #tw constraint
    service_time = torch.ones(batch_size, problem_size) * 0.2


    diff = node_xy.unsqueeze(2) - depot_xy.unsqueeze(1)  # [num_nodes, num_depots, 2]
    # 计算欧氏距离
    distances = diff.norm(p=2, dim=-1)  # [num_nodes, num_depots]
    # 取最大值 (沿 depot 维度)
    max_dist, _ = distances.max(dim=-1)  # [num_nodes]
    # 转换成 travel_time
    travel_time = max_dist / speed  # [num_nodes]



    a, b = depot_start + travel_time, depot_end - travel_time - service_time
    time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
    time_half_width = (service_time / 2 - depot_end / 3) * torch.rand(batch_size,problem_size) + depot_end / 3
    tw_start = torch.clamp(time_centers - time_half_width, min=depot_start, max=depot_end)
    tw_end = torch.clamp(time_centers + time_half_width, min=depot_start, max=depot_end)
    # shape: (batch, problem)

    # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
    instance_invalid, round_error_epsilon = False, 0.00001
    # total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / speed, tw_start) + service_time + (
    #             node_xy - depot_xy).norm(p=2, dim=-1) / speed > depot_end + round_error_epsilon
    # (batch, problem)

    diff_to_depot = node_xy.unsqueeze(2) - depot_xy.unsqueeze(1)  # [B, P, D, 2]
    # 每个 node 到各 depot 的距离
    dist_to_depot = diff_to_depot.norm(p=2, dim=-1)  # [B, P, D]
    # 取最大距离（最远 depot）
    max_dist, _ = dist_to_depot.max(dim=2)  # [B, P]
    # 计算 total_time
    total_time = (torch.max(max_dist / speed, tw_start)  # 等待时间 or travel_time
                         + service_time
                         + max_dist / speed ) > (depot_end + round_error_epsilon)

    instance_invalid = total_time.any()

    if instance_invalid:
        print(">> Invalid instances, Re-generating ...")
        return get_random_problems_mdvrp_mixed(batch_size, problem_size,capacity,depot_num=depot_num,problem_name=problem_name)
    elif normalized:
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)
        if problem_name is not None:
            if 'b' in problem_name:
                backhauls_index = torch.randperm(problem_size)[
                                  :int(problem_size * backhaul_ratio)]  # randomly select 20% customers as backhaul ones
                node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            if 'tw' not in problem_name:
                service_time = service_time*0
                tw_start = tw_start*0
                tw_end = torch.ones(batch_size, problem_size)* float('inf')
            if 'l' not in problem_name:
                route_limit = torch.ones(batch_size)*float("inf")

        xy = torch.cat((depot_xy,node_xy),dim=1)
        depot_demand = torch.zeros(size=(batch_size,depot_num))
        demand = torch.cat((depot_demand,node_demand),dim=-1)
        depot_tw_start = torch.zeros(size=(batch_size,depot_num))
        if "tw" in problem_name:
            depot_tw_end = torch.ones(size=(batch_size,depot_num))*3.0
        else:
            depot_tw_end = torch.ones(size=(batch_size, depot_num)) * float('inf')
        depot_service_time = torch.zeros(size=(batch_size,depot_num))
        tw_start = torch.cat((depot_tw_start,tw_start),dim=-1)
        tw_end = torch.cat((depot_tw_end,tw_end),dim=-1)
        service_time = torch.cat((depot_service_time,service_time),dim=-1)

        data_dict = {
            'xy': xy,
            'demand': demand,
            "tw_start": tw_start,
            "tw_end":tw_end,
            "service_time": service_time,
            "route_limit": route_limit
        }

        return data_dict


def get_random_problems_amdvrp_mixed(batch_size, problem_size,capacity,**kwargs):
    speed = 1.0
    depot_start, depot_end = 0., 1.
    backhaul_ratio = 0.2
    problem_name = kwargs.get("problem_name", None)
    depot_num = kwargs.get("depot_num", 3)

    problem_gen_params = {
        'int_min': 0,
        'int_max': 1000 * 1000,
        'scaler': 1000 * 1000
    }

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, problem_size + depot_num, problem_size + depot_num))
    # shape: (batch, node, node)
    problems[:, torch.arange(problem_size + depot_num), torch.arange(problem_size + depot_num)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    dist = problems.float() / scaler

    demand = torch.randint(1, 10, size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_demand = demand / float(capacity)


    # l constraints
    if 'l' in problem_name:
        route_limit = torch.ones(batch_size) * 0.5
    else:
        route_limit = torch.ones(batch_size) * float("inf")

    # b constraints
    if 'b' in problem_name:
        backhauls_index = torch.randperm(problem_size)[
                          :int(problem_size * backhaul_ratio)]  # randomly select 20% customers as backhaul ones
        node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]

    # tw constraints
    if 'tw' in problem_name:
        dist_matrix = dist
        travel_time_depot_to_node = dist_matrix[:, :depot_num, depot_num:] / speed
        travel_time_node_to_depot = dist_matrix[:, depot_num:, :depot_num] / speed

        travel_time_depot_to_node,_ = travel_time_depot_to_node.max(dim=1)
        travel_time_node_to_depot,_ = travel_time_node_to_depot.max(dim=-1)


        service_time = torch.ones(batch_size, problem_size) * 0.2
        a, b = depot_start + travel_time_depot_to_node, depot_end - travel_time_node_to_depot - service_time
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (service_time / 2 - depot_end / 3) * torch.rand(batch_size, problem_size) + depot_end / 3
        tw_start = torch.clamp(time_centers - time_half_width, min=depot_start, max=depot_end)
        tw_end = torch.clamp(time_centers + time_half_width, min=depot_start, max=depot_end)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + travel_time_depot_to_node,
                               tw_start) + service_time + travel_time_node_to_depot > depot_end + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return get_random_problems_avrpmix(batch_size, problem_size, capacity, problem_name=problem_name)
    else:
        service_time = torch.zeros(batch_size, problem_size)
        tw_start = torch.zeros(batch_size, problem_size)
        tw_end = torch.ones(batch_size, problem_size) * float("inf")

    depot_demand = torch.zeros(size=(batch_size, depot_num))
    demand = torch.cat((depot_demand, node_demand), dim=-1)
    depot_tw_start = torch.zeros(size=(batch_size, depot_num))
    if "tw" in problem_name:
        depot_tw_end = torch.ones(size=(batch_size, depot_num)) * 1.0
    else:
        depot_tw_end = torch.ones(size=(batch_size, depot_num)) * float('inf')
    depot_service_time = torch.zeros(size=(batch_size, depot_num))
    tw_start = torch.cat((depot_tw_start, tw_start), dim=-1)
    tw_end = torch.cat((depot_tw_end, tw_end), dim=-1)
    service_time = torch.cat((depot_service_time, service_time), dim=-1)

    data_dict = {
        'dist': dist,
        'demand': demand,
        'tw_start':tw_start,
        'tw_end':tw_end,
        'service_time': service_time,
        'route_limit': route_limit
    }

    return data_dict





def get_random_problems_aop(batch_size, node_cnt,depot_num=1):

    ################################
    # "tmat" type
    # Following MatNet: https://github.com/yd-kwon/MatNet/blob/main/ATSP/ATSProblemDef.py
    ################################
    problem_gen_params = {
        'int_min': 0,
        'int_max': 1000 * 1000,
        'scaler': 1000 * 1000
    }

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt+depot_num, node_cnt+depot_num))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt+depot_num), torch.arange(node_cnt+depot_num)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    prize_ = scaled_problems[:,0,:]
    prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.
    prize[:, 0] = 0.


    data = {
        'dist':scaled_problems,
        'prize': prize
    }

    return data
    # shape: (batch, node, node)



def get_random_problems_vrp_mixed_mvmoe_version(batch_size, problem_size,capacity,**kwargs):
    #follow mvmoe:https://github.com/RoyalSkye/Routing-MVMoE/blob/main/envs/OVRPBLTWEnv.py

    normalized = True   #demand有没有除以demand_scaler
    speed = 1.0
    depot_start, depot_end = 0., 3.
    backhaul_ratio = 0.2
    problem_name = kwargs.get("problem_name",None)


    depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

    demand_scaler = capacity


    route_limit = torch.ones(batch_size) * 3.0

    #tw constraint
    service_time = torch.ones(batch_size, problem_size) * 0.2
    travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / speed
    a, b = depot_start + travel_time, depot_end - travel_time - service_time
    time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
    time_half_width = (service_time / 2 - depot_end / 3) * torch.rand(batch_size,problem_size) + depot_end / 3
    tw_start = torch.clamp(time_centers - time_half_width, min=depot_start, max=depot_end)
    tw_end = torch.clamp(time_centers + time_half_width, min=depot_start, max=depot_end)
    # shape: (batch, problem)

    # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
    instance_invalid, round_error_epsilon = False, 0.00001
    total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / speed, tw_start) + service_time + (
                node_xy - depot_xy).norm(p=2, dim=-1) / speed > depot_end + round_error_epsilon
    # (batch, problem)
    instance_invalid = total_time.any()

    if instance_invalid:
        print(">> Invalid instances, Re-generating ...")
        return get_random_problems_vrp_mixed_mvmoe_version(batch_size, problem_size,capacity)
    elif normalized:
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)
        if problem_name is not None:
            if 'b' in problem_name:
                backhauls_index = torch.randperm(problem_size)[
                                  :int(problem_size * backhaul_ratio)]  # randomly select 20% customers as backhaul ones
                node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            if 'tw' not in problem_name:
                service_time = service_time*0
                tw_start = tw_start*0
                tw_end = torch.ones(batch_size, problem_size)*float("inf")
            if 'l' not in problem_name:
                route_limit = torch.ones(batch_size)*float("inf")
        return depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end
    else:
        node_demand = torch.Tensor(
            np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)
        if problem_name is not None:
            if 'b' in problem_name:
                backhauls_index = torch.randperm(problem_size)[
                                  :int(problem_size * backhaul_ratio)]  # randomly select 20% customers as backhaul ones
                node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            if 'tw' not in problem_name:
                service_time = service_time*0
                tw_start = tw_start*0
                tw_end = torch.ones(batch_size, problem_size)*3.0
            if 'l' not in problem_name:
                route_limit = torch.ones(batch_size)*float("inf")

        capacity = torch.Tensor(np.full(batch_size, demand_scaler))
        return depot_xy, node_xy, node_demand, capacity, route_limit, service_time, tw_start, tw_end



def use_saved_problems_pdvrp_pkl(filename, total_episodes,device, start=0, solution_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:start + total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32,device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1 #LKH10000


    problems = torch.cat((raw_data_depot, raw_data_nodes), dim=1)

    data_dict = {
        'xy':problems,
    }


    return data_dict,optimal_score

def use_saved_problems_apdp_pt(filename, total_episodes,device, start=0, solution_name=None):
    data_dict = torch.load(filename)

    dist = data_dict['dist'][start: start + total_episodes].to(device)

    data = {
        'dist':dist
    }

    if solution_name is not None:
        solution = torch.load(solution_name)
        optimal = solution.mean()

    return data,optimal


if __name__ == "__main__":

    seed_everything(1234)
    batch_size = 16
    problem_size_list = [1000,2000,3000,4000,5000]
    capacity = None
    problem_list = ['cvrptw','ovrptw','ovrp','vrpb']

    for problem in problem_list:
        for problem_size in problem_size_list:
            if problem_size > 1000:
                capacity = 300
            else:
                capacity = 200
            depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end = get_random_problems_vrp_mixed_mvmoe_version(batch_size,problem_size,capacity,problem_name=problem)
            xy = torch.cat((depot_xy,node_xy),dim = 1)
            depot_demand = torch.zeros(batch_size, 1)
            demand = torch.cat((depot_demand,node_demand),dim = 1)

            data_dict = {
                'xy':xy,
                'demand':demand,
                'route_limit': route_limit,
                "service_time":service_time,
                "tw_start":tw_start,
                "tw_end":tw_end,
            }
            path = f"/home/zhengyp/ych/AAA/data/{problem}/{problem}{problem_size}_C{capacity}_n{batch_size}_seed1234.pt"
            torch.save(data_dict, path)
            print(f"数据保存到{path}")









