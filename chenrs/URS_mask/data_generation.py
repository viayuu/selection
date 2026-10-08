import torch
import numpy as np

def get_random_problems_tsp(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems

def get_random_problems_cvrp(batch_size, problem_size,capacity,depot_num=1):
    depot_xy = torch.rand(size=(batch_size, depot_num, 2))
    # shape: (batch, 1, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)
    demand = torch.randint(1, 10, size=(batch_size, problem_size))
    # shape: (batch, problem)
    node_demand = demand / float(capacity)
    # shape: (batch, problem)
    return depot_xy, node_xy, node_demand

def get_random_problems_atsp(batch_size, node_cnt, problem_gen_params):
    ################################
    # "tmat" type
    # Following MatNet: https://github.com/yd-kwon/MatNet/blob/main/ATSP/ATSProblemDef.py
    ################################
    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()
        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)
        if (problems == old_problems).all():
            break
    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)

def get_random_problems_op(batch_size, problem_size,coords=None, test=False):
    if coords is not None:
        problems = coords
    else:
        problems = torch.rand(size=(batch_size, problem_size+1, 2))
    prize_ = (problems[:, 0:1] - problems).norm(p=2, dim=-1)
    prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.
    prize[:, 0] = 0.
    problems = torch.cat((problems, prize.unsqueeze(-1)), dim=2)
    return problems  #batch,problem+1,3

#for pctsp
K_n = {
    20: 2,
    50: 3,
    100: 4,
    500: 9,
    1000: 12,
    5000: 20,
    10000: 38
}

def get_random_problems_pctsp(batch_size, problem_size, coords=None, fix_problem_size=True):
    if coords is not None:
        problems = coords
    else:
        problems = torch.rand(size=(batch_size, problem_size+1, 2))
    prizes = torch.rand(size=(batch_size, problem_size)) * 4 / problem_size
    prize = torch.cat((torch.zeros((batch_size, 1)), prizes), dim=1)
    if fix_problem_size == True:
        K = K_n[problem_size]
    else:
        K = np.random.randint(4, 9 + 1)  #one scalar  100到500
    beta = torch.rand(size=(batch_size, problem_size)) * 3 * K / problem_size
    c = torch.cat((torch.zeros((batch_size, 1)), beta), dim=1)  # (n+1,) penalty
    # problems.shape: (batch, problem, 2)
    problems = torch.cat((problems, prize.unsqueeze(-1), c.unsqueeze(-1)), dim=2)
    return problems   #batch,problem+1,4


def get_random_problems_pdtsp(batch_size, problem_size):
    #pdtsp 有一个depot
    #  问题的前一半是pickup节点 后一半是deliver节点，(1,n/2+1),(2,n/2+2) ...
    #  比如pdtsp100 0号点为depot, 1为pickup节点，它对应的deliver节点为 1+100/2=51

    if problem_size % 2 !=0:
        problem_size +=1  #Number of locations must be even
    problems = torch.rand(size=(batch_size, problem_size+1, 2))
    # problems.shape: (batch, problem, 2)
    return problems


def get_random_problems_vrp_mixed_mvmoe_version(batch_size, problem_size,capacity,**kwargs):

    normalized = True
    speed = 1.0
    depot_start, depot_end = 0., 3.
    backhaul_ratio = 0.2
    problem_name = kwargs.get("problem_name",None)


    depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
    node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

    # if problem_size == 20:
    #     demand_scaler = 30
    # elif problem_size == 50:
    #     demand_scaler = 40
    # elif problem_size == 100:
    #     demand_scaler = 50
    # elif problem_size == 200:
    #     demand_scaler = 70
    # else:
    #     raise NotImplementedError
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

def get_random_problems_acvrp(batch_size, node_cnt, capacity,depot_num=1):

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

    demand = torch.randint(1, 10, size=(batch_size, node_cnt+depot_num))
    # shape: (batch, problem)

    node_demand = demand / float(capacity)

    node_demand[:,:depot_num] = 0

    data = {
        'dist':scaled_problems,
        'demand':node_demand,
    }

    return data