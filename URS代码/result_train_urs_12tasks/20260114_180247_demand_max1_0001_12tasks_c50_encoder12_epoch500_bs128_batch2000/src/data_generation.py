import torch
import numpy as np

#for pctsp/spctsp
K_n = {
    20: 2,
    50: 3,
    100: 4,
    500: 9,
    1000: 12,
    5000: 20,
    10000: 38
}

def get_random_problems_tsp(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems

def get_random_problems_pdtsp(batch_size, problem_size):
    if problem_size % 2 !=0:
        problem_size +=1  #Number of locations must be even
    problems = torch.rand(size=(batch_size, problem_size+1, 2))
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
    c = torch.cat((torch.zeros((batch_size, 1)), beta), dim=1)  # (n+1,)
    # problems.shape: (batch, problem, 2)
    problems = torch.cat((problems, prize.unsqueeze(-1), c.unsqueeze(-1)), dim=2)
    return problems   #batch,problem+1,4

def get_random_problems_spctsp(batch_size, problem_size, coords=None, fix_problem_size=True):
    if coords is not None:
        problems = coords
    else:
        problems = torch.rand(size=(batch_size, problem_size+1, 2))

    prizes = torch.rand(size=(batch_size, problem_size)) * 4 / problem_size
    prize = torch.cat((torch.zeros((batch_size, 1)), prizes), dim=1)
    stochastic_prize = torch.rand(batch_size, problem_size+1) * prize * 2
    if fix_problem_size == True:
        K = K_n[problem_size]
    else:
        K = np.random.randint(4, 9 + 1)  # one scalar
    beta = torch.rand(size=(batch_size, problem_size)) * 3 * K / problem_size
    c = torch.cat((torch.zeros((batch_size, 1)), beta), dim=1)  # (n+1,)
    # problems.shape: (batch, problem, 2)
    problems = torch.cat((problems, stochastic_prize.unsqueeze(-1), prize.unsqueeze(-1), c.unsqueeze(-1)), dim=2)
    return problems  # xy, real_prize, prize, penalty

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

def get_random_problems_mdvrp_mixed(batch_size, problem_size,capacity,**kwargs):
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
            return get_random_problems_amdvrp_mixed(batch_size, problem_size, capacity, problem_name=problem_name)
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

def get_random_problems_pdcvrp(batch_size, problem_size,capacity,depot_num=1):

    xy = torch.rand(size=(batch_size, depot_num+problem_size, 2))
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

def get_random_problems_apdcvrp(batch_size, node_cnt,capacity,depot_num=1):
    problem_gen_params = {
        'int_min': 0,
        'int_max': 1000 * 1000,
        'scaler': 1000 * 1000
    }
    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt + depot_num, node_cnt + depot_num))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt + depot_num), torch.arange(node_cnt + depot_num)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler


    demand = torch.randint(1, 10, size=(batch_size, node_cnt//2))
    demand = torch.cat((demand,-demand),dim=-1)
    # shape: (batch, problem)
    depot_demand = torch.zeros(size=(batch_size,1))
    demand = torch.cat((depot_demand,demand),dim=-1)

    node_demand = demand / float(capacity)
    # shape: (batch, problem)

    data_dict = {
        "dist": scaled_problems,
        "demand": node_demand,
    }

    return data_dict




