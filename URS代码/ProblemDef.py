from multi_hot_set import get_problem_list
from data_utils.data_generation import *
from data_utils.use_saved_data import *

def get_random_problems(batch_size, problem_size, capacity, problem_name,**kwargs):
    xy = None
    demand = None
    prize = None  #op的prize和pctsp的prize都用这个变量
    fake_prize = None
    penalty = None
    dist = None
    service_time = None
    tw_start = None
    tw_end = None
    route_limit = None

    depot_num = kwargs.get("depot_num")
    depot_start = kwargs.get("depot_start",0)
    depot_end = kwargs.get("depot_end",float('inf'))

    if problem_name == "tsp":
        xy = get_random_problems_tsp(batch_size,problem_size) #batch,problem_size,2 (x,y)
    elif problem_name in ["cvrp","sdvrp","mdcvrp"]:
        depot_xy, node_xy, node_demand = get_random_problems_cvrp(batch_size, problem_size,capacity,depot_num=depot_num)
        xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(batch_size, depot_num))
        demand = torch.cat((depot_demand, node_demand), dim=1)
    elif problem_name == "atsp":
        problem_gen_params = kwargs.get('problem_gen_params')
        dist = get_random_problems_atsp(batch_size,problem_size,problem_gen_params)
    elif problem_name == "op":
        problem = get_random_problems_op(batch_size,problem_size) #batch,problem_size+1,3
        xy = problem[:,:,:2]
        prize = problem[:,:,2]
    elif problem_name == "pctsp":
        problem = get_random_problems_pctsp(batch_size, problem_size)  # batch,problem_size+1,4
        xy = problem[:, :, :2]
        prize = problem[:, :, 2]
        penalty = problem[:,:,3]
    elif problem_name == "spctsp":
        #xy, real_prize, prize, penalty
        problem = get_random_problems_spctsp(batch_size, problem_size)  # batch,problem_size+1,5
        xy = problem[:, :, :2]
        prize = problem[:, :, 2]
        fake_prize = problem[:, :, 3]
        penalty = problem[:, :, 4]
    elif problem_name == "pdtsp":
        problem = get_random_problems_pdtsp(batch_size, problem_size)
        xy = problem[:, :, :2]
    elif problem_name == "acvrp":
        problem = get_random_problems_acvrp(batch_size,problem_size,capacity)
        dist = problem['dist']
        demand = problem['demand']
    elif problem_name in ['cvrpb','cvrpl','cvrptw','ocvrp','ocvrptw', 'ocvrpb','cvrpbl','cvrpltw','ocvrpbtw','cvrpbltw','ocvrpl','cvrpbtw','ocvrpbl','ocvrpltw','ocvrpbltw']:
        depot_xy, node_xy, node_demand, route_limit, service_time, tw_start, tw_end = get_random_problems_vrp_mixed_mvmoe_version(
            batch_size, problem_size, capacity, problem_name=problem_name)
        xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(batch_size, depot_num))
        demand = torch.cat((depot_demand, node_demand), dim=1)
        depot_service_time = torch.zeros(size=(batch_size, 1))
        depot_tw_start = torch.ones(size=(batch_size, 1)) * depot_start
        depot_tw_end = torch.ones(size=(batch_size, 1)) * depot_end
        service_time = torch.cat((depot_service_time, service_time), dim=1)
        # shape: (batch, problem+1)
        tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
        # shape: (batch, problem+1)
        tw_end = torch.cat((depot_tw_end, tw_end), dim=1)
        
    # ! add
    elif problem_name in ['acvrptw', 'acvrpb', 'acvrpl', 'acvrpbtw', 'acvrpltw', 'acvrpbl', 'acvrpbltw']:
        # asymmetric VRP variants with time windows, backhaul, or limit constraints
        problem = get_random_problems_avrpmix(batch_size, problem_size, capacity, problem_name=problem_name)
        dist = problem['dist']
        demand = problem['demand']
        route_limit = problem['route_limit']
        service_time = problem['service_time']
        tw_start = problem['tw_start']
        tw_end = problem['tw_end']
        # Add depot values
        depot_service_time = torch.zeros(size=(batch_size, 1))
        depot_tw_start = torch.ones(size=(batch_size, 1)) * depot_start
        depot_tw_end = torch.ones(size=(batch_size, 1)) * depot_end
        service_time = torch.cat((depot_service_time, service_time), dim=1)
        tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
        tw_end = torch.cat((depot_tw_end, tw_end), dim=1)


    #没有对应属性的，用全0或者inf填充，根据问题性质
    if xy is None:
        xy = torch.zeros(size=(batch_size, problem_size+depot_num, 2))
    if demand is None:
        demand = torch.zeros(size=(batch_size, problem_size+depot_num))
    if dist is None:
        dist = torch.cdist(xy, xy, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    if prize is None:
        prize = torch.zeros(size=(batch_size, problem_size+depot_num))
    if penalty is None:
        penalty = torch.zeros(size=(batch_size, problem_size + depot_num))
    if fake_prize is None:
        fake_prize = torch.zeros(size=(batch_size, problem_size + depot_num))
    if service_time is None:
        service_time = torch.zeros(size=(batch_size, problem_size+depot_num))
    if tw_start is None:
        tw_start = torch.zeros(size=(batch_size, problem_size+depot_num))
    if tw_end is None:
        tw_end = torch.full((batch_size, problem_size+depot_num), float('inf'))
    if route_limit is None:
        route_limit = torch.full([batch_size], float('inf'))

    data = {
        'xy': xy,
        'demand': demand,
        'dist': dist,
        'prize': prize,
        'penalty': penalty,
        'fake_prize': fake_prize,
        'service_time': service_time,
        'tw_start': tw_start,
        'tw_end': tw_end,
        'route_limit': route_limit,
    }

    return data

def augment_xy_data_by_8_fold(problems,aug_factor=None):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    if aug_factor == 2:
        aug_problems = torch.cat((dat1, dat4), dim=0)

    else:
        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def get_saved_data(filename, problem_name,total_episodes,device, start=0, solution_name=None):

    data_type = filename.split('.')[-1]

    if problem_name == 'tsp':
        data_loader = {
            'pkl': use_saved_problems_tsp_pkl,
            'txt': use_saved_problems_tsp_txt,
            'pt': use_saved_problems_tsp_pt,
        }
    elif problem_name == 'cvrp':
        data_loader = {
            'pkl': use_saved_problems_cvrp_pkl,
            'txt': use_saved_problems_cvrp_txt,
        }
    elif problem_name == "op":
        data_loader = {
            'pkl': use_saved_problems_op_pkl,
            'pt' : use_saved_problems_op_pt,
        }
    elif problem_name == "pctsp":
        data_loader = {
            'pkl': use_saved_problems_pctsp_pkl,
            'pt': use_saved_problems_pctsp_pt,
        }
    elif problem_name == "spctsp":
        data_loader ={
            'pkl':use_saved_problems_spctsp_pkl,
            'pt': use_saved_problems_spctsp_pt
        }
    elif problem_name == "pdtsp":
        data_loader ={
            'pkl':use_saved_problems_pdtsp_pkl,
        }
    elif problem_name == "atsp":
        data_loader = {
            'pt': use_saved_problems_atsp_pt,
        }
    elif problem_name in get_problem_list("vrpmix_list"):
        data_loader = {
            'pkl': use_saved_problems_vrpmix_pkl,
            'pt': use_saved_problems_vrpmix_pt,
        }
    elif problem_name in get_problem_list("avrpmix_list"):
        data_loader = {
            'pt': use_saved_problems_avrpmix_pt,
        }
    elif problem_name in get_problem_list("pdcvrp_list"):
        data_loader = {
            'pt': use_saved_problems_pdcvrp_pt,
        }
    elif problem_name in ['amdcvrp']:
        data_loader = {
            'pt': use_saved_problems_amdcvrp_pt,
        }
    elif problem_name in get_problem_list("multi_depot_list"):
        data_loader = {
            'pt': use_saved_problems_multi_depot_pt,
        }
    elif problem_name in ['apdtsp']:
        data_loader = {
            'pt': use_saved_problems_apdp_pt,
        }
    else:
        raise NotImplementedError(f"Problem name {problem_name} is not supported.")

    if data_type not in data_loader.keys():
        assert False, f"Unsupported file type: {data_type}. Supported types are: {list(data_loader.keys())}"


    data,optimal_score = data_loader[data_type](filename, total_episodes, device, start, solution_name,problem_name=problem_name)


    return data,optimal_score

