import torch
import pickle
import numpy as np

def use_saved_problems_tsp_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        problems = torch.tensor(out_1, dtype=torch.float32,device=device)
        # shape: (batch, problem, 2)
    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:start + total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32,device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1.0

    data = {
        'xy': problems
    }
    return data,optimal_score

def use_saved_problems_tsp_txt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    nodes_coords = []
    solution = []
    for line in open(filename, "r").readlines()[start:start + total_episodes]:
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes_coords.append(
            [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
        )
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
        solution.append(tour_nodes)

    problems = torch.tensor(nodes_coords,device=device)  # shape: (batch, problem, 2)
    solution = torch.tensor(solution,device=device)  # shape: (batch, problem)
    gathering_index = solution.unsqueeze(2).expand(-1, -1, 2)
    # shape: (batch, problem, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
    # shape: (batch, problem)
    travel_distances = segment_lengths.sum(1)
    # shape: (batch,)
    optimal_score = travel_distances.mean().item()

    data_dict = {
        'xy': problems
    }
    return data_dict,optimal_score

def use_saved_problems_tsp_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename)
    data_dict = {
        'xy':data['xy'][start:start + total_episodes].to(device)
    }
    optimal_score = 1.0
    return data_dict,optimal_score

def use_saved_problems_cvrp_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]

        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
        raw_data_demand = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
        # shape: (batch, problem)
        capacity = float(out[0, 3])
        raw_data_demand = (raw_data_demand / capacity).to(device)
    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 15.504


    xy = torch.cat((raw_data_depot, raw_data_nodes), dim=1)
    depot_demand = torch.zeros(size=(total_episodes, 1))
    demand = torch.cat((depot_demand, raw_data_demand), dim=-1)
    dataset_dict = {
        'xy': xy,
        'demand': demand,
        'capacity': capacity,
    }

    return dataset_dict, optimal_score

def use_saved_problems_cvrp_txt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    raw_data_nodes = []
    raw_data_depot = []
    raw_data_demand = []
    raw_cost = []
    capacity = 0

    for line in open(filename, "r").readlines()[start:start + total_episodes]:
        line = line.split(",")

        depot_index = int(line.index('depot'))
        customer_index = int(line.index('customer'))
        capacity_index = int(line.index('capacity'))
        demand_index = int(line.index('demand'))
        cost_index = int(line.index('cost'))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                    range(customer_index + 1, capacity_index, 2)]
        raw_data_nodes.append(customer)
        raw_data_depot.append(depot)

        if capacity == 0:
            capacity = float(line[capacity_index + 1])

        demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        raw_data_demand.append(demand)
        raw_cost.append(float(line[cost_index + 1]))

    raw_data_depot = torch.tensor(raw_data_depot, device=device)
    # shape: (batch, 1, 2)
    raw_data_nodes = torch.tensor(raw_data_nodes, device=device)
    # shape: (batch, problem, 2)
    raw_data_demand = torch.tensor(raw_data_demand, device=device) / capacity
    # shape: (batch, problem)
    optimal_score = np.mean(raw_cost)

    xy = torch.cat((raw_data_depot, raw_data_nodes), dim=1)
    depot_demand = torch.zeros(size=(total_episodes, 1))
    demand = torch.cat((depot_demand, raw_data_demand), dim=-1)
    dataset_dict = {
        'xy': xy,
        'demand': demand,
        'capacity': capacity,
    }

    return dataset_dict, optimal_score

def use_saved_problems_op_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
        raw_data_prize = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
        # shape: (batch, problem)  #不包含depot的prize
        max_length = float(out[0, 3])
    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1

    xy = torch.cat((raw_data_depot, raw_data_nodes), dim=1)
    depot_prize = torch.zeros(size=(total_episodes,1))
    prize = torch.cat((depot_prize,raw_data_prize),dim=1)
    dataset_dict = {
        'xy':xy,
        'prize':prize,
        'capacity': max_length,  #4.0
    }

    return dataset_dict, optimal_score

def use_saved_problems_op_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename)[start:start + total_episodes]
    raw_data_depot = data[:,0,:2].to(device)
    raw_data_nodes = data[:,1:,:2].to(device)
    raw_data_prize = data[:,1:,2].to(device)

    optimal_score = 33.19

    xy = torch.cat((raw_data_depot.unsqueeze(1), raw_data_nodes), dim=1)
    depot_prize = torch.zeros(size=(total_episodes, 1))
    prize = torch.cat((depot_prize, raw_data_prize), dim=1)

    dataset_dict = {
        'xy': xy,
        'prize': prize,
        'tour_length': 4.0,  #4.0
    }

    return dataset_dict, optimal_score

def use_saved_problems_pctsp_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
        raw_data_penalty = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
        raw_data_prize = torch.tensor(out[:, 3].tolist(), dtype=torch.float32)

    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 18.29

    xy = torch.cat((raw_data_depot, raw_data_nodes), dim=1)
    depot_zero_pad = torch.zeros(size=(total_episodes, 1))
    penalty = torch.cat((depot_zero_pad, raw_data_penalty), dim=1)
    prize = torch.cat((depot_zero_pad, raw_data_prize), dim=1)
    dataset_dict = {
        'xy': xy,
        'penalty': penalty,
        'prize': prize,

    }

    return dataset_dict, optimal_score

def use_saved_problems_pctsp_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename)[start:start + total_episodes]
    raw_data_depot = data[:,0,:2].to(device)
    raw_data_nodes = data[:,1:,:2].to(device)
    raw_data_prize = data[:,1:,2].to(device)
    raw_data_penalty = data[:,1:,3].to(device)

    optimal_score = 5.98

    xy = torch.cat((raw_data_depot.unsqueeze(1), raw_data_nodes), dim=1)
    depot_prize = torch.zeros(size=(total_episodes, 1))
    prize = torch.cat((depot_prize, raw_data_prize), dim=-1)
    depot_penalty = torch.zeros(size=(total_episodes, 1))
    penalty = torch.cat((depot_penalty, raw_data_penalty), dim=-1)

    dataset_dict = {
        'xy': xy,
        'penalty': penalty,
        'prize': prize,
    }
    return dataset_dict, optimal_score

def use_saved_problems_spctsp_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
        raw_data_penalty = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
        raw_data_prize = torch.tensor(out[:, 3].tolist(), dtype=torch.float32)
        raw_data_sto_prize = torch.tensor(out[:, 4].tolist(), dtype=torch.float32)

    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1   #以MDAM为基准

    xy = torch.cat((raw_data_depot.unsqueeze(1), raw_data_nodes), dim=1)
    depot_prize = torch.zeros(size=(total_episodes, 1))
    sto_prize = torch.cat((depot_prize, raw_data_sto_prize), dim=-1)
    prize = torch.cat((depot_prize, raw_data_prize), dim=-1)
    depot_penalty = torch.zeros(size=(total_episodes, 1))
    penalty = torch.cat((depot_penalty, raw_data_penalty), dim=-1)


    dataset_dict = {
        'xy': xy,
        'penalty': penalty,
        'prize':sto_prize,
        'fake_prize': prize,
    }

    return dataset_dict, optimal_score

def use_saved_problems_spctsp_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename)[start:start + total_episodes]
    raw_data_depot = data[:,0,:2].to(device)
    raw_data_nodes = data[:,1:,:2].to(device)
    raw_data_sto_prize = data[:,1:,2].to(device)
    raw_data_prize = data[:,1:,3].to(device)
    raw_data_penalty = data[:,1:,4].to(device)

    optimal_score = 6.16 # AM

    xy = torch.cat((raw_data_depot.unsqueeze(1), raw_data_nodes), dim=1)
    depot_prize = torch.zeros(size=(total_episodes, 1))
    sto_prize = torch.cat((depot_prize, raw_data_sto_prize), dim=-1)
    prize = torch.cat((depot_prize, raw_data_prize), dim=-1)
    depot_penalty = torch.zeros(size=(total_episodes, 1))
    penalty = torch.cat((depot_penalty, raw_data_penalty), dim=-1)

    dataset_dict = {
        'xy': xy,
        'penalty': penalty,
        'prize': sto_prize,
        'fake_prize': prize,
    }

    return dataset_dict, optimal_score

def use_saved_problems_atsp_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename, map_location=device,weights_only=False)
    problems = torch.from_numpy(data['distance_matrix'])[start:start + total_episodes].to(device)
    if total_episodes == 1000:
        optimal = 1.5711
    else:
        optimal = data['optimal'] # 1.5777 for 10K条数据

    data = {
        'dist':problems
    }

    return data, optimal

def use_saved_problems_multi_depot_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename, map_location=device)
    if solution_name is not None:
        solution = torch.load(solution_name, map_location=device)
        optimal_score = solution['cost'].mean()
    else:
        md_pyvrp20s_optimal={
            'mdcvrpbp': 11.5892,
            'mdocvrpbp':8.7014,
            'mdcvrpbpl':11.6182,
            'mdocvrpbptw':11.4134,
            'mdcvrpbpltw':17.8289,
            'mdcvrpbptw':17.9382,
            'mdocvrpbpl':8.6903,
            'mdocvrpbpltw':11.4001,
        }
        optimal_score = md_pyvrp20s_optimal[problem_name]

    demand = data['demand'][start:start + total_episodes].to(device)
    route_limit = data["route_limit"][start:start + total_episodes].to(device)
    tw_start = data['tw_start'][start:start + total_episodes].to(device)
    tw_end = data['tw_end'][start:start + total_episodes].to(device)
    service_time = data['service_time'][start:start + total_episodes].to(device)

    if "xy" in data:
        xy = data['xy'][start:start + total_episodes].to(device)
        dataset_dict = {
            'xy': xy,
            'demand': demand,
            'route_limit': route_limit,
            'tw_start': tw_start,
            'tw_end': tw_end,
            'service_time': service_time,
        }

    if "dist" in data:
        dist = data['dist'][start:start + total_episodes].to(device)
        dataset_dict = {
            'dist': dist,
            'demand': demand,
            'route_limit': route_limit,
            'tw_start': tw_start,
            'tw_end': tw_end,
            'service_time': service_time,
        }

    return dataset_dict, optimal_score

def use_saved_problems_pdtsp_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
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
        optimal_score = 9.428 #LKH10000


    problems = torch.cat((raw_data_depot, raw_data_nodes), dim=1)

    data_dict = {
        'xy':problems,
    }

    return data_dict,optimal_score

def use_saved_problems_vrpmix_pkl(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
        raw_data_demand = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
        # shape: (batch, problem)
        capacity = float(out[0, 3])

        raw_data_demand = (raw_data_demand / capacity).to(device)
        if 'l' in problem_name:
            route_limit = float(out[0, 4])
        if 'tw' in problem_name:
            raw_data_node_serviceTime = torch.tensor(out[:, -3].tolist(), dtype=torch.float32)
            raw_data_node_earlyTW = torch.tensor(out[:, -2].tolist(), dtype=torch.float32)
            raw_data_node_lateTW = torch.tensor(out[:, -1].tolist(), dtype=torch.float32)

    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1.0

    xy = torch.cat((raw_data_depot,raw_data_nodes),dim=1)
    depot_demand = torch.zeros(size=(total_episodes, 1))
    demand = torch.cat((depot_demand,raw_data_demand),dim=-1)


    dataset_dict = {
        'xy': xy,
        'demand': demand,
        'capacity': capacity,
    }
    if 'tw' in problem_name:
        depot_service_time = torch.zeros(size=(total_episodes, 1))
        depot_tw_start = torch.zeros(size=(total_episodes, 1))
        depot_tw_end = torch.ones(size=(total_episodes, 1)) * 3.0
        service_time = torch.cat((depot_service_time, raw_data_node_serviceTime), dim=1)
        # shape: (batch, problem+1)
        tw_start = torch.cat((depot_tw_start, raw_data_node_earlyTW), dim=1)
        # shape: (batch, problem+1)
        tw_end = torch.cat((depot_tw_end, raw_data_node_lateTW), dim=1)
        dataset_dict["service_time"] = service_time
        dataset_dict["tw_start"] = tw_start
        dataset_dict["tw_end"] = tw_end
    if 'l' in problem_name:
        dataset_dict["route_limit"] = route_limit * torch.ones(total_episodes)

    return dataset_dict, optimal_score

def use_saved_problems_vrpmix_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename,map_location=device)

    xy = data['xy'][start:start + total_episodes].to(device)
    demand = data['demand'][start:start + total_episodes].to(device)
    route_limit = data["route_limit"][start:start + total_episodes].to(device)
    tw_start = data['tw_start'][start:start + total_episodes].to(device)
    tw_end = data['tw_end'][start:start + total_episodes].to(device)
    service_time = data['service_time'][start:start + total_episodes].to(device)


    depot_service_time = torch.zeros(size=(total_episodes, 1))
    depot_tw_start = torch.zeros(size=(total_episodes, 1))

    if 'tw' in problem_name:
        depot_tw_end = torch.ones(size=(total_episodes, 1)) * 3.0
    else:
        depot_tw_end = torch.ones(size=(total_episodes, 1)) * float('inf')
    service_time = torch.cat((depot_service_time, service_time), dim=1)
    # shape: (batch, problem+1)
    tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
    # shape: (batch, problem+1)
    tw_end = torch.cat((depot_tw_end, tw_end), dim=1)

    if solution_name is not None:
        solution = torch.load(solution_name)
        optimal_score = solution['cost'].mean()
    else:
        optimal_score = 1.0

    data_dict = {
        'xy': xy,
        'demand': demand,
        'route_limit': route_limit,
        'tw_start': tw_start,
        'tw_end': tw_end,
        'service_time': service_time,
    }

    return data_dict, optimal_score

def use_saved_problems_avrpmix_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data = torch.load(filename,map_location=device)
    if problem_name == 'acvrp':
        dist = data['dist_matrix'][start:start + total_episodes].to(device)
        demand = data['demand'][start:start + total_episodes].to(device)
        if solution_name == None:
            score = data['result'][start:start + total_episodes].to(device)
            optimal_score = score.mean()
        else:
            solution = torch.load(solution_name, map_location=device)
            optimal_score = solution['cost'].mean()
        dataset_dict = {
            'dist': dist,
            'demand': demand,
        }
        return dataset_dict, optimal_score

    dist = data['dist_matrix'][start:start + total_episodes].to(device)
    node_demand = data['node_demand'][start:start + total_episodes].to(device)
    route_limit = data["route_limit"][start:start + total_episodes].to(device)
    tw_start = data['tw_start'][start:start + total_episodes].to(device)
    tw_end = data['tw_end'][start:start + total_episodes].to(device)
    service_time = data['service_time'][start:start + total_episodes].to(device)

    depot_demand = torch.zeros(size=(total_episodes,1))
    demand = torch.cat((depot_demand,node_demand),dim=-1)

    depot_tw_start = torch.zeros(size=(total_episodes,1))
    if 'tw' in problem_name:
        depot_tw_end = torch.ones(size=(total_episodes, 1))
    else:
        depot_tw_end = torch.ones(size=(total_episodes, 1)) * float('inf')
    depot_service_time = torch.zeros(size=(total_episodes,1))
    tw_start = torch.cat((depot_tw_start,tw_start),dim=-1)
    tw_end = torch.cat((depot_tw_end,tw_end),dim=-1)
    service_time = torch.cat((depot_service_time,service_time),dim=-1)

    solution = torch.load(solution_name,map_location=device)
    optimal_score = solution['cost'].mean()

    dataset_dict = {
        'dist':dist,
        'demand':demand,
        'route_limit':route_limit,
        'tw_start':tw_start,
        'tw_end':tw_end,
        'service_time':service_time,
    }

    return dataset_dict, optimal_score

def use_saved_problems_amdcvrp_pt(filename, total_episodes, device, start=0, solution_name=None,problem_name=None):
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

def use_saved_problems_pdcvrp_pt(filename, total_episodes, device, start=0, solution_name=None,problem_name=None):
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

def use_saved_problems_apdp_pt(filename, total_episodes,device, start=0, solution_name=None,problem_name=None):
    data_dict = torch.load(filename)

    dist = data_dict['dist'][start: start + total_episodes].to(device)

    data = {
        'dist':dist
    }

    if solution_name is not None:
        solution = torch.load(solution_name)
        optimal = solution.mean()

    return data,optimal