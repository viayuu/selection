import torch
import argparse
import os
import numpy as np
import math
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from EasyNCO.exact_solvers.params_setting import get_options
from EasyNCO.data.data_utils import get_gaussian_mixture,generate_by_distribution
from EasyNCO.neural_solvers.methods.omni.train_utils import _fast_val
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

def generate_task_set(data_type):
    """
    For Omni generates Training Task Set
    Current setting:
        size: (n,) \in [20, 150]
        distribution: (m, c) \in {(0, 0) + [1-9] * [1, 10, 20, 30, 40, 50]}
        size_distribution: (n, m, c) \in [50, 200, 5] * {(0, 0) + (1, 1) + [3, 5, 7] * [10, 30, 50]}
    """
    if data_type == "distribution":  # focus on TSP100 with gaussian mixture distributions
        task_set = [(0, 0)] + [(m, c) for m in range(1, 10) for c in [1, 10, 20, 30, 40, 50]]
    elif data_type == "size":  # focus on uniform distribution with different sizes
        task_set = [(n,) for n in range(20, 151)]
    elif data_type == "size_distribution":
        dist_set = [(0, 0), (1, 1)] + [(m, c) for m in [3, 5, 7] for c in [10, 30, 50]]
        task_set = [(n, m, c) for n in range(50, 201, 5) for (m, c) in dist_set]
    else:
        raise NotImplementedError

    return task_set

def _get_data(data_type, batch_size, task_params,problem_size,env_name):
    if data_type == 'size':
        assert len(task_params) == 1
        data = get_random_problems(batch_size, task_params[0], num_modes=0, cdist=0, distribution='uniform', problem=env_name)
    elif data_type == 'distribution':
        assert len(task_params) == 2
        data = get_random_problems(batch_size, problem_size, num_modes=task_params[0], cdist=task_params[1], distribution='gaussian_mixture', problem=env_name)
    elif data_type == "size_distribution":
        assert len(task_params) == 3
        data = get_random_problems(batch_size, task_params[0], num_modes=task_params[1], cdist=task_params[2], distribution='gaussian_mixture', problem=env_name)
    else:
        raise NotImplementedError

    if len(data) == 4:
        depot_xy, node_xy, node_demand, capacity = data
        norm_node_demand = node_demand / capacity.view(-1, 1)
        node_with_depot = torch.cat([depot_xy, node_xy],dim=1)
        norm_demand_with_depot = torch.cat([torch.zeros(batch_size, 1), norm_node_demand],dim=1).unsqueeze(2)
        demand_with_depot = torch.cat([torch.zeros(batch_size, 1), node_demand], dim=1).unsqueeze(2)
        norm_data = torch.cat((node_with_depot, norm_demand_with_depot), dim=-1)
        capacity = capacity.view(-1, 1, 1)
        capacity = capacity.repeat(1,node_with_depot.shape[1],1)
        data = torch.cat((node_with_depot,demand_with_depot,capacity),dim=-1)
    else:
        norm_data = data

    return data,norm_data

def generate_gaussian_mixture_tsp(dataset_size, graph_size, num_modes=0, cdist=0):
    #修改返回的数据类型
    if num_modes == 0:  # (0, 0) - uniform
        return np.random.uniform(0, 1, [dataset_size, graph_size, 2])
    elif num_modes == 1 and cdist == 1:  # (1, 1) - gaussian
        return generate_tsp_dist(dataset_size, graph_size, "gaussian")
    else:
        res = []
        for i in range(dataset_size):
            res.append(get_gaussian_mixture(graph_size=graph_size, num_modes=num_modes, cdist=cdist))
        return torch.stack(res, dim=0)


def generate_tsp_dist(batch_size, problem_size, distribution):
    result = []
    for i in range(batch_size):
        data = generate_by_distribution(problem_size, distribution)
        result.append(data)

    final_result = torch.stack(result, dim=0)

    assert final_result.shape[0] == batch_size
    assert final_result.shape[1] == problem_size
    assert final_result.shape[2] == 2

    return final_result


def set_lkh_opt(problem_type):
    opts = get_options()
    opts.solver = "lkh"
    opts.MAX_TRIALS = 100
    opts.ptype = problem_type
    opts.int_coord_scale = 1000000
    opts.SEED = 1234
    opts.cpus, opts.n, opts.progress_bar_mininterval = None, None, 0.1
    opts.delete = False
    opts.solver_mode = "multi"
    return opts

def _update_task_weight(self,tasks, weights,meta_params,env):
    """
    Update the weights of tasks.
    For LKH3, set MAX_TRIALS = 100 to reduce time.
    """
    from EasyNCO.exact_solvers.exact_solver_main import exact_solver_main
    solver = meta_params['solver']
    global run_func
    gap = torch.zeros(weights.size(0))
    batch_size = 200 if solver == "lkh3_offline" else 50
    idx = torch.randperm(batch_size)[:50]
    for i in range(gap.size(0)):
        selected = tasks[i]
        data,norm_data = _get_data(data_type=meta_params['data_type'],batch_size=batch_size, task_params=selected,problem_size=env.problem_size,env_name=env.env_name)
        logger.info(f"lkh3_running,i={i},total={gap.size(0)}")
        # only use lkh3 at the first iteration of updating task weights
        if solver == "lkh3_offline":
            if selected not in self.val_data.keys():
                self.val_data[selected] = data
                opts = set_lkh_opt(env.env_name)
                dataset = [(instance.cpu().numpy(),) for instance in data]
                data_list = []
                if env.env_name == "tsp":
                    for batch_idx, batch in enumerate(dataset):
                        for instance in batch:
                            node_coords = [(xy[0].item(), xy[1].item()) for xy in instance]
                            data_list.append(node_coords)
                elif env.env_name == "cvrp":
                    for batch_idx, batch in enumerate(dataset):
                        for instance in batch:
                            depot = instance[0, 0:2]
                            locs = instance[1:, 0:2]
                            demand = instance[1:, 2]
                            capacity = int(instance[0, 3])

                            demand = demand.tolist()
                            locs = locs.tolist()
                            depot = depot.tolist()
                            data_list.append([depot, locs, demand, capacity])
                results = exact_solver_main(opts, data_list=data_list)

                self.val_opt[selected] = [j for _,j in results]
            data = self.val_data[selected][idx]

        model_score = _fast_val(self, env, self.policy, data=data, mode="eval", return_all=True)
        model_score = (-model_score).tolist()[0]

        if meta_params["solver"] == "lkh3_online":
            # get results from LKH3
            opts = set_lkh_opt(env.env_name)
            dataset = [(instance.cpu().numpy(),) for instance in data]
            if env.env_name == "tsp":
                for batch_idx, batch in enumerate(dataset):
                    for instance in batch:
                        node_coords = [(xy[0].item(), xy[1].item()) for xy in instance]
                        data_list.append(node_coords)
            elif env.env_name == "cvrp":
                for batch_idx, batch in enumerate(dataset):
                    for instance in batch:
                        depot = instance[0, 0:2]
                        locs = instance[1:, 0:2]
                        demand = instance[1:, 2]
                        capacity = int(instance[0, 3])

                        demand = demand.tolist()
                        locs = locs.tolist()
                        depot = depot.tolist()
                        data_list.append([depot, locs, demand, capacity])
            results = exact_solver_main(opts, data_list=data_list)
            gap_list = [(model_score[j] - results[j][0]) / results[j][0] * 100 for j in range(len(results))]
            gap[i] = sum(gap_list) / len(gap_list)
        elif solver == "lkh3_offline":
            lkh_score = [self.val_opt[selected][j] for j in idx.tolist()]
            gap_list = [(model_score[j] - lkh_score[j]) / lkh_score[j] * 100 for j in range(len(lkh_score))]
            gap[i] = sum(gap_list) / len(gap_list)
        else:
            raise NotImplementedError

    temp = 1.0
    gap_temp = torch.Tensor([i/temp for i in gap.tolist()])
    weights = torch.softmax(gap_temp, dim=0)
    return weights


def get_random_problems(batch_size, problem_size, num_modes=0, cdist=0, distribution='uniform', path=None, problem="tsp"):
    """
    Generate TSP data within range of [0, 1]
    """
    # assert problem in ["tsp", "cvrp"], "Problems not support."

    # uniform distribution problems.shape: (batch, problem, 2)
    if distribution == "uniform":
        problems = np.random.uniform(0, 1, [batch_size, problem_size, 2])
        # problems = torch.rand(size=(batch_size, problem_size, 2))
    elif distribution == "gaussian_mixture":
        problems = generate_gaussian_mixture_tsp(batch_size, problem_size, num_modes=num_modes, cdist=cdist)  #ndarry (batch,problem,2)
    elif distribution in ["uniform_rectangle", "gaussian", "cluster", "diagonal", "tsplib", "cvrplib"]:
        problems = generate_tsp_dist(batch_size, problem_size, distribution)
    else:
        raise NotImplementedError

    if problem == "cvrp":
        depot_xy = np.random.uniform(size=(batch_size, 1, 2))  # shape: (batch, 1, 2)
        node_demand = np.random.randint(1, 10, size=(batch_size, problem_size))  # (unnormalized) shape: (batch, problem)
        demand_scaler = math.ceil(30 + problem_size/5) if problem_size >= 20 else 20
        capacity = np.full(batch_size, demand_scaler)

    if not torch.is_tensor(problems):
        problems = torch.Tensor(problems)

    if problem == "tsp":
        return problems
    else:
        return torch.Tensor(depot_xy), problems, torch.Tensor(node_demand), torch.Tensor(capacity)



