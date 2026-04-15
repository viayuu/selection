import os
import numpy as np
import pyvrp
import torch
from pyvrp import Model
from time import time
from multiprocessing import Pool
from EasyNCO.data.LIBUtils import CVRPLIBWriter
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))

def pyvrp_solver_multiprocess(data_list, par_args):
    '''
        Multiprocess the HGS solver on a given dataset of CVRP problems.
        Parameters:
        - data_loader: A DataLoader object that yields batches of CVRP problem instances.
        - par_args: An object containing arguments for the HGS algorithm.
        Return:
        -result: [(tour_1, cost_1),(tour_2, cost_2),......(tour_n,cost_n)]
    '''

    problem_index = 0
    solver_task_param_list = list()

    for instance in data_list:

        depot = instance[0]
        locs = instance[1]
        demand = instance[2]
        capacity = instance[3]

        solver_task_param = (depot, locs, demand, capacity, par_args, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1

    result = multiprocess(pyvrp_solver, solver_task_param_list, par_args.cpus)
    return result


def pyvrp_solver(depot, locs, demands, capacity, par_args, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a single CVRP problem using the LKH algorithm.
        Parameters:
        - depot: The coordinates of the depot.
        - locs: A list of node coordinates.
        - demands: A list of node demands.
        - capacity: The capacity of the vehicle.
        - par_args: An object containing arguments for the HGS algorithm.
        - problem_index: it is used in multiprocess
        - distribution: Distribution of nodes
        - attributes: Distribution attributes set during data generation
        Returns:
        - A tuple containing the tour data and the total cost.
    '''
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the vehicle capacity
    result_dir = os.path.join(current_dir, "results_pyvrp_cvrp")
    os.makedirs(result_dir, exist_ok=True)
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)

    int_coord_scale = par_args.int_coord_scale
    demand_scale = par_args.demand_scale
    ptype = par_args.ptype
    
    CVRPLIBWriter(directory, depot, locs, demands, capacity, int_coord_scale, demand_scale, ptype, par_args)

    final_tour, manual_calculate_cost, duration = solve_eucilide_cvrp(depot, locs, demands, capacity, par_args, problem_index, distribution, attributes)
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{len(locs)}_c{capacity}.txt"
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)

        with open(file_path_txt, 'w') as f:

            f.write("depot_xy:")
            coord = depot
            coord[0] = coord[0]
            coord[1] = coord[1]
            f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            f.write("node_xy:")
            for coord in locs:
                coord[0] = coord[0]
                coord[1] = coord[1]
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            f.write("node_demand:")
            for demand in demands:
                f.write(f" {demand}")
            f.write("\n")
            f.write(f"capacity: {capacity}\n")

            f.write(f"scale: {len(locs)}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            f.write("solution: ")
            f.write(" ".join(map(str, final_tour)))
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("pyvrp settings: \n")
            f.write(f"num_vehicles: {par_args.pyvrp_num_vehicles}\n")
            f.write(f"pyvrp_stop: {par_args.pyvrp_stop}\n")
            f.write(f"MaxIterations: {par_args.pyvrp_MaxIterations}\n")
            f.write(f"MaxRuntime: {par_args.pyvrp_MaxRuntime}\n")
            f.write(f"seed: {par_args.SEED}\n")
            f.write(f"\n")
    return final_tour, manual_calculate_cost
    
def solve_eucilide_cvrp(depot, locs, demands, capacity, par_args, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a CVRP problem using the pyvrp library.
        Parameters:
        - node_coords: A list of node coordinates, including the depot as the first element.
        - demands: A list of node demands, with the depot demand as 0.
        - capacity: The capacity of the vehicle.
        - par_args: An object containing arguments for the HGS algorithm.
        Returns:
        - A tuple containing the tour data and the total cost.
    '''
    int_coord_scale = par_args.int_coord_scale
    demand_scale = par_args.demand_scale
    num_vehicles = par_args.pyvrp_num_vehicles
    demands = [int(d * demand_scale) for d in demands]
    capacity = int(capacity * demand_scale)

    int_coord_scale = par_args.int_coord_scale
    to_int_coord = lambda arr: (np.array(arr) * int_coord_scale + 0.5).astype(int)
    to_int_locs = to_int_coord(locs)


    m = Model()
    m.add_vehicle_type(num_vehicles, capacity=capacity)
    m.add_depot(x=depot[0], y=depot[1])
    clients = [
        m.add_client(x=to_int_locs[idx][0], y=to_int_locs[idx][1], delivery=demands[idx])
        for idx in range(0, len(locs))
    ]
    for frm in m.locations:
        for to in m.locations:
            import math
            distance = math.sqrt((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2)
            m.add_edge(frm, to, distance=distance)

    from pyvrp.stop import MaxRuntime
    from pyvrp.stop import MaxIterations
    if par_args.pyvrp_stop == 'MaxIterations':
        res = m.solve(stop=MaxIterations(par_args.pyvrp_MaxIterations), seed=par_args.SEED, display=False)
    elif par_args.pyvrp_stop == 'MaxRuntime':
        res = m.solve(stop=MaxRuntime(par_args.pyvrp_MaxRuntime), seed = par_args.SEED, display=False)  # one second
    if res.is_feasible:
        cost = res.cost()/par_args.int_coord_scale
        routes = res.best.routes()
        duration = res.runtime
        final_tour = []
        for i, route in enumerate(routes):
            final_tour.extend(route.visits())
            if i < len(routes) - 1:
                final_tour.append(0)
        final_tour = [0] + final_tour + [0]
        final_tour = np.array(final_tour).astype(int)
        all_node_xy = [depot] + locs
        # Recalculate cost to preserve more decimal places
        solution = final_tour
        node_xy_tensor = torch.tensor(all_node_xy, dtype=torch.float32)
        node_xy_tensor = node_xy_tensor.unsqueeze(0)
        sequence_tensor = torch.tensor(solution, dtype=torch.int64)

        # node_xy.shape=(1, node_num,2), sequence.shape=(m,)
        problem = node_xy_tensor[0]
        # shape: (node_num, 2)
        gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
        # shape: (m, 2)
        ordered_seq = problem.gather(dim=0, index=gathering_index)
        # shape:(m,2)
        rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
        manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())
        return final_tour, manual_calculate_cost, duration
    else:
        return [], 0