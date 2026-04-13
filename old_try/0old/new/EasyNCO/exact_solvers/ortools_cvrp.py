import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from time import time
from EasyNCO.data import CVRPGenerator
from EasyNCO.data.LIBUtils import CVRPLIBWriter
from EasyNCO.data.LIBUtils import TSPLIBReader
from EasyNCO.data.LIBUtils import CVRPLIBReader
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

from multiprocessing import Pool


def multiprocess(func, tasks, cpus=None):
    """
    Utility function to run a target function in parallel using multiprocessing.
    
    Args:
        func: Target function to execute (here: ortools_cvrp_solver for single CVRP instances)
        tasks: List of parameter tuples, where each tuple contains args for one call to `func`
        cpus: Number of CPU cores to use; defaults to all available cores if None
    
    Returns:
        List of results from each task (order matches input `tasks` list)
    """
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))


def ortools_cvrp_solver_multiprocess(data_list, par_args):
    """
    Entry point for parallel solving of batch CVRP instances.
    
    Args:
        data_list: List of CVRP instances; each instance is a list [depot, locs, demand, capacity]
        par_args: Parameter configuration object (contains solver settings, save flags, etc.)
    
    Returns:
        List of solving results for all instances; each result is (routes, cost)
    """
    problem_index = 0
    solver_task_param_list = list()
    # Iterate over each batch and instance in the data loader
    for instance in data_list:

        depot = instance[0]
        locs = instance[1]
        demand = instance[2]
        capacity = instance[3]

        solver_task_param = (depot, locs, demand, capacity, par_args, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1

    result = multiprocess(ortools_cvrp_solver, solver_task_param_list, par_args.cpus)
    return result


def ortools_cvrp_solver(depot, locs, demands, capacity, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Solve a single CVRP instance using OR-Tools, and save results to a text file.
    
    Args:
        depot: Tuple (x, y) of depot coordinates
        locs: List of (x, y) tuples for customer locations
        demands: List of demands (one per customer; original list excludes depot)
        capacity: Maximum capacity per vehicle (converted to integer for consistency)
        par_args: Configuration object (e.g., time limits, save flags, scaling factors)
        problem_index: Unique index for the instance (for result directory)
        distribution: Customer location distribution (e.g., uniform, clustered; for logging)
        attributes: Additional instance metadata (e.g., node density; for logging)
    
    Returns:
        routes: List of routes (one per vehicle); each route is a list of node indices
        cost: Total cost of the solution (from OR-Tools, scaled back to original coordinates)
    """
    # Get directory of the current script (base path for saving results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Ensure vehicle capacity is an integer (avoids floating-point errors in constraints)
    capacity = int(capacity)
    # Number of customers (CVRP "scale" — excludes depot)
    scale = len(locs)
    
    # Create result directories (avoid errors if directories already exist)
    result_dir = os.path.join(current_dir, "results_ortools_cvrp")  # Root result directory
    os.makedirs(result_dir, exist_ok=True)
    directory = os.path.join(result_dir, str(problem_index))       # Instance-specific directory
    os.makedirs(directory, exist_ok=True)

    # Record start time to measure solving duration
    start = time()
    # Call core CVRP solving function to get vehicle routes and total cost
    routes, cost = solve_euclidian_cvrp(depot, locs, demands, capacity, par_args, problem_index, distribution, attributes)
    # Calculate total solving time (seconds)
    duration = time() - start
    
    # Define filename for the result text file (includes instance metadata for easy identification)
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{scale}_c{capacity}.txt"
    # Import PyTorch for high-precision cost recalculation (avoids OR-Tools integer scaling loss)
    import torch
    # Combine depot and customer coordinates into a single list (depot is first node: index 0)
    all_node_xy = [depot] + locs
    # Build a single "final_tour" by concatenating all vehicle routes (for result visualization/logging)
    final_tour = []
    for i, route in enumerate(routes):
        # Only add non-trivial routes (skip empty routes or routes with just the depot)
        if len(route) > 1:
            final_tour.extend(route)
    # Append depot (index 0) to the end to close the full loop (optional for logging)
    final_tour.append(0)
    # Convert final_tour to an integer numpy array (for consistency in downstream use)
    final_tour = np.array(final_tour).astype(int)
    
    # Recalculate cost using original floating-point coordinates (OR-Tools uses scaled integers)
    solution = final_tour  # Reuse final_tour as the solution sequence
    # Convert coordinates to a PyTorch tensor (float32 for memory efficiency)
    node_xy_tensor = torch.tensor(all_node_xy, dtype=torch.float32)
    node_xy_tensor = node_xy_tensor.unsqueeze(0)  # Add batch dimension (required for gather operation)
    # Convert solution sequence to integer tensor (for indexing coordinates)
    sequence_tensor = torch.tensor(solution, dtype=torch.int64)

    # Extract coordinates in the order of the solution sequence
    problem = node_xy_tensor[0]  # Remove batch dimension: shape (total_nodes, 2)
    # Reshape sequence tensor to match coordinate dimensions (shape: (sequence_length, 2))
    gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
    # Gather coordinates in the solution order: shape (sequence_length, 2)
    ordered_seq = problem.gather(dim=0, index=gathering_index)
    # Shift sequence by 1 (to calculate distance between adjacent nodes: node i → node i+1)
    rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
    # Calculate Euclidean distance between adjacent nodes, sum to get total cost
    manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())
    
    # Save results to a text file if the save flag is enabled
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        # Open file in write mode (overwrites existing file; creates new if missing)
        with open(file_path_txt, 'w') as f:
            # Write depot coordinates (format: depot_xy: x y)
            f.write("depot_xy:")
            coord = depot
            # Redundant assignments (coord[0] = coord[0]) kept for consistency with original code
            coord[0] = coord[0]
            coord[1] = coord[1]
            f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            
            # Write customer coordinates (format: node_xy: x1 y1 x2 y2 ...)
            f.write("node_xy:")
            for coord in locs:
                coord[0] = coord[0]
                coord[1] = coord[1]
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            
            # Write customer demands (skips depot demand, which is 0; format: node_demand: d1 d2 ...)
            f.write("node_demand:")
            for demand in demands[1:]:
                f.write(f" {demand}")
            f.write("\n")
            
            # Write key CVRP parameters: capacity, number of customers, distribution, attributes
            f.write(f"capacity: {capacity}\n")
            f.write(f"scale: {scale}\n")
            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            # Format final_tour as a space-separated string (e.g., "0 1 2 0 3 ...")
            tour_str_list = final_tour.astype(str).tolist()  # Convert numpy array to string list
            tour_str = " ".join(tour_str_list)               # Join with spaces
            f.write("solution: ")
            f.write(tour_str)
            f.write("\n")
            
            # Write high-precision cost, solving time, and OR-Tools settings
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("ortools settings: \n")
            f.write(f"num_vehicles: {par_args.ortools_num_vehicles}\n")
            f.write(f"time_limit: {par_args.ortools_time_limit}\n")
            f.write(f"\n")
    
    # Return OR-Tools-generated routes and scaled cost
    return routes, cost


def solve_euclidian_cvrp(depot, locs, demands, capacity, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Core CVRP solving logic using OR-Tools (Euclidean distance for routing costs).
    
    Args:
        depot: Depot coordinates (x, y)
        locs: Customer coordinates (list of (x, y) tuples)
        demands: Customer demands (list; original excludes depot)
        capacity: Vehicle capacity (integer)
        par_args: Configuration object (seed, coordinate scale, time limit, etc.)
        Other args: Metadata for logging (no impact on solving)
    
    Returns:
        routes: List of vehicle routes (each route = list of node indices)
        cost: Total solution cost (scaled back to original coordinate units)
    """
    # Unpack configuration parameters
    seed = par_args.SEED  # Random seed (for potential randomized strategies; not used here)
    int_coord_scale = par_args.int_coord_scale  # Scaling factor to convert coords to integers
    num_vehicles = par_args.ortools_num_vehicles  # Number of vehicles available for CVRP
    
    # Combine depot and customers into a single node list (depot = index 0)
    node_coords = [depot] + locs
    # Lambda function to scale floating-point coordinates to integers (rounds via +0.5)
    to_int_coord = lambda arr: (np.array(arr) * int_coord_scale + 0.5).astype(int)
    # Convert demands to integers (avoids constraint errors) and add depot demand (0)
    demands = [int(num) for num in demands]
    demands = [0] + demands  # Depot has 0 demand (no need to deliver to itself)

    # Calculate integer-distance matrix (OR-Tools uses this for routing cost calculations)
    distance_matrix = get_distance_matrix(to_int_coord(node_coords))

    # Build OR-Tools data model (contains problem constraints: distances + demands)
    data = {'distance_matrix': distance_matrix, 'demands': demands}
    # Initialize OR-Tools Index Manager: maps node indices to solver-internal indices
    # Params: total_nodes, num_vehicles, depot_index (0 = first node in node_coords)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), num_vehicles, 0)

    # Create OR-Tools Routing Model (core object for solving CVRP)
    routing = pywrapcp.RoutingModel(manager)

    # Define distance callback: returns cost between two nodes (used by OR-Tools)
    def distance_callback(from_index, to_index):
        """Convert solver-internal indices to node numbers and return distance."""
        from_node = manager.IndexToNode(from_index)  # Solver index → actual node number
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]  # Get distance from matrix

    # Register distance callback with the routing model
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Set all vehicles to use the distance callback for cost calculation
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint: vehicles cannot exceed their maximum capacity
    def demand_callback(index):
        """Return demand of a node (used to track capacity usage per vehicle)."""
        node = manager.IndexToNode(index)
        return data['demands'][node]

    # Register demand callback (unary callback = single input: node index)
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    # Add capacity dimension to the routing model:
    # Params: callback_index, slack (0 = no extra capacity), capacity_per_vehicle, fix_start_cumul (True = start at 0), dimension_name
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [capacity] * num_vehicles, True, 'Capacity')

    # Configure OR-Tools solving parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Set initial solution strategy: PATH_CHEAPEST_ARC (greedy: always pick lowest-cost edge)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Add time limit if specified (prevents solver from running indefinitely)
    if par_args.ortools_time_limit:
        search_parameters.time_limit.seconds = par_args.ortools_time_limit

    # Solve the CVRP and get the solution object
    solution = routing.SolveWithParameters(search_parameters)

    # Extract routes and cost if a valid solution exists
    if solution:
        routes, cost = get_routes_and_cost(manager, routing, solution)
        # Scale cost back to original coordinate units (undo int_coord_scale)
        cost = cost / par_args.int_coord_scale
        return routes, cost
def get_distance_matrix(node_coords):
    """Calculates the distance matrix for the given node coordinates."""
    import math
    n = len(node_coords)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            distance = math.sqrt((node_coords[i][0] - node_coords[j][0]) ** 2 +
                                 (node_coords[i][1] - node_coords[j][1]) ** 2)
            distance_matrix[i][j] = round(distance)
            distance_matrix[j][i] = round(distance)
    return distance_matrix


def get_routes_and_cost(manager, routing, solution):
    """
    Extract vehicle routes and total solution cost from an OR-Tools CVRP solution.
    
    Args:
        manager: OR-Tools IndexManager (maps solver indices to node numbers)
        routing: OR-Tools RoutingModel (CVRP model object)
        solution: OR-Tools solution object (contains optimized routes)
    
    Returns:
        routes: List of routes (one per vehicle); each route = list of node indices
        total_cost: Total cost of the solution (from OR-Tools, integer-scaled)
    """
    routes = []
    total_cost = 0
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        routes.append(route)

    total_cost = solution.ObjectiveValue()
    return routes, total_cost

