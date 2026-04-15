import os
import shutil
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from time import time
from EasyNCO.data import TSPGenerator
from EasyNCO.data.LIBUtils import TSPLIBWriter
from EasyNCO.data.LIBUtils import TSPLIBReader
from EasyNCO.data.LIBUtils import CVRPLIBReader
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

from multiprocessing import Pool

def multiprocess(func, tasks, cpus=None):
    """
    Utility function for parallel execution of a target function using multiprocessing.
    
    Args:
        func: Target function to execute in parallel (here, it's ortools_tsp_solver)
        tasks: List of tasks, where each element is a tuple of parameters for the target function
        cpus: Number of CPU cores to use for parallelism; defaults to all available cores
    
    Returns:
        List of results from all tasks (order matches the input 'tasks' list)
    """
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))
def ortools_tsp_solver_multiprocess(data_list, par_args):
    """
    Entry point for parallel solving of batch TSP instances.
    
    Args:
        data_list: List of TSP instances, where each instance is a collection of node coordinates
        par_args: Parameter configuration object (contains solver settings, save paths, etc.)
    
    Returns:
        List of solving results for all instances (each result includes the final route and cost)
    """
    problem_index = 0
    solver_task_param_list = list()
    for instance in data_list:
        node_coords = [(xy[0], xy[1]) for xy in instance]

        solver_task_param = (node_coords, par_args, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1

    result = multiprocess(ortools_tsp_solver, solver_task_param_list, par_args.cpus)
    return result

def ortools_tsp_solver(node_coords, par_args, problem_index = None, distribution = None, attributes = None):
    """
    OR-Tools solving logic for a single TSP instance (including result saving).
    
    Args:
        node_coords: List of node coordinates for the current instance (format: [(x1,y1), (x2,y2), ...])
        par_args: Parameter configuration object (contains time limits, save switches, etc.)
        problem_index: Instance index (used to generate a unique save directory)
        distribution: Node distribution type (e.g., uniform, clustered; for logging purposes)
        attributes: Additional instance attributes (for logging purposes, e.g., node density)
    
    Returns:
        final_route: Final solved TSP route (node numbers start from 1)
        manual_calculate_cost: Manually recalculated total route cost (preserves higher precision)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "results_ortools_tsp")
    os.makedirs(result_dir, exist_ok=True)
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)
    
    start = time()
    final_route, manual_calculate_cost = solve_euclidian_tsp(node_coords, par_args, problem_index, distribution, attributes)
    duration = time() - start

    # Save as .txt file
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{len(node_coords)}.txt"
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:
            f.write("node_xy:")
            for coord in node_coords:
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")

            f.write(f"scale: {len(node_coords)}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")
            f.write("solution: ")
            f.write(" ".join(map(str, final_route)))  # Return to start point
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("ortools settings: \n")
            f.write(f"time_limit: {par_args.ortools_time_limit}\n")

    return final_route, manual_calculate_cost

def solve_euclidian_tsp(node_coords, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Core TSP solving logic (based on OR-Tools, using Euclidean distance).
    
    Args:
        node_coords: List of node coordinates
        par_args: Parameter configuration object (contains random seeds, coordinate scaling factors, etc.)
        Other parameters: For logging purposes, no actual impact on solving
    
    Returns:
        final_route: Post-processed final route (node numbers start from 1, no duplicate end points)
        manual_calculate_cost: Manually recalculated high-precision route cost
    """
    # Get random seed (for potential randomized solving strategies; not directly used here)
    seed = par_args.SEED
    # Coordinate scaling factor (converts floating-point coordinates to integers to reduce distance calculation errors)
    int_coord_scale = par_args.int_coord_scale
    # Define a function to convert coordinates: floating-point → integer (add 0.5 to implement rounding)
    to_int_coord = lambda arr: (np.array(arr) * int_coord_scale + 0.5).astype(int)
    # Calculate the distance matrix for integer coordinates (used by OR-Tools to compute route costs)
    distance_matrix = get_distance_matrix(to_int_coord(node_coords))
    # Get the number of nodes (TSP problem scale)
    n = len(distance_matrix)
    scale = len(distance_matrix)  # Same as 'n', used for redundant recording

    # 1. Initialize OR-Tools Routing Index Manager
    # Parameters: number of nodes (n), number of vehicles (1, since TSP has only one "salesman"), start node index (0, starts from the first node)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    # 2. Initialize the routing model (based on the index manager)
    routing = pywrapcp.RoutingModel(manager)

    # 3. Define a distance callback function (used by OR-Tools to compute costs between any two nodes)
    def distance_callback(from_index, to_index):
        """
        Callback function: Converts OR-Tools internal indices to actual node distances.
        
        Args:
            from_index: OR-Tools internal index of the start node
            to_index: OR-Tools internal index of the end node
        
        Returns:
            Distance between the two nodes (retrieved from the distance matrix)
        """
        # Convert OR-Tools internal indices to actual node numbers (0-based)
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Return the distance between the corresponding nodes from the distance matrix
        return distance_matrix[from_node][to_node]
    
    # 4. Register the distance callback function and get the callback index (for internal OR-Tools calls)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # 5. Set the cost evaluation method for all vehicles (only 1 vehicle here) using the above distance callback
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 6. Configure OR-Tools solving parameters
    # Get default solving parameters (basic configuration)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Set initial solution generation strategy: PATH_CHEAPEST_ARC (greedy strategy, expands from the start node by selecting the lowest-cost arc)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Add a time constraint if a solving time limit is set (prevents excessive solving time)
    if par_args.ortools_time_limit:
        search_parameters.time_limit.seconds = par_args.ortools_time_limit

    # 7. Execute solving and get the solution object
    solution = routing.SolveWithParameters(search_parameters)

    # 8. Extract the route from the solution (convert to actual node numbers)
    # Get the OR-Tools internal index of the start node (start node for vehicle 0)
    index = routing.Start(0)
    # Initialize the route list (first add the actual node number of the start node)
    route = [manager.IndexToNode(index)]
    # Traverse the solution to extract the complete route (until reaching the end node)
    while not routing.IsEnd(index):
        # Get the internal index of the next node
        index = solution.Value(routing.NextVar(index))
        # Convert the internal index to an actual node number and add it to the route list
        route.append(manager.IndexToNode(index))

    # 9. Post-process the route: OR-Tools returns a route with a duplicate end node (e.g., [0,1,2,0]), so remove the last end node
    # Also convert node numbers from 0-based to 1-based (matches common TSP numbering conventions)
    final_route = [x + 1 for x in route[:-1]]

    # 10. Manually recalculate route cost (using original floating-point coordinates to avoid precision loss from integer distance matrices)
    # Convert the route from 1-based back to 0-based (matches the index of the original coordinate list)
    solution_0based = [x - 1 for x in final_route]
    # Import PyTorch (for efficient route cost calculation; can also use numpy)
    import torch
    # Convert original node coordinates to a PyTorch tensor (float32 type to reduce memory usage)
    node_xy_tensor = torch.tensor(node_coords, dtype=torch.float32)
    node_xy_tensor = node_xy_tensor.unsqueeze(0)  # Add a batch dimension (adapts to subsequent gather operations)
    # Convert the route to an integer tensor (used to index node coordinates)
    sequence_tensor = torch.tensor(solution_0based, dtype=torch.int64)

    # Calculate route cost: extract node coordinates in route order → compute distance between adjacent nodes → sum all distances
    problem = node_xy_tensor[0]  # Remove the batch dimension, shape: (number of nodes, 2)
    # Construct an index tensor: reshape from (route length,) to (route length, 2) (adapts to x/y dimensions of coordinates)
    gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
    # Extract node coordinates in route order, shape: (route length, 2)
    ordered_seq = problem.gather(dim=0, index=gathering_index)
    # Construct a shifted route (for calculating adjacent nodes: e.g., [a,b,c] → [b,c,a], where the last node connects back to the start)
    rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
    # Calculate Euclidean distance and sum: (x1-x2)² + (y1-y2)² → square root → sum all adjacent distances
    manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())

    # Return the post-processed final route and high-precision cost
    return final_route, manual_calculate_cost


def get_distance_matrix(node_coords):
    """
        Callback function: Converts OR-Tools internal indices to actual node distances.
        
        Args:
            from_index: OR-Tools internal index of the start node
            to_index: OR-Tools internal index of the end node
        
        Returns:
            Distance between the two nodes (retrieved from the distance matrix)
    """
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




