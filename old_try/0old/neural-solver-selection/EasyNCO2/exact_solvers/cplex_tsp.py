import docplex.mp.model as cpx
import math
from multiprocessing import Pool
import os
from time import time
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

def multiprocess(func, tasks, cpus=None):
    """
    Utility function to parallelize execution of a target function.
    
    Args:
        func: Target function to run in parallel (here: `cplex_tsp_solver` for single TSP instances)
        tasks: List of parameter tuples, where each tuple contains arguments for one function call
        cpus: Number of CPU cores to use; defaults to all available cores if `None`
    
    Returns:
        List of results from all tasks (order matches input `tasks` list)
    """
    # Run serially if only 1 core or 1 task (avoids multiprocessing overhead)
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    # Use a process pool to execute tasks in parallel
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))  # `starmap` unpacks parameter tuples


def cplex_tsp_solver_multiprocess(data_list, par_args):
    """
    Entry point for parallel solving of multiple TSP instances using CPLEX.
    
    Args:
        data_list: List of TSP instances; each instance is a list of node coordinates
        par_args: Parameter object containing solver settings (threads, time limit, etc.)
    
    Returns:
        List of results for all instances (each result = (optimal_path, total_distance))
    """
    # Track instance index (for unique result directories)
    problem_index = 0
    # Store parameters for each TSP instance to be solved
    solver_task_param_list = list()
    
    # Iterate over each TSP instance to prepare task parameters
    for instance in data_list:
        # Convert node coordinates to tuples (immutable for safety)
        node_coords = [(xy[0], xy[1]) for xy in instance]
        # Package parameters for the current instance
        solver_task_param = (node_coords, par_args, problem_index)
        solver_task_param_list.append(solver_task_param)
        # Increment index for the next instance
        problem_index += 1

    # Solve all instances in parallel and return results
    result = multiprocess(cplex_tsp_solver, solver_task_param_list, par_args.cpus)
    return result


def cplex_tsp_solver(node_coords, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Solve a single TSP instance using CPLEX and save results to a text file.
    
    Args:
        node_coords: List of (x, y) tuples representing node coordinates
        par_args: Parameter object with solver configurations (time limit, gap, etc.)
        problem_index: Unique index for the instance (for result storage)
        distribution: Node distribution type (e.g., uniform; for logging)
        attributes: Additional instance metadata (for logging)
    
    Returns:
        path: Optimal TSP path (list of node indices, 1-based)
        total_distance: Total distance of the optimal path
    """
    # Get directory of the current script (base path for results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create root directory for CPLEX TSP results
    result_dir = os.path.join(current_dir, "results_cplex_tsp")
    os.makedirs(result_dir, exist_ok=True)  # Avoid error if directory exists
    # Create instance-specific directory using problem index
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)
    
    # Record start time to measure solving duration
    start = time()
    # Solve the TSP instance using the core solver function
    path, total_distance = solve_euclidian_tsp(node_coords, par_args, problem_index, distribution, attributes)
    # Calculate total solving time
    duration = time() - start
    
    # Define filename for the result text file (includes instance metadata)
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{len(node_coords)}.txt"
    # Save results to file if enabled
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:
            # Write node coordinates (format: node_xy: x1 y1 x2 y2 ...)
            f.write("node_xy:")
            for coord in node_coords:
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            
            # Write problem scale (number of nodes)
            f.write(f"scale: {len(node_coords)}\n")
            
            # Write metadata (distribution and attributes)
            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")
            
            # Write the optimal path (space-separated node indices)
            f.write("solution: ")
            f.write(" ".join(map(str, path)))
            f.write("\n")
            
            # Write total cost, solving time, and CPLEX settings
            f.write(f"cost: {total_distance}\n")
            f.write(f"time: {duration} s\n")
            f.write("cplex settings: \n")
            f.write(f"cplex_num_vehicles: {par_args.cplex_num_vehicles}\n")
            f.write(f"cplex_time_limit: {par_args.cplex_time_limit}\n")
            f.write(f"cplex_gap: {par_args.cplex_gap}\n")
            f.write(f"cplex_threads: {par_args.cplex_threads}\n")
            f.write(f"\n")
    
    # Return the optimal path and its total distance
    return path, total_distance


def solve_euclidian_tsp(node_coords, par_args, problem_index = None, distribution = None, attributes = None):
    """
    Solves the Euclidean Traveling Salesman Problem (TSP) using CPLEX.
    TSP aims to find the shortest possible route that visits each city exactly once and returns to the origin city.
    """
    
    n = len(node_coords)  # Number of cities

    if n <= 1:
        return [], 0.0  # Trivial solution for 0 or 1 city
    
    # Initialize distance matrix to store Euclidean distances between all pairs of cities
    distance = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        x1, y1 = node_coords[i]
        for j in range(n):
            if i != j:  # Skip distance from a city to itself
                x2, y2 = node_coords[j]
                # Calculate Euclidean distance using math.hypot: sqrt((x1-x2)² + (y1-y2)²)
                distance[i][j] = math.hypot(x1 - x2, y1 - y2)
    try:
        # Create a CPLEX model instance named "TSP_Model"
        model = cpx.Model(name="TSP_Model")
        # Create binary variables x_ij: x_ij = 1 if the route goes from city i to city j, 0 otherwise
        x = model.binary_var_matrix(n, n, name="x")
        
        # Add constraint: Eliminate self-loops (no city can travel to itself)
        for i in range(n):
            model.add_constraint(x[i, i] == 0, f"no_self_loop_{i}")
        
        # Add constraint: Each city has exactly one outgoing edge (must depart to exactly one other city)
        for i in range(n):
            model.add_constraint(model.sum(x[i, j] for j in range(n)) == 1, f"out_degree_{i}")
        
        # Add constraint: Each city has exactly one incoming edge (must arrive from exactly one other city)
        for j in range(n):
            model.add_constraint(model.sum(x[i, j] for i in range(n)) == 1, f"in_degree_{j}")

        # Add subtour elimination constraints using the Miller-Tucker-Zemlin (MTZ) method
        # Prevents formation of disconnected sub-routes (ensures a single global tour)
        if n > 2:  # No need for subtour constraints when there are 2 or fewer cities
            # Continuous variables u[i] to track the position of city i in the tour
            u = model.continuous_var_list(n, lb=0, ub=n-1, name="u")
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        # MTZ constraint: u[i] - u[j] + n*x[i,j] ≤ n-1
                        # Ensures if there's an edge from i to j, i comes before j in the tour
                        model.add_constraint(u[i] - u[j] + n * x[i, j] <= n - 1, f"mtz_{i}_{j}")
        
        # Set objective function: Minimize the total travel distance of the tour
        model.minimize(model.sum(x[i, j] * distance[i][j] for i in range(n) for j in range(n)))
        
        # Configure solver parameters
        if par_args.cplex_time_limit:  # Set maximum solving time (seconds) if specified
            time_limit = par_args.cplex_time_limit
            model.parameters.timelimit.set(time_limit)
        if par_args.cplex_gap:  # Set acceptable MIP gap (optimality tolerance) if specified
            mip_gap = par_args.cplex_gap
            model.parameters.mip.tolerances.mipgap.set(mip_gap)
        # Set number of threads to use for solving
        model.parameters.threads.set(par_args.cplex_threads)
        # Solve the model to find the optimal tour
        solution = model.solve()

        # Check if a valid solution was found
        if not solution:
            return None, None
        
        # Construct the optimal tour from the solution
        path = []
        current = 0  # Start the tour from city 0
        path.append(current)
        
        # Traverse through the cities to build the complete route
        for _ in range(n-1):
            for j in range(n):
                # Check if the edge from current city to j is selected (x[current,j] ≈ 1)
                if current != j and solution.get_value(x[current, j]) > 0.9:
                    current = j
                    path.append(current)
                    break
        
        # Get the total distance of the optimal tour from the objective function
        total_distance = model.objective_value
        # Adjust the path format (increment all indices by 1 for consistency)
        path = [num + 1 for num in path]
        return path, total_distance
    except Exception as e:
        # Return None if an error occurs during solving
        return None, None