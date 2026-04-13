import numpy as np
from gurobipy import *
import math
import itertools
from multiprocessing import Pool
import os
from time import time
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

def multiprocess(func, tasks, cpus=None):
    """
    Utility function to parallelize execution of a target function.
    
    Args:
        func: Target function to run in parallel (here: gurobi_tsp_solver)
        tasks: List of parameter tuples, where each tuple contains arguments for one function call
        cpus: Number of CPU cores to use; defaults to all available cores if None
    
    Returns:
        List of results from all tasks (order matches input tasks)
    """
    # Run serially if only 1 core or 1 task (avoids multiprocessing overhead)
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    # Use a process pool to run tasks in parallel
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))


def gurobi_tsp_solver_multiprocess(data_list, par_args):
    """
    Entry point for parallel solving of multiple TSP instances using Gurobi.
    
    Args:
        data_list: List of TSP instances; each instance is a list of node coordinates
        par_args: Parameter object containing solver settings (threads, time limit, etc.)
    
    Returns:
        List of results for all instances (each result is a tuple of tour and cost)
    """
    # Track the index of each problem (for unique result directories)
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
    result = multiprocess(gurobi_tsp_solver, solver_task_param_list, par_args.cpus)
    return result


def gurobi_tsp_solver(node_coords, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Solve a single TSP instance using Gurobi and save results to a text file.
    
    Args:
        node_coords: List of (x, y) tuples representing node coordinates
        par_args: Parameter object with solver configurations
        problem_index: Unique index for the instance (for result storage)
        distribution: Node distribution type (e.g., uniform; for logging)
        attributes: Additional instance metadata (for logging)
    
    Returns:
        tour: Optimal TSP tour (list of node indices)
        cost: Total cost of the optimal tour
    """
    # Get the directory of the current script (base path for results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create root directory for Gurobi TSP results
    result_dir = os.path.join(current_dir, "results_gurobi_tsp")
    os.makedirs(result_dir, exist_ok=True)  # Avoid error if directory exists
    # Create instance-specific directory using problem index
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)
    
    # Record start time to measure solving duration
    start = time()
    # Solve the TSP instance using the core solver function
    tour, cost = solve_euclidian_tsp(node_coords, par_args, problem_index, distribution, attributes)
    # Calculate total solving time
    duration = time() - start
    
    # Define filename for the result text file (includes instance metadata)
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{len(node_coords)}.txt"
    # Save results to file if enabled
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:
            # Write node coordinates
            f.write("node_xy:")
            for coord in node_coords:
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            
            # Write problem scale (number of nodes)
            f.write(f"scale: {len(node_coords)}\n")
            
            # Write metadata (distribution and attributes)
            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")
            
            # Write the optimal tour (space-separated node indices)
            f.write("solution: ")
            f.write(" ".join(map(str, tour)))
            f.write("\n")
            
            # Write total cost, solving time, and Gurobi settings
            f.write(f"cost: {cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("gurobi settings: \n")
            f.write(f"gurobi_threads: {par_args.gurobi_threads}\n")
            f.write(f"gurobi_time_limit: {par_args.gurobi_time_limit}\n")
            f.write(f"gurobi_gap: {par_args.gurobi_gap}\n")
            f.write(f"gurobi_num_vehicles: {par_args.gurobi_num_vehicles}\n")
            f.write(f"\n")
    
    # Return the optimal tour and its cost
    return tour, cost


def solve_euclidian_tsp(node_coords, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Core function to solve the Euclidean TSP to optimality using Gurobi's MIP solver.
    Uses the Miller-Tucker-Zemlin (MTZ) formulation for subtour elimination.
    
    Args:
        node_coords: List of (x, y) tuples for node coordinates
        par_args: Parameter object with Gurobi settings
        Other args: Metadata for logging (no impact on solving)
    
    Returns:
        tour: List of node indices in the optimal order
        cost: Total distance of the optimal tour
    """
    # Number of nodes in the TSP instance
    n = len(node_coords)
    
    # Handle edge case: if 1 node or fewer, return trivial solution
    if n <= 1:
        return [0, 0], 0.0  # Trivial tour (start and end at the only node)
    
    # 1. Precompute the distance matrix between all pairs of nodes
    dist = {}  # dist[(i, j)] = Euclidean distance from node i to node j
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops (distance from node to itself is irrelevant)
                x1, y1 = node_coords[i]
                x2, y2 = node_coords[j]
                # Calculate Euclidean distance using math.hypot (sqrt((x1-x2)² + (y1-y2)²))
                dist[(i, j)] = math.hypot(x1 - x2, y1 - y2)
    
    try:
        # 2. Create a new Gurobi model
        m = Model("TSP")
        
        # 3. Create binary decision variables: x[i,j] = 1 if the tour goes from i to j, else 0
        x = m.addVars(dist.keys(), vtype=GRB.BINARY, name="x")
        
        # 4. Create continuous variables for MTZ subtour elimination (u[i])
        # These variables help enforce that the tour is a single cycle (no subtours)
        u = m.addVars(n, vtype=GRB.CONTINUOUS, name="u")
        # Fix u[0] to 1 (arbitrary starting point to break symmetry)
        m.addConstr(u[0] == 1)
        # Constraints to bound u[i] (ensures valid ordering for subtour elimination)
        m.addConstrs(u[i] >= 2 for i in range(1, n))  # u[i] ≥ 2 for non-start nodes
        m.addConstrs(u[i] <= n for i in range(n))      # u[i] ≤ n for all nodes
        
        # 5. Set objective: minimize the total distance of the tour
        m.setObjective(quicksum(dist[i,j] * x[i,j] for i,j in dist.keys()), GRB.MINIMIZE)
        
        # 6. Add core TSP constraints:
        # a. Each node has exactly 1 outgoing edge (outdegree = 1)
        m.addConstrs(quicksum(x[i,j] for j in range(n) if i != j) == 1 for i in range(n))
        # b. Each node has exactly 1 incoming edge (indegree = 1)
        m.addConstrs(quicksum(x[i,j] for i in range(n) if i != j) == 1 for j in range(n))
        
        # 7. Add MTZ subtour elimination constraints
        # For all pairs of non-start nodes (i, j), prevent i → j from forming a subtour
        for i, j in itertools.product(range(1, n), range(1, n)):
            if i != j and (i, j) in dist:
                # u[i] - u[j] + (n-1)*x[i,j] ≤ n-2
                # Intuition: If x[i,j] = 1, then u[i] < u[j], enforcing a global order
                m.addConstr(u[i] - u[j] + (n-1)*x[i,j] <= n-2)
        
        # 8. Configure Gurobi solver parameters
        m.Params.MIPGap = par_args.gurobi_gap          # Acceptable optimality gap (e.g., 0.01 = 1%)
        m.Params.TimeLimit = par_args.gurobi_time_limit  # Maximum solving time (seconds)
        m.Params.Threads = par_args.gurobi_threads      # Number of CPU threads to use
        
        # 9. Solve the model
        m.optimize()
        
        # 10. Extract the optimal tour from the solution
        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):  # Check if solution is found (optimal or time-limited)
            tour = []
            current = 0  # Start at node 0
            tour.append(current)
            # Traverse the tour to collect node indices (n-1 steps for n nodes)
            for _ in range(n-1):
                for j in range(n):
                    # Check if the edge from current node to j is selected (x[current,j] ≈ 1)
                    if current != j and (current, j) in dist and x[current,j].x > 0.5:
                        tour.append(j)
                        current = j  # Move to next node in the tour
                        break
            # Adjust the path format (increment all indices by 1 for consistency)
            tour = [num + 1 for num in tour]
            return tour, m.objVal  # Return tour and total cost
        else:
            # No solution found (e.g., infeasible or interrupted)
            return None, None   
    
    except Exception as e:
        # Log any errors during solving
        logger.info(f"Error during solving: {e}")
        return None, None
