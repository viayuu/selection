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
    Utility function to run a target function in parallel using multiprocessing.
    
    Args:
        func: Target function to parallelize (here: `gurobi_cvrp_solver` for single CVRP instances)
        tasks: List of parameter tuples—each tuple contains arguments for one call to `func`
        cpus: Number of CPU cores to use; defaults to all available cores if `None`
    
    Returns:
        List of results from all tasks (order matches the input `tasks` list)
    """
    # Run serially if only 1 core is allowed or only 1 task exists (avoids multiprocessing overhead)
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    # Create a process pool, execute tasks in parallel, and return results
    # `starmap` unpacks each parameter tuple to pass arguments to `func`
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))


def gurobi_cvrp_solver_multiprocess(data_list, par_args):
    """
    Entry point for parallel solving of batch CVRP instances using Gurobi.
    
    Args:
        data_list: List of CVRP instances—each instance should contain (depot, locs, demands, capacity)
        par_args: Parameter object with solver settings (e.g., threads, time limit, save flags)
    
    Returns:
        List of solving results for all instances (each result = (routes, total_cost))
    """
    # Track the index of each CVRP instance (for unique result directory names)
    problem_index = 0
    # Store parameter tuples for each instance (feeds into the multiprocessing function)
    solver_task_param_list = list()
    
    # Iterate over each CVRP instance to prepare task parameters
    for instance in data_list:
        # Convert node coordinates to tuples (immutable, preventing accidental modifications)
        depot = instance[0]
        locs = instance[1]
        demand = instance[2]
        capacity = instance[3]
        # Package parameters for the current instance: (depot, locs, demand, capacity, par_args, problem_index)
        solver_task_param = (depot, locs, demand, capacity, par_args, problem_index)
        solver_task_param_list.append(solver_task_param)
        # Increment index for the next instance
        problem_index += 1

    # Solve all instances in parallel and return the results
    result = multiprocess(gurobi_cvrp_solver, solver_task_param_list, par_args.cpus)
    # Note: The original code missing a return statement—added here to pass results upstream
    return result


def gurobi_cvrp_solver(depot, locs, demands, capacity, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Solve a single CVRP instance using Gurobi and save results to a text file.
    
    Args:
        depot: Tuple (x, y) of the depot coordinates (start/end point for all vehicles)
        locs: List of (x, y) tuples for customer locations
        demands: List of customer demands (one value per customer; excludes depot)
        capacity: Maximum load capacity per vehicle (integer)
        par_args: Parameter object with solver/config settings (e.g., threads, time limit)
        problem_index: Unique index for the instance (for creating a dedicated result directory)
        distribution: Customer location distribution (e.g., uniform, clustered; for logging)
        attributes: Additional instance metadata (e.g., node density; for logging)
    
    Returns:
        routes: List of vehicle routes—each route is a list of customer indices (0 = depot)
        cost: Total distance of the optimal CVRP solution
    """
    # Get the directory of the current script (base path for saving results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create a root directory for Gurobi CVRP results (avoids errors if it already exists)
    result_dir = os.path.join(current_dir, "results_gurobi_cvrp")
    os.makedirs(result_dir, exist_ok=True)
    # Create an instance-specific directory (using `problem_index` to avoid overwriting)
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)
    
    # Record the start time to calculate total solving duration
    start = time()
    # Call the core CVRP solving function to get routes and total cost
    routes, cost = solve_euclidian_cvrp(depot, locs, demands, capacity, par_args, problem_index, distribution, attributes)
    # Calculate total time spent solving the instance
    duration = time() - start
    
    # Define the filename for the result text file (includes metadata for easy identification)
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{len(locs)}.txt"
    # Save results to a text file if the `save_as_txt` flag is enabled in `par_args`
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        # Open the file in write mode (overwrites existing files; creates new if missing)
        with open(file_path_txt, 'w') as f:
            # Write depot coordinates (format: depot: x y)
            f.write("depot:")
            f.write(f" {depot[0]} {depot[1]}")
            f.write("\n")

            # Write customer locations (format: locs: x1 y1 x2 y2 ...)
            f.write("locs:")
            for coord in locs:
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")

            # Write customer demands (format: demands: d1 d2 ...)
            f.write("demands:")
            for d in demands:
                f.write(f" {d}")
            f.write("\n")

            # Write key CVRP parameters: vehicle capacity, number of vehicles
            f.write(f"capacity: {capacity}\n")
            f.write(f"num_vehicles: {par_args.gurobi_num_vehicles}\n")

            # Write instance metadata (distribution, attributes)
            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")
            
            # Write the optimal solution (routes)
            f.write("solution:\n")
            f.write("0 ")  # Start all routes at the depot (index 0)
            for idx, route in enumerate(routes):
                # Convert route indices to strings and join with spaces
                f.write(" ".join(map(str, route)))
                f.write(" 0 ")  # End each route at the depot (index 0)
            f.write("\n")
            
            # Write total cost, solving time, and Gurobi solver settings
            f.write(f"cost: {cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("gurobi settings: \n")
            f.write(f"gurobi_threads: {par_args.gurobi_threads}\n")
            f.write(f"gurobi_time_limit: {par_args.gurobi_time_limit}\n")
            f.write(f"gurobi_gap: {par_args.gurobi_gap}\n")
            f.write(f"gurobi_num_vehicles: {par_args.gurobi_num_vehicles}\n")
            f.write(f"\n")
    
    # Return the optimal routes and total distance cost
    return routes, cost


def solve_euclidian_cvrp(depot, locs, demands, capacity, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Core function to solve the Euclidean CVRP using Gurobi's MIP solver.
    Uses Miller-Tucker-Zemlin (MTZ) constraints to enforce capacity limits and eliminate subtours.
    
    Args:
        depot: Tuple (x, y) of depot coordinates
        locs: List of (x, y) tuples for customer locations
        demands: List of customer demands (excludes depot)
        capacity: Maximum capacity per vehicle
        par_args: Parameter object with Gurobi solver settings
        Other args: Metadata for logging (no impact on solving)
    
    Returns:
        routes: List of optimal vehicle routes (each route = list of customer indices)
        cost: Total distance of the optimal solution
    """
    # Combine depot and customer coordinates into a single list (depot = index 0)
    node_coords = [depot] + locs
    # Add 0 demand for the depot (depots don't require delivery)
    demands = [0] + demands
    # Total number of nodes (depot + customers)
    n = len(node_coords)
    # List of customer indices (exclude depot, which is index 0)
    customers = [i for i in range(n) if i != 0]
    # Number of available vehicles (from configuration parameters)
    num_vehicles = par_args.gurobi_num_vehicles

    # 1. Precompute Euclidean distance matrix between all pairs of nodes
    dist = {}  # Key: (i, j) (node pair), Value: Euclidean distance from i to j
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-loops (distance from a node to itself is irrelevant)
                x1, y1 = node_coords[i]
                x2, y2 = node_coords[j]
                # Calculate Euclidean distance using `math.hypot` (efficient for 2D coordinates)
                dist[i, j] = math.hypot(x1 - x2, y1 - y2)

    try:
        # 2. Create a new Gurobi MIP model named "CVRP"
        m = Model("CVRP")
        
        # 3. Define decision variables
        # x[i, j]: Binary variable (1 = vehicle travels from node i to j; 0 = no travel)
        x = m.addVars(dist.keys(), vtype=GRB.BINARY, name="x")
        # u[i]: Continuous variable (cumulative load of the vehicle when arriving at customer i)
        u = m.addVars(customers, vtype=GRB.CONTINUOUS, name="u")
        
        # 4. Set objective function: Minimize total travel distance
        m.setObjective(quicksum(dist[i, j] * x[i, j] for i, j in dist.keys()), GRB.MINIMIZE)
        
        # 5. Add core CVRP constraints
        # Constraint 1: Each customer has exactly 1 incoming edge (ensures all customers are visited)
        m.addConstrs(
            quicksum(x[i, j] for i, j in dist.keys() if j == customer) == 1 
            for customer in customers
        )
        
        # Constraint 2: Each customer has exactly 1 outgoing edge (ensures no stuck customers)
        m.addConstrs(
            quicksum(x[i, j] for i, j in dist.keys() if i == customer) == 1 
            for customer in customers
        )
        
        # Constraint 3: Limit the number of vehicles used (no more than available vehicles)
        # Counts how many vehicles leave the depot (each vehicle starts at depot: x[0, j])
        m.addConstr(quicksum(x[0, j] for j in customers) <= num_vehicles)
        
        # Constraint 4: MTZ capacity constraints (prevent subtours and enforce load limits)
        # For all pairs of customers (i, j), if a vehicle travels i→j, its load at i must be ≥ load at j + demand of j
        for i, j in itertools.product(customers, customers):
            if i != j and (i, j) in dist:
                m.addConstr(u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j])
        
        # Constraint 5: Bounds for cumulative load variable `u`
        m.addConstrs(u[i] >= demands[i] for i in customers)  # Load at i ≥ demand of i (can't deliver less than needed)
        m.addConstrs(u[i] <= capacity for i in customers)    # Load at i ≤ vehicle capacity (no overloading)
            
        # 6. Configure Gurobi solver parameters
        m.Params.MIPGap = par_args.gurobi_gap          # Acceptable optimality gap (e.g., 0.01 = 1% error allowed)
        m.Params.TimeLimit = par_args.gurobi_time_limit  # Maximum solving time (seconds; prevents infinite runs)
        m.Params.Threads = par_args.gurobi_threads      # Number of CPU threads to use (speeds up solving)
        
        # 7. Solve the MIP model
        m.optimize()
        
        # 8. Extract optimal routes from the solved model
        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
            # Print total distance (for debugging/verification)
            print(f"Optimal total distance: {m.objVal:.2f}")
            
            # Reconstruct routes from the binary variable `x`
            routes = []  # Stores all vehicle routes
            used_customers = set()  # Tracks customers already assigned to a route
            
            # Iterate over all possible depot exits (x[0, j] = 1 means a vehicle starts at depot → j)
            for j in customers:
                if x[0, j].x > 0.5 and j not in used_customers:
                    current_route = []  # Route for the current vehicle
                    current_node = j    # Start at the first customer of the route
                    
                    # Traverse until returning to the depot (current_node = 0)
                    while current_node != 0:
                        current_route.append(current_node)  # Add current customer to the route
                        used_customers.add(current_node)    # Mark customer as used
                        
                        # Find the next node in the route (x[current_node, k] = 1)
                        for k in range(n):
                            if current_node != k and x[current_node, k].x > 0.5:
                                current_node = k
                                break
                    
                    # Add the completed route to the list of routes
                    routes.append(current_route)
            
            # Return the list of routes and total distance cost
            return routes, m.objVal
        else:
            # No valid solution found (e.g., infeasible problem, solver interrupted)
            return [], 0
    
    # Catch and log any errors during model creation or solving
    except Exception as e:
        logger.info(f"Error during CVRP solving: {e}")
        return [], 0