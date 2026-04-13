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
        func: Target function to run in parallel (here: `cplex_cvrp_solver` for single CVRP instances)
        tasks: List of parameter tuples—each tuple contains arguments for one function call
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


def cplex_cvrp_solver_multiprocess(data_list, par_args):
    """
    Entry point for parallel solving of batch CVRP instances using CPLEX.
    
    Args:
        data_list: List of CVRP instances—each instance is a tuple (depot, locs, demands, capacity)
        par_args: Parameter object with solver settings (threads, time limit, etc.)
    
    Returns:
        List of solving results for all instances (each result = (routes, total_cost))
    """
    # Track instance index (for unique result directory names)
    problem_index = 0
    # Store parameter tuples for each CVRP instance
    solver_task_param_list = list()
    
    # Iterate over each CVRP instance to prepare task parameters
    for instance in data_list:
        # Unpack CVRP components from the instance tuple
        depot = instance[0]       # Depot coordinates (x, y)
        locs = instance[1]        # Customer locations (list of (x, y) tuples)
        demand = instance[2]      # Customer demands (list)
        capacity = instance[3]    # Vehicle capacity
        
        # Package parameters for the current instance
        solver_task_param = (depot, locs, demand, capacity, par_args, problem_index)
        solver_task_param_list.append(solver_task_param)
        # Increment index for the next instance
        problem_index += 1

    # Solve all instances in parallel and return results
    result = multiprocess(cplex_cvrp_solver, solver_task_param_list, par_args.cpus)
    return result


def cplex_cvrp_solver(depot, locs, demands, capacity, par_args, problem_index=None, distribution=None, attributes=None):
    """
    Solve a single CVRP instance using CPLEX and save results to a text file.
    
    Args:
        depot: Tuple (x, y) of depot coordinates (start/end for all vehicles)
        locs: List of (x, y) tuples for customer locations
        demands: List of customer demands (excludes depot)
        capacity: Maximum load capacity per vehicle
        par_args: Parameter object with solver configurations (time limit, gap, etc.)
        problem_index: Unique index for the instance (for result storage)
        distribution: Customer location distribution (e.g., uniform; for logging)
        attributes: Additional instance metadata (for logging)
    
    Returns:
        routes: List of vehicle routes—each route is a list of customer indices (0 = depot)
        cost: Total distance of the optimal solution
    """
    # Get directory of the current script (base path for results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create root directory for CPLEX CVRP results
    result_dir = os.path.join(current_dir, "results_cplex_cvrp")
    os.makedirs(result_dir, exist_ok=True)  # Avoid error if directory exists
    # Create instance-specific directory using problem index
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)
    
    # Record start time to measure solving duration
    start = time()
    # Solve the CVRP instance using the core solver function
    routes, cost = solve_euclidian_cvrp(depot, locs, demands, capacity, par_args, problem_index, distribution, attributes)
    # Calculate total solving time
    duration = time() - start
    
    # Define filename for the result text file (includes metadata)
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{len(locs)}.txt"
    # Save results to file if enabled
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
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
            f.write(f"num_vehicles: {par_args.cplex_num_vehicles}\n")

            # Write instance metadata (distribution, attributes)
            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")
            
            # Write the optimal solution (routes)
            f.write("solution: ")
            f.write("0 ")  # All routes start at depot (index 0)
            for idx, route in enumerate(routes):
                # Convert route indices to strings and join with spaces
                f.write(" ".join(map(str, route)))
                f.write(" 0 ")  # All routes end at depot (index 0)
            f.write("\n")
            
            # Write total cost, solving time, and CPLEX settings
            f.write(f"cost: {cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("cplex settings: \n")
            f.write(f"cplex_num_vehicles: {par_args.cplex_num_vehicles}\n")
            f.write(f"cplex_time_limit: {par_args.cplex_time_limit}\n")
            f.write(f"cplex_gap: {par_args.cplex_gap}\n")
            f.write(f"cplex_threads: {par_args.cplex_threads}\n")
            f.write(f"\n")
    
    # Return the optimal routes and total distance cost
    return routes, cost


def solve_euclidian_cvrp(depot, locs, demands, capacity, par_args, problem_index = None, distribution = None, attributes = None):
    """
    Solve the Capacitated Vehicle Routing Problem (CVRP) using CPLEX.
    
    Args:
        distance_matrix: n×n distance matrix between nodes
        demands: Demand list for each node (index 0 is depot, usually with demand 0)
        capacity: Maximum load capacity of each vehicle
        num_vehicles: Number of available vehicles
        depot: Index of the depot (default: 0)
    Returns:
        List of vehicle routes, total distance, solving time
    """
    # Combine depot coordinates with customer coordinates into a complete node list (depot = index 0)
    node_coords = [depot] + locs
    # Add demand 0 for the depot (depot has no delivery demand) and merge with customer demands
    demands = [0] + demands  # Depot demand is 0
    # Calculate total number of nodes (depot + customers)
    n = len(node_coords)  # Total number of nodes (including depot)
    # Initialize an n×n distance matrix to store Euclidean distances between all node pairs
    distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    # Iterate over all node pairs to compute Euclidean distances and populate the matrix
    for i in range(n):
        x1, y1 = node_coords[i]
        for j in range(n):
            if i != j:  # Skip distance from a node to itself (default 0, no need to compute)
                x2, y2 = node_coords[j]
                # Compute Euclidean distance using math.hypot: sqrt((x1-x2)² + (y1-y2)²)
                distance_matrix[i][j] = math.hypot(x1 - x2, y1 - y2)

    # Filter indices of all customer nodes (exclude depot node 0)
    customers = [i for i in range(n) if i != 0]  # Customer node indices
    
    try:
        # 1. Create a CPLEX model instance named "CVRP_Model"
        model = cpx.Model(name="CVRP_Model")
        
        # 2. Create binary decision variables x[i,j]: 
        # x[i,j] = 1 if a vehicle travels from node i to node j; 0 otherwise
        x = model.binary_var_matrix(n, n, name="x")
        
        # 3. Add constraint: Eliminate self-loops (vehicles cannot travel from a node to itself)
        for i in range(n):
            model.add_constraint(x[i, i] == 0, f"no_self_loop_{i}")  # Constraint name with index for debugging
        
        # 4. Add flow conservation constraints: For all customers (excluding depot),
        # each customer has exactly 1 outgoing edge and 1 incoming edge (ensures each customer is visited once)
        for i in customers:
            # Out-degree constraint: Total outgoing edges from customer i = 1 (can only go to one next node)
            model.add_constraint(model.sum(x[i, j] for j in range(n)) == 1, 
                                f"out_degree_{i}")
            # In-degree constraint: Total incoming edges to customer i = 1 (can only come from one previous node)
            model.add_constraint(model.sum(x[j, i] for j in range(n)) == 1, 
                                f"in_degree_{i}")
        
        # 5. Add in/out-degree constraints for the depot:
        # The number of outgoing/incoming edges from/to the depot equals the number of available vehicles
        # (each vehicle starts and ends at the depot)
        # Out-degree constraint for depot: Edges from depot 0 to customers = number of vehicles
        model.add_constraint(model.sum(x[0, j] for j in customers) == par_args.cplex_num_vehicles,
                            "depot_out_degree")
        # In-degree constraint for depot: Edges from customers to depot 0 = number of vehicles
        model.add_constraint(model.sum(x[j, 0] for j in customers) == par_args.cplex_num_vehicles,
                            "depot_in_degree")
        
        # 6. Add capacity constraints: Use Miller-Tucker-Zemlin (MTZ) variables "u" to track cumulative demand
        # at each node, preventing vehicles from exceeding capacity
        if n > 1:  # No need for capacity constraints if there's only 1 node (the depot)
            # Create continuous variables u[i]: Cumulative demand of the vehicle when arriving at node i
            # Range: 0 (no load) to "capacity" (maximum vehicle load)
            u = model.continuous_var_list(n, lb=0, ub=capacity, name="u")
            
            # Constraint: Cumulative demand at the depot is 0 (vehicles start at the depot with no load)
            model.add_constraint(u[0] == 0, "depot_cumulative_demand")
            
            # Add cumulative demand-related constraints for each customer node
            for i in customers:
                # Constraint 1: Cumulative demand at customer i ≥ demand of i
                # (Ensures the vehicle has enough load to satisfy the customer's demand)
                model.add_constraint(u[i] >= demands[i], f"min_demand_{i}")
                
                # Constraint 2: Core MTZ constraint (prevents subtours and limits cumulative demand)
                # If a vehicle travels from i to j (x[i,j] = 1), then u[i] + demands[j] ≤ u[j]
                # Rewritten as: u[i] - u[j] + capacity * x[i,j] ≤ capacity - demands[j]
                for j in customers:
                    if i != j:  # Skip the case where i equals j
                        model.add_constraint(
                            u[i] - u[j] + capacity * x[i, j] <= capacity - demands[j],
                            f"mtz_capacity_{i}_{j}"
                        )
        
        # 7. Set objective function: Minimize the total travel distance of all routes
        model.minimize(model.sum(
            x[i, j] * distance_matrix[i][j] for i in range(n) for j in range(n)
        ))
        
        # 8. Configure CPLEX solver parameters (read from par_args)
        # Set solving time limit (seconds): Takes effect if a time limit is specified in par_args
        if par_args.cplex_time_limit:
            time_limit = par_args.cplex_time_limit
            model.parameters.timelimit.set(time_limit)
        # Set MIP gap (optimality tolerance): Takes effect if specified (e.g., 0.01 = 1% error allowed)
        if par_args.cplex_gap:
            mip_gap = par_args.cplex_gap
            model.parameters.mip.tolerances.mipgap.set(mip_gap)
        # Execute the solver to find the optimal solution
        model.solve()
        
        # 9. Extract vehicle routes from the solved model
        routes = []  # Store routes for all vehicles (each route = list of customer node indices)
        used = set()  # Track customers already assigned to a route (avoid duplicate assignment)
        
        # Iterate over all possible initial edges from the depot
        # (x[0,j] = 1 means a vehicle starts from the depot to customer j)
        for j in customers:
            # Check if customer j is unassigned and x[0,j] solution value > 0.9
            # (Treat values > 0.9 as 1 to avoid floating-point errors)
            if j not in used and model.solution.get_value(x[0, j]) > 0.9:
                route = []  # Store the route for the current vehicle
                current = j  # Start tracking the route from customer j
                # Loop to track the route until returning to the depot (node 0)
                while current != 0:
                    route.append(current)  # Add the current node to the route
                    used.add(current)  # Mark the current node as used
                    # Find the next node in the route (k where x[current,k] = 1)
                    for k in range(n):
                        if current != k and model.solution.get_value(x[current, k]) > 0.9:
                            current = k  # Update current node to k
                            break
                # Add the complete route of the current vehicle to the routes list
                routes.append(route)
        
        # Get the objective function value (total travel distance of the optimal solution)
        total_distance = model.solution.get_objective_value()
        # Return the extracted routes and total distance
        return routes, total_distance
    
    # Catch exceptions during solving (e.g., model construction errors, solving failures)
    except Exception as e:
        # Return None to indicate solving failure
        return [], 0


    