import numpy as np
import scipy.optimize as opt
def solve(m: int, n: int, warehouses: list, customers: list) -> dict:
    """
    Solves the Uncapacitated Warehouse Location Problem.
    Input kwargs:
      - m: Number of potential warehouses (int)
      - n: Number of customers (int)
      - warehouses: A list of dictionaries, each with keys:
            'fixed_cost': Fixed cost for opening the warehouse.
      - customers: A list of dictionaries, each with keys:
            'costs': A list of floats representing the cost of assigning the entire customer to each warehouse.
    Evaluation Metric:
      The objective is to minimize the total cost, computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum of assignment costs for each customer assigned to a warehouse)
      Each customer must be assigned entirely to exactly one open warehouse.
      If a solution violates this constraint (i.e., a customer is unassigned or is assigned to more than one warehouse), then the solution is considered infeasible and no score is provided.
    Returns:
      A dictionary with the following keys:
         'total_cost': (float) The computed objective value (cost) if the solution is feasible; otherwise, no score is provided.
         'warehouse_open': (list of int) A list of m integers (0 or 1) indicating whether each warehouse is closed or open.
         'assignments': (list of list of int) A 2D list (n x m) where each entry is 1 if customer i is assigned to warehouse j, and 0 otherwise.
    """
    import random
    import math

    # Greedy initialization
    def greedy_initial_solution():
        open_warehouses = [0] * m
        assignments = [[0 for _ in range(m)] for _ in range(n)]
        total_cost = 0.0

        # Open warehouses with lowest fixed_cost / potential_assignment_ratio
        warehouse_costs = []
        for j in range(m):
            min_assignment_cost = sum(customers[i]['costs'][j] for i in range(n))
            ratio = warehouses[j]['fixed_cost'] / (min_assignment_cost + 1e-6)
            warehouse_costs.append((ratio, j))

        # Sort warehouses by ratio and open top k
        warehouse_costs.sort()
        k = max(1, m // 2)  # Arbitrary initial choice
        selected_warehouses = [j for _, j in warehouse_costs[:k]]
        for j in selected_warehouses:
            open_warehouses[j] = 1
            total_cost += warehouses[j]['fixed_cost']

        # Assign customers to the cheapest open warehouse
        for i in range(n):
            min_cost = float('inf')
            best_j = -1
            for j in selected_warehouses:
                if customers[i]['costs'][j] < min_cost:
                    min_cost = customers[i]['costs'][j]
                    best_j = j
            assignments[i][best_j] = 1
            total_cost += min_cost

        return open_warehouses, assignments, total_cost

    # Local search with swap moves
    def local_search(open_warehouses, assignments, total_cost):
        improved = True
        while improved:
            improved = False
            # Try closing a warehouse and reassigning customers
            for j in [j for j in range(m) if open_warehouses[j]]:
                new_open = open_warehouses.copy()
                new_open[j] = 0
                new_assignments = [[0 for _ in range(m)] for _ in range(n)]
                new_cost = 0.0
                # Reassign customers to next best open warehouse
                feasible = True
                for i in range(n):
                    min_cost = float('inf')
                    best_j = -1
                    for k in range(m):
                        if new_open[k] and customers[i]['costs'][k] < min_cost:
                            min_cost = customers[i]['costs'][k]
                            best_j = k
                    if best_j == -1:
                        feasible = False
                        break
                    new_assignments[i][best_j] = 1
                    new_cost += min_cost
                if feasible:
                    new_cost += sum(warehouses[k]['fixed_cost'] for k in range(m) if new_open[k])
                    if new_cost < total_cost:
                        open_warehouses, assignments, total_cost = new_open, new_assignments, new_cost
                        improved = True
                        break
            if improved:
                continue
            # Try opening a closed warehouse and reassigning customers
            for j in [j for j in range(m) if not open_warehouses[j]]:
                new_open = open_warehouses.copy()
                new_open[j] = 1
                new_assignments = [[0 for _ in range(m)] for _ in range(n)]
                new_cost = 0.0
                # Reassign customers to best open warehouse (including new one)
                for i in range(n):
                    min_cost = float('inf')
                    best_j = -1
                    for k in range(m):
                        if new_open[k] and customers[i]['costs'][k] < min_cost:
                            min_cost = customers[i]['costs'][k]
                            best_j = k
                    new_assignments[i][best_j] = 1
                    new_cost += min_cost
                new_cost += sum(warehouses[k]['fixed_cost'] for k in range(m) if new_open[k])
                if new_cost < total_cost:
                    open_warehouses, assignments, total_cost = new_open, new_assignments, new_cost
                    improved = True
                    break
        return open_warehouses, assignments, total_cost

    # Simulated annealing
    def simulated_annealing(initial_open, initial_assignments, initial_cost):
        current_open = initial_open.copy()
        current_assignments = [row.copy() for row in initial_assignments]
        current_cost = initial_cost
        best_open = current_open.copy()
        best_assignments = [row.copy() for row in current_assignments]
        best_cost = current_cost
        temperature = 1000.0
        cooling_rate = 0.99
        iterations = 1000

        for _ in range(iterations):
            # Random neighbor: flip a warehouse's status
            j = random.randint(0, m - 1)
            new_open = current_open.copy()
            new_open[j] = 1 - new_open[j]
            new_assignments = [[0 for _ in range(m)] for _ in range(n)]
            new_cost = 0.0
            feasible = True
            for i in range(n):
                min_cost = float('inf')
                best_j = -1
                for k in range(m):
                    if new_open[k] and customers[i]['costs'][k] < min_cost:
                        min_cost = customers[i]['costs'][k]
                        best_j = k
                if best_j == -1:
                    feasible = False
                    break
                new_assignments[i][best_j] = 1
                new_cost += min_cost
            if not feasible:
                continue
            new_cost += sum(warehouses[k]['fixed_cost'] for k in range(m) if new_open[k])
            # Accept or reject
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
                current_open, current_assignments, current_cost = new_open, new_assignments, new_cost
                if current_cost < best_cost:
                    best_open, best_assignments, best_cost = current_open, current_assignments, current_cost
            temperature *= cooling_rate
        return best_open, best_assignments, best_cost

    # Main execution
    open_warehouses, assignments, total_cost = greedy_initial_solution()
    open_warehouses, assignments, total_cost = local_search(open_warehouses, assignments, total_cost)
    open_warehouses, assignments, total_cost = simulated_annealing(open_warehouses, assignments, total_cost)

    # Verify feasibility
    for i in range(n):
        if sum(assignments[i]) != 1:
            return {'total_cost': None, 'warehouse_open': open_warehouses, 'assignments': assignments}

    return {'total_cost': total_cost, 'warehouse_open': open_warehouses, 'assignments': assignments}

