
import numpy as np
import scipy.optimize as opt
def solve(m: int, n: int, warehouses: list, customers: list) -> dict:
    """
    Solves the Capacitated Warehouse Location Problem with Splittable Customer Demand.
    Input kwargs:
  - m (int): Number of potential warehouses
  - n (int): Number of customers
  - warehouses (list of dict): A list of dictionaries, each with keys 'capacity' (float) and 'fixed_cost' (float)
  - customers (list of dict): A list of dictionaries, each with keys 'demand' (float) and 'costs' (list of float) representing the per-unit assignment cost from each warehouse
    Evaluation Metric:
      The objective is to minimize the total cost, computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum of per-unit assignment costs for each unit of demand allocated from warehouses to customers)
      For each customer, the sum of allocations from all warehouses must equal the customer's demand.
      For each warehouse, the total allocated demand across all customers must not exceed its capacity.
      If a solution violates any of these constraints, the solution is considered infeasible and no score is provided.
    Returns:
      A dictionary with the following keys:
         'total_cost': (float) The computed objective value (cost) if the solution is feasible;
                         otherwise, no score is provided.
         'warehouse_open': (list of int) A list of m integers (0 or 1) indicating whether each warehouse is closed or open.
         'assignments': (list of list of float) A 2D list (n x m) where each entry represents the amount of customer i's demand supplied by warehouse j.
    """
    warehouse_open = [0] * m
    assignments = [[0.0 for _ in range(m)] for _ in range(n)]
    remaining_demand = [customer['demand'] for customer in customers]
    remaining_capacity = [warehouse['capacity'] for warehouse in warehouses]

    # Calculate cost-effectiveness (fixed_cost / capacity + avg assignment cost)
    cost_effectiveness = []
    for j in range(m):
        avg_cost = sum(customers[i]['costs'][j] for i in range(n)) / n
        cost_effectiveness.append((warehouses[j]['fixed_cost'] / warehouses[j]['capacity'] + avg_cost, j))

    # Sort warehouses by cost-effectiveness
    cost_effectiveness.sort()

    # Open warehouses greedily and assign demand
    for _, j in cost_effectiveness:
        if warehouse_open[j]:
            continue

        # Try to open this warehouse
        warehouse_open[j] = 1
        total_assigned = 0.0

        # Assign demand to this warehouse in order of cheapest cost
        for i in range(n):
            if remaining_demand[i] <= 0:
                continue

            if customers[i]['costs'][j] < float('inf'):
                assign = min(remaining_demand[i], remaining_capacity[j])
                assignments[i][j] = assign
                remaining_demand[i] -= assign
                remaining_capacity[j] -= assign
                total_assigned += assign

        # If no demand assigned, close warehouse
        if total_assigned == 0:
            warehouse_open[j] = 0

    # Local search improvement
    improved = True
    while improved:
        improved = False

        # Try to close warehouses and redistribute demand
        for j in range(m):
            if not warehouse_open[j]:
                continue

            # Check if closing this warehouse is beneficial
            temp_open = warehouse_open.copy()
            temp_open[j] = 0
            temp_assignments = [row.copy() for row in assignments]
            temp_remaining_demand = [0.0] * n
            temp_remaining_capacity = remaining_capacity.copy()
            temp_remaining_capacity[j] = warehouses[j]['capacity']

            # Redistribute demand from closed warehouse
            feasible = True
            for i in range(n):
                if temp_assignments[i][j] > 0:
                    temp_remaining_demand[i] = temp_assignments[i][j]
                    temp_assignments[i][j] = 0

            # Try to redistribute to other open warehouses
            for i in range(n):
                if temp_remaining_demand[i] <= 0:
                    continue

                # Find cheapest open warehouse with capacity
                best_cost = float('inf')
                best_j = -1
                for k in range(m):
                    if temp_open[k] and customers[i]['costs'][k] < best_cost and temp_remaining_capacity[k] >= temp_remaining_demand[i]:
                        best_cost = customers[i]['costs'][k]
                        best_j = k

                if best_j == -1:
                    feasible = False
                    break

                temp_assignments[i][best_j] += temp_remaining_demand[i]
                temp_remaining_capacity[best_j] -= temp_remaining_demand[i]
                temp_remaining_demand[i] = 0

            if feasible:
                # Calculate new cost
                new_cost = sum(warehouses[k]['fixed_cost'] for k in range(m) if temp_open[k])
                new_cost += sum(sum(temp_assignments[i][k] * customers[i]['costs'][k] for k in range(m)) for i in range(n))

                current_cost = sum(warehouses[k]['fixed_cost'] for k in range(m) if warehouse_open[k])
                current_cost += sum(sum(assignments[i][k] * customers[i]['costs'][k] for k in range(m)) for i in range(n))

                if new_cost < current_cost:
                    warehouse_open = temp_open
                    assignments = temp_assignments
                    remaining_capacity = temp_remaining_capacity
                    improved = True

    # Check feasibility
    feasible = True
    for i in range(n):
        if abs(sum(assignments[i][j] for j in range(m)) - customers[i]['demand']) > 1e-6:
            feasible = False
            break

    for j in range(m):
        if warehouse_open[j] and sum(assignments[i][j] for i in range(n)) > warehouses[j]['capacity'] + 1e-6:
            feasible = False
            break

    if not feasible:
        return {
            'warehouse_open': warehouse_open,
            'assignments': assignments
        }

    total_cost = sum(warehouses[j]['fixed_cost'] for j in range(m) if warehouse_open[j])
    total_cost += sum(sum(assignments[i][j] * customers[i]['costs'][j] for j in range(m)) for i in range(n))

    return {
        'total_cost': total_cost,
        'warehouse_open': warehouse_open,
        'assignments': assignments
    }

