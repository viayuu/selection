import math
from collections import defaultdict
import random


def solve(best_known: float, n: int, p: int, Q: float, customers: list) -> dict:
    """
    Solve the Capacitated P-Median Problem.
    This function receives the data for one problem instance via keyword arguments:
      - best_known (float): Best known solution value for reference.
      - n (int): Number of customers.
      - p (int): Number of medians to choose.
      - Q (float): Capacity limit for each median.
      - customers (list of tuples): Each tuple is (customer_id, x, y, demand).
    The goal is to select p medians (from the customers) and assign every customer to one
    of these medians so that the total cost is minimized. The cost for a customer is the
    Euclidean distance (rounded down to the nearest integer) to its assigned median, and the
    total demand assigned to each median must not exceed Q.
    Evaluation Metric:
      The solution is evaluated by computing the ratio:
          score = best_known / computed_total_cost,
      where computed_total_cost is the sum over all customers of the (floored) Euclidean distance
      to its assigned median.
    Note: This is a placeholder function. Replace the placeholder with an actual algorithm.
    Returns:
      A dictionary with the following keys:
        - 'objective': (numeric) the total cost (objective value) computed by the algorithm.
        - 'medians': (list of int) exactly p customer IDs chosen as medians.
        - 'assignments': (list of int) a list of n integers, where the i-th integer is the customer
                         ID (from the chosen medians) assigned to customer i.
    """
    customer_data = {cid: (x, y, demand) for cid, x, y, demand in customers}
    cids = [cid for cid, _, _, _ in customers]

    # Precompute all pairwise distances (floored)
    distances = {}
    for i in cids:
        for j in cids:
            if i == j:
                distances[(i, j)] = 0
            else:
                x1, y1, _ = customer_data[i]
                x2, y2, _ = customer_data[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distances[(i, j)] = math.floor(dist)

    # Greedy initial solution: select p medians with highest demand
    sorted_customers = sorted(customers, key=lambda x: -x[3])
    medians = [cid for cid, _, _, _ in sorted_customers[:p]]

    # Assign customers to nearest medians respecting capacity
    assignments = []
    total_cost = 0
    median_demands = defaultdict(float)

    for cid in cids:
        nearest_median = None
        min_dist = float('inf')
        for m in medians:
            if median_demands[m] + customer_data[cid][2] <= Q:
                dist = distances[(cid, m)]
                if dist < min_dist:
                    min_dist = dist
                    nearest_median = m
        if nearest_median is None:
            # Feasibility repair: assign to least violating median
            min_violation = float('inf')
            for m in medians:
                violation = median_demands[m] + customer_data[cid][2] - Q
                if violation < min_violation:
                    min_violation = violation
                    nearest_median = m
        assignments.append(nearest_median)
        median_demands[nearest_median] += customer_data[cid][2]
        total_cost += distances[(cid, nearest_median)]

    # Local search: try to swap medians and improve assignments
    improved = True
    while improved:
        improved = False
        for i in range(p):
            current_median = medians[i]
            for candidate in cids:
                if candidate in medians:
                    continue
                # Try swapping current_median with candidate
                new_medians = medians.copy()
                new_medians[i] = candidate

                # Reassign customers
                new_assignments = []
                new_total_cost = 0
                new_median_demands = defaultdict(float)
                feasible = True

                for cid in cids:
                    nearest_median = None
                    min_dist = float('inf')
                    for m in new_medians:
                        if new_median_demands[m] + customer_data[cid][2] <= Q:
                            dist = distances[(cid, m)]
                            if dist < min_dist:
                                min_dist = dist
                                nearest_median = m
                    if nearest_median is None:
                        feasible = False
                        break
                    new_assignments.append(nearest_median)
                    new_median_demands[nearest_median] += customer_data[cid][2]
                    new_total_cost += distances[(cid, nearest_median)]

                if feasible and new_total_cost < total_cost:
                    medians = new_medians
                    assignments = new_assignments
                    total_cost = new_total_cost
                    improved = True
                    break
            if improved:
                break

    # Check feasibility
    median_demands = defaultdict(float)
    feasible = True
    for cid, m in zip(cids, assignments):
        median_demands[m] += customer_data[cid][2]
        if median_demands[m] > Q + 1e-6:
            feasible = False
            break

    if not feasible:
        return {'objective': 0, 'medians': [], 'assignments': []}

    return {
        'objective': total_cost,
        'medians': medians,
        'assignments': assignments
    }





