import numpy as np
from scipy.optimize import linear_sum_assignment
def solve(num_items: int, cost_matrix: np.ndarray) -> dict:
    """
    Solves an instance of the Assignment Problem.
    Given n items and an n×n cost matrix (where cost_matrix[i][j] is the cost of assigning
    item (i+1) to agent (j+1)), the goal is to determine a permutation (a one-to-one assignment
    between items and agents) that minimizes the total cost. The returned solution is a
    dictionary with:
      - "total_cost": The sum of the costs of the chosen assignments.
      - "assignment": A list of n tuples (i, j), where i is the item number (1-indexed)
                      and j is the assigned agent number (1-indexed).
    Input kwargs:
      - n: int, the number of items/agents.
      - cost_matrix: numpy.ndarray, a 2D array with shape (n, n) containing the costs.
    Returns:
      A dictionary with keys "total_cost" and "assignment" representing the optimal solution.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    assignment = [(i + 1, j + 1) for i, j in zip(row_ind, col_ind)]
    return {"total_cost": total_cost, "assignment": assignment}
