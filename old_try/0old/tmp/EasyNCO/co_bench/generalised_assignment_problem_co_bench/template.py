import numpy as np
import scipy.optimize as opt
def solve(m: int, n: int, cost_matrix: list, consumption_matrix: list, capacities: list, problem_type: str='max') -> dict:
    """
    Solve the Generalised Assignment Problem (GAP) for a single case.
    Input arguments (passed as keyword arguments):
      - m: (int) Number of agents.
      - n: (int) Number of jobs.
      - cost_matrix: (list of list of float) A matrix of size m×n where cost_matrix[i][j]
                     represents the cost of assigning job j to agent i.
      - consumption_matrix: (list of list of float) A matrix of size m×n where consumption_matrix[i][j]
                     represents the resource consumed when job j is assigned to agent i.
      - capacities: (list of float) A list of length m containing the resource capacity for each agent.
      - problem_type: (str, optional) Indicates whether the problem is a 'max' or 'min' problem.
                     Defaults to 'max'.
    Returns:
      A dictionary with the key 'assignments' whose value is a list of n integers.
      Each integer is an agent number (using 1-indexing) that is assigned to the corresponding job.
    """
    assignments = [0] * n
    remaining_capacities = capacities.copy()

    # Greedy initial assignment
    for job in range(n):
        best_agent = -1
        best_value = -float('inf') if problem_type == 'max' else float('inf')
        for agent in range(m):
            if remaining_capacities[agent] >= consumption_matrix[agent][job]:
                current_value = cost_matrix[agent][job] / (consumption_matrix[agent][job] + 1e-10)
                if (problem_type == 'max' and current_value > best_value) or (problem_type == 'min' and current_value < best_value):
                    best_value = current_value
                    best_agent = agent
        if best_agent != -1:
            assignments[job] = best_agent + 1  # 1-indexing
            remaining_capacities[best_agent] -= consumption_matrix[best_agent][job]

    # Local search to improve the solution
    improved = True
    while improved:
        improved = False
        for job1 in range(n):
            for job2 in range(job1 + 1, n):
                agent1 = assignments[job1] - 1
                agent2 = assignments[job2] - 1
                if agent1 == agent2:
                    continue
                # Check if swapping improves the solution
                original_cost = cost_matrix[agent1][job1] + cost_matrix[agent2][job2]
                new_cost = cost_matrix[agent1][job2] + cost_matrix[agent2][job1]
                if (problem_type == 'max' and new_cost > original_cost) or (problem_type == 'min' and new_cost < original_cost):
                    # Check capacity constraints
                    if (remaining_capacities[agent1] + consumption_matrix[agent1][job1] >= consumption_matrix[agent1][job2] and
                        remaining_capacities[agent2] + consumption_matrix[agent2][job2] >= consumption_matrix[agent2][job1]):
                        # Perform the swap
                        remaining_capacities[agent1] += consumption_matrix[agent1][job1] - consumption_matrix[agent1][job2]
                        remaining_capacities[agent2] += consumption_matrix[agent2][job2] - consumption_matrix[agent2][job1]
                        assignments[job1], assignments[job2] = assignments[job2], assignments[job1]
                        improved = True
    return {'assignments': assignments}


