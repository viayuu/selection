import numpy as np
import scipy.optimize as opt
def solve(data: list[list[int]]) -> dict:
    """
    Partition individuals into 8 groups so that for every binary attribute the count of 1's is as evenly
    distributed across the groups as possible.
    Input kwargs:
      - data (list of list of int): A matrix where each inner list represents the binary attributes (0 or 1)
        of one individual.
    Evaluation Metric:
      For each attribute, calculate the number of 1’s in each group,
      then compute the absolute difference between each group’s count and the mean count for that attribute.
      Average these differences over all groups to obtain the attribute’s imbalance.
      The final score is the sum of these attribute imbalances across all attributes.
      A lower score indicates a more balanced partitioning.
    Returns:
      dict: A dictionary with one key 'assignment' whose value is a list of positive integers (one per individual)
            indicating the group assignment (using 1-based indexing). For example:
            { "assignment": [1, 1, 1, ...] }
    """
    import numpy as np
    import random
    from collections import defaultdict

    num_individuals = len(data)
    num_attributes = len(data[0]) if num_individuals > 0 else 0
    num_groups = 8

    if num_individuals == 0:
        return {"assignment": []}

    # Convert data to numpy array for efficiency
    data_np = np.array(data, dtype=int)

    # Calculate target means for each attribute
    total_ones = data_np.sum(axis=0)
    target_means = total_ones / num_groups

    # Initialize assignments randomly
    assignments = np.random.randint(1, num_groups + 1, size=num_individuals)

    def compute_imbalance(assignments):
        group_counts = np.zeros((num_groups, num_attributes))
        for i in range(num_individuals):
            group = assignments[i] - 1
            group_counts[group] += data_np[i]
        abs_diffs = np.abs(group_counts - target_means)
        avg_diffs = np.mean(abs_diffs, axis=0)
        return np.sum(avg_diffs)

    current_imbalance = compute_imbalance(assignments)
    best_imbalance = current_imbalance
    best_assignments = assignments.copy()

    # Simulated annealing parameters
    temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    max_iter = 10000

    for _ in range(max_iter):
        if temp < min_temp:
            break

        # Randomly select an individual and a new group
        idx = random.randint(0, num_individuals - 1)
        old_group = assignments[idx]
        new_group = random.choice([g for g in range(1, num_groups + 1) if g != old_group])

        # Make the change and compute new imbalance
        assignments[idx] = new_group
        new_imbalance = compute_imbalance(assignments)

        # Decide whether to keep the change
        if new_imbalance < current_imbalance or random.random() < np.exp((current_imbalance - new_imbalance) / temp):
            current_imbalance = new_imbalance
            if new_imbalance < best_imbalance:
                best_imbalance = new_imbalance
                best_assignments = assignments.copy()
        else:
            assignments[idx] = old_group

        temp *= cooling_rate

    return {"assignment": best_assignments.tolist()}


