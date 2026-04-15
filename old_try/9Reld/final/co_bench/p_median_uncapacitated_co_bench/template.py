def solve(n: int, m: int, p: int, dist: list) -> dict:
    """
    Solves the uncapacitated p-median problem on a given graph.
    Input kwargs:
        - n: int, number of vertices.
        - m: int, number of edges.
        - p: int, number of medians to choose.
        - dist: list of lists, the complete cost matrix (n x n) computed via Floyd’s algorithm.
    Evaluation metric:
        The total assignment cost, defined as the sum (over all vertices) of the shortest distance
        from that vertex to its closest chosen median.
    Returns:
        A dictionary with a single key:
            - 'medians': a list of exactly p distinct integers (each between 1 and n) representing
              the indices of the chosen medians.
    Note: This is a placeholder. The actual solution logic should populate the 'medians' list.
    """
    import random
    import math

    def compute_cost(medians):
        total = 0
        for v in range(n):
            total += min(dist[v][s] for s in medians)
        return total

    # Greedy initialization: iteratively add the best median
    medians = set()
    for _ in range(p):
        best_median = None
        best_cost = float('inf')
        for candidate in range(n):
            if candidate not in medians:
                temp_medians = medians | {candidate}
                cost = compute_cost(temp_medians)
                if cost < best_cost:
                    best_cost = cost
                    best_median = candidate
        medians.add(best_median)

    # Local search with simulated annealing
    current_cost = compute_cost(medians)
    temperature = 1000
    cooling_rate = 0.995
    min_temperature = 1e-3

    while temperature > min_temperature:
        # Random swap: replace one median with a non-median
        old_median = random.choice(list(medians))
        new_median = random.choice([v for v in range(n) if v not in medians])
        new_medians = (medians - {old_median}) | {new_median}
        new_cost = compute_cost(new_medians)

        # Accept if better or probabilistically if worse
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            medians = new_medians
            current_cost = new_cost

        temperature *= cooling_rate

    return {'medians': sorted(medians)}

