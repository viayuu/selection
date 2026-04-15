def solve(n: int, m: int, p: list, r: list, b: list) -> dict:
    """
    Solves a multidimensional knapsack problem instance.
    Input kwargs (for one test case):
      - n: int, number of decision variables.
      - m: int, number of constraints.
      - p: list of floats, profit coefficients (length n).
      - r: list of m lists, each of length n, representing the resource consumption per constraint.
      - b: list of floats, right-hand side values for each constraint (length m).
    Evaluation metric:
    The score is computed as:
        score = sum(p[j] * x[j] for j in range(n))
    if and only if all constraints are satisfied—that is, for every constraint i, the total resource consumption
        sum(r[i][j] * x[j] for j in range(n))
    does not exceed b[i].
    If any constraint is violated, the solution receives no score. A higher score is better.
    Returns:
      A dict with key 'x' whose value is a list of n binary decisions (0 or 1).
    """
    import random
    import math

    def is_feasible(x):
        for i in range(m):
            if sum(r[i][j] * x[j] for j in range(n)) > b[i]:
                return False
        return True

    def get_score(x):
        return sum(p[j] * x[j] for j in range(n)) if is_feasible(x) else -1

    # Greedy initialization
    x = [0] * n
    indices = sorted(range(n), key=lambda j: -p[j] / sum(r[i][j] for i in range(m)) if sum(r[i][j] for i in range(m)) != 0 else float('inf'))
    for j in indices:
        temp_x = x.copy()
        temp_x[j] = 1
        if is_feasible(temp_x):
            x = temp_x

    # Local search with simulated annealing
    best_x = x.copy()
    best_score = get_score(x)
    current_x = x.copy()
    current_score = best_score

    temperature = 1.0
    cooling_rate = 0.99
    iterations = 1000

    for _ in range(iterations):
        temp_x = current_x.copy()
        j = random.randint(0, n - 1)
        temp_x[j] = 1 - temp_x[j]
        new_score = get_score(temp_x)

        if new_score > current_score or (new_score >= 0 and random.random() < math.exp((new_score - current_score) / temperature)):
            current_x = temp_x
            current_score = new_score
            if new_score > best_score:
                best_x = temp_x.copy()
                best_score = new_score

        temperature *= cooling_rate

    return {'x': best_x}

