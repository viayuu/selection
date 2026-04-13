import random


def solve(n: int, m: int, q: int, A_leq: list, b_leq: list, A_geq: list, b_geq: list, cost_vector: list,
          cost_type: str) -> dict:
    """
    Solve a given MDMKP test instance.
    Input (via kwargs):
      - n: int
          Number of decision variables.
      - m: int
          Number of <= constraints.
      - q: int
          Number of active >= constraints (subset of the full set).
      - A_leq: list of lists of int
          Coefficient matrix for <= constraints (dimensions: m x n).
      - b_leq: list of int
          Right-hand side for <= constraints (length m).
      - A_geq: list of lists of int
          Coefficient matrix for >= constraints (dimensions: q x n).
      - b_geq: list of int
          Right-hand side for >= constraints (length q).
      - cost_vector: list of int
          Objective function coefficients (length n).
      - cost_type: str
          Type of cost coefficients ("positive" or "mixed").
    Output:
      A dictionary with the following keys:
        - 'optimal_value': int/float
             The optimal objective function value (if found).
        - 'x': list of int
             Binary vector (0 or 1) representing the decision variable assignment.
    TODO: Implement the actual solution algorithm for the MDMKP instance.
    """
    POP_SIZE = 10
    MAX_GENS = 5
    MUTATION_RATE = 0.05
    CROSSOVER_RATE = 0.7

    def fitness(x):
        total_cost = sum(cost_vector[j] * x[j] for j in range(n))
        penalty = 0

        # Penalize <= constraint violations
        for i in range(m):
            constraint_sum = sum(A_leq[i][j] * x[j] for j in range(n))
            if constraint_sum > b_leq[i]:
                penalty += (constraint_sum - b_leq[i]) * 1000

        # Penalize >= constraint violations
        for i in range(q):
            constraint_sum = sum(A_geq[i][j] * x[j] for j in range(n))
            if constraint_sum < b_geq[i]:
                penalty += (b_geq[i] - constraint_sum) * 1000

        return total_cost - penalty

    def repair(x):
        # Repair <= constraints
        for i in range(m):
            while True:
                constraint_sum = sum(A_leq[i][j] * x[j] for j in range(n))
                if constraint_sum <= b_leq[i]:
                    break
                # Randomly turn off a variable contributing to violation
                candidates = [j for j in range(n) if x[j] == 1 and A_leq[i][j] > 0]
                if not candidates:
                    break
                x[random.choice(candidates)] = 0

        # Repair >= constraints
        for i in range(q):
            while True:
                constraint_sum = sum(A_geq[i][j] * x[j] for j in range(n))
                if constraint_sum >= b_geq[i]:
                    break
                # Randomly turn on a variable that helps satisfy constraint
                candidates = [j for j in range(n) if x[j] == 0 and A_geq[i][j] > 0]
                if not candidates:
                    break
                x[random.choice(candidates)] = 1

        return x

    def crossover(parent1, parent2):
        if random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, n - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(x):
        for j in range(n):
            if random.random() < MUTATION_RATE:
                x[j] = 1 - x[j]
        return x

    # Initialize population
    population = []
    for _ in range(POP_SIZE):
        x = [random.randint(0, 1) for _ in range(n)]
        x = repair(x)
        population.append(x)

    best_solution = None
    best_fitness = -float('inf')

    for _ in range(MAX_GENS):
        # Evaluate fitness
        fitnesses = [fitness(x) for x in population]

        # Update best solution
        current_best = max(fitnesses)
        if current_best > best_fitness:
            best_idx = fitnesses.index(current_best)
            best_solution = population[best_idx]
            best_fitness = current_best

        # Selection (tournament)
        new_population = []
        for _ in range(POP_SIZE // 2):
            # Select parents
            parents = random.sample(range(POP_SIZE), 4)
            parent1 = population[parents[0]] if fitnesses[parents[0]] > fitnesses[parents[1]] else population[
                parents[1]]
            parent2 = population[parents[2]] if fitnesses[parents[2]] > fitnesses[parents[3]] else population[
                parents[3]]

            # Crossover and mutate
            child1, child2 = crossover(parent1, parent2)
            child1 = repair(mutate(child1))
            child2 = repair(mutate(child2))

            new_population.extend([child1, child2])

        population = new_population

    # Final check
    final_fitnesses = [fitness(x) for x in population]
    best_idx = final_fitnesses.index(max(final_fitnesses))
    best_solution = population[best_idx]
    best_fitness = fitness(best_solution)

    return {
        'optimal_value': best_fitness if best_fitness > -float('inf') else 0,
        'x': best_solution if best_solution else [0] * n
    }

