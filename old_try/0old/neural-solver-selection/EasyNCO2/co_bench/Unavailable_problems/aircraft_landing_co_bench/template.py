def solve(num_planes: int, num_runways: int, freeze_time: float, planes: list[dict],
          separation: list[list[int]]) -> dict:
    """
    Problem:
        Given an instance of the Aircraft Landing Scheduling Problem, schedule the landing time for each plane and assign a runway so that:
          - Each landing time is within its allowed time window.
          - Each plane is assigned to one runway (from the available runways).
          - For any two planes assigned to the same runway, if plane i lands at or before plane j, then the landing times must be separated by at least
            the specified separation time (provided in the input data).
          - The overall penalty is minimized. For each plane, if its landing time is earlier than its target time, a penalty
            is incurred proportional to the earliness; if later than its target time, a penalty proportional to the lateness is incurred.
          - If any constraint is violated, the solution receives no score.
    Input kwargs:
        num_planes  : (int) Number of planes.
        num_runways : (int) Number of runways.
        freeze_time : (float) Freeze time (unused in scheduling decisions).
        planes      : (list of dict) Each dictionary contains:
                        - "appearance"    : float, time the plane appears.
                        - "earliest"      : float, earliest landing time.
                        - "target"        : float, target landing time.
                        - "latest"        : float, latest landing time.
                        - "penalty_early" : float, penalty per unit time landing early.
                        - "penalty_late"  : float, penalty per unit time landing late.
        separation  : (list of lists) separation[i][j] is the required gap after plane i lands before plane j can land
                      when they are assigned to the same runway.
    Returns:
        A dictionary named "schedule" mapping each plane id (1-indexed) to a dictionary with its scheduled landing time
        and assigned runway, e.g., { plane_id: {"landing_time": float, "runway": int}, ... }.
    """
    import random
    from itertools import permutations

    # Genetic Algorithm parameters
    POP_SIZE = 10
    GENERATIONS = 5
    MUTATION_RATE = 0.1

    def initialize_population():
        population = []
        for _ in range(POP_SIZE):
            runway_assign = [random.randint(1, num_runways) for _ in range(num_planes)]
            population.append(runway_assign)
        return population

    def evaluate(assignment):
        runway_planes = {r: [] for r in range(1, num_runways + 1)}
        for plane_id, runway in enumerate(assignment, 1):
            runway_planes[runway].append(plane_id)

        total_penalty = 0
        schedule = {}

        for runway, plane_ids in runway_planes.items():
            # Try all permutations to find minimal penalty
            best_perm_penalty = float('inf')
            best_perm_schedule = {}

            # sample 10 combinations of permutations to avoid combinatorial explosion
            per_lists = random.sample(list(permutations(plane_ids)), min(10, len(plane_ids)))

            for perm in per_lists:
                current_time = 0
                perm_penalty = 0
                perm_schedule = {}
                valid = True

                for i, plane_id in enumerate(perm):
                    plane = planes[plane_id - 1]
                    earliest = plane["earliest"]
                    target = plane["target"]
                    latest = plane["latest"]
                    penalty_early = plane["penalty_early"]
                    penalty_late = plane["penalty_late"]

                    if i == 0:
                        start_time = max(earliest, plane["appearance"])
                    else:
                        prev_plane_id = perm[i - 1]
                        sep = separation[prev_plane_id - 1][plane_id - 1]
                        start_time = max(earliest, perm_schedule[prev_plane_id]["landing_time"] + sep,
                                         plane["appearance"])

                    if start_time > latest:
                        valid = False
                        break

                    # Choose time to minimize penalty
                    if start_time <= target:
                        landing_time = min(target, latest)
                        penalty = (target - landing_time) * penalty_early
                    else:
                        landing_time = start_time
                        penalty = (landing_time - target) * penalty_late

                    perm_penalty += penalty
                    perm_schedule[plane_id] = {"landing_time": landing_time, "runway": runway}

                if valid and perm_penalty < best_perm_penalty:
                    best_perm_penalty = perm_penalty
                    best_perm_schedule = perm_schedule

            if best_perm_penalty == float('inf'):
                return float('inf'), None
            total_penalty += best_perm_penalty
            schedule.update(best_perm_schedule)

        return total_penalty, schedule

    def crossover(parent1, parent2):
        point = random.randint(1, num_planes - 1)
        child = parent1[:point] + parent2[point:]
        return child

    def mutate(individual):
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, num_planes - 1)
            individual[idx] = random.randint(1, num_runways)
        return individual

    # Main GA loop
    population = initialize_population()
    best_schedule = None
    best_penalty = float('inf')

    for _ in range(GENERATIONS):
        evaluated = [evaluate(ind) for ind in population]
        valid = [penalty for penalty, _ in evaluated if penalty != float('inf')]

        if not valid:
            continue

        # Selection: tournament
        new_population = []
        for _ in range(POP_SIZE):
            candidates = random.sample(range(POP_SIZE), 3)
            winner = min(candidates, key=lambda x: evaluated[x][0] if evaluated[x][0] != float('inf') else float('inf'))
            new_population.append(population[winner])
        population = new_population

        # Crossover and mutation
        next_population = []
        for i in range(0, POP_SIZE, 2):
            parent1, parent2 = population[i], population[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            next_population.extend([mutate(child1), mutate(child2)])
        population = next_population

        # Update best
        for penalty, schedule in evaluated:
            if penalty < best_penalty:
                best_penalty = penalty
                best_schedule = schedule

    best_schedule = {'schedule': best_schedule}

    return best_schedule if best_schedule else {i + 1: {"landing_time": planes[i]["target"], "runway": 1} for i in
                                                range(num_planes)}

