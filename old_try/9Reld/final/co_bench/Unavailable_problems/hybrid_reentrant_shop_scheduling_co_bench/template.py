import random
from typing import List


def solve(n_jobs: int, n_machines: int, init_time: int, setup_times: list, processing_times: list, **kwargs) -> dict:
    """
    Input:
      - n_jobs: Integer; the number of jobs.
      - n_machines: Integer; the number of primary machines.
      - init_time: Integer; the initialization time for every job on a primary machine.
      - setup_times: List of integers; the setup times for each job on the remote server.
      - processing_times: List of integers; the processing times for each job in the main processing stage.
    Output:
      A dictionary with the following keys:
        - 'permutation': A list of integers of length n_jobs. This list represents the order in which the jobs are processed on the remote server.
        - 'batch_assignment': A list of integers of length n_jobs. Each element indicates the primary machine to which the corresponding job (or batch) is assigned.
    """
    population_size = kwargs.get('population_size', 10)
    generations = kwargs.get('generations', 5)
    mutation_rate = kwargs.get('mutation_rate', 0.1)
    crossover_rate = kwargs.get('crossover_rate', 0.8)

    # Initialize population
    population = []
    for _ in range(population_size):
        permutation = random.sample(range(n_jobs), n_jobs)
        batch_assignment = [random.randint(0, n_machines - 1) for _ in range(n_jobs)]
        population.append((permutation, batch_assignment))

    # Evaluate fitness (makespan)
    def evaluate(permutation: List[int], batch_assignment: List[int]) -> int:
        init_end = [0] * n_machines
        server_end = 0
        main_end = [0] * n_machines

        for job in range(n_jobs):
            machine = batch_assignment[job]
            init_start = init_end[machine]
            init_end[machine] = init_start + init_time

        for job in permutation:
            machine = batch_assignment[job]
            server_start = max(init_end[machine], server_end)
            server_end = server_start + setup_times[job]

            main_start = max(server_end, main_end[machine])
            main_end[machine] = main_start + processing_times[job]

        return max(main_end)

    # Genetic Algorithm loop
    for _ in range(generations):
        # Selection (tournament)
        new_population = []
        for _ in range(population_size):
            a, b = random.sample(population, 2)
            a_fit = evaluate(a[0], a[1])
            b_fit = evaluate(b[0], b[1])
            new_population.append(a if a_fit < b_fit else b)

        # Crossover (OX for permutation, uniform for batch)
        offspring = []
        for i in range(0, population_size, 2):
            if i + 1 >= population_size:
                break
            parent1, parent2 = new_population[i], new_population[i + 1]
            if random.random() < crossover_rate:
                # Permutation crossover (OX)
                start, end = sorted(random.sample(range(n_jobs), 2))
                child1_p = parent1[0][start:end]
                child2_p = parent2[0][start:end]
                remaining1 = [x for x in parent2[0] if x not in child1_p]
                remaining2 = [x for x in parent1[0] if x not in child2_p]
                child1_p = remaining1[:start] + child1_p + remaining1[start:]
                child2_p = remaining2[:start] + child2_p + remaining2[start:]

                # Batch assignment crossover (uniform)
                child1_b = [parent1[1][j] if random.random() < 0.5 else parent2[1][j] for j in range(n_jobs)]
                child2_b = [parent2[1][j] if random.random() < 0.5 else parent1[1][j] for j in range(n_jobs)]

                offspring.append((child1_p, child1_b))
                offspring.append((child2_p, child2_b))
            else:
                offspring.append(parent1)
                offspring.append(parent2)

        # Mutation (swap for permutation, random reset for batch)
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                # Permutation mutation (swap)
                a, b = random.sample(range(n_jobs), 2)
                offspring[i][0][a], offspring[i][0][b] = offspring[i][0][b], offspring[i][0][a]
            if random.random() < mutation_rate:
                # Batch assignment mutation (random reset)
                j = random.randint(0, n_jobs - 1)
                offspring[i][1][j] = random.randint(0, n_machines - 1)

        population = offspring

    # Select best solution
    best_solution = min(population, key=lambda x: evaluate(x[0], x[1]))
    return {
        'permutation': best_solution[0],
        'batch_assignment': best_solution[1]
    }

