import math
import random


def solve(nodes: list) -> dict:
    """
    Solve a TSP instance.
    Args:
        - nodes (list): List of (x, y) coordinates representing cities in the TSP problem
                     Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
    Returns:
        dict: Solution information with:
            - 'tour' (list): List of node indices representing the solution path
                            Format: [0, 3, 1, ...] where numbers are indices into the nodes list
    """

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def total_distance(tour):
        return sum(distance(nodes[tour[i]], nodes[tour[i + 1]]) for i in range(len(tour) - 1)) + distance(
            nodes[tour[-1]], nodes[tour[0]])

    n = len(nodes)
    population_size = 5
    population = [random.sample(range(n), n) for _ in range(population_size)]
    temperature = 1.0
    cooling_rate = 0.995

    for _ in range(10):
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(population, 2)
            split = random.randint(0, n - 1)
            child = parent1[:split] + [gene for gene in parent2 if gene not in parent1[:split]]

            if random.random() < temperature:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]

            new_population.append(child)

        population = sorted(new_population, key=total_distance)[:population_size]
        temperature *= cooling_rate

    best_tour = min(population, key=total_distance)
    return {'tour': best_tour}



