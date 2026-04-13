import numpy as np
import scipy.optimize as opt
def solve(n: int, edges: list, adjacency: dict) -> dict:
    """
    Problem:
        Given a graph in DIMACS format (with vertices, edges, and an adjacency list),
        assign a positive integer color to each vertex (1..n) so that no two adjacent vertices
        share the same color. The objective is to use as few colors as possible.
    Input kwargs:
    The keyword arguments are expected to include at least:
      - n: int (int), the number of vertices.
      - edges: list of (u, v) tuples (tuple of int (int), int (int)) representing edges.
      - adjacency: dict mapping each vertex (1..n) (int) to a set of its adjacent vertices (set of int).
    Evaluation Metric:
        Let  k  be the number of distinct colors used.
        For every edge connecting two vertices with the same color, count one conflict ( C ).
        If  C > 0 , the solution is invalid and receives no score.
        Otherwise, the score is simply  k , with a lower  k  being better.
    Returns:
        A dictionary representing the solution, mapping each vertex_id (1..n) to a positive integer color.
    """
    color = {}
    saturation = {v: 0 for v in range(1, n + 1)}
    uncolored = set(range(1, n + 1))

    while uncolored:
        v = max(uncolored, key=lambda x: (saturation[x], len(adjacency[x])))
        used_colors = {color[u] for u in adjacency[v] if u in color}
        for c in range(1, n + 1):
            if c not in used_colors:
                color[v] = c
                break
        for u in adjacency[v]:
            if u in uncolored:
                saturation[u] = len({color[w] for w in adjacency[u] if w in color})
        uncolored.remove(v)

    return color


