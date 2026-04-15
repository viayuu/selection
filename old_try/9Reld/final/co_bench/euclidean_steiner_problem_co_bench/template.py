import math
from itertools import combinations


def solve(points: list) -> dict:
    """
    Solves a single instance of the Euclidean Steiner Problem.
    Problem Description:
      Given a set of 2D points (terminals), the goal is to compute additional Steiner points
      such that when you compute the MST over the union of the original terminals and these Steiner points,
      the total length (measured via Euclidean distances) is minimized.
      (Recall, the Euclidean distance between two points (x1, y1) and (x2, y2) is sqrt((x1-x2)^2 + (y1-y2)^2).)
    Input kwargs:
      - points: a list of points, where each point is a tuple of floats (x, y),
                representing the coordinates of an original terminal.
    Returns:
      A dictionary with one key:
         - "steiner_points": a list of (x, y) tuples representing the additional Steiner points.
      It is assumed that the candidate solution’s computed total length can be derived by computing
      the MST over the union of the original terminals and the returned Steiner points.
    """
    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def mst_length(nodes):
        if len(nodes) <= 1:
            return 0.0
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((euclidean_distance(nodes[i], nodes[j]), i, j))
        edges.sort()
        parent = list(range(len(nodes)))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        total = 0.0
        for d, u, v in edges:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u
                total += d
        return total

    def fermat_torricelli_point(a, b, c):
        def angle(p1, p2, p3):
            x1, y1 = p1[0] - p2[0], p1[1] - p2[1]
            x2, y2 = p3[0] - p2[0], p3[1] - p2[1]
            dot = x1 * x2 + y1 * y2
            det = x1 * y2 - y1 * x2
            return math.atan2(det, dot)
        A = angle(b, a, c)
        B = angle(c, b, a)
        C = angle(a, c, b)
        if abs(A) >= 2 * math.pi / 3:
            return a
        if abs(B) >= 2 * math.pi / 3:
            return b
        if abs(C) >= 2 * math.pi / 3:
            return c
        def compute_fermat_point(a, b, c):
            def equilateral_point(p1, p2):
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1
                x3 = x1 + (dx - math.sqrt(3) * dy) / 2
                y3 = y1 + (math.sqrt(3) * dx + dy) / 2
                return (x3, y3)
            pab = equilateral_point(a, b)
            pbc = equilateral_point(b, c)
            pca = equilateral_point(c, a)
            def line_intersection(p1, p2, p3, p4):
                def det(a, b):
                    return a[0] * b[1] - a[1] * b[0]
                xdiff = (p1[0] - p2[0], p3[0] - p4[0])
                ydiff = (p1[1] - p2[1], p3[1] - p4[1])
                div = det(xdiff, ydiff)
                if div == 0:
                    return None
                d = (det(p1, p2), det(p3, p4))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return (x, y)
            return line_intersection(a, pbc, b, pca)
        return compute_fermat_point(a, b, c)

    terminals = points.copy()
    steiner_points = []
    improved = True
    while improved:
        improved = False
        best_ratio = float('inf')
        best_new_point = None
        best_terminals = None
        for triplet in combinations(terminals, 3):
            new_point = fermat_torricelli_point(*triplet)
            new_terminals = terminals + [new_point]
            new_length = mst_length(new_terminals)
            original_length = mst_length(terminals)
            ratio = new_length / original_length
            if ratio < best_ratio:
                best_ratio = ratio
                best_new_point = new_point
                best_terminals = triplet
        if best_ratio < 1.0:
            steiner_points.append(best_new_point)
            terminals.append(best_new_point)
            improved = True
    return {"steiner_points": steiner_points}

