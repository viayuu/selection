import numpy as np
import scipy.optimize as opt
def solve(n: int, cx: float, cy: float, R: float, radii: list) -> dict:
    """
    Solve the Unequal Circle Packing problem (Maximize Area version).
    Problem Description:
      Given a circular container with center (cx, cy) and radius R, and n circles
      with specified radii (provided in 'radii'), decide which circles to pack and
      determine the centers (x_i, y_i) for the packed circles such that:
      1. Containment: Each packed circle i must lie completely within the container.
         (x_i - cx)^2 + (y_i - cy)^2 <= α_i * (R - radii[i])^2,  for i = 1,...,n.
         (If α_i = 0, then the circle is not packed and its center is set to (cx, cy).)
      2. Non-Overlap: For every pair of circles i and j (with i < j), if both are packed,
         their centers must satisfy:
         (x_i - x_j)^2 + (y_i - y_j)^2 >= ( (α_i + α_j - 1) * (radii[i] + radii[j]) )^2.
         (This is a linearized version of the product α_i * α_j used in the paper.)
      3. Binary decisions: α_i ∈ {0, 1} for i = 1,...,n, where α_i = 1 indicates circle i is packed.
         (For circles not packed, we force (x_i, y_i) to equal (cx, cy).)
      4. Objective: Maximize the total area of the circles packed:
         maximize sum_{i=1}^n α_i * (pi * radii[i]^2).
    Input kwargs:
      - n     : int, the number of circles.
      - cx    : float, x-coordinate of the container's center.
      - cy    : float, y-coordinate of the container's center.
      - R     : float, the radius of the container.
      - radii : list of float, each element is the radius of a circle.
    Returns:
      A dictionary with one key:
        - "coords": a list of n (x, y) tuples corresponding to the centers of the circles.
                    For circles not packed (α_i = 0), (x, y) should be (-1, -1).
    """
    import math
    import random
    import numpy as np

    # Initialize parameters for simulated annealing
    initial_temp = 1000
    cooling_rate = 0.995
    min_temp = 1e-3
    max_iter = 10000

    # Precompute areas and sort circles by descending radius (for greedy initialization)
    sorted_indices = sorted(range(n), key=lambda i: -radii[i])
    sorted_radii = [radii[i] for i in sorted_indices]
    areas = [math.pi * r ** 2 for r in radii]

    # Initialize solution: pack as many large circles as possible greedily
    packed = [False] * n
    coords = [(-1, -1) for _ in range(n)]
    current_area = 0

    for i in sorted_indices:
        r = radii[i]
        if r > R:
            continue  # Cannot pack this circle

        # Try to place the circle as close to the center as possible without overlaps
        best_x, best_y = cx, cy
        best_distance = 0
        found = False

        # Generate candidate positions in spiral around the center
        for angle in np.linspace(0, 2 * math.pi, 20, endpoint=False):
            for distance in np.linspace(0, R - r, 10):
                x = cx + distance * math.cos(angle)
                y = cy + distance * math.sin(angle)

                # Check containment
                if (x - cx) ** 2 + (y - cy) ** 2 > (R - r) ** 2:
                    continue

                # Check overlap with already packed circles
                overlap = False
                for j in range(n):
                    if packed[j]:
                        dx = x - coords[j][0]
                        dy = y - coords[j][1]
                        if dx ** 2 + dy ** 2 < (r + radii[j]) ** 2:
                            overlap = True
                            break

                if not overlap:
                    best_x, best_y = x, y
                    found = True
                    break

            if found:
                break

        if found:
            packed[i] = True
            coords[i] = (best_x, best_y)
            current_area += areas[i]

    # Simulated annealing to improve the solution
    temperature = initial_temp
    best_coords = coords.copy()
    best_packed = packed.copy()
    best_area = current_area

    for _ in range(max_iter):
        if temperature < min_temp:
            break

        # Randomly choose a circle to flip (pack/unpack)
        i = random.randint(0, n - 1)
        r = radii[i]

        if packed[i]:
            # Try to unpack this circle
            new_area = current_area - areas[i]
            delta = new_area - current_area

            if delta > 0 or random.random() < math.exp(delta / temperature):
                packed[i] = False
                coords[i] = (-1, -1)
                current_area = new_area
        else:
            # Try to pack this circle
            if r > R:
                continue

            # Generate random position within container
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, R - r)
            x = cx + distance * math.cos(angle)
            y = cy + distance * math.sin(angle)

            # Check containment and overlap
            valid = True
            if (x - cx) ** 2 + (y - cy) ** 2 > (R - r) ** 2:
                valid = False
            else:
                for j in range(n):
                    if packed[j]:
                        dx = x - coords[j][0]
                        dy = y - coords[j][1]
                        if dx ** 2 + dy ** 2 < (r + radii[j]) ** 2:
                            valid = False
                            break

            if valid:
                new_area = current_area + areas[i]
                delta = new_area - current_area

                if delta > 0 or random.random() < math.exp(delta / temperature):
                    packed[i] = True
                    coords[i] = (x, y)
                    current_area = new_area

        # Update best solution
        if current_area > best_area:
            best_area = current_area
            best_coords = coords.copy()
            best_packed = packed.copy()

        temperature *= cooling_rate

    # Prepare output
    output_coords = [(-1, -1) if not best_packed[i] else best_coords[i] for i in range(n)]
    return {"coords": output_coords}


