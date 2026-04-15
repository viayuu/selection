import math
def solve(n: int, cx: float, cy: float, R: float, radii: list) -> dict:
    """
    Solve the unequal circle packing problem for the maximize-number case.
    Problem Description:
      Given a circular container with center (cx, cy) and radius R, and n circles with specified radii (sorted in increasing order),
      the task is to select and pack a prefix of the sorted list—i.e., if circle i is packed, then all circles with a smaller index must also be packed—in order to maximize the number of circles placed.
      Each packed circle must be fully contained within the container, meaning that the distance from its center to (cx, cy) plus its radius must not exceed R, and no two packed circles may overlap, which requires that the distance between any two centers is at least the sum of their respective radii.
    Input kwargs:
      - n     : int, the number of circles.
      - cx    : float, x-coordinate of the container's center.
      - cy    : float, y-coordinate of the container's center.
      - R     : float, the radius of the container.
      - radii : list of float, the radius of each circle (assumed sorted in increasing order).
    Returns:
      A dictionary with one key:
        - "coords": a list of n (x, y) tuples corresponding to the centers of the circles.
          For circles that are not packed, the coordinates default to (-1, -1).
    """
    coords = [(-1, -1) for _ in range(n)]
    packed = []

    for i in range(n):
        r_i = radii[i]
        if r_i > R:
            continue

        if not packed:
            coords[i] = (cx, cy)
            packed.append((cx, cy, r_i))
            continue

        angle = 0.0
        distance = R - r_i
        max_attempts = 100
        found = False

        for attempt in range(max_attempts):
            x = cx + distance * math.cos(angle)
            y = cy + distance * math.sin(angle)
            valid = True

            for (px, py, pr) in packed:
                dx = x - px
                dy = y - py
                if math.sqrt(dx * dx + dy * dy) < r_i + pr:
                    valid = False
                    break

            if valid:
                coords[i] = (x, y)
                packed.append((x, y, r_i))
                found = True
                break

            angle += math.pi / 4

        if not found:
            break

    return {"coords": coords}

