import random
import math

def solve(stock_length: int, stock_width: int, pieces: list) -> dict:
    """
    Solves the constrained non-guillotine cutting problem.
    Input kwargs:
      - stock_length (int): Length of the stock rectangle.
      - stock_width (int): Width of the stock rectangle.
      - pieces (list of dict): List of pieces, where each dict has:
            'length' (int), 'width' (int),
            'min' (int): minimum number required,
            'max' (int): maximum allowed,
            'value' (int): value of the piece.
    Evaluation Metric:
      The solution is scored as the sum of the values of all placed pieces,
      provided that every placement is valid (i.e., pieces lie within bounds,
      do not overlap, and the count for each type meets the specified [min, max] range).
      If any constraint is violated, the solution receives no score.
    Returns:
      A dictionary with one key:
          'placements': a list of placements, where each placement is a 4-tuple:
                        (piece_type, x, y, r)
                       - piece_type: 1-indexed index of the piece type.
                       - x, y: integer coordinates for the placement (bottom-left corner).
                       - r: rotation flag (0 for no rotation, 1 for 90° rotation).
    """

    def is_valid_placement(placements):
        used_area = []
        piece_counts = [0] * len(pieces)

        for placement in placements:
            piece_idx, x, y, r = placement
            piece = pieces[piece_idx - 1]
            length = piece['length'] if not r else piece['width']
            width = piece['width'] if not r else piece['length']

            # Check bounds
            if x < 0 or y < 0 or x + length > stock_length or y + width > stock_width:
                return False

            # Check overlap
            new_rect = (x, y, x + length, y + width)
            for rect in used_area:
                if not (new_rect[2] <= rect[0] or new_rect[0] >= rect[2] or
                        new_rect[3] <= rect[1] or new_rect[1] >= rect[3]):
                    return False
            used_area.append(new_rect)
            piece_counts[piece_idx - 1] += 1

        # Check min/max constraints
        for i, piece in enumerate(pieces):
            if not (piece['min'] <= piece_counts[i] <= piece['max']):
                return False
        return True

    def evaluate(placements):
        if not is_valid_placement(placements):
            return 0
        total_value = 0
        for placement in placements:
            piece_idx = placement[0]
            total_value += pieces[piece_idx - 1]['value']
        return total_value

    def generate_initial_solution():
        placements = []
        remaining_pieces = []
        for i, piece in enumerate(pieces, 1):
            remaining_pieces.extend([i] * piece['min'])

        random.shuffle(remaining_pieces)

        for piece_idx in remaining_pieces:
            piece = pieces[piece_idx - 1]
            for _ in range(100):  # Try 100 random positions
                r = random.randint(0, 1)
                length = piece['length'] if not r else piece['width']
                width = piece['width'] if not r else piece['length']
                if length > stock_length or width > stock_width:
                    continue
                x = random.randint(0, stock_length - length)
                y = random.randint(0, stock_width - width)
                new_placement = (piece_idx, x, y, r)
                temp_placements = placements.copy()
                temp_placements.append(new_placement)
                if is_valid_placement(temp_placements):
                    placements = temp_placements
                    break
        return placements

    def neighbor(current_solution):
        new_solution = current_solution.copy()
        if not new_solution:
            return new_solution

        action = random.choice(['add', 'remove', 'move', 'rotate'])

        if action == 'add' and len(new_solution) < sum(p['min'] for p in pieces) + 10:
            piece_idx = random.randint(1, len(pieces))
            piece = pieces[piece_idx - 1]
            r = random.randint(0, 1)
            length = piece['length'] if not r else piece['width']
            width = piece['width'] if not r else piece['length']
            if length <= stock_length and width <= stock_width:
                x = random.randint(0, stock_length - length)
                y = random.randint(0, stock_width - width)
                new_placement = (piece_idx, x, y, r)
                new_solution.append(new_placement)

        elif action == 'remove' and len(new_solution) > sum(p['min'] for p in pieces):
            idx = random.randint(0, len(new_solution) - 1)
            new_solution.pop(idx)

        elif action == 'move' or action == 'rotate':
            idx = random.randint(0, len(new_solution) - 1)
            piece_idx, x, y, r = new_solution[idx]
            piece = pieces[piece_idx - 1]

            if action == 'rotate':
                r = 1 - r
                length = piece['length'] if not r else piece['width']
                width = piece['width'] if not r else piece['length']
                if length > stock_length or width > stock_width:
                    return current_solution
                x = min(x, stock_length - length)
                y = min(y, stock_width - width)
            else:
                length = piece['length'] if not r else piece['width']
                width = piece['width'] if not r else piece['length']
                dx = random.randint(-length, length)
                dy = random.randint(-width, width)
                x = max(0, min(stock_length - length, x + dx))
                y = max(0, min(stock_width - width, y + dy))

            new_solution[idx] = (piece_idx, x, y, r)

        return new_solution

    # Simulated Annealing
    current_solution = generate_initial_solution()
    current_value = evaluate(current_solution)
    best_solution = current_solution.copy()
    best_value = current_value

    temperature = 1.0
    cooling_rate = 0.999
    iterations = 10000

    for _ in range(iterations):
        new_solution = neighbor(current_solution)
        new_value = evaluate(new_solution)

        if new_value > current_value or (new_value > 0 and
                                         random.random() < math.exp((new_value - current_value) / temperature)):
            current_solution = new_solution
            current_value = new_value

            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value = current_value

        temperature *= cooling_rate

    return {'placements': best_solution}

