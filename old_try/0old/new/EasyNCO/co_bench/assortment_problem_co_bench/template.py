import numpy as np
from scipy.optimize import linear_sum_assignment
def solve(m: int, stocks: list, pieces: list) -> dict:
    """
    Solves the rectangular piece arrangement optimization problem to minimize the overall waste area percentage.
    Given:
      - m (int): Number of piece types.
      - stocks (list of dict): Each dict represents a stock type with keys:
            'length' (float), 'width' (float), 'fixed_cost' (float).
      - pieces (list of dict): Each dict represents a piece type with keys:
            'length' (float), 'width' (float), 'min' (int), 'max' (int), 'value' (float).
    Objective:
      Arrange rectangular pieces (which may be rotated by 90°) into stock rectangles such that the overall waste area percentage is minimized.
      The waste area percentage is computed as:
             Waste Percentage = (Total Stock Area - Total Used Area) / (Total Stock Area)
    Constraints:
      • Each piece must lie entirely within its assigned stock rectangle.
      • Pieces must not overlap within the same stock rectangle.
      • The number of pieces placed for each piece type must lie within its specified minimum and maximum bounds.
      • You may use unlimited many instances of each selected stock type, but the solution can include at most 2 distinct stock types.
    Output:
      Returns a dictionary with two keys (exactly follow this format):
        - "objective": The overall waste area percentage (float) as computed by the evaluation function.
        - "placements": A dictionary mapping stock instance ids (1-indexed) to their placement details.
          Each stock instance is represented by a dictionary with the following keys:
              'stock_type': (the 1-indexed id of the stock type used for this instance),
              'placements': a list of placements for pieces within that stock instance.
                  Each placement is a dict with keys:
                      'piece'       (piece type, 1-indexed, 1 <= piece type <= m),
                      'x'           (x-coordinate of the bottom-left corner),
                      'y'           (y-coordinate of the bottom-left corner),
                      'orientation' (0 for normal, 1 for rotated 90°).
    NOTE: The returned data should adhere to the output format required for evaluation.
    """
    import random
    from itertools import combinations

    def evaluate(stock_comb, pieces_placed):
        total_stock_area = 0
        total_used_area = 0
        for stock_idx, stock in stock_comb:
            stock_area = stock['length'] * stock['width']
            total_stock_area += stock_area * pieces_placed[stock_idx]
            total_used_area += sum(p['length'] * p['width'] for p in pieces if p['type'] in pieces_placed[stock_idx])
        if total_stock_area == 0:
            return float('inf')
        return (total_stock_area - total_used_area) / total_stock_area

    def greedy_pack(stock, piece_list):
        placements = []
        remaining_pieces = piece_list.copy()
        stock_length = stock['length']
        stock_width = stock['width']
        current_x = 0
        current_y = 0
        max_height_in_row = 0
        while remaining_pieces:
            placed = False
            for i, piece in enumerate(remaining_pieces):
                for orientation in [0, 1]:
                    length = piece['length'] if orientation == 0 else piece['width']
                    width = piece['width'] if orientation == 0 else piece['length']
                    if current_x + length <= stock_length and current_y + width <= stock_width:
                        placements.append({
                            'piece': piece['type'],
                            'x': current_x,
                            'y': current_y,
                            'orientation': orientation
                        })
                        max_height_in_row = max(max_height_in_row, width)
                        current_x += length
                        remaining_pieces.pop(i)
                        placed = True
                        break
                    elif current_x + width <= stock_length and current_y + length <= stock_width:
                        placements.append({
                            'piece': piece['type'],
                            'x': current_x,
                            'y': current_y,
                            'orientation': 1 - orientation
                        })
                        max_height_in_row = max(max_height_in_row, length)
                        current_x += width
                        remaining_pieces.pop(i)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                if current_x == 0:
                    break
                current_x = 0
                current_y += max_height_in_row
                max_height_in_row = 0
        return placements, remaining_pieces

    best_solution = {'objective': float('inf'), 'placements': {}}
    stock_combinations = combinations(enumerate(stocks, 1), min(2, len(stocks)))

    for comb in stock_combinations:
        stock_indices = [s[0] for s in comb]
        stock_types = [s[1] for s in comb]
        pieces_to_place = []
        for i, piece in enumerate(pieces, 1):
            count = random.randint(piece['min'], piece['max'])
            pieces_to_place.extend([{'type': i, 'length': piece['length'], 'width': piece['width']}] * count)

        placements_dict = {}
        remaining_pieces = pieces_to_place.copy()
        stock_instance_id = 1
        while remaining_pieces:
            best_stock_idx = None
            best_placements = None
            best_remaining = None
            for stock_idx, stock in comb:
                placements, remaining = greedy_pack(stock, remaining_pieces)
                if len(remaining) < len(best_remaining) if best_remaining is not None else True:
                    best_stock_idx = stock_idx
                    best_placements = placements
                    best_remaining = remaining
            if best_stock_idx is None:
                break
            placements_dict[stock_instance_id] = {
                'stock_type': best_stock_idx,
                'placements': best_placements
            }
            remaining_pieces = best_remaining
            stock_instance_id += 1

        if not remaining_pieces:
            total_stock_area = sum(stock['length'] * stock['width'] * len([k for k in placements_dict if placements_dict[k]['stock_type'] == s[0]]) for s in comb)
            total_used_area = sum(p['length'] * p['width'] for p in pieces_to_place)
            waste_percentage = (total_stock_area - total_used_area) / total_stock_area if total_stock_area > 0 else float('inf')
            if waste_percentage < best_solution['objective']:
                best_solution = {
                    'objective': waste_percentage,
                    'placements': placements_dict
                }

    return best_solution


