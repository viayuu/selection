import numpy as np
import scipy.optimize as opt
def solve(m: int, stock_width: int, stock_height: int, pieces: dict, allow_rotation: bool=False) -> dict:
    """
    Solves the unconstrained guillotine cutting problem.
    Given a stock rectangle (with dimensions 'stock_width' and 'stock_height') and a set of pieces
    (provided as a dictionary 'pieces' mapping each piece_id to its specification {'l', 'w', 'value'}),
    the goal is to select and place some pieces (each used at most once) within the stock rectangle.
    If the keyword argument 'allow_rotation' is True, each piece may be placed in its original orientation or rotated 90° (swapping its dimensions);
    otherwise, pieces must be placed in their original orientation. In all cases, placements must not overlap and must lie entirely within the stock.
    Input kwargs:
        - m (int): Number of available pieces.
        - stock_width (int): The width of the stock rectangle.
        - stock_height (int): The height of the stock rectangle.
        - pieces (dict): A dictionary mapping piece_id (1-indexed) to a dict with keys:
              'l' (length), 'w' (width), and 'value' (value of the piece).
        - allow_rotation (bool): Indicates whether a piece is allowed to be rotated 90°.
    Evaluation metric:
        The performance is measured as the total value of the placed pieces (sum of individual values).
    Returns:
        A dictionary with a key "placements" whose value is a list.
        Each element in the list is a dictionary representing a placement with keys:
            - piece_id (int): Identifier of the placed piece.
            - x (int): x-coordinate of the bottom-left corner in the stock rectangle.
            - y (int): y-coordinate of the bottom-left corner in the stock rectangle.
            - orientation (int): 0 for original orientation; 1 if rotated 90° (only applicable if allow_rotation is True, otherwise default to 0).
    NOTE: This is a placeholder function. Replace the body with an actual algorithm if desired.
    """
    from collections import deque
    import heapq

    # Preprocess pieces: calculate value density and generate possible orientations
    piece_list = []
    for piece_id, spec in pieces.items():
        l, w, value = spec['l'], spec['w'], spec['value']
        orientations = []
        orientations.append((l, w, value, 0))
        if allow_rotation and l != w:
            orientations.append((w, l, value, 1))
        for (pl, pw, pv, rot) in orientations:
            density = pv / (pl * pw)
            piece_list.append((piece_id, pl, pw, pv, rot, density))

    # Sort pieces by descending value density
    piece_list.sort(key=lambda x: -x[5])

    # Initialize the stock as a list of free rectangles (x, y, width, height)
    free_rectangles = deque()
    free_rectangles.append((0, 0, stock_width, stock_height))

    placements = []
    used_pieces = set()

    for piece in piece_list:
        piece_id, pl, pw, pv, rot, _ = piece
        if piece_id in used_pieces:
            continue

        # Try to place the piece in the best free rectangle
        best_rect = None
        best_idx = -1
        for idx, rect in enumerate(free_rectangles):
            x, y, rw, rh = rect
            if (pl <= rw and pw <= rh) or (allow_rotation and pw <= rw and pl <= rh):
                best_rect = rect
                best_idx = idx
                break

        if best_rect is not None:
            x, y, rw, rh = best_rect
            del free_rectangles[best_idx]

            # Place the piece
            if pl <= rw and pw <= rh:
                placement = {'piece_id': piece_id, 'x': x, 'y': y, 'orientation': rot}
            else:
                placement = {'piece_id': piece_id, 'x': x, 'y': y, 'orientation': 1 - rot}

            placements.append(placement)
            used_pieces.add(piece_id)

            # Generate remaining free rectangles after placement (guillotine split)
            if x + pl <= stock_width and y + pw <= stock_height:
                # Horizontal split
                if rw - pl > 0:
                    free_rectangles.append((x + pl, y, rw - pl, rh))
                if rh - pw > 0:
                    free_rectangles.append((x, y + pw, pl, rh - pw))
            else:
                # Vertical split
                if rh - pl > 0:
                    free_rectangles.append((x, y + pw, rw, rh - pw))
                if rw - pw > 0:
                    free_rectangles.append((x + pw, y, rw - pw, pl))

            # Re-sort free rectangles by area to encourage better fits
            free_rectangles = deque(sorted(free_rectangles, key=lambda r: -(r[2] * r[3])))

    return {'placements': placements}

