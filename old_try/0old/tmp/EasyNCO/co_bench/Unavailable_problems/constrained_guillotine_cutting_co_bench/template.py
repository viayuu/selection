def solve(m: int, stock_length: int, stock_width: int, piece_types: list) -> dict:
    """
    Solves the Fixed Orientation Guillotine Cutting problem.
    Problem Description:
      Given a rectangular stock sheet (with specified length and width) and a set of piece types
      (each defined by a length, width, an upper bound on the number of times it may appear, and a value),
      the goal is to determine a placement for the pieces such that:
        - Each placed piece lies entirely within the stock sheet.
        - Pieces do not overlap.
        - The number of pieces placed for any type does not exceed its allowed maximum.
        - The orientation of the pieces is fixed (i.e. no rotation is allowed).
        - The total value reported equals the sum of the values of the placed pieces.
    Input kwargs (for one case):
      - m: integer, the number of piece types.
      - stock_length: integer, the length of the stock sheet.
      - stock_width: integer, the width of the stock sheet.
      - piece_types: list of dictionaries. Each dictionary has the keys:
            'length' : int, the length of the piece.
            'width'  : int, the width of the piece.
            'max'    : int, maximum number of pieces allowed.
            'value'  : int, value of the piece.
    Returns:
      A dictionary with the following keys:
        - total_value: int, the computed total value (must equal the sum of the piece values in placements).
        - placements: list of placements, where each placement is a tuple of 6 integers:
              (piece_type_index, x, y, placed_length, placed_width, orientation_flag)
          The orientation_flag is always 0 since rotation is not allowed.
    """
    from functools import lru_cache

    # Preprocess piece types to include index and filter out impossible pieces
    pieces = []
    for idx, pt in enumerate(piece_types):
        if pt['length'] <= stock_length and pt['width'] <= stock_width and pt['max'] > 0:
            pieces.append((idx, pt['length'], pt['width'], pt['max'], pt['value']))

    # Memoization table for DP: (length, width) -> (max_value, placements)
    memo = {}

    def dp(L, W):
        if (L, W) in memo:
            return memo[(L, W)]

        max_value = 0
        best_placements = []

        # Try placing each piece type in the current rectangle
        for idx, l, w, max_p, val in pieces:
            if l <= L and w <= W and max_p > 0:
                # Place the piece at (0, 0)
                remaining_pieces = [(i, li, wi, mp - (1 if i == idx else 0), vi)
                                    for i, li, wi, mp, vi in pieces]

                # Recursively solve the remaining sub-rectangles
                # Option 1: Horizontal cut above the piece
                if L - l > 0:
                    val_below, placements_below = dp(L - l, W)
                    total_val = val + val_below
                    if total_val > max_value:
                        max_value = total_val
                        best_placements = [(idx, 0, 0, l, w, 0)] + placements_below

                # Option 2: Vertical cut to the right of the piece
                if W - w > 0:
                    val_right, placements_right = dp(L, W - w)
                    total_val = val + val_right
                    if total_val > max_value:
                        max_value = total_val
                        best_placements = [(idx, 0, 0, l, w, 0)] + placements_right

        # Try all possible guillotine cuts
        # Horizontal cuts
        for cut in range(1, L):
            val_top, placements_top = dp(cut, W)
            val_bottom, placements_bottom = dp(L - cut, W)
            total_val = val_top + val_bottom
            if total_val > max_value:
                max_value = total_val
                best_placements = placements_top + placements_bottom

        # Vertical cuts
        for cut in range(1, W):
            val_left, placements_left = dp(L, cut)
            val_right, placements_right = dp(L, W - cut)
            total_val = val_left + val_right
            if total_val > max_value:
                max_value = total_val
                best_placements = placements_left + placements_right

        memo[(L, W)] = (max_value, best_placements)
        return max_value, best_placements

    total_value, placements = dp(stock_length, stock_width)

    # Filter placements to ensure max constraints are not violated
    piece_counts = [0] * m
    valid_placements = []
    for placement in placements:
        idx = placement[0]
        if piece_counts[idx] < piece_types[idx]['max']:
            valid_placements.append(placement)
            piece_counts[idx] += 1

    # Recalculate total value based on valid placements
    total_value = sum(piece_types[p[0]]['value'] for p in valid_placements)

    return {
        'total_value': total_value,
        'placements': valid_placements
    }

