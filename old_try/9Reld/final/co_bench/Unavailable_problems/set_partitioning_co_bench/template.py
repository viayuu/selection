from itertools import combinations


def solve(num_rows: int, num_columns: int, columns_info: dict) -> dict:
    """
    Solve a set partitioning problem instance.
    The problem: Given a set of rows and a set of columns (each with an associated cost and a set
    of rows it covers), select a subset of columns so that each row is covered exactly once and the
    total cost is minimized.
    Input kwargs:
  - num_rows (int): Total number of rows. (int)
  - num_columns (int): Total number of columns. (int)
  - columns_info (dict): Dictionary mapping 1-indexed column indices (int) to a tuple:
                         (cost (int), set of row indices (set[int]) covered by that column).
    Evaluation metric:
      The objective score equals the sum of the costs of the selected columns if the solution is feasible,
      i.e., if every row is covered exactly once. Otherwise, the solution is invalid and receives no score.
    Returns:
      A dictionary with key "selected_columns" containing a list of chosen column indices in strictly increasing order.
      (This is a placeholder implementation.)
    """
    columns = sorted(columns_info.keys())
    best_solution = None
    best_cost = float('inf')

    def backtrack(remaining_rows, selected_columns, current_cost, start_index):
        nonlocal best_solution, best_cost

        if not remaining_rows:
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = sorted(selected_columns)
            return

        for i in range(start_index, len(columns)):
            col = columns[i]
            cost, rows_covered = columns_info[col]
            new_remaining = remaining_rows - rows_covered
            if len(new_remaining) != len(remaining_rows) - len(rows_covered):
                continue  # Overlapping rows, skip
            if current_cost + cost >= best_cost:
                continue  # Prune if cost exceeds best
            backtrack(new_remaining, selected_columns + [col], current_cost + cost, i + 1)

    all_rows = set(range(1, num_rows + 1))
    backtrack(all_rows, [], 0, 0)

    return {"selected_columns": best_solution if best_solution else []}

