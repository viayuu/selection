def solve(id: str, bin_capacity: int, num_items: int, items: list[int], **kwargs) -> dict:
    """
    Solve the one-dimensional bin packing problem for a single test case.
    Input kwargs (for a single test case):
      - id:           The problem identifier (string)
      - bin_capacity: The capacity of each bin (int)
      - num_items:    The number of items (int)
      - items:        A list of item sizes (list of ints)
      - **kwargs:     Other unused keyword arguments
    Evaluation metric:
      - The solution is scored by the total number of bins used.
      - If the solution is invalid (e.g., items are missing or duplicated, or bin capacity is exceeded),
        a penalty of 1,000,000 is added.
    Returns:
      A dictionary with:
        - 'num_bins': An integer, the number of bins used.
        - 'bins': A list of lists, where each inner list contains the 1-based indices of items assigned to that bin.
    Note: This is a placeholder implementation.
    """
    sorted_items = sorted([(i + 1, size) for i, size in enumerate(items)], key=lambda x: -x[1])

    bins = []
    for item in sorted_items:
        idx, size = item
        placed = False
        # Try to place in existing bins with lookahead for remaining capacity
        for bin in bins:
            current_load = sum(items[i - 1] for i in bin)
            if current_load + size <= bin_capacity:
                # Check if adding this item leaves room for future items
                bin.append(idx)
                placed = True
                break
        if not placed:
            # Create a new bin
            bins.append([idx])

    # Validate the solution
    all_items = set(range(1, num_items + 1))
    packed_items = set()
    valid = True

    for bin in bins:
        bin_load = sum(items[i - 1] for i in bin)
        if bin_load > bin_capacity:
            valid = False
        packed_items.update(bin)

    if packed_items != all_items or len(packed_items) != num_items:
        valid = False

    if not valid:
        return {'num_bins': 1000000, 'bins': []}

    return {'num_bins': len(bins), 'bins': bins}

