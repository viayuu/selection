import networkx
import numpy as np
import networkx as nx
def solve(graph: nx.Graph):
    """
    Solve the Maximum Independent Set problem for a given test case.
   Input:
        kwargs (dict): A dictionary with the following keys:
            - graph (networkx.Graph): The graph to solve
    Returns:
        dict: A solution dictionary containing:
            - mis_nodes (list): List of node indices in the maximum independent set
    """
    if not graph.nodes:
        return {"mis_nodes": []}

    # Greedy selection: pick node with minimum degree
    nodes_sorted = sorted(graph.degree, key=lambda x: x[1])
    independent_set = set()
    remaining_graph = graph.copy()

    while remaining_graph.nodes:
        node, _ = nodes_sorted.pop(0)
        independent_set.add(node)
        # Remove selected node and its neighbors
        neighbors = list(remaining_graph.neighbors(node))
        remaining_graph.remove_nodes_from(neighbors + [node])
        # Update sorted list
        nodes_sorted = [n for n in nodes_sorted if n[0] in remaining_graph]
        nodes_sorted.sort(key=lambda x: x[1])

    return {"mis_nodes": list(independent_set)}


