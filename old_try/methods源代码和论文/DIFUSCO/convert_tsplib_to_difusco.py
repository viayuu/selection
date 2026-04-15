#!/usr/bin/env python3
"""
Convert TSPLIB format to DIFUSCO format

TSPLIB format (.tsp):
    NAME: berlin52
    TYPE: TSP
    DIMENSION: 52
    NODE_COORD_SECTION
    1 565.0 575.0
    2 25.0 185.0
    ...

DIFUSCO format:
    x1 y1 x2 y2 ... xn yn output tour1 tour2 ... tourN tour1
"""

import argparse
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm
import tsplib95

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TSPLIB to DIFUSCO format")
    parser.add_argument("--input", type=str, required=True, help="Path to TSPLIB file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--normalize", action="store_true", help="Normalize coordinates to [0,1]")
    parser.add_argument("--solver", type=str, default="none", choices=["none", "lkh", "concorde"],
                        help="Solver to use for optimal tour (default: none, use sequential order)")
    parser.add_argument("--lkh_path", type=str, default="LKH-3.0.6/LKH", help="Path to LKH executable")
    parser.add_argument("--lkh_trials", type=int, default=1000, help="LKH max trials")
    return parser.parse_args()


def load_tsplib(filepath):
    """Load TSPLIB file and return nodes as numpy array"""
    problem = tsplib95.load(filepath)
    nodes = []
    for node_id in range(1, problem.dimension + 1):
        coord = problem.node_coords[node_id]
        nodes.append([coord[0], coord[1]])
    return np.array(nodes)


def solve_tsp_with_lkh(nodes, lkh_path, max_trials=1000):
    """Solve TSP using LKH solver"""
    try:
        import lkh
    except ImportError:
        print("Warning: LKH not installed, using sequential order")
        return list(range(len(nodes)))

    num_nodes = len(nodes)
    scale = 1e6

    # Create TSP problem
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = num_nodes
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {n + 1: nodes[n] * scale for n in range(num_nodes)}

    # Solve with LKH
    solution = lkh.solve(lkh_path, problem=problem, max_trials=max_trials, runs=10)
    tour = [n - 1 for n in solution[0]]  # Convert to 0-indexed

    return tour


def solve_tsp_with_concorde(nodes):
    """Solve TSP using Concorde solver"""
    try:
        from concorde.tsp import TSPSolver
    except ImportError:
        print("Warning: Concorde not installed, using sequential order")
        return list(range(len(nodes)))

    scale = 1e6
    solver = TSPSolver.from_data(
        nodes[:, 0] * scale,
        nodes[:, 1] * scale,
        norm="EUC_2D"
    )
    solution = solver.solve(verbose=False)
    tour = solution.tour  # Already 0-indexed

    return tour


def normalize_coordinates(nodes):
    """Normalize coordinates to [0, 1] range"""
    min_val = nodes.min(axis=0)
    max_val = nodes.max(axis=0)
    return (nodes - min_val) / (max_val - min_val + 1e-8)


def nodes_to_difusco_format(nodes, tour=None):
    """Convert nodes and tour to DIFUSCO format string"""
    n = len(nodes)

    # Generate coordinate string: x1 y1 x2 y2 ...
    coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in nodes)

    # Generate tour string (1-indexed, returns to start)
    if tour is None:
        # Use sequential order as fallback
        tour = list(range(n))

    # Convert to 1-indexed and add return to start
    tour_1indexed = [str(t + 1) for t in tour] + [str(tour[0] + 1)]
    tour_str = " ".join(tour_1indexed)

    return f"{coords_str} output {tour_str}\n"


def process_single_file(input_file, args):
    """Process a single TSPLIB file"""
    print(f"Processing: {input_file}")

    # Load TSPLIB
    nodes = load_tsplib(input_file)
    num_nodes = len(nodes)
    print(f"  Nodes: {num_nodes}")

    # Normalize if requested
    if args.normalize:
        nodes = normalize_coordinates(nodes)
        print(f"  Normalized to [0, 1]")

    # Solve for optimal tour if requested
    tour = None
    if args.solver == "lkh":
        print(f"  Solving with LKH...")
        tour = solve_tsp_with_lkh(nodes, args.lkh_path, args.lkh_trials)
    elif args.solver == "concorde":
        print(f"  Solving with Concorde...")
        tour = solve_tsp_with_concorde(nodes)

    # Convert to DIFUSCO format
    difusco_line = nodes_to_difusco_format(nodes, tour)

    return difusco_line


def process_directory(input_dir, args):
    """Process all .tsp files in a directory"""
    input_path = Path(input_dir)
    tsp_files = list(input_path.glob("*.tsp"))

    if not tsp_files:
        print(f"No .tsp files found in {input_dir}")
        return

    print(f"Found {len(tsp_files)} TSP files")

    # Process all files
    results = []
    for tsp_file in tqdm.tqdm(tsp_files, desc="Converting"):
        try:
            line = process_single_file(tsp_file, args)
            results.append(line)
        except Exception as e:
            print(f"Error processing {tsp_file}: {e}")
            continue

    # Write to output file
    with open(args.output, 'w') as f:
        f.writelines(results)

    print(f"\nSaved {len(results)} instances to {args.output}")


def main():
    args = parse_args()

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        line = process_single_file(args.input, args)

        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, 'w') as f:
            f.write(line)

        print(f"Saved to {args.output}")

    elif input_path.is_dir():
        # Directory with multiple files
        process_directory(args.input, args)

    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()