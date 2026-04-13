#!/usr/bin/env python3
"""
Convert TSPLIB format to DACT format

TSPLIB format (.tsp):
    NAME: berlin52
    TYPE: TSP
    DIMENSION: 52
    NODE_COORD_SECTION
    1 565.0 575.0
    2 25.0 185.0
    ...

DACT format (.pkl):
    List of coordinate arrays: [(x1,y1), (x2,y2), ...]
    Each array is a torch.FloatTensor of shape (n_nodes, 2)
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import tsplib95

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TSPLIB to DACT format")
    parser.add_argument("--input", type=str, required=True, help="Path to TSPLIB file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output .pkl file path")
    parser.add_argument("--normalize", action="store_true", help="Normalize coordinates to [0,1]")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of instances to convert")
    return parser.parse_args()


def load_tsplib(filepath):
    """Load TSPLIB file and return nodes as numpy array"""
    problem = tsplib95.load(filepath)
    nodes = []
    for node_id in range(1, problem.dimension + 1):
        coord = problem.node_coords[node_id]
        nodes.append([coord[0], coord[1]])
    return np.array(nodes)


def normalize_coordinates(nodes):
    """Normalize coordinates to [0, 1] range"""
    min_val = nodes.min(axis=0)
    max_val = nodes.max(axis=0)
    return (nodes - min_val) / (max_val - min_val + 1e-8)


def tsplib_to_dact_format(nodes):
    """Convert nodes to DACT format (torch.FloatTensor)"""
    return torch.FloatTensor(nodes)


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

    # Convert to DACT format
    coordinates = tsplib_to_dact_format(nodes)

    return coordinates


def process_directory(input_dir, args):
    """Process all .tsp files in a directory"""
    input_path = Path(input_dir)
    tsp_files = list(input_path.glob("*.tsp"))

    if not tsp_files:
        print(f"No .tsp files found in {input_dir}")
        return None

    print(f"Found {len(tsp_files)} TSP files")

    # Apply num_samples limit if specified
    if args.num_samples is not None and args.num_samples < len(tsp_files):
        tsp_files = tsp_files[:args.num_samples]
        print(f"Limited to {args.num_samples} files")

    # Process all files
    results = []
    for tsp_file in tsp_files:
        try:
            coords = process_single_file(tsp_file, args)
            results.append(coords)
        except Exception as e:
            print(f"Error processing {tsp_file}: {e}")
            continue

    return results


def main():
    args = parse_args()

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file - create list with one instance
        coords = process_single_file(args.input, args)
        results = [coords]

        # Get size for naming
        size = coords.shape[0]

    elif input_path.is_dir():
        # Directory with multiple files
        results = process_directory(args.input, args)

        if results is None or len(results) == 0:
            print("No valid instances found")
            return

        # Get size from first instance
        size = results[0].shape[0]

    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to .pkl file
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nSaved {len(results)} instances to {args.output}")
    print(f"Instance size: {size} nodes")
    print(f"Data format: List of {len(results)} torch.Tensor of shape ({size}, 2)")


if __name__ == "__main__":
    main()
