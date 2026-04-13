import os
import numpy as np
import torch
from EasyNCO.data.LIBUtils import CVRPLIBReader
from EasyNCO.data.LIBUtils import TSPLIBReader
from EasyNCO.data.LIBUtils import ATSPLIBReader
from EasyNCO.data.lkh_params_utils import read_lkh_params

"""
This module contains functions to parse solution files from various optimization algorithms and save the results.

Each function reads a solution file from a given directory, extracts relevant information such as the tour and cost,
and then optionally saves this data to a .txt file for further processing or analysis.
"""

def lkh_tsp_solution(directory, nodes, scale, distribution, attributes, duration, par_args, problem_index = None):
    # Read the params.par file
    params_dict = read_lkh_params(directory)

    # Read the output.tour file
    tour_file = os.path.join(directory, 'output.tour')
    with open(tour_file, 'r') as f:
        tour_lines = f.readlines()

    cost = None
    tour = []
    started = False
    for line in tour_lines:
        if line.startswith('COMMENT : Length ='):
            cost = float(line.split('Length = ')[1].strip())
        elif line.startswith('TOUR_SECTION'):
            started = True
        elif started and line.strip().isdigit():
            loc = int(line.strip())
            if loc == -1:
                break
            tour.append(loc)
    tour = np.array(tour).astype(int)
    # Convert tour to 0-based index
    cost = cost / par_args.int_coord_scale
    # Recalculate cost to preserve more decimal places
    solution = [x - 1 for x in tour]
    node_xy_tensor = torch.tensor(nodes, dtype=torch.float32)
    node_xy_tensor = node_xy_tensor.unsqueeze(0)
    sequence_tensor = torch.tensor(solution, dtype=torch.int64)

    # node_xy.shape=(1, node_num,2), sequence.shape=(m,)
    problem = node_xy_tensor[0]
    # shape: (node_num, 2)
    gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
    # shape: (m, 2)
    ordered_seq = problem.gather(dim=0, index=gathering_index)
    # shape:(m,2)
    rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
    manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())

    # Save as .txt file
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{scale}.txt"
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:
            f.write("node_xy:")
            for coord in nodes:
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")

            f.write(f"scale: {scale}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            f.write("solution: ")
            f.write(" ".join(map(str, tour)))  # Return to start point
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")

            f.write("LKH settings:\n")
            for key, value in params_dict.items():
                f.write(f"{key}: {value}\n")

    return tour, manual_calculate_cost


def lkh_atsp_solution(directory, node_matrix, scale, distribution, attributes, duration, par_args, problem_index = None):
    # Read the params.par file
    params_dict = read_lkh_params(directory)

    # Read the output.tour file
    tour_file = os.path.join(directory, 'output.tour')
    with open(tour_file, 'r') as f:
        tour_lines = f.readlines()

    cost = None
    tour = []
    started = False
    for line in tour_lines:
        if line.startswith('COMMENT : Length ='):
            cost = float(line.split('Length = ')[1].strip())
        elif line.startswith('TOUR_SECTION'):
            started = True
        elif started and line.strip().isdigit():
            loc = int(line.strip())
            if loc == -1:
                break
            tour.append(loc)
    tour = np.array(tour).astype(int)
    # Convert tour to 0-based index
    cost = cost / par_args.int_matrix_scale
    # Recalculate cost to preserve more decimal places
    manual_calculate_cost = 0
    solution = [x - 1 for x in tour]
    for i in range(len(solution)-1):
        manual_calculate_cost = manual_calculate_cost + node_matrix[solution[i]][solution[i+1]]
    manual_calculate_cost = manual_calculate_cost + node_matrix[solution[-1]][solution[0]]
    # Save as .txt file
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{scale}.txt"
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:
            f.write("node_xy:")

            f.write("\n")

            f.write(f"scale: {scale}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            f.write("solution: ")
            f.write(" ".join(map(str, tour)))  # Return to start point
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")

            f.write("LKH settings:\n")
            for key, value in params_dict.items():
                f.write(f"{key}: {value}\n")

    return tour, manual_calculate_cost


def lkh_cvrp_solution(directory, depot, locs, demands, capacity, scale, distribution, attributes, duration, par_args, problem_index = None):
    # Read the params.par file
    params_dict = read_lkh_params(directory)

    # Read the output.tour file
    tour_file = os.path.join(directory, 'output.tour')
    with open(tour_file, 'r') as f:
        tour_lines = f.readlines()

    cost = None
    tour = []
    started = False
    for line in tour_lines:
        if line.startswith('COMMENT : Length ='):
            cost = float(line.split('Length = ')[1].strip())
        elif line.startswith('TOUR_SECTION'):
            started = True
        elif started and line.strip().isdigit():
            loc = int(line.strip())
            if loc == -1:
                break
            tour.append(loc)
    tour = np.array(tour).astype(int)
    # Convert tour to 0-based index
    cost = cost / par_args.int_coord_scale

    all_node_xy = [depot] + locs
    modified_tour = np.array(tour).astype(int) - 1
    modified_tour[modified_tour > scale] = 0
    final_tour = np.append(modified_tour, 0)
    final_tour = np.array(final_tour).astype(int)

    # Recalculate cost to preserve more decimal places
    solution = final_tour
    node_xy_tensor = torch.tensor(all_node_xy, dtype=torch.float32)
    node_xy_tensor = node_xy_tensor.unsqueeze(0)
    sequence_tensor = torch.tensor(solution, dtype=torch.int64)

    # node_xy.shape=(1, node_num,2), sequence.shape=(m,)
    problem = node_xy_tensor[0]
    # shape: (node_num, 2)
    gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
    # shape: (m, 2)
    ordered_seq = problem.gather(dim=0, index=gathering_index)
    # shape:(m,2)
    rolled_seq = ordered_seq.roll(dims=0, shifts=-1)

    manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())

    # Save as .txt file
    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{scale}_c{capacity}.txt"
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:

            f.write("depot_xy:")
            coord = depot
            coord[0] = coord[0]
            coord[1] = coord[1]
            f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            f.write("node_xy:")
            for coord in locs:
                coord[0] = coord[0]
                coord[1] = coord[1]
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")

            f.write("node_demand:")
            for demand in demands:
                f.write(f" {demand}")
            f.write("\n")
            f.write(f"capacity: {capacity}\n")

            f.write(f"scale: {scale}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            f.write("solution: ")
            f.write(" ".join(map(str, final_tour)))  # Return to start point
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")

            f.write("LKH settings:\n")
            for key, value in params_dict.items():
                f.write(f"{key}: {value}\n")

    return final_tour, manual_calculate_cost


def eax_solution(directory, node_coords, scale, distribution, attributes, duration, EAXargs, problem_index):

    # Check if the directory contains the required solution file
    result_file = os.path.join(directory, "problem_BestSol")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"The file {result_file} does not exist.")

    # Read the solution file
    with open(result_file, 'r') as f:
        lines = f.readlines()

        # Initialize variables to store the minimum cost and corresponding tour
        min_cost = float('inf')
        best_tour = None

        # Iterate through the lines in pairs
        for i in range(0, len(lines)-1, 2):
            # Check if there are at least two more lines
            if i + 1 >= len(lines):
                break

            # Parse cost from the current line
            parts = lines[i].strip().split()
            if len(parts) < 2:
                continue  # Skip lines that don't have enough elements

            try:
                cost = int(parts[1])
            except ValueError:
                continue  # Skip lines where conversion to int fails

            if cost < min_cost:
                min_cost = cost
                # Parse tour from the next line
                tour = list(map(int, lines[i + 1].strip().split()))
                best_tour = tour

                # Check if a valid solution was found
            if best_tour is None:
                raise ValueError("No valid solution with odd cost found.")

    # Recalculate cost to preserve more decimal places
    solution = [x - 1 for x in best_tour]
    node_xy_tensor = torch.tensor(node_coords, dtype=torch.float32)
    node_xy_tensor = node_xy_tensor.unsqueeze(0)
    sequence_tensor = torch.tensor(solution, dtype=torch.int64)

    # node_xy.shape=(1, node_num,2), sequence.shape=(m,)
    problem = node_xy_tensor[0]
    # shape: (node_num, 2)
    gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
    # shape: (m, 2)
    ordered_seq = problem.gather(dim=0, index=gathering_index)
    # shape:(m,2)
    rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
    manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())

    # Save as .txt file
    txt_filename = f"{EAXargs.ptype}_num{problem_index}_{distribution}_node{scale}.txt"
    if EAXargs.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)
        with open(file_path_txt, 'w') as f:
            f.write("node_xy:")
            for coord in node_coords:
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")

            f.write(f"scale: {scale}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            f.write("solution: ")
            f.write(" ".join(map(str, best_tour)))  # Return to start point
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("EAX settings:\n")
            f.write(f"trials: {EAXargs.trials}\n")
            f.write(f"population: {EAXargs.population}\n")
            f.write(f"offspring: {EAXargs.offspring}\n")

    return best_tour, manual_calculate_cost


def hgs_solution(directory, depot, locs, demands, capacity, scale, distribution, attributes, duration, par_args, problem_index = None):
    # Read the output.tour file
    filename = os.path.join(directory, 'output.tour')
    with open(filename, 'r') as f:
        lines = f.readlines()

    cost = None
    routes = []
    for line in lines:
        if line.startswith('Cost'):
            cost = float(line.split()[1])
        elif line.startswith('Route #'):
            route = [int(x) for x in line.strip().split(':')[1].split()]
            routes.append(route)
    cost = cost / par_args.int_coord_scale

    all_node_xy = [depot] + locs
    final_tour = []
    for i, route in enumerate(routes):
        final_tour.extend(route)
        if i < len(routes) - 1:
            final_tour.append(0)
    final_tour = [0] + final_tour + [0]
    final_tour = np.array(final_tour).astype(int)
    # Recalculate cost to preserve more decimal places
    solution = final_tour
    node_xy_tensor = torch.tensor(all_node_xy, dtype=torch.float32)
    node_xy_tensor = node_xy_tensor.unsqueeze(0)
    sequence_tensor = torch.tensor(solution, dtype=torch.int64)

    # node_xy.shape=(1, node_num,2), sequence.shape=(m,)
    problem = node_xy_tensor[0]
    # shape: (node_num, 2)
    gathering_index = torch.Tensor(sequence_tensor).long()[:, None].expand(-1, 2)
    # shape: (m, 2)
    ordered_seq = problem.gather(dim=0, index=gathering_index)
    # shape:(m,2)
    rolled_seq = ordered_seq.roll(dims=0, shifts=-1)
    manual_calculate_cost = float(((ordered_seq - rolled_seq) ** 2).sum(1).sqrt().sum().numpy())

    txt_filename = f"{par_args.ptype}_num{problem_index}_{distribution}_node{scale}_c{capacity}.txt"
    if par_args.save_as_txt:
        file_path_txt = os.path.join(directory, txt_filename)

        with open(file_path_txt, 'w') as f:

            f.write("depot_xy:")
            coord = depot
            coord[0] = coord[0]
            coord[1] = coord[1]
            f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            f.write("node_xy:")
            for coord in locs:
                coord[0] = coord[0]
                coord[1] = coord[1]
                f.write(f" {coord[0]} {coord[1]}")
            f.write("\n")
            f.write("node_demand:")
            for demand in demands:
                f.write(f" {demand}")
            f.write("\n")
            f.write(f"capacity: {capacity}\n")

            f.write(f"scale: {scale}\n")

            f.write(f"distribution: {distribution}\n")
            f.write(f"attributes: {attributes}\n")

            f.write("solution: ")
            f.write("0 ")
            for idx, route in enumerate(routes):
                f.write(" ".join(map(str, route)))  # Return to start point
                f.write(" 0 ")
            f.write("\n")
            f.write(f"cost: {manual_calculate_cost}\n")
            f.write(f"time: {duration} s\n")
            f.write("HGS settings: \n")
            f.write(f"time_threshold: {par_args.time_threshold}s\n")
            f.write(f"seed: {par_args.SEED}\n")
            f.write(f"\n")
    return final_tour, manual_calculate_cost

