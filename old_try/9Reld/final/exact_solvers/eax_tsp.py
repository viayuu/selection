import os
import subprocess
from urllib.parse import urlparse
import requests
import zipfile
import torch
from EasyNCO.data import TSPGenerator
from EasyNCO.data.LIBUtils import TSPLIBWriter
import shutil
import numpy as np
from time import time
from EasyNCO.exact_solvers.solution import eax_solution
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)
from multiprocessing import Pool

def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))

def get_ga_eax_executable(url="https://github.com/nagata-yuichi/GA-EAX/archive/refs/heads/main.zip"):
    '''
        Downloads and compiles the GA-EAX executable from a given URL.
        If the executable already exists, it returns the path to the executable.
        Parameters:
        - url: The URL from which to download the GA-EAX source code.
        Returns:
        - The path to the compiled GA-EAX executable.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.join(current_dir, "use_exe/ga_eax")
    os.makedirs(cwd, exist_ok=True)
    filedir = os.path.join(cwd, "GA-EAX-main", "GA_EAX_1.0", "Normal")
    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    if not os.path.isdir(os.path.join(cwd, "GA-EAX-main", "GA_EAX_1.0", "Normal")):
        logger.info("GA-EAX not found, downloading and compiling")

        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {file}")

            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(cwd)
            logger.info(f"Extracted to {cwd}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading the file: {e}")
            return None
        try:
            subprocess.check_call(["g++", "-o", "jikken", "-O3", "main.cpp", "env.cpp", "cross.cpp", "evaluator.cpp", "indi.cpp", "rand.cpp", "kopt.cpp", "sort.cpp", "-lm"], cwd=filedir)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile EAX: {e}")
            return None
    cwd_path = os.getcwd()
    executable = os.path.join(filedir, "jikken")
    executable_rel = os.path.relpath(executable, cwd_path)
    assert os.path.isfile(executable), "Executable file does not exist: {}".format(executable)
    return executable_rel


def run_eax_cmd(directory, eax_executable, EAXargs):
    '''
        Runs the GA-EAX algorithm using the provided command-line arguments.
        Parameters:
        - directory: The directory where the TSP problem and result files are located.
        - EAXargs: An object containing arguments for the GA-EAX algorithm.
    '''

    if eax_executable is None:
        logger.error("Failed to get GA-EAX executable path")
        return

    results_dir = os.path.join(directory)
    os.makedirs(results_dir, exist_ok=True)

    eax_cmd = [
        eax_executable,
        str(EAXargs.trials),
        os.path.join(results_dir, "problem"),
        str(EAXargs.population),
        str(EAXargs.offspring),
        os.path.join(directory, 'problem.tsp'),
    ]
    start = time()
    # logger.info(f"Running command: {eax_cmd}")
    parent_dir = os.path.dirname(os.path.abspath(directory))
    try:
        subprocess.check_call(eax_cmd)
    except Exception as e:
        logger.info(f"error:{e}")
        logger.info(f"location:{directory}")
        txt_filename = "false.txt"
        file_path_txt = os.path.join(parent_dir, txt_filename)
        with open(file_path_txt, 'a') as f:
            f.write(f"error:{e}\n")
            f.write(f"location:{directory}\n")

    duration = time() - start
    return duration

def eax_solver_multiprocess(data_list, EAXargs):
    '''
        Multiprocess the GA-EAX solver on a given dataset of TSP problems.
        Parameters:
        - data_loader: A DataLoader object that yields batches of TSP problem instances.
        - EAXargs: An object containing arguments for the GA-EAX algorithm.
    '''
    eax_executable = get_ga_eax_executable()
    problem_index = 0
    solver_task_param_list = list()

    for instance in data_list:
        node_coords = [(xy[0], xy[1]) for xy in instance]

        solver_task_param = (node_coords, EAXargs, eax_executable, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1
    result = multiprocess(eax_solver_base, solver_task_param_list, EAXargs.cpus)
    return result


def eax_solver(node_coords, EAXargs, problem_index = None, distribution = None, attributes = None):
    '''
        Runs the GA-EAX algorithm on a single TSP problem instance.
        Parameters:
        - node_coords: A list of node coordinates for the TSP problem.
        - EAXargs: An object containing arguments for the GA-EAX algorithm.
        - problem_index: it is used in multiprocess
        Returns:
        - The tour and cost of the best solution found by the GA-EAX algorithm.
    '''
    eax_executable = get_ga_eax_executable()
    tour, cost = eax_solver_base(node_coords, EAXargs, eax_executable, problem_index, distribution, attributes)
    return tour, cost

def eax_solver_base(node_coords, EAXargs, eax_executable, problem_index = None, distribution = None, attributes = None):
    cwd_path = os.getcwd()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "results_eax_tsp")
    os.makedirs(result_dir, exist_ok=True)
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)
    # In the problem.Bestsol file, the new results do not overwrite the historical results, so the historial file needs to be deleted
    if os.path.isdir(os.path.join(directory)):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

    int_coord_scale = EAXargs.int_coord_scale
    ptype = EAXargs.ptype
    scale = len(node_coords)
    # Generate the TSPLIB format problem file
    TSPLIBWriter(directory, node_coords, int_coord_scale, ptype, EAXargs)
    # convert absolute path to relative path
    directory_rel = os.path.relpath(directory, cwd_path)
    # Run the EAX algorithm
    duration = run_eax_cmd(directory_rel, eax_executable, EAXargs)
    # Read the result from the EAX algorithm
    tour, cost = eax_solution(directory, node_coords, scale, distribution, attributes, duration, EAXargs, problem_index)
    # Delete temporary file_dir
    if EAXargs.delete:
        shutil.rmtree(directory)
    return tour, cost