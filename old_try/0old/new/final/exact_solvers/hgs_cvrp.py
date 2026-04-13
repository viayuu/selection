import os
import sys
import subprocess
from urllib.parse import urlparse
import numpy as np
from time import time
from EasyNCO.data import CVRPGenerator
from EasyNCO.data.LIBUtils import CVRPLIBWriter
import shutil
from EasyNCO.exact_solvers.solution import hgs_solution
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

from multiprocessing import Pool

def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))

def get_hgs_executable(url="https://github.com/vidalt/HGS-CVRP.git"):
    '''
        Downloads and compiles the HGS executable from a given GitHub repository.
        If the executable already exists, it returns the path to the executable.
        Parameters:
        - url: The URL of the HGS repository.
        Returns:
        - The path to the compiled HGS executable.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.join(current_dir, "use_exe/hgs")
    os.makedirs(cwd, exist_ok=True)
    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        logger.info("{} not found, downloading and compiling".format(filedir))

        try:
            subprocess.check_call(f"git clone {url}", cwd=cwd, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return None

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        buildbuild = os.path.abspath(os.path.join(filedir, "build"))
        os.makedirs(buildbuild, exist_ok=True)

        try:
            subprocess.check_call("cmake .. -DCMAKE_BUILD_TYPE=Release -G 'Unix Makefiles'", cwd=os.path.join(filedir, "build"), shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile HGS: {e}")
            return None

        try:
            subprocess.check_call("make bin", cwd=os.path.join(filedir, "build"), shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile HGS: {e}")
            return None

    executable = os.path.join(os.path.join(filedir, "build"), "hgs")
    assert os.path.isfile(executable), f'Cannot find HGS executable file at {executable}'
    return os.path.abspath(executable)




def run_hgs_cmd(directory, time_threshold, hgs_executable, seed=0):
    '''
        Runs the HGS algorithm with the specified command and parameters.
        Parameters:
        - directory: The directory containing the problem file.
        - time_threshold: The maximum time allowed for HGS to run.
        - seed: The random seed for HGS.
        Returns:
        - A function that runs HGS and returns the output.
    '''

    problem_path = os.path.join(directory, 'problem.vrp')
    output_path = os.path.join(directory, 'output.tour')
    
    command = [hgs_executable, problem_path, output_path, '-seed', f'{seed}', '-t', f'{time_threshold}']
    # logger.info(f"Running command: {command}")
    log_path = os.path.join(directory, 'output.log')
    parent_dir = os.path.dirname(os.path.abspath(directory))
    start = time()
    try:
        with open(log_path, 'w') as f:
            subprocess.check_call(command, stdout=f, stderr=f)
    except Exception as e:
        logger.info(f"error:{e}")
        logger.info(f"location:{directory}")
        txt_filename = "false.txt"
        file_path_txt = os.path.join(parent_dir, txt_filename)
        with open(file_path_txt, 'a') as f:
            f.write(f"error:{e}\n")
            f.write(f"location:{directory}\n")        
    duration = time() - start
    # logger.info(f"HGS execution time: {duration:.2f} seconds")
    return duration



def hgs_solver_multiprocess(data_list, par_args):
    '''
        Multiprocess the HGS solver on a given dataset of CVRP problems.
        Parameters:
        - data_loader: A DataLoader object that yields batches of CVRP problem instances.
        - par_args: An object containing arguments for the HGS algorithm.
        Return:
        -result: [(tour_1, cost_1),(tour_2, cost_2),......(tour_n,cost_n)]
    '''
    hgs_executable = get_hgs_executable()
    problem_index = 0
    solver_task_param_list = list()

    for instance in data_list:

        depot = instance[0]
        locs = instance[1]
        demand = instance[2]
        capacity = instance[3]

        solver_task_param = (depot, locs, demand, capacity, par_args, hgs_executable, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1

    result = multiprocess(hgs_solver_base, solver_task_param_list, par_args.cpus)
    return result


def hgs_solver(depot, locs, demands, capacity, par_args, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a single CVRP problem using the HGS algorithm.
        Parameters:
        - depot: The coordinates of the depot.
        - locs: A list of node coordinates.
        - demands: A list of node demands.
        - capacity: The capacity of the vehicle.
        - par_args: An object containing arguments for the HGS algorithm.
        - problem_index: it is used in multiprocess
        - distribution: Distribution of nodes
        - attributes: Distribution attributes set during data generation
        Returns:
        - A tuple containing the tour data and the total cost.
    '''
    hgs_executable = get_hgs_executable()
    tour, cost = hgs_solver_base(depot, locs, demands, capacity, par_args, hgs_executable, problem_index, distribution, attributes)
    return tour, cost
    

def hgs_solver_base(depot, locs, demands, capacity, par_args, hgs_executable, problem_index = None, distribution = None, attributes = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(current_dir, "results_hgs_cvrp")
    os.makedirs(directory, exist_ok=True)
    directory = os.path.join(directory, str(problem_index))
    os.makedirs(directory, exist_ok=True)

    int_coord_scale = par_args.int_coord_scale
    ptype = par_args.ptype
    demand_scale = par_args.demand_scale
    time_threshold = par_args.time_threshold
    SEED = par_args.SEED
    scale = len(locs)

    CVRPLIBWriter(directory, depot, locs, demands, capacity, int_coord_scale, demand_scale, ptype, par_args)
    duration = run_hgs_cmd(directory, time_threshold, hgs_executable, SEED)
    tour, cost = hgs_solution(directory, depot, locs, demands, capacity, scale, distribution, attributes, duration, par_args, problem_index)
    # Delete temporary file_dir
    if par_args.delete:
        shutil.rmtree(directory)
    return tour, cost