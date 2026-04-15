import os
import subprocess
from subprocess import check_call, check_output
from urllib.parse import urlparse
import requests
import tarfile
import torch
from time import time
from EasyNCO.data import CVRPGenerator
from EasyNCO.data.LIBUtils import CVRPLIBWriter
import shutil
import re
import numpy as np
from EasyNCO.exact_solvers.solution import lkh_cvrp_solution
from EasyNCO.data.lkh_params_utils import write_lkh_cvrp_params
from EasyNCO.utils.utils import getLogger
logger = getLogger(__name__)

from multiprocessing import Pool

def multiprocess(func, tasks, cpus=None):
    if cpus == 1 or len(tasks) == 1:
        return [func(*t) for t in tasks]
    with Pool(cpus or os.cpu_count()) as pool:
        return list(pool.starmap(func, tasks))


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.4.tgz"):
    '''
        Downloads and compiles the LKH executable from a given URL.
        If the executable already exists, it returns the path to the executable.
        Parameters:
        - url: The URL from which to download the LKH source code.
        Returns:
        - The path to the compiled LKH executable.
    '''

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.join(current_dir, "use_exe/lkh")
    os.makedirs(cwd, exist_ok=True)
    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        logger.info("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable), f'Cannot find LKH-3 executable file at {executable}'
    return os.path.abspath(executable)

def write_lkh_tours(tours, n_jobs, directory, cost=None):
    init_tour_path = os.path.join(directory, 'input.tour')
    n_veh = len(tours)
    with open(init_tour_path, 'w') as f:
        f.write('TYPE : TOUR\n')
        f.write('DIMENSION : {}\n'.format(n_jobs + n_veh))
        f.write('TOUR_SECTION\n')
        for tour_idx, tour in enumerate(tours):  # 1: depot, 2 to n_jobs + 1: jobs, n_jobs + 1+1 - : depot
            f.write('{}\n'.format(1 if tour_idx == 0 else n_jobs + n_veh + 1 - tour_idx))
            for a in tour:  # depot starts at 1 (originally 0), jobs start at 2 (originally 1)
                f.write('{}\n'.format(a + 1))
        f.write('-1\n')
        f.write('EOF')


def run_lkh_cmd(directory, executable=None):
    '''
        Runs the LKH algorithm with the specified command and parameters.
        Parameters:
        - directory: The directory containing the LKH parameter file.
        Returns:
        - A function that runs LKH and logs the execution time.
    '''
    param_path = os.path.join(directory, 'params.par')
    log_path = os.path.join(directory, 'output.log')
    command = [executable, param_path]
    #logger.info(f"Running command: {command}")
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
    # logger.info(f"LKH execution time: {duration:.2f} seconds")
    return duration

def lkh_cvrp_solver_multiprocess(data_list, par_args):
    '''
        Multiprocess the LKH solver on a given dataset of CVRP problems.
        Parameters:
        - data_loader: A DataLoader object that yields batches of CVRP problem instances.
        - par_args: An object containing arguments for the LKH algorithm.
        Return:
        -result: [(tour_1, cost_1),(tour_2, cost_2),......(tour_n,cost_n)]
    '''
    lkh_executable = get_lkh_executable()
    problem_index = 0

    solver_task_param_list = list()


    for instance in data_list:

        depot = instance[0]
        locs = instance[1]
        demand = instance[2]
        capacity = instance[3]

        solver_task_param = (depot, locs, demand, capacity, par_args, lkh_executable, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1

    result = multiprocess(lkh_cvrp_solver_base, solver_task_param_list, par_args.cpus)
    return result


def lkh_cvrp_solver(depot, locs, demands, capacity, par_args, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a single CVRP problem using the LKH algorithm.
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
    lkh_executable = get_lkh_executable()
    tour, cost = lkh_cvrp_solver_base(depot, locs, demands, capacity, par_args, lkh_executable, problem_index, distribution, attributes)
    return tour, cost

def lkh_cvrp_solver_base(depot, locs, demands, capacity, par_args, lkh_executable, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a batch of CVRP problems using the LKH algorithm.
        Parameters:
        - data_loader: A DataLoader object that yields batches of CVRP problem instances.
        - par_args: An object containing arguments for the LKH algorithm.
        Returns:
        - A list of tuples, each containing the tour data and total cost for a CVRP problem.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "results_lkh_cvrp")
    os.makedirs(result_dir, exist_ok=True)
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)

    int_coord_scale = par_args.int_coord_scale
    ptype = par_args.ptype
    demand_scale = par_args.demand_scale
    scale = len(locs)

    CVRPLIBWriter(directory, depot, locs, demands, capacity, int_coord_scale, demand_scale, ptype, par_args)
    write_lkh_cvrp_params(directory, par_args)
    duration = run_lkh_cmd(directory, lkh_executable)
    tour, cost = lkh_cvrp_solution(directory, depot, locs, demands, capacity, scale, distribution, attributes, duration, par_args, problem_index)
    # Delete temporary file_dir
    if par_args.delete:
        shutil.rmtree(directory)
    return tour, cost