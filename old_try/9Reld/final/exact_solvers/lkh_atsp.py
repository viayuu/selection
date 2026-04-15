import os
import subprocess
from urllib.parse import urlparse
import requests
import tarfile
import torch
from time import time
from EasyNCO.data import TSPGenerator
from EasyNCO.data.LIBUtils import ATSPLIBWriter
import shutil
import re
import numpy as np
from EasyNCO.exact_solvers.solution import lkh_atsp_solution
from EasyNCO.data.lkh_params_utils import write_lkh_atsp_params
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

        # Use requests to download
        response = requests.get(url)
        with open(file, 'wb') as f:
            f.write(response.content)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        subprocess.check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)



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
    # logger.info(f"Running command: {command}")
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


def lkh_atsp_solver_multiprocess(data_list, par_args):
    '''
        Multiprocess the LKH solver on a given dataset of ATSP problems.
        Parameters:
        - data_loader: A DataLoader object that yields batches of ATSP problem instances.
        - par_args: An object containing arguments for the LKH algorithm.
        Return:
        -result: [(tour_1, cost_1),(tour_2, cost_2),......(tour_n,cost_n)]
    '''
    lkh_executable = get_lkh_executable()
    problem_index = 0
    solver_task_param_list = list()

    for instance in data_list:

        node_matrix = instance

        solver_task_param = (node_matrix, par_args, lkh_executable, problem_index)
        solver_task_param_list.append(solver_task_param)

        problem_index = problem_index + 1

    result = multiprocess(lkh_atsp_solver_base, solver_task_param_list, par_args.cpus)
    return result


def lkh_atsp_solver(node_matrix, par_args, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a single ATSP problem using the LKH algorithm.
        Parameters:
        - node_coords: A list of node coordinates.
        - par_args: An object containing arguments for the LKH algorithm.
        - problem_index: it is used in multiprocess
        Returns:
        - A tuple containing the tour data and the total cost.
    '''
    # Use temporary file, return optimal tour, cost
    lkh_executable = get_lkh_executable()
    tour, cost = lkh_atsp_solver_base(node_matrix, par_args, lkh_executable, problem_index, distribution, attributes)
    return tour, cost

def lkh_atsp_solver_base(node_matrix, par_args, lkh_executable, problem_index = None, distribution = None, attributes = None):
    '''
        Solves a single ATSP problem using the LKH algorithm.
        Parameters:
        - node_coords: A list of node coordinates.
        - par_args: An object containing arguments for the LKH algorithm.
        - problem_index: it is used in multiprocess
        Returns:
        - A tuple containing the tour data and the total cost.
    '''
    # Use temporary file, return optimal tour, cost
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "results_lkh_atsp")
    os.makedirs(result_dir, exist_ok=True)
    directory = os.path.join(result_dir, str(problem_index))
    os.makedirs(directory, exist_ok=True)

    ptype = par_args.ptype
    int_matrix_scale = par_args.int_matrix_scale
    scale = len(node_matrix)

    ATSPLIBWriter(directory, node_matrix, int_matrix_scale, ptype, par_args)
    write_lkh_atsp_params(directory, par_args)
    duration = run_lkh_cmd(directory, lkh_executable)
    tour, cost = lkh_atsp_solution(directory, node_matrix, scale, distribution, attributes, duration, par_args, problem_index)
    # Delete temporary file_dir
    if par_args.delete:
        shutil.rmtree(directory)
    return tour, cost
