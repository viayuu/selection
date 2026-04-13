import os


def write_lkh_tsp_params(directory, par_args):
    '''
        Writes the LKH parameters for a TSP problem into a file named params.par.
        The file includes various settings for the LKH algorithm.
        Parameters:
        - directory: The directory where the params.par file will be saved.
        - par_args: An object containing parameters for LKH.
    '''
    param_path = os.path.join(directory, 'params.par')
    output_path = os.path.join(directory, 'output.tour')
    problem_path = os.path.join(directory, 'problem.tsp')
    default_parameters = {
        "MAX_TRIALS": 10000,
        "RUNS": 1,
        "TRACE_LEVEL": 1,
        "SEED": 0,
        "PROBLEM_FILE": problem_path,
        "TOUR_FILE": output_path,
        "CANDIDATE_SET_TYPE": 'ALPHA'
    }
    if par_args is not None:
        parameters = {
            "MAX_TRIALS": par_args.MAX_TRIALS,
            "RUNS": par_args.RUNS,
            "TRACE_LEVEL": par_args.TRACE_LEVEL,
            "SEED": par_args.SEED,
            "CANDIDATE_SET_TYPE": par_args.CANDIDATE_SET_TYPE
        }
    else:
        parameters = default_parameters

    with open(param_path, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write(f"{k}\n")
            else:
                f.write(f"{k} = {v}\n")

def write_lkh_atsp_params(directory, par_args):
    '''
        Writes the LKH parameters for a ATSP problem into a file named params.par.
        The file includes various settings for the LKH algorithm.
        Parameters:
        - directory: The directory where the params.par file will be saved.
        - par_args: An object containing parameters for LKH.
    '''
    param_path = os.path.join(directory, 'params.par')
    output_path = os.path.join(directory, 'output.tour')
    problem_path = os.path.join(directory, 'problem.atsp')
    default_parameters = {
        "MAX_TRIALS": 10000,
        "RUNS": 1,
        "TRACE_LEVEL": 1,
        "SEED": 0,
        "PROBLEM_FILE": problem_path,
        "TOUR_FILE": output_path,
        "CANDIDATE_SET_TYPE": 'ALPHA'
    }
    if par_args is not None:
        parameters = {
            "MAX_TRIALS": par_args.MAX_TRIALS,
            "RUNS": par_args.RUNS,
            "TRACE_LEVEL": par_args.TRACE_LEVEL,
            "SEED": par_args.SEED,
            "CANDIDATE_SET_TYPE": par_args.CANDIDATE_SET_TYPE
        }
    else:
        parameters = default_parameters

    with open(param_path, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write(f"{k}\n")
            else:
                f.write(f"{k} = {v}\n")

def write_lkh_cvrp_params(directory, par_args):
    '''
        Writes the LKH parameters for a CVRP problem into a file named params.par.
        The file includes various settings for the LKH algorithm.
        Parameters:
        - directory: The directory where the params.par file will be saved.
        - par_args: An object containing parameters for LKH.
    '''
    param_path = os.path.join(directory, 'params.par')
    output_path = os.path.join(directory, 'output.tour')
    problem_path = os.path.join(directory, 'problem.vrp')
    init_tour_path = os.path.join(directory, 'input.tour')
    default_parameters = {
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 1,
        "TRACE_LEVEL": 1,
        "SEED": 0,
        "PROBLEM_FILE": problem_path,
        "OUTPUT_TOUR_FILE": output_path,
        "CANDIDATE_SET_TYPE": 'ALPHA'
    }
    if par_args is not None:
        parameters = {
            "SPECIAL": par_args.SPECIAL,
            "MAX_TRIALS": par_args.MAX_TRIALS,
            "RUNS": par_args.RUNS,
            "TRACE_LEVEL": par_args.TRACE_LEVEL,
            "SEED": par_args.SEED,
            "CANDIDATE_SET_TYPE": par_args.CANDIDATE_SET_TYPE
        }
    else:
        parameters = default_parameters

    with open(param_path, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write(f"{k}\n")
            else:
                f.write(f"{k} = {v}\n")


def read_lkh_params(directory):
    params_file = os.path.join(directory, 'params.par')
    with open(params_file, 'r') as f:
        params_lines = f.readlines()

# Parse parameter information
    params_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in params_lines if '=' in line}

    return params_dict

