import argparse

"""
This module provides functionality for parsing command-line arguments for various optimization solvers.

The function returns an object containing the parsed arguments, which can be used to configure and run
the solvers with the specified parameters.
"""


def get_options(args=None):
    parser = argparse.ArgumentParser()
    # solver setting
    parser.add_argument('--solver', type=str, choices=['lkh', 'hgs', 'eax', 'ortools', 'pyvrp', 'cplex', 'gurobi'], default='gurobi',
                        help="Name of the solver to evaluate")
    # LKH parameters
    parser.add_argument('--SPECIAL', default=None,
                        help="LKH: The params for LKH")
    parser.add_argument('--MAX_TRIALS', type=int, default=1000,
                        help="LKH: The number of iterations for LKH")
    parser.add_argument('--RUNS', type=int, default=1,
                        help="LKH: The number of times a problem is run")
    parser.add_argument('--SEED', type=int, default=1234,
                        help="Generate seeds for random numbers")
    parser.add_argument('--TRACE_LEVEL', type=int, default=1,
                        help="Log information")
    parser.add_argument("--CANDIDATE_SET_TYPE", type=str, choices=['POPMUSIC', 'ALPHA', 'DELAUNAY', 'NEAREST-NEIGHBOR'],
                        default='ALPHA',
                        help="LKH: Accelerate algorithm")
    # HGS parameters
    parser.add_argument('--time_threshold', type=int, default=10,
                        help="HGS: The running time of HGS")
    # EAX parameters
    parser.add_argument("--trials", type=int, default=3,
                        help="EAX: The number of trials for EAX")
    parser.add_argument("--population", type=int, default=300,
                        help="EAX: The count of candidate solutions maintained per generational iteration")
    parser.add_argument("--offspring", type=int, default=30,
                        help="EAX: The number of offspring for EAX")
    # OR-Tools parameters
    parser.add_argument("--ortools_num_vehicles", type=int, default=6,
                        help="ortools: number of vehicles")
    parser.add_argument("--ortools_time_limit", type=int, default=10,
                        help="time limit in seconds for ortools")
    # PyVRP parameters
    parser.add_argument("--pyvrp_num_vehicles", type=int, default=10,
                        help="pyvrp: number of vehicles")
    parser.add_argument("--pyvrp_stop", type=str, choices=['MaxRuntime', 'MaxIterations'], default='MaxRuntime',
                        help="pyvrp: stop limit")
    parser.add_argument("--pyvrp_MaxRuntime", type=int, default=10,
                        help="pyvrp: run time limit")
    parser.add_argument("--pyvrp_MaxIterations", type=int, default=10,
                        help="pyvrp: max iterations number")
    # cplex parameters
    parser.add_argument("--cplex_num_vehicles", type=int, default=6,
                        help="number of vehicles for cplex (CVRP)")
    parser.add_argument("--cplex_time_limit", type=int, default=10,
                        help="time limit in seconds for cplex")
    parser.add_argument("--cplex_gap", type=float, default=0.01,
                        help="MIP gap for cplex, e.g")
    parser.add_argument("--cplex_threads", type=int, default=2,
                        help="number of threads for cplex, 0 means using all available threads")

    # gurobi parameters
    parser.add_argument("--gurobi_num_vehicles", type=int, default=6,
                        help="number of vehicles for gurobi (CVRP)")
    parser.add_argument("--gurobi_time_limit", type=int, default=10,
                        help="time limit in seconds for gurobi")
    parser.add_argument("--gurobi_gap", type=float, default=0.01,
                        help="MIP gap for gurobi, e.g")
    parser.add_argument("--gurobi_threads", type=int, default=2,
                        help="number of threads for gurobi, 0 means using all available threads")
    

    # problem setting
    parser.add_argument('--ptype', type=str, choices=['tsp', 'cvrp', 'atsp'], default='tsp',
                        help="Problem type")
    parser.add_argument('--problem_number', type=int, default=10,
                        help="Number of instances")
    parser.add_argument('--problem_scale', type=int, default=20,
                        help="Nodes number of each instance")
    parser.add_argument('--batch_size', type=int, default=5,
                        help="Instance number of each batch")
    parser.add_argument('--int_coord_scale', type=int, default=100,
                        help="Expand coordinate values, transform coordinates into integers")
    parser.add_argument('--int_matrix_scale', type=int, default=100,
                        help="For ATSP problem: Expand edge weight values, transform edge weight into integers")
    parser.add_argument('--demand_scale', type=int, default=1,
                        help="Demand range: demand/demand_scale")
    parser.add_argument('--capacity', type=int, default=50,
                        help="vehicle capacity, it is only used in CVRP")


    # working setting
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help="device")
    parser.add_argument('--save_as_txt', default=True,
                        help="Convert to TXT file after solving")
    parser.add_argument("--solver_mode", default="multi", choices=['single', 'multi'], 
                        help="'single' means solving one instance each time, 'multi' means solving a batch of instances each time")
    parser.add_argument("--cpus", type=int, default=4,
                        help="multiprocess，cpu number")
    parser.add_argument("--delete", default=False,
                        help="delete the lib file, including .par,.tsp,.vrp....")
    if args is None:
        args = []  # Default to an empty list; do not read command-line arguments

    opts = parser.parse_args(args)
    return opts
