from EasyNCO.exact_solvers.hgs_cvrp import hgs_solver_multiprocess
from EasyNCO.exact_solvers.lkh_cvrp import lkh_cvrp_solver_multiprocess
from EasyNCO.exact_solvers.lkh_tsp import lkh_tsp_solver_multiprocess
from EasyNCO.exact_solvers.lkh_atsp import lkh_atsp_solver_multiprocess
from EasyNCO.exact_solvers.eax_tsp import eax_solver_multiprocess
from EasyNCO.exact_solvers.ortools_tsp import ortools_tsp_solver_multiprocess
from EasyNCO.exact_solvers.ortools_cvrp import ortools_cvrp_solver_multiprocess
from EasyNCO.exact_solvers.pyvrp_run import pyvrp_solver_multiprocess
from EasyNCO.exact_solvers.cplex_tsp import cplex_tsp_solver_multiprocess
from EasyNCO.exact_solvers.cplex_cvrp import cplex_cvrp_solver_multiprocess
from EasyNCO.exact_solvers.gurobi_tsp import gurobi_tsp_solver_multiprocess
from EasyNCO.exact_solvers.gurobi_cvrp import gurobi_cvrp_solver_multiprocess

from EasyNCO.exact_solvers.hgs_cvrp import hgs_solver
from EasyNCO.exact_solvers.lkh_cvrp import lkh_cvrp_solver
from EasyNCO.exact_solvers.lkh_tsp import lkh_tsp_solver
from EasyNCO.exact_solvers.lkh_atsp import lkh_atsp_solver
from EasyNCO.exact_solvers.eax_tsp import eax_solver
from EasyNCO.exact_solvers.ortools_tsp import ortools_tsp_solver
from EasyNCO.exact_solvers.ortools_cvrp import ortools_cvrp_solver
from EasyNCO.exact_solvers.pyvrp_run import pyvrp_solver
from EasyNCO.exact_solvers.cplex_tsp import cplex_tsp_solver
from EasyNCO.exact_solvers.cplex_cvrp import cplex_cvrp_solver
from EasyNCO.exact_solvers.gurobi_tsp import gurobi_tsp_solver
from EasyNCO.exact_solvers.gurobi_cvrp import gurobi_cvrp_solver


def exact_solver_main(cfg, data_list=None, node_coords=None, depot=None, locs=None, demand=None, capacity=None, node_matrix=None):
    if cfg.solver_mode == 'multi' and data_list is not None:

        if cfg.ptype == 'tsp':
            if cfg.solver == 'lkh':
                result = lkh_tsp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'eax':
                result = eax_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'ortools':
                result = ortools_tsp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'cplex':
                result = cplex_tsp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'gurobi':
                result = gurobi_tsp_solver_multiprocess(data_list, cfg)

        elif cfg.ptype == 'cvrp':
            if cfg.solver == 'lkh':
                result = lkh_cvrp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'hgs':
                result = hgs_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'ortools':
                result = ortools_cvrp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'pyvrp':
                result = pyvrp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'cplex':
                result = cplex_cvrp_solver_multiprocess(data_list, cfg)
            elif cfg.solver == 'gurobi':
                result = gurobi_cvrp_solver_multiprocess(data_list, cfg)
        elif cfg.ptype == 'atsp':
            if cfg.solver == 'lkh':
                result = lkh_atsp_solver_multiprocess(data_list, cfg)
        return result

    elif cfg.solver_mode == 'single':
        if cfg.ptype == 'tsp':
            if cfg.solver == 'lkh':
                tour, cost = lkh_tsp_solver(node_coords, cfg)
            elif cfg.solver == 'eax':
                tour, cost = eax_solver(node_coords, cfg)
            elif cfg.solver == 'ortools':
                tour, cost = ortools_tsp_solver(node_coords, cfg)
            elif cfg.solver == 'cplex':
                tour, cost = cplex_tsp_solver(node_coords, cfg)
            elif cfg.solver == 'gurobi':
                tour, cost = gurobi_tsp_solver(node_coords, cfg)

        elif cfg.ptype == 'cvrp':
            if cfg.solver == 'lkh':
                tour, cost = lkh_cvrp_solver(depot, locs, demand, capacity, cfg)
            elif cfg.solver == 'hgs':
                tour, cost = hgs_solver(depot, locs, demand, capacity, cfg)
            elif cfg.solver == 'ortools':
                tour, cost = ortools_cvrp_solver(depot, locs, demand, capacity, cfg)
            elif cfg.solver == 'pyvrp':
                tour, cost = pyvrp_solver(depot, locs, demand, capacity, cfg)
            elif cfg.solver == 'cplex':
                tour, cost = cplex_cvrp_solver(depot, locs, demand, capacity, cfg)
            elif cfg.solver == 'gurobi':
                tour, cost = gurobi_cvrp_solver(depot, locs, demand, capacity, cfg)

        elif cfg.ptype == 'atsp':
            if cfg.solver == 'lkh':
                tour, cost = lkh_atsp_solver(node_matrix, cfg)
        return tour, cost
