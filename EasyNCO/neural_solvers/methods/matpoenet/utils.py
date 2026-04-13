import torch
from ctypes import CDLL, c_int, c_double, byref
import numpy as np
import argparse
import os

from torch.utils.data import DataLoader,Dataset
from EasyNCO.exact_solvers.lkh_atsp import lkh_atsp_solver


def make_positional_encoding_cosh_recur(dim, n_node, scaler=10):
    
    dim_phi = torch.arange(0, dim, dtype=torch.float) / dim * 2 - 1 # T = 2
    position_phi = torch.arange(0, n_node, dtype=torch.float).unsqueeze(1) / n_node * 2
    phi = dim_phi + position_phi
    next_period_phi_idx = phi > 1
    phi[next_period_phi_idx] -= 2
    pe = 1 / torch.cosh(phi * scaler)
    
    return pe



# ===========================================================================
# Base solver (original: utils/base_methods)
# ===========================================================================
class BaseSolver:
    def __init__(self, n, lib_path="./libtsp.so", scaler=1e6):
        '''
        n int: number of nodes in the problems
        '''
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(base_dir, "libtsp.so")       
        self.lib_tsp = CDLL(lib_path)
        
    
        self.n = n
        self.path = (c_int * n)(*(list(range(n))))
        self.cost = c_double(0)
        self.scaler = scaler

    def _make_problem(self, dist):
        
        
        self.n = dist.size(0)
        # transform the dist(distance matrix) to the format that the C++ library can read
        return (c_double * (self.n * self.n))(*dist.reshape(self.n * self.n).tolist()) 

    def get_cost(self):
        return self.cost.value

    def get_path(self):
        return list(self.path)

    def solve_random_walk(self, dist):
        pass

    def solve_nearest_neighbor(self, dist):
        '''
        dist numpy array: the distance matrix
        '''
        self.lib_tsp.nearest_neighbor(self.n, 
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))
        
    def solve_nearest_insertion(self, dist):
        self.lib_tsp.nearest_insertion(self.n, 
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))
    
    def solve_farthest_insertion(self, dist):
        self.lib_tsp.farthest_insertion(self.n, 
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))
    
    def solve_rand_perm(self, dist):
        self.path = np.random.permutation(self.n).tolist()
        
        
        
    def solve_lkh(self, dist, runs=1, max_trials=1000):
        """
        dist numpy array: the distance matrix
        self.path: solved path
        
        results = lkh_atsp_solver_multiprocess(atsp_dataloader, par_args)
        result: (route, distance)
        lkh_atsp_solver(node_matrix, par_args, problem_index = None, distribution = None, attributes = None)
        
        """
        
        import warnings
        from copy import deepcopy
        warnings.filterwarnings('ignore')
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--RUNS", type=int, default=1)
        parser.add_argument("--MAX_TRIALS", type=int, default=10)
        parser.add_argument("--ptype", type=str, default="TSP")
        parser.add_argument("--int_matrix_scale", type=int, default=1000)
        parser.add_argument("--cpus", type=int, default=4,help="multiprocess, cpu number")
        parser.add_argument('--TRACE_LEVEL', type=int, default=1,
                        help="Log information")
        parser.add_argument('--SEED', type=int, default=1,
                        help="Generate seeds for random numbers")
        parser.add_argument("--CANDIDATE_SET_TYPE", type=str, choices=['POPMUSIC', 'ALPHA', 'DELAUNAY', 'NEAREST-NEIGHBOR'],
                        default='ALPHA',
                        help="LKH: Accelerate algorithm")
        parser.add_argument('--save_as_txt', default=True,
                        help="Convert to TXT file after solving")
        parser.add_argument("--delete", default=False,
                        help="delete the lib file, including .par,.tsp,.vrp....")
        
        
        par_args = argparse.Namespace(
            RUNS=runs,
            MAX_TRIALS=max_trials, 
            ptype="atsp",
            int_matrix_scale=10000,
            cpus=4,
            TRACE_LEVEL=1,
            SEED=1,
            CANDIDATE_SET_TYPE="ALPHA",
            save_as_txt=True,
            delete=False,)
        
        
        result = lkh_atsp_solver(dist, par_args, problem_index = None, distribution = None, attributes = None)
        lkh_path = result[0]
        self.path = deepcopy(lkh_path)  
        tour_len = result[1]
        tour_len *= 1e-6
        return tour_len
        

