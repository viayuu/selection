import sys
import torch
import random
import time
import itertools
import numpy as np
from functools import cached_property
import concurrent.futures
import os
import platform
from ctypes import (
    Structure,
    CDLL,
    POINTER,
    c_int,
    c_double,
    c_char,
    sizeof,
    cast,
    byref,
    c_char_p,
)
from dataclasses import dataclass
from typing import List
from EasyNCO.neural_solvers.utils.post_search import sampling_search

CAPACITY = 1.0 # The input demands shall be normalized
  
class ACO_CVRP():

    def __init__(self,  # 0: depot
                 distances, # (n, n)
                 demand,   # (n, )
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu',
                 adaptive=False,
                 capacity=CAPACITY,
                 swapstar = False,
                 positions = None,
                 inference = False,
                 ):
        
        self.problem_size = len(distances)
        self.distances = distances
        self.capacity = capacity
        self.demand = demand
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or adaptive
        self.min_max = min_max
        self.adaptive = adaptive
        self.swapstar = swapstar
        self.positions = positions
        self.inference = inference

        assert positions is not None if swapstar else True
        
        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None
        
        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone
        
        if self.adaptive:
            self.elite_pool = []

        self.heuristic = 1 / distances if heuristic is None else heuristic
        self.probmat = None
        self.shortest_path = None
        self.lowest_cost = float('inf')
        self.device = device
    
    def sample(self):
        self.probmat = self.heatmap()
        paths, probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, probs, paths

    def sample_nls(self, paths):
        self.probmat = self.heatmap()
        paths, log_probs = self.gen_path(require_prob=True)
        self.multiple_swap_star(paths)
        costs = self.gen_path_costs(paths).detach()
        return costs
    
    def train_instance(self, infer = False, nls = False):
        if not infer:
            costs, probs, paths = self.sample() # costs[ants,]; log_probs[nodes-1, ants]; paths[nodes, ants]
            baseline = costs.mean()
            if nls:
                # Local search interleaved with neural-guided perturbation(NLS) is applied to the loss function.
                W = 0.95
                costs_nls = self.sample_nls(paths) # costs_nls[ants,]; paths[nodes, ants]
                baseline_nls = costs_nls.mean()
                cost = (costs_nls - baseline_nls) * W + (costs - baseline) * (1 - W)
            else:
                cost = costs - baseline
            # reinforce_loss = torch.sum(cost.detach() * log_probs.sum(dim=0)) / len(cost)
            return -cost, probs

        else:
            costs, _, _ = self.sample()
            baseline = costs.mean()
            best_cost = torch.min(costs)
            heatmap = self.heatmap()
            result = self.search(heatmap)
            return np.array([baseline.item(), best_cost.item(), result.item()])

    @torch.no_grad()
    def multiple_swap_star(self, paths, indexes = None):
        subroutes_all = []
        for i in range(paths.size(1)) if indexes is None else indexes:
            subroutes = get_subroutes(paths[:, i])
            subroutes_all.append((i, subroutes))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, p in subroutes_all:
                future = executor.submit(neural_swapstar, self.demand_cpu, self.distances_cpu, self.heuristic_dist, self.positions_cpu, p,
                                        limit = 100000 if self.inference else max(self.problem_size, 50))
                futures.append((i, future))
            for i, future in futures:
                paths[:, i] = merge_subroutes(future.result(), paths.size(0), self.device)
        
    @cached_property
    @torch.no_grad()
    def heuristic_dist(self):
        heu = self.heuristic.detach().cpu().numpy()
        return (1 / (heu/heu.max(-1, keepdims=True) + 1e-5))

    def heatmap(self):
        probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        return probmat

    @torch.no_grad()
    def search(self, probmat):
        self.probmat = probmat
        paths = self.gen_path(require_prob=False)
        costs = self.gen_path_costs(paths)

        if self.adaptive:
            self.improvement_phase(paths, costs)

        if self.swapstar:
            indexes = costs.topk(8, largest=False).indices
            self.multiple_swap_star(paths, indexes=indexes)
            costs = self.gen_path_costs(paths)

        improved = False
        best_cost, best_idx = costs.min(dim=0)
        if best_cost < self.lowest_cost:
            self.shortest_path = paths[:, best_idx].clone()
            self.lowest_cost = best_cost
            if self.adaptive:
                self.intensification_phase(paths, costs, best_idx)
            if self.min_max:
                max = self.problem_size / self.lowest_cost
                if self.max is None:
                    self.pheromone *= max / self.pheromone.max()
                self.max = max
            improved = True

        if not self.adaptive or improved:
            self.update_pheronome(paths, costs)
            if self.adaptive:
                self.elite_pool.insert(0, (self.shortest_path, self.lowest_cost))
                if len(self.elite_pool) > 5:  # pool_size = 5
                    del self.elite_pool[5:]
        else:
            self.diversification_phase()

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour = paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
        
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self, require_prob=False):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants
        probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        done = self.check_done(visit_mask, actions)
        while not done:
            actions, probs = self.pick_move(actions, visit_mask, capacity_mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                probs_list.append(probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)
            
        if require_prob:
            return torch.stack(paths_list), torch.stack(probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, visit_mask, capacity_mask, require_prob):
        dist = (self.probmat[prev] * visit_mask * capacity_mask) # shape: (n_ants, p_size)
        dist = dist.unsqueeze(0) # shape: (1, n_ants, p_size)
        actions, probs = sampling_search(dist)
        actions = actions.squeeze(0)
        probs = probs.squeeze(0)
        # log_probs = torch.log(probs)
        return actions, probs
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    
    @cached_property
    @torch.no_grad()
    def distances_cpu(self):
        return self.distances.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def demand_cpu(self):
        return self.demand.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def positions_cpu(self):
        return self.positions.cpu().numpy() if self.positions is not None else None
    
    # ======== code for adaptive elitist AS ========
    # These code are unrelated to DeepACO, and are kept for comparisons.
    def insertion_single(self, route, index):
        # route starts from 0, terminates with 0
        insertion_cost = (((self.distances[p1, index]+self.distances[index,p2]-self.distances[p1, p2]).item(), i) 
                          for i,(p1,p2) in enumerate(zip(route,route[1:])))
        min_deltacost, min_index = min(insertion_cost)
        return min_index, min_deltacost
    
    def insertion(self, node_indexes, shuffle = False):
        route = [node_indexes[0].item()]*2
        cost = 0
        if shuffle:
            perm = torch.randperm(len(node_indexes)-1)+1
            nodes = node_indexes[perm]
        else:
            nodes = node_indexes[1:]
        for i in nodes:
            bestpos, deltacost = self.insertion_single(route, i)
            route.insert(bestpos+1, i.item())
            cost += deltacost
        return route, cost


    @torch.no_grad()
    def N1_neighbourhood(self, subroutes, demands, count = 5):
        # N1 neighbourhood: Pick a random node and insert it in other subroutes.
        best_insertion = (None, 0.0)
        for _ in range(count):
            subroute_index = random.randint(0, len(subroutes)-1)
            route = subroutes[subroute_index]
            node_index = random.randint(1, len(route)-2) # exclude depots
            pred, node, next = route[node_index-1: node_index+2]
            demand = self.demand[node]
            avaliable = demands + demand <= self.capacity
            avaliable[subroute_index] = False
            if not avaliable.any(): # no avaliable subroute
                continue
            cost = self.distances[pred, next] - self.distances[pred, node] - self.distances[node, next]
            for i, r in itertools.compress(enumerate(subroutes), avaliable):
                loc, insertion_cost = self.insertion_single(r, node)
                insertion_cost += cost
                if insertion_cost < best_insertion[1]:
                    best_insertion = ((subroute_index, node_index, i, loc+1), insertion_cost)

        if best_insertion[0] is not None: # perform insertion
            sri, sni, tri, tni = best_insertion[0]
            subroutes = subroutes[:]
            source_route, target_route = subroutes[sri], subroutes[tri]
            node = subroutes[sri][sni]
            subroutes[tri] = torch.cat([target_route[:tni], node.unsqueeze(0), target_route[tni:]])
            if len(subroutes[sri])==3:
                del subroutes[sri]
            else:
                subroutes[sri] = torch.cat([source_route[:sni], source_route[sni+1:]])
            return subroutes, best_insertion[1]
        else:
            return best_insertion
    
    @torch.no_grad()
    def N2_neighbourhood(self, subroutes, demands, count = 5):
        # N2 neighbourhood: Randomly swap 2 nodes and insert them in the best position.
        best_insertion = (None, 0.0)
        for _ in range(count):
            sr1_index, sr2_index = np.random.choice(len(subroutes), size=2, replace=False)
            sr1, sr2 = subroutes[sr1_index], subroutes[sr2_index]
            node1_index = random.randint(1, len(sr1)-2)
            pred1, node1, next1 = sr1[node1_index-1:node1_index+2]
            demand1 = self.demand[node1]
            # avaliable nodes to swap
            avaliable = torch.bitwise_and(
                demands[sr2_index] + demand1 - self.demand[sr2] <= self.capacity,
                demands[sr1_index] - demand1 + self.demand[sr2] <= self.capacity,
            )
            avaliable[0] = avaliable[-1] = False
            if not avaliable.any():
                continue
            # remove node1 from sr1
            cost = self.distances[pred1, next1] - self.distances[pred1, node1] - self.distances[node1, next1]
            sr1_mod = torch.concat([sr1[:node1_index], sr1[node1_index+1:]])
            # choose a node from sr2
            avaliable_index = torch.arange(len(sr2))[avaliable]
            node2_index = np.random.choice(avaliable_index)
            pred2, node2, next2 = sr2[node2_index-1: node2_index+2]
            # remove node2 from sr2
            cost += self.distances[pred2, next2] - self.distances[pred2, node2] - self.distances[node2, next2]
            sr2_mod = torch.concat([sr2[:node2_index], sr2[node2_index+1:]])
            # insert node1 into sr2_mod
            loc1, inscost1 = self.insertion_single(sr2_mod, node1)
            cost += inscost1
            sr2_mod = torch.concat([sr2_mod[:loc1+1], node1.unsqueeze(0), sr2_mod[loc1+1:]])
            # insert node2 into sr1_mod
            loc2, inscost2 = self.insertion_single(sr1_mod, node2)
            cost += inscost2
            sr1_mod = torch.concat([sr1_mod[:loc2+1], node2.unsqueeze(0), sr1_mod[loc2+1:]])
            if cost < best_insertion[1]:
                best_insertion = ((sr1_index, sr1_mod, sr2_index, sr2_mod), cost)

        if best_insertion[0] is not None: # perform insertion
            sr1_index, sr1, sr2_index, sr2 = best_insertion[0]
            subroutes = subroutes[:]
            subroutes[sr1_index] = sr1
            subroutes[sr2_index] = sr2
            return subroutes, best_insertion[1]
        else:
            return best_insertion

    @torch.no_grad()
    def improvement_phase(self, paths, costs, topk = 5):
        # local search
        if topk <= 0 or topk >= self.n_ants:
            target_indexes = range(paths.size(1))
        else:
            target_indexes = costs.topk(5, largest=False).indices

        for i in target_indexes:
            subroutes = get_subroutes(paths[:, i], end_with_zero=False)
            # ILS (not implemented)
            pass
            # insertion
            new_subroutes = []
            new_cost=0
            for r in subroutes:
                new_subroute, c = self.insertion(r)
                new_cost += c
                new_subroutes.append(new_subroute)
            if new_cost < costs[i]:
                paths[:, i] = merge_subroutes(new_subroutes, paths.size(0), self.device)
                costs[i] = new_cost
    
    @torch.no_grad()
    def intensification_phase(self, paths, costs, best_idx):
        ogroute, ogcost = self.shortest_path, self.lowest_cost
        subroutes = get_subroutes(ogroute, end_with_zero=True)
        demands = torch.tensor([self.demand[r].sum() for r in subroutes], device=self.device)
        # print(*subroutes, sep='\n')
        best_neighbour = (None, 0.0)
        for func in [self.N1_neighbourhood, self.N2_neighbourhood]:
            route, cost = func(subroutes, demands)
            if cost < best_neighbour[1]:
                best_neighbour = (route, cost)
        if best_neighbour[0] is not None:
            self.shortest_path = merge_subroutes(best_neighbour[0], self.shortest_path.size(0), self.device)
            self.lowest_cost = ogcost + best_neighbour[1]
            paths[:, best_idx] = self.shortest_path
            costs[best_idx] = self.lowest_cost

    @torch.no_grad()
    def diversification_phase(self):
        # reinitialize pheromone trails
        self.pheromone = self.pheromone * (self.decay * 0.5) + 0.01
        for path, cost in self.elite_pool:
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost

def neural_swapstar(demand, distances, heu_dist, positions, p, disturb=10, limit=10000):
    p0 = p
    p1 = swapstar(demand, distances, positions, p0, count = limit)
    p2 = swapstar(demand, heu_dist, positions, p1, count = disturb)
    p3 = swapstar(demand, distances, positions, p2, count = limit)
    return p3

def get_subroutes(route, end_with_zero = True):
    x = torch.nonzero(route == 0).flatten()
    subroutes = []
    for i, j in zip(x, x[1:]):
        if j-i>1:
            if end_with_zero:
                j=j+1
            subroutes.append(route[i:j])
    return subroutes

def merge_subroutes(subroutes, length, device):
    route = torch.zeros(length, dtype = torch.long, device=device)
    i=0
    for r in subroutes:
        if len(r)>2:
            if isinstance(r, list):
                r = torch.tensor(r[:-1])
            else:
                r = r[:-1].clone().detach()
            route[i: i+len(r)] = r
            i+=len(r)
    return route

# Import the HGS solver
# Adapted from https://github.com/chkwon/PyHygese/blob/master/hygese/hygese.py

def write_routes(routes: List[List[int]], filepath: str):
    with open(filepath, "w") as f:
        for i, r in enumerate(routes):
            f.write(f"Route #{i + 1}: " + ' '.join([str(x.item()) for x in r if x > 0]) + "\n")
    return

def read_routes(filepath):
    routes = []
    with open(filepath, "r") as f:
        while 1:
            line = f.readline().strip()
            if line.startswith("Route"):
                routes.append(torch.tensor([0, *map(int, line.split(":")[1].split()), 0]))
            else:
                break
    return routes


def get_lib_filename():
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    path = os.path.join(current_dir, "HGS-CVRP-main", "build", "libhgscvrp.so")
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(f"Shared library file `{path}` not found")


# basedir = os.path.abspath(os.path.dirname(__file__))
basedir = os.path.dirname(os.path.realpath(__file__))
# os.add_dll_directory(basedir)
HGS_LIBRARY_FILEPATH = os.path.join(basedir, get_lib_filename())

c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)
C_INT_MAX = 2 ** (sizeof(c_int) * 8 - 1) - 1
C_DBL_MAX = sys.float_info.max

# Must match with AlgorithmParameters.h in HGS-CVRP: https://github.com/vidalt/HGS-CVRP
class CAlgorithmParameters(Structure):
    _fields_ = [
        ("nbGranular", c_int),
        ("mu", c_int),
        ("lambda", c_int),
        ("nbElite", c_int),
        ("nbClose", c_int),
        ("targetFeasible", c_double),
        ("seed", c_int),
        ("nbIter", c_int),
        ("timeLimit", c_double),
        ("useSwapStar", c_int),
    ]

@dataclass
class AlgorithmParameters:
    nbGranular: int = 20
    mu: int = 25
    lambda_: int = 40
    nbElite: int = 4
    nbClose: int = 5
    targetFeasible: float = 0.2
    seed: int = 0
    nbIter: int = 20000
    timeLimit: float = 0.0
    useSwapStar: bool = True

    @property
    def ctypes(self) -> CAlgorithmParameters:
        return CAlgorithmParameters(
            self.nbGranular,
            self.mu,
            self.lambda_,
            self.nbElite,
            self.nbClose,
            self.targetFeasible,
            self.seed,
            self.nbIter,
            self.timeLimit,
            int(self.useSwapStar),
        )

class _SolutionRoute(Structure):
    _fields_ = [("length", c_int), ("path", c_int_p)]

class _Solution(Structure):
    _fields_ = [
        ("cost", c_double),
        ("time", c_double),
        ("n_routes", c_int),
        ("routes", POINTER(_SolutionRoute)),
    ]

class RoutingSolution:
    def __init__(self, sol_ptr):
        if not sol_ptr:
            raise TypeError("The solution pointer is null.")

        self.cost = sol_ptr[0].cost
        self.time = sol_ptr[0].time
        self.n_routes = sol_ptr[0].n_routes
        self.routes = []
        for i in range(self.n_routes):
            r = sol_ptr[0].routes[i]
            path = r.path[0: r.length]
            self.routes.append(path)

class Solver:
    def __init__(self, parameters=AlgorithmParameters(), verbose=False):
        if platform.system() == "Windows":
            hgs_library = CDLL(HGS_LIBRARY_FILEPATH, winmode=0)
        else:
            hgs_library = CDLL(HGS_LIBRARY_FILEPATH)

        self.algorithm_parameters = parameters
        self.verbose = verbose

        # solve_cvrp
        self._c_api_solve_cvrp = hgs_library.solve_cvrp
        self._c_api_solve_cvrp.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double,
            c_double,
            c_char,
            c_char,
            c_int,
            POINTER(CAlgorithmParameters),
            c_char,
        ]
        self._c_api_solve_cvrp.restype = POINTER(_Solution)

        # solve_cvrp_dist_mtx
        self._c_api_local_search = hgs_library.local_search
        self._c_api_local_search.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double,
            c_double,
            c_char,
            c_int,
            POINTER(CAlgorithmParameters),
            c_char,
            c_int,
            c_int,
        ]
        self._c_api_local_search.restype = c_int

        # delete_solution
        self._c_api_delete_sol = hgs_library.delete_solution
        self._c_api_delete_sol.restype = None
        self._c_api_delete_sol.argtypes = [POINTER(_Solution)]

    def local_search(self, data, routes: List[List[int]], count: int = 1, rounding=True, ):
        # required data
        demand = np.asarray(data["demands"])
        vehicle_capacity = data["vehicle_capacity"]
        n_nodes = len(demand)

        # optional depot
        depot = data.get("depot", 0)
        if depot != 0:
            raise ValueError("In HGS, the depot location must be 0.")

        # optional num_vehicles
        maximum_number_of_vehicles = data.get("num_vehicles", C_INT_MAX)

        # optional service_times
        service_times = data.get("service_times")
        if service_times is None:
            service_times = np.zeros(n_nodes)
        else:
            service_times = np.asarray(service_times)

        # optional duration_limit
        duration_limit = data.get("duration_limit")
        if duration_limit is None:
            is_duration_constraint = False
            duration_limit = C_DBL_MAX
        else:
            is_duration_constraint = True

        is_rounding_integer = rounding

        x_coords = data.get("x_coordinates")
        y_coords = data.get("y_coordinates")
        dist_mtx = data.get("distance_matrix")

        if x_coords is None or y_coords is None:
            assert dist_mtx is not None
            x_coords = np.zeros(n_nodes)
            y_coords = np.zeros(n_nodes)
        else:
            x_coords = np.asarray(x_coords)
            y_coords = np.asarray(y_coords)

        assert len(x_coords) == len(y_coords) == len(service_times) == len(demand)
        assert (x_coords >= 0.0).all()
        assert (y_coords >= 0.0).all()
        assert (service_times >= 0.0).all()
        assert (demand >= 0.0).all()

        dist_mtx = np.asarray(dist_mtx)
        assert dist_mtx.shape[0] == dist_mtx.shape[1]
        assert (dist_mtx >= 0.0).all()

        callid = (time.time_ns() * 10000 + random.randint(0, 10000)) % C_INT_MAX

        tmppath = "/tmp/route-{}".format(callid)
        resultpath = "/tmp/swapstar-result-{}".format(callid)
        write_routes(routes, tmppath)
        try:
            self._local_search(
                x_coords,
                y_coords,
                dist_mtx,
                service_times,
                demand,
                vehicle_capacity,
                duration_limit,
                is_duration_constraint,
                maximum_number_of_vehicles,
                self.algorithm_parameters,
                self.verbose,
                callid,
                count,
            )

            result = read_routes(resultpath)
        except Exception as e:
            print(routes)
            print([demand[r].sum() for r in routes])
        else:
            os.remove(resultpath)
        finally:
            os.remove(tmppath)

        return result

    def _local_search(
            self,
            x_coords: np.ndarray,
            y_coords: np.ndarray,
            dist_mtx: np.ndarray,
            service_times: np.ndarray,
            demand: np.ndarray,
            vehicle_capacity: int,
            duration_limit: float,
            is_duration_constraint: bool,
            maximum_number_of_vehicles: int,
            algorithm_parameters: AlgorithmParameters,
            verbose: bool,
            callid: int,
            count: int,
    ):
        n_nodes = x_coords.size

        x_ct = x_coords.astype(c_double).ctypes
        y_ct = y_coords.astype(c_double).ctypes
        s_ct = service_times.astype(c_double).ctypes
        d_ct = demand.astype(c_double).ctypes

        m_ct = dist_mtx.reshape(n_nodes * n_nodes).astype(c_double).ctypes
        ap_ct = algorithm_parameters.ctypes

        # struct Solution *solve_cvrp_dist_mtx(
        # 	int n, double* x, double* y, double *dist_mtx, double *serv_time, double *dem,
        # 	double vehicleCapacity, double durationLimit, char isDurationConstraint,
        # 	int max_nbVeh, const struct AlgorithmParameters *ap, char verbose);
        sol_p = self._c_api_local_search(
            n_nodes,
            cast(x_ct, c_double_p),
            cast(y_ct, c_double_p),
            cast(m_ct, c_double_p),
            cast(s_ct, c_double_p),
            cast(d_ct, c_double_p),
            vehicle_capacity,
            duration_limit,
            is_duration_constraint,
            maximum_number_of_vehicles,
            byref(ap_ct),
            verbose,
            callid,
            count,
        )

        result = sol_p
        return result

def swapstar(demands, matrix, positions, routes, count=1):
    ap = AlgorithmParameters()
    hgs_solver = Solver(parameters=ap, verbose=False)

    data = dict()
    x = positions[:, 0]
    y = positions[:, 1]
    data['x_coordinates'] = x
    data['y_coordinates'] = y

    data['depot'] = 0
    data['demands'] = demands * 1000
    data["num_vehicles"] = len(routes)
    data['vehicle_capacity'] = 1000.001  # to avoid floating-point error

    # Solve with calculated distances
    data['distance_matrix'] = matrix
    try:
        result = hgs_solver.local_search(data, routes, count)
    except Exception as e:
        print(e)
        return routes
    return result
