import torch
import numpy as np
import numba
from functools import cached_property

from EasyNCO.neural_solvers.utils.post_search import sampling_search

CAPACITY = 150


@numba.njit()
def count_last_zero(x: np.ndarray):
    ret = np.zeros(len(x),)
    n, m = x.shape
    for i in range(n):
        count = 0
        for j in range(m-1, -1, -1):
            if x[i][j] == 0:
                count += 1
            else:
                ret[i] = count
                break
    return ret

@numba.njit()
def cal_fitness(s: np.ndarray, demand: np.ndarray, n_bins: np.ndarray):
    ret = np.zeros(len(s),)
    n, m =  s.shape
    for i in range(n):
        f = 0
        sub_f = 0
        for j in range(1, m):
            if s[i, j] != 0: # not dummy node
                sub_f += demand[s[i, j]]
            else:
                f += (sub_f / CAPACITY)**2
                sub_f = 0
        ret[i] = f / n_bins[i]
    return ret
            
class ACO_BPP():

    # Levine, J., & Ducatelle, F. (2004). Ant colony optimization and local search for bin packing and cutting stock problems. 
    # Journal of the Operational Research society, 55(7), 705-716.

    def __init__(self,  # 0: depot
                 demand,   # (n, )
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 pheromone=None,
                 heuristic=None,
                 device='cpu',
                 capacity=CAPACITY
                 ):
        
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        
        self.pheromone = torch.ones(self.problem_size, self.problem_size) if pheromone is None else pheromone
        self.heuristic = self.demand.unsqueeze(0).repeat(len(demand), 1) if heuristic is None else heuristic
        self.heuristic[:, 0] = 1e-5
        self.probmat = None
        self.shortest_path = None
        self.best_fitness = 0

        self.device = device
    
    def sample(self):
        self.probmat = self.heatmap()
        paths, probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, probs, paths

    def train_instance(self, infer = False):
        if not infer:
            costs, probs, paths = self.sample() # costs[ants,]; log_probs[nodes-1, ants]; paths[nodes, ants]
            baseline = costs.mean()
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


    def heatmap(self):
        probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        return probmat

    @torch.no_grad()
    def search(self, probmat):
        self.probmat = probmat
        paths = self.gen_path(require_prob=False)
        costs = self.gen_path_costs(paths)
        best_cost, best_idx = costs.min(dim=0)
        if  - best_cost > self.best_fitness:
            self.shortest_path = paths[:, best_idx]
            self.best_fitness = - best_cost
        self.update_pheronome(paths, -costs)
        return self.best_fitness
       
    @torch.no_grad()
    def update_pheronome(self, paths, fits):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_fit, best_idx = fits.max(dim=0)
            best_tour = paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += best_fit
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                fit = fits[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += fit / self.n_ants
        
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths:torch.Tensor):
        u = paths.permute(1, 0).cpu().numpy() # shape: (n_ants, max_seq_len)
        last_zeros = count_last_zero(u)
        n_bins = u.shape[1] - last_zeros - self.problem_size + 1 # number of bins
        fit = cal_fitness(u, self.demand_numpy, n_bins)
        return -torch.tensor(fit)


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
    def demand_numpy(self):
        return self.demand.cpu().numpy()

