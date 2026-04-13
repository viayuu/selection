import numpy as np
import torch
from EasyNCO.neural_solvers.utils.post_search import sampling_search

class ACO_SMTWTP():

    def __init__(self,
                 due_time, # [n,]
                 weights,  # [n,]
                 processing_time, # [n]
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu'
                 ):
        
        self.n = len(due_time)
        self.due_time = due_time
        self.weights = weights
        self.processing_time = processing_time
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        
        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = 1
        
        if pheromone is None:
            self.pheromone = torch.ones(size=(self.n+1, self.n+1), device=device) # [n+1, n+1], includes dummy node 0
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        # A Weighted Population Update Rule for PACO Applied to the Single Machine Total Weighted Tardiness Problem
        # perfer jobs with smaller due time, [n+1, n+1], includes dummy node 0
        self.heuristic = (1 / torch.cat([torch.tensor([1], device=device), self.due_time])).repeat(self.n+1, 1) if heuristic is None else heuristic 
        self.probmat = None
        self.best_sol = None
        self.lowest_cost = float('inf')

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
        if best_cost < self.lowest_cost:
            self.best_sol = paths[:, best_idx]
            self.lowest_cost = best_cost
        self.update_pheronome(paths, costs)
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
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/(best_cost + 1)
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/(cost + 1)
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        paths = (paths - 1).T # due to the dummy node 0, (n_ants, problem_size)
        ants_time = self.processing_time[paths] # corresponding processing time (n_ants, problem_size)
        ants_presum_time = torch.stack([ants_time[:, :i].sum(dim=1) for i in range(1, self.n + 1)]).T # presum (total) time (n_ants, problem_size)
        ants_due_time = self.due_time[paths] # (n_ants, problem_size)
        ants_weights = self.weights[paths] # (n_ants, problem_size)
        diff = ants_presum_time - ants_due_time
        diff[diff < 0] = 0
        ants_weighted_tardiness = (ants_weights * diff).sum(dim=1)
        return ants_weighted_tardiness # (n_ants,)
        
    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.zeros(size=(self.n_ants,), dtype=torch.long, device=self.device)
        
        visit_mask = torch.ones(size=(self.n_ants, self.n + 1), device=self.device)
        visit_mask[:, 0] = 0 # exlude the dummy node (starting node) 0
                
        paths_list = [] # paths_list[i] is the ith action (tensor) for all ants
        probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for _ in range(self.n):
            actions, probs = self.pick_move(prev, visit_mask)
            paths_list.append(actions)
            if require_prob:
                probs_list.append(probs)
                visit_mask = visit_mask.clone()
            prev = actions
            visit_mask[torch.arange(self.n_ants), actions] = 0
            
        if require_prob:
            return torch.stack(paths_list), torch.stack(probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask):
        dist = (self.probmat[prev] * mask) # shape: (n_ants, p_size)
        dist = dist.unsqueeze(0) # shape: (1, n_ants, p_size)
        actions, probs = sampling_search(dist)
        actions = actions.squeeze(0)
        probs = probs.squeeze(0)
        # log_probs = torch.log(probs)
        return actions, probs
        
        
