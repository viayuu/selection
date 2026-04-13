import numpy as np
import torch
from EasyNCO.neural_solvers.utils.post_search import sampling_search

class ACO_SOP():

    def __init__(self, 
                 distances,
                 prec_cons,
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
        
        self.problem_size = len(distances)
        self.distances  = distances
        self.prec_cons = prec_cons
        
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
            self.max = None
        
        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

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
            self.shortest_path = paths[:, best_idx]
            self.lowest_cost = best_cost
            if self.min_max:
                max = self.problem_size / self.lowest_cost
                if self.max is None:
                    self.pheromone *= max/self.pheromone.max()
                self.max = max

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
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=-1, dims=1)  # shape: (n_ants, problem_size)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.zeros(size=(self.n_ants,), dtype=torch.long, device=self.device)
        prec_cons_ants = self.prec_cons.repeat(self.n_ants, 1, 1)
        
        prec_cons_ants = self.update_prec_cons(prec_cons_ants, start)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask[:, 0] = 0
        
        prec_mask = (prec_cons_ants == 0).all(dim=-1) # [n_ant, problem_size]
                
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for _ in range(self.problem_size-1):
            actions, probs = self.pick_move(prev, visit_mask, prec_mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                probs_list.append(probs)
                
            prec_cons_ants = self.update_prec_cons(prec_cons_ants, actions)
            
            prev = actions
            
            if require_prob:
                visit_mask = visit_mask.clone()
                prec_mask = prec_mask.clone()
            
            visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
            prec_mask = (prec_cons_ants == 0).all(dim=-1)
            
        if require_prob:
            return torch.stack(paths_list), torch.stack(probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask1, mask2, require_prob):
        dist = (self.probmat[prev] * mask1 * mask2) # shape: (n_ants, p_size)
        dist = dist.unsqueeze(0) # shape: (1, n_ants, p_size)
        actions, probs = sampling_search(dist)
        actions = actions.squeeze(0)
        probs = probs.squeeze(0)
        # log_probs = torch.log(probs)
        return actions, probs
    
    def update_prec_cons(self, prec_cons_ants, actions):
        '''
        Args:
            prec_cons_ants: [n_ants, p_size, p_size], [i,j,k] = 1 means:
                for ant i, node k precedes node j and k has yet been visited
            actions: [n_ants, ], the visited nodes for all ants
        '''
        prec_cons_ants[torch.arange(self.n_ants), : , actions] = 0
        return prec_cons_ants
        
