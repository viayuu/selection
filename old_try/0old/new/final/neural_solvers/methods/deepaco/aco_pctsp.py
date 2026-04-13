import torch
import numpy as np
from copy import deepcopy

from EasyNCO.neural_solvers.utils.post_search import sampling_search


class ACO_PCTSP():

    def __init__(self,
                 distances,
                 prizes,
                 penalties,
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
        
        self.n = prizes.size(0)
        self.distances = distances
        self.prizes = prizes
        self.penalties = penalties
        self.min_prizes = self.n / 4
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        
        self.ants_idx = torch.arange(n_ants)
        
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
        
        _distances = deepcopy(self.distances)
        _distances[torch.arange(self.n), torch.arange(self.n)] = 1e9
        self.heuristic = (1e-10 + prizes.repeat(self.n, 1)) / _distances if heuristic is None else heuristic
        self.probmat = None
        self.alltime_best_obj = 1e10
        self.alltime_best_sol = None

        self.device = device
    
    def sample(self):
        self.probmat = self.heatmap()
        sols, probs = self.gen_sol(require_prob=True)
        objs = self.gen_sol_obj(sols)
        return objs, probs, objs

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
        sols = self.gen_sol(require_prob=False)
        objs = self.gen_sol_obj(sols)
        sols = sols.T
        best_obj, best_idx = objs.max(dim=0)
        if best_obj < self.alltime_best_obj:
            self.alltime_best_obj = best_obj
            self.alltime_best_sol = sols[best_idx]
            if self.min_max:
                max = (self.n - 1) / self.alltime_best_obj
                if self.max is None:
                    self.pheromone *= max/self.pheromone.max()
                self.max = max
        self.update_pheronome(sols, objs, best_obj, best_idx)
        return self.alltime_best_obj
       
    
    @torch.no_grad()
    def update_pheronome(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay 
        if self.elitist:
            best_sol= sols[best_idx]
            self.pheromone[best_sol[:-1], torch.roll(best_sol, shifts=-1)[:-1]] += 1.0/best_obj
        
        else:
            for i in range(self.n_ants):
                sol = sols[i]
                obj = objs[i]
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] +=  1.0/obj
                
        if self.min_max:
            self.pheromone[(self.pheromone>1e-9) * (self.pheromone)<self.min] = self.min
            self.pheromone[self.pheromone>self.max] = self.max
    
    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        u = solutions.permute(1, 0)
        v = torch.roll(u, shifts=-1, dims=1)
        length = torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)
        penalty_bool = self.gen_penalty_bool(u, self.n)
        penalty = []
        for ant_id in range(self.n_ants):
            ant_penalty = self.penalties[penalty_bool[ant_id]].sum()
            penalty.append(ant_penalty)
        return length + torch.stack(penalty)

    def gen_penalty_bool(self, sol, n):
        '''
        Args:
            sol: (n_ants, max_seq_len)
        '''
        n_ants = sol.size(0)
        seq_len = sol.size(1)
        expanded_nodes = torch.arange(n, device=self.device).repeat(n_ants, seq_len, 1) # (n_ants, seq_len, n)
        expanded_sol = torch.repeat_interleave(sol, n, dim=-1).reshape(n_ants, seq_len, n)
        return (torch.eq(expanded_nodes, expanded_sol)==0).all(dim=1)

    def gen_sol(self, require_prob=False):
        solutions = []
        probs_list = []

        cur_node = torch.zeros(size=(self.n_ants,), dtype=torch.int64, device=self.device)
        solutions = [cur_node]
        
        visit_mask = torch.ones(size=(self.n_ants, self.n), device=self.device) # 1) mask the visted regular node; 2) once return to depot, mask all
        depot_mask = torch.ones(size=(self.n_ants, self.n), device=self.device) 
        depot_mask[: , 0] = 0 # unmask the depot when 1) enough prize collected; 2) all nodes visited
        
        collected_prize = torch.zeros(size=(self.n_ants,), device=self.device)
        done = False
        # construction
        while not done:
            cur_node, prob = self.pick_node(visit_mask, depot_mask, cur_node, require_prob) # pick action
            # update solution and log_probs
            solutions.append(cur_node) 
            probs_list.append(prob)
            # update collected_prize and mask
            collected_prize += self.prizes[cur_node]
            if require_prob:
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask, depot_mask = self.update_mask(visit_mask, depot_mask, cur_node, collected_prize)
            # check done
            done = self.check_done(cur_node)
        if require_prob:
            return torch.stack(solutions), torch.stack(probs_list)  # shape: [n_ant, max_seq_len]
        else:
            return torch.stack(solutions)
    
    def pick_node(self, visit_mask, depot_mask, cur_node, require_prob):
        dist = (self.probmat[cur_node] * visit_mask * depot_mask)
        dist = dist.unsqueeze(0) # shape: (1, n_ants, p_size)
        actions, probs = sampling_search(dist)
        actions = actions.squeeze(0)
        probs = probs.squeeze(0)
        # log_probs = torch.log(probs)
        return actions, probs  # (n_ants,)
    
    def update_mask(self, visit_mask, depot_mask, cur_node, collected_prize):
        # mask regular visted node
        visit_mask[self.ants_idx, cur_node] = 0
        # if at depot, mask all regular nodes, and unmask depot
        at_depot = cur_node == 0
        visit_mask[at_depot, 0] = 1
        visit_mask[at_depot, 1:] = 0
        # unmask the depot for in either case
        # 1) not at depot and enough prize collected
        depot_mask[(~at_depot) * (collected_prize > self.min_prizes), 0] = 1
        # 2) not at depot and all nodes visited
        depot_mask[(~at_depot) * ((visit_mask[:, 1:]==0).all(dim=1)), 0] = 1
        return visit_mask, depot_mask

    def check_done(self, cur_node):
        # is all at depot ?
        return (cur_node == 0).all()        


    