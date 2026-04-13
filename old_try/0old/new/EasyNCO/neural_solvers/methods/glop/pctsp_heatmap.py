

from torch_geometric.data import Data
import torch
from torch.distributions import Categorical



def pctsp_solution_heatmap(data,main_net,n_subset,k_sparse=None,Train_flag=False):
    """
    Solve a batch of data using a heatmap approach.

    """
    device=data.device
    max_seq_len=0
    penaltys=[]
    pi_all=[]
    probs=[]
    for i in range(data.shape[0]):
        coors=data[i,:,0:2]
        prize=data[i,:,2]
        penalty=data[i,:,3]
        if Train_flag:
            heatmap = infer_heatmap(main_net, prize, penalty, coors,k_sparse)

        else:
            with torch.no_grad():
                # main_net.eval()  #heatmaps 的输出居然与训练和评估有关
                heatmap = infer_heatmap(main_net, prize, penalty, coors,k_sparse)
        sampler = Sampler(prize, heatmap, n_subset, device)
        subset, prob = sampler.gen_subsets(require_prob=Train_flag,
                                           greedy_mode=(n_subset == 1))
        assert subset.size(0) == n_subset
        if subset.size(1) > max_seq_len:
            max_seq_len = subset.size(1)
        penalty = sampler.gen_penalty(subset, penalty)
        penaltys.append(penalty[:,None])
        pi_all.append(subset)
        probs.append(prob)
    pi_all = add_padding(pi_all, max_seq_len, n_subset,data.shape[0])  # n_val*n_subset, max_len变成最大的大小
    penaltys=torch.cat((penaltys),dim=0)
    if Train_flag:
        probs=torch.cat((probs),dim=0)
    return pi_all, penaltys,probs



#################### heatmap

k_sparse_table = {
    500: 50,
    1000: 100,
    5000: 200,
}
def infer_heatmap(model, prizes, penalties, coor,k_sparse):
    # n = prizes.size(0)-1
    dist_mat = torch.norm(coor[:, None] - coor, dim=2, p=2).to(prizes.device)
    pyg_data = gen_pyg_data_pctsp(prizes, penalties, dist_mat,k_sparse)
    heatmap = model.reshape(pyg_data, model(pyg_data,k_sparse=k_sparse)) + 1e-10
    return heatmap



class Sampler():
    '''
    To sample node subsets given a PCTSP heatmap.
    '''

    def __init__(self, prizes, heatmap, bs=20, device='cpu'):
        self.n = prizes.size(0)
        self.prizes = prizes
        self.heatmap = heatmap
        self.min_prizes = 1
        self.bs = bs
        self.ants_idx = torch.arange(bs)
        self.device = device

    def gen_subsets(self, require_prob=False, greedy_mode=False):
        if greedy_mode:
            assert not require_prob
        solutions = []
        log_probs_list = []
        cur_node = torch.zeros(size=(self.bs,), dtype=torch.int64, device=self.device)
        visit_mask = torch.ones(size=(self.bs, self.n),
                                device=self.device)  # 1) mask the visted regular node; 2) once return to depot, mask all
        depot_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        depot_mask[:, 0] = 0  # unmask the depot when 1) enough prize collected; 2) all nodes visited

        collected_prize = torch.zeros(size=(self.bs,), device=self.device)
        done = False
        # construction
        while not done:
            cur_node, log_prob = self.pick_node(visit_mask, depot_mask, cur_node, require_prob,
                                                greedy_mode)  # pick action
            # update solution and log_probs
            solutions.append(cur_node)
            log_probs_list.append(log_prob)
            # update collected_prize and mask
            collected_prize += self.prizes[cur_node]
            if require_prob:
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask, depot_mask = self.update_mask(visit_mask, depot_mask, cur_node, collected_prize)
            # check done
            done = self.check_done(cur_node)
        if require_prob:
            return torch.stack(solutions).permute(1, 0), torch.stack(log_probs_list).permute(1,
                                                                                             0)  # shape: [bs, max_seq_len]
        else:
            return torch.stack(solutions).permute(1, 0),None

    def gen_penalty_bool(self, sol, n):
        '''
        Args:
            sol: (width, max_seq_len)
        '''
        width = sol.size(0)
        seq_len = sol.size(1)
        expanded_nodes = torch.arange(n, device=self.device).repeat(width, seq_len, 1)  # (width, seq_len, n)
        expanded_sol = torch.repeat_interleave(sol, n, dim=-1).reshape(width, seq_len, n)
        return (torch.eq(expanded_nodes, expanded_sol) == 0).all(dim=1)

    def gen_penalty(self, solutions, node_penalty):
        '''
        Args:
            solutions: (width, max_len)
        '''
        penalty_bool = self.gen_penalty_bool(solutions, self.n)
        sols_penalty = []
        for idx in range(solutions.size(0)):
            penalty = node_penalty[penalty_bool[idx]].sum()
            sols_penalty.append(penalty)
        return torch.stack(sols_penalty)

    def pick_node(self, visit_mask, depot_mask, cur_node, require_prob, greedy_mode=False):
        log_prob = None
        heatmap = self.heatmap[cur_node]
        dist = (heatmap * visit_mask * depot_mask)
        if not greedy_mode:
            dist = Categorical(dist)
            item = dist.sample()
            log_prob = dist.log_prob(item) if require_prob else None
        else:
            _, item = dist.max(dim=1)
        return item, log_prob  # (bs,)

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
        depot_mask[(~at_depot) * ((visit_mask[:, 1:] == 0).all(dim=1)), 0] = 1
        return visit_mask, depot_mask

    def check_done(self, cur_node):
        # is all at depot ?
        return (cur_node == 0).all()


def gen_pyg_data_pctsp(prizes, penalties, dist_mat, k_sparse):
    n_nodes = prizes.size(0)
    if n_nodes-1<k_sparse:
        k_sparse=n_nodes-1
    topk_values, topk_indices = torch.topk(dist_mat,
                                           k=k_sparse,
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    x = torch.stack((prizes, penalties)).permute(1, 0) # (n+1, 2)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data
def add_padding(pi_all, max_seq_len, n_subset,val_size):
    ret = []
    for subset_pi in pi_all:
        assert subset_pi.size(0) == n_subset
        diff = max_seq_len - subset_pi.size(-1)
        subset_pi = torch.cat([subset_pi, torch.zeros((n_subset, diff), dtype=torch.int64 ,device=subset_pi.device)], dim=1)
        ret.append(subset_pi)
    ret = torch.cat(ret, dim=0) # n_val*n_subset, max_len
    assert ret.shape == (val_size * n_subset, max_seq_len)
    return ret