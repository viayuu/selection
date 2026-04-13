import math
import torch
from torch_geometric.data import Data
from torch.distributions import Categorical

K_SPARSE = {
    1000: 100,
    2000: 200,
    5000: 200,
    7000: 200
}
capacity_mapping = {
    1000: 200.,
    2000: 300.,
    5000: 300.,
    7000: 300.
}


def cvrp_solution_heatmap(problem, main_net, n_subset,k_sparse=None,Train_flag=False):
    '''
    Solve a batch of data using a heatmap approach.

    '''
    problem=problem.squeeze()
    problem_size=problem.shape[0]-1
    capacity=capacity_mapping[problem_size]
    coors=problem[:,0:2]
    demands=problem[:,2]*capacity
    if Train_flag:
        main_net.train()
        heatmap = infer_heatmap_cvrp(main_net, coors, demands, capacity, k_sparse=k_sparse)
        sampler = Sampler_cvrp(demands, heatmap, capacity, n_subset, coors.device)
        routes, probs = sampler.gen_subsets(require_prob=True, greedy_mode=False)  # n_subset, max_len
    else:
        with torch.no_grad():
            main_net.eval()#改为train()性能会好些
            heatmap = infer_heatmap_cvrp(main_net, coors, demands, capacity, k_sparse=k_sparse)
        sampler = Sampler_cvrp(demands, heatmap, capacity, n_subset, coors.device)
        routes, probs = sampler.gen_subsets(require_prob=False, greedy_mode=True)  # n_subset, max_len
    return routes,probs


def cvrp_trans_tsp(routes, min_reviser_size=20):
    tsp_pis = []
    n_tsps_per_route = []
    for route in routes:
        start = 0
        sub_route_count = 0
        for idx, node in enumerate(route):
            if idx == 0:
                continue
            if node == 0:
                if route[idx-1] != 0:
                    tsp_pis.append(route[start: idx])
                    sub_route_count += 1
                start = idx
        n_tsps_per_route.append(sub_route_count)
    max_tsp_len = max([len(tsp_pis[i]) for i in range(len(tsp_pis))])
    max_tsp_len = max(min_reviser_size, max_tsp_len)
    padded_tsp_pis = []
    for pi in tsp_pis:
        padded_pi = torch.nn.functional.pad(pi, (0, max_tsp_len-len(pi)), mode='constant', value=0)#根据最长的把后面补0
        padded_tsp_pis.append(padded_pi)
    padded_tsp_pis = torch.stack(padded_tsp_pis)
    return padded_tsp_pis,n_tsps_per_route




def infer_heatmap_cvrp(model, coors, demand, capacity, k_sparse=None, is_cvrplib=False):
    n = demand.size(0)-1
    k_sparse = K_SPARSE[n] if k_sparse is None else k_sparse
    pyg_data = gen_pyg_data(coors, demand, capacity, k_sparse, cvrplib=is_cvrplib)
    heatmap = model(pyg_data,k_sparse=k_sparse)
    heatmap = heatmap / (heatmap.min() + 1e-5)
    heatmap = model.reshape(pyg_data, heatmap) + 1e-5
    return heatmap

def gen_cos_sim_matrix(shift_coors):
    dot_products = torch.mm(shift_coors, shift_coors.t())
    magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=1)).unsqueeze(1)
    magnitude_matrix = torch.mm(magnitudes, magnitudes.t()) + 1e-5
    cosine_similarity_matrix = dot_products / magnitude_matrix
    return cosine_similarity_matrix

def gen_pyg_data(coors, demand, capacity, k_sparse, cvrplib=False):
    n_nodes = demand.size(0)
    norm_demand = demand / capacity
    shift_coors = coors - coors[0]#坐标移动到仓库节点为中心
    _x, _y = shift_coors[:, 0], shift_coors[:, 1]
    r = torch.sqrt(_x**2 + _y**2)#客户节点到仓库节点的距离
    theta = torch.atan2(_y, _x)#极角
    x = torch.stack((norm_demand, r, theta)).transpose(1, 0)#需求，极坐标
    cos_mat = gen_cos_sim_matrix(shift_coors) #余弦相似度矩阵
    if n_nodes-1<k_sparse:
        k_sparse=n_nodes-1
    if cvrplib:
        cos_mat = (cos_mat + cos_mat.min()) / cos_mat.max()
        euc_mat = torch.norm(coors[:, None] - coors, dim=2, p=2)
        euc_aff = 1 - euc_mat
        topk_values, topk_indices = torch.topk(cos_mat + euc_aff,
                                            k=k_sparse,
                                            dim=1, largest=True)
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                    repeats=k_sparse),
            torch.flatten(topk_indices)
            ])
        edge_attr1 = euc_aff[edge_index[0], edge_index[1]].reshape(k_sparse*n_nodes, 1)
        edge_attr2 = cos_mat[edge_index[0], edge_index[1]].reshape(k_sparse*n_nodes, 1)
    else:
        topk_values, topk_indices = torch.topk(cos_mat,
                                        k=k_sparse,
                                        dim=1, largest=True)#余弦相似度最大K个值
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                    repeats=k_sparse),
            torch.flatten(topk_indices)
            ])#这100个最近的边的
        edge_attr1 = topk_values.reshape(-1, 1)
        edge_attr2 = cos_mat[edge_index[0], edge_index[1]].reshape(k_sparse*n_nodes, 1)
    edge_attr = torch.cat((edge_attr1, edge_attr2), dim=1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

class Sampler_cvrp():
    def __init__(self, demand, heatmap, capacity, bs, device):
        self.n = demand.size(0)
        self.demand = demand
        self.heatmap = heatmap
        self.capacity = capacity
        self.max_vehicle = math.ceil(sum(self.demand) / capacity) + 1
        self.total_demand = self.demand.sum()
        self.bs = bs
        self.ants_idx = torch.arange(bs)
        self.device = device

    def gen_subsets(self, require_prob=False, greedy_mode=False):
        if greedy_mode:
            assert not require_prob
        actions = torch.zeros((self.bs,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.bs,), device=self.device)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

        vehicle_count = torch.zeros((self.bs,), device=self.device)
        demand_count = torch.zeros((self.bs,), device=self.device)
        depot_mask, vehicle_count, demand_count = self.update_depot_mask(vehicle_count, demand_count, actions,
                                                                         capacity_mask, visit_mask)

        paths_list = [actions]
        log_probs_list = []
        done = self.check_done(visit_mask, actions)
        while not done:
            actions, log_probs = self.pick_node(actions, visit_mask, capacity_mask, depot_mask, require_prob,
                                                greedy_mode)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            depot_mask, vehicle_count, demand_count = self.update_depot_mask(vehicle_count, demand_count, actions,
                                                                             capacity_mask, visit_mask)
            done = self.check_done(visit_mask, actions)
        if require_prob:
            return torch.stack(paths_list).permute(1, 0), torch.stack(log_probs_list).permute(1, 0)
        else:
            return torch.stack(paths_list).permute(1, 0),None

    def pick_node(self, prev, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode=False):
        log_prob = None
        heatmap = self.heatmap[prev]
        dist = (heatmap * visit_mask * capacity_mask * depot_mask)
        if not greedy_mode:
            try:
                dist = Categorical(dist)
                item = dist.sample()
                log_prob = dist.log_prob(item) if require_prob else None
            except:
                dist = torch.softmax(torch.log(dist), dim=1)
                item = torch.multinomial(dist, num_samples=1).squeeze()
                log_prob = torch.log(dist[torch.arange(self.bs), item])
        else:
            _, item = dist.max(dim=1)
        return item, log_prob

    def update_depot_mask(self, vehicle_count, demand_count, actions, capacity_mask, visit_mask):
        depot_mask = torch.ones((self.bs, self.n), device=self.device)
        # update record
        vehicle_count[actions == 0] += 1
        demand_count += self.demand[actions]
        remaining_demand = self.total_demand - demand_count
        # mask
        depot_mask[remaining_demand > self.capacity * (self.max_vehicle - vehicle_count), 0] = 0
        # unmask
        depot_mask[((visit_mask[:, 1:] * capacity_mask[:, 1:]) == 0).all(dim=1), 0] = 1
        # depot_mask[(capacity_mask[:, 1:]==0).all(dim=1), 0] = 1
        return depot_mask, vehicle_count, demand_count

    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.bs, device=self.device), actions] = 0
        visit_mask[:, 0] = 1  # depot can be revisited with one exception
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0  # one exception is here
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity):
        capacity_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        # update capacity
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity  # (bs,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.n)  # (bs, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.bs, 1)  # (bs, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0

        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
