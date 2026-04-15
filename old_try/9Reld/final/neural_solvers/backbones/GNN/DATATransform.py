'''
This file is mainly used to convert data into a format that can be processed by network structures,
primarily suitable for graph data in GNN.
'''

import torch
from torch import Tensor
from torch_geometric.data import Data

def gen_pyg_data_tsp(tsp_coordinates, k_sparse=100, start_node=None, infinity=True, udc_flag=False):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
        k_sparse: int for the sparse graph
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    if k_sparse >= n_nodes:
        k_sparse = n_nodes
    distances = gen_distance_matrix(tsp_coordinates, infinity=infinity)
    topk_values, topk_indices = torch.topk(distances,
                                           k=k_sparse,
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
    ])
    edge_attr = topk_values.reshape(-1, 1)
    if udc_flag:
        edge_attr = -torch.cat((edge_attr, distances[edge_index[0], edge_index[1]].reshape(k_sparse*n_nodes, 1)), dim=1)

    if start_node is None:
        node_feature = tsp_coordinates
    else:
        node_feature = torch.zeros((n_nodes, 1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def gen_pyg_data_cvrp(demands, distances, device, k_sparse=5):
    n = demands.size(0)
    # Sparsify
    topk_values, topk_indices = torch.topk(distances[1:, 1:], k = k_sparse, dim=1, largest=False)
    edge_index_1 = torch.stack([
        torch.repeat_interleave(torch.arange(n-1).to(topk_indices.device), repeats=k_sparse),
        torch.flatten(topk_indices)
    ]) + 1
    edge_attr_1 = topk_values.reshape(-1, 1)
    # Keep all edges connected to depot
    edge_index_2 = torch.stack([
        torch.zeros(n-1, device=device, dtype=torch.long),
        torch.arange(1, n, device=device, dtype=torch.long),
    ])
    edge_attr_2 = distances[1:, 0].reshape(-1, 1)
    edge_index_3 = torch.stack([
        torch.arange(1, n, device=device, dtype=torch.long),
        torch.zeros(n-1, device=device, dtype=torch.long),
    ])
    edge_index = torch.concat([edge_index_1, edge_index_2, edge_index_3], dim=1)
    edge_attr = torch.concat([edge_attr_1, edge_attr_2, edge_attr_2])

    x = demands
    pyg_data = Data(x=x.unsqueeze(1).float(), edge_attr=edge_attr.float(), edge_index=edge_index)
    return pyg_data

def gen_pyg_data_op(tsp_coordinates, prizes, k_sparse):
    n_nodes = len(tsp_coordinates)
    dis_to_depot = (tsp_coordinates - tsp_coordinates[0]).norm(dim=-1)
    x = torch.stack((dis_to_depot, prizes)).T
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances,
                                           k=k_sparse,
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

def gen_pyg_data_pctsp(prizes, penalties, dist_mat):
    n_nodes = prizes.size(0)
    x = torch.stack((prizes, penalties)).permute(1, 0) # (n+1, 2)
    nodes = torch.arange(n_nodes, device=prizes.device) # (n+1,)
    v = nodes.repeat(n_nodes)
    u = torch.repeat_interleave(nodes, n_nodes)
    edge_index = torch.stack([u, v]) # (2, n+1)
    edge_attr = dist_mat.reshape(-1,)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.unsqueeze(-1))
    return pyg_data

def gen_pyg_data_sop(distances, adj):
    edge_index = torch.nonzero(adj).T
    edge_attr = distances[adj.bool()].unsqueeze(-1)
    x = distances[0, :].unsqueeze(-1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

def gen_pyg_data_smtwtp(due_time, weights, processing_time, device):
    n = len(due_time)
    due_time_norm = due_time / n

    x = torch.stack([due_time_norm, weights]).T  # (n, 2)
    x_depot = torch.zeros(size=(1, 2), device=device)
    x = torch.cat([x_depot, x], dim=0)

    _edge_attr = torch.cat([torch.zeros(size=(1,), device=device), processing_time])  # (n+1,)
    edge_attr = torch.repeat_interleave(_edge_attr, n + 1).unsqueeze(-1)  # attr of <i,j> is the processing time of j
    nodes = torch.arange(n + 1, device=device)
    u = nodes.repeat(n + 1)
    v = torch.repeat_interleave(nodes, n + 1)
    edge_index = torch.stack([u, v])
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data

def gen_pyg_data_mkp(prize, weight_matrix):
    device = prize.device
    n = prize.size(0)
    x = weight_matrix
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_attr = prize.repeat(n).unsqueeze(-1)
    pyg_data = Data(x=x, edge_index=torch.stack((u, v)), edge_attr=edge_attr)
    return pyg_data

def gen_pyg_data_bpp(demands, device='cpu'):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = torch.ones((edge_index.size(1), 1))
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr ,edge_index=edge_index)
    return pyg_data

def gen_distance_matrix(coordinates, infinity = True):
    '''
    This function is used to generate the distance matrix between nodes in the routing problem.
    Note that the values for the diagonal elements may be different;
    some are set to a very large value, while others are set to a very small value close to 0.

    Args:
        coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    if infinity:
        distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9
    else:
        distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10
    return distances

def reformat(price: Tensor, weight: Tensor):
    '''
    This function is used to process data for the transformer.
    Concatenate price tensor and weight tensor into input features for Transformer.
    '''
    src = torch.cat((price.T.unsqueeze(-1), weight.T), dim=-1)
    src.unsqueeze_(1)
    return src # [seq_len, batch_size=1, emb_size=m+1]