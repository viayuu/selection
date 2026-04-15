import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
    def reset_parameters(self):
        raise NotImplementedError
    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn, udc_flag=False):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.udc_flag = udc_flag
        if udc_flag:
            self.lin_f = nn.ModuleList(
                [
                    nn.Linear(self.units_list[i], self.units_list[i + 1])
                    for i in range(self.depth)
                ]
            )
            self.lin_l = nn.ModuleList(
                [
                    nn.Linear(self.units_list[i], self.units_list[i + 1])
                    for i in range(self.depth)
                ]
            )
            self.lin_v = nn.ModuleList(
                [
                    nn.Linear(self.units_list[i], self.units_list[i + 1])
                    for i in range(self.depth)
                ]
            )
        if udc_flag == 'cvrp':
            self.lin_d = nn.ModuleList(
                [
                    nn.Linear(self.units_list[i], self.units_list[i + 1])
                    for i in range(self.depth)
                ]
            )
    def forward(self, x, k_sparse, **kwargs):
        if self.udc_flag:
            node = kwargs['node']
            solution = kwargs['solution']
            visited = kwargs['visited']
            sample_size = solution.size(0)
            if self.udc_flag == 'cvrp':
                x_depot = node.clone()[0, :][None, None, :].repeat(
                    solution.size(0), 1, 1
                )
            x_first = (
                node.clone()[None, :]
                .repeat(solution.size(0), 1, 1)
                .gather(1, solution[:, 0][:, None, None].expand(-1, -1, node.size(-1)))
            )
            x_last = (
                node.clone()[None, :]
                .repeat(solution.size(0), 1, 1)
                .gather(1, solution[:, 1][:, None, None].expand(-1, -1, node.size(-1)))
            )
            x_visited = node.clone()
        for i in range(self.depth):
            x = self.lins[i](x)
            if self.udc_flag:
                x_first = self.lin_f[i](x_first)
                x_last = self.lin_l[i](x_last)
                if self.udc_flag == 'cvrp':
                    x_depot = self.lin_d[i](x_depot)
                x_visited = self.lin_v[i](x_visited)
            if i < self.depth - 1:
                x = self.act_fn(x)
                if self.udc_flag:
                    x_first = self.act_fn(x_first)
                    x_last = self.act_fn(x_last)
                    if self.udc_flag == 'cvrp':
                        x_depot = self.act_fn(x_depot)
                    x_visited = self.act_fn(x_visited)
            else:
                if self.udc_flag:
                    q = x_visited[None, :].repeat(solution.size(0), 1, 1) * visited[
                        :, :, None
                    ].expand(-1, -1, node.size(-1))
                    q = q.sum(1) / visited[:, :, None].expand(
                        -1, -1, node.size(-1)
                    ).sum(1)
                    if self.udc_flag == 'tsp':
                        q += x_first.squeeze(1) + x_last.squeeze(1)
                    elif self.udc_flag == 'cvrp':
                        q += x_first.squeeze(1) + x_last.squeeze(1) + x_depot.squeeze(1)
                    x = torch.mm(q, x.T).reshape(sample_size, -1, k_sparse)
                    x = torch.softmax(x, dim=-1)
                else:
                    x = x.reshape(-1, k_sparse)
                    x = torch.softmax(x, dim=1)
                    x = x.flatten()
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, k_sparse, depth=3, units=48, preds=1, act_fn='silu', udc_flag=False):
        self.units = units
        self.preds = preds
        self.k_sparse = k_sparse
        self.udc_flag = udc_flag
        if udc_flag:
            super().__init__([self.units] * depth, act_fn, udc_flag=udc_flag)
        else:
            super().__init__([self.units] * depth + [self.preds], act_fn)
    def forward(self, x, **kwargs):
        if self.udc_flag:
            return super().forward(x, self.k_sparse, **kwargs).squeeze(dim=-1)
        return super().forward(x, self.k_sparse).squeeze(dim = -1)


class glop_partition_net(nn.Module):
    def __init__(self, units, feats, k_sparse=50, edge_feats=1, depth=12, udc_flag=False,**kwargs):
        super().__init__()
        self.emb_net = EmbNet(depth=depth, units=units, feats=feats, edge_feats=edge_feats)
        self.par_net_heu = ParNet(units=units, k_sparse=k_sparse, udc_flag=udc_flag)
        self.x_emb = None
        self.udc_flag = udc_flag

    def pre(self, pyg):
        """
        Args:
            pyg: torch_geometric.data.Data instance with x, edge_index, and edge attr
        Returns:
            heu: heuristic vector [n_nodes * k_sparsification,]
        """
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr)

    def forward(self, pyg=None, solution=None, visited=None, selected=None,k_sparse=None):
        '''
        Args:
            pyg: torch_geometric.data.Data instance with x, edge_index, and edge attr
        Returns:
            heu: heuristic vector [n_nodes * k_sparsification,]
        '''
        if self.udc_flag:
            if self.udc_flag == 'tsp':
                solution_cat = torch.cat((solution[:, 0].unsqueeze(-1), solution[:, -1].unsqueeze(-1)), dim=-1)
            if self.udc_flag == 'cvrp':
                solution_cat = torch.cat(
                    (solution[:, 0].unsqueeze(-1), selected), dim=-1
                )
            heu = self.par_net_heu(self.emb, node=self.x_emb, solution=solution_cat, visited=visited)
            return self.x_emb, heu
        else:
            x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
            _, emb = self.emb_net(x, edge_index, edge_attr)#100个点的48维度的emb
            problem_size=x.shape[0]-1
            if k_sparse is not None:
                self.par_net_heu.k_sparse=k_sparse
            if problem_size<self.par_net_heu.k_sparse:
                self.par_net_heu.k_sparse=problem_size
            heu = self.par_net_heu(emb)
            return heu

    @staticmethod
    def reshape(pyg, vector, udc_flag=False):
        '''Turn heu vector into matrix with zero padding 
        '''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        if udc_flag:
            matrix = torch.zeros(size=(vector.size(0), n_nodes, n_nodes), device=device)
            idx = torch.repeat_interleave(torch.arange(vector.size(0)).to(device), repeats=pyg.edge_index[0].shape[0])
            idx0 = pyg.edge_index[0].repeat(vector.size(0))
            idx1 = pyg.edge_index[1].repeat(vector.size(0))
            matrix[idx, idx0, idx1] = vector.view(-1)
            dim = 2
        else:
            matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
            matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
            dim = 1
        try:
            assert (matrix.sum(dim=dim) >= 0.99).all()
        except:
            torch.save(matrix, './error_reshape.pt')
        return matrix
