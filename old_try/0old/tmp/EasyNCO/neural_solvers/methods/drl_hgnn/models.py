import torch.nn as nn
import torch
from torch.nn import Identity
import torch.nn.functional as F

class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch, batch_idxes, feat):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        #   W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if not hasattr(self, 'fc_src'):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_dst)
        else:
            # Deprecated in final experiment
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        feat_edge = self.fc_edge(feat[2].unsqueeze(-1))

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)
        el_add_ee = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a)
        ekk = self.leaky_relu(er + er)

        # Normalize attention coefficients
        mask = torch.cat((ope_ma_adj_batch[batch_idxes].unsqueeze(-1)==1,
                          torch.full(size=(ope_ma_adj_batch[batch_idxes].size(0), 1,
                                           ope_ma_adj_batch[batch_idxes].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        e[~mask] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        nu_k_prime = torch.sigmoid(b+c)
        return nu_k_prime


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))

            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        b = torch.sum(a, dim=-2)
        c = self.project(b)
        return c

class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # Calculate an return operation embedding
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime