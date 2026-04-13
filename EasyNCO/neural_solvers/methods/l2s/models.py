import torch.nn.functional as F
import torch
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU

class DGHANlayer(torch.nn.Module):  #CAM
    def __init__(self, in_chnl, out_chnl, dropout, concat, num_heads=2):
        super(DGHANlayer, self).__init__()
        self.dropout = dropout
        self.opsgrp_conv = GATConv(in_chnl, out_chnl, heads=num_heads, dropout=dropout, concat=concat)
        self.mchgrp_conv = GATConv(in_chnl, out_chnl, heads=num_heads, dropout=dropout, concat=concat)

    def forward(self, node_h, edge_index_pc, edge_index_mc):
        node_h_pc = F.elu(self.opsgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_pc))
        node_h_mc = F.elu(self.mchgrp_conv(F.dropout(node_h, p=self.dropout, training=self.training), edge_index_mc))
        node_h = torch.mean(torch.stack([node_h_pc, node_h_mc]), dim=0, keepdim=False)
        return node_h

class DGHAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, layer_dghan=4, num_heads=2):
        super(DGHAN, self).__init__()
        self.layer_dghan = layer_dghan
        self.hidden_dim = hidden_dim

        ## DGHAN conv layers
        self.DGHAN_layers = torch.nn.ModuleList()

        # init DGHAN layer
        if layer_dghan == 1:
            # only DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=False, num_heads=num_heads))
        else:
            # first DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(in_dim, hidden_dim, dropout, concat=True, num_heads=num_heads))
            # following DGHAN layers
            for layer in range(layer_dghan - 2):
                self.DGHAN_layers.append(DGHANlayer(num_heads * hidden_dim, hidden_dim, dropout, concat=True, num_heads=num_heads))
            # last DGHAN layer
            self.DGHAN_layers.append(DGHANlayer(num_heads * hidden_dim, hidden_dim, dropout, concat=False, num_heads=1))

    def forward(self, x, edge_index_pc, edge_index_mc, batch_size):

        # initial layer forward
        h_node = self.DGHAN_layers[0](x, edge_index_pc, edge_index_mc)
        for layer in range(1, self.layer_dghan):
            h_node = self.DGHAN_layers[layer](h_node, edge_index_pc, edge_index_mc)

        return h_node, torch.mean(h_node.reshape(batch_size, -1, self.hidden_dim), dim=1)

class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_gin=4):
        super(GIN, self).__init__()
        self.layer_gin = layer_gin

        ## GIN conv layers
        self.GIN_layers = torch.nn.ModuleList()

        # init gin layer
        self.GIN_layers.append(
            GINConv(
                Sequential(Linear(in_dim, hidden_dim),
                           torch.nn.BatchNorm1d(hidden_dim),
                           ReLU(),
                           Linear(hidden_dim, hidden_dim)),
                eps=0,
                train_eps=False,
                aggr='mean',
                flow="source_to_target")
        )

        # rest gin layers
        for layer in range(layer_gin - 1):
            self.GIN_layers.append(
                GINConv(
                    Sequential(Linear(hidden_dim, hidden_dim),
                               torch.nn.BatchNorm1d(hidden_dim),
                               ReLU(),
                               Linear(hidden_dim, hidden_dim)),
                    eps=0,
                    train_eps=False,
                    aggr='mean',
                    flow="source_to_target")
            )

    def forward(self, x, edge_index, batch):

        hidden_rep = []
        node_pool_over_layer = 0
        # initial layer forward
        h = self.GIN_layers[0](x, edge_index)
        node_pool_over_layer += h
        hidden_rep.append(h)
        # rest layers forward
        for layer in range(1, self.layer_gin):
            h = self.GIN_layers[layer](h, edge_index)
            node_pool_over_layer += h
            hidden_rep.append(h)

        # Graph pool
        gPool_over_layer = 0
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += g_pool

        return node_pool_over_layer, gPool_over_layer