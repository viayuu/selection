import torch
from torch import nn
from EasyNCO.neural_solvers.backbones.Transformer.attr_component import reshape_by_heads,multi_head_attention
from EasyNCO.neural_solvers.backbones import TransformerNet, FeedForward
from EasyNCO.neural_solvers.methods.mvmoe.moe_layer import Add_And_Normalization_Module


class MTPOMOEncoder(nn.Module):
    def __init__(self, embedding_dim,ff_hidden_dim,encoder_layer_num,head_num=8,qkv_dim=16,normalization="batch"):
        super().__init__()

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers=TransformerNet(num_layers=encoder_layer_num,num_heads=head_num,qkv_dim=qkv_dim,
                                    embed_dim=embedding_dim,normalization=normalization,feedforward_hidden=ff_hidden_dim,
                                    bias=False,bias_k=False,bias_v=False,bias_combine=True)
        # self.layers = nn.ModuleList([EncoderLayer(embedding_dim,head_num,qkv_dim,ff_hidden_dim,norm=normalization) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_TW):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand_TW)
        # input shape: (batch, problem, 5)
        # 5 features are: x_coord, y_coord, demands, earlyTW, lateTW
        # embedded_node shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)
        out=self.layers(out)
        return out

