
import torch.nn as nn
import torch

from EasyNCO.neural_solvers.utils import GraphMeanEmbedding
from EasyNCO.neural_solvers.backbones import TransformerNet,reshape_by_heads,multi_head_attention
from EasyNCO.neural_solvers.backbones.Transformer.attr_component import positional_encoding_init

from EasyNCO.neural_solvers.methods.lih.actor import LIH_Actor


class LIHPolicy(nn.Module):
    VEHICLE_CAPACITY = 1.

    def __init__(
        self,
        env_name: str = "tsp",  # tsp or cvrp (currently only tsp and cvrp are supported)
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        num_heads: int = 1,
        qkv_dim: int = 128,
        num_encoder_layers: int = 3,
        normalization: str = "batch", # norm can be None, batch, or instance
        feedforward_hidden: int = 512,
        **kwargs
    ):
        super(LIHPolicy, self).__init__()

        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim
        self.feedforward_hidden = feedforward_hidden
        self.num_encoder_layers = num_encoder_layers
        self.decode_type = None
        self.env_name = env_name
        self.num_heads = num_heads

        # cvrp: x_t-1 , x_t , x_t+1 , demand 组成一个点，共7维
        if self.env_name == 'tsp':
            node_dim = 2
        elif self.env_name == 'cvrp':
            node_dim = 7

        self.init_embed = nn.Linear(node_dim, embed_dim)

        self.embedder = LIH_Actor(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            normalization=normalization,
            env_name=env_name,
        )


    def set_decoder_strategy(self, strategy: str = "sampling"):
        self.decoder_strategy = strategy

    def forward(self,td,**kwargs):

        td['init_embed'] = self.init_embed(td['input_info'])

        if self.env_name == 'tsp':
            soft_max, exchange , _ = self.embedder(td)
            # log_likelihood
            ll = soft_max.squeeze()

            td.update(
                {
                    'log_likelihood' : ll,
                    'exchange' : exchange[0],
                }
            )

            return td


        elif self.env_name == 'cvrp':
            policy_old = kwargs.get('policy_old',True)

            if policy_old :
                soft_max, exchange, action = self.embedder(td,policy_old=policy_old)
                ll = soft_max.squeeze()

                td.update(
                    {
                        'log_likelihood': ll,
                        'exchange': exchange[0],
                        'action' :action,  # row*n + col
                    }
                )

                return td

            else:
                soft_max = self.embedder(td,policy_old=policy_old)
                ll = soft_max.squeeze()
                td['log_likelihood'] = ll
                td.update(
                    {
                        'log_likelihood': ll,
                    }
                )
                return td

    def get_input_and_pe(self,td):
        if self.env_name == "tsp":
            input_info, pos_enc = tsp_embedding(td['locs'], td['solution'])
        elif self.env_name == "cvrp":
            input_info, pos_enc = cvrp_embedding(td['locs'], td['solution'])

        return input_info, pos_enc




def tsp_embedding(input, rec):
    """
    input: (batch_size, graph_size, 2) - 每个节点的2D坐标
    rec: (batch_size, graph_size) - 访问顺序

    返回:
    node_2_cor: (batch_size, graph_size, 2) - 访问顺序对应的坐标
    pos_enc: (batch_size, graph_size, 128) - 位置编码
    """
    bs, gs = rec.shape

    # 计算位置编码
    enc = positional_encoding_init(gs, 128).cuda()  # (graph_size, 128)
    enc_b = enc.expand(bs, gs, 128)  # 扩展 batch 维度

    # 获取访问顺序的索引
    seq_tensor_index = rec.long()

    # 获取每个节点在路径中的位置
    cor = torch.argsort(seq_tensor_index, dim=1)  # (batch_size, graph_size)

    # 取出每个节点的前继节点索引
    pre_indices = seq_tensor_index.gather(1, cor)  # (batch_size, graph_size)

    # 获取前继节点的坐标
    node_2_cor = input.gather(1, pre_indices.unsqueeze(-1).expand(-1, -1, 2))

    # 获取位置编码
    pos_enc = enc_b.gather(1, cor.unsqueeze(-1).expand(-1, -1, 128))

    return node_2_cor, pos_enc


#将x-1,x,x+1组成一个节点信息
def cvrp_embedding(input, seq_tensor):
    #input dict 3
    #'locs' (batch,problem*2,2)
    #'demand' (batch,problem*2)
    #'depot' (batch,2)
    #seq_tensor (batch,problem*2) solution 从1开始
    bs, graph_2 = seq_tensor.size()

    loc_with_depot = input[:,:,0:2]

    enc = positional_encoding_init(graph_2, 128)

    enc = enc.cuda()

    enc_b = enc.expand(bs, graph_2, 128)

    seq_tensor_index = torch.cat((seq_tensor[:, -1][:, None], seq_tensor, seq_tensor[:, 0][:, None]), 1)

    cor_3_demand = []

    pos_enc = []

    for i in range(1, graph_2 + 1):

        cor = torch.nonzero(seq_tensor == i)

        pre = seq_tensor_index[cor[:, 0], cor[:, 1]]
        mid = seq_tensor_index[cor[:, 0], cor[:, 1] + 1]
        las = seq_tensor_index[cor[:, 0], cor[:, 1] + 2]

        dem = input[:,:,2].gather(1, mid[:, None] - 1)   #batch,1

        cor_indice = torch.cat((pre[:, None] - 1, mid[:, None] - 1, las[:, None] - 1), 1)

        cor_single = loc_with_depot.gather(1, cor_indice[..., None].expand(*cor_indice.size(),
                                                                           loc_with_depot.size(-1)))

        cor_single_fla = cor_single.view(cor_single.size(0), -1)

        cor_with_demand = torch.cat((cor_single_fla, dem), 1)

        cor_3_demand.append(cor_with_demand)

        single_pos = enc_b.gather(1, cor[:, 1][:, None][..., None].expand(bs, 1, 128))

        pos_enc.append(single_pos.squeeze())

    return torch.stack(cor_3_demand, 1), torch.stack(pos_enc, 1)




