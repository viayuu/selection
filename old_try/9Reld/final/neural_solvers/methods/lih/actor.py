from typing import Tuple, Optional
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch
import numpy as np
import torch.nn.functional as F

from EasyNCO.neural_solvers.backbones import TransformerNet,reshape_by_heads


class LIH_Actor(nn.Module):
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
        linear_bias: bool = False,
        net: nn.Module = None,
        # GraphMeanEmbedding: nn.Module = None,
    ):
        super(LIH_Actor, self).__init__()

        self.env_name = env_name
        self.max_timescale = 10000.0
        self.min_timescale = 1.0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.linear_bias = linear_bias
        self.num_encoder_layers = num_encoder_layers


        self.net = (TransformerNet(
                    num_encoder_layers,
                    num_heads,
                    qkv_dim,
                    embed_dim,
                    normalization,
                    feedforward_hidden,
                    bias_combine=False
                ) if net is None
                else net
        )

        self.Wq = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=linear_bias)
        self.Wk = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=linear_bias)

        self.maxpooling = nn.Linear(embed_dim, embed_dim, bias=False)

        self.project_node = nn.Linear(embed_dim, embed_dim, bias=False)


    def forward(self, td: TensorDict, mask: Optional[Tensor] = None,**kwargs):

        from torch.cuda.amp import autocast as autocast
        with autocast(enabled=False):
            out = self.net(td['init_embed']+td['pos_enc'])
            graph_embed = out.max(1)[0]

            # batch,1,embed_dim
            fixed_context = self.maxpooling(graph_embed)[:, None, :]

            node_feature = self.project_node(out)

            fusion = node_feature + fixed_context.expand_as(node_feature)

            q = reshape_by_heads(self.Wq(fusion),head_num=self.num_heads)
            k = reshape_by_heads(self.Wk(fusion),head_num=self.num_heads)

            #计算log softmax和softmax  compatibility layer
            if self.env_name =='tsp':
                log_att_s,att_s = multi_head_attention(q,k,None,LIH=True,exchange=td['exchange'])
            elif self.env_name == 'cvrp':
                if 'exchange' in td.keys():
                    log_att_s, att_s = multi_head_attention(q, k, None, LIH=True, exchange=td['exchange'],
                                                        che_mask=td['che_mask'].reshape(-1))
                else:
                    log_att_s, att_s = multi_head_attention(q, k, None, LIH=True,
                                                            che_mask=td['che_mask'].reshape(-1))

            n = q.size(-2) #graph_size

            if self.env_name == 'tsp':
                inde = att_s.squeeze().multinomial(1)
                softmax_max = log_att_s.squeeze().gather(1, inde)
                col = inde % n
                row = inde // n
                exc = torch.cat((row, col), -1)
                exc = exc[None, :, :]

                return (
                    softmax_max,  # 每个最大的概率取log
                    exc,  # 点对 exchange
                    inde  # action
                )
            elif self.env_name == 'cvrp':
                policy_old = kwargs.get("policy_old")
                if policy_old :
                    inde = att_s.squeeze().multinomial(1)
                    softmax_max = log_att_s.squeeze().gather(1, inde)
                    col = inde % n
                    row = inde // n
                    exc = torch.cat((row, col), -1)
                    exc = exc[None, :, :]

                    return (
                        softmax_max,  # 每个最大的概率取log
                        exc,  # 点对 exchange
                        inde  #action
                    )
                else:
                    softmax_max = log_att_s.squeeze().gather(1, td['action'])
                    return (
                        softmax_max
                    )


def multi_head_attention(q, k, v,score_dim=None,**kwargs):
    batch_size = q.size(0)
    head_num = q.size(-3)
    n = q.size(-2)
    key_dim = q.size(-1)

    score = torch.matmul(q, k.transpose(-2, -1))
    score_scaled = score / torch.sqrt(
            torch.tensor(key_dim if score_dim is None else score_dim, dtype=torch.float)
    )

    score_scaled = (F.tanh(score_scaled) * 10.).permute(1, 0, 2, 3)

    # mask for dia
    mask_dia = torch.eye(n).view(1, 1, n, n).expand_as(score_scaled)

    che_mask = kwargs.get('che_mask', None)
    if che_mask is not None:
        mask_dia = torch.tril(torch.ones(n, n)).view(1, 1, n, n).expand_as(
            score_scaled)
        score_scaled[mask_dia.bool()] = -np.inf
        tt = score_scaled[~mask_dia.bool()].clone()
        tt[~che_mask.bool()] = -np.inf
        score_scaled[~mask_dia.bool()] = tt
        if n == 128:  # cvrp100
            score_scaled[:, :, -28:, :] = -np.inf
        else:
            score_scaled[:, :, -(n // 2):, :] = -np.inf
    else:
        score_scaled[mask_dia.bool()] = -np.inf

    exchange = kwargs.get('exchange', torch.zeros((batch_size, 2), dtype=torch.long))

    is_all_zero = torch.all(exchange == 0).item()

    if not is_all_zero:
        score_scaled[0][torch.arange(batch_size), exchange[:, 1], exchange[:, 0]] = -np.inf
        score_scaled[0][torch.arange(batch_size), exchange[:, 0], exchange[:, 1]] = -np.inf

    if n == 128:
        score_scaled[:, :, :, 100] = -np.inf

    im = score_scaled.view(head_num, batch_size, -1)
    im_l = F.log_softmax(im, dim=-1)  # for sample

    im_s = F.softmax(im, dim=-1)  # for sample

    return im_l, im_s