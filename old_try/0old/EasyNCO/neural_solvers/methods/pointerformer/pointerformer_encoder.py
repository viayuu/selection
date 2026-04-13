from typing import Tuple, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from EasyNCO.utils.utils import getLogger
import revtorch as rv
from EasyNCO.neural_solvers.backbones.Transformer.attention import MultiHeadAttentionLayer
logger = getLogger(__name__)

from EasyNCO.neural_solvers.backbones import FeedForward
class PointerformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int=6,
        n_heads: int=8,
        embedding_dim: int=128,
        input_dim: int=24,
        intermediate_dim: int=512,
        add_init_projection=True,
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.num_hidden_layers = n_layers
        blocks = []
        for _ in range(n_layers):
            f_func = MHABlock(embedding_dim, n_heads)
            g_func = FFBlock(embedding_dim, intermediate_dim)
            # we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x: Tensor, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)   #把aug后拼接的数据映射到128维度
        x = torch.cat([x, x], dim=-1)#又复制一份
        out = self.sequence(x)
        return torch.stack(out.chunk(2, dim=-1))[-1]





class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.mha=MultiHeadAttentionLayer(embed_dim=hidden_size,num_heads=num_heads,qkv_dim=int(hidden_size/num_heads),bias=False,bias_k=False,bias_v=False,bias_combine=False)

    def forward(self, hidden_states: Tensor):

        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        mha_output=self.mha(hidden_states, hidden_states)



        return mha_output

class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        # self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.FF=FeedForward(hidden_size, intermediate_size,activation="Gelu")
        # self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor):
        hidden_states = (
            self.output_layer_norm(hidden_states.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
        # intermediate_output = self.feed_forward(hidden_states)
        # intermediate_output = self.activation(intermediate_output)
        # output1 = self.output_dense(intermediate_output)
        output=self.FF(hidden_states)
        return output