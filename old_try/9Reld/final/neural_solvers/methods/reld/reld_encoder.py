from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
import torch
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class ReLDEncoder(nn.Module):
    """
    ReLD Encoder - 与作者CVRP_Encoder架构完全一致
    保持ReLD的核心改进：移除归一化层，简化编码器负担
    """

    def __init__(self, **model_params):
        super(ReLDEncoder, self).__init__()
        self.model_params = model_params
        
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params.get('encoder_layer_num', 6)

        # 🔥 与作者完全一致：直接线性嵌入层
        self.embedding_depot = nn.Linear(2, embedding_dim)    # 仓库节点: (x, y)
        self.embedding_node = nn.Linear(3, embedding_dim)     # 客户节点: (x, y, demand)
        
        # 🔥 与作者一致：层堆叠结构
        self.layers = nn.ModuleList([ReLDEncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, td: TensorDict, mask: Optional[Tensor] = None, weights=None) -> Tuple[Tensor, Tensor]:
        """
        与作者CVRP_Encoder forward方法完全一致
        """
        # 提取数据 - 适配TensorDict到作者格式
        depot_xy = td["locs"][:, :1, :-1]  # 仓库坐标 (batch, 1, 2)
        node_xy_demand = td["locs"][:, 1:, :]     # 客户坐标和需求 (batch, problem, 3)
        
        # 与作者一致的嵌入处理
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)
        
        # 通过所有编码器层
        for layer in self.layers:
            out = layer(out)
        
        # 返回编码结果 - 与平台兼容
        return out, out


class ReLDEncoderLayer(nn.Module):
    """
    ReLD编码器层 - 与作者EncoderLayer架构完全一致
    保持ReLD的改进：移除归一化层
    """

    def __init__(self, **model_params):
        super(ReLDEncoderLayer, self).__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        # 🔥 与作者完全一致：多头注意力机制
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # 🔥 ReLD改进：移除归一化，直接使用FeedForward
        self.feed_forward = ReLDFeedForward(**model_params)

    def forward(self, input1):
        """
        与作者EncoderLayer forward方法完全一致
        """
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # 多头注意力计算
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        # 残差连接1：input + multi_head_out
        out1 = input1 + multi_head_out
        
        # 🔥 ReLD改进：直接前馈处理，无归一化
        out2 = self.feed_forward(out1)
        
        # 残差连接2：out1 + out2
        out3 = out1 + out2

        return out3
        # shape: (batch, problem, embedding)


class ReLDFeedForward(nn.Module):
    """
    ReLD前馈网络 - 与作者FeedForward架构完全一致
    使用W1/W2命名以匹配作者权重
    """

    def __init__(self, **model_params):
        super(ReLDFeedForward, self).__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params.get('ff_hidden_dim', 512)

        # 🔥 与作者完全一致：使用W1/W2命名
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))


########################################
# 工具函数 - 与作者代码完全一致
########################################

def reshape_by_heads(qkv, head_num):
    """
    重塑张量以适应多头注意力
    与作者CVRPModel.py中的函数完全一致
    """
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    if len(qkv.shape) == 4:
        qkv = qkv.reshape(qkv.size(0) * qkv.size(1), qkv.size(2), qkv.size(3))
        # shape: (batch * multi, n, head_num*key_dim)
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    多头注意力计算
    与作者CVRPModel.py中的函数完全一致
    """
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat