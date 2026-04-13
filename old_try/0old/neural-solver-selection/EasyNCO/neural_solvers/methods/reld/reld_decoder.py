from typing import Tuple
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn
from EasyNCO.neural_solvers.utils import dynamic_embedding, special_selected
from EasyNCO.neural_solvers.backbones import (reshape_by_heads,
                                           multi_head_attention,
                                           Compatibility)
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


class ReLDDecoder(nn.Module):
    """
    ReLD Decoder - 基于AM解码器的增强版本，包含ReLD的三大核心改进：
    1. IDT (Identity Mapping) - 直接融合上下文信息
    2. FF (Feed Forward) - 增强非线性建模能力
    3. 距离启发式 - 引入物理距离先验

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        env_name: Name of the environment used to initialize embeddings
        context_embedding: Context embedding module
        linear_bias: Whether to use a bias in the linear layer
        use_graph_mean: Whether to use the graph context
        am_mode: Whether to use the AM mode
        first_placeholder: Whether to use the first placeholder network
    """

    def __init__(
        self,
        env_name: str = "cvrp",
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = None,
        key_dim: int = None,
        logit_clipping: float = 50,  # ReLD使用更大的截断值
        context_embedding: nn.Module = None,
        linear_bias: bool = False,
        use_graph_mean: bool = False,  # ReLD不使用图均值
        am_mode: bool = True,
        first_placeholder: bool = False,  # ReLD不使用first placeholder
        feedforward_hidden: int = 512,
        sub_glop: bool = False,
        **kwargs
    ):
        super().__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.first_placeholder = first_placeholder
        self.logit_clipping = logit_clipping
        self.feedforward_hidden = feedforward_hidden
        self.sub_glop = sub_glop
        self.sqrt_embedding_dim = embed_dim ** (1/2)

        if key_dim is None:
            self.key_dim = embed_dim
        else:
            self.key_dim = key_dim

        if qkv_dim is None:
            self.qkv_dim = embed_dim // num_heads
        else:
            self.qkv_dim = qkv_dim

        assert embed_dim % num_heads == 0

        # 动态嵌入层
        # self.dynamic_embedding = (dynamic_embedding(self.env_name,
        #                                     {"embed_dim": embed_dim,
        #                                            "first_placeholder": first_placeholder})
        #     if context_embedding is None
        #     else context_embedding
        # )

        # 兼容性层
        # self.compatibility = Compatibility(
        #         embed_dim=embed_dim,
        #         n_heads=num_heads,
        #         qkv_dim=qkv_dim,
        #         key_dim=self.key_dim,
        #         am_mode=am_mode,
        #         sub_glop=self.sub_glop,
        # )

        if self.sub_glop:
            self.project_node_embed= nn.Linear(self.embed_dim, 3*self.embed_dim, bias=False)
        else:
            self.Wk = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=False)
            self.Wv = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=False)

        self.Wq_last = nn.Linear(self.embed_dim + 1, self.num_heads * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.num_heads * self.qkv_dim, self.embed_dim)

        self.graph_mean_embedding = 0.0  # 保持与AM结构一致，但不使用

        self.encoded_nodes = None  # saved encoded nodes
        self.sing_head_k=None

        # ReLD核心组件1: IDT (Identity Mapping) - 容量映射
        self.capacity_mapping = nn.Linear(1, self.embed_dim, bias=False)

        # ReLD核心组件2: FF (Feed Forward) - 与作者架构一致的独立线性层
        self.feed_forward = FeedForward(self.embed_dim, self.feedforward_hidden)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None


    def set_kv(self, encoded_nodes, weights=None):
        # encoded_nodes.shape: (batch, problem, embedding)
        if self.sub_glop:
            k,v,self.sing_head_k=self.project_node_embed(encoded_nodes).chunk(3,dim=-1)
            self.k=reshape_by_heads(k,head_num=self.num_heads)
            self.v=reshape_by_heads(v,head_num=self.num_heads)
            # shape: (batch, head_num, n_start, qkv_dim)

        else:
            if weights is None:
                self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.num_heads)
                self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.num_heads)
                # shape: (batch, head_num, n_start, qkv_dim)
            else:
                self.k = reshape_by_heads(F.linear(encoded_nodes, weights['decoder.Wk.weight'], bias=None),
                                          head_num=self.num_heads)
                self.v = reshape_by_heads(F.linear(encoded_nodes, weights['decoder.Wv.weight'], bias=None),
                                          head_num=self.num_heads)
        self.encoded_nodes = encoded_nodes
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, problem, embedding)

    def forward(self, encoded_last_node: Tensor, load: Tensor, td: TensorDict, first_mode: str = None, ninf_mask=None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state
        Args:
            encoded_last_node: Tensor with the embedding of the last selected node
            load: Tensor with the current load of the vehicle
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """
        head_num = self.num_heads
        encoded_last_node=encoded_last_node.squeeze(2)
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)

        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # IDT
        mh_atten_out = mh_atten_out + encoded_last_node + self.capacity_mapping(load[:,:,None].clone())
        # FF
        q_refined = self.feed_forward(mh_atten_out) + mh_atten_out

        #  Single-Head Attention, for probability calculation
        score = torch.matmul(q_refined, self.single_head_key)
        # shape: (batch, pomo, problem)
 
        sqrt_embedding_dim = self.embed_dim ** 0.5
        logit_clipping = self.logit_clipping

        score_scaled = score / sqrt_embedding_dim

        # ReLD核心组件3: 距离启发式
        # 从 kwargs 中获取 cur_dist
        cur_dist = kwargs.get('cur_dist', None)
        
        # -log(dist) heuristic

        score = score_scaled - torch.log(cur_dist)


        score = logit_clipping * torch.tanh(score)

        score_masked = score + ninf_mask

        probs = F.softmax(score_masked, dim=2)

        selected = None


        # if probs is None:
            # 获取当前状态信息
            # load = td.get("load", None)  # 当前车辆负载
            # ninf_mask = td.get("ninf_mask", None)  # 不可访问节点掩码
            # q = reshape_by_heads(self.graph_mean_embedding + self.dynamic_embedding(self.encoded_nodes, td),
            #                      head_num=self.num_heads)
            # 获取前一节点的嵌入
            # current_node = td.get("current_node", None)
            # if current_node is not None and hasattr(self.encoded_nodes, 'gather'):
            #     # 从编码节点中获取当前节点嵌入
            #     batch_size = current_node.size(0)
            #     pomo_size = current_node.size(1)
            #     embedding_dim = self.embed_dim
                
            #     gathering_index = current_node[:, :, None].expand(batch_size, pomo_size, embedding_dim)
            #     encoded_last_node = self.encoded_nodes.gather(dim=1, index=gathering_index)
            # else:
            #     # 如果没有当前节点信息，使用动态嵌入
            #     encoded_last_node = self.dynamic_embedding(self.encoded_nodes, td)
            # mask = td["next"]["ninf_mask"].clone()
            # if self.sub_glop :
            #     assert mask.shape[1]==2,"In glop,The AM pomo_size should be 2"
            #     problem_size = mask.shape[2]
            #     if td["next"]["selected_count"][0,0]==0:
            #         mask[:, 0, 1:problem_size] = float("-inf")
            #         mask[:, 1, 0:problem_size - 1] = float("-inf")
            #     else:
            #         if td["next"]["selected_count"][0,0]<problem_size-1:
            #             mask[:,0,problem_size-1]=float("-inf")
            #             mask[:,1,0]=float("-inf")
            #         else:
            #             mask[:,0,problem_size-1]=0.
            #             mask[:,1,0]=0.
            # out_concat = multi_head_attention(q, self.k, self.v, mask)
            # if weights is None:
            #     mh_atten_out = self.multi_head_combine(out_concat)
            # else:
            #     mh_atten_out = F.linear(out_concat, weights['decoder.multi_head_combine.weight'],
            #                             bias=weights['decoder.multi_head_combine.bias'])
            # penalty = kwargs.get('penalty', 0)

            # ReLD核心组件1: IDT (Identity Mapping)
            # 直接融合原始节点嵌入和容量信息



            # mh_atten_out = mh_atten_out + encoded_last_node
            
            # if load is not None:
            #     # 添加容量映射
            #     capacity_embed = self.capacity_mapping(load[:, :, None].clone())
            #     mh_atten_out = mh_atten_out + capacity_embed

            # # ReLD核心组件2: FF (Feed Forward)
            # # 引入非线性建模能力
            # q_refined = self.feed_forward(mh_atten_out) + mh_atten_out

            # # 兼容性计算
            # # ReLD核心组件3: 距离启发式
            # cur_dist = kwargs.get('cur_dist', None)  # 当前距离信息
            
            # # 基础得分计算
            # score = torch.matmul(q_refined, self.single_head_key)
            # sqrt_embedding_dim = self.embed_dim ** 0.5
            # score_scaled = score / sqrt_embedding_dim
            
            # # 添加距离启发式：-log(dist)
            # if cur_dist is not None and (cur_dist > 0).all():
            #     score = score_scaled - torch.log(cur_dist)
            # else:
            #     score = score_scaled

            # # logit截断和稳定性处理
            # score = self.logit_clipping * torch.tanh(score)
            # score_masked = score + ninf_mask

            # # 最终概率
            # probs = F.softmax(score_masked, dim=2)
            # selected = None

        return selected,probs
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, feedforward_hidden):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, feedforward_hidden)
        self.W2 = nn.Linear(feedforward_hidden, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))