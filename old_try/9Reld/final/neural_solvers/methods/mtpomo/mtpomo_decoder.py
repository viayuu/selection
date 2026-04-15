import torch
from torch import nn
import torch.nn.functional as F

from EasyNCO.neural_solvers.backbones.Transformer.attr_component import multi_head_attention,reshape_by_heads

class MTPOMODecoder(nn.Module):
    def __init__(self,embedding_dim,head_num,qkv_dim,sqrt_embedding_dim,logit_clipping):
        super().__init__()
        self.sqrt_embedding_dim=sqrt_embedding_dim
        self.logit_clipping=logit_clipping
        self.head_num=head_num
        self.Wq_last = nn.Linear(embedding_dim+4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.head_num

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.head_num
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.head_num
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, time,length, route_open, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.head_num

        #  Multi-Head Attention
        #######################################################
        encoded_last_node=encoded_last_node.squeeze(2)
        input_cat = torch.cat((encoded_last_node, load[:, :, None], time[:, :, None],length[:, :, None], route_open[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+3)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q_last shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        #print("ninf_mask",ninf_mask)
        out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)
        #print("score",score)

        sqrt_embedding_dim = self.sqrt_embedding_dim
        logit_clipping = self.logit_clipping

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        #print("score_scaled",score_scaled)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs