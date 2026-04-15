import torch
from torch import nn
import torch.nn.functional as F

from EasyNCO.neural_solvers.backbones.Transformer.attr_component import reshape_by_heads,multi_head_attention
from EasyNCO.neural_solvers.methods.mvmoe.moe_layer import MoE



class MTMoEDecoder(nn.Module):
    def __init__(self,embedding_dim,head_num,qkv_dim,sqrt_embedding_dim,logit_clipping,MoE_param=None):
        super().__init__()
        self.sqrt_embedding_dim=sqrt_embedding_dim
        self.logit_clipping=logit_clipping
        self.head_num=head_num
        self.light=MoE_param["light"]
        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hierarchical_gating = False
        # [Option 3]: Use MoEs in Decoder
        if MoE_param['num_experts'] > 1 and 'Dec' in MoE_param['expert_loc']:
            self.multi_head_combine = MoE(input_size=head_num * qkv_dim, output_size=embedding_dim, num_experts=MoE_param['num_experts'],
                                          k=MoE_param['topk'], T=1.0, noisy_gating=True, routing_level=MoE_param['routing_level'],
                                          routing_method=MoE_param['routing_method'], moe_model="Linear")
            if self.light:
                self.hierarchical_gating = True
                self.dense_or_moe = nn.Linear(head_num * qkv_dim, 2, bias=False)
                self.multi_head_combine_dense = nn.Linear(head_num * qkv_dim, embedding_dim)
        else:
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=self.head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.head_num
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=self.head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, attr, ninf_mask,T=1.0,step=2):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # attr.shape: (batch, pomo, 4)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num, moe_loss = self.head_num, 0

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        # shape = (batch, group, EMBEDDING_DIM + 4)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)
        if self.light:
            if self.hierarchical_gating:
                if step == 2:  # this line could be removed if using Gating_Network_1: dense_or_moe in every decoding step
                    self.probs = F.softmax(self.dense_or_moe(out_concat.mean(0).mean(0).unsqueeze(0)) / T,
                                           dim=-1)  # [1, 2]
                selected = self.probs.multinomial(1).squeeze(0)
                if selected.item() == 1:
                    mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
                else:
                    mh_atten_out = self.multi_head_combine_dense(out_concat)
                mh_atten_out = mh_atten_out * self.probs.squeeze(0)[selected]
            else:
                mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
        else:
            if isinstance(self.multi_head_combine, MoE):
                mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
            else:
                mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.sqrt_embedding_dim
        logit_clipping = self.logit_clipping

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, moe_loss