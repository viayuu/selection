import torch
from torch import nn
from torch.nn import functional as F
from EasyNCO.utils.utils import getLogger
from EasyNCO.neural_solvers.methods.icam.icam_encoder import adaptation_attention_free_module

logger = getLogger(__name__)

class TSPICAMDecoder(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        sqrt_embedding_dim=128**0.5,
        logit_clipping=50,
        **kwargs  # Additional parameters for flexibility
    ):
        super().__init__()
        self.udc_flag = kwargs.get("udc_flag", False)
        self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.AFT_dist_alpha_1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.sqrt_embedding_dim = sqrt_embedding_dim
        self.logit_clipping = logit_clipping

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = self.Wq_first(encoded_q1)
        # shape: (batch, problem, embedding)

    def forward(self, encoded_last_node, cur_dist, log_scale, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)

        q_last = self.Wq_last(encoded_last_node)
        # shape: (batch, pomo, embedding_dim)
        q = self.q_first + q_last
        # shape: (batch, pomo, embedding_dim)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_adaptation_bias = -1 * self.AFT_dist_alpha_1 * log_scale * cur_dist
        # shape: (batch, pomo, problem)
        AAFM_OUT = adaptation_attention_free_module(
            q, self.k, self.v, alpha_adaptation_bias, ninf_mask
        )
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(AAFM_OUT, self.single_head_key)
        # shape: (batch, pomo, problem)
        score_scaled = score / self.sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        score_scaled = score_scaled - self.dist_alpha_1 * log_scale * cur_dist #! +
        # shape: (batch, pomo, problem)
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, pomo, problem)

        return probs


class CVRPICAMDecoder(nn.Module):

    def __init__(
        self,
        embedding_dim=128,
        head_num=8,
        qkv_dim=16,
        sqrt_embedding_dim=128**0.5,
        logit_clipping=50,
        **kwargs  # Additional parameters for flexibility
    ):
        super().__init__()
        self.udc_flag = kwargs.get("udc_flag", False)
        if self.udc_flag:
            self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.q_first = None  # saved q1, for multi-head attention
            self.q1 = None  # saved q1, for multi-head attention
            self.q2 = None  # saved q2, for multi-head attention
            self.Wq_last = nn.Linear(embedding_dim + 2, embedding_dim, bias=False)
        else:
            self.Wq_last = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        self.probs_dist_alpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.sqrt_embedding_dim = sqrt_embedding_dim
        self.logit_clipping = logit_clipping

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)

        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        # shape: (batch, problem+1, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q1 = self.Wq_1(encoded_q1)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q2 = self.Wq_2(encoded_q2)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, cur_dist, log_scale, ninf_mask, left=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # cur_dist.shape: (batch, pomo, problem+1)
        # ninf_mask.shape: (batch, pomo, problem+1)

        if self.udc_flag:
            input_cat = torch.cat(
                (encoded_last_node, load[:, :, None], left[:, :, None]), dim=2
            )
            # shape = (batch, group, EMBEDDING_DIM+1)
        else:
            input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, pomo, embedding+1)
        q_last = self.Wq_last(input_cat)
        # shape: (batch, pomo, embedding+1)
        if self.udc_flag:
            q = self.q1 + self.q2 + q_last
            # shape: (batch, pomo, embedding+1)
        else:
            q = q_last
            # shape: (batch, pomo, embedding+1)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_adaptation_bias = -1 * self.AFT_dist_alpha * log_scale * cur_dist
        # shape: (batch, pomo, problem+1)
        AAFM_OUT = adaptation_attention_free_module(
            q, self.k, self.v, alpha_adaptation_bias, ninf_mask, self.udc_flag
        )
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(AAFM_OUT, self.single_head_key)
        # shape: (batch, pomo, problem+1)
        sqrt_embedding_dim = self.sqrt_embedding_dim
        logit_clipping = self.logit_clipping
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem+1)
        score_scaled = score_scaled - log_scale * self.probs_dist_alpha * cur_dist
        # shape: (batch, pomo, problem+1)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, pomo, problem+1)

        return probs
