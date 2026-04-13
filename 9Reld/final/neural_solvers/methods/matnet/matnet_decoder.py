import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from EasyNCO.neural_solvers.backbones import reshape_by_heads,multi_head_attention

class ATSPDecoder(nn.Module):
    def __init__(
        self,
        head_num: int = 16,
        embedding_dim: int = 256,
        qkv_dim: int = 16,
        logit_clipping: float = 10.0,
    ):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.logit_clipping = logit_clipping
        self.Wq_0 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention
        
        self.embedding_dim = embedding_dim

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)

        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=self.head_num)
        # shape: (batch, head_num, job, qkv_dim)
        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (batch, embedding, job)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo

        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=self.head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_q0, ninf_mask):
        # encoded_q4.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job)

        #  Multi-Head Attention
        #######################################################
        q0 = reshape_by_heads(self.Wq_0(encoded_q0), head_num=self.head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q1 + q0
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job)

        score_scaled = score / math.sqrt(self.embedding_dim)
        # shape: (batch, pomo, job)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, job)

        return probs


class FFSPDecoder(nn.Module):

    def __init__(
        self,
        head_num: int = 16,
        embedding_dim: int = 256,
        qkv_dim: int = 16,
        logit_clipping: float = 10.0,
    ):
        super().__init__()
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.logit_clipping = logit_clipping
        self.embedding_dim = embedding_dim
        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)
        batch_size = encoded_jobs.size(0)

        encoded_no_job = self.encoded_NO_JOB.expand(
            size=(batch_size, 1, self.embedding_dim)
        )
        encoded_jobs_plus_1 = torch.cat((encoded_jobs, encoded_no_job), dim=1)
        # shape: (batch, job_cnt+1, embedding)

        self.k = reshape_by_heads(self.Wk(encoded_jobs_plus_1), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs_plus_1), head_num=self.head_num)
        # shape: (batch, head_num, job+1, qkv_dim)
        self.single_head_key = encoded_jobs_plus_1.transpose(1, 2)
        # shape: (batch, embedding, job+1)

    def forward(self, encoded_machine, ninf_mask):
        # encoded_machine.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job_cnt+1)

        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=self.head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job_cnt+1)

        score_scaled = score / math.sqrt(self.embedding_dim)
        # shape: (batch, pomo, job_cnt+1)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, job_cnt+1)
        return probs
