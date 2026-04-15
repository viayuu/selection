from typing import Tuple
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch.nn.functional as F

from EasyNCO.neural_solvers.utils import (dynamic_embedding,
                                          GraphMeanEmbedding,
                                          special_selected)
from EasyNCO.neural_solvers.backbones import (reshape_by_heads,
                                       multi_head_attention,
                                       Compatibility)
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)
class PSLDecoder(nn.Module):
    """
    Autoregressive decoder based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    In this case we additionally have a `pre_decoder_hook` method that allows to precompute the embeddings before
    the decoder is called, which saves a lot of computation.


    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        env_name: Name of the environment used to initialize embeddings
        context_embedding: Context embedding module
        linear_bias: Whether to use a bias in the linear layer
        use_graph_mean: Whether to use the graph context
        am_mode: Whether to use the AM mode
        first_placeholder: Whether to use the first placeholder network, to select the first node
    """

    def __init__(
        self,
        env_name: str = "motsp",
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        key_dim: int = None,
        logit_clipping: float = 10, # clipping value for logits, used in Compatibility
        context_embedding: nn.Module = None,
        linear_bias: bool = False,
        use_graph_mean: bool = False, # It is False in POMO and POMO_based models
        am_mode: bool = True,
        first_placeholder: bool = True, # only used in TSP of Attention Model
        sub_glop: bool = False,
        hyper_input_dim:int = 3,
        **kwargs
    ):
        super().__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.first_placeholder = first_placeholder
        self.logit_clipping = logit_clipping
        self.graph_mean_embedding = 0.0
        self.sub_glop=sub_glop

        if key_dim is None:
            self.key_dim = embed_dim
        else:
            self.key_dim = key_dim

        if qkv_dim is None:
            self.qkv_dim = embed_dim // num_heads
        else:
            self.qkv_dim = qkv_dim

        assert embed_dim % num_heads == 0

        # if self.env_name == "motsp" or self.env_name == "mocvrp" or self.env_name == "mokp":
        #     self.env_name = self.env_name[2:]
        #     # self.dynamic_embedding = (dynamic_embedding(self.env_name,
        #     #                                     {"embed_dim": embed_dim,
        #     #                                            "first_placeholder": first_placeholder})
        #     #     if context_embedding is None
        #     #     else context_embedding
        #     # )
        #     self.env_name = "mo" + self.env_name

        # self.GraphMeanEmbedding = GraphMeanEmbedding(embed_dim=embed_dim)

        # self.compatibility = Compatibility(
        #         embed_dim=embed_dim,
        #         n_heads=num_heads,
        #         qkv_dim=qkv_dim,
        #         key_dim=self.key_dim,
        #         am_mode=am_mode,
        #         sub_glop=self.sub_glop,
        # )

        # self.Wk = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=linear_bias)
        # self.Wv = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=linear_bias)

        # self.multi_head_combine = nn.Linear(self.num_heads * self.qkv_dim, self.embed_dim,bias=not sub_glop)
        self.graph_embedding = None
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.encoded_nodes = None  # saved encoded nodes
        self.sing_head_k=None


        #MOEAD
        # hyper_input_dim = 2
        hyper_hidden_embd_dim = 256
        self.embd_dim = hyper_input_dim
        # self.embd_dim = 3
        self.head_num = self.num_heads
        if self.env_name == "motsp":
            self.hyper_output_dim = 5 * self.embd_dim
            self.hyper_Wq_first = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
            self.hyper_Wq_last = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
        elif self.env_name == "mocvrp":
            self.hyper_output_dim = 4 * self.embd_dim
            self.hyper_Wq_last = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
        elif self.env_name == "mokp":
            hyper_input_dim = 2
            self.embd_dim = hyper_input_dim
            self.hyper_output_dim = 4 * self.embd_dim
            self.hyper_Wq = nn.Linear(self.embd_dim, (1 + self.embed_dim) * self.head_num * qkv_dim, bias=False)


        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)

        # self.hyper_Wq_first = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
        # self.hyper_Wq_last = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, self.embed_dim * self.head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, self.head_num * qkv_dim * self.embed_dim, bias=False)

        self.Wq_last_para = None
        self.Wq_para = None
        self.multi_head_combine_para = None

        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def assign(self, pref):

        embedding_dim = self.embed_dim
        head_num = self.head_num
        qkv_dim = self.qkv_dim

        if not isinstance(pref, torch.Tensor):
            pref = torch.tensor(pref, dtype=torch.float32)
            
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)

        if self.env_name == "motsp":
            self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
            self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(embedding_dim,
                                                                                                      head_num * qkv_dim)
            self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim,
                                                                                                 head_num * qkv_dim)
            self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim,
                                                                                                 head_num * qkv_dim)
            self.multi_head_combine_para = self.hyper_multi_head_combine(
                mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)

        elif self.env_name == "mocvrp":
            self.Wq_last_para = self.hyper_Wq_last(mid_embd[:1 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim)
            self.Wk_para = self.hyper_Wk(mid_embd[1 * self.embd_dim: 2 * self.embd_dim]).reshape(embedding_dim,
                                                                                                 head_num * qkv_dim)
            self.Wv_para = self.hyper_Wv(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim,
                                                                                                 head_num * qkv_dim)
            self.multi_head_combine_para = self.hyper_multi_head_combine(
                mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)

        elif self.env_name == "mokp":
            self.Wq_para = self.hyper_Wq(mid_embd[:self.embd_dim]).reshape(head_num * qkv_dim, (1 + embedding_dim))
            self.Wk_para = self.hyper_Wk(mid_embd[1 * self.embd_dim: 2 * self.embd_dim]).reshape(head_num * qkv_dim,
                                                                                                 embedding_dim)
            self.Wv_para = self.hyper_Wv(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(head_num * qkv_dim,
                                                                                                 embedding_dim)
            self.multi_head_combine_para = self.hyper_multi_head_combine(
                mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.head_num

        self.q_first = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_kv(self, encoded_nodes, weights=None):
        # encoded_nodes.shape: (batch, problem, embedding)

        if weights is None:
            self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=self.head_num)
            self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=self.head_num)
            # shape: (batch, head_num, n_start, qkv_dim)
        else:
            self.k = reshape_by_heads(F.linear(encoded_nodes, weights['decoder.Wk.weight'], bias=None),
                                          head_num=self.num_heads)
            self.v = reshape_by_heads(F.linear(encoded_nodes, weights['decoder.Wv.weight'], bias=None),
                                          head_num=self.num_heads)
        self.encoded_nodes = encoded_nodes
        # shape: (batch, problem, embedding)

        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_graph_mean(self, embeddings, weights=None):
        self.graph_mean_embedding = self.GraphMeanEmbedding(embeddings,weights)
        # shape:(batch,1,embedding)

    def forward(self,encoded_last_node, td: TensorDict, first_mode: str = None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state
        Args:
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """
        weights = kwargs.get('weights')
        pomo_size = kwargs.get('pomo_size', None)
        if self.env_name == "motsp" or self.env_name == "mocvrp" or self.env_name == "mokp":
            self.env_name = self.env_name[2:]
            selected, probs = special_selected(self.env_name,
                                               {
                                                   "td": td,
                                                   "first_mode": first_mode,
                                                   "first_placeholder": self.first_placeholder,
                                                   "pomo_size": pomo_size
                                               })
            self.env_name = "mo" + self.env_name


        if probs is None:
            embedding_dim = self.embed_dim
            head_num = self.head_num
            qkv_dim = self.qkv_dim

            if self.env_name == "motsp" or self.env_name == "mocvrp":
                #  Multi-Head Attention
                #######################################################
                q_last = reshape_by_heads(F.linear(encoded_last_node, self.Wq_last_para), head_num=head_num)

                if self.env_name == 'motsp':
                    q = self.q_first + q_last
                else:
                    q = q_last
                # shape: (batch, head_num, pomo, qkv_dim)

                ninf_mask = td["next"]["ninf_mask"].clone()

                out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)
                # shape: (batch, pomo, head_num*qkv_dim)

                mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)
                # shape: (batch, pomo, embedding)

                #  Single-Head Attention, for probability calculation
                #######################################################
                score = torch.matmul(mh_atten_out, self.single_head_key)
                # shape: (batch, pomo, problem)

                sqrt_embedding_dim = self.embed_dim **(1/2)
                logit_clipping = self.logit_clipping

                score_scaled = score / sqrt_embedding_dim
                # shape: (batch, pomo, problem)

                score_clipped = logit_clipping * torch.tanh(score_scaled)

                score_masked = score_clipped + ninf_mask

                probs = F.softmax(score_masked, dim=2)
                # shape: (batch, pomo, problem)

                selected = None

            elif self.env_name == "mokp":
                capacity = td["capacity"].clone()
                batch_size = capacity.size(0)
                group_size = capacity.size(1)

                #  Multi-Head Attention
                #######################################################
                graph = self.encoded_nodes.mean(dim=1, keepdim=True)
                input1 = graph.expand(batch_size, group_size, embedding_dim)
                input2 = capacity[:, :, None]
                input_cat = torch.cat((input1, input2), dim=2)

                #  Multi-Head Attention
                #######################################################
                q = reshape_by_heads(F.linear(input_cat, self.Wq_para), head_num=head_num)
                ninf_mask = td["next"]["ninf_mask"].clone()
                out_concat = multi_head_attention(q, self.k, self.v, mask=ninf_mask)

                mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)
                # shape: (batch, pomo, embedding)

                #  Single-Head Attention, for probability calculation
                #######################################################
                score = torch.matmul(mh_atten_out, self.single_head_key)
                # shape: (batch, pomo, problem)

                sqrt_embedding_dim = self.embed_dim **(1/2)
                logit_clipping = self.logit_clipping

                score_scaled = score / sqrt_embedding_dim
                # shape: (batch, pomo, problem)

                score_clipped = logit_clipping * torch.tanh(score_scaled)

                # score_masked = score_clipped + ninf_mask
                if ninf_mask is None:
                    score_masked = score_clipped
                else:
                    score_masked = score_clipped + ninf_mask

                probs = F.softmax(score_masked, dim=2)
                # shape: (batch, pomo, problem)

        return selected,probs


