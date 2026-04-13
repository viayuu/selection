from typing import Tuple
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
class AttentionModelDecoder(nn.Module):
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
        env_name: str = "tsp",
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = None,
        key_dim: int = None,
        logit_clipping: float = 10, # clipping value for logits, used in Compatibility
        context_embedding: nn.Module = None,
        linear_bias: bool = False,
        use_graph_mean: bool = True, # It is False in POMO and POMO_based models
        am_mode: bool = True,
        first_placeholder: bool = True, # only used in TSP of Attention Model
        sub_glop: bool = False,
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



        self.dynamic_embedding = (dynamic_embedding(self.env_name,
                                            {"embed_dim": embed_dim,
                                                   "first_placeholder": first_placeholder})
            if context_embedding is None
            else context_embedding
        )
        if use_graph_mean:
            self.GraphMeanEmbedding = GraphMeanEmbedding(embed_dim=embed_dim)

        self.compatibility = Compatibility(
                embed_dim=embed_dim,
                n_heads=num_heads,
                qkv_dim=qkv_dim,
                key_dim=self.key_dim,
                am_mode=am_mode,
                sub_glop=self.sub_glop,
        )

        if self.sub_glop:
            self.project_node_embed= nn.Linear(self.embed_dim, 3*self.embed_dim, bias=False)
        else:
            self.Wk = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=linear_bias)
            self.Wv = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=linear_bias)


        self.multi_head_combine = nn.Linear(self.num_heads * self.qkv_dim, self.embed_dim,bias=not sub_glop)
        self.graph_embedding = None
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.encoded_nodes = None  # saved encoded nodes
        self.sing_head_k=None

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
        # shape: (batch, problem, embedding)

    def set_graph_mean(self, embeddings, weights=None):
        self.graph_mean_embedding = self.GraphMeanEmbedding(embeddings,weights)
        # shape:(batch,1,embedding)

    def forward(self, td: TensorDict, first_mode: str = None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state
        Args:
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """
        weights = kwargs.get('weights')
        if self.env_name != 'cvrp' or (self.env_name == 'cvrp' and td.batch_size[1] > 1):
            pomo_size = kwargs.get('pomo_size', None)
            selected,probs = special_selected(self.env_name,
                                              {
                                                    "td": td,
                                                    "first_mode": first_mode,
                                                    "first_placeholder": self.first_placeholder,
                                                    "pomo_size" : pomo_size
                                              })

        if probs is None:
            q = reshape_by_heads(self.graph_mean_embedding + self.dynamic_embedding(self.encoded_nodes, td),
                                 head_num=self.num_heads)
            # q.shape: (batch, head_num, n_start, qkv_dim)

            # Compute logits
            mask = td["next"]["ninf_mask"].clone()
            if self.sub_glop :
                assert mask.shape[1]==2,"In glop,The AM pomo_size should be 2"
                problem_size = mask.shape[2]
                if td["next"]["selected_count"][0,0]==0:
                    mask[:, 0, 1:problem_size] = float("-inf")
                    mask[:, 1, 0:problem_size - 1] = float("-inf")
                else:
                    if td["next"]["selected_count"][0,0]<problem_size-1:
                        mask[:,0,problem_size-1]=float("-inf")
                        mask[:,1,0]=float("-inf")
                    else:
                        mask[:,0,problem_size-1]=0.
                        mask[:,1,0]=0.
            out_concat = multi_head_attention(q, self.k, self.v, mask)
            # mh_atten_out.shape: (batch, head_num, n_start, qkv_dim)
            if weights is None:
                mh_atten_out = self.multi_head_combine(out_concat)
            else:
                mh_atten_out = F.linear(out_concat, weights['decoder.multi_head_combine.weight'],
                                        bias=weights['decoder.multi_head_combine.bias'])


            penalty = kwargs.get('penalty', 0)
            probs = self.compatibility(mh_atten_out,
                                       self.encoded_nodes,
                                       mask=mask,
                                       logit_clipping=self.logit_clipping,
                                       penalty = penalty,sing_head_k=self.sing_head_k)
            selected = None

        return selected,probs


