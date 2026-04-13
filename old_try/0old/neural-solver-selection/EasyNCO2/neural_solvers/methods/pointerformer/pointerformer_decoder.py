import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from EasyNCO.utils.utils import getLogger
import torch.nn.functional as F
from EasyNCO.neural_solvers.utils import special_selected
logger = getLogger(__name__)
from EasyNCO.utils.utils import _get_encoding


class PointerformerDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        n_heads=8,
        tanh_clipping=10.0,
        multi_pointer=1,
        multi_pointer_level=1,
        add_more_query=True,
        env_name="tsp"
    ):
        super().__init__()
        self.env_name=env_name
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.W_visited = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state
        self.multi_pointer = multi_pointer  #
        self.multi_pointer_level = multi_pointer_level
        self.add_more_query = add_more_query

    def set_kv(self, coordinates, embeddings, G, trainging=True):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        # q_graph.hape = [B, n_heads, 1, key_dim]
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]

        B, N, H = embeddings.shape
        # G = group_ninf_mask.size(1)

        self.coordinates = coordinates  # [:,:2]
        self.embeddings = embeddings
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, G, N, H)
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.q_first = None
        self.logit_k = embeddings.transpose(1, 2)

        if self.multi_pointer > 1:
            k = self.Wk(embeddings)
            k = k.reshape(k.size(0), k.size(1), self.multi_pointer, -1).transpose(1, 2)
            self.logit_k = k.transpose(2, 3)  # [B, n_heads, key_dim, N]

    def forward(self, td: TensorDict, first_mode: str = None, **kwargs) -> Tuple[Tensor, Tensor]:
        pomo_size = kwargs.get('pomo_size', None)
        selected, probs = special_selected(self.env_name,
                                           {
                                               "td": td,
                                               "pomo_size": pomo_size
                                           })

        if probs is None:
            last_node=td['action']
            mask=td['next']['ninf_mask']
            batch_size, pomo_size, emb_dim = self.embeddings.shape
            problem_dim = mask.size(1)
            last_node_embedding = _get_encoding(self.embeddings, last_node[:,:,None]).squeeze()  # shape (batch, pomo, 1, embed_dim)
            q_last=self.Wq_last(last_node_embedding)
            if self.q_first is None:
                self.q_first = self.Wq_first(last_node_embedding)
            mask = mask.detach()
            coors_dim = self.coordinates.size(-1)
            last_node_coordinate = self.coordinates.gather(dim=1, index=last_node.unsqueeze(-1).expand(batch_size, problem_dim, coors_dim))
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            if self.add_more_query:
                mask_visited = mask.clone()
                mask_visited[mask_visited == -np.inf] = 1.0
                q_visited = self.W_visited(torch.bmm(mask_visited, self.embeddings) / pomo_size)
                final_q = q_last + self.q_first + self.q_graph + q_visited
            else:
                final_q = q_last + self.q_first + self.q_graph

            if self.multi_pointer > 1:
                q=self.wq(final_q)
                final_q = q.reshape(q.size(0), q.size(1), self.n_heads, -1).transpose(1, 2)
                # (B,n_head,G,H)  (B,n_head,H,N)
                score = (torch.matmul(final_q, self.logit_k) / math.sqrt(emb_dim)) - (distances / math.sqrt(2)).unsqueeze(1)
                # (B,n_head,G,N)
                if self.multi_pointer_level == 1:
                    score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
                elif self.multi_pointer_level == 2:
                    score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
                else:
                    # add mask
                    score_clipped = self.tanh_clipping * torch.tanh(score)
            else:
                score = torch.matmul(final_q, self.logit_k) / math.sqrt(emb_dim) - distances / math.sqrt(2)
                score_clipped = self.tanh_clipping * torch.tanh(score)
            score_masked = score_clipped + mask
            probs = F.softmax(score_masked, dim=2)
            selected = None

        return selected, probs
