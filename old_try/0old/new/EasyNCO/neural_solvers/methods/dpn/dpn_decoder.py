from typing import Tuple
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
import torch
import math
from typing import NamedTuple

from EasyNCO.neural_solvers.backbones import multi_head_attention
from EasyNCO.utils.utils import getLogger
from EasyNCO.neural_solvers.utils.post_search import (
    get_post_search_strategy,
)

logger = getLogger(__name__)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class DPNDecoder(nn.Module):

    def __init__(
        self,
        env_name: str = "mtsp",
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = None,
        key_dim: int = None,
        tanh_clipping: float = 50.0,  # the clipping in the original code is 50.0, instead of logit_clipping with 10.0
        mask_logits=True,
    ):
        super().__init__()
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.graph_mean_embedding = 0.0
        self.mask_logits = mask_logits
        self.temp = 1.0
        if key_dim is None:
            self.key_dim = embed_dim
        else:
            self.key_dim = key_dim
        if qkv_dim is None:
            self.qkv_dim = embed_dim // num_heads
        else:
            self.qkv_dim = qkv_dim
        # Problem specific context parameters (placeholder and step context dimension)
        step_context_dim = (
            2 * embed_dim + 2
        )  # Embedding of current_agent, current node, # of left cities and # of left agents
        self.embeddings = None  # saved encoded nodes
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embed_dim, bias=False)
        self.dis_emb = nn.Sequential(
            nn.Linear(
                5 if self.env_name == "mpdp" else 3, embed_dim, bias=False
            )
        )
        assert embed_dim % num_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fixed = None
        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def set_fixed(self, embeddings: Tensor, agent_embeddings=None, td=None):
        self.embeddings = embeddings
        self.agent_embeddings = agent_embeddings
        if td != None:
            self.depot_num = td["depot_num"][0]
        # shape: [batch_size, num_nodes, embed_dim]
        self.fixed = self._precompute(self.embeddings)

    def forward(
        self,
        td: TensorDict,
        decode_type: str = None,
    ) -> Tuple[Tensor, Tensor]:
        self.curr_dist = torch.squeeze(td["dist_out"][:, 0, :, :], dim=1).gather(
            1, td["prev_a"].expand(-1, -1, td["dist_out"].size(-1))
        )
        self.decode_type = decode_type

        # if probs is None:
        probs, _ = self._get_probs(self.fixed, td, self.agent_embeddings)
        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected, probs = get_post_search_strategy(self.decode_type, probs.exp())
        return selected, probs

    def _precompute(self, embeddings):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = (
            self.project_node_embeddings(embeddings[:, :, :]).chunk(3, dim=-1)
        )

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed,
        )
        return AttentionModelFixed(
            embeddings, fixed_context, *fixed_attention_node_data
        )

    def _make_heads(self, v):
        batch_s = v.size(0)
        n = v.size(1)
        q_reshaped = v.reshape(batch_s, n, self.num_heads, -1)
        # shape: (batch, n, head_num, key_dim)
        q_transposed = q_reshaped.transpose(1, 2)
        return q_transposed

    def _get_probs(self, fixed, state, agent_embeddings=None):
        query = fixed.context_node_projected + self.project_step_context(
            self._get_parallel_step_context(
                fixed.node_embeddings, state, agent_embeddings
            )
        )
        if self.env_name == "mpdp":
            query += self.dis_emb(
                torch.cat(
                    (
                        state["lengths"].gather(-1, state["count_depot"]),
                        state["remain_pickup_max_distance"],
                        state["remain_delivery_max_distance"],
                        state["longest_lengths"].gather(-1, state["count_depot"]),
                        state["remain_sum_paired_distance"]
                        / (state["agent_per"].size(2) - state["count_depot"]),
                    ),
                    -1,
                )
            )
        elif (
            self.env_name == "mtsp"
            or self.env_name == "mdvrp"
            or self.env_name == "fmdvrp"
        ):
            query += self.dis_emb(
                torch.cat(
                    (
                        state["lengths"].gather(-1, state["count_depot"]),
                        state["max_distance"],
                        state["remain_max_distance"],
                    ),
                    -1,
                )
            )

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        # Compute the mask
        mask = self.get_mask(state)
        # Compute logits (unnormalized log_p)
        log_p = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, td, agent_embeddings=None):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param td: (batch_size, num_steps)
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = td["prev_a"]
        batch_size, pomo_size, _ = current_node.size()
        batch_size, _, embedding_dim = embeddings.size()
        gathering_index = current_node.expand(batch_size, pomo_size, embedding_dim)
        gathering_agent = td["agent_idx"].expand(batch_size, pomo_size, embedding_dim)
        # shape: (batch, pomo, embedding)
        picked_nodes = embeddings.gather(dim=1, index=gathering_index)
        if agent_embeddings is None:
            picked_agents = embeddings.gather(dim=1, index=gathering_agent)
        else:
            picked_agents = agent_embeddings.gather(dim=1, index=gathering_agent)

        if self.env_name == "mtsp":
            left = td["left_city"] / (td["loc"].size(2) - td["agent_per"].size(2))
        elif self.env_name == "mpdp":
            left = td["left_request"] / (
                (self.embeddings.size(1) - td["agent_per"].size(2)) // 2
            )
        elif self.env_name == "mdvrp" or self.env_name == "fmdvrp":
            left = td["left_city"] / (td["loc"].size(2) - self.depot_num)
        return torch.cat(
            (
                torch.cat((picked_nodes, picked_agents), dim=-1),
                1.0
                - torch.ones(size=td["count_depot"].shape, device=embeddings.device)
                * (td["count_depot"] + 1)
                / td["agent_per"].size(2),
                left,
            ),
            2,
        )

    def _get_attention_node_data(self, fixed):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def get_mask(self, td):

        agent_num = td["lengths"].size(2)  # number of agent
        visited_loc = td["visited_"].clone()
        if self.env_name == "mtsp":
            agent_idx = td["agent_idx"]
            mask_loc = td["visited_"].clone()
            mask_loc[:, :, :agent_num] = 1
            mask_loc = mask_loc.scatter_(-1, agent_idx, 0)
            condition = (td["count_depot"] == agent_num - 1).squeeze(-1) & (
                (visited_loc[:, :, agent_num:] == 0).sum(dim=-1) != 0
            )
            src = torch.ones_like(condition, dtype=torch.uint8) * condition
            mask_loc = mask_loc.scatter_(-1, agent_idx, src[:, :, None])
        elif self.env_name == "mpdp":
            agent_idx = td["agent_idx"]
            n_loc = visited_loc.size(-1) - agent_num  # num of customers
            mask_loc = visited_loc.to(td["to_delivery"].device) | (
                1 - td["to_delivery"]
            )
            # depot
            mask_loc[:, :, :agent_num] = 1
            # if deliver nodes which is assigned agent is complete, then agent can go to depot
            no_item_to_delivery = (
                visited_loc[:, :, n_loc // 2 + agent_num + 1 :]
                == td["to_delivery"][:, :, n_loc // 2 + agent_num + 1 :]
            ).all(dim=-1)
            condition = (td["count_depot"] == agent_num - 1).squeeze(-1) & (
                (visited_loc[:, :, agent_num:] == 0).sum(dim=-1) != 0
            )
            condition = ~(no_item_to_delivery & ~condition)
            src = torch.ones_like(condition, dtype=torch.uint8) * condition
            mask_loc = mask_loc.scatter_(-1, agent_idx, src[:, :, None])
        elif self.env_name == "mdvrp":
            mask_loc = td["visited_"].clone()
            depot_size = self.depot_num
            mask_loc[:, :, :depot_size] = 1
            agent_idx = td["assign"].gather(2, td["count_depot"])
            mask_loc = mask_loc.scatter_(-1, agent_idx, 0)
            condition = (td["count_depot"] == agent_num - 1).squeeze(-1) & (
                (visited_loc[:, :, depot_size:] == 0).sum(dim=-1) != 0
            )
            src = torch.ones_like(condition, dtype=torch.uint8) * condition
            mask_loc = mask_loc.scatter_(-1, agent_idx, src[:, :, None])
            mask_loc[td["to_assign"].squeeze(), :] = 1
            mask_loc[:, :, :depot_size][td["to_assign"].squeeze(), :] = 0
        elif self.env_name == "fmdvrp":
            mask_loc = td["visited_"].clone()
            depot_size = self.depot_num
            mask_loc[:, :, :depot_size] = 0
            condition = (td["count_depot"] == agent_num - 1).squeeze(-1) & (
                (visited_loc[:, :, depot_size:] == 0).sum(dim=-1) != 0
            )
            mask_loc[:, :, :depot_size][condition] = 1
            mask_loc[td["to_assign"].squeeze(), :] = 1
            mask_loc[:, :, :depot_size][td["to_assign"].squeeze(), :] = 0

        return (
            mask_loc > 0
        )  # Hacky way to return bool or uint8 depending on pytorch version

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        query = self._make_heads(query)
        out = multi_head_attention(query, glimpse_K, glimpse_V)
        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        logits = torch.matmul(
            self.project_out(out), logit_K.transpose(-2, -1)
        ) / math.sqrt(out.size(-1))
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = (
                torch.tanh(logits + self.dist_alpha_1 * self.curr_dist)
                * self.tanh_clipping
            )
        if self.mask_logits:
            logits[mask] = -math.inf
        log_p = torch.log_softmax(
            logits / self.temp, dim=-1
        )  # orginal code is log_softmax, which is different from the other code
        return log_p
