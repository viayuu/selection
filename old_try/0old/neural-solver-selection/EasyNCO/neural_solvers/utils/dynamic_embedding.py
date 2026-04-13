import torch
import torch.nn as nn
import torch.nn.functional as F

from EasyNCO.utils.utils import _get_encoding

'''
This file contains the dynamic embeddings for the CO problems.
It is used to modify the context embedding of the problem node of the current partial solution for RL training.
For example, in the CVRP, the dynamic embedding is the remaining capacity of the vehicle and the current node embedding.
And in the TSP, the dynamic embedding is the current node embedding and the first node embedding.
'''

def dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """
    Get environment dynamic embedding in context embedding.
    The dynamic embedding is used to modify the query embedding of the problem node of the current partial solution.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPDynamicEmbedding,
        "cvrp": CVRPDynamicEmbedding,
        "kp": KPDynamicEmbedding,
    }

    if env_name not in embedding_registry.keys():
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)

class DynamicEmbedding(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, context_dim=None, linear_bias=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        context_dim = context_dim if context_dim is not None else embed_dim
        self.W_dynamic = nn.Linear(context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        # td["current_node"].shape: (batch, n_start)
        cur_node_embedding = _get_encoding(embeddings, td["action"][:, :, None]).squeeze(-2)
        # shape: (batch, n_start, embedding)
        return cur_node_embedding

    def _cur_state_append(self, embeddings, td):
        """
        Get state embedding.
        For example, it is the remaining capacity in CVRP.
        """
        raise NotImplementedError("Implement for each environment, please see the specific Context class.")

    def forward(self, embeddings, td, weights=None):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_append = self._cur_state_append(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_append], dim=-1)

        if weights is None:
            query_embedding = self.W_dynamic(context_embedding)
        else:
            query_embedding = F.linear(context_embedding,weights['decoder.dynamic_embedding.W_dynamic.weight'],bias=None)
        return query_embedding

class TSPDynamicEmbedding(DynamicEmbedding):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embed_dim, **kwargs):
        super(TSPDynamicEmbedding, self).__init__(embed_dim=embed_dim,
                                                  context_dim=2 * embed_dim)
        first_placeholder = kwargs.get("first_placeholder", False)
        # For the first node, we use a placeholder embedding，this operation is only used in the Attention Model.
        self.first_placeholder = first_placeholder
        if self.first_placeholder:
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embed_dim))
            self.W_placeholder.data.uniform_(-1, 1) # initialize with uniform distribution

    def forward(self, embeddings, td, weights=None):
        batch_size = embeddings.size(0)
        n_start = td["action"].size(1)

        if (td["action"]== -1).all() and self.first_placeholder:
            context_embedding = self.W_placeholder[None, None, :].expand(batch_size, n_start, self.W_placeholder.size(-1))
            # shape: (batch, n_start, 2*embedding)
        else:
            assert (td["first_node"] != -1).any(), "First node is not provided."
            first_node_embedding = _get_encoding(embeddings, td["first_node"][:,:,None]).squeeze(-2)
            last_node_embedding = _get_encoding(embeddings, td["action"][:,:,None]).squeeze(-2)
            context_embedding = torch.cat([first_node_embedding, last_node_embedding], dim=-1)
            # shape: (batch, n_start, 2*embedding)

        if weights is None:
            return self.W_dynamic(context_embedding) # shape: (batch, n_start, embedding)
        else:
            return F.linear(context_embedding, weights['decoder.dynamic_embedding.W_dynamic.weight'], bias=None)


class CVRPDynamicEmbedding(DynamicEmbedding):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim, **kwargs):
        super(CVRPDynamicEmbedding, self).__init__(embed_dim=embed_dim,
                                                   context_dim=embed_dim + 1)

    def _cur_state_append(self, embeddings, td):
        state_append = td["vehicle_capacity"] - td["used_capacity"]
        # -0.00001 is used to avoid negative remaining capacity due to floating point errors
        assert (state_append >= -0.00001).all(), "Negative remaining capacity, please check the capacity calculation."
        # shape: (batch, n_start, 1)
        return state_append

class KPDynamicEmbedding(DynamicEmbedding):
    """Context embedding for the Knapsack (KP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim, **kwargs):
        super(KPDynamicEmbedding, self).__init__(embed_dim=embed_dim,
                                                   context_dim=1)

    def forward(self, embedding, td):
        context_embedding = td['capacity'][:, :, None]

        query_embedding = self.W_dynamic(context_embedding)

        return query_embedding
