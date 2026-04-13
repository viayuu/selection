import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This file contains the static embeddings for the CO problems, including the initial embedding and the graph mean embedding.
- Initial embedding: Embed the node features to the embedding space.
- Graph mean embedding: Embed the graph mean to the embedding space.
'''


class GraphMeanEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=False):
        super(GraphMeanEmbedding, self).__init__()
        self.graph_mean_embedding = nn.Linear(embed_dim, embed_dim, bias=linear_bias)

    def forward(self, embeddings, weights=None):
        graph_mean_embedding = embeddings.mean(dim=1, keepdim=True)
        if weights is None:
            return self.graph_mean_embedding(graph_mean_embedding)
        else:
            return F.linear(graph_mean_embedding,weights['decoder.GraphMeanEmbedding.graph_mean_embedding.weight'],bias=None)
        # shape:(batch,1,embedding)



def initial_embedding(env_name: str, config: dict) -> nn.Module:
    """
    Get environment initial embedding.
    The init embedding is used to initialize the general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPInitEmbedding,
        "cvrp": VRPInitEmbedding,
        "kp": KPInitEmbedding,

    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPInitEmbedding(nn.Module):
    """
    Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td, weights=None):
        if weights is None:
            out = self.init_embed(td["locs"])
        else:
            out = F.linear(td["locs"], weights['encoder.initial_embedding.init_embed.weight'],
                           weights['encoder.initial_embedding.init_embed.bias'])
        # shape: (batch, N, embed_dim)
        return out


class VRPInitEmbedding(nn.Module):
    """
    Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(VRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, demand
        self.init_embed_customers = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td, weights=None):

        depot, customers = td["locs"][:, :1, :-1], td["locs"][:, 1:, :]
        if weights is None:
            depot_embedding = self.init_embed_depot(depot)
        else:
            depot_embedding = F.linear(depot, weights['encoder.initial_embedding.init_embed_depot.weight'],
                                       weights['encoder.initial_embedding.init_embed_depot.bias'])
        # shape: (batch, 1, embed_dim)

        #customers_demands = torch.cat((customers, td["demand"][..., None]), -1)
        customers_demands = customers
        # shape: (batch, N, 3)
        if weights is None:
            node_embeddings = self.init_embed_customers(customers_demands)
        else:
            node_embeddings = F.linear(customers_demands,
                                       weights['encoder.initial_embedding.init_embed_customers.weight'],
                                       weights['encoder.initial_embedding.init_embed_customers.bias'])
        # shape: (batch, N, embed_dim)
        out = torch.cat((depot_embedding, node_embeddings), dim=-2)
        # shape: (batch, N+1, embed_dim)

        return out

class KPInitEmbedding(nn.Module):
    """
    Initial embedding for the Knapsack Problem (KP).
    Embed the following node features to the embedding space:
        - items: weight, and values of the items
    """
    def __init__(self, embed_dim, linear_bias=True):
        super(KPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["items"])
        # shape: (batch, N, embed_dim)
        return out



