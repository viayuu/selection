import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
from torch_geometric.utils import add_self_loops
import numpy as np
from torch.distributions.categorical import Categorical
from EasyNCO.neural_solvers.methods.l2s.models import GIN,DGHAN




class L2SPolicy(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 embedding_l=4,
                 policy_l=3,
                 embedding_type='gin',
                 num_heads=4,
                 dropout=0.6):
        super(L2SPolicy, self).__init__()
        self.embedding_l = embedding_l
        self.policy_l = policy_l
        self.embedding_type = embedding_type

        if self.embedding_type == 'gin':
            self.embedding = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=embedding_l)
        elif self.embedding_type == 'dghan':
            self.embedding = DGHAN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, layer_dghan=embedding_l, num_heads=num_heads)
        elif self.embedding_type == 'gin+dghan':
            self.embedding_gin = GIN(in_dim=in_dim, hidden_dim=hidden_dim, layer_gin=embedding_l)
            self.embedding_dghan = DGHAN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, layer_dghan=embedding_l, num_heads=num_heads)
        else:
            raise Exception('embedding type should be either "gin", "dghan", or "gin+dghan".')

        # policy
        self.policy = torch.nn.ModuleList()
        if policy_l == 1:
            if self.embedding_type == 'gin+dghan':
                self.policy.append(Sequential(Linear(hidden_dim * 4, hidden_dim),
                                              # torch.nn.BatchNorm1d(hidden_dim),
                                              torch.nn.Tanh(),
                                              Linear(hidden_dim, hidden_dim)))
            else:
                self.policy.append(Sequential(Linear(hidden_dim * 2, hidden_dim),
                                              # torch.nn.BatchNorm1d(hidden_dim),
                                              torch.nn.Tanh(),
                                              Linear(hidden_dim, hidden_dim)))
        else:
            for layer in range(policy_l):
                if layer == 0:
                    if self.embedding_type == 'gin+dghan':
                        self.policy.append(Sequential(Linear(hidden_dim * 4, hidden_dim),
                                                      # torch.nn.BatchNorm1d(hidden_dim),
                                                      torch.nn.Tanh(),
                                                      Linear(hidden_dim, hidden_dim)))
                    else:
                        self.policy.append(Sequential(Linear(hidden_dim * 2, hidden_dim),
                                                      # torch.nn.BatchNorm1d(hidden_dim),
                                                      torch.nn.Tanh(),
                                                      Linear(hidden_dim, hidden_dim)))
                else:
                    self.policy.append(Sequential(Linear(hidden_dim, hidden_dim),
                                                  # torch.nn.BatchNorm1d(hidden_dim),
                                                  torch.nn.Tanh(),
                                                  Linear(hidden_dim, hidden_dim)))

    # def set_decoder_strategy(self, strategy: str = "sampling"):
    #     self.decoder_strategy = strategy

    def forward(self, batch_states, feasible_actions):

        if self.embedding_type == 'gin':
            node_embed, graph_embed = self.embedding(batch_states.x,
                                                     add_self_loops(torch.cat([batch_states.edge_index_pc,
                                                                               batch_states.edge_index_mc],
                                                                              dim=-1))[0],
                                                     batch_states.batch)
        elif self.embedding_type == 'dghan':
            node_embed, graph_embed = self.embedding(batch_states.x,
                                                     add_self_loops(batch_states.edge_index_pc)[0],
                                                     add_self_loops(batch_states.edge_index_mc)[0],
                                                     len(feasible_actions))
        elif self.embedding_type == 'gin+dghan':
            node_embed_gin, graph_embed_gin = self.embedding_gin(batch_states.x,
                                                                 add_self_loops(torch.cat([batch_states.edge_index_pc,
                                                                                           batch_states.edge_index_mc],
                                                                                          dim=-1))[0],
                                                                 batch_states.batch)
            node_embed_dghan, graph_embed_dghan = self.embedding_dghan(batch_states.x,
                                                                       add_self_loops(batch_states.edge_index_pc)[0],
                                                                       add_self_loops(batch_states.edge_index_mc)[0],
                                                                       len(feasible_actions))
            node_embed = torch.cat([node_embed_gin, node_embed_dghan], dim=-1)
            graph_embed = torch.cat([graph_embed_gin, graph_embed_dghan], dim=-1)
        else:
            raise Exception('embedding type should be either "gin", "dghan", or "gin+dghan".')

        device = node_embed.device
        batch_size = graph_embed.shape[0]
        n_nodes_per_state = node_embed.shape[0] // batch_size

        # augment node embedding with graph embedding then forwarding policy
        node_embed_augmented = torch.cat([node_embed, graph_embed.repeat_interleave(repeats=n_nodes_per_state, dim=0)], dim=-1).reshape(batch_size, n_nodes_per_state, -1)
        for layer in range(self.policy_l):
            node_embed_augmented = self.policy[layer](node_embed_augmented)

        # action score
        action_score = torch.bmm(node_embed_augmented, node_embed_augmented.transpose(-1, -2))

        # prepare mask
        carries = np.arange(0, batch_size * n_nodes_per_state, n_nodes_per_state)
        a_merge = []  # merge index of actions of all states
        action_count = []  # list of #actions for each state
        for i in range(len(feasible_actions)):
            action_count.append(len(feasible_actions[i]))
            for j in range(len(feasible_actions[i])):
                a_merge.append([feasible_actions[i][j][0] + carries[i], feasible_actions[i][j][1]])
        a_merge = np.array(a_merge)
        mask = torch.ones(size=[batch_size * n_nodes_per_state, n_nodes_per_state], dtype=torch.bool, device=device)
        mask[a_merge[:, 0], a_merge[:, 1]] = False
        mask.resize_as_(action_score)

        # pi
        action_score.masked_fill_(mask, -np.inf)
        action_score_flat = action_score.reshape(batch_size, 1, -1)
        pi = F.softmax(action_score_flat, dim=-1)


        dist = Categorical(probs=pi)
        # if self.decoder_strategy == "greedy":
        #     actions_id = torch.argmax(pi, dim=-1)
        # else:   #sampling
        actions_id = dist.sample()

        sampled_actions = [[actions_id[i].item() // n_nodes_per_state, actions_id[i].item() % n_nodes_per_state] for i in range(len(feasible_actions))]
        log_prob = dist.log_prob(actions_id)  # log_prob using Pytorch API, this will have a gradient shift, reference: https://github.com/pytorch/pytorch/issues/61727. Used in paper submission version.
        # log_prob = torch.log(torch.gather(pi, -1, actions_id.unsqueeze(-1)) + -1e-7).squeeze(-1)  # log_prob calculated manually, this will not have a gradient shift. Switch to this after paper submission.
        return sampled_actions, log_prob