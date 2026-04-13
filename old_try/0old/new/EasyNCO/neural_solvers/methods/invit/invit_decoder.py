from typing import Tuple

import torch.nn as nn
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import ModuleList
from torch_cluster import knn

from EasyNCO.neural_solvers.backbones import (MultiHeadAttentionLayer,
                                       reshape_by_heads,
                                       multi_head_attention,
                                       FeedForward,
                                       SkipConnection,
                                       Normalization,
                                       Compatibility)

from EasyNCO.neural_solvers.utils import special_selected
from EasyNCO. neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)

class INVITDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 2,
        env_name: str = 'tsp',
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        feedforward_dim: int = 512,
        action_embed_layer_num: int = 2,
        state_embed_layer_num: int = 2,
        state_num: int = 3,
        decoder_layer_num: int = 3,
        normalization: str = 'layer',
        first_placeholder: bool = True,
        state_k: list = [15, 35, 50],
        action_k: int = 15,
        logit_clipping: float = 10
    ):
        super().__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.decoder_layer_num = decoder_layer_num
        self.first_placeholder = first_placeholder
        self.state_k = state_k
        self.action_k = action_k
        self.logit_clipping = logit_clipping

        self.action_embed = INVITEmbedLayer(
                                env_name=env_name,
                                node_dim=node_dim,
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                qkv_dim=qkv_dim,
                                feedforward_dim=feedforward_dim,
                                normalization=normalization,
                                neighbor_type= 'action',
                                embed_layer_num=action_embed_layer_num,
                                bias = False,
                                bias_combine = False)

        self.state_embed = ModuleList( [INVITEmbedLayer(
                                            env_name=env_name,
                                            node_dim=node_dim,
                                            embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            qkv_dim=qkv_dim,
                                            feedforward_dim=feedforward_dim,
                                            normalization=normalization,
                                            neighbor_type='state',
                                            embed_layer_num=state_embed_layer_num,
                                            bias=False,
                                            bias_combine = False)
                                        for _ in range(state_num)])

        if self.env_name == 'tsp':
            self.Wq = nn.Linear(embed_dim*(1 + 2*state_num), embed_dim)
        elif self.env_name == 'cvrp':
            self.Wq = nn.Linear(embed_dim*(2 + 2*state_num), embed_dim)
        else:
            raise RuntimeError(f"Invalid problem type: {self.env_name}.")

        self.Wk = nn.Linear(embed_dim*(1 + state_num), embed_dim*decoder_layer_num)
        self.Wv = nn.Linear(embed_dim*(1 + state_num), embed_dim*decoder_layer_num)
        self.Wq_final = nn.Linear(embed_dim, embed_dim)
        self.autoregressive = ModuleList([AutoRegressiveDecoderLayer(embed_dim=embed_dim, num_heads=num_heads) for _ in range(decoder_layer_num-1)])
        self.compatibility = Compatibility(embed_dim=embed_dim, n_heads=num_heads, qkv_dim=qkv_dim, key_dim=embed_dim, am_mode=False)

    def forward(self, problems, decoder_strategy, td: TensorDict, first_mode: str = None) -> Tuple[Tensor, Tensor]:
        selected, probs = special_selected(self.env_name,
                                           {
                                               "td": td,
                                               "first_mode": first_mode,
                                               "problem_size": problems.shape[1],
                                               "first_placeholder": self.first_placeholder,
                                           })

        if probs is None:
            batch_size = td.batch_size[0]
            pomo_size = td.batch_size[1]
            left_node_size = torch.max(problems.shape[1] - int(self.env_name=='cvrp') - torch.count_nonzero(td['next']['ninf_mask'][:, :, int(self.env_name=='cvrp'):], dim=-1))
            idx = torch.arange(batch_size).repeat(pomo_size).sort()[0]
            last_node = problems.clone()[idx, td['action'].reshape(-1), :].reshape(batch_size, pomo_size, 1, -1)
            k_max = min(self.state_k[-1], left_node_size)
            k_action = min(self.action_k, left_node_size)

            # embed action nodes
            if self.env_name == 'tsp':
                knn_sorted_node, knn_sorted_node_idx, knn_mask = self.knn_sorted(problems, td, last_node, k_max)
                first_node = problems.clone()[idx, td['first_node'].reshape(-1), :].reshape(batch_size, pomo_size, 1, -1)
                action_node = torch.cat((knn_sorted_node[:, :, :k_action, :], last_node, last_node), dim=-2)
                action_embed_node = self.action_embed(action_node, k_action, knn_mask[:, :, :k_action])
                embed_q = action_embed_node[:, :, [-1], :]
                embed_other = action_embed_node[:, :, :k_action, :]
            elif self.env_name == 'cvrp':
                last_node[:, :, :, 2] = td['vehicle_capacity'] - td['used_capacity']
                knn_sorted_node, knn_sorted_node_idx, knn_mask = self.knn_sorted(problems, td, last_node, k_max)
                first_node = problems.clone()[:, [0], :][:, None, :, :].expand(-1, pomo_size, -1, -1)
                action_node = torch.cat((knn_sorted_node[:, :, :k_action, :], last_node, first_node), dim=-2)
                action_embed_node = self.action_embed(action_node, k_action, knn_mask[:, :, :k_action])
                embed_q = torch.cat((action_embed_node[:, :, [-2], :], action_embed_node[:, :, [-1], :]), dim=-1)
                embed_other = torch.cat((action_embed_node[:, :, :k_action, :], action_embed_node[:, :, [-1], :]), dim=-2)
            else:
                raise RuntimeError(f"Invalid problem type: {self.env_name}.")

            # embed state nodes
            for i in range(0, len(self.state_k)):
                k_state = min(self.state_k[i], left_node_size)
                state_node = torch.cat((knn_sorted_node[:, :, :k_state, :], last_node, first_node), dim=-2)
                state_embed_node = self.state_embed[i](state_node, k_state, knn_mask[:, :, :k_state])
                embed_q = torch.cat((embed_q, state_embed_node[:, :, [-2],:], state_embed_node[:, :, [-1], :]), dim=-1)
                if self.env_name == 'tsp':
                    embed_other = torch.cat((embed_other, state_embed_node[:, :, :k_action, :]), dim=-1)
                elif self.env_name == 'cvrp':
                    embed_other_temp = torch.cat((state_embed_node[:, : ,:k_action, :], state_embed_node[:, :, [-1], :]), dim=-2)
                    embed_other = torch.cat((embed_other, embed_other_temp), dim=-1)
                else:
                    raise RuntimeError(f"Invalid problem type: {self.env_name}.")

            # decoder
            q = self.Wq(embed_q)
            k = self.Wk(embed_other)
            v = self.Wv(embed_other)

            if self.env_name == 'cvrp':
                decoder_mask = torch.cat((knn_mask[:, :, :k_action], td['next']['ninf_mask'][:, :, [0]]), dim=-1)
            else:
                decoder_mask = None
            h_l = q
            for i in range(self.decoder_layer_num):
                k_l = k[:, :, :, i * self.embed_dim:(i + 1) * self.embed_dim].contiguous()
                v_l = v[:, :, :, i * self.embed_dim:(i + 1) * self.embed_dim].contiguous()
                if i < self.decoder_layer_num - 1:
                    h_l = self.autoregressive[i](h_l, k_l, v_l, decoder_mask)
                else:
                    q_final = self.Wq_final(h_l)
                    probs = self.compatibility(q_final, k_l, decoder_mask, logit_clipping=self.logit_clipping)

            selected_temp, probs = get_post_search_strategy(decoder_strategy, probs)
            sorted_node_idx = torch.cat((knn_sorted_node_idx[:, :, :k_action], torch.zeros(size=(batch_size, pomo_size, 1),
                                                dtype=knn_sorted_node_idx.dtype)), dim=-1)
            selected = sorted_node_idx[idx, torch.arange(pomo_size).repeat(batch_size), selected_temp.reshape(-1)].reshape(batch_size, pomo_size)

        return selected, probs

    def knn_sorted(self, problems, td, last_node, k_max):

        batch_size = td.batch_size[0]
        pomo_size = td.batch_size[1]
        # In env_mask, true for unvisited nodes.
        if self.env_name == 'tsp':
            env_mask = (td['next']['ninf_mask'] == 0)
            knn_problems = problems.clone()
        elif self.env_name == 'cvrp':
            env_mask = (td['next']['ninf_mask'][:, :, 1:] == 0) # shape (batch, pomo, problem)
            knn_problems = problems[:, 1:, :].clone()
        else:
            raise RuntimeError(f"Invalid problem type: {self.env_name}.")
        problem_size = knn_problems.shape[1]
        node_dim = knn_problems.shape[2]
        nodes = knn_problems[:, None, :, :].expand(-1, pomo_size, -1, -1) # shape (batch, pomo, problem, node_dim)
        last_node_index = td['action'].clone()

        unvisited_nodes = nodes[env_mask]
        available_num_vec = torch.sum(env_mask, dim=-1).long()
        available_num_vec[available_num_vec > k_max] = k_max #shape (batch, pomo)
        last_node_to_knn = last_node.reshape(-1, node_dim)
        batch_pomo_expand =torch.arange(batch_size * pomo_size)[:, None].expand(-1, problem_size).reshape(batch_size, pomo_size, -1)
        # shape (batch, pomo, problem)
        unvisited_node_batch_index = batch_pomo_expand[env_mask]
        knn_out = knn(unvisited_nodes[:, :2], last_node_to_knn[:, :2], k_max, unvisited_node_batch_index, torch.arange(batch_size*pomo_size))

        unvisited_knn_mask = torch.arange(k_max)[None, None, :].expand(batch_size, pomo_size, -1) < available_num_vec[:, :, None]
        unvisited_knn_nodes_index = last_node_index[:, :, None].expand(-1, -1, k_max).clone()
        if self.env_name == 'tsp':
            unvisited_knn_nodes_index[unvisited_knn_mask] = (torch.arange(problem_size)[None, None, :].expand(batch_size, pomo_size, -1)[env_mask])[knn_out[1, :]]
        elif self.env_name == 'cvrp':
            unvisited_knn_nodes_index[unvisited_knn_mask] = (torch.arange(problem_size)[None, None, :].expand(batch_size, pomo_size, -1)[env_mask] + 1)[knn_out[1, :]]
        unvisited_knn_nodes = problems.clone()[:, None, :, :].expand(-1, pomo_size, -1 ,-1).gather(dim=-2, index=unvisited_knn_nodes_index[:, :, :, None].expand(-1, -1, -1, node_dim))

        unvisited_mask = torch.zeros_like(unvisited_knn_mask, dtype=torch.float32)
        unvisited_mask[~unvisited_knn_mask] = float('-inf')

        return unvisited_knn_nodes, unvisited_knn_nodes_index, unvisited_mask

class INVITEmbedLayer(nn.Module):
    def __init__(
        self,
        env_name: str = 'tsp',
        node_dim: int = 2,
        embed_dim: int = 128,
        num_heads: int = 8,
        qkv_dim: int = 16,
        feedforward_dim: int = 512,
        normalization: str = 'layer',
        neighbor_type: str = 'action',
        embed_layer_num: int = 2,
        bias = False,
        bias_combine = True,
        multi_head_combine_used = True,
        ):
        super().__init__()
        self.env_name = env_name
        self.neighbor_type = neighbor_type
        self.embed_layer_num = embed_layer_num

        self.embed_other = nn.Linear(node_dim, embed_dim)
        self.embed_last = nn.Linear(node_dim, embed_dim)
        if not (self.env_name == 'tsp' and self.neighbor_type == 'action'):
            self.embed_first = nn.Linear(2, embed_dim)

        self.MHA_layers = nn.ModuleList(
            [SkipConnection(
                SkipConnection(
                    MultiHeadAttentionLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        bias=bias,
                        bias_combine = bias_combine,
                        multi_head_combine_used = multi_head_combine_used)
                ),
                Normalization(embed_dim=embed_dim, normalization=normalization, eps=1e-6))
            for _ in range(embed_layer_num)])

        self.feedforward_layers = nn.ModuleList([SkipConnection(FeedForward(embedding_dim=embed_dim, ff_hidden_dim=feedforward_dim)) for _ in range(embed_layer_num)])
        self.norm_1 = nn.ModuleList([Normalization(embed_dim=embed_dim, normalization=normalization) for _ in range(embed_layer_num)])
        self.norm_2 = nn.ModuleList([Normalization(embed_dim=embed_dim, normalization=normalization) for _ in range(embed_layer_num)])

    def forward(self, x, k, mask):
        x = self.InvariantLayer(x)
        embed_last_node = self.embed_last(x[:, :, [-2], :])
        embed_other_node = self.embed_other(x[:, :, :k, :])
        if self.env_name == 'tsp' and self.neighbor_type == 'action':
            embed_node = torch.cat((embed_other_node, embed_last_node), dim=-2)
        else:
            embed_first_node = self.embed_first(x[:, :, [-1], :2])
            embed_node = torch.cat((embed_other_node, embed_last_node, embed_first_node), dim=-2)

        mask = torch.cat((mask, torch.zeros(mask.shape[0], mask.shape[1], embed_node.shape[2]-mask.shape[2])),dim=-1)
        out = embed_node
        for i in range(self.embed_layer_num):
            params_MHA = [{'mask': mask, 'score_dim':128}]
            out = self.MHA_layers[i](out, params_MHA)
            out = self.norm_1[i](out)
            out = self.feedforward_layers[i](out)
            out = self.norm_2[i](out)

        return out

    def InvariantLayer(self, nodes):
        new_nodes = nodes.clone() # shape (batch, pomo, problem, node_dim)
        xy_min = torch.min(new_nodes[:, :, :-1, :2], dim=-2).values.unsqueeze(-2)  # shape (batch, pomo, 1, 2)
        xy_max = torch.max(new_nodes[:, :, :-1, :2], dim=-2).values.unsqueeze(-2)
        ratio = torch.max((xy_max - xy_min), dim=-1).values.unsqueeze(-1)
        ratio[ratio==0] = 1
        new_nodes[:, :, :, :2] = (new_nodes[:, :, :, :2] - xy_min).div(ratio)
        new_nodes[:, :, [-1], :2] = torch.clip(new_nodes[:, :, [-1], :2], 0, 1)
        return new_nodes

class AutoRegressiveDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8
        ):
        super().__init__()

        self.head_num = num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wa = nn.Linear(embed_dim, embed_dim)
        self.norm_1 = Normalization(embed_dim=embed_dim, normalization='layer')
        self.norm_2 = Normalization(embed_dim=embed_dim, normalization='layer')
        self.skip = SkipConnection(FeedForward(embedding_dim=embed_dim, ff_hidden_dim=embed_dim))

    def forward(self, q, k, v, mask):
        Q = reshape_by_heads(self.Wq(q), head_num=self.head_num)
        K = reshape_by_heads(k, head_num=self.head_num)
        V = reshape_by_heads(v, head_num=self.head_num)
        q = q + self.Wa(multi_head_attention(Q, K, V, mask))
        q = self.norm_1(q)

        out = self.skip(q)
        out = self.norm_2(out)
        return out