import torch
import torch.nn as nn
from absl.logging import exception
from tensordict import TensorDict
from typing import Literal

from EasyNCO.neural_solvers.methods import AttentionModelEncoder, AttentionModelDecoder
from EasyNCO.neural_solvers.utils import get_post_search_strategy
from EasyNCO.utils.utils import getLogger
from EasyNCO.neural_solvers.backbones.Transformer.attr_component import positional_encoding_ELG
from EasyNCO.neural_solvers.backbones.Transformer.attention import MultiHeadAttentionLayer

logger = getLogger(__name__)

class ELGPolicy(nn.Module):
    def __init__(self,
                 env_name: str = "tsp",
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 qkv_dim: int = 16,
                 num_encoder_layers: int = 6,
                 normalization: str = "instance",
                 feedforward_hidden: int = 512,
                 logit_clipping: float = 10,  # clipping value for logits, used in Compatibility
                 use_graph_mean: bool = False,  # It is False in POMO and POMO_based models
                 am_mode: bool = True,
                 first_placeholder: bool = True,  # only used in TSP of Attention Model
                 first_mode: Literal["random", "placeholder"] = "random",
                 local_enable: bool = False,
                 local_size: int = 30,
                 xi: int = -1,
                 euclidean: bool = False
                 ):
        super().__init__()

        self.env_name = env_name
        self.local_size = int(local_size)
        self.local_enable = local_enable
        self.xi = xi

        euclidean = euclidean
        if env_name == "tsp":
            self.local_policy = local_policy_att_tsp(env_name=self.env_name,
                                                local_size=self.local_size,
                                                euclidean=euclidean,
                                                positional=True, )
        elif env_name == "cvrp":
            self.local_policy = local_policy_att_cvrp(env_name=self.env_name,
                                                 local_size=self.local_size,
                                                 euclidean=euclidean,
                                                 positional=True, )
        else:
            raise NotImplementedError(f"local policy for {env_name} is not implemented yet")
        self.local_policy.requires_grad_(False)  # freeze the gradient of local policy in the beginning

        self.encoder = AttentionModelEncoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_dim=qkv_dim,
            num_layers=num_encoder_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )
        self.decoder = AttentionModelDecoder(
                        env_name=env_name,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        logit_clipping=logit_clipping,
                        use_graph_mean=use_graph_mean,
                        am_mode=am_mode,
                        first_placeholder=first_placeholder,
        )
        self.decoder_strategy = None
        self.use_graph_mean = use_graph_mean
        self.first_mode = first_mode

    def set_decoder_strategy(self, strategy: str="sampling"):
        self.decoder_strategy = strategy

    def pre_forward(self, td:TensorDict):
        encoded_nodes, _ = self.encoder(td)
        self.decoder.set_kv(encoded_nodes)
        if self.use_graph_mean:
            self.decoder.set_graph_mean(encoded_nodes)

    def enable_local_policy(self):
        self.local_enable = True
        self.local_policy.requires_grad_(True)

    def forward(self, td : TensorDict) -> TensorDict:

        if self.local_enable and 'local_feature' in td.keys():
            penalty = elg_forward(td=td, local_size=self.local_size, xi=self.xi, problem=self.env_name)
            local_score = self.local_policy(td)
        else:
            penalty = 0
            local_score = 0
        penalty = penalty + local_score
        pomo_selected, probs = self.decoder(td, first_mode=self.first_mode, penalty = penalty)
        if pomo_selected is not None:
            selected = pomo_selected
            prob = probs
        else:
            selected, prob = get_post_search_strategy(self.decoder_strategy, probs)

        td.set("action", selected)
        td.set("prob", prob)

        return td



class local_policy_att_tsp(nn.Module):
    def __init__(self,
                 env_name: str = "tsp",
                 init_embedding : nn.Module = None,
                 embed_dim : int = 32,
                 num_heads : int = 4,
                 qkv_dim : int = 8,
                 local_size : int = 30,
                 euclidean : bool = False,
                 positional : bool = True,
                 ):
        super().__init__()
        self.env_name = env_name
        self.local_size = local_size
        self.euclidean = euclidean
        self.positional = positional

        self.qkv_dim = qkv_dim
        self.num_heads = num_heads

        self.embed_dim = embed_dim
        self.max_timescale = 10000.0
        self.min_timescale = 1.0
        input_dim = 2 if env_name == 'tsp' else 3

        self.initial_embedding = nn.Linear(input_dim, self.embed_dim)

        self.cur_token_emb = nn.Parameter(torch.Tensor(self.embed_dim))
        self.cur_token_emb.data.uniform_(-1, 1)

        self.mha = MultiHeadAttentionLayer(embed_dim = self.embed_dim,
                                           num_heads = self.num_heads,
                                           qkv_dim = self.qkv_dim,
                                        )

    def forward(self, td : TensorDict):
        # dist.shape: (batch, pomo, problem)
        # theta.shape: (batch, pomo, problem)
        dist = td['local_feature']['cur_dist']
        ninf_mask = td['next']['ninf_mask']
        theta = td['local_feature']['cur_theta']
        xy = td['local_feature']['relative_xy']

        valid_nodes = dist.shape[2]
        pomo_size = dist.shape[1]

        dist -= ninf_mask
        valid_nodes -= ninf_mask.isinf().sum(-1).min()

        if self.local_size > valid_nodes:
            local_size = valid_nodes
        else:
            local_size = self.local_size

        if self.env_name == 'tsp':
            sorted_dist, idx = dist.topk(local_size, dim = -1, largest = False)
            # shape: (batch, pomo, local)
            sorted_mask = torch.take_along_dim(ninf_mask, idx, dim = -1)
            # shape: (batch, pomo, local)

        # Polar coordinate distance
        if self.euclidean:
            norm_fac = (sorted_dist.max(-1)[0].unsqueeze(-1) + 1e-6)
            sorted_dist = torch.take_along_dim(xy[:, :, :, 0], idx, dim = -1) / norm_fac
            sorted_theta = torch.take_along_dim(xy[:, :, :, 1], idx, dim = -1) / norm_fac
        else:
            sorted_dist = sorted_dist / (sorted_dist.max(-1)[0].unsqueeze(-1) + 1e-6) # avoid division by zero
            sorted_theta = torch.take_along_dim(theta, idx, dim = -1)

        sorted_dist_theta = torch.cat((sorted_dist[:, :, :, None], sorted_theta[:, :, :, None]), dim = -1)
        # shape: (batch, pomo, local, 2)
        cur_token = self.cur_token_emb[None, None, :].expand(dist.shape[0], dist.shape[1], self.embed_dim)
        # (batch, pomo, embed_dim)

        # Positional encoding
        pos_encode = positional_encoding_ELG(sorted_dist_theta, self.embed_dim, self.max_timescale, self.min_timescale)
        pos_encode = pos_encode.view((1, sorted_dist_theta.size()[2], self.embed_dim))
        signal = (pos_encode[:, None, :, :]).expand(-1, pomo_size, -1, -1)

        init_k = self.initial_embedding(sorted_dist_theta) + signal
        # init_k.shape: (batch, pomo, local, embed_dim)

        mh_atten_out = self.mha(q_input = cur_token[:, :, None, :], kv_input = init_k, mask = sorted_mask)
        # shape: (batch, pomo, 1, embed_dim)

        score = torch.matmul(mh_atten_out, init_k.transpose(2, 3)).squeeze(2)
        # shape: (batch, pomo, local)
        sqrt_emb_dim = self.embed_dim ** 0.5
        score_scaled = score / sqrt_emb_dim

        out = score_scaled
        out_mat = torch.zeros(dist.shape, device = dist.device, dtype = out.dtype)
        out = out_mat.scatter_(-1, idx, out)
        # shape: (batch, pomo, problem + 1)

        return out

class local_policy_att_cvrp(nn.Module):
    def __init__(self,
                 env_name: str = "tsp",
                 init_embedding : nn.Module = None,
                 embed_dim : int = 32,
                 num_heads : int = 4,
                 qkv_dim : int = 8,
                 local_size : int = 30,
                 euclidean : bool = False,
                 positional : bool = True,
                 ):
        super().__init__()
        self.env_name = env_name
        self.local_size = local_size
        self.euclidean = euclidean
        self.positional = positional

        self.qkv_dim = qkv_dim
        self.num_heads = num_heads

        self.embed_dim = embed_dim
        self.max_timescale = 10000.0
        self.min_timescale = 1.0
        input_dim = 2 if env_name == 'tsp' else 3

        self.initial_embedding = nn.Linear(input_dim, self.embed_dim)

        self.cur_token_emb = nn.Parameter(torch.Tensor(self.embed_dim))
        self.cur_token_emb.data.uniform_(-1, 1)

        self.mha = MultiHeadAttentionLayer(embed_dim = self.embed_dim,
                                           num_heads = self.num_heads,
                                           qkv_dim = self.qkv_dim,
                                        )

    def forward(self, td: TensorDict):
        # dist.shape: (batch, pomo, problem)
        # theta.shape: (batch, pomo, problem)
        dist = td['local_feature']['cur_dist']
        ninf_mask = td['next']['ninf_mask']
        theta = td['local_feature']['cur_theta']
        xy = td['local_feature']['relative_xy']
        demand = td['local_feature']['norm_demand']

        valid_nodes = dist.shape[2]
        pomo_size = dist.shape[1]

        dist -= ninf_mask
        valid_nodes -= ninf_mask.isinf().sum(-1).min()

        if self.local_size > valid_nodes:
            local_size = valid_nodes
        else:
            local_size = self.local_size


        depot_idx = torch.zeros(dist.shape[0], dist.shape[1], 1).long()
        # select top-K except depot, add 1 to idx for alignment
        dist_, idx = dist[:, :, 1:].topk(local_size, dim=-1, largest=False)
        # norm factor
        if dist_.isinf().any():
            dist_[dist_.isinf()] = 0.
        norm_idx = dist_.max(-1)[0] != 0
        norm_fac = dist_[norm_idx].max(-1)[0].unsqueeze(-1) + 1e-6
        idx += 1
        # add depot idx
        idx = torch.cat((depot_idx, idx), dim=-1)

        sorted_dist = torch.take_along_dim(dist, idx, dim=-1)
        sorted_theta = torch.take_along_dim(theta, idx, dim=-1)
        sorted_demand = torch.take_along_dim(demand, idx, dim=-1)
        sorted_mask = torch.take_along_dim(ninf_mask, idx, dim=-1)

        # Polar coordinate distance
        if self.euclidean:
            sorted_x = torch.take_along_dim(xy[:, :, :, 0], idx, dim=-1)
            sorted_y = torch.take_along_dim(xy[:, :, :, 1], idx, dim=-1)

        # padding 0 to align different mask
        sorted_demand[torch.where(torch.isnan(sorted_demand))] = 0

        if sorted_dist.isinf().any():
            sorted_theta[sorted_dist.isinf()] = 0.
            sorted_demand[sorted_dist.isinf()] = 0.
            if self.euclidean == True:
                sorted_x[sorted_dist.isinf()] = 0.
                sorted_y[sorted_dist.isinf()] = 0.
            sorted_dist[sorted_dist.isinf()] = 0.

        if norm_idx is None:
            norm_idx = sorted_dist.max(-1)[0] != 0
            norm_fac = sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1) + 1e-6  # avoid division by zero
            sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac
            if self.euclidean == True:
                sorted_x[norm_idx] = sorted_x[norm_idx] / norm_fac
                sorted_y[norm_idx] = sorted_y[norm_idx] / norm_fac
        else:
            sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac
            if self.euclidean == True:
                sorted_x[norm_idx] = sorted_x[norm_idx] / norm_fac
                sorted_y[norm_idx] = sorted_y[norm_idx] / norm_fac

        if self.euclidean == True:
            sorted_dist_theta = torch.cat((sorted_x[:, :, :, None], sorted_y[:, :, :, None]), dim=-1)
        else:
            sorted_dist_theta = torch.cat((sorted_dist[:, :, :, None], sorted_theta[:, :, :, None]), dim=-1)

        sorted_input = torch.cat((sorted_dist_theta, sorted_demand[:, :, :, None]), dim = -1)

        cur_token = self.cur_token_emb[None, None, :].expand(dist.shape[0], dist.shape[1], self.embed_dim)
        # (batch, pomo, embed_dim)

        # Positional encoding
        pos_encode = positional_encoding_ELG(sorted_input, self.embed_dim, self.max_timescale,
                                             self.min_timescale)
        pos_encode = pos_encode.view((1, sorted_dist_theta.size()[2], self.embed_dim))
        signal = (pos_encode[:, None, :, :]).expand(-1, pomo_size, -1, -1)

        init_k = self.initial_embedding(sorted_input) + signal
        # init_k.shape: (batch, pomo, local, embed_dim)

        mh_atten_out = self.mha(q_input=cur_token[:, :, None, :], kv_input=init_k, mask=sorted_mask)


        # shape: (batch, pomo, 1, embed_dim)

        score = torch.matmul(mh_atten_out, init_k.transpose(2, 3)).squeeze(2)
        # shape: (batch, pomo, local)
        sqrt_emb_dim = self.embed_dim ** 0.5
        score_scaled = score / sqrt_emb_dim

        out = score_scaled
        out_mat = torch.zeros(dist.shape, device=dist.device, dtype=out.dtype)
        out = out_mat.scatter_(-1, idx, out)
        # shape: (batch, pomo, problem + 1)

        return out



def add_distance_penalty_tsp(cur_dist, mask, local_size, xi):

    valid_nodes = cur_dist.shape[2]
    dist = cur_dist.clone()

    dist -= mask
    valid_nodes -= mask.isinf().sum(-1).min()

    if local_size > valid_nodes:
        local_size = valid_nodes
    else:
        local_size = local_size

    sorted_dist, idx = dist.topk(local_size, dim=-1, largest=False)
    # shape: (batch, pomo, local)
    dist_penalty = - sorted_dist / (sorted_dist.max(-1)[0].unsqueeze(-1) + 1e-6)
    out_mat = xi * torch.ones(cur_dist.shape, device=cur_dist.device)
    penalty = out_mat.scatter_(-1, idx, dist_penalty)

    return penalty


def add_distance_penalty_cvrp(cur_dist, mask, local_size, xi):
    valid_nodes = cur_dist.shape[2]
    dist = cur_dist.clone()

    dist -= mask
    valid_nodes -= mask.isinf().sum(-1).min()

    if local_size > valid_nodes:
        local_size = valid_nodes
    else:
        local_size = local_size

    depot_idx = torch.zeros(dist.shape[0], dist.shape[1], 1).long()

    dist_, idx = dist[:, :, 1:].topk(local_size, dim=-1, largest=False)

    if dist_.isinf().any():
        dist_[dist_.isinf()] = 0.
    norm_idx = dist_.max(-1)[0] != 0
    norm_fac = dist_[norm_idx].max(-1)[0].unsqueeze(-1) + 1e-6  # avoid division by zero

    idx += 1
    idx = torch.cat((depot_idx, idx), dim=-1)

    sorted_dist = torch.take_along_dim(dist, idx, dim=-1)

    if sorted_dist.isinf().any():
        sorted_dist[sorted_dist.isinf()] = 0.

    if norm_idx is None:
        norm_idx = sorted_dist.max(-1)[0] != 0
        sorted_dist[norm_idx] = sorted_dist[norm_idx] / sorted_dist[norm_idx].max(-1)[0].unsqueeze(-1)
    else:
        sorted_dist[norm_idx] = sorted_dist[norm_idx] / norm_fac

    dist_penalty = - sorted_dist
    out_mat = xi * torch.ones(cur_dist.shape, device=cur_dist.device)
    penalty = out_mat.scatter_(-1, idx, dist_penalty)

    return penalty

def elg_forward(td, local_size, xi, problem):

    if 'local_feature' in td.keys():
        cur_dist = td["local_feature"]['cur_dist']
        mask = td['next']['ninf_mask']
        if problem == 'tsp':
            penalty = add_distance_penalty_tsp(cur_dist, mask, local_size, xi)
        elif problem == 'cvrp':
            penalty = add_distance_penalty_cvrp(cur_dist, mask, local_size, xi)
    else:
        penalty = 0

    return penalty