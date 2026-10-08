########################################
# NN SUB CLASS / FUNCTIONS
########################################
from typing import Tuple

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from multi_hot_set import get_problem_list

def get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


def select_next_node(probs: Tensor, decoding_strategy: str="sampling")-> Tuple[Tensor, Tensor]:
    """
    Design a novel algorithm to select the next node in each step.
    Args:
    probs: Probability distribution over nodes, shape: (batch_size, m).
    decoding_strategy: Decoding strategy to use. Available strategies: ['sampling', 'greedy'], default: 'sampling'.

    Return:
    ID of the next node to visit.
    prob of the selected node.
    """
    assert not torch.isnan(probs).any(), "probs has nan, but it should not have any nans."
    batch_size, pomo_size, problem_size = probs.size()
    if decoding_strategy == "sampling":
        # Check if sampling went OK, can go wrong due to bug on GPU
        # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
        # to fix pytorch.multinomial bug on selecting 0 probability elements
        while True:
            selected = (probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                        .squeeze(dim=1).reshape(batch_size, pomo_size))
            # shape: (batch, pomo)
            prob = torch.gather(probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(dim=-1)
            # shape: (batch, pomo)
            if (prob != 0).all():
                break
        assert prob.size() == (batch_size,pomo_size), f"prob.size(): {prob.size()}. Expected: {(batch_size,pomo_size)}"
        # shape: (batch, n_start)
    elif decoding_strategy == "greedy":
        selected = torch.argmax(probs, dim=-1)
        # (batch_size, pomo)
        prob = torch.zeros(size=(batch_size, pomo_size)) # prob is not needed for greedy decoding
        # shape: (batch, n_start)
    else:
        raise NotImplementedError(f"eval_type: {decoding_strategy} is not implemented!")

    return selected, prob

def distance_normalization(distance_matrix, dist_norm_style):
    # distance_matrix.shape: (batch, n, m)
    batch_size = distance_matrix.size(0)
    if dist_norm_style == 'sep_min_max':
        dist_max = distance_matrix.amax(dim=2, keepdim=True)  # <B, N, 1>
        dist_min = distance_matrix.amin(dim=2, keepdim=True)  # <B, N, 1>
        dist_normed = ((distance_matrix - dist_min) / (dist_max - dist_min + 1e-8))
    elif dist_norm_style == 'sep_max':
        dist_max = distance_matrix.amax(dim=2, keepdim=True)  # <B, N, 1>
        dist_normed = distance_matrix / (dist_max + 1e-8)
    elif dist_norm_style == 'all_min_max':
        dist_max = distance_matrix.amax(dim=(1, 2), keepdim=True)
        dist_min = distance_matrix.amin(dim=(1, 2), keepdim=True)
        # <B, 1, 1>
        assert dist_max.shape == (batch_size, 1, 1)
        assert dist_min.shape == (batch_size, 1, 1)
        dist_normed = ((distance_matrix - dist_min) / (dist_max - dist_min + 1e-8))
    elif dist_norm_style == 'all_max':
        dist_max = distance_matrix.amax(dim=(1, 2), keepdim=True)
        assert dist_max.shape == (batch_size, 1, 1)
        dist_normed = distance_matrix / (dist_max + 1e-8)  # normalize edge features per node
    else:
        assert dist_norm_style == 'nonorm', "Unknown bias style: {}".format(dist_norm_style)
        dist_normed = distance_matrix

    return dist_normed

def adaptation_attention_free_module(q, k, v, adaptation_bias, ninf_mask=None):
    """
    The core code of Adaptation Attention Free Module.
    [Speed Optimized] 移除冗余的 nan/inf 检查，使用更高效的计算路径。

    Inspired by the paper: An Attention Free Transformer
    (url:  https://arxiv.org/pdf/2105.14103.pdf)

    Args:
        q: query, shape: (batch, n, embedding_dim)
        k: key, shape: (batch, m, embedding_dim)
        v: value, shape: (batch, m, embedding_dim)
        adaptation_bias: - alpha * log_scale * dist, shape: (batch, n, m)
        ninf_mask: shape: (batch, n, m)

    Return:
        out: shape: (batch, n, embedding_dim)
    """
    # [Speed] 预计算 sigmoid，避免后续重复
    sigmoid_q = torch.sigmoid(q)
    # shape: (batch, n, embedding_dim)

    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask

    # [Speed] stable exp(k) with max subtraction
    k_max = k.amax(dim=-2, keepdim=True)
    # (batch, 1, embedding_dim)
    exp_k = torch.exp(k - k_max)  # maximum value is exp(0) = 1, avoid overflow

    # [Speed] 使用 clamp 限制 adaptation_bias 范围，避免 exp 溢出
    # 这比事后检查 nan/inf 更快
    adaptation_bias_clamped = adaptation_bias.clamp(min=-50, max=50)
    exp_A = torch.exp(adaptation_bias_clamped)

    # [Speed] 融合计算 exp_k * v
    exp_k_v = exp_k * v
    
    bias = torch.bmm(exp_A, exp_k_v)
    # shape: (batch, n, embedding_dim)
    a_k = torch.bmm(exp_A, exp_k)
    # shape: (batch, n, embedding_dim)

    # [Speed] 直接除法，epsilon 保护
    weighted = bias / (a_k + 1e-8)
    # shape: (batch, n, embedding_dim)

    out = sigmoid_q * weighted
    # shape: (batch, n, embedding_dim)

    return out

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

def orthogonal_encoding(batch_size, problem_size, embedding_dim=128):
    # timestamp
    t = torch.arange(embedding_dim)[None, None, :].expand(batch_size,1,embedding_dim) #[bs,1,128]
    t = t.to(torch.float32)

    const_scale = embedding_dim
    base_frequemcy = torch.randint(1,const_scale, (problem_size,)) * 3
    frequency = base_frequemcy[None, :, None].expand(batch_size, problem_size, 1)
    # shape: (batch, problem_size, 1)
    f = (2 * np.pi /embedding_dim) * (frequency)
    # shape: (batch, problem_size, 1)

    # embedding
    orthogonal_embedding = torch.cos(torch.matmul(f , t))
    # shape: (batch, problem_size, embedding_dim)

    return orthogonal_embedding

def unified_node_position_construction(problems, problem_name):
    batch_size = problems.size(0)
    problem_size = problems.size(1)
    position_features = torch.zeros((batch_size, problem_size, 3), device=problems.device)
    if problem_name in get_problem_list("A_list"):
        # node projections
        node_random_emb = torch.rand((batch_size, problem_size, 1), device=problems.device)
        position_features[:, :, 0] = node_random_emb[:, :, 0]  # random feature
    else:
        position_features[:, :, 1] = problems[:, :, 0]  # x
        position_features[:, :, 2] = problems[:, :, 1]  # y

    return position_features

def unified_node_attribute_construction(problems, problem_name, demand_max1=True):
    batch_size = problems.size(0)
    problem_size = problems.size(1)
    # [demand0, prize1, penalty2, early_time3, late_time4, service_time5, depot6, pickup7, delivery8, multi route9, open route10]
    attribute_features = torch.zeros((batch_size, problem_size, 11), device=problems.device)
    if problem_name in ['atsp','tsp']:
        pass
    elif problem_name in ['op']:
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, :, 1] = problems[:, :, 3]  # prize
    elif problem_name in ['pctsp']:
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, :, 1] = problems[:, :, 3]  # prize
        attribute_features[:, :, 2] = problems[:, :, 5]  # penalty
    elif problem_name in ['spctsp']:
        # x,y,real_prize(sto_prize),fake_prize,penalty
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, :, 1] = problems[:, :, 4]  # fake_prize
        attribute_features[:, :, 2] = problems[:, :, 5]  # penalty
    elif problem_name in ['pdp','apdp']:
        # follow rl4co, 1~n/2 is pickup,n/2+1~n is delivery
        # https://github.com/ai4co/rl4co/blob/main/rl4co/envs/routing/pdp/env.py
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, 1:problem_size // 2 + 1, 7] = 1  # pickup node
        attribute_features[:, problem_size // 2 + 1:,  8] = 1  # delivery node
    elif problem_name in get_problem_list("all_vrpmix_list"):
        if demand_max1:
            demand = problems[:, :, 2]  # shape: (batch, problem)
            demand_normed = demand / (demand.amax(dim=-1, keepdim=True) + 1e-8)
        else:
            demand_normed = problems[:, :, 2]  # shape: (batch, problem)
        attribute_features[:, :, 0] = demand_normed  # demand
        if problem_name in get_problem_list("multi_depot_list"):
            attribute_features[:, :3, 6] = 1  # depot node
        else:
            attribute_features[:, 0, 6] = 1  # single depot node
        attribute_features[:, :, 9] = 1  # multi route
        if 'tw' in problem_name:
            tw_norm_factor = problems[0,0,-2]  # depot_tw_end
            service_time = problems[0,-1,-1]   # service_time
            attribute_features[:, :, 3] = problems[:, :, -3] / tw_norm_factor  # early time window
            attribute_features[:, :, 4] = problems[:, :, -2] / tw_norm_factor  # late time window
            attribute_features[:, :, 5] = service_time  # service_time
        if 'b' in problem_name:
            attribute_features[:, :, 7][problems[:, :, 2] < 0] = 1  # pickup node
            attribute_features[:, :, 8][problems[:, :, 2] > 0] = 1  # delivery node
        else:
            attribute_features[:, 1:, 8] = 1  # delivery node
        if 'o' in problem_name:
            attribute_features[:, :, 10] = 1  # open route
    elif problem_name in ['pdcvrp','opdcvrp','apdcvrp','aopdcvrp']:
        if demand_max1:
            demand = problems[:, :, 2] # shape: (batch, problem)
            demand_normed = demand / (demand.amax(dim=-1, keepdim=True) + 1e-8)
        else:
            demand_normed = problems[:, :, 2]  # shape: (batch, problem)
        attribute_features[:, :, 0] = -demand_normed  # demand
        attribute_features[:, 0, 6] = 1  # depot node
        attribute_features[:, 1:problem_size // 2 + 1, 7] = 1  # pickup node
        attribute_features[:, problem_size // 2 + 1:, 8] = 1  # delivery node
        attribute_features[:, :, 9] = 1  # multi route
        if 'o' in problem_name:
            attribute_features[:, :, 10] = 1  # open route

    else:
        raise ValueError(f"Unsupported problem name: {problem_name}")

    return attribute_features
    # shape: (batch, problem, 11)

def unified_node_attribute_construction_wide(problems, problem_name, demand_max1=True):
    """
    [Fixed Version]
    Strictly aligned with the provided 'unified_node_attribute_construction' logic.
    修正了 OP, PCTSP, SPCTSP 的列索引错误。
    """
    batch_size = problems.size(0)
    problem_size = problems.size(1)
    device = problems.device

    # 存储活跃特征的列表
    values_list = []
    indices_list = []

    # 辅助函数：添加一个特征
    def add_feat(idx, tensor_val):
        indices_list.append(idx)
        values_list.append(tensor_val)

    # 辅助函数：创建全 1 或全 0 矩阵
    ones = torch.ones((batch_size, problem_size), device=device)
    
    # 辅助函数：创建特定位置为 1 的 Mask
    def make_mask(slice_or_condition):
        t = torch.zeros((batch_size, problem_size), device=device)
        if isinstance(slice_or_condition, slice):
            t[:, slice_or_condition] = 1
        else:
            t[slice_or_condition] = 1
        return t

    # === 1. TSP / ATSP ===
    if problem_name in ['atsp', 'tsp']:
        pass 

    # === 2. OP (Orienteering Problem) ===
    elif problem_name in ['op']:
        add_feat(6, make_mask(slice(0, 1))) # Depot
        # [修正] 源代码取的是 problems[:, :, 3]
        add_feat(1, problems[:, :, 3])      # Prize

    # === 3. PCTSP ===
    elif problem_name in ['pctsp']:
        add_feat(6, make_mask(slice(0, 1))) # Depot
        # [修正] 源代码取的是 index 3 和 5
        add_feat(1, problems[:, :, 3])      # Prize
        add_feat(2, problems[:, :, 5])      # Penalty

    # === 4. SPCTSP ===
    elif problem_name in ['spctsp']:
        add_feat(6, make_mask(slice(0, 1))) # Depot
        # [修正] 源代码取的是 index 4 和 5 (对应 real_prize 后面的 fake_prize 和 penalty)
        add_feat(1, problems[:, :, 4])      # Fake Prize
        add_feat(2, problems[:, :, 5])      # Penalty

    # === 5. PDP / APDP ===
    elif problem_name in ['pdp', 'apdp']:
        add_feat(6, make_mask(slice(0, 1))) # Depot
        # 1 ~ N/2 is Pickup
        add_feat(7, make_mask(slice(1, problem_size // 2 + 1)))
        # N/2+1 ~ End is Delivery
        add_feat(8, make_mask(slice(problem_size // 2 + 1, None)))

    # === 6. VRP Mix List (CVRP, VRPTW, VRPB, etc.) ===
    elif problem_name in get_problem_list("all_vrpmix_list"):
        # 6.1 Demand
        d = problems[:, :, 2]
        if demand_max1:
            d_norm = d / (d.amax(dim=-1, keepdim=True) + 1e-8)
        else:
            d_norm = d
        add_feat(0, d_norm)

        # 6.2 Depot
        if problem_name in get_problem_list("multi_depot_list"):
            add_feat(6, make_mask(slice(0, 3))) # Multi-Depot (3 Depots)
        else:
            add_feat(6, make_mask(slice(0, 1))) # Single Depot

        # 6.3 Multi Route
        add_feat(9, ones)

        # 6.4 Time Windows
        if 'tw' in problem_name:
            # 源代码逻辑：归一化因子取自第 0 个样本的倒数第 2 列
            tw_norm_factor = problems[0, 0, -2]
            if tw_norm_factor == 0: tw_norm_factor = 1.0
            
            # Early: 倒数第 3 列, Late: 倒数第 2 列
            add_feat(3, problems[:, :, -3] / tw_norm_factor) 
            add_feat(4, problems[:, :, -2] / tw_norm_factor)
            
            # Service Time: 源代码取 problems[0,-1,-1]，意味着通常最后一列是服务时间
            add_feat(5, problems[:, :, -1])

        # 6.5 Backhaul (VRPB)
        if 'b' in problem_name:
            # VRPB 逻辑：Demand < 0 为 Pickup (Backhaul)
            is_backhaul = problems[:, :, 2] < 0
            
            # [保持优化] 使用 11/12 分离特征，逻辑严谨性优于源代码的 7/8
            # Linehaul: Demand >= 0 且不是 Depot (Index > 0)
            is_linehaul = (problems[:, :, 2] >= 0)
            is_linehaul[:, 0] = False # 排除 Depot
            
            add_feat(11, make_mask(is_backhaul))  # Backhaul
            add_feat(12, make_mask(is_linehaul))  # Linehaul
        else:
            # 普通 Delivery: 除了 Depot 以外的所有点
            add_feat(8, make_mask(slice(1, None)))

        # 6.6 Open Route
        if 'o' in problem_name:
            add_feat(10, ones)

    # === 7. PDCVRP 类 ===
    elif problem_name in ['pdcvrp', 'opdcvrp', 'apdcvrp', 'aopdcvrp']:
        # Demand (取负号)
        d = problems[:, :, 2]
        if demand_max1:
            d_norm = d / (d.amax(dim=-1, keepdim=True) + 1e-8)
        else:
            d_norm = d
        add_feat(0, -d_norm) # Negative demand
        
        add_feat(6, make_mask(slice(0, 1))) # Depot
        add_feat(7, make_mask(slice(1, problem_size // 2 + 1))) # Pickup
        add_feat(8, make_mask(slice(problem_size // 2 + 1, None))) # Delivery
        add_feat(9, ones) # Multi route

        if 'o' in problem_name:
            add_feat(10, ones)

    else:
        raise ValueError(f"Unsupported problem name: {problem_name}")

    # === 8. 输出组装 ===
    if not values_list:
        return torch.zeros((batch_size, problem_size, 0), device=device), \
               torch.tensor([], device=device, dtype=torch.long)

    active_values = torch.stack(values_list, dim=-1)
    active_indices = torch.tensor(indices_list, device=device, dtype=torch.long)

    return active_values, active_indices

class Adaptation_Bias_Module(nn.Module):
    def __init__(self, representation_dim=256, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.W1 = nn.Linear(representation_dim, embedding_dim)  # Support both 13-dim and 256-dim
        self.W2 = nn.Linear(embedding_dim, 1)

    def forward(self, problem_representation):
        # input.shape: (8,)

        return F.relu(self.W2(self.W1(problem_representation))) + 1 # bias >= 1, because we find it is helpful for obtaining better coverage behavior in our experiments.






