

import torch

from .mask_registry import register_mask

@register_mask("vrp")
def capacity_constraint_mask(env) -> torch.Tensor:
    demands = env.depot_node_demand[:, None, :].expand(env.batch_size, env.pomo_size, -1)
    current_capacity = env.load.clone()
    problem_size = env.problem_size + env.depot_num


    demand_too_large = demands > current_capacity.unsqueeze(-1) + 0.00001
    exceed_capacity = current_capacity.unsqueeze(-1) - demands > 1.0 + 0.00001

    # Combine the masks
    illegal_mask = demand_too_large| exceed_capacity

    mask = torch.zeros((env.batch_size, env.pomo_size, problem_size))
    mask[illegal_mask] = float('-inf')

    return mask