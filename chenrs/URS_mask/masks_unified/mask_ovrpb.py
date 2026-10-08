import torch
from .mask_registry import register_mask


@register_mask("ovrpb")
def mask_ovrpb(env, selected):
    num_nodes = env.problem_size + env.depot_num
    visited_mask = env.visited_ninf_flag.clone()
    visited_mask[:, :, 0][~env.at_the_depot] = 0
    demand_list = env.demand[:, None, :].expand(
        env.batch_size, env.pomo_size, num_nodes
    )
    demand_too_large = demand_list > env.load[:, :, None] + 1e-6
    exceed_capacity = env.load[:, :, None] - demand_list > 1.0 + 1e-6
    capacity_mask = torch.zeros_like(visited_mask)
    capacity_mask[demand_too_large | exceed_capacity] = float("-inf")
    mask = visited_mask + capacity_mask
    mask[:, :, : env.depot_num][
        env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    ] = float("-inf")
    depot_mask = ~env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    mask[:, :, : env.depot_num][depot_mask] = 0
    return mask
