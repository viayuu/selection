import torch
from .mask_registry import register_mask


@register_mask("vrpbl")
def mask_vrpbl(env, selected):
    num_nodes = env.problem_size + env.depot_num
    round_error_epsilon = 1e-6
    visited_mask = env.visited_ninf_flag.clone()
    visited_mask[:, :, 0][~env.at_the_depot] = 0
    demand_list = env.demand[:, None, :].expand(
        env.batch_size, env.pomo_size, num_nodes
    )
    demand_too_large = demand_list > env.load[:, :, None] + round_error_epsilon
    exceed_capacity = env.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
    capacity_mask = torch.zeros_like(visited_mask)
    capacity_mask[demand_too_large | exceed_capacity] = float("-inf")
    current_node_to_all_dist = env.dist.gather(
        1, env.current_node[:, :, None].expand(-1, -1, num_nodes)
    )
    current_all_to_depot_dist = env.dist.gather(
        2, env.current_depot[:, None, :].expand(-1, num_nodes, -1)
    ).permute(0, 2, 1)
    route_too_large = (
        env.cumulative_length[:, :, None]
        + current_node_to_all_dist
        + current_all_to_depot_dist
        > env.route_limit[:, :, None] + round_error_epsilon
    )
    route_too_large[:, :, : env.depot_num] = False
    l_mask = torch.zeros_like(visited_mask)
    l_mask[route_too_large] = float("-inf")
    mask = visited_mask + capacity_mask + l_mask
    mask[:, :, : env.depot_num][
        env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    ] = float("-inf")
    depot_mask = ~env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    mask[:, :, : env.depot_num][depot_mask] = 0
    return mask
