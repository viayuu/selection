import torch
from .mask_registry import register_mask


@register_mask("ovrpbltw")
def mask_ovrpbltw(env, selected):
    num_nodes = env.problem_size + env.depot_num
    round_error_epsilon = 1e-6
    speed = 1.0
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
    route_too_large = (
        env.cumulative_length[:, :, None] + current_node_to_all_dist
        > env.route_limit[:, :, None] + round_error_epsilon
    )
    route_too_large[:, :, : env.depot_num] = False
    l_mask = torch.zeros_like(visited_mask)
    l_mask[route_too_large] = float("-inf")
    travel_time = current_node_to_all_dist / speed
    next_time_required = env.current_time[:, :, None] + travel_time
    arrival_time = torch.max(
        next_time_required, env.tw_start[:, None, :].expand(-1, env.pomo_size, -1)
    )
    out_of_tw = (
        arrival_time
        > env.tw_end[:, None, :].expand(-1, env.pomo_size, -1) + round_error_epsilon
    )
    tw_mask = torch.zeros_like(visited_mask)
    tw_mask[out_of_tw] = float("-inf")
    mask = visited_mask + capacity_mask + l_mask + tw_mask
    mask[:, :, : env.depot_num][
        env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    ] = float("-inf")
    depot_mask = ~env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    mask[:, :, : env.depot_num][depot_mask] = 0
    return mask
