import torch
from .mask_registry import register_mask


@register_mask("mdovrpbpl")
def mask_mdovrpbpl(env, selected):
    num_nodes = env.problem_size + env.depot_num
    round_error_epsilon = 0.00001
    visited_mask = env.visited_ninf_flag.clone()
    at_depot = env.current_node < env.depot_num
    visited_mask[:, :, : env.depot_num][
        ~at_depot[:, :, None].expand(-1, -1, env.depot_num)
    ] = 0
    demand_list = env.demand[:, None, :].expand(
        env.batch_size, env.pomo_size, num_nodes
    )
    demand_too_large = demand_list > env.load[:, :, None] + round_error_epsilon
    exceed_capacity = env.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
    capacity_mask = torch.zeros_like(visited_mask)
    capacity_mask[demand_too_large | exceed_capacity] = float("-inf")
    gathering_index = env.current_node[:, :, None]
    current_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
    at_backhaul = (current_demand < 0)[:, :, None]
    linehaul_nodes = demand_list > 0
    bp_mask = torch.zeros_like(visited_mask)
    bp_mask[at_backhaul & linehaul_nodes] = float("-inf")
    current_node_to_all_dist = env.dist.gather(
        1, env.current_node[:, :, None].expand(-1, -1, num_nodes)
    )
    route_limit_expanded = env.route_limit.expand(-1, env.pomo_size)
    will_exceed_length = (
        env.cumulative_length[:, :, None] + current_node_to_all_dist
        > route_limit_expanded[:, :, None] + round_error_epsilon
    )
    l_mask = torch.zeros_like(visited_mask)
    l_mask[will_exceed_length] = float("-inf")
    mask = visited_mask + capacity_mask + bp_mask + l_mask
    mask[:, :, : env.depot_num][at_depot[:, :, None].expand(-1, -1, env.depot_num)] = (
        float("-inf")
    )
    depot_mask = ~at_depot[:, :, None].expand(-1, -1, env.depot_num)
    mask[:, :, : env.depot_num][depot_mask] = 0
    return mask
