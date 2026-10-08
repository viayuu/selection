import torch
from masks_unified.mask_registry import register_mask


@register_mask("cvrptw")
def mask_cvrptw(env, selected):
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.batch_size
    pomo_size = env.pomo_size
    mask = torch.zeros((batch_size, pomo_size, num_nodes), device=env.device)
    if env.selected_node_list.size(2) > 0:
        mask.scatter_(-1, env.selected_node_list, float("-inf"))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float("-inf"))
    demand_list = env.demand[:, None, :].expand(batch_size, pomo_size, -1)
    demand_too_large = env.load[:, :, None] + env.round_error_epsilon < demand_list
    exceed_capacity = env.load[:, :, None] - demand_list > 1.0 + env.round_error_epsilon
    demand_mask = torch.zeros_like(mask)
    demand_mask[demand_too_large | exceed_capacity] = float("-inf")
    mask = mask + demand_mask
    round_error_epsilon = env.round_error_epsilon
    speed = 1.0
    depot_end = env.tw_end[0, 0]
    current_node_to_all_dist = env.dist.gather(
        1, env.current_node[:, :, None].expand(-1, -1, num_nodes)
    )
    current_all_to_depot_dist = env.dist.gather(
        2, env.current_depot[:, None, :].expand(-1, num_nodes, -1)
    )
    current_all_to_depot_dist = current_all_to_depot_dist.permute(0, 2, 1)
    next_time_required = env.current_time[:, :, None] + current_node_to_all_dist / speed
    arrival_time = torch.max(
        next_time_required, env.tw_start[:, None, :].expand(-1, pomo_size, -1)
    )
    out_of_tw = (
        arrival_time
        > env.tw_end[:, None, :].expand(-1, pomo_size, -1) + round_error_epsilon
    )
    tw_mask = torch.zeros_like(mask)
    tw_mask[out_of_tw] = float("-inf")
    fail_return_depot = (
        arrival_time
        + env.service_time[:, None, :].expand(-1, pomo_size, -1)
        + current_all_to_depot_dist / speed
        > depot_end + round_error_epsilon
    )
    tw_mask[fail_return_depot] = float("-inf")
    mask = mask + tw_mask
    at_depot_expanded = env.at_the_depot[:, :, None].expand(-1, -1, env.depot_num)
    mask[:, :, : env.depot_num][at_depot_expanded] = float("-inf")
    not_at_depot_expanded = (~env.at_the_depot)[:, :, None].expand(
        -1, -1, env.depot_num
    )
    mask[:, :, : env.depot_num][not_at_depot_expanded] = 0.0
    return mask
