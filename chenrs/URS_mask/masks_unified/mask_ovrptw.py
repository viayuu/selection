import torch
from masks_unified.mask_registry import register_mask


@register_mask("ovrptw")
def mask_ovrptw(env, selected):
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
    mask[demand_too_large | exceed_capacity] = float("-inf")
    current_node_expanded = env.current_node.unsqueeze(-1).expand(-1, -1, num_nodes)
    current_to_all_dist = env.dist.gather(dim=1, index=current_node_expanded)
    next_time_required = (
        env.current_time[:, :, None] + current_to_all_dist / env.speed
    )
    tw_start_expanded = env.tw_start[:, None, :].expand(-1, pomo_size, -1)
    arrival_time = torch.max(next_time_required, tw_start_expanded)
    tw_end_expanded = env.tw_end[:, None, :].expand(-1, pomo_size, -1)
    out_of_tw = arrival_time > tw_end_expanded + env.round_error_epsilon
    out_of_tw[:, :, 0] = False
    mask[out_of_tw] = float("-inf")
    mask[:, :, 0][env.at_the_depot] = float("-inf")
    depot_mask = ~env.at_the_depot
    mask[:, :, 0][depot_mask] = 0
    return mask
