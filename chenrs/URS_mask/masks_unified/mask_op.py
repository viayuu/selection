import torch
from masks_unified.mask_registry import register_mask


@register_mask("op")
def mask_op(env, selected):
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.batch_size
    pomo_size = env.pomo_size
    mask = torch.zeros((batch_size, pomo_size, num_nodes), device=env.device)
    if env.selected_node_list.size(2) > 0:
        mask.scatter_(-1, env.selected_node_list, float("-inf"))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float("-inf"))
    current_node_to_all_dist = env.dist.gather(
        1, selected[:, :, None].expand(-1, -1, num_nodes)
    )
    current_depot = torch.zeros(
        (batch_size, 1), dtype=torch.long, device=selected.device
    )
    current_all_to_depot_dist = env.dist.gather(
        2, current_depot[:, None, :].expand(-1, num_nodes, -1)
    )
    current_all_to_depot_dist = current_all_to_depot_dist.permute(0, 2, 1)
    current_all_to_depot_dist = current_all_to_depot_dist.expand(-1, pomo_size, -1)
    dist_via_node_to_depot = current_node_to_all_dist + current_all_to_depot_dist
    length_exceeded = (
        dist_via_node_to_depot
        > env.tour_maxlength.unsqueeze(-1) + env.round_error_epsilon
    )
    mask[length_exceeded] = float("-inf")
    at_depot_expanded = env.at_the_depot.unsqueeze(-1).expand(-1, -1, env.depot_num)
    mask[:, :, : env.depot_num][at_depot_expanded] = float("-inf")
    not_at_depot_expanded = (
        (~env.at_the_depot).unsqueeze(-1).expand(-1, -1, env.depot_num)
    )
    mask[:, :, : env.depot_num][not_at_depot_expanded] = 0
    finished = env.at_the_depot & (env.selected_count > 1)
    finished_expanded = finished.unsqueeze(-1).expand(-1, -1, env.problem_size)
    mask[:, :, 1:][finished_expanded] = float("-inf")
    return mask
