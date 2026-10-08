import torch
from masks_unified.mask_registry import register_mask


@register_mask("cvrp")
def mask_cvrp(env, selected):
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
    at_depot_expanded = env.at_the_depot[:, :, None].expand(-1, -1, 1)
    mask[:, :, 0:1] = torch.where(
        at_depot_expanded,
        torch.tensor(float("-inf"), device=mask.device),
        torch.tensor(0.0, device=mask.device),
    )
    return mask
