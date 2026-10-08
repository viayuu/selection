import torch
from masks_unified.mask_registry import register_mask


@register_mask("tsp")
def mask_tsp(env, selected):
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.current_node.size(0)
    pomo_size = env.current_node.size(1)
    mask = torch.zeros((batch_size, pomo_size, num_nodes), device=env.device)
    if env.selected_node_list.size(2) > 0:
        mask.scatter_(-1, env.selected_node_list, float("-inf"))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float("-inf"))
    return mask
