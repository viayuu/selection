
import torch
from .mask_registry import register_mask
@register_mask("visited")
def mask_visited_nodes(env) -> torch.Tensor:
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.batch_size
    pomo_size = env.pomo_size


    mask = torch.zeros((batch_size, pomo_size, num_nodes))
    mask.scatter_(-1, env.selected_node_list, float('-inf'))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float('-inf'))
    return mask