
import torch
from .mask_registry import register_mask

@register_mask("bp")
def bp_constraint_mask(env) -> torch.Tensor:
    current_node = env.current_node.clone()
    demands = env.depot_node_demand[:, None, :].expand(env.batch_size, env.pomo_size, -1)
    num_nodes = env.problem_size + env.depot_num

    mask = torch.zeros((current_node.size(0), current_node.size(1), num_nodes))
    gathering_index = current_node.unsqueeze(-1)
    selected_demand = demands.gather(dim=2, index=gathering_index).squeeze(dim=2)
    load_with_backhaul = (selected_demand < 0).unsqueeze(-1)
    mask[load_with_backhaul&(demands>0)] = float('-inf') #如果访问了B点，L点便不可访问

    return mask

