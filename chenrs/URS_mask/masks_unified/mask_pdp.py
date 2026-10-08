import torch
from masks_unified.mask_registry import register_mask


@register_mask("pdp")
def mask_pdp(env, selected):
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.current_node.size(0)
    pomo_size = env.current_node.size(1)
    mask = torch.zeros((batch_size, pomo_size, num_nodes), device=env.device)
    if env.selected_node_list.size(2) > 0:
        mask.scatter_(-1, env.selected_node_list, float("-inf"))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float("-inf"))
    visited_nodes = env.selected_node_list
    pd_mask = torch.zeros_like(mask)
    pd_mask[~env.to_deliver] = float("-inf")
    cond = (visited_nodes <= num_nodes // 2) & (visited_nodes > 0)
    target_idx = visited_nodes + num_nodes // 2
    b_idx, p_idx, s_idx = torch.where(cond)
    m_idx = target_idx[b_idx, p_idx, s_idx]
    pd_mask[b_idx, p_idx, m_idx] = 0
    mask = mask + pd_mask
    return mask
