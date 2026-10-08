import torch
from masks_unified.mask_registry import register_mask


@register_mask("pctsp")
def mask_pctsp(env, selected):
    num_nodes = env.problem_size + env.depot_num
    batch_size = env.current_node.size(0)
    pomo_size = env.current_node.size(1)
    mask = torch.zeros((batch_size, pomo_size, num_nodes), device=env.device)
    if env.selected_node_list.size(2) > 0:
        mask.scatter_(-1, env.selected_node_list, float("-inf"))
    mask.scatter_(-1, env.current_node.unsqueeze(-1), float("-inf"))
    mask[:, :, 0] = torch.where(
        env.at_the_depot,
        torch.tensor(float("-inf"), device=env.device),
        torch.tensor(0.0, device=env.device),
    )
    need_more_prize = (env.collected_prize < 1.0) & (env.selected_count < num_nodes)
    mask[:, :, 0] = torch.where(
        need_more_prize,
        torch.tensor(float("-inf"), device=env.device),
        mask[:, :, 0],
    )
    return mask
