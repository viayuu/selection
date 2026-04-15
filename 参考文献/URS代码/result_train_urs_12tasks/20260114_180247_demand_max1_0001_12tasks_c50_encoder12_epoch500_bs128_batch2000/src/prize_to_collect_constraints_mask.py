
import torch


from .mask_registry import register_mask
@register_mask("pc")
def prize_to_collect_constraint_mask(env) -> torch.Tensor:

    num_nodes = env.problem_size + env.depot_num


    mask = torch.zeros((env.batch_size, env.pomo_size, num_nodes))
    mask[:,:,0] = float('-inf')
    allow = (env.collected_prize >= 1.) | (env.selected_count == num_nodes)

    # 将finished的其他action设置为-inf
    finished = env.at_the_depot & (env.selected_count > 1)
    finished_extend = finished.unsqueeze(-1).expand(-1, -1, num_nodes)
    mask[finished_extend] = float('-inf')
    mask[:, :, 0][allow] = 0

    return mask