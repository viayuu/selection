import torch
from .mask_registry import register_mask


@register_mask("sdvrp")
def mask_sdvrp(env, selected):
    round_error_epsilon = 1e-6
    if hasattr(env, "dynamic_demand") and env.dynamic_demand is not None:
        mask_index = env.dynamic_demand == 0
        env.visited_ninf_flag[mask_index] = float("-inf")
    env.visited_ninf_flag[:, :, 0][~env.at_the_depot] = 0
    mask = env.visited_ninf_flag.clone()
    full_index = (
        (env.load <= round_error_epsilon).unsqueeze(-1).expand(-1, -1, env.problem_size)
    )
    mask[:, :, 1:][full_index] = float("-inf")
    newly_finished = (env.visited_ninf_flag[..., 1:] == float("-inf")).all(dim=2)
    env.finished = env.finished + newly_finished
    return mask
