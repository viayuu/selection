import torch


from .mask_registry import register_mask
@register_mask("l")
def route_limit_constraint_mask(env) -> torch.Tensor:

    num_nodes = env.problem_size + env.depot_num
    round_error_epsilon = 0.00001
    batch_size = env.batch_size
    pomo_size = env.pomo_size

    route_limit = env.route_limit[:, :, None].expand(batch_size, pomo_size, num_nodes)
    current_node_to_all_dist = env.dist.gather(
        1, env.current_node[:, :, None].expand(-1, -1, num_nodes)
    )
    current_all_to_depot_dist = env.dist.gather(
        2, env.current_depot[:, None, :].expand(-1, num_nodes, -1)
    )
    current_all_to_depot_dist = current_all_to_depot_dist.permute(0, 2, 1)

    mask = torch.zeros((batch_size, pomo_size, num_nodes))


    if env.open_route:
        # check route limit constraint: length + cur->next <= route_limit
        route_too_large = env.length[:, :,None] + current_node_to_all_dist > route_limit + round_error_epsilon
        route_too_large[:, :, 0] = False
        # shape: (batch, pomo, problem+1)
        mask[route_too_large] = float('-inf')
    else:
        # check route limit constraint: length + cur->next->depot <= route_limit
        route_too_large = env.length[:, :,None] + current_node_to_all_dist + current_all_to_depot_dist > route_limit + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        mask[route_too_large] = float('-inf')

    return mask
