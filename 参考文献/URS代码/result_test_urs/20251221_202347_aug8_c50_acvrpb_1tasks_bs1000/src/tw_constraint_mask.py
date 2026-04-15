import torch

from .mask_registry import register_mask
@register_mask("tw")
def tw_constraint_mask(env) -> torch.Tensor:
    round_error_epsilon = 0.00001
    speed = 1.0
    num_nodes = env.problem_size + env.depot_num

    depot_end = env.tw_end[0,0]
    batch_size = env.batch_size
    pomo_size = env.pomo_size
    #当前点到其他全部点的距离
    current_node_to_all_dist = env.dist.gather(1, env.current_node[:, :, None].expand(-1, -1, num_nodes))
    #其他点的距离回到depot的距离
    current_all_to_depot_dist = env.dist.gather(2, env.current_depot[:, None, :].expand(-1, num_nodes, -1))
    current_all_to_depot_dist = current_all_to_depot_dist.permute(0, 2, 1)

    next_time_required= env.current_time[:, :, None] + current_node_to_all_dist / speed
    arrival_time = torch.max(next_time_required,env.tw_start[:, None, :].expand(-1, pomo_size, -1))

    out_of_tw = arrival_time > env.tw_end[:, None, :].expand(-1, pomo_size, -1) + round_error_epsilon
    mask = torch.zeros((batch_size, pomo_size, num_nodes))


    if env.open_route:
        out_of_tw[:, :, 0] = False
        mask[out_of_tw] = float('-inf')
    else:
        mask[out_of_tw] = float('-inf')
        fail_return_depot = arrival_time + env.service_time[:, None, :].expand(-1, pomo_size, -1) + current_all_to_depot_dist/speed > depot_end + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        mask[fail_return_depot] = float('-inf')

    return mask