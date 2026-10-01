"""Input-only sparse relations. An edge (i, j) carries j's message into i."""

import torch
import torch.nn.functional as F


EDGE_FIELDS = (
    'forward_cost', 'reverse_cost', 'forward_normalized', 'reverse_normalized',
    'dx_normalized', 'dy_normalized', 'source_depot_distance', 'target_depot_distance',
    'source_demand', 'target_demand', 'positive_demand_sum', 'negative_demand_sum',
    'demand_sign_product', 'tw_overlap', 'forward_slack', 'reverse_slack',
    'route_limit_ratio', 'geometric_neighbor', 'temporal_neighbor',
    'low_outgoing_cost', 'low_incoming_cost', 'depot_connection',
    'geometry_valid', 'capacity_valid', 'backhaul_valid', 'time_window_valid',
    'route_limit_valid', 'open_route', 'directed_matrix', 'source_depot', 'target_depot',
)


def nearest(values, valid, k):
    """Include all kth-value ties; exclude invalid entries and self edges."""
    values = values.masked_fill(~valid, float('inf'))
    count = valid.sum(-1)
    ordered = values.topk(min(k, values.size(-1)), largest=False).values
    position = count.clamp(min=1, max=ordered.size(-1)) - 1
    boundary = ordered.gather(-1, position.unsqueeze(-1))
    return valid & (values <= boundary)


@torch.no_grad()
def build_relation_graph(batch, geometric_k=16, temporal_k=8):
    mask = batch['node_mask'].bool()
    b, n = mask.shape
    device = mask.device
    valid = mask[:, :, None] & mask[:, None, :]
    valid &= ~torch.eye(n, dtype=torch.bool, device=device)[None]
    zeros = torch.zeros_like(valid)
    coord = batch['kind'] == 'coord'
    cbits = batch['cbits'].float()
    if coord:
        node = F.pad(batch['node'].float(), (0, max(0, 8 - batch['node'].size(-1))))[..., :8]
        node = node.masked_fill(~mask[..., None], 0)
        distance = torch.cdist(node[..., :2], node[..., :2])
        depot = (node[..., 3] > .5) & mask
        customers = mask & ~depot
        customer_pairs = customers[:, :, None] & customers[:, None, :] & valid
        geometric = nearest(distance, valid, geometric_k)
        time_difference = (node[..., 6, None] - node[:, None, :, 6]).abs()
        temporal = nearest(time_difference, customer_pairs, temporal_k) & (cbits[:, 4] > .5)[:, None, None]
        depot_edges = valid & (depot[:, :, None] | depot[:, None, :])
        outgoing, incoming = zeros, zeros
    else:
        distance = batch['matrix'].float().masked_fill(~valid, 0)
        outgoing = nearest(distance, valid, geometric_k)
        incoming = nearest(distance.transpose(1, 2), valid, geometric_k)
        geometric, temporal, depot_edges = zeros, zeros, zeros
        depot = torch.zeros_like(mask)
    adjacency = geometric | temporal | depot_edges | outgoing | incoming
    sample, source, target = adjacency.nonzero(as_tuple=True)
    scale = (distance.abs().masked_fill(~valid, 0).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)).clamp_min(1e-6)
    forward, reverse = distance[sample, source, target], distance[sample, target, source]
    ell = scale[sample]
    features = torch.zeros(len(sample), len(EDGE_FIELDS), device=device)
    features[:, 0:4] = torch.stack((forward, reverse, forward / ell, reverse / ell), -1)
    features[:, 17:22] = torch.stack([x[sample, source, target] for x in
                                    (geometric, temporal, outgoing, incoming, depot_edges)], -1).float()
    features[:, 22] = float(coord)
    features[:, 28] = float(not coord)
    if coord:
        src, dst = node[sample, source], node[sample, target]
        flags = cbits[sample]
        demand_valid = flags[:, 0] > .5
        backhaul_valid = flags[:, 2] > .5
        time_valid = (flags[:, 4] > .5) & ~depot[sample, source] & ~depot[sample, target]
        limit_count = ((node[..., 4] > 0) & mask).sum(1).clamp_min(1)
        limit = node[..., 4].masked_fill(~mask, 0).sum(1) / limit_count
        limit_valid = (flags[:, 3] > .5) & (limit[sample] > 0)
        depot_xy = (node[..., :2] * depot[..., None]).sum(1)
        endpoint_distance = (node[..., :2] - depot_xy[:, None]).norm(dim=-1)
        features[:, 4:6] = (dst[:, :2] - src[:, :2]) / ell[:, None]
        has_depot = depot.any(1)[sample]
        features[:, 6] = endpoint_distance[sample, source] / ell * has_depot
        features[:, 7] = endpoint_distance[sample, target] / ell * has_depot
        demands = torch.stack((src[:, 2], dst[:, 2]), -1)
        features[:, 8:10] = demands * demand_valid[:, None]
        features[:, 10] = demands.clamp_min(0).sum(1) * demand_valid
        features[:, 11] = (-demands).clamp_min(0).sum(1) * backhaul_valid
        features[:, 12] = demands.sign().prod(1) * backhaul_valid
        overlap = (torch.minimum(src[:, 7], dst[:, 7]) - torch.maximum(src[:, 6], dst[:, 6])).clamp_min(0)
        span = (torch.maximum(src[:, 7], dst[:, 7]) - torch.minimum(src[:, 6], dst[:, 6])).clamp_min(1e-6)
        time_scale = (node[..., 7] * customers).sum(1) / customers.sum(1).clamp_min(1)
        time_scale = time_scale[sample].clamp_min(1e-6)
        # Original routing environments use Euclidean travel distance / speed=1.
        features[:, 13] = overlap / span * time_valid
        features[:, 14] = (dst[:, 7] - src[:, 6] - src[:, 5] - forward) / time_scale * time_valid
        features[:, 15] = (src[:, 7] - dst[:, 6] - dst[:, 5] - reverse) / time_scale * time_valid
        path = endpoint_distance[sample, source] + forward + (1 - flags[:, 1]) * endpoint_distance[sample, target]
        features[:, 16] = path / limit[sample].clamp_min(1e-6) * limit_valid
        features[:, 23:28] = torch.stack((demand_valid, backhaul_valid, time_valid, limit_valid, flags[:, 1] > .5), -1).float()
        features[:, 29] = depot[sample, source]
        features[:, 30] = depot[sample, target]
    index = torch.stack((sample * n + source, sample * n + target))
    return dict(index=index, features=features, degree=adjacency.sum(-1).reshape(-1), shape=(b, n))
