"""Input-only sparse graphs shared by all candidate recurrent workspaces."""

import torch

from code.unified_selector.registry import P2I


NEIGHBORS = 16
SPECIAL_ROLES = ('cls', 'condition', 'stats')
DISTANCE_FEATURES = ('d_ij', 'd_ij_over_scale', 'd_ji', 'd_ji_over_scale')
# Node role IDs: -1 padding, 0 customer, 1 depot, 2 CLS, 3 condition, 4 stats.


@torch.no_grad()
def build_workspace_graph(batch):
    """Return directed edges i->j in flattened [B, N+3] node coordinates.

    The original encoder prepends CLS/condition/stats; physical nodes follow in
    their original order, including padding. ``n_nodes`` is that padded stride,
    not the number of valid nodes. Messages gather at ``flat_source`` and scatter
    to ``flat_receiver``. One graph serves all candidate workspaces and both
    dropout forwards of a microbatch.

    Distance scale is the mean absolute valid off-diagonal physical travel
    distance, clamped to 1e-6. Special-token edges have four zero features;
    ``physical_edge`` distinguishes these from zero-distance physical edges.
    Self edges are excluded; the recurrent target state supplies local memory.
    """
    mask = batch['node_mask']
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError('Expected the original physical [B,N] boolean node_mask')
    batch_size, physical_n = mask.shape
    special_n = len(SPECIAL_ROLES)
    n_nodes = physical_n + special_n
    device = mask.device
    with torch.autocast(device_type=device.type, enabled=False):
        if batch['kind'] == 'coord':
            node = batch['node']
            if node.shape[:2] != mask.shape or node.shape[-1] < 2:
                raise ValueError('Coordinate node fields do not match the physical mask')
            xy = node[..., :2].detach().float()
            distance = torch.cdist(xy, xy)
            depot = torch.zeros_like(mask)
            if batch['problem_id'] not in (P2I['TSP'], P2I['ATSP']) and node.shape[-1] >= 4:
                depot = mask & (node[..., 3] > .5)
        elif batch['kind'] == 'matrix':
            distance = batch['matrix'].detach().float()
            if distance.shape != (batch_size, physical_n, physical_n):
                raise ValueError('Directed matrix dimensions do not match the physical mask')
            depot = torch.zeros_like(mask)
        else:
            raise ValueError('Unknown original instance input kind')

        diagonal = torch.eye(physical_n, dtype=torch.bool, device=device)[None]
        physical_valid = mask[:, :, None] & mask[:, None, :] & ~diagonal
        ranking = distance.masked_fill(~physical_valid, float('inf'))
        physical_edges = torch.zeros_like(physical_valid)
        k = min(NEIGHBORS, max(physical_n - 1, 0))
        if k:
            # Outgoing: choose destinations per row. Incoming: sources per column.
            physical_edges.scatter_(2, ranking.topk(k, dim=2, largest=False).indices, True)
            physical_edges.scatter_(1, ranking.topk(k, dim=1, largest=False).indices, True)
        physical_edges &= physical_valid

        node_mask = torch.cat((torch.ones(batch_size, special_n, dtype=torch.bool, device=device), mask), dim=1)
        hubs = torch.cat((torch.ones(batch_size, special_n, dtype=torch.bool, device=device), depot), dim=1)
        edges = (node_mask[:, :, None] & node_mask[:, None, :] &
                 (hubs[:, :, None] | hubs[:, None, :]))
        edges[:, special_n:, special_n:] |= physical_edges
        edges &= ~torch.eye(n_nodes, dtype=torch.bool, device=device)[None]
        graph_id, source, receiver = edges.nonzero(as_tuple=True)
        physical_edge = (source >= special_n) & (receiver >= special_n)

        scale = (distance.masked_fill(~physical_valid, 0.).abs().sum((1, 2)) /
                 physical_valid.sum((1, 2)).clamp_min(1)).clamp_min(1e-6)
        distance_features = torch.zeros(source.numel(), len(DISTANCE_FEATURES), dtype=torch.float32, device=device)
        b = graph_id[physical_edge]
        i, j = source[physical_edge] - special_n, receiver[physical_edge] - special_n
        forward, reverse = distance[b, i, j], distance[b, j, i]
        distance_features[physical_edge] = torch.stack((forward, forward / scale[b],
                                                        reverse, reverse / scale[b]), dim=-1)

    physical_roles = torch.where(depot, 1, 0).masked_fill(~mask, -1)
    node_roles = torch.cat((torch.arange(2, 5, device=device)[None].expand(batch_size, -1), physical_roles), dim=1)
    return dict(flat_source=graph_id * n_nodes + source, flat_receiver=graph_id * n_nodes + receiver,
                distance_features=distance_features, physical_edge=physical_edge, distance_scale=scale,
                node_mask=node_mask, node_roles=node_roles, n_nodes=n_nodes, batch_size=batch_size,
                special_count=special_n, special_roles=SPECIAL_ROLES, distance_feature_names=DISTANCE_FEATURES)
