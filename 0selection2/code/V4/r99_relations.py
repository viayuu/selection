"""Input-only local or globally balanced distance messages.

No routes, solver outcomes, labels or learned parameters enter this transform.
The soft assignment is a finite-iteration travel-graph view, not a VRP bound.
"""

import torch
import torch.nn.functional as F

from .r88_model import prepare_role_inputs


WIDTH = 32
VARIANTS = ('row', 'balanced')
SPEC = dict(format='r99_competition_messages_v1', width=WIDTH,
    temperature=.05, iterations=64, diagonal='excluded',
    scale='mean_valid_off_diagonal_input_distance',
    channels='outgoing8,incoming8,outgoing_twostep8,incoming_twostep8',
    role='R88B_physical_TSP_node0', precision='FP32_transform_then_FP16',
    matrix_semantics='directed_input_costs', vrp_bound=False,
    hard_constraint_filter=False, labels_used=False)


def input_geometry(batch):
    """Original eight node attributes, or original ATSP row/column summaries."""
    mask = batch['node_mask']
    if mask.dtype != torch.bool or mask.ndim != 2 or not (mask.sum(1) >= 2).all():
        raise ValueError('At least two valid nodes and a boolean mask are required')
    n = mask.shape[1]
    valid = mask[:, :, None] & mask[:, None, :]
    valid &= ~torch.eye(n, dtype=torch.bool, device=mask.device)[None]
    if batch['kind'] == 'coord':
        prepared = prepare_role_inputs(batch, 'B_execution_anchor')
        node = prepared['node'].float()
        node = F.pad(node, (0, max(0, 8 - node.shape[-1])))[..., :8]
        distance = torch.cdist(node[..., :2], node[..., :2],
                               compute_mode='donot_use_mm_for_euclid_dist')
    elif batch['kind'] == 'matrix':
        distance = batch['matrix'].float()
        if distance.shape != (len(mask), n, n):
            raise ValueError('Directed matrix shape does not match valid nodes')
        zero = distance.masked_fill(~valid, 0.)
        row_count = valid.sum(2).clamp_min(1)
        col_count = valid.sum(1).clamp_min(1)
        low = distance.masked_fill(~valid, torch.inf)
        high = distance.masked_fill(~valid, -torch.inf)
        node = torch.stack((zero.sum(2) / row_count, zero.sum(1) / col_count,
            low.amin(2), low.amin(1), high.amax(2), high.amax(1),
            (distance-distance.transpose(1, 2)).abs().masked_fill(~valid, 0.).sum(2)/row_count,
            distance.diagonal(dim1=1, dim2=2)), -1)
        node = node.masked_fill(~mask[..., None], 0.)
    else:
        raise ValueError('Use the existing coordinate or matrix input kind')
    if (not torch.isfinite(node[mask]).all()
            or not torch.isfinite(distance[valid]).all()
            or (distance[valid] < 0).any()):
        raise ValueError('Input distances/attributes must be finite and nonnegative distances')
    return node.masked_fill(~mask[..., None], 0.), distance, valid


@torch.no_grad()
def transport(batch, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown competition transform')
    with torch.autocast(device_type=batch['node_mask'].device.type, enabled=False):
        attributes, distance, valid = input_geometry(batch)
        mask = batch['node_mask']
        scale = distance.masked_fill(~valid, 0.).sum((1, 2), keepdim=True)
        scale = (scale / valid.sum((1, 2), keepdim=True).clamp_min(1)).clamp_min(1e-6)
        logp = (-distance / (SPEC['temperature'] * scale)).masked_fill(~valid, -torch.inf)
        if variant == 'balanced':
            for _ in range(SPEC['iterations']):
                row = logp.logsumexp(2, keepdim=True).masked_fill(~mask[:, :, None], 0.)
                logp = logp - row
                col = logp.logsumexp(1, keepdim=True).masked_fill(~mask[:, None, :], 0.)
                logp = logp - col
        else:
            norm = logp.logsumexp(2, keepdim=True).masked_fill(~mask[:, :, None], 0.)
            logp = logp - norm
        p = logp.exp().masked_fill(~valid, 0.)
        row_mass, col_mass = p.sum(2), p.sum(1)
        outgoing = p / row_mass[:, :, None].clamp_min(1e-20)
        incoming = (p / col_mass[:, None, :].clamp_min(1e-20)).transpose(1, 2)
        first_out = outgoing @ attributes
        first_in = incoming @ attributes
        messages = torch.cat((first_out, first_in,
                              outgoing @ first_out, incoming @ first_in), -1)
        messages = messages.masked_fill(~mask[..., None], 0.)
        if not torch.isfinite(messages).all():
            raise ValueError('Nonfinite input-only competition messages')
        residual = torch.maximum((row_mass-1.).abs().masked_fill(~mask, 0.).amax(),
                                 (col_mass-1.).abs().masked_fill(~mask, 0.).amax())
    return messages.to(torch.float16), residual


def node_messages(batch, variant):
    return transport(batch, variant)[0]
