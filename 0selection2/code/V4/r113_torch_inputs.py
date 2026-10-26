"""Device-resident, vectorized R113 v1 inference-input compilation.

API: ``compile_inputs_torch(raw, problem, spec=ProgramSpec())``. K=12 and
neighbors=2 are fixed. Outputs have the original INPUT_KEYS, shapes and dtypes.
No teacher/target/reference execution, NumPy conversion, host scalar extraction,
or Python loop over examples is used. CPU and CUDA follow the same Torch path;
dynamic validity checks use device-side asynchronous assertions on CUDA.

Pass original R84/R88 pre-role raw fields. TSP [B,N,2] is canonical; zero-padded
[B,N,8] with an optional binary slot-3 customer role is also accepted, but the
role is deliberately discarded for this separate program's canonical anchor.
``node_geom`` and listed candidate metadata are accepted but never consumed.
Raw tensors are not mutated. Compilation is detached, as in the NumPy reference.

Coordinate workspace is O(B*(K+1)*N), not O(B*N*N). ATSP retains its directed
matrix. Geometry and ordering use FP64 before final FP32 feature conversion.
No GPU measurements have been made by this module's CPU-only test suite.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .r113_process_data import (
    BINARY_FIELDS, INPUT_KEYS, PROBLEMS, ProgramSpec, RAW_KEYS,
    SCALAR_FIELDS, TASK_TYPES,
)


ALLOWED_RAW_KEYS = frozenset(RAW_KEYS) | {'node_geom'}
LABEL_KEYS = frozenset({
    'cost', 'costs', 'ind', 'winner', 'raw_winner', 'label', 'labels',
    'target', 'targets', 'hint', 'hints', 'teacher', 'teacher_states',
    'route', 'routes', 'solution', 'solutions', 'cmp_target', 'cmp_valid',
    'cmp_true3', 'cost_regret', 'scalar', 'binary', 'pointer', 'reached',
    'frontier', 'compatible', 'scalar_delta', 'state_mask', 'transition_mask',
})
INTEGER_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)


def _assert(condition, message):
    # Unlike a Python bool(tensor), this does not synchronize CUDA with the host.
    torch._assert_async(condition, message)


def _lexsort(keys, valid):
    """Stable lexicographic sort; keys run from least to most significant."""
    batch, size = valid.shape
    order = torch.arange(size, device=valid.device).expand(batch, -1)
    for key in keys:
        scores = key.masked_fill(~valid, float('inf')).gather(1, order)
        rank = torch.argsort(scores, dim=1, stable=True)
        order = order.gather(1, rank)
    return order


def _gather_nodes(values, indices):
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))


def _metadata(raw, key, device, dtype=None):
    return torch.as_tensor(raw[key], device=device, dtype=dtype)


@torch.no_grad()
def compile_inputs_torch(raw, problem, spec=ProgramSpec()):
    """Compile only allowed raw inference inputs, entirely on their device.

    Optional metadata may be Python constants or tensors. node/matrix and n must
    be tensors; n and masks are moved to the node/matrix device if necessary.
    Invalid tensor values raise RuntimeError (asynchronously on CUDA); structural
    errors and forbidden keys raise ValueError/TypeError before tensor execution.
    ``node_geom`` is safe to pass and ignored, never reinterpreted as raw fields.
    """
    if not isinstance(raw, Mapping):
        raise TypeError('raw must be an input-only mapping')
    labels = set(raw) & LABEL_KEYS
    if labels:
        raise ValueError(f'Supervision is forbidden in the compiler: {sorted(labels)}')
    unknown = set(raw) - ALLOWED_RAW_KEYS
    if unknown:
        raise ValueError(f'Only listed raw input keys are allowed: {sorted(unknown)}')
    problem = str(problem).upper()
    if problem not in TASK_TYPES:
        raise ValueError(f'Unknown explicit task type: {problem}')
    if (spec.max_steps, spec.neighbors) != (12, 2):
        raise ValueError('This deployment compiler is frozen to max_steps=12, neighbors=2')
    kind = 'matrix' if problem == 'ATSP' else 'coord'
    if raw.get('kind', kind) != kind:
        raise ValueError('Task type and raw representation disagree')
    if ('node' in raw and kind == 'matrix') or ('matrix' in raw and kind == 'coord'):
        raise ValueError('Supply node OR matrix for the declared task, not both')
    values, sizes = raw['matrix' if kind == 'matrix' else 'node'], raw['n']
    if not torch.is_tensor(values) or not torch.is_tensor(sizes):
        raise TypeError('node/matrix and n must be Torch tensors')
    if (values.ndim != 3 or not values.shape[0] or values.shape[1] < 2
            or sizes.ndim != 1 or sizes.shape[0] != values.shape[0]
            or sizes.dtype not in INTEGER_DTYPES or values.is_complex()):
        raise ValueError('Require nonempty raw [B,N,D]/[B,N,N] and integer n[B]')
    batch, size, _ = values.shape
    device = values.device
    c, opened, backhaul, length, tw = TASK_TYPES[problem]
    if kind == 'matrix' and values.shape[2] != size:
        raise ValueError('ATSP matrix must be square')
    dimension = 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    if kind == 'coord' and values.shape[2] != dimension:
        if problem != 'TSP' or values.shape[2] != 8:
            raise ValueError(f'{problem}: expected {dimension} raw fields (TSP also accepts role-pad 8)')
    sizes = sizes.to(device=device, dtype=torch.int64)
    _assert(((sizes >= 2) & (sizes <= size)).all(), 'Need 2 <= n <= raw padded size')
    indices = torch.arange(size, device=device)
    valid = indices[None] < sizes[:, None]
    if 'node_mask' in raw:
        mask = _metadata(raw, 'node_mask', device)
        if mask.dtype != torch.bool or mask.shape != valid.shape:
            raise ValueError('node_mask must be boolean [B,N]')
        _assert((mask == valid).all(), 'node_mask disagrees with true n prefix')
    problem_index = PROBLEMS.index(problem)
    if 'problem_id' in raw:
        _assert((_metadata(raw, 'problem_id', device) == problem_index).all(), 'Raw problem_id mismatch')
    if 'cbits' in raw:
        bits = _metadata(raw, 'cbits', device)
        if bits.shape not in ((5,), (1, 5), (batch, 5)):
            raise ValueError('cbits must have shape [5], [1,5] or [B,5]')
        _assert((bits == torch.tensor(TASK_TYPES[problem], device=device)).all(), 'Raw constraint bits mismatch')
    if 'coord_dist' in raw:
        coord_dist = _metadata(raw, 'coord_dist', device, torch.int64)
        if coord_dist.shape != (batch,):
            raise ValueError('coord_dist must be distribution metadata [B]')
    else:
        coord_dist = torch.full((batch,), 4, device=device, dtype=torch.int64)

    # Disable outer AMP explicitly: both canonical ranking and neighbor ties are FP64.
    with torch.autocast(device_type=device.type, enabled=False):
        if kind == 'coord':
            if problem == 'TSP' and values.shape[2] == 8:
                padding = values[..., (2, 4, 5, 6, 7)]
                role = values[..., 3]
                _assert(((padding == 0) | ~valid[..., None]).all(), 'TSP role-pad has nonzero raw constraint slots')
                _assert((((role == 0) | (role == 1)) | ~valid).all(), 'TSP slot 3 must be a binary customer role')
            raw_nodes = F.pad(values[..., :dimension].to(torch.float64), (0, 8 - dimension))
            raw_nodes = raw_nodes.masked_fill(~valid[..., None], 0.)
            _assert(torch.isfinite(raw_nodes).all(), 'Nonfinite valid raw node input')
            if c:
                depot_slot = raw_nodes[..., 3]
                depot = valid & (depot_slot == 1)
                _assert((((depot_slot == 0) | (depot_slot == 1)) | ~valid).all(), 'Depot flag must be binary')
                _assert((depot.sum(1) == 1).all(), 'Exactly one physical depot is required')
                anchor_index = depot.to(torch.int64).argmax(1)
                _assert((raw_nodes[..., 2].gather(1, anchor_index[:, None]) == 0).all(), 'Depot demand must be zero')
                customers = valid & ~depot
                if not backhaul:
                    _assert((raw_nodes[..., 2] >= 0).all(), 'Negative demand on a non-B task')
                if length:
                    limits = raw_nodes[..., 4]
                    low = limits.masked_fill(~customers, float('inf')).amin(1)
                    high = limits.masked_fill(~customers, float('-inf')).amax(1)
                    _assert(((low > 0) & (low == high)).all(), 'L needs one positive customer route limit')
                if tw:
                    _assert((raw_nodes[..., 5:7] >= 0).all(), 'Negative service/window start')
                    _assert(((raw_nodes[..., 7] >= raw_nodes[..., 6]) | ~customers).all(), 'Reversed customer window')
            keys = (raw_nodes[..., 2], raw_nodes[..., 1], raw_nodes[..., 0])
            if tw:
                keys = (*keys, raw_nodes[..., 7])
            order = _lexsort(keys, valid)
        else:
            pair_valid = valid[:, :, None] & valid[:, None]
            matrix = values.to(torch.float64).masked_fill(~pair_valid, 0.)
            _assert(torch.isfinite(matrix).all(), 'Nonfinite valid matrix input')
            _assert((matrix >= 0).all(), 'Negative directed travel is outside this program')
            diagonal = matrix.diagonal(dim1=1, dim2=2)
            mean_out = (matrix.sum(2) - diagonal) / (sizes - 1)[:, None]
            mean_in = (matrix.sum(1) - diagonal) / (sizes - 1)[:, None]
            order = _lexsort((mean_in, mean_out), valid)
        if not c:
            anchor_index = order[:, 0]
        eligible = valid & (indices[None] != anchor_index[:, None])
        # A stable partition removes each row's anchor without a row-wise loop.
        partition = torch.argsort((~eligible.gather(1, order)).to(torch.int64), dim=1, stable=True)
        order = order.gather(1, partition)
        steps = torch.ceil(torch.sqrt((sizes - 1).to(torch.float64))).to(torch.int64).clamp(max=12)
        node_mask = torch.arange(12, device=device)[None] < steps[:, None]
        selected = F.pad(order[:, :min(12, size)], (0, max(0, 12 - size)), value=0)
        selected = selected.masked_fill(~node_mask, 0)
        chosen = torch.cat((anchor_index[:, None], selected), 1)

        if kind == 'coord':
            xy = raw_nodes[..., :2]
            points = _gather_nodes(xy, chosen)
            dx = points[:, :, None, 0] - xy[:, None, :, 0]
            dy = points[:, :, None, 1] - xy[:, None, :, 1]
            distances = torch.sqrt(dx.square() + dy.square())
            mean = distances.masked_fill(~valid[:, None], 0.).sum(2) / (sizes - 1)[:, None]
            from_anchor = distances[:, 0].gather(1, chosen)
            to_anchor = from_anchor
            mean_out_chosen = mean_in_chosen = mean
            base = _gather_nodes(raw_nodes, chosen)
            local_distance = distances[:, 1:].gather(2, selected[:, None].expand(-1, 12, -1))
        else:
            from_anchor = matrix.gather(1, anchor_index[:, None, None].expand(-1, 1, size)).squeeze(1).gather(1, chosen)
            to_anchor = matrix.gather(2, anchor_index[:, None, None].expand(-1, size, 1)).squeeze(2).gather(1, chosen)
            mean_out_chosen, mean_in_chosen = mean_out.gather(1, chosen), mean_in.gather(1, chosen)
            base = matrix.new_zeros(batch, 13, 8)
            local_distance = matrix.gather(1, selected[:, :, None].expand(-1, -1, size)).gather(
                2, selected[:, None].expand(-1, 12, -1))
        features = torch.cat((base, from_anchor[..., None], to_anchor[..., None],
                              mean_out_chosen[..., None], mean_in_chosen[..., None]), -1).to(torch.float32)
        nodes = features[:, 1:].masked_fill(~node_mask[..., None], 0.).contiguous()
        anchor = features[:, 0].clone()
        if length:
            anchor[:, 4] = features[:, 1, 4]
        valid_pair = node_mask[:, :, None] & node_mask[:, None]
        offdiag = ~torch.eye(12, device=device, dtype=torch.bool)[None]
        edge_mask = valid_pair & offdiag
        scores = local_distance.masked_fill(~edge_mask, float('inf'))
        neighbors = torch.argsort(scores, dim=2, stable=True)[..., :2]
        topology = torch.zeros_like(edge_mask).scatter(2, neighbors, True) & edge_mask
        edge_features = torch.stack((local_distance, local_distance.transpose(1, 2)), -1)
        edge_features = edge_features.to(torch.float32).masked_fill(~valid_pair[..., None], 0.)

    task_flags = torch.tensor((c, opened, backhaul, length, tw, problem == 'ATSP', c),
                              device=device, dtype=torch.bool).expand(batch, -1).contiguous()
    graph_mask = torch.tensor((not c, bool(c), bool(backhaul)), device=device, dtype=torch.bool).expand(batch, -1).contiguous()
    scalar_mask = torch.tensor((bool(c),) * 7 + (True,) * 2 + (bool(tw),) * 5 + (bool(length), bool(c)),
                              device=device, dtype=torch.bool).expand(batch, len(SCALAR_FIELDS)).contiguous()
    binary_mask = torch.tensor((c, c, backhaul, backhaul, tw, length, backhaul),
                              device=device, dtype=torch.bool).expand(batch, len(BINARY_FIELDS)).contiguous()
    result = dict(nodes=nodes, anchor=anchor, edge_features=edge_features, topology=topology,
        node_mask=node_mask, sequence=selected.masked_fill(~node_mask, -1), anchor_index=anchor_index,
        steps=steps, n=sizes, problem_id=torch.full((batch,), problem_index, device=device, dtype=torch.int64),
        coord_dist=coord_dist, task_flags=task_flags, graph_mask=graph_mask,
        scalar_mask=scalar_mask, binary_mask=binary_mask)
    return {key: result[key] for key in INPUT_KEYS}
