"""R92 C/D shared, deterministic input-only structural statistics.

The whitelist is intentional: no labels, costs, row IDs, candidate identities,
solver routes, or learned embeddings enter phi. Padding never enters a moment.
"""

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from .local_geometry import FIELDS


VERSION = 'r92_structural_phi_v1'
INPUT_KEYS = ('kind', 'node', 'matrix', 'node_mask', 'node_geom', 'n', 'd_in',
              'cbits', 'problem_desc', 'coord_dist')
STATS = ('mean', 'std', 'min', 'max', 'q25', 'q50', 'q75')
RAW_FIELDS = ('x', 'y', 'demand', 'is_depot', 'route_limit', 'service_time', 'tw_start', 'tw_end')
MATRIX_FIELDS = ('out_mean', 'out_min', 'out_std', 'in_mean', 'in_min', 'in_std',
                 'asym_abs_mean', 'asym_relative_mean', 'out_nearest1', 'out_nearest4',
                 'out_nearest8', 'in_nearest1', 'in_nearest4', 'in_nearest8')
XY_FIELDS = ('cov_xx', 'cov_xy', 'cov_yy', 'inertia', 'eigen_min', 'eigen_max',
             'anisotropy', 'bbox_area', 'depot_x', 'depot_y', 'has_depot')
CORR_FIELDS = ('x', 'y', 'demand', 'service_time', 'tw_start', 'tw_end', 'tw_width',
               'radius_centroid', 'radius_depot')
CORR_PAIRS = tuple(itertools.combinations(range(len(CORR_FIELDS)), 2))


def feature_columns():
    columns = ['n', 'log1p_n', 'd_in', 'is_matrix', 'coord_dist']
    columns += [f'cbits_{i}' for i in range(5)]
    columns += [f'problem_desc_{i}' for i in range(16)]
    columns += [f'raw_{field}_{stat}' for field in RAW_FIELDS for stat in STATS]
    columns += [f'geom_{field}_{stat}' for field in FIELDS for stat in STATS]
    columns += ['xy_' + field for field in XY_FIELDS]
    columns += [f'grid4x4_{i}' for i in range(16)]
    columns += [f'angle_{origin}_{i}' for origin in ('centroid', 'depot') for i in range(8)]
    columns += [f'radius_{origin}_{stat}' for origin in ('centroid', 'depot') for stat in STATS]
    columns += [f'corr_{CORR_FIELDS[i]}_{CORR_FIELDS[j]}' for i, j in CORR_PAIRS]
    columns += [f'matrix_{field}_{stat}' for field in MATRIX_FIELDS for stat in STATS]
    return tuple(columns)


COLUMNS = feature_columns()
FEATURE_DEFINITION = dict(version=VERSION, columns=list(COLUMNS), input_whitelist=list(INPUT_KEYS),
    statistics=list(STATS), std='population', quantiles='linear interpolation over valid nodes only',
    geometry='existing 12-channel R84 cached local geometry; no normalization/refitting',
    xy='inertia/covariance/eigenvalues, bounding-box-normalized grid4x4, angle8 at centroid/depot',
    correlations='all pairs of x/y/demand/service/TW endpoints/width/centroid radius/depot radius',
    matrix='directed row/column mean/min/std; absolute/relative asymmetry; nearest 1/4/8 means',
    matrix_diagonal='excluded from all edge statistics, including nearest neighbors',
    missing_channels='zero; kind/d_in/depot flags distinguish absent attributes',
    padding='excluded, even when padding contains NaN/Inf', output_dtype='float32',
    train_fitted_feature_statistics=False)


def input_batch(batch):
    return {key: batch[key] for key in INPUT_KEYS if key in batch}


def _moments(values, mask):
    """B,N,F -> B,F,7 without per-instance loops or padding quantiles."""
    valid = mask[..., None]
    values = values.float().masked_fill(~valid, 0.)
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite real input attribute')
    count = mask.sum(1).clamp_min(1)
    mean = values.sum(1) / count[:, None]
    var = (values - mean[:, None]).square().masked_fill(~valid, 0.).sum(1) / count[:, None]
    ordered = values.masked_fill(~valid, float('inf')).sort(dim=1).values
    quantiles = []
    for q in (0., 1., .25, .5, .75):
        position = (count - 1).float() * q
        low, high = position.floor().long(), position.ceil().long()
        shape = (-1, 1, values.shape[-1])
        lo = ordered.gather(1, low[:, None, None].expand(shape)).squeeze(1)
        hi = ordered.gather(1, high[:, None, None].expand(shape)).squeeze(1)
        quantiles.append(lo + (hi - lo) * (position - low)[:, None])
    result = torch.stack((mean, var.clamp_min(0).sqrt(), *quantiles), -1)
    return result.masked_fill((mask.sum(1) == 0)[:, None, None], 0.).flatten(1)


def _histogram(indices, mask, bins):
    result = torch.zeros(len(mask), bins, device=mask.device)
    result.scatter_add_(1, indices.clamp(0, bins - 1), mask.float())
    return result / mask.sum(1).clamp_min(1)[:, None]


def _coordinate(node, geometry, mask):
    node = node.float().masked_fill(~mask[..., None], 0.)
    if node.shape[-1] > 8:
        raise ValueError('R92 expects at most eight real node attributes')
    node = F.pad(node, (0, 8 - node.shape[-1]))
    xy = node[..., :2]
    count = mask.sum(1).clamp_min(1)[:, None]
    center = xy.sum(1) / count
    delta = xy - center[:, None]
    covariance = torch.einsum('bni,bnj->bij', delta.masked_fill(~mask[..., None], 0.),
                              delta.masked_fill(~mask[..., None], 0.)) / count[..., None]
    xx, cross, yy = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    trace = xx + yy
    gap = ((xx - yy).square() + 4 * cross.square()).clamp_min(0).sqrt()
    lower = xy.masked_fill(~mask[..., None], float('inf')).amin(1)
    upper = xy.masked_fill(~mask[..., None], -float('inf')).amax(1)
    span = upper - lower
    depot_mask = mask & (node[..., 3] > .5)
    has_depot = depot_mask.any(1)
    depot = (xy * depot_mask[..., None]).sum(1) / depot_mask.sum(1).clamp_min(1)[:, None]
    # TSP has no arbitrary first-node anchor; its absent depot falls back to centroid.
    depot = torch.where(has_depot[:, None], depot, center)
    depot_delta = xy - depot[:, None]
    radius_center, radius_depot = delta.norm(dim=-1), depot_delta.norm(dim=-1)
    shape = torch.stack((xx, cross, yy, trace, (trace-gap).clamp_min(0)/2,
                         (trace+gap)/2, gap/trace.clamp_min(1e-12), span.prod(1),
                         depot[:, 0], depot[:, 1], has_depot.float()), 1)
    cells = (((xy-lower[:, None]) / span[:, None].clamp_min(1e-12))*4).long().clamp(0, 3)
    grid = _histogram(cells[..., 0]*4+cells[..., 1], mask, 16)
    angles = []
    for displacement in (delta, depot_delta):
        angle = torch.atan2(displacement[..., 1], displacement[..., 0])
        indices = ((angle + math.pi) * (8/(2*math.pi))).long().clamp(0, 7)
        # Coincident nodes carry no angular direction.
        angles.append(_histogram(indices, mask & (displacement.norm(dim=-1) > 0), 8))
    attributes = torch.stack((node[..., 0], node[..., 1], node[..., 2], node[..., 5],
                              node[..., 6], node[..., 7], node[..., 7]-node[..., 6],
                              radius_center, radius_depot), -1)
    attributes = attributes.masked_fill(~mask[..., None], 0.)
    centered = (attributes-attributes.sum(1)[:, None]/count[..., None]).masked_fill(~mask[..., None], 0.)
    cov = torch.einsum('bni,bnj->bij', centered, centered) / count[..., None]
    std = cov.diagonal(dim1=1, dim2=2).clamp_min(0).sqrt()
    corr = (cov / (std[:, :, None]*std[:, None, :]).clamp_min(1e-12)).clamp(-1, 1)
    correlations = torch.stack([corr[:, i, j] for i, j in CORR_PAIRS], 1)
    return torch.cat((_moments(node, mask), _moments(geometry, mask), shape, grid,
                      *angles, _moments(radius_center[..., None], mask),
                      _moments(radius_depot[..., None], mask), correlations), 1)


def _matrix(matrix, mask):
    size = matrix.shape[1]
    valid = mask[:, :, None] & mask[:, None, :]
    valid = valid & ~torch.eye(size, dtype=torch.bool, device=matrix.device)[None]
    edges = matrix.float().masked_fill(~valid, 0.)
    if not torch.isfinite(edges).all():
        raise ValueError('Nonfinite valid directed edge')
    degree = valid.sum(-1).clamp_min(1)
    out_mean = edges.sum(-1)/degree
    in_mean = edges.sum(-2)/degree
    out_std = ((edges-out_mean[..., None]).square().masked_fill(~valid, 0.).sum(-1)/degree).sqrt()
    in_std = ((edges-in_mean[:, None]).square().masked_fill(~valid, 0.).sum(-2)/degree).sqrt()
    out_min = edges.masked_fill(~valid, float('inf')).amin(-1)
    in_min = edges.masked_fill(~valid, float('inf')).amin(-2)
    has_edge = valid.any(-1)
    out_min = out_min.masked_fill(~has_edge, 0.)
    in_min = in_min.masked_fill(~has_edge, 0.)
    asym = (edges-edges.transpose(1, 2)).abs()
    asym_mean = asym.sum(-1)/degree
    relative = (asym/(edges.abs()+edges.transpose(1, 2).abs()).clamp_min(1e-12)).sum(-1)/degree
    fields = [out_mean, out_min, out_std, in_mean, in_min, in_std, asym_mean, relative]
    for dimension in (-1, -2):
        ordered = edges.masked_fill(~valid, float('inf')).topk(min(8, size), dim=dimension, largest=False).values
        if dimension == -2:
            ordered = ordered.transpose(1, 2)
        for k in (1, 4, 8):
            neighbors = ordered[..., :min(k, size)]
            finite = torch.isfinite(neighbors)
            fields.append(neighbors.masked_fill(~finite, 0.).sum(-1)/finite.sum(-1).clamp_min(1))
    return _moments(torch.stack(fields, -1), mask)


@torch.no_grad()
def features(batch):
    """Return compact CPU FP32 rows plus immutable column names on CPU or CUDA."""
    source = input_batch(batch)
    mask = source['node_mask'].bool()
    if mask.ndim != 2 or mask.shape[0] == 0 or not mask.any(1).all():
        raise ValueError('Each feature row needs at least one real node')
    n = source['n'].reshape(-1).to(mask.device)
    if not torch.equal(mask.sum(1), n.long()):
        raise ValueError('True lengths disagree with the padding mask')
    kind = source['kind']
    if kind not in ('coord', 'matrix'):
        raise ValueError('Unknown input kind')
    b, device = len(n), mask.device
    cbits, desc = source['cbits'].to(device).float(), source['problem_desc'].to(device).float()
    if cbits.shape != (b, 5) or desc.shape != (b, 16):
        raise ValueError('Expected five constraint bits and sixteen descriptor fields')
    d_in = source['node'].shape[-1] if kind == 'coord' else 0
    common = torch.cat((n.float()[:, None], n.float().log1p()[:, None],
                        torch.full((b, 1), float(d_in), device=device),
                        torch.full((b, 1), float(kind == 'matrix'), device=device),
                        source['coord_dist'].to(device).float().reshape(b, 1), cbits, desc), 1)
    coordinate_width = len(COLUMNS)-common.shape[1]-len(MATRIX_FIELDS)*len(STATS)
    if kind == 'coord':
        geometry = source['node_geom']
        if geometry.shape != (*mask.shape, len(FIELDS)):
            raise ValueError('Use the existing twelve-channel cached geometry')
        coordinate = _coordinate(source['node'], geometry, mask)
        directed = torch.zeros(b, len(MATRIX_FIELDS)*len(STATS), device=device)
    else:
        coordinate = torch.zeros(b, coordinate_width, device=device)
        directed = _matrix(source['matrix'], mask)
    result = torch.cat((common, coordinate, directed), 1)
    if result.shape != (b, len(COLUMNS)) or not torch.isfinite(result).all():
        raise ValueError('Feature columns are nonfinite or inconsistent')
    return np.ascontiguousarray(result.cpu().numpy(), dtype=np.float32), COLUMNS
