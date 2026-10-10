"""R53: an independent, solver-free MOEL/MTL classifier and fixed Top2 gate."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .V4Model import InstanceEncoder, _masked_max, _masked_mean


PAIR = ('RELD_MOEL', 'RELD_MTL')
INPUT_FIELDS = ('kind', 'problem_id', 'node', 'node_mask', 'matrix', 'n', 'cbits',
                'problem_desc', 'coord_dist', 'node_geom')


class PairSpecialist(nn.Module):
    """Positive logit means MOEL; neither solver identity nor labels enter forward."""

    def __init__(self, params):
        super().__init__()
        self.params = dict(params, node_only=True)
        self.instance_encoder = InstanceEncoder(**self.params)
        d = self.params['embedding_dim']
        self.head = nn.Sequential(nn.Linear(2*d+1, d), nn.GELU(),
                                  nn.Dropout(self.params['dropout']), nn.Linear(d, 1))

    def forward(self, batch):
        clean = {k: batch[k] for k in INPUT_FIELDS if k in batch}
        nodes, mask = self.instance_encoder(clean)
        summary = torch.cat((_masked_mean(nodes, mask), _masked_max(nodes, mask),
                             clean['n'].float().log1p()[:, None]/6.), dim=-1)
        return self.head(summary).squeeze(-1)


def pair_targets(costs, pool):
    costs = np.asarray(costs)
    if costs.dtype != np.float64 or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Pair targets require original, finite positive FP64 costs')
    a, b = [list(pool).index(name) for name in PAIR]
    delta = costs[:, a] - costs[:, b]
    label = np.where(delta == 0, -1, (delta < 0).astype(np.int64))
    weight = np.abs(delta) / costs.min(axis=1)
    return label, weight


def binary_objective(logit, labels, weights, weighted, training_mean):
    if (labels < 0).any():
        raise ValueError('Exact ties must not enter binary supervision')
    values = F.binary_cross_entropy_with_logits(logit.float(), labels.float(), reduction='none')
    if weighted:
        if not np.isfinite(training_mean) or training_mean <= 0:
            raise ValueError('The fixed training-only weight mean must be positive')
        values = values * weights.float() / training_mean
    return values.mean()


def stable_order(logits, pool_ids):
    by_id = np.argsort(pool_ids, kind='stable')
    return by_id[np.argsort(-np.asarray(logits)[:, by_id], axis=1, kind='stable')]


def gate_mask(order, pool):
    if not all(name in pool for name in PAIR) or order.shape[1] < 2:
        return np.zeros(len(order), dtype=bool)
    a, b = [list(pool).index(name) for name in PAIR]
    return ((order[:, 0] == a) & (order[:, 1] == b)) | ((order[:, 0] == b) & (order[:, 1] == a))


def apply_specialist(order, pool, expert_logits):
    """Only reorder the original two candidates, including tied baseline scores."""
    result = np.asarray(order).copy()
    gate = gate_mask(result, pool)
    if not gate.any():
        return result, gate
    z = np.asarray(expert_logits)
    if z.shape != (len(order),) or not np.isfinite(z).all():
        raise ValueError('Every specialist output must be finite and row-aligned')
    a, b = [list(pool).index(name) for name in PAIR]
    # MOEL has the lower global solver ID, so a zero margin selects MOEL.
    chosen = np.where(z >= 0, a, b)
    changed = gate & (chosen != result[:, 0])
    result[changed, :2] = result[changed, :2][:, ::-1]
    return result, gate


def ordinal_scores(order):
    """An exact ranking representation, not calibrated classification logits."""
    return -np.argsort(order, axis=1).astype(np.float64)


def specialist_metrics(z, raw, training_mean):
    labels, weights = pair_targets(raw['costs'], raw['pool'])
    valid = labels >= 0
    z = np.asarray(z, dtype=np.float64)
    if z.shape != labels.shape or not np.isfinite(z).all():
        raise ValueError('Invalid binary predictions')
    ce = np.logaddexp(0, z) - np.maximum(labels, 0) * z
    a, b = [raw['pool'].index(name) for name in PAIR]
    chosen = np.where(z >= 0, a, b)
    selected = raw['costs'][np.arange(len(z)), chosen]
    pair_min = raw['costs'][:, [a, b]].min(1)
    wrong = ((z >= 0).astype(np.int64) != labels)
    return dict(n=int(valid.sum()), ties=int((~valid).sum()),
        accuracy=float((~wrong[valid]).mean()), ce=float(ce[valid].mean()),
        weighted_ce=float((ce[valid]*weights[valid]/training_mean).mean()),
        pair_regret_pct=float(((selected[valid]-pair_min[valid])/pair_min[valid]*100).mean()),
        extra_full_pool_regret_pct=float((weights[valid]*wrong[valid]*100).mean()),
        moel_prediction_pct=float((z[valid] >= 0).mean()*100),
        moel_win_pct=float((labels[valid] == 1).mean()*100))
