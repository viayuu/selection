"""R124: frozen full-pool Top2 plus an independent pair-conditioned encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .V4Model import InstanceEncoder, MultiHeadAttention, _masked_mean, _masked_max
from .dual_stream import JointEncoderLayer, SolverFeatureEncoder
from .r123_meta_data import INPUT_FIELDS


QUERY_FIELDS = INPUT_FIELDS - {'mask'}


def query_inputs(batch):
    """An allowlist, not a blacklist of currently known supervision fields."""
    return {key: value for key, value in batch.items() if key in QUERY_FIELDS}


def candidate_ids(batch, scores):
    ids = batch['pool_ids'].to(device=scores.device, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids[None].expand(len(scores), -1)
    mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool))
    mask = mask.to(device=scores.device, dtype=torch.bool)
    if scores.ndim != 2 or ids.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError('Scores, method IDs and legal mask must align')
    if (mask.sum(1) < 2).any() or not torch.isfinite(scores[mask]).all():
        raise ValueError('Top2 requires at least two finite legal candidates')
    ordered_ids = ids.masked_fill(~mask, torch.iinfo(torch.long).max).sort(1).values
    if ((ordered_ids[:, 1:] == ordered_ids[:, :-1])
            & (ordered_ids[:, 1:] != torch.iinfo(torch.long).max)).any():
        raise ValueError('Legal method IDs must be distinct')
    return ids, mask


def top2_indices(scores, ids, mask):
    """Stable score rank, with global method ID as the exact-tie secondary key."""
    id_order = ids.masked_fill(~mask, torch.iinfo(torch.long).max).argsort(dim=1, stable=True)
    values = scores.masked_fill(~mask, -torch.inf).gather(1, id_order)
    return id_order.gather(1, values.argsort(dim=1, descending=True, stable=True)[:, :2])


def replace_top2(scores, pair, difference, ids, mask):
    """Lift both pair members above rank3 without disturbing any other score.

    nextafter guards avoid rounded sigmoid ties, including after the FP32 cast
    used by existing LIB consumers. This transport is NOT the loss head.
    """
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    result = scores.detach().to(dtype=dtype).masked_fill(~mask, -torch.inf).clone()
    other = mask.clone().scatter(1, pair, False)
    maximum = result.masked_fill(~other, -torch.inf).amax(1)
    maximum = torch.where(other.any(1), maximum, result.gather(1, pair).amax(1))
    upward = torch.full_like(maximum, torch.inf)
    base = torch.maximum(maximum + 1., torch.nextafter(maximum, upward))
    d = difference.detach().to(dtype=dtype)
    values = base[:, None] + torch.stack((d.sigmoid(), (-d).sigmoid()), 1)
    values = torch.maximum(values, torch.nextafter(maximum, upward)[:, None])
    collapsed = (values[:, 0] == values[:, 1]) & (d != 0)
    higher = torch.nextafter(values[:, 0], upward)
    values[:, 0] = torch.where(collapsed & (d > 0), higher, values[:, 0])
    values[:, 1] = torch.where(collapsed & (d < 0), higher, values[:, 1])
    if not torch.isfinite(values).all():
        raise ValueError('First-stage score magnitude cannot be safely reranked')
    return result.scatter(1, pair, values)


class Top2Reranker(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.params = dict(params)
        d = params['embedding_dim']
        self.instance_encoder = InstanceEncoder(**dict(params, node_only=True))
        self.solver_encoder = SolverFeatureEncoder(**params)
        self.stage_score = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.joint_layers = nn.ModuleList([JointEncoderLayer(**params) for _ in range(2)])
        self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.pair_readout = MultiHeadAttention(d, params['head_num'], params['qkv_dim'],
                                               params['dropout'], params.get('sdpa', False))
        self.score_head = nn.Sequential(nn.Linear(3 * d, params['head_hidden_dim']), nn.GELU(),
            nn.Dropout(params['dropout']), nn.Linear(params['head_hidden_dim'], 1))
        self.quality_head = nn.Sequential(nn.Linear(3 * d, params['head_hidden_dim']), nn.GELU(),
            nn.Dropout(params['dropout']), nn.Linear(params['head_hidden_dim'], 1))

    def compare(self, batch, stage1_scores, pair=None):
        inputs = query_inputs(batch)
        scores = stage1_scores.detach()
        ids, legal = candidate_ids(inputs, scores)
        actual = top2_indices(scores, ids, legal)
        pair = actual if pair is None else pair.to(device=scores.device, dtype=torch.long)
        if pair.shape != actual.shape or not torch.equal(pair.sort(1).values, actual.sort(1).values):
            raise ValueError('Comparator must receive the actual first-stage Top2, in either order')
        # Canonical proposal order also makes antisymmetry exact in floating point.
        orientation = torch.where(pair[:, 0] == actual[:, 0], 1., -1.)
        nodes, node_mask = self.instance_encoder(inputs)
        methods = self.solver_encoder(ids.gather(1, actual))
        masked = scores.double().masked_fill(~legal, 0.)
        count = legal.sum(1, keepdim=True)
        mean = masked.sum(1, keepdim=True) / count
        variance = ((masked - mean).square().masked_fill(~legal, 0.)).sum(1, keepdim=True) / count
        z = ((scores.double().gather(1, actual) - mean) / variance.sqrt().clamp_min(1e-6)).float()
        # Own score and opponent score swap together with each method token.
        methods = methods + self.stage_score(torch.stack((z, z.flip(1)), -1).to(methods.dtype))
        pair_mask = torch.ones_like(pair, dtype=torch.bool)
        for layer in self.joint_layers:
            nodes, methods = layer(nodes, methods, node_mask, pair_mask)
        pooled = self.pool(torch.cat((_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)), -1))
        structure = self.pair_readout(methods, nodes, nodes, key_mask=node_mask)
        features = torch.cat((methods, structure, pooled[:, None].expand_as(methods)), -1)
        utilities = self.score_head(features).squeeze(-1).float()
        difference = (utilities[:, 0] - utilities[:, 1]) * orientation
        quality = self.quality_head(features).squeeze(-1).float()
        return dict(difference=difference, pair_logits=torch.stack((difference / 2., -difference / 2.), 1),
                    pair_quality=.01 * (quality[:, 0] - quality[:, 1]) * orientation,
                    pair_indices=pair, pair_ids=ids.gather(1, pair), solver_mask=legal)

    def forward(self, batch, stage1_scores):
        output = self.compare(query_inputs(batch), stage1_scores)
        ids, legal = candidate_ids(batch, stage1_scores)
        output['logits'] = replace_top2(stage1_scores, output['pair_indices'], output['difference'], ids, legal)
        return output


class FrozenTop2Selector(nn.Module):
    """Direct inference factory: compute frozen f0(X), then rerank its actual pair."""
    def __init__(self, reranker, first_stage):
        super().__init__()
        self.reranker = reranker
        self.first_stage = first_stage.eval().requires_grad_(False)
        self.params = reranker.params

    @property
    def instance_encoder(self):
        return self.reranker.instance_encoder

    def state_dict(self, *args, **kwargs):
        """New-learner checkpoint scope; the factory hash-verifies external f0."""
        return self.reranker.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self.reranker.load_state_dict(state_dict, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.first_stage.eval()
        return self

    def forward(self, batch):
        inputs = query_inputs(batch)
        device = next(self.first_stage.parameters()).device
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            scores = self.first_stage(inputs)['logits'].float()
        return dict(self.reranker(inputs, scores), stage1_scores=scores)


def pair_objective(output, costs, winner, *, row_denominator=None):
    """Supervision stays outside the comparator, with FP64 native cost semantics."""
    pair, logits = output['pair_indices'], output['pair_logits']
    if costs.dtype != torch.float64 or winner.dtype != torch.long:
        raise ValueError('Loss requires unchanged native FP64 costs and int64 winners')
    legal = output['solver_mask']
    if (costs.shape != legal.shape or not torch.isfinite(costs[legal]).all()
            or (costs[legal] <= 0).any() or not legal.gather(1, winner[:, None]).all()):
        raise ValueError('Invalid native cost/winner supervision')
    optimum = costs.masked_fill(~legal, torch.inf).amin(1)
    if not torch.equal(costs.gather(1, winner[:, None]).squeeze(1), optimum):
        raise ValueError('Native winner must be an exact legal optimum; ties are not relabeled')
    matches = pair == winner[:, None]
    covered = matches.any(1)
    ce_sum = (F.cross_entropy(logits[covered], matches[covered].long().argmax(1), reduction='sum')
              if covered.any() else logits.sum() * 0.)
    selected_costs = costs.gather(1, pair)
    target = (selected_costs[:, 1] - selected_costs[:, 0]) / optimum
    quality_sum = F.smooth_l1_loss(output['pair_quality'].double() / .01,
                                   target / .01, beta=1., reduction='sum')
    nrow = len(costs) if row_denominator is None else int(row_denominator)
    if nrow <= 0:
        raise ValueError('Logical row denominator must be positive')
    ce, quality = ce_sum / nrow, quality_sum / nrow
    loss = ce + quality
    return loss, dict(native_ce=ce.detach(), pair_quality_huber=quality.detach(),
                      covered=covered.sum().detach(), loss=loss.detach())
