"""One raw-query selector and a shared centered advantage potential for D1."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_selector import ScoreDifferenceSelector, maximin_logits

ADVANTAGE_UNIT = .01
LOSS_WEIGHTS = dict(natural_ce=1., endpoint_huber=1., response_huber=1.)
FORWARD_KEYS = frozenset(('problem_id', 'kind', 'node', 'matrix', 'node_mask', 'd_in',
    'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'solver_mask', 'mask', 'n', 'node_geom'))


def query_inputs(batch):
    return {key: value for key, value in batch.items() if key in FORWARD_KEYS}


def center(values, mask):
    if values.ndim != 2 or mask.shape != values.shape or not mask.any(-1).all():
        raise ValueError('Every query needs a nonempty legal candidate pool')
    clean = values.masked_fill(~mask, 0.)
    mean = clean.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
    return (clean - mean).masked_fill(~mask, 0.)


class StructuralFamilySelector(ScoreDifferenceSelector):
    def __init__(self, **params):
        if params.get('pair_mode') != 'score_difference' or params.get('solver_representation', 'handcrafted') != 'handcrafted':
            raise ValueError('R121 retains the original handcrafted ScoreDifferenceSelector backbone')
        super().__init__(**params)
        self.advantage_head = nn.Sequential(nn.Linear(3 * params['embedding_dim'], params['head_hidden_dim']),
            nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['head_hidden_dim'], 1))

    def forward(self, batch):
        features, mask = self.encode_pairs(query_inputs(batch))
        utility = self.score_head(features).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        units = center(self.advantage_head(features).squeeze(-1).float(), mask)
        return dict(logits=maximin_logits(margins, mask), utility=utility,
            pair_margin=margins, solver_mask=mask, advantage_units=units,
            centered_log_advantage=ADVANTAGE_UNIT * units)


def advantage_targets(costs, mask):
    if costs.dtype != torch.float64 or costs.requires_grad or costs.shape != mask.shape:
        raise ValueError('Detached original FP64 costs and aligned masks required')
    if not mask.any(-1).all() or not torch.isfinite(costs[mask]).all() or (costs[mask] <= 0).any():
        raise ValueError('Complete positive native costs required; no epsilon filling')
    return center(torch.log(costs.masked_fill(~mask, 1.)), mask) / ADVANTAGE_UNIT


def natural_loss(output, batch):
    winner, costs, mask = batch['ind'], batch['costs'], output['solver_mask']
    if (winner.dtype != torch.long or costs.dtype != torch.float64 or costs.requires_grad
            or winner.shape != costs.shape[:1] or not torch.isfinite(costs[mask]).all()
            or not mask.gather(1, winner[:, None]).all()
            or not torch.equal(costs.gather(1, winner[:, None])[:, 0], costs.masked_fill(~mask, torch.inf).min(-1).values)):
        raise ValueError('Natural CE uses retained native winners, not relabeled FP32 argmin')
    return F.cross_entropy(output['logits'].float(), winner, reduction='none')


def family_losses(output, batch):
    prediction, mask = output['advantage_units'], output['solver_mask']
    if len(prediction) % 2:
        raise ValueError('Family microbatches must retain complete adjacent original/changed pairs')
    if not torch.equal(mask[::2], mask[1::2]):
        raise ValueError('A family edge cannot cross candidate pools')
    target = advantage_targets(batch['costs'], mask).to(prediction.dtype)
    endpoint = F.huber_loss(prediction, target, reduction='none', delta=1.)
    endpoint = endpoint.masked_fill(~mask, 0.).sum(-1) / mask.sum(-1)
    difference = prediction[1::2] - prediction[::2]
    truth = target[1::2] - target[::2]
    response = F.huber_loss(difference, truth, reduction='none', delta=1.)
    response = response.masked_fill(~mask[::2], 0.).sum(-1) / mask[::2].sum(-1)
    return dict(endpoint_huber=endpoint, response_huber=response)


def make_model(params, state_dict=None):
    model = StructuralFamilySelector(**params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def initialize_shared(initial, geometry):
    from .r77_data import parameter_hash
    if initial.get('initialization') != 'scratch' or initial.get('seed') != 2:
        raise ValueError('Only the original untrained seed2 common initialization is allowed')
    reference = initial['trainable']
    if parameter_hash(reference) != initial['trainable_sha256']:
        raise ValueError('Common fresh initialization hash changed')
    torch.manual_seed(2)
    params = dict(initial['model_params'])
    if not params.get('ignore_coord_dist'):
        raise ValueError('Keep raw-query-only original distribution-independent conditioning')
    model = make_model(params)
    actual = dict(model.named_parameters())
    common = {key: value for key, value in actual.items() if not key.startswith('advantage_head.')}
    if set(common) != set(reference) or any(common[k].shape != reference[k].shape for k in common):
        raise ValueError('Common raw-query selector parameter identities differ')
    with torch.no_grad():
        for key, value in common.items():
            value.copy_(reference[key])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    receipt = dict(initialization='scratch_shared_r84_plus_fresh_advantage_head', seed=2,
        historical_trained_checkpoint=False, common_trainable_sha256=parameter_hash(common),
        new_parameter_names=[k for k in actual if k.startswith('advantage_head.')],
        all_trainable_sha256=parameter_hash(actual), geometry_statistics='original full81000 TRAIN only')
    return model, params, receipt
