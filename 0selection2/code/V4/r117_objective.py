"""Unchanged R77 row-wise losses; only explicitly safe native inputs enter forward."""

import torch

from .pairwise_objective import predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import LOSS_RECIPE, PART_KEYS, RDropObjective as OriginalObjective


DONOR_FIELDS = ('nco_nodes', 'nco_node_mask', 'native_nodes', 'native_node_mask')
INPUT_KEYS = frozenset(('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind',
    'problem_id', 'problem_name', 'coord_dist', 'cbits', 'problem_desc', 'd_in',
    'pool_ids', 'solver_mask', 'mask', *DONOR_FIELDS))


def inputs_only(batch):
    return {key: value for key, value in batch.items() if key in INPUT_KEYS}


def raw_inputs(batch):
    return {key: value for key, value in inputs_only(batch).items() if key not in DONOR_FIELDS}


class NativeObjective(OriginalObjective):
    """Share one frozen preparation between the two stochastic selector forwards."""

    def forward(self, batch, consistency_weight):
        if batch is None:
            return super().forward(batch, consistency_weight)
        device = batch['ind'].device
        prepared = self.selector.prepare_inputs(inputs_only(batch))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first, second = self.selector(prepared), self.selector(prepared)
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])


RDropObjective = NativeObjective
