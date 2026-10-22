"""Unchanged R77 supervision; share deterministic inputs across dropout passes."""

import torch

from .pairwise_objective import predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective as OriginalObjective
from .r99_model import inputs_with_relations


class CompetitionObjective(OriginalObjective):
    def forward(self, batch, consistency_weight):
        if batch is None:
            return super().forward(batch, consistency_weight)
        prepared = self.selector.prepare_inputs(inputs_with_relations(batch))
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first, second = self.selector(prepared), self.selector(prepared)
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5*(loss1+loss2) + consistency_weight*kl
        parts = {key: .5*(parts1[key]+parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])
