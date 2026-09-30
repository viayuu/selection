"""R40 control: shared instance encoder followed by a global solver classifier."""

import torch
from torch import nn

from code.unified_selector.registry import M_GLOBAL
from .V4Model import InstanceEncoder, _masked_mean, _masked_max, make_selector


class DirectSelector(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        d = params['embedding_dim']
        self.instance_encoder = InstanceEncoder(**dict(params, node_only=True))
        self.classifier = nn.Sequential(
            nn.Linear(3 * d + 1, 512), nn.GELU(), nn.Dropout(params['dropout']),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(params['dropout']),
            nn.Linear(256, M_GLOBAL),
        )

    def forward(self, batch):
        nodes, mask = self.instance_encoder(batch)
        encoder = self.instance_encoder
        if batch['kind'] == 'coord':
            stats = encoder._coord_stats(batch['node'], mask)
        else:
            stats = encoder._matrix_stats(batch['matrix'], mask)
        condition, _ = encoder.condition_encoder(batch, stats, int(batch['kind'] != 'coord'))
        size = batch['n'].float().to(nodes.device).log1p().unsqueeze(-1) / 6.
        features = torch.cat([_masked_mean(nodes, mask), _masked_max(nodes, mask), condition, size], dim=-1)
        global_logits = self.classifier(features)
        ids = batch['pool_ids'].to(nodes.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0).expand(nodes.size(0), -1)
        solver_mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device)
        if solver_mask.shape != ids.shape or not solver_mask.any(1).all():
            raise ValueError('Every instance needs a nonempty candidate mask matching pool_ids')
        logits = global_logits.gather(1, ids.masked_fill(~solver_mask, 0))
        return {'logits': logits.masked_fill(~solver_mask, float('-inf'))}


def make_r40_model(params):
    return DirectSelector(**params) if params['architecture'] == 'direct' else make_selector(params)
