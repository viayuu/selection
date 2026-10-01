"""R42B: each solver reads node-level evidence through four independent views."""

import torch
from torch import nn

from .dual_stream import CrossUpdate, DualStreamSelector, SolverFeatureEncoder
from .relation_encoder import RelationEncoder
from .V4Model import _masked_mean, _masked_max


class SolverQuerySelector(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        d = params['embedding_dim']
        self.instance_encoder = RelationEncoder(**params)
        self.solver_encoder = SolverFeatureEncoder(**params)
        self.view = nn.Embedding(4, d)
        self.condition_proj = nn.Linear(d, d)
        self.query_layers = nn.ModuleList([CrossUpdate(**params) for _ in range(2)])
        self.summary = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.score_head = nn.Sequential(nn.Linear(6 * d, params['head_hidden_dim']), nn.GELU(),
                                        nn.Dropout(params['dropout']), nn.Linear(params['head_hidden_dim'], 1))

    def forward(self, batch):
        memory, memory_mask, nodes, node_mask, condition = self.instance_encoder(batch)
        ids = batch['pool_ids'].to(nodes.device)
        if ids.dim() == 1:
            ids = ids[None].expand(nodes.size(0), -1)
        mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device).bool()
        if mask.shape != ids.shape or not mask.any(1).all():
            raise ValueError('Every instance needs a nonempty valid solver pool')
        solver = self.solver_encoder(ids.masked_fill(~mask, 0)).masked_fill(~mask[..., None], 0)
        b, k, d = solver.shape
        query = solver[:, :, None] + self.view.weight[None, None] + self.condition_proj(condition)[:, None, None]
        query = query.reshape(b, 4 * k, d)
        query_mask = mask[:, :, None].expand(-1, -1, 4).reshape(b, 4 * k)
        for layer in self.query_layers:
            query = layer(query, memory, query_mask, memory_mask)
        summary = self.summary(torch.cat((_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)), -1))
        features = torch.cat((query.reshape(b, k, 4 * d), solver, summary[:, None].expand(-1, k, -1)), -1)
        return {'logits': self.score_head(features).squeeze(-1).masked_fill(~mask, float('-inf'))}


def make_r42_model(params):
    return SolverQuerySelector(**params) if params['architecture'] == 'relation_query' else DualStreamSelector(**params)
