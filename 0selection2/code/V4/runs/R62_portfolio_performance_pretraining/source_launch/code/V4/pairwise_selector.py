"""R43: score differences or joint pair queries, followed by maximin selection."""

import math

import torch
from torch import nn

from .dual_stream import CrossUpdate, DualStreamSelector


def maximin_logits(margins, mask):
    k = margins.size(-1)
    opponents = mask[:, None, :] & ~torch.eye(k, device=margins.device, dtype=torch.bool)[None]
    scores = margins.masked_fill(~opponents, float('inf')).min(-1).values
    scores = torch.where(mask.sum(-1, keepdim=True) == 1, torch.zeros_like(scores), scores)
    return scores.masked_fill(~mask, float('-inf'))


def rank_by_solver_ids(scores, ids):
    if ids.dim() == 1:
        ids = ids[None].expand_as(scores)
    order = ids.argsort(dim=-1, stable=True)
    ranked = scores.gather(-1, order).argsort(dim=-1, descending=True, stable=True)
    return order.gather(-1, ranked)


class ScoreDifferenceSelector(DualStreamSelector):
    def forward(self, batch):
        features, mask = self.encode_pairs(batch)
        utility = self.score_head(features).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        return dict(logits=maximin_logits(margins, mask), pair_margin=margins, solver_mask=mask)


class ExplicitPairSelector(DualStreamSelector):
    def __init__(self, **params):
        super().__init__(**params)
        # Construct the common backbone identically to A before replacing its decoder.
        del self.decoder, self.score_head
        d, dropout = params['embedding_dim'], params['dropout']
        self.pair_mlp = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.query_views = nn.Parameter(torch.randn(2, d) / math.sqrt(d))
        self.pair_layers = nn.ModuleList([CrossUpdate(**params) for _ in range(2)])
        self.delta_proj = nn.Linear(d, d, bias=False)
        self.pair_out = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, batch):
        nodes, solvers, h, node_mask, mask = self.encode_state(batch)
        b, k, d = solvers.shape
        i, j = torch.triu_indices(k, k, 1, device=nodes.device)
        margins = solvers.new_zeros((b, k, k), dtype=torch.float32)
        if i.numel():
            first, second = solvers[:, i], solvers[:, j]
            pair = self.pair_mlp(torch.cat([h[:, None].expand_as(first), first + second,
                                          (first - second).abs(), first * second], -1))
            pair_mask = mask[:, i] & mask[:, j]
            query = (pair[:, :, None] + self.query_views).reshape(b, -1, d)
            query_mask = pair_mask.repeat_interleave(2, -1)
            for layer in self.pair_layers:
                query = layer(query, nodes, query_mask, node_mask)
            evidence = query.reshape(b, i.numel(), 2 * d)
            direction = self.delta_proj(first - second).float()
            response = self.pair_out(torch.cat([pair, evidence], -1)).float()
            values = (direction * response).sum(-1) / math.sqrt(d)
            values = values.masked_fill(~pair_mask, 0.)
            margins[:, i, j], margins[:, j, i] = values, -values
        return dict(logits=maximin_logits(margins, mask), pair_margin=margins, solver_mask=mask)


def make_r43_model(params):
    cls = ScoreDifferenceSelector if params['pair_mode'] == 'score_difference' else ExplicitPairSelector
    return cls(**params)
