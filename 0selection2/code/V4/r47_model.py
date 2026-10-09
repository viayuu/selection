"""R47B: shared learned retrieval and solver-specific historical behavior values."""

import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from .V4Model import _masked_max, _masked_mean
from .behavior_memory import BehaviorBank, instance_summary
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits


class BehaviorRetrieval(nn.Module):
    def __init__(self, d, dropout=.1, neighbors=32):
        super().__init__()
        self.neighbors = neighbors
        self.key = nn.Sequential(nn.LayerNorm(2*d+1), nn.Linear(2*d+1, d))
        self.value = nn.Sequential(nn.Linear(d+2, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.correction = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d, d, bias=False))

    def retrieve(self, summary, base_ids, bank):
        # Index selection is discrete; recompute the selected reference keys with gradients.
        with torch.autocast(device_type=summary.device.type, enabled=False):
            query = self.key(summary.float())
            with torch.no_grad():
                keys = self.key(bank.summary)
                distances = (query.detach().square().sum(-1, keepdim=True) + keys.square().sum(-1)[None]
                             - 2*query.detach() @ keys.T).clamp_min(0)
                excluded = base_ids[:, None] == bank.base_ids[None]
                distances.masked_fill_(excluded, float('inf'))
                if (excluded.logical_not().sum(-1) < self.neighbors).any():
                    raise ValueError('Not enough nonrelated training references for the configured context')
                indices = distances.topk(self.neighbors, largest=False, sorted=True).indices
            selected = self.key(bank.summary[indices])
            weights = (-(query[:, None]-selected).square().sum(-1)).softmax(-1)
        return query, selected, indices, weights

    def forward(self, summary, solvers, ids, mask, base_ids, bank):
        query, selected, indices, weights = self.retrieve(summary, base_ids, bank)
        if ids.ndim == 1:
            ids = ids[None].expand(solvers.size(0), -1)
        mapping = ids[:, :, None] == bank.pool_ids[None, None]
        if not mapping.any(-1).logical_or(~mask).all():
            raise ValueError('Candidate identity is absent from this problem training memory')
        columns = mapping.long().argmax(-1)[:, None].expand(-1, self.neighbors, -1)
        valid = bank.solver_mask[indices].gather(-1, columns) & mask[:, None]
        gap = bank.gap_input[indices].gather(-1, columns)
        winner = (bank.winner_ids[indices, None] == ids[:, None]).float()
        solver = solvers[:, None].expand(-1, self.neighbors, -1, -1)
        values = self.value(torch.cat([solver, gap[..., None], winner[..., None]], -1))
        delta = (query[:, None]-selected)[:, :, None].expand_as(solver)
        values = values + self.correction(torch.cat([delta, solver], -1))
        per_solver_weights = weights[:, :, None] * valid
        per_solver_weights = per_solver_weights / per_solver_weights.sum(1, keepdim=True).clamp_min(1e-12)
        behavior = (per_solver_weights[..., None] * values).sum(1)
        return behavior.masked_fill(~mask[..., None], 0.), indices, weights


class BehaviorMemorySelector(ScoreDifferenceSelector):
    def __init__(self, **params):
        super().__init__(**params)
        d = params['embedding_dim']
        # Preserve common scratch initialization, including the old score-head block.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(params.get('retrieval_init_seed', 470002))
            self.retrieval = BehaviorRetrieval(d, params['dropout'], params.get('neighbors', 32))
            old = self.score_head[0]
            expanded = nn.Linear(4*d, old.out_features)
            with torch.no_grad():
                expanded.weight[:, :3*d].copy_(old.weight)
                expanded.bias.copy_(old.bias)
            self.score_head[0] = expanded
        self.memory = nn.ModuleDict()

    def encode_for_retrieval(self, batch):
        nodes, node_mask = self.instance_encoder(batch)
        summary = instance_summary(nodes, node_mask, batch['n'])
        ids = batch['pool_ids'].to(nodes.device)
        if ids.ndim == 1:
            ids = ids[None].expand(nodes.size(0), -1)
        mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device).bool()
        if mask.shape != ids.shape or not mask.any(-1).all():
            raise ValueError('Every query requires a nonempty legal solver pool')
        solvers = self.solver_encoder(ids.masked_fill(~mask, 0)).masked_fill(~mask[..., None], 0.)
        for layer in self.joint_layers:
            nodes, solvers = layer(nodes, solvers, node_mask, mask)
        h = self.pool(torch.cat([_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)], -1))
        context = self.decoder(h[:, None], solvers, solvers, key_mask=mask)
        features = torch.cat([h[:, None].expand_as(solvers), context.expand_as(solvers), solvers], -1)
        return features, summary, solvers, ids, mask

    def forward(self, batch):
        problem = PROBLEMS[int(batch['problem_id'])]
        if problem not in self.memory or 'base_id' not in batch:
            raise ValueError('R47 requires training memory and input-derived base_id for self exclusion')
        features, summary, solvers, ids, mask = self.encode_for_retrieval(batch)
        behavior, indices, weights = self.retrieval(summary, solvers, ids, mask, batch['base_id'], self.memory[problem])
        utility = self.score_head(torch.cat([features, behavior], -1)).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        return dict(logits=maximin_logits(margins, mask), pair_margin=margins, solver_mask=mask,
                    neighbor_indices=indices, neighbor_weights=weights, behavior=behavior)


def load_r47_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    params = dict(checkpoint['args']['model_params'], sdpa=str(device).startswith('cuda'))
    model = BehaviorMemorySelector(**params)
    state = checkpoint['model']
    for problem in checkpoint['args']['sizes']:
        get = lambda name: state[f'memory.{problem}.{name}']
        model.memory[problem] = BehaviorBank(get('summary'), get('gap'), get('winner_ids'), get('pool_ids'),
                                             get('nodes'), get('base_ids'), get('solver_mask'))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint
