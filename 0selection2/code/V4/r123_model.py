"""Input-only reliability gate and sparse routing to frozen complete selectors."""

from __future__ import annotations

import torch
from torch import nn

from .V4Model import InstanceEncoder, MultiHeadAttention
from .r123_meta_data import DEFAULT_EXPERTS, model_inputs


GATE_FIELDS = frozenset(('problem_id', 'kind', 'n', 'node', 'matrix', 'node_mask',
    'cbits', 'problem_desc', 'coord_dist', 'node_geom', 'd_in'))


def gate_inputs(batch):
    return {key: value for key, value in batch.items() if key in GATE_FIELDS}


def input_rows(batch, rows):
    """Slice only query inputs; global one-dimensional pool IDs are not rows."""
    clean = model_inputs(batch)
    return {key: (value[rows] if torch.is_tensor(value) and value.ndim
                         and not (key == 'pool_ids' and value.ndim == 1) else value)
            for key, value in clean.items()}


class ReliabilityGate(nn.Module):
    def __init__(self, model_params, experts=DEFAULT_EXPERTS):
        super().__init__()
        self.params = dict(model_params, node_only=True)
        self.expert_ids = tuple(experts)
        if not self.expert_ids or len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError('Require distinct fixed expert identities')
        d = self.params['embedding_dim']
        self.instance_encoder = InstanceEncoder(**self.params)
        self.expert_queries = nn.Parameter(torch.randn(len(experts), d) * .02)
        self.read = MultiHeadAttention(d, self.params['head_num'], self.params['qkv_dim'],
                                       self.params['dropout'], self.params.get('sdpa', False))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                 nn.Dropout(self.params['dropout']), nn.Linear(d, 1))

    def forward(self, batch):
        nodes, mask = self.instance_encoder(gate_inputs(batch))
        query = self.expert_queries[None].expand(nodes.size(0), -1, -1)
        evidence = self.read(query, nodes, nodes, key_mask=mask)
        logits = self.head(torch.cat((query, evidence), -1)).squeeze(-1).float()
        return dict(reliability_logits=logits, reliability=logits.sigmoid())


def correctness_targets(expert_logits, native_winner, pool_ids):
    """Native multi-label targets, including multiple ones and all-zero rows."""
    from .pairwise_selector import rank_by_solver_ids
    if expert_logits.ndim != 3 or native_winner.shape != expert_logits.shape[:1]:
        raise ValueError('Require bank scores [B,E,K] and saved native winners [B]')
    if native_winner.dtype != torch.long or not torch.isfinite(expert_logits).all():
        raise ValueError('Finite complete bank scores and native int64 winners required')
    b, e, k = expert_logits.shape
    if pool_ids.ndim != 1 or pool_ids.numel() != k or torch.unique(pool_ids).numel() != k:
        raise ValueError('Require distinct fixed global candidate IDs')
    if ((native_winner < 0) | (native_winner >= k)).any():
        raise ValueError('Native winner outside the legal pool')
    actions = rank_by_solver_ids(expert_logits.detach().reshape(b * e, k), pool_ids)[:, 0].reshape(b, e)
    return (actions == native_winner[:, None]).float(), actions


def brier_loss(reliability_logits, targets):
    if reliability_logits.shape != targets.shape or targets.requires_grad:
        raise ValueError('Detached complete expert-correctness targets required')
    if not torch.isfinite(reliability_logits).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('Require finite logits and exact binary multi-label supervision')
    return (reliability_logits.float().sigmoid() - targets.float()).square().mean()


class HistoricalModelRouter(nn.Module):
    """Compute gate first, then execute one frozen expert per query, never mix."""

    def __init__(self, gate, experts, candidate_pool_ids=None):
        super().__init__()
        if len(experts) != len(gate.expert_ids):
            raise ValueError('Frozen bank must match gate expert identities and order')
        self.gate = gate
        self.candidate_pool_ids = candidate_pool_ids
        self.experts = nn.ModuleList(experts)
        self.experts.requires_grad_(False).eval()

    @property
    def instance_encoder(self):
        return self.gate.instance_encoder

    def train(self, mode=True):
        super().train(mode)
        self.experts.eval()
        return self

    def forward(self, batch, *, expert_logits=None):
        if self.candidate_pool_ids is not None:
            from code.unified_selector.registry import PROBLEMS
            expected = self.candidate_pool_ids[PROBLEMS[int(batch['problem_id'])]]
            ids = batch['pool_ids']
            values = ids[None] if ids.ndim == 1 else ids
            required = torch.tensor(sorted(expected), device=ids.device, dtype=ids.dtype)
            if values.shape[-1] != len(expected) or not torch.equal(values.sort(-1).values, required[None].expand_as(values)):
                raise ValueError('Router requires the unchanged complete current legal candidate pool')
        output = self.gate(gate_inputs(batch))
        chosen = output['reliability_logits'].argmax(-1)  # fixed bank-order ties
        n, k = len(chosen), batch['pool_ids'].shape[-1]
        clean = model_inputs(batch)
        mask = clean.get('solver_mask')
        if mask is None:
            mask = torch.ones(n, k, dtype=torch.bool, device=chosen.device)
            clean['solver_mask'] = mask
        if expert_logits is not None:
            if expert_logits.shape != (n, len(self.experts), k) or not torch.isfinite(expert_logits).all():
                raise ValueError('Exact complete cached bank score tensor required')
            scores = expert_logits.to(chosen.device)[torch.arange(n, device=chosen.device), chosen]
        else:
            scores = output['reliability_logits'].new_empty(n, k)
            for expert, model in enumerate(self.experts):
                rows = (chosen == expert).nonzero(as_tuple=True)[0]
                if rows.numel():
                    # Frozen expert inference has the same explicit FP32 policy
                    # as cache generation, even inside an outer gate AMP scope.
                    with torch.no_grad(), torch.autocast(device_type=chosen.device.type, enabled=False):
                        scores[rows] = model(input_rows(clean, rows))['logits'].float()
        return dict(output, logits=scores, chosen_expert=chosen,
                    solver_mask=mask)
