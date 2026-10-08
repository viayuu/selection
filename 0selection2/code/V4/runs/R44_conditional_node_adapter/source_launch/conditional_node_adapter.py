"""R44: one metadata-conditioned bottleneck before node/solver interaction."""

import math

import torch
from torch import nn

from code.unified_selector.registry import PROBLEM_DESCRIPTOR_FIELDS
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits


CONDITION_FIELDS = (*PROBLEM_DESCRIPTOR_FIELDS, 'train_normalized_log_real_nodes')
MODES = {'C0': 'none', 'C1': 'shared', 'C2': 'conditional'}


def adapter_dimensions(d):
    rank = 32 if d >= 128 else min(d, max(8, d // 4))
    conditional = 2 * d * rank + len(CONDITION_FIELDS) * rank + rank * rank + 2 * rank
    shared_rank = math.ceil(conditional / (2 * d))
    return rank, shared_rank, conditional, 2 * d * shared_rank


class NodeResidual(nn.Module):
    def __init__(self, d, mode, log_n_mean=0., log_n_std=1.):
        super().__init__()
        if mode not in ('shared', 'conditional'):
            raise ValueError('NodeResidual requires shared or conditional mode')
        self.mode = mode
        conditional_rank, shared_rank, _, _ = adapter_dimensions(d)
        self.rank = conditional_rank if mode == 'conditional' else shared_rank
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.down = nn.Linear(d, self.rank, bias=False)
        self.up = nn.Linear(self.rank, d, bias=False)
        self.activation = nn.GELU()
        nn.init.zeros_(self.up.weight)
        if mode == 'conditional':
            self.gate_in = nn.Linear(len(CONDITION_FIELDS), self.rank)
            self.gate_out = nn.Linear(self.rank, self.rank)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)
            self.register_buffer('log_n_mean', torch.tensor(log_n_mean, dtype=torch.float32))
            self.register_buffer('log_n_std', torch.tensor(max(log_n_std, 1e-6), dtype=torch.float32))

    def condition(self, batch, node_mask):
        # Read only existing semantic metadata and real nodes (including a depot).
        n = node_mask.sum(-1).float().clamp_min(1)
        size = (n.log() - self.log_n_mean) / self.log_n_std
        return torch.cat([batch['problem_desc'].float(), size[:, None]], -1)

    def gate(self, batch, node_mask):
        q = self.condition(batch, node_mask)
        return 2. * self.gate_out(self.activation(self.gate_in(q))).sigmoid()

    def forward(self, nodes, node_mask, batch):
        value = self.activation(self.down(self.norm(nodes)))
        if self.mode == 'conditional':
            value = value * self.gate(batch, node_mask)[:, None, :]
        delta = self.up(value).masked_fill(~node_mask[:, :, None], 0.)
        return nodes + delta


class ConditionalNodeSelector(ScoreDifferenceSelector):
    def __init__(self, **params):
        super().__init__(**params)
        if params.get('pair_mode') != 'score_difference':
            raise ValueError('R44 continues R43A, not the explicit pair-query model')
        mode = params.get('node_adapter_mode', 'none')
        if mode not in MODES.values():
            raise ValueError('node_adapter_mode must be none, shared or conditional')
        self.node_adapter = None if mode == 'none' else NodeResidual(
            params['embedding_dim'], mode, params.get('log_n_mean', 0.), params.get('log_n_std', 1.))

    def adapt_nodes(self, nodes, node_mask, batch):
        if self.node_adapter is None:
            return nodes
        return self.node_adapter(nodes, node_mask, batch)

    def forward(self, batch, return_utility=False):
        features, mask = self.encode_pairs(batch)
        utility = self.score_head(features).squeeze(-1).float()
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        output = dict(logits=maximin_logits(margins, mask), pair_margin=margins, solver_mask=mask)
        if return_utility:
            output['utility'] = utility
        return output


def load_baseline(model, state):
    result = model.load_state_dict(state, strict=False)
    expected = {key for key in model.state_dict() if key.startswith('node_adapter.')}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError(f'Unexpected baseline mismatch: {result}')
