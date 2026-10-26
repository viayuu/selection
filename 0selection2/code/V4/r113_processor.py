"""Reusable typed R113 processor; deploy with self-initialized prediction feedback.

This module has no runner, candidate readout, checkpoint loading or native solver.
It consumes only ``compile_inputs`` output. Teacher states are allowed solely by
an explicit training-mode call to ``rollout(..., teacher_forcing=True)``.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .r113_process_data import (
    BINARY_FIELDS, GRAPH_FIELDS, INPUT_KEYS, NODE_FIELDS, PROBLEMS,
    SCALAR_FIELDS, TASK_FIELDS, initial_state, load_shard,
)


RESOURCE_INDICES = (0, 1, 2, 3, 4, 5, 6, 15)
DISTANCE_INDICES = (7, 8)
TIME_INDICES = (9, 10, 11, 12, 13)
LIMIT_INDICES = (14,)
STATE_KEYS = ('scalar', 'binary', 'pointer', 'reached', 'frontier', 'compatible')


def _mlp(in_dim, out_dim, hidden):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))


def soft_directed_propagation(reached, compatible, topology):
    """Differentiable OR; exact Boolean propagation on 0/1 predictions.

    This executes on predicted compatibility and predicted previous reachability,
    never the teacher graph. Source axis precedes target axis throughout.
    """
    messages = reached[..., :, None] * compatible * topology[:, None].to(compatible.dtype)
    incoming = 1. - (1. - messages).prod(dim=-2)
    frontier = (1. - reached) * incoming
    return reached + frontier, frontier


class TypedProcessOperator(nn.Module):
    """Distinct transition modules for signed resources, clocks and distances."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        if hidden_dim < 8:
            raise ValueError('hidden_dim must be at least 8')
        self.hidden_dim = hidden_dim
        self.node_encoder = _mlp(len(NODE_FIELDS) + 1, hidden_dim, hidden_dim)
        self.task_encoder = nn.Linear(len(TASK_FIELDS), hidden_dim)
        self.problem_embedding = nn.Embedding(len(PROBLEMS), hidden_dim)
        common_dim = 3 * hidden_dim + len(SCALAR_FIELDS) + len(BINARY_FIELDS) + 2 * len(GRAPH_FIELDS) + 1
        self.resource_operator = _mlp(common_dim, len(RESOURCE_INDICES), hidden_dim)
        self.distance_operator = _mlp(common_dim, len(DISTANCE_INDICES), hidden_dim)
        self.time_operator = _mlp(common_dim, len(TIME_INDICES), hidden_dim)
        self.limit_operator = _mlp(common_dim, len(LIMIT_INDICES), hidden_dim)
        self.flag_operator = _mlp(common_dim, len(BINARY_FIELDS), hidden_dim)
        self.pointer_query = nn.Linear(common_dim, hidden_dim)
        self.compatibility_operators = nn.ModuleList(
            _mlp(2 * hidden_dim + 3, 1, hidden_dim) for _ in GRAPH_FIELDS)
        summary_dim = 2 * (len(SCALAR_FIELDS) + len(BINARY_FIELDS) + 2 * len(GRAPH_FIELDS)) + len(GRAPH_FIELDS)
        self.trace_readout = _mlp(summary_dim, hidden_dim, hidden_dim)

    def _encode(self, program):
        if set(program) != set(INPUT_KEYS):
            raise ValueError('Processor accepts the compiled input contract only, never targets')
        nodes, valid = program['nodes'], program['node_mask']
        batch, width, _ = nodes.shape
        if (program['steps'].shape != (batch,) or (program['steps'] < 1).any()
                or (program['steps'] > width).any()
                or not torch.equal(valid.sum(-1), program['steps'])):
            raise ValueError('Execution lengths must come from valid compiled input nodes')
        task = (self.task_encoder(program['task_flags'].to(nodes.dtype))
                + self.problem_embedding(program['problem_id']))
        position = torch.arange(width, device=nodes.device, dtype=nodes.dtype) / max(1, width - 1)
        position = position[None, :, None].expand(batch, -1, -1)
        encoded = self.node_encoder(torch.cat((nodes, position), -1)) + task[:, None]
        encoded = encoded * valid[..., None]
        anchor = self.node_encoder(torch.cat((program['anchor'], nodes.new_full((batch, 1), -1.)), -1)) + task
        pool = encoded.sum(1) / valid.sum(-1, keepdim=True).clamp_min(1)
        return dict(nodes=encoded, anchor=anchor, pool=pool,
                    pointer_keys=torch.cat((encoded, anchor[:, None]), 1))

    def _step(self, program, encoded, previous, step):
        nodes, valid = program['nodes'], program['node_mask']
        batch, width, _ = nodes.shape
        count = valid.sum(-1).clamp_min(1)
        graph_stats = [previous[key].sum(-1) / count[:, None] for key in ('reached', 'frontier')]
        pointer_context = torch.einsum('bk,bkd->bd', previous['pointer'], encoded['pointer_keys'])
        fraction = nodes.new_full((batch, 1), step + 1) / program['steps'][:, None]
        common = torch.cat((encoded['pool'], encoded['nodes'][:, step], pointer_context,
                            previous['scalar'], previous['binary'], *graph_stats, fraction), -1)
        scalar = previous['scalar'].clone()
        for indices, operator in ((RESOURCE_INDICES, self.resource_operator),
                                  (DISTANCE_INDICES, self.distance_operator),
                                  (TIME_INDICES, self.time_operator), (LIMIT_INDICES, self.limit_operator)):
            scalar[:, indices] = previous['scalar'][:, indices] + operator(common)
        scalar = scalar * program['scalar_mask']
        binary = self.flag_operator(common).sigmoid() * program['binary_mask']
        query = self.pointer_query(common)
        pointer_logits = torch.einsum('bd,bkd->bk', query, encoded['pointer_keys']) / math.sqrt(self.hidden_dim)
        pointer_mask = torch.cat((valid, torch.ones(batch, 1, dtype=torch.bool, device=valid.device)), -1)
        pointer = pointer_logits.masked_fill(~pointer_mask, float('-inf')).softmax(-1)
        left = encoded['nodes'][:, :, None].expand(-1, -1, width, -1)
        right = encoded['nodes'][:, None].expand(-1, width, -1, -1)
        pair = valid[:, :, None] & valid[:, None] & ~torch.eye(width, dtype=torch.bool, device=valid.device)[None]
        compatibles = []
        for graph, operator in enumerate(self.compatibility_operators):
            features = torch.cat((left, right, program['edge_features'], previous['compatible'][:, graph, :, :, None]), -1)
            compatibles.append(operator(features).squeeze(-1).sigmoid())
        compatible = torch.stack(compatibles, 1) * pair[:, None] * program['graph_mask'][:, :, None, None]
        reached, frontier = soft_directed_propagation(previous['reached'], compatible, program['topology'])
        reached = reached * program['graph_mask'][:, :, None] * valid[:, None]
        frontier = frontier * program['graph_mask'][:, :, None] * valid[:, None]
        return dict(scalar=scalar, binary=binary, pointer=pointer, reached=reached,
                    frontier=frontier, compatible=compatible)

    @staticmethod
    def _teacher_state(program, targets, step):
        batch, width, _ = program['nodes'].shape
        dtype = program['nodes'].dtype
        if targets['scalar'].shape != (batch, width + 1, len(SCALAR_FIELDS)):
            raise ValueError('Teacher shape must match the input-derived bounded program')
        state = {key: targets[key][:, step].to(dtype) for key in ('scalar', 'binary', 'reached', 'frontier')}
        state['pointer'] = F.one_hot(targets['pointer'][:, step].long(), width + 1).to(dtype)
        state['compatible'] = (targets['compatible'].to(dtype) if step > 0
                               else initial_state(program)['compatible'])
        return state

    def rollout(self, program, *, targets=None, teacher_forcing=False):
        """The default path never accepts or looks up true hints/teacher lengths.

        A fixed input tensor width bounds the loop. Each row stops at its own
        input-derived steps, holding its last prediction thereafter. Targets can
        condition transitions only with explicit teacher_forcing in train mode.
        """
        if teacher_forcing and (not self.training or targets is None):
            raise PermissionError('Teacher forcing requires train mode and explicit TRAIN targets')
        if targets is not None and not teacher_forcing:
            raise PermissionError('Do not pass targets on the self-feedback path')
        encoded = self._encode(program)
        previous = initial_state(program)
        history = {key: [value] for key, value in previous.items()}
        deltas = []
        width = program['nodes'].shape[1]
        for step in range(width):
            state_in = self._teacher_state(program, targets, step) if teacher_forcing else previous
            predicted = self._step(program, encoded, state_in, step)
            active = step < program['steps']
            previous = {key: torch.where(active.reshape(-1, *([1] * (value.ndim - 1))), value, state_in[key])
                        for key, value in predicted.items()}
            deltas.append(previous['scalar'] - state_in['scalar'])
            for key, value in previous.items():
                history[key].append(value)
        outputs = {key: torch.stack(value, 1) for key, value in history.items()}
        outputs['scalar_delta'] = torch.stack(deltas, 1)
        outputs['representation'] = self.summarize(program, outputs, encoded['pool'])
        return outputs

    def forward(self, program):
        return self.rollout(program)

    def summarize(self, program, outputs, raw_pool):
        """Candidate-independent trace features for an external selector readout."""
        batch, width, _ = program['nodes'].shape
        count = program['node_mask'].sum(-1).clamp_min(1)
        parts = [outputs['scalar'], outputs['binary']]
        parts.extend(outputs[key].sum(-1) / count[:, None, None] for key in ('reached', 'frontier'))
        trace = torch.cat(parts, -1)
        mask = torch.arange(width + 1, device=count.device)[None] <= program['steps'][:, None]
        mean = (trace * mask[..., None]).sum(1) / mask.sum(-1, keepdim=True)
        last = trace[torch.arange(batch, device=count.device), program['steps']]
        compatibility = outputs['compatible'][torch.arange(batch, device=count.device), program['steps']]
        pair_count = (count * (count - 1)).clamp_min(1)
        compatibility = compatibility.sum((-1, -2)) / pair_count[:, None]
        return self.trace_readout(torch.cat((last, mean, compatibility), -1)) + raw_pool


def _masked_mean(value, mask):
    mask = torch.broadcast_to(mask, value.shape).to(value.dtype)
    axes = tuple(range(1, value.ndim))
    per_example = (value * mask).sum(axes) / mask.sum(axes).clamp_min(1)
    return per_example.mean()


def _binary_loss(predicted, target, mask):
    value = F.binary_cross_entropy(predicted.float().clamp(1e-6, 1. - 1e-6), target.float(), reduction='none')
    return _masked_mean(value, mask)


def process_loss(outputs, targets, program, weights=None):
    """Masked state AND transition losses. Targets never enter model.forward.

    Each component first averages its active elements within each example, then
    averages examples (inactive types contribute zero). For microbatches, weight
    each returned loss by actual_microbatch_rows / logical_batch_rows. This gives
    the same objective as an unsplit logical batch, including variable lengths.
    """
    active = targets['transition_mask']
    expected = torch.arange(active.shape[1], device=active.device)[None] < program['steps'][:, None]
    if not torch.equal(active, expected):
        raise ValueError('Teacher transition lengths must equal the input-derived schedule')
    scalar_mask = active[..., None] & program['scalar_mask'][:, None]
    state = F.smooth_l1_loss(outputs['scalar'][:, 1:].float(), targets['scalar'][:, 1:].float(), reduction='none')
    transition = F.smooth_l1_loss(outputs['scalar_delta'].float(), targets['scalar_delta'].float(), reduction='none')
    losses = dict(scalar_state=_masked_mean(state, scalar_mask),
                  scalar_transition=_masked_mean(transition, scalar_mask),
                  binary_state=_binary_loss(outputs['binary'][:, 1:], targets['binary'][:, 1:],
                                            active[..., None] & program['binary_mask'][:, None]))
    pointer = outputs['pointer'][:, 1:].float().clamp_min(1e-8)
    pointer_loss = -pointer.log().gather(-1, targets['pointer'][:, 1:, None].long()).squeeze(-1)
    losses['pointer_state'] = _masked_mean(pointer_loss, active)
    node_mask = program['node_mask']
    pair = node_mask[:, :, None] & node_mask[:, None]
    pair &= ~torch.eye(pair.shape[-1], dtype=torch.bool, device=pair.device)[None]
    graph_step = active[:, :, None] & program['graph_mask'][:, None]
    losses['compatible'] = _binary_loss(outputs['compatible'][:, 1:],
        targets['compatible'][:, None].expand_as(outputs['compatible'][:, 1:]),
        graph_step[..., None, None] & pair[:, None, None])
    for key in ('reached', 'frontier'):
        losses[key] = _binary_loss(outputs[key][:, 1:], targets[key][:, 1:],
                                   graph_step[..., None] & node_mask[:, None, None])
    weights = {} if weights is None else weights
    if set(weights) - set(losses):
        raise ValueError('Unknown process loss weight')
    losses['total'] = sum(value * weights.get(key, 1.) for key, value in losses.items())
    return losses


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU forward-only R113 processor interface check; no training')
    parser.add_argument('--shard', required=True)
    parser.add_argument('--hidden-dim', type=int, default=64)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    program, targets, _ = load_shard(args.shard)
    torch.manual_seed(2)
    model = TypedProcessOperator(args.hidden_dim).eval()
    with torch.no_grad():
        outputs = model(program)
        loss = process_loss(outputs, targets, program)
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
                          outputs={key: list(value.shape) for key, value in outputs.items()},
                          random_initialization_loss=float(loss['total']),
                          trained=False, device='cpu'), indent=2))


if __name__ == '__main__':
    main()
