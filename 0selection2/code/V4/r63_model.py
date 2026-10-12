"""R63 direct native-winner classification with tied node/solver workspaces."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .pairwise_objective import inputs_only
from .r42_experiment import symmetric_kl
from .r45_experiment import parameters
from .solver_code_encoder import make_r45_model
from .tensor_loader import TensorBatchLoader
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


def full_tokens(encoder, batch):
    """Keep the original dense encoder computation, including its three tokens."""
    node_mask = batch['node_mask']
    if batch['kind'] == 'coord':
        node = encoder._pad_node(batch['node'])
        stats = encoder._coord_stats(node, node_mask)
        condition, _ = encoder.condition_encoder(batch, stats, kind_id=0)
        physical = encoder.coord_proj(node)
        if hasattr(encoder, 'geometry_residual'):
            physical = physical + encoder.geometry_residual(node, node_mask, batch['n'], batch.get('node_geom'))
        bias = encoder._coord_bias(node, node_mask, 3)
    else:
        stats = encoder._matrix_stats(batch['matrix'], node_mask)
        condition, _ = encoder.condition_encoder(batch, stats, kind_id=1)
        features, bias = encoder._matrix_features_and_bias(batch['matrix'], node_mask, 3)
        physical = encoder.matrix_proj(features)
    token = torch.cat([encoder.cls_token.expand(len(physical), 1, -1), condition[:, None],
                       encoder.stats_token_proj(stats)[:, None], physical], dim=1)
    mask = torch.cat([torch.ones(len(token), 3, dtype=torch.bool, device=token.device), node_mask], dim=1)
    for layer in encoder.layers:
        token = layer(token, mask=mask, attn_bias=bias)
    return token, mask


class WorkspaceCell(nn.Module):
    def __init__(self, dim, message_dim, edge_dim, dropout):
        super().__init__()
        self.source = nn.Linear(dim, message_dim)
        self.target = nn.Linear(dim, message_dim)
        self.source_gate = nn.Linear(message_dim, 1, bias=False)
        self.target_gate = nn.Linear(message_dim, 1, bias=False)
        self.solver_gate = nn.Linear(dim, 1, bias=False)
        self.edge_gate = nn.Linear(edge_dim, 1)
        self.message = nn.Linear(message_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, current, memory, solvers, graph, valid):
        b, n, k, d = current.shape
        source, target = graph['flat_source'], graph['flat_receiver']
        # Only width-32 messages, not full-width states, are gathered per edge.
        projected_source = self.source(memory).reshape(b*n, k, -1)
        projected_target = self.target(current).reshape(b*n, k, -1)
        scalar = (self.source_gate(projected_source)[source] + self.target_gate(projected_target)[target]
                  + self.solver_gate(solvers)[target // n]
                  + self.edge_gate(graph['distance_features'])[:, None])
        gate = torch.sigmoid(scalar.float())
        numerator = torch.zeros(b*n, k, projected_source.size(-1), device=current.device, dtype=torch.float32)
        denominator = torch.zeros(b*n, k, 1, device=current.device, dtype=torch.float32)
        numerator.index_add_(0, target, projected_source[source].float() * gate)
        denominator.index_add_(0, target, gate)
        aggregate = numerator / denominator.clamp_min(1e-8)
        message = self.dropout(self.message(aggregate))
        updated = self.gru(message.reshape(-1, d), current.reshape(-1, d)).reshape(b, n, k, d)
        return updated.masked_fill(~valid[..., None], 0.)


class RecurrentWorkspaceSelector(nn.Module):
    def __init__(self, params, communicating):
        super().__init__()
        from .r63_graph import DISTANCE_FEATURES
        self.params, self.communicating = params, bool(communicating)
        base = make_r45_model(params)
        for name in ('instance_encoder', 'solver_encoder', 'pool', 'decoder', 'score_head'):
            setattr(self, name, getattr(base, name))
        self.initial_norm = nn.LayerNorm(params['embedding_dim'])
        self.cell = WorkspaceCell(params['embedding_dim'], params.get('message_dim', 32),
                                  len(DISTANCE_FEATURES), params['dropout'])
        self.rounds = params.get('workspace_rounds', 8)

    def forward(self, batch):
        from .r63_graph import build_workspace_graph
        clean = inputs_only(batch)
        nodes, node_mask = full_tokens(self.instance_encoder, clean)
        ids = clean['pool_ids'].to(nodes.device)
        if ids.ndim == 1:
            ids = ids[None].expand(len(nodes), -1)
        solver_mask = clean.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device).bool()
        if solver_mask.shape != ids.shape or not solver_mask.any(-1).all():
            raise ValueError('A legal solver is required in every query')
        solvers = self.solver_encoder(ids.masked_fill(~solver_mask, 0)).masked_fill(~solver_mask[..., None], 0.)
        valid = node_mask[:, :, None] & solver_mask[:, None, :]
        initial = self.initial_norm(nodes[:, :, None] + solvers[:, None]).masked_fill(~valid[..., None], 0.)
        graph = clean.get('workspace_graph')
        if graph is None:
            graph = build_workspace_graph(clean)
        current = initial
        for _ in range(self.rounds):
            memory = current if self.communicating else initial
            current = self.cell(current, memory, solvers, graph, valid)
        count = node_mask.sum(1).clamp_min(1)[:, None, None]
        mean = current.sum(1) / count
        maximum = current.masked_fill(~node_mask[:, :, None, None], float('-inf')).amax(1)
        pooled = self.pool(torch.cat([mean, maximum], -1))
        context = self.decoder(pooled, solvers, solvers, key_mask=solver_mask)
        utility = self.score_head(torch.cat([pooled, context, solvers], -1)).squeeze(-1).float()
        margins = (utility[:, :, None] - utility[:, None, :]).masked_fill(
            ~(solver_mask[:, :, None] & solver_mask[:, None, :]), 0.)
        return dict(logits=utility.masked_fill(~solver_mask, float('-inf')),
                    pair_margin=margins, solver_mask=solver_mask)


def initialize(arm, geometry, device='cpu', seed=2):
    if arm not in ('A', 'B'):
        raise ValueError('R63 has exactly one static and one communicating arm')
    set_seed(seed)
    params = dict(parameters('A', sdpa=str(device).startswith('cuda')),
                  architecture='recurrent_workspaces', workspace_rounds=8, message_dim=32,
                  neighbor_memory='previous_round' if arm == 'B' else 'initial_state')
    model = RecurrentWorkspaceSelector(params, communicating=arm == 'B').to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


def classification_objective(first, second, winner, consistency_weight):
    mask = first['solver_mask']
    if not torch.equal(mask, second['solver_mask']) or not mask.gather(1, winner[:, None]).all():
        raise ValueError('Native winner must belong to the same legal pool in both forwards')
    ce = .5 * (F.cross_entropy(first['logits'].float(), winner)
               + F.cross_entropy(second['logits'].float(), winner))
    kl = symmetric_kl(first['logits'], second['logits'], mask)
    return ce + consistency_weight * kl, dict(ce=ce, kl=kl)


def update_batch(model, optimizer, scaler, batch, consistency_weight, microbatch_size=16):
    from .r63_graph import build_workspace_graph
    rng, rows = capture_rng(), len(batch['ind'])
    for retry in range(20):
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        totals = dict(loss=0., ce=0., kl=0.)
        for micro in TensorBatchLoader(batch, microbatch_size, False, False):
            clean = inputs_only(micro)
            clean['workspace_graph'] = build_workspace_graph(clean)
            with torch.autocast(device_type=micro['ind'].device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                first, second = model(clean), model(clean)
            loss, parts = classification_objective(first, second, micro['ind'], consistency_weight)
            weight = len(micro['ind']) / rows
            scaler.scale(loss * weight).backward()
            for key, value in dict(loss=loss, **parts).items():
                totals[key] += float(value.detach()) * weight
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        if not scaler.is_enabled() and (not torch.isfinite(norm) or not all(math.isfinite(v) for v in totals.values())):
            raise RuntimeError('Nonfinite full-batch gradients; no examples skipped')
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            if not torch.isfinite(norm) or not all(math.isfinite(v) for v in totals.values()):
                raise RuntimeError('Nonfinite successful update')
            return dict(totals, grad_norm=float(norm), skipped=retry, microbatches=math.ceil(rows/microbatch_size))
        print(f'[R63 AMP retry] entire original batch; lower scale={scaler.get_scale()}', flush=True)
    raise RuntimeError('Repeated AMP overflow on the same R63 original batch')
