"""R92 input-only multiresolution graph encoders, with the original selector.

Runner API: ``make_model(params, arm, state_dict=None)`` for arms A/B only.
Forward adds scalar ``auxiliary_loss`` and ``auxiliary_parts`` (link/entropy).
Both are per-graph means; a runner averages the two dropout forwards and weights
microbatches by their actual row counts. Evaluation skips auxiliary computation.

Fixed choices: 32/8 TOTAL clusters, including singleton cluster 0 for coordinate
graphs. TSP uses R88's execution-role slot 3, identically in A/B; other coordinate
graphs protect the first valid physical node (depot). ATSP has no protected node.
TW is carried in raw attributes/conditions, not guessed into a relation channel.
The two scalar directed relations are off-diagonal-mean-normalized input distance
and exp(-distance). There are no edge embeddings, cost heads or route targets.

The four backbone blocks are fine, 32-cluster, 8-cluster, 8-cluster GNNs. B's
separate assignment MLPs read outgoing/incoming relation messages, not PMA tokens.
At subsequent pools, S_eff = diag(input_mass) S preserves original-node mass:
X' = S_eff.T X / mass', E' = S_eff.T E S_eff / outer(mass', mass').
Hard farthest/nearest assignments use symmetrized input distance only, whereas
both propagation and soft relation reconstruction retain directed relations.
No old global-attention layers or unused CLS/bias parameters are retained.
"""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from code.unified_selector.registry import P2I
from .V4Model import InstanceEncoder
from .local_geometry import GeometryResidual
from .pairwise_selector import ScoreDifferenceSelector
from .r88_model import prepare_role_inputs as _r88_prepare_role_inputs


ARMS = ('A', 'B')
SELECTION_HEAD = 'original_score_mlp_maximin'
ARCHITECTURE = dict(
    name='r92_multiresolution_directed_gnn', version=1,
    clusters=[32, 8], protected_anchor_in_cluster_budget=True,
    message_blocks=4, embedding_dim=128, ff_hidden_dim=512, dropout=.1,
    relations=['normalized_input_distance', 'exp_negative_normalized_distance'],
    coordinate_anchor='R88_execution_role_TSP_else_first_valid_depot',
    atsp_anchor='none', raw_time_windows='attributes_only',
    mass_semantics='original_physical_node_mass',
    auxiliary_link_weight=.1, auxiliary_entropy_weight=.01,
)


def prepare_role_inputs(batch):
    """The same validated, label-free execution role for both R92 arms."""
    return _r88_prepare_role_inputs(batch, 'B_execution_anchor')


def _pairs(mask):
    return mask[:, :, None] & mask[:, None, :]


def input_relations(distance, mask):
    """Two scalar channels only; normalization ignores padding and self pairs."""
    valid = _pairs(mask)
    off = valid & ~torch.eye(mask.size(1), dtype=torch.bool, device=mask.device)[None]
    with torch.autocast(device_type=distance.device.type, enabled=False):
        distance = distance.float().masked_fill(~valid, 0.)
        scale = distance.abs().masked_fill(~off, 0.).sum((1, 2), keepdim=True)
        scale = scale / off.sum((1, 2), keepdim=True).clamp_min(1)
        normalized = distance / scale.clamp_min(1e-6)
        affinity = torch.exp(-normalized.clamp_min(0.))
        return torch.stack([normalized, affinity], -1).masked_fill(~valid[..., None], 0.)


def directed_messages(nodes, relation, mask):
    """O(N^2 d) matmuls, never an N x N x d tensor or an N^3 product."""
    with torch.autocast(device_type=nodes.device.type, enabled=False):
        affinity = relation[..., 1].float().masked_fill(~_pairs(mask), 0.)
        outgoing = affinity / affinity.sum(-1, keepdim=True).clamp_min(1e-12)
        incoming = affinity.transpose(1, 2)
        incoming = incoming / incoming.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.bmm(outgoing, nodes.float()), torch.bmm(incoming, nodes.float())


class DirectedGraphBlock(nn.Module):
    """One shared nodewise update of state, out/in messages and condition."""

    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(4 * d, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d))
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, relation, mask, condition):
        outgoing, incoming = directed_messages(nodes, relation, mask)
        context = condition[:, None].expand_as(nodes)
        delta = self.ffn(torch.cat([nodes, outgoing, incoming, context], -1))
        return self.norm(nodes + self.dropout(delta)).masked_fill(~mask[..., None], 0.)


class RelationAssignment(nn.Module):
    """Learned, relation-conditioned row-softmax assignments, not attention pools."""

    def __init__(self, d, hidden, dropout, clusters):
        super().__init__()
        self.clusters = clusters
        self.network = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, hidden),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, clusters))

    def forward(self, nodes, relation, mask, condition, protected):
        outgoing, incoming = directed_messages(nodes, relation, mask)
        context = condition[:, None].expand_as(nodes)
        logits = self.network(torch.cat([nodes, outgoing, incoming, context], -1)).float()
        k = self.clusters
        count = mask.sum(-1).clamp_max(k)
        allowed = torch.arange(k, device=mask.device)[None] < count[:, None]
        has_anchor = protected.any(-1)
        allowed = allowed & ~((torch.arange(k, device=mask.device)[None] == 0)
                              & has_anchor[:, None])
        # Finite fill also handles a graph consisting solely of its anchor.
        logits = logits.masked_fill(~allowed[:, None], torch.finfo(logits.dtype).min)
        assignment = logits.softmax(-1) * allowed[:, None].float()
        anchor_assignment = F.one_hot(torch.zeros_like(mask, dtype=torch.long), k).float()
        assignment = torch.where(protected[..., None], anchor_assignment, assignment)
        return assignment.masked_fill(~mask[..., None], 0.)


@torch.no_grad()
def hard_assignment(relation, mask, clusters, protected):
    """Batched farthest anchors, nearest cells; anchors own their tie cells."""
    distance = (relation[..., 0] + relation[..., 0].transpose(1, 2)) * .5
    b, n = mask.shape
    rows = torch.arange(b, device=mask.device)
    anchors = torch.zeros(b, clusters, dtype=torch.long, device=mask.device)
    active = torch.zeros(b, clusters, dtype=torch.bool, device=mask.device)
    selected = torch.zeros_like(mask)
    nearest = distance.new_full((b, n), float('inf'))
    first = torch.where(protected.any(-1), protected.long().argmax(-1), mask.long().argmax(-1))
    for index in range(clusters):
        candidates = mask & ~selected
        exists = candidates.any(-1)
        chosen = first if index == 0 else nearest.masked_fill(~candidates, -1.).argmax(-1)
        anchors[:, index] = chosen
        active[:, index] = exists
        selected[rows, chosen] |= exists
        to_anchor = distance.gather(2, chosen[:, None, None].expand(-1, n, 1)).squeeze(-1)
        nearest = torch.minimum(nearest, to_anchor.masked_fill(~exists[:, None], float('inf')))
    distances = distance.gather(2, anchors[:, None].expand(-1, n, -1))
    distances = distances.masked_fill(~active[:, None], float('inf'))
    distances[:, :, 0] = distances[:, :, 0].masked_fill(
        protected.any(-1)[:, None] & ~protected, float('inf'))
    cells = distances.argmin(-1)
    for index in range(clusters):
        cells[rows[active[:, index]], anchors[active[:, index], index]] = index
    cells = cells.masked_fill(protected, 0)
    return F.one_hot(cells, clusters).float().masked_fill(~mask[..., None], 0.)


def coarsen(nodes, attributes, marker, mass, relation, assignment):
    """Mass-normalized raw pooling, including the actual directed coarse graph."""
    with torch.autocast(device_type=nodes.device.type, enabled=False):
        weights = assignment.float() * mass.float()[..., None]
        transpose = weights.transpose(1, 2)
        next_mass = weights.sum(1)
        next_mask = next_mass > 0
        denominator = next_mass.clamp_min(1e-12)[..., None]
        next_nodes = torch.bmm(transpose, nodes.float()) / denominator
        next_attributes = torch.bmm(transpose, attributes.float()) / denominator
        next_marker = torch.bmm(transpose, marker.float()) / denominator
        pair_mass = next_mass[:, :, None] * next_mass[:, None, :]
        channels = [torch.bmm(transpose, torch.bmm(relation[..., channel].float(), weights))
                    / pair_mass.clamp_min(1e-12) for channel in range(relation.size(-1))]
        next_relation = torch.stack(channels, -1).masked_fill(~_pairs(next_mask)[..., None], 0.)
        return (next_nodes.masked_fill(~next_mask[..., None], 0.),
                next_attributes.masked_fill(~next_mask[..., None], 0.),
                next_marker.masked_fill(~next_mask[..., None], 0.),
                next_mass, next_relation, next_mask)


def structural_auxiliary(assignment, relation, coarse_relation, mask, mass, protected):
    """Per-graph input-relation MSE and mass-weighted normalized row entropy."""
    with torch.autocast(device_type=assignment.device.type, enabled=False):
        assignment = assignment.float()
        valid_pairs = _pairs(mask) & ~torch.eye(mask.size(1), device=mask.device, dtype=torch.bool)[None]
        pair_weight = mass[:, :, None] * mass[:, None, :] * valid_pairs
        link = mass.new_zeros(mass.size(0))
        for channel in range(relation.size(-1)):
            reconstruction = torch.bmm(torch.bmm(assignment, coarse_relation[..., channel]),
                                       assignment.transpose(1, 2))
            error = (reconstruction - relation[..., channel]).square()
            link = link + (error * pair_weight).sum((1, 2)) / pair_weight.sum((1, 2)).clamp_min(1.)
        link = link / relation.size(-1)
        eligible = mask & ~protected
        used = (assignment.sum(1) > 0).sum(-1) - protected.any(-1).long()
        normalizer = used.float().clamp_min(2).log()
        entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(-1) / normalizer[:, None]
        row_weight = mass * eligible
        entropy = (entropy * row_weight).sum(-1) / row_weight.sum(-1).clamp_min(1.)
        return dict(link=link, entropy=entropy)


class MultiresolutionInstanceEncoder(nn.Module):
    _pad_node = InstanceEncoder._pad_node
    _coord_stats = InstanceEncoder._coord_stats
    _matrix_stats = InstanceEncoder._matrix_stats
    _matrix_features_and_bias = InstanceEncoder._matrix_features_and_bias

    def __init__(self, original, arm):
        super().__init__()
        self.params = original.params
        self.arm = arm
        d = self.params['embedding_dim']
        hidden, dropout = self.params['ff_hidden_dim'], self.params['dropout']
        self.coord_proj = original.coord_proj
        self.matrix_proj = original.matrix_proj
        self.condition_encoder = original.condition_encoder
        self.stats_token_proj = original.stats_token_proj
        self.geometry_residual = getattr(original, 'geometry_residual', None)
        if self.geometry_residual is None:
            self.geometry_residual = GeometryResidual(d)
        self.input_norm = nn.LayerNorm(d)
        self.raw_pool_proj = nn.Linear(10, d)
        self.pool_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.blocks = nn.ModuleList([DirectedGraphBlock(d, hidden, dropout) for _ in range(4)])
        self.assignment_nets = nn.ModuleList(
            [RelationAssignment(d, hidden, dropout, k) for k in (32, 8)] if arm == 'B' else [])
        self.compute_auxiliary = True
        self.capture_hierarchy = False
        self.auxiliary_parts = {}
        self.auxiliary_per_graph = {}
        self.hierarchy_debug = ()

    def _matrix_features(self, matrix, mask):
        # Reuse the original feature primitive without allocating its unused
        # H x N x N attention bias or retaining trainable bias-scale parameters.
        class FeaturePrimitive:
            params = {'head_num': 0}
            matrix_bias_scale = matrix.new_empty(0)
        features, _ = InstanceEncoder._matrix_features_and_bias(FeaturePrimitive(), matrix, mask, 0)
        return features

    def forward(self, batch):
        mask = batch['node_mask']
        if mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError('R92 requires a nonempty boolean node_mask for every graph')
        protected = torch.zeros_like(mask)
        if batch['kind'] == 'coord':
            attributes = self._pad_node(batch['node']).masked_fill(~mask[..., None], 0.)
            stats = self._coord_stats(attributes, mask)
            condition, _ = self.condition_encoder(batch, stats, 0)
            nodes = self.coord_proj(attributes)
            nodes = nodes + self.geometry_residual(attributes, mask, batch['n'], batch.get('node_geom'))
            with torch.autocast(device_type=nodes.device.type, enabled=False):
                distance = torch.cdist(attributes[..., :2].float(), attributes[..., :2].float(),
                                       compute_mode='donot_use_mm_for_euclid_dist')
            if batch['problem_id'] == P2I['TSP']:
                protected = (attributes[..., 3] == 1) & mask
            else:
                protected.scatter_(1, mask.long().argmax(-1)[:, None], True)
        elif batch['kind'] == 'matrix':
            distance = batch['matrix'].masked_fill(~_pairs(mask), 0.)
            stats = self._matrix_stats(distance, mask)
            condition, _ = self.condition_encoder(batch, stats, 1)
            attributes = self._matrix_features(distance, mask).masked_fill(~mask[..., None], 0.)
            nodes = self.matrix_proj(attributes)
        else:
            raise ValueError('R92 supports coordinate or directed matrix inputs only')
        nodes = self.input_norm(nodes + condition[:, None] + self.stats_token_proj(stats)[:, None])
        nodes = nodes.masked_fill(~mask[..., None], 0.)
        relation = input_relations(distance, mask)
        mass = mask.float()
        marker = protected.float()[..., None]
        nodes = self.blocks[0](nodes, relation, mask, condition)
        debug, losses = [], []
        for stage, clusters in enumerate((32, 8)):
            if self.arm == 'A':
                assignment = hard_assignment(relation, mask, clusters, protected)
            else:
                assignment = self.assignment_nets[stage](nodes, relation, mask, condition, protected)
            next_values = coarsen(nodes, attributes, marker, mass, relation, assignment)
            next_nodes, next_attributes, next_marker, next_mass, next_relation, next_mask = next_values
            if self.arm == 'B' and self.training and self.compute_auxiliary:
                losses.append(structural_auxiliary(assignment, relation, next_relation, mask, mass, protected))
            if self.capture_hierarchy:
                debug.append({key: value.detach() for key, value in dict(
                    assignment=assignment, input_relation=relation, input_mask=mask,
                    input_mass=mass, input_protected=protected,
                    relation=next_relation, mask=next_mask, mass=next_mass,
                    attributes=next_attributes, marker=next_marker).items()})
            nodes, attributes, marker, mass, relation, mask = next_values
            raw_features = torch.cat([attributes, marker, mass.log1p()[..., None]], -1)
            nodes = self.pool_norms[stage](nodes + self.raw_pool_proj(raw_features))
            nodes = nodes.masked_fill(~mask[..., None], 0.)
            protected = (marker.squeeze(-1) > .5) & mask
            nodes = self.blocks[stage + 1](nodes, relation, mask, condition)
        nodes = self.blocks[3](nodes, relation, mask, condition)
        zeros = nodes.new_zeros(nodes.size(0), dtype=torch.float32)
        self.auxiliary_per_graph = {key: torch.stack([part[key] for part in losses]).mean(0)
                                    if losses else zeros for key in ('link', 'entropy')}
        self.auxiliary_parts = {key: value.mean() for key, value in self.auxiliary_per_graph.items()}
        self.hierarchy_debug = tuple(debug)
        return nodes, mask


class R92HierarchicalSelector(ScoreDifferenceSelector):
    def __init__(self, arm, **params):
        if arm not in ARMS:
            raise ValueError('R92 neural factory supports only A and B')
        params = copy.deepcopy(params)
        for name, required in (('embedding_dim', 128), ('ff_hidden_dim', 512),
                               ('dropout', .1), ('joint_layer_num', 2)):
            if params.get(name, required) != required:
                raise ValueError(f'R92 requires {name}={required}')
            params[name] = required
        if (params.get('solver_representation', 'handcrafted') != 'handcrafted'
                or params.get('pair_mode', 'score_difference') != 'score_difference'
                or params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD
                or any(name.startswith('residual_') for name in params)):
            raise ValueError('R92 retains the original ID/34-feature encoder and score/maximin head')
        if params.get('r92_arm', arm) != arm or params.get('r92_architecture', ARCHITECTURE) != ARCHITECTURE:
            raise ValueError('R92 checkpoint arm or architecture does not match')
        params.update(r92_arm=arm, r92_architecture=copy.deepcopy(ARCHITECTURE),
                      selection_head=SELECTION_HEAD, encoder_layer_num=4)
        super().__init__(**params)
        self.arm = arm
        self.instance_encoder = MultiresolutionInstanceEncoder(self.instance_encoder, arm)

    def prepare_role_inputs(self, batch):
        return prepare_role_inputs(batch)

    def forward(self, batch, return_hierarchy=False, return_auxiliary=None):
        encoder = self.instance_encoder
        encoder.compute_auxiliary = self.training and return_auxiliary is not False
        encoder.capture_hierarchy = bool(return_hierarchy)
        # The original forward -> encode_pairs -> encode_state, solver encoder,
        # joint layers, pooled decoder, score MLP and maximin are unchanged.
        output = super().forward(self.prepare_role_inputs(batch))
        output['auxiliary_parts'] = dict(encoder.auxiliary_parts)
        output['auxiliary_loss'] = (.1 * encoder.auxiliary_parts['link']
                                    + .01 * encoder.auxiliary_parts['entropy'])
        if return_hierarchy:
            output['hierarchy_debug'] = encoder.hierarchy_debug
        return output


def make_model(params, arm, state_dict=None):
    """Fresh construction or strict R92-only reconstruction; never old checkpoints."""
    model = R92HierarchicalSelector(arm, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
