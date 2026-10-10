"""EGT-inspired directed node/relation encoding for the MOEL/MTL specialist."""

import math

import torch
from torch import nn

from code.unified_selector.registry import K_CBITS, K_PROBLEM_DESC
from .local_geometry import GeometryResidual
from .pair_specialist import INPUT_FIELDS
from .V4Model import ConditionEncoder, InstanceEncoder, _masked_max, _masked_mean


class JointRelationLayer(nn.Module):
    """EGT equations 3-7: gated node attention and persistent edge residuals."""

    def __init__(self, d=128, edge_dim=32, heads=4, dropout=.1):
        super().__init__()
        if d % heads:
            raise ValueError('Node dimension must be divisible by attention heads')
        self.heads, self.width = heads, d // heads
        self.node_norm = nn.LayerNorm(d)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.qkv = nn.Linear(d, 3*d)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.edge_gate = nn.Linear(edge_dim, heads)
        self.node_out = nn.Linear(d, d)
        self.edge_out = nn.Linear(heads, edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.node_ff_norm = nn.LayerNorm(d)
        self.edge_ff_norm = nn.LayerNorm(edge_dim)
        self.node_ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d, d))
        self.edge_ff = nn.Sequential(nn.Linear(edge_dim, 4*edge_dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(4*edge_dim, edge_dim))

    def forward(self, nodes, edges, mask):
        b, n, d = nodes.shape
        valid = mask[:, :, None] & mask[:, None, :]
        h, e = self.node_norm(nodes), self.edge_norm(edges)
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, self.width).unbind(2)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        dot = (q @ k.transpose(-1, -2)) / math.sqrt(self.width)
        score = dot.clamp(-5., 5.) + self.edge_bias(e).permute(0, 3, 1, 2)
        # Gate after softmax, without renormalizing, as in EGT equation 4.
        probability = score.float().masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
        gate = self.edge_gate(e).permute(0, 3, 1, 2).float().sigmoid()
        attention = self.dropout(probability * gate).to(v.dtype)
        message = (attention @ v).transpose(1, 2).reshape(b, n, d)
        new_nodes = nodes + self.dropout(self.node_out(message))
        new_nodes = new_nodes + self.dropout(self.node_ff(self.node_ff_norm(new_nodes)))
        new_edges = edges + self.dropout(self.edge_out(score.permute(0, 2, 3, 1)))
        new_edges = new_edges + self.dropout(self.edge_ff(self.edge_ff_norm(new_edges)))
        return new_nodes.masked_fill(~mask[..., None], 0.), new_edges.masked_fill(~valid[..., None], 0.)


class RelationalInstanceEncoder(nn.Module):
    """Replace the old node layers; reuse only original input/condition helpers."""

    _pad_node = InstanceEncoder._pad_node
    _coord_stats = InstanceEncoder._coord_stats

    def __init__(self, **params):
        super().__init__()
        self.params = dict(params)
        d, e = params['embedding_dim'], params.get('relation_dim', 32)
        self.coord_proj = nn.Linear(8, d)
        if params.get('local_geometry', False):
            self.geometry_residual = GeometryResidual(d, params.get('geometry_mode', 'real'))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.stats_token_proj = nn.Linear(params['stats_dim'], d)
        self.condition_encoder = ConditionEncoder(**params)
        self.distance_projection = nn.Linear(2, e)
        self.source_projection = nn.Linear(9, e, bias=False)
        self.target_projection = nn.Linear(9, e, bias=False)
        self.relation_condition = nn.Linear(K_CBITS+K_PROBLEM_DESC+1, e, bias=False)
        self.layers = nn.ModuleList([JointRelationLayer(d, e, params['head_num'], params['dropout'])
                                    for _ in range(params['encoder_layer_num'])])
        self.final_node_norm = nn.LayerNorm(d)

    def initial_states(self, batch):
        if batch['kind'] != 'coord':
            raise ValueError('R56 is restricted to the 15 coordinate MVRP tasks')
        mask = batch['node_mask'].bool()
        node = self._pad_node(batch['node']).float().masked_fill(~mask[..., None], 0.)
        stats = self._coord_stats(node, mask)
        condition, _ = self.condition_encoder(batch, stats, kind_id=0)
        h = self.coord_proj(node)
        if hasattr(self, 'geometry_residual'):
            h = h + self.geometry_residual(node, mask, batch['n'], batch.get('node_geom'))
        b, n = mask.shape
        h = torch.cat((self.cls_token.expand(b, 1, -1), condition[:, None],
                       self.stats_token_proj(stats)[:, None], h), dim=1)
        all_mask = torch.cat((torch.ones(b, 3, dtype=torch.bool, device=mask.device), mask), dim=1)
        valid = all_mask[:, :, None] & all_mask[:, None, :]
        with torch.autocast(device_type=node.device.type, enabled=False):
            distance = torch.cdist(node[..., :2], node[..., :2])
            real_valid = mask[:, :, None] & mask[:, None, :]
            off_diagonal = real_valid & ~torch.eye(n, dtype=torch.bool, device=node.device)[None]
            scale = (distance.masked_fill(~off_diagonal, 0.).sum((1, 2)) /
                     off_diagonal.sum((1, 2)).clamp_min(1)).clamp_min(1e-6)
            distances = node.new_zeros(b, n+3, n+3, 2)
            distances[:, 3:, 3:, 0] = distance.masked_fill(~real_valid, 0.)
            distances[:, 3:, 3:, 1] = (distance/scale[:, None, None]).masked_fill(~real_valid, 0.)
        # Separate source/target projections preserve direction without materializing
        # a B x N x N x 42 concatenation of identical endpoint/condition inputs.
        endpoints = torch.cat((node, torch.ones(b, n, 1, device=node.device)), dim=-1)
        endpoints = torch.cat((node.new_zeros(b, 3, 9), endpoints), dim=1)
        relation_condition = torch.cat((batch['cbits'].float(), batch['problem_desc'].float(),
                                         batch['n'].float().log1p()[:, None]/6.), dim=-1)
        edges = (self.distance_projection(distances) + self.source_projection(endpoints)[:, :, None]
                 + self.target_projection(endpoints)[:, None, :]
                 + self.relation_condition(relation_condition)[:, None, None])
        return h.masked_fill(~all_mask[..., None], 0.), edges.masked_fill(~valid[..., None], 0.), all_mask

    def forward(self, batch, return_states=False):
        nodes, edges, mask = self.initial_states(batch)
        states = []
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, mask)
            if return_states:
                states.append((nodes, edges))
        nodes = self.final_node_norm(nodes).masked_fill(~mask[..., None], 0.)
        result = (nodes[:, 3:], mask[:, 3:])
        if return_states:
            return (*result, dict(states=states, final_relations=edges, all_mask=mask))
        return result


class RelationalPairSpecialist(nn.Module):
    """Positive logit means MOEL; the ordinary R53A pooling/head is unchanged."""

    def __init__(self, params):
        super().__init__()
        self.params = dict(params, node_only=True, relation_dim=32)
        self.instance_encoder = RelationalInstanceEncoder(**self.params)
        d = self.params['embedding_dim']
        self.head = nn.Sequential(nn.Linear(2*d+1, d), nn.GELU(),
                                  nn.Dropout(self.params['dropout']), nn.Linear(d, 1))

    def forward(self, batch, return_states=False):
        clean = {k: batch[k] for k in INPUT_FIELDS if k in batch}
        encoded = self.instance_encoder(clean, return_states=return_states)
        nodes, mask = encoded[:2]
        summary = torch.cat((_masked_mean(nodes, mask), _masked_max(nodes, mask),
                             clean['n'].float().log1p()[:, None]/6.), dim=-1)
        z = self.head(summary).squeeze(-1)
        return (z, encoded[2]) if return_states else z
