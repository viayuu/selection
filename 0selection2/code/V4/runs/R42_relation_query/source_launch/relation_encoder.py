"""Sparse relation messages and global attention, updated in parallel."""

import torch
from torch import nn

from .relation_graph import EDGE_FIELDS, build_relation_graph
from .V4Model import InstanceEncoder, MultiHeadAttention, FeedForward


class RelationLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        d, e = params['embedding_dim'], params.get('edge_dim', 32)
        self.source = nn.Linear(d, e)
        self.target = nn.Linear(d, e)
        self.edge_value = nn.Linear(e, e)
        self.gate = nn.Linear(e, e)
        self.local_out = nn.Linear(e, d)
        self.edge_update = nn.Sequential(nn.Linear(e, 2 * e), nn.GELU(), nn.Linear(2 * e, e)) if params.get('update_edges', True) else None
        self.edge_norm = nn.LayerNorm(e) if self.edge_update is not None else None
        self.attn = MultiHeadAttention(d, params['head_num'], params['qkv_dim'], params['dropout'], params.get('sdpa', False))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = FeedForward(d, params['ff_hidden_dim'], params['dropout'])
        self.dropout = nn.Dropout(params['dropout'])

    def forward(self, nodes, edges, graph, mask, bias):
        b, n, d = nodes.shape
        source, target = graph['index']
        flat = nodes.reshape(b * n, d)
        pair = self.source(flat)[source] + self.target(flat)[target]
        message = torch.tanh(pair + self.edge_value(edges)) * torch.sigmoid(self.gate(pair + edges))
        # nonzero() emits edges grouped by source; segment sums avoid CUDA atomics.
        degree = graph['degree']
        aggregate = torch.segment_reduce(message, 'sum', lengths=degree)
        local = self.local_out(aggregate / degree.clamp_min(1).to(message.dtype)[:, None]).reshape(b, n, d)
        global_message = self.attn(nodes, nodes, nodes, key_mask=mask, attn_bias=bias)
        next_edges = self.edge_norm(edges + self.dropout(self.edge_update(pair + edges))) if self.edge_update is not None else edges
        nodes = self.norm1(nodes + self.dropout(local) + self.dropout(global_message))
        nodes = self.norm2(nodes + self.dropout(self.ffn(nodes)))
        return nodes.masked_fill(~mask[..., None], 0), next_edges


class RelationEncoder(InstanceEncoder):
    def __init__(self, **params):
        super().__init__(**dict(params, node_only=True))
        self.layers = nn.ModuleList([RelationLayer(**dict(params, update_edges=i < params['encoder_layer_num'] - 1))
                                     for i in range(params['encoder_layer_num'])])
        del self.cls_token, self.stats_token_proj
        e = params.get('edge_dim', 32)
        self.edge_proj = nn.Sequential(nn.Linear(len(EDGE_FIELDS), 64), nn.GELU(), nn.Linear(64, e), nn.LayerNorm(e))
        self.level = nn.Embedding(2, params['embedding_dim'])

    def forward(self, batch):
        mask = batch['node_mask'].bool()
        if batch['kind'] == 'coord':
            node = self._pad_node(batch['node']).masked_fill(~mask[..., None], 0)
            stats = self._coord_stats(node, mask)
            condition, _ = self.condition_encoder(batch, stats, 0)
            nodes = self.coord_proj(node)
            if hasattr(self, 'geometry_residual'):
                nodes = nodes + self.geometry_residual(node, mask, batch['n'], batch.get('node_geom'))
            bias = self._coord_bias(node, mask, 0)
        else:
            matrix = batch['matrix'].masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0)
            stats = self._matrix_stats(matrix, mask)
            condition, _ = self.condition_encoder(batch, stats, 1)
            features, bias = self._matrix_features_and_bias(matrix, mask, 0)
            nodes = self.matrix_proj(features)
        nodes = (nodes + condition[:, None]).masked_fill(~mask[..., None], 0)
        graph = batch.get('relation_graph')
        if graph is None:
            graph = build_relation_graph(batch)
        edges = self.edge_proj(graph['features'])
        levels = []
        for i, layer in enumerate(self.layers):
            nodes, edges = layer(nodes, edges, graph, mask, bias)
            if i in (1, 3):
                levels.append((nodes + self.level.weight[len(levels)]).masked_fill(~mask[..., None], 0))
        return torch.cat(levels, 1), torch.cat((mask, mask), 1), nodes, mask, condition
