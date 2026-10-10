"""Node-aligned probe of two frozen ReLD encoders; no solver decision inputs."""

import torch
from torch import nn


RAW_FIELDS = ('x', 'y', 'demand', 'service_time', 'tw_start', 'tw_end', 'is_depot')


class SolverEncoderReadout(nn.Module):
    def __init__(self, encoder_dim=128, hidden_dim=128, dropout=.1):
        super().__init__()
        self.moel_norm = nn.LayerNorm(encoder_dim)
        self.mtl_norm = nn.LayerNorm(encoder_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * encoder_dim + len(RAW_FIELDS), hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * hidden_dim + 1, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, encoded, raw_node, node_mask):
        if encoded.ndim != 4 or encoded.shape[2] != 2 or node_mask.dtype != torch.bool:
            raise ValueError('Expected [B,N,MOEL/MTL,D] and a boolean node mask')
        if not node_mask[:, 0].all() or not node_mask[:, 1:].any(1).all():
            raise ValueError('Each instance needs depot0 and at least one customer')
        features = torch.cat((self.moel_norm(encoded[:, :, 0]),
                              self.mtl_norm(encoded[:, :, 1]), raw_node), -1)
        nodes = self.node_mlp(features)
        customer_mask = node_mask.clone()
        customer_mask[:, 0] = False
        count = customer_mask.sum(1, keepdim=True)
        mean = nodes.masked_fill(~customer_mask[..., None], 0).sum(1) / count
        maximum = nodes.masked_fill(~customer_mask[..., None], -torch.inf).amax(1)
        scale = torch.log1p(count.to(nodes.dtype)) / 6
        return self.head(torch.cat((mean, maximum, nodes[:, 0], scale), -1))
