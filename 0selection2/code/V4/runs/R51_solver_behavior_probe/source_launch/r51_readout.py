"""R50 static readout and graph-level fusion of native prefix behavior."""

import torch
from torch import nn

from .r50_readout import SolverEncoderReadout
from .r51_probe import BEHAVIOR_DIM


class BehaviorReadout(SolverEncoderReadout):
    def __init__(self, behavior=False, mean=None, std=None):
        super().__init__()
        self.behavior_enabled = behavior
        if behavior:
            self.register_buffer('behavior_mean', torch.zeros(BEHAVIOR_DIM) if mean is None else mean.clone())
            self.register_buffer('behavior_std', torch.ones(BEHAVIOR_DIM) if std is None else std.clone())
            self.behavior_mlp = nn.Sequential(nn.Linear(BEHAVIOR_DIM, 64), nn.GELU(), nn.Dropout(.1),
                                             nn.Linear(64, 64), nn.GELU())
            self.head[0] = nn.Linear(385 + 64, 128)

    def graph_summary(self, encoded, raw_node, node_mask):
        if encoded.ndim != 4 or encoded.shape[2] != 2 or node_mask.dtype != torch.bool:
            raise ValueError('Expected node-aligned dual solver features and boolean mask')
        if not node_mask[:, 0].all() or not node_mask[:, 1:].any(1).all():
            raise ValueError('A graph needs depot0 and at least one customer')
        features = torch.cat((self.moel_norm(encoded[:, :, 0]), self.mtl_norm(encoded[:, :, 1]), raw_node), -1)
        nodes = self.node_mlp(features)
        customer_mask = node_mask.clone()
        customer_mask[:, 0] = False
        count = customer_mask.sum(1, keepdim=True)
        mean = nodes.masked_fill(~customer_mask[..., None], 0).sum(1) / count
        maximum = nodes.masked_fill(~customer_mask[..., None], -torch.inf).amax(1)
        scale = torch.log1p(count.to(nodes.dtype)) / 6
        return torch.cat((mean, maximum, nodes[:, 0], scale), -1)

    def forward(self, encoded, raw_node, node_mask, behavior=None):
        graph = self.graph_summary(encoded, raw_node, node_mask)
        if self.behavior_enabled:
            if behavior is None or behavior.shape[-1] != BEHAVIOR_DIM:
                raise ValueError('Behavior features are required by the behavior-enhanced readout')
            evidence = self.behavior_mlp((behavior - self.behavior_mean) / self.behavior_std)
            graph = torch.cat((graph, evidence), -1)
        return self.head(graph)


def copy_common_initialization(static, enhanced):
    target = enhanced.state_dict()
    for name, value in static.state_dict().items():
        if name == 'head.0.weight':
            target[name][:, :value.shape[1]].copy_(value)
        else:
            target[name].copy_(value)
    enhanced.load_state_dict(target, strict=True)


def common_state(model):
    return {name: (value[:, :385] if name == 'head.0.weight' else value)
            for name, value in model.state_dict().items()
            if not name.startswith(('behavior_mlp.', 'behavior_mean', 'behavior_std'))}
