"""R55: supervised billed-edge differences are the only route to a decision."""

import torch
from torch import nn

from code.unified_selector.registry import K_CBITS, K_PROBLEM_DESC
from .pair_specialist import INPUT_FIELDS
from .V4Model import InstanceEncoder


def analytic_cost_difference(difference, distance, valid):
    return (difference.float().masked_fill(~valid, 0.) * distance.float().masked_fill(~valid, 0.)).sum(-1)


class SymmetricEdgeDecoder(nn.Module):
    def __init__(self, d=128, dropout=.1):
        super().__init__()
        self.node_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32), nn.GELU())
        self.condition_projection = nn.Sequential(nn.Linear(K_CBITS + K_PROBLEM_DESC + 1, 16), nn.GELU())
        self.edge_mlp = nn.Sequential(nn.Linear(3 * 32 + 2 + 16, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        # A small nonzero readout avoids an O(N^2) random cost offset at initialization.
        nn.init.normal_(self.edge_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(self, nodes, mask, xy, cbits, description, n):
        row, col = torch.triu_indices(nodes.shape[1], nodes.shape[1], offset=1, device=nodes.device)
        valid = mask[:, row] & mask[:, col]
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            distance = torch.linalg.vector_norm(xy[:, row].float() - xy[:, col].float(), dim=-1)
            distance = distance.masked_fill(~valid, 0.)
            scale = (distance.sum(-1) / valid.sum(-1).clamp_min(1)).clamp_min(1e-6)
        projected = self.node_projection(nodes)
        left, right = projected[:, row], projected[:, col]
        condition = self.condition_projection(torch.cat((cbits.float(), description.float(),
                                                         n.float().log1p()[:, None] / 6.), -1))
        feature = torch.cat((left + right, (left - right).abs(), left * right,
            distance[..., None], (distance / scale[:, None])[..., None],
            condition[:, None].expand(-1, len(row), -1)), -1)
        difference = self.edge_mlp(feature).squeeze(-1).float().masked_fill(~valid, 0.)
        delta = analytic_cost_difference(difference, distance, valid)
        return dict(cost_difference=delta, edge_difference=difference, edge_distance=distance,
                    edge_mask=valid, row=row, col=col, node_mask=mask)


class CostDifferenceSpecialist(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = dict(params, node_only=True)
        self.instance_encoder = InstanceEncoder(**self.params)
        self.edge_decoder = SymmetricEdgeDecoder(self.params['embedding_dim'], self.params['dropout'])

    def forward(self, batch, return_edges=False):
        clean = {k: batch[k] for k in INPUT_FIELDS if k in batch}
        nodes, mask = self.instance_encoder(clean)
        out = self.edge_decoder(nodes, mask, clean['node'][..., :2], clean['cbits'], clean['problem_desc'], clean['n'])
        return out if return_edges else out['cost_difference']


def difference_objective(out, target, cost_difference, oracle_cost):
    valid = out['edge_mask']
    truth = target[:, out['row'], out['col']].float()
    squared = (out['edge_difference'].float() - truth).square()
    groups = (valid & (truth > 0), valid & (truth < 0), valid & (truth == 0))
    counts = torch.stack([g.sum(-1) for g in groups], -1)
    group_mse = torch.stack([(squared * g).sum(-1) / count.clamp_min(1)
                            for g, count in zip(groups, counts.unbind(-1))], -1)
    active = counts > 0
    per_instance = (group_mse * active).sum(-1) / active.sum(-1).clamp_min(1)
    edge_loss = per_instance.mean()
    if not torch.isfinite(oracle_cost).all() or not (oracle_cost > 0).all():
        raise ValueError('Relative loss requires finite positive training-only Oracle costs')
    cost_loss = ((out['cost_difference'].float() - cost_difference.float()) / (.01 * oracle_cost.float())).square().mean()
    stats = dict(edge_loss=edge_loss, relative_cost_loss=cost_loss,
        positive_edge_mse=group_mse[:, 0].mean(), negative_edge_mse=group_mse[:, 1].mean(),
        zero_edge_mse=group_mse[:, 2].mean(),
        cost_difference_mae=(out['cost_difference'].float() - cost_difference.float()).abs().mean(),
        zero_prediction_edge_loss=((truth.square() * groups[0]).sum(-1) / counts[:, 0].clamp_min(1) +
                                   (truth.square() * groups[1]).sum(-1) / counts[:, 1].clamp_min(1))
                                   .div(active.sum(-1).clamp_min(1)).mean())
    return edge_loss + cost_loss, stats
