"""Label-free, tie-inclusive multiscale geometry and a zero-initialized adapter."""

import torch
from torch import nn


SCALES = (4, 8, 16)
FIELDS = tuple(f"k{k}_{name}" for k in SCALES for name in ("radius", "mean_distance", "distance_cv", "anisotropy"))


@torch.no_grad()
def local_geometry(xy, mask, eps=1e-12):
    """Population moments; all neighbors tied at the kth boundary are included."""
    xy = xy.double().masked_fill(~mask[..., None], 0)
    distances = torch.cdist(xy, xy, compute_mode="donot_use_mm_for_euclid_dist")
    n = xy.size(1)
    valid = mask[:, :, None] & mask[:, None, :]
    valid &= ~torch.eye(n, dtype=torch.bool, device=xy.device)[None]
    ell = distances.masked_fill(~valid, 0).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    ell = ell.clamp_min(eps)
    normalized = distances / ell[:, None, None]
    neighbor_dist = normalized.masked_fill(~valid, float("inf"))
    largest = normalized.masked_fill(~valid, 0).max(-1).values
    available = valid.sum(-1)
    dx = (xy[:, None, :, 0] - xy[:, :, None, 0]) / ell[:, None, None]
    dy = (xy[:, None, :, 1] - xy[:, :, None, 1]) / ell[:, None, None]
    features = []
    for k in SCALES:
        radius = neighbor_dist.topk(min(k, n), largest=False).values[..., -1]
        radius = torch.where(available < k, largest, radius)
        neighbors = valid & (normalized <= radius[..., None])
        count = neighbors.sum(-1).clamp_min(1)
        weight = neighbors.double()
        mean = (normalized * weight).sum(-1) / count
        variance = ((normalized - mean[..., None]).square() * weight).sum(-1) / count
        cv = variance.sqrt() / (mean + eps)
        # The center has zero displacement and participates in covariance.
        cluster_count = neighbors.sum(-1) + 1
        mx = (dx * weight).sum(-1) / cluster_count
        my = (dy * weight).sum(-1) / cluster_count
        xx = (dx.square() * weight).sum(-1) / cluster_count - mx.square()
        yy = (dy.square() * weight).sum(-1) / cluster_count - my.square()
        cross = (dx * dy * weight).sum(-1) / cluster_count - mx * my
        anisotropy = ((xx - yy).square() + 4 * cross.square()).sqrt() / ((xx + yy).clamp_min(0) + eps)
        features.extend((radius, mean, cv, anisotropy.clamp(0, 1)))
    return torch.stack(features, dim=-1).masked_fill(~mask[..., None], 0)


class GeometryResidual(nn.Module):
    def __init__(self, embedding_dim, mode="real"):
        super().__init__()
        if mode not in ("real", "zero"):
            raise ValueError("Geometry mode must be real or zero")
        self.mode = mode
        self.register_buffer("mean", torch.zeros(len(FIELDS)))
        self.register_buffer("std", torch.ones(len(FIELDS)))
        self.projection = nn.Sequential(nn.Linear(21, 64), nn.GELU(), nn.Linear(64, embedding_dim))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, node, mask, n, geometry=None):
        if geometry is None:
            geometry = local_geometry(node[..., :2], mask).float()
        geometry = (geometry.float() - self.mean) / self.std
        if self.mode == "zero":
            geometry = torch.zeros_like(geometry)
        geometry = geometry.masked_fill(~mask[..., None], 0)
        size = n.float().log1p()[:, None, None].expand(-1, node.size(1), 1) / 6.0
        return self.projection(torch.cat([node, size, geometry], -1)).masked_fill(~mask[..., None], 0)
