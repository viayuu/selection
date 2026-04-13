from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature extraction config.

    Default uses the paper-style encoder implemented locally in `my/paper_encoder.py`
    (Naive_Encoder / Encoder_h) to match the original selector architecture.
    """

    extractor: Literal["paper", "simple"] = "paper"

    # Paper-style encoder params (defaults match `config_TSP.yml:model_params`).
    paper_pooling: bool = True
    paper_downsample_ratio: float = 0.8
    paper_embedding_dim: int = 128
    paper_encoder_layer_num: int = 2
    paper_block_num: int = 2
    paper_head_num: int = 8
    paper_qkv_dim: int = 16
    paper_ff_hidden_dim: int = 512
    paper_norm: str = "rezero"
    paper_include_scale: bool = True  # match `Selection_model.acquire_feature(...)`

    # Simple handcrafted fallback (only used when extractor="simple")
    use_centroid_stats: bool = True
    use_bbox_stats: bool = True

    # Gate2 condition features (data + sol0)
    use_length0: bool = True  # only for Gate2
    use_tour_stats: bool = True  # only for Gate2


def build_paper_tsp_encoder(cfg: FeatureConfig):
    """
    Build the paper-style instance encoder (local copy in `my/paper_encoder.py`).

    Returns:
        - encoder (nn.Module) if cfg.extractor == "paper"
        - None otherwise
    """
    if cfg.extractor != "paper":
        return None

    # Local import: only needed when running training (requires torch).
    from my import paper_encoder as paper_model

    model_params = {
        "problem_type": "TSP",
        "pooling": bool(cfg.paper_pooling),
        "downsample_ratio": float(cfg.paper_downsample_ratio),
        "embedding_dim": int(cfg.paper_embedding_dim),
        "encoder_layer_num": int(cfg.paper_encoder_layer_num),
        "block_num": int(cfg.paper_block_num),
        "head_num": int(cfg.paper_head_num),
        "qkv_dim": int(cfg.paper_qkv_dim),
        "ff_hidden_dim": int(cfg.paper_ff_hidden_dim),
        "norm": str(cfg.paper_norm),
    }

    if cfg.paper_pooling:
        return paper_model.Encoder_h(**model_params)
    return paper_model.Naive_Encoder(**model_params)


def _tsp_default_mask(coords):
    import torch

    return torch.zeros((coords.size(0), coords.size(1)), device=coords.device, dtype=coords.dtype)


def tsp_instance_features(coords, *, cfg: FeatureConfig, encoder=None, mask=None):
    """
    coords: Tensor[batch, n, 2]
    returns: Tensor[batch, d]
    """
    # Local import so `import my.*` doesn't require torch unless used.
    import torch

    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError(f"Expected coords shape [B,N,2], got {tuple(coords.shape)}")

    if cfg.extractor == "paper":
        if encoder is None:
            raise ValueError("cfg.extractor='paper' requires an encoder; call build_paper_tsp_encoder(cfg) first.")
        if mask is None:
            mask = _tsp_default_mask(coords)

        graph_emb = encoder(coords, mask)  # matches `Selection_model.encoder(points, mask)`
        feats = [graph_emb]
        if cfg.paper_include_scale:
            scale = torch.full((coords.size(0), 1), float(coords.size(1)), device=coords.device, dtype=coords.dtype)
            feats.append(scale)
        return torch.cat(feats, dim=1)

    if cfg.extractor != "simple":
        raise ValueError(f"Unknown feature extractor: {cfg.extractor}")

    feats = []

    if cfg.use_centroid_stats:
        mean = coords.mean(dim=1)  # [B,2]
        std = coords.std(dim=1).clamp_min(1e-6)  # [B,2]
        feats.extend([mean, std])

        centroid = mean[:, None, :]  # [B,1,2]
        dist = ((coords - centroid) ** 2).sum(dim=-1).sqrt()  # [B,N]
        feats.append(dist.mean(dim=1, keepdim=True))  # [B,1]
        feats.append(dist.std(dim=1, keepdim=True).clamp_min(1e-6))  # [B,1]

    if cfg.use_bbox_stats:
        min_xy = coords.min(dim=1).values  # [B,2]
        max_xy = coords.max(dim=1).values  # [B,2]
        span = (max_xy - min_xy).clamp_min(1e-6)  # [B,2]
        feats.append(span)
        feats.append((span[:, 0:1] * span[:, 1:2]).clamp_min(1e-6))  # area [B,1]

    if not feats:
        return torch.zeros((coords.size(0), 0), device=coords.device, dtype=coords.dtype)

    return torch.cat(feats, dim=1)


def tsp_tour_features(coords, tour):
    """
    coords: Tensor[batch, n, 2]
    tour:   Tensor[batch, n]  (permutation of 0..n-1)
    returns: Tensor[batch, d]
    """
    import torch

    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError(f"Expected coords shape [B,N,2], got {tuple(coords.shape)}")
    if tour.ndim != 2 or tour.size(0) != coords.size(0) or tour.size(1) != coords.size(1):
        raise ValueError(f"Expected tour shape [B,N], got {tuple(tour.shape)}")

    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))  # [B,N,2]
    rolled = ordered.roll(shifts=-1, dims=1)
    seg = ((ordered - rolled) ** 2).sum(dim=-1).sqrt()  # [B,N]

    mean = seg.mean(dim=1, keepdim=True)
    std = seg.std(dim=1, keepdim=True).clamp_min(1e-6)
    minv = seg.min(dim=1, keepdim=True).values
    maxv = seg.max(dim=1, keepdim=True).values

    return torch.cat([mean, std, minv, maxv], dim=1)


def tsp_gate2_features(coords, tour0, tour0_length, *, cfg: FeatureConfig, encoder=None, mask=None):
    """
    Gate2 condition: data + sol0.
    For the prototype we use (length0 + simple tour statistics) as condition features.
    """
    import torch

    base = tsp_instance_features(coords, cfg=cfg, encoder=encoder, mask=mask)

    extras = []
    if cfg.use_length0:
        if tour0_length.ndim != 1 or tour0_length.size(0) != coords.size(0):
            raise ValueError(
                f"Expected length0 shape [B], got {tuple(tour0_length.shape)}"
            )
        extras.append(tour0_length[:, None])

    if cfg.use_tour_stats:
        extras.append(tsp_tour_features(coords, tour0))

    if not extras:
        return base

    return torch.cat([base, *extras], dim=1)
