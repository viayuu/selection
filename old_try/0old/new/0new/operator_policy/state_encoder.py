from __future__ import annotations

import math

import torch
import torch.nn as nn



def _cyclic_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device) / max(length, 1)
    half = dim // 2
    freq = torch.arange(1, half + 1, dtype=torch.float32, device=device)
    angles = 2.0 * math.pi * position[:, None] * freq[None, :]
    pe = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if pe.size(1) < dim:
        pe = torch.cat([pe, torch.zeros((length, dim - pe.size(1)), device=device)], dim=1)
    return pe



def _tour_node_features(coords: torch.Tensor, tour: torch.Tensor, cpe_dim: int) -> torch.Tensor:
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    prev_ordered = ordered.roll(shifts=1, dims=1)
    next_ordered = ordered.roll(shifts=-1, dims=1)
    rel_prev = ordered - prev_ordered
    rel_next = next_ordered - ordered
    dist_prev = rel_prev.norm(p=2, dim=-1, keepdim=True)
    dist_next = rel_next.norm(p=2, dim=-1, keepdim=True)
    pos = torch.arange(tour.size(1), dtype=coords.dtype, device=coords.device)
    pos = (pos / max(tour.size(1), 1)).view(1, -1, 1).expand(coords.size(0), -1, -1)
    cpe = _cyclic_positional_encoding(tour.size(1), cpe_dim, coords.device).unsqueeze(0).expand(coords.size(0), -1, -1)
    return torch.cat(
        [ordered, prev_ordered, next_ordered, rel_prev, rel_next, dist_prev, dist_next, pos, cpe],
        dim=-1,
    )



def _tour_edge_stats(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    rolled = ordered.roll(shifts=-1, dims=1)
    edges = (ordered - rolled).norm(p=2, dim=-1)
    mean = edges.mean(dim=1, keepdim=True)
    std = edges.std(dim=1, keepdim=True)
    minv = edges.min(dim=1, keepdim=True).values
    maxv = edges.max(dim=1, keepdim=True).values
    return torch.cat([mean, std, minv, maxv], dim=1)


class GraphPoolEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.output_proj(pooled)


class InitStateEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.graph_encoder = GraphPoolEncoder(2, hidden_dim, num_heads, num_layers, dropout)
        self.scale_proj = nn.Linear(1, hidden_dim)
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        graph_emb = self.graph_encoder(coords)
        scale = torch.full((coords.size(0), 1), float(coords.size(1)), dtype=coords.dtype, device=coords.device)
        scale_emb = self.scale_proj(scale)
        return self.final_proj(torch.cat([graph_emb, scale_emb], dim=1))


class OperatorStateEncoder(nn.Module):
    def __init__(
        self,
        num_initializers: int,
        num_operators: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        cpe_dim: int = 32,
        dropout: float = 0.0,
        action_embed_dim: int = 32,
    ):
        super().__init__()
        self.cpe_dim = cpe_dim
        self.static_encoder = GraphPoolEncoder(2, hidden_dim, num_heads, num_layers, dropout)
        self.solution_encoder = GraphPoolEncoder(13 + cpe_dim, hidden_dim, num_heads, num_layers, dropout)
        self.init_embedding = nn.Embedding(num_initializers, action_embed_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + action_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        tour: torch.Tensor,
        current_length: torch.Tensor,
        best_length: torch.Tensor,
        init_length: torch.Tensor,
        init_action: torch.Tensor,
        prev_operator: torch.Tensor,
        step_index: torch.Tensor,
        rollout_steps: int,
        stagnation: torch.Tensor,
    ) -> torch.Tensor:
        del current_length, best_length, init_length, prev_operator, step_index, rollout_steps, stagnation
        static_emb = self.static_encoder(coords)
        solution_feats = _tour_node_features(coords, tour, self.cpe_dim)
        solution_emb = self.solution_encoder(solution_feats)
        edge_stats = _tour_edge_stats(coords, tour)
        global_emb = self.global_proj(edge_stats)
        init_emb = self.init_embedding(init_action)
        return self.final_proj(torch.cat([static_emb, solution_emb, global_emb, init_emb], dim=1))
