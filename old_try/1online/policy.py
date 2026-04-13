from __future__ import annotations

import torch
import torch.nn as nn

from state_encoder import InitStateEncoder, OperatorStateEncoder


def _build_head(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (128, 128), dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hid in hidden_dims:
        layers.append(nn.Linear(prev, hid))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hid
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class TSPSelectorPolicy(nn.Module):
    def __init__(
        self,
        num_initializers: int,
        num_operators: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        cpe_dim: int = 32,
        dropout: float = 0.0,
        head_hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        self.init_encoder = InitStateEncoder(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.op_encoder = OperatorStateEncoder(
            num_initializers=num_initializers,
            num_operators=num_operators,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            cpe_dim=cpe_dim,
            dropout=dropout,
        )
        self.init_head = _build_head(hidden_dim, num_initializers, head_hidden_dims, dropout)
        self.op_head = _build_head(hidden_dim, num_operators, head_hidden_dims, dropout)

    def init_logits(self, coords: torch.Tensor) -> torch.Tensor:
        return self.init_head(self.init_encoder(coords))

    def operator_logits(
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
        state_emb = self.op_encoder(
            coords=coords,
            tour=tour,
            current_length=current_length,
            best_length=best_length,
            init_length=init_length,
            init_action=init_action,
            prev_operator=prev_operator,
            step_index=step_index,
            rollout_steps=rollout_steps,
            stagnation=stagnation,
        )
        return self.op_head(state_emb)

