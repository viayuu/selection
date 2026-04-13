from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateMLPConfig:
    in_dim: int
    out_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)
    dropout: float = 0.0


def build_mlp(cfg: GateMLPConfig):
    import torch.nn as nn

    layers = []
    prev = cfg.in_dim
    for hid in cfg.hidden_dims:
        layers.append(nn.Linear(prev, hid))
        layers.append(nn.GELU())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(p=cfg.dropout))
        prev = hid
    layers.append(nn.Linear(prev, cfg.out_dim))
    return nn.Sequential(*layers)


def build_value_mlp(in_dim: int, hidden_dims: tuple[int, ...] = (128, 128), dropout: float = 0.0):
    """
    Value function approximator V(s) -> R.
    """
    import torch.nn as nn

    layers = []
    prev = in_dim
    for hid in hidden_dims:
        layers.append(nn.Linear(prev, hid))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        prev = hid
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class TwoGateAC:
    """
    Gate1: pi(a1 | data)
    Gate2: pi(a2 | data, sol0)  (here conditioned by length0 in features)
    """

    def __init__(self, *, gate1, gate2, value1=None, value2=None, encoder=None):
        import torch.nn as nn

        super().__init__()
        # Keep it an nn.Module for easier checkpointing.
        self._module = nn.Module()
        self._module.encoder = encoder
        self._module.gate1 = gate1
        self._module.gate2 = gate2
        self._module.value1 = value1
        self._module.value2 = value2

    @property
    def encoder(self):
        return getattr(self._module, "encoder", None)

    @property
    def gate1(self):
        return self._module.gate1

    @property
    def gate2(self):
        return self._module.gate2

    @property
    def value(self):
        # Backward-compatible alias: historical code used a single `value` critic.
        return getattr(self._module, "value1", None)

    @property
    def value1(self):
        return getattr(self._module, "value1", None)

    @property
    def value2(self):
        return getattr(self._module, "value2", None)

    def to(self, device):
        self._module.to(device)
        return self

    def train(self):
        self._module.train()

    def eval(self):
        self._module.eval()

    def parameters(self):
        yield from self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        return self._module.load_state_dict(state_dict)
