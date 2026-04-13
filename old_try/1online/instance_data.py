from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from easynco_bootstrap import ensure_local_easynco

ensure_local_easynco()

from EasyNCO.data.LIBUtils import TSPLIBReader
from EasyNCO.data.data_utils import normalize_to_unit_board


@dataclass(frozen=True)
class TSPInstance:
    name: str
    path: str
    problem_size: int
    coords: torch.Tensor
    edge_weight_type: str | None = None



def _match_instance_name(candidate_path: Path, target_name: str) -> bool:
    try:
        name, _, _, _ = TSPLIBReader(str(candidate_path))
    except Exception:
        return False
    if name is not None and str(name).lower() == target_name.lower():
        return True
    return candidate_path.stem.lower() == target_name.lower()



def resolve_instance_path(*, tsplib_root: str | None, instance_name: str | None, instance_path: str | None) -> str:
    if instance_path is not None:
        path = Path(instance_path)
        if not path.is_file():
            raise FileNotFoundError(f"TSPLIB instance file not found: {path}")
        return str(path)

    if tsplib_root is None or instance_name is None:
        raise ValueError("Either instance_path must be set, or tsplib_root + instance_name must both be provided")

    root = Path(tsplib_root)
    if not root.is_dir():
        raise FileNotFoundError(f"TSPLIB root directory not found: {root}")

    matches: list[Path] = []
    for current_root, _dirs, files in os.walk(root):
        for file in files:
            if not file.lower().endswith('.tsp'):
                continue
            candidate = Path(current_root) / file
            if _match_instance_name(candidate, instance_name):
                matches.append(candidate)

    if not matches:
        raise FileNotFoundError(f"Could not find TSPLIB instance '{instance_name}' under {root}")
    if len(matches) > 1:
        joined = ', '.join(str(p) for p in matches)
        raise RuntimeError(f"Multiple TSPLIB files matched '{instance_name}': {joined}")
    return str(matches[0])



def load_tsplib_instance(*, tsplib_root: str | None = None, instance_name: str | None = None, instance_path: str | None = None, device: str = 'cpu') -> TSPInstance:
    resolved = resolve_instance_path(tsplib_root=tsplib_root, instance_name=instance_name, instance_path=instance_path)
    name, dimension, locs, edge_weight_type = TSPLIBReader(resolved)
    if name is None or dimension is None or locs is None:
        raise RuntimeError(f"Unsupported or invalid TSPLIB file: {resolved}")

    coords_raw = torch.tensor(locs, dtype=torch.float32)
    coords_norm, _ = normalize_to_unit_board(coords_raw)
    coords_norm = coords_norm.to(device=device)
    return TSPInstance(
        name=str(name),
        path=resolved,
        problem_size=int(dimension),
        coords=coords_norm,
        edge_weight_type=(str(edge_weight_type) if edge_weight_type is not None else None),
    )
