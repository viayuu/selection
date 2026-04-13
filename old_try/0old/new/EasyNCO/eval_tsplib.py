#!/usr/bin/env python3
"""
Evaluate EasyNCO HTSP on TSPLIB with *official* TSPLIB distance (EUC_2D / CEIL_2D rounding),
so the resulting gap (%) is comparable to `final/logs/tsplib_70/*`.

Why this script exists
----------------------
EasyNCO's standard `eval.py` computes `gap = 100 * ((score - optimal) / optimal)` where:
  - `optimal` is the TSPLIB official optimal tour length (integer, TSPLIB rounding), but
  - `score` for HTSP comes from `HTSPEnv` which uses continuous Euclidean distance on normalized coords.
Those two "cost" definitions are inconsistent, so the reported gap is not aligned with TSPLIB.

This script keeps HTSP inference as-is (using EasyNCO's `HTSPPolicy`), but:
  1) reads the original TSPLIB `.tsp` file (coords + EDGE_WEIGHT_TYPE),
  2) runs HTSP to obtain a tour (node permutation),
  3) re-computes tour length on the original TSPLIB coordinates using TSPLIB rounding,
  4) computes gap (%) w.r.t. the official optimal cost.

Default instance set
--------------------
By default it evaluates exactly the 70 instances used by the platform logs, derived from:
  `EasyNCO/data/datasets/tsplib/TSPlib_70instances.txt`.

Example
-------
CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval_tsplib.py \\
  --tsplib_root EasyNCO/data/datasets/tsplib \\
  --ckpt EasyNCO/pretrained/htsp_tsp1000.ckpt \\
  --val_m 8 --seeds 0,1,2,3,4,5,6,7,8,9 \\
  --device cuda:0 \\
  --out results/htsp_tsplib70_gap.tsv
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import sys
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow running as a script: `python EasyNCO/eval_tsplib.py ...`
# (so absolute imports like `import EasyNCO.*` work the same as `python -m EasyNCO.eval_tsplib ...`)
_this_dir = os.path.dirname(os.path.abspath(__file__))  # .../EasyNCO
_repo_root = os.path.dirname(_this_dir)  # .../
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from EasyNCO.neural_solvers.methods.htsp.policy import HTSPEnv, HTSPPolicy
from EasyNCO.neural_solvers.methods.htsp.utils_htsp import RLSolver
from EasyNCO.utils.utils import load_model


def _parse_int_list_arg(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [0]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_to_unit_board_isotropic(coords: torch.Tensor) -> torch.Tensor:
    """Isotropic normalization to [0,1]^2: (coords - min_xy) / max(range_x, range_y)."""
    min_xy = coords.min(dim=0).values
    max_xy = coords.max(dim=0).values
    scale = (max_xy - min_xy).max()
    return (coords - min_xy) / (scale + 1e-8)


def _augment_coords_pomo_style(coords: torch.Tensor, val_m: int) -> torch.Tensor:
    """
    POMO-style coordinate augmentation used across the platform.

    Input:
      coords: (1, N, 2), assumed in [0,1]
    Output:
      (val_m, N, 2)
    """
    assert coords.dim() == 3 and coords.size(0) == 1 and coords.size(2) == 2
    assert val_m in (1, 4, 6, 8), f"val_m must be one of 1/4/6/8, got {val_m}"

    bs, n, _ = coords.size()
    aug = coords.unsqueeze(1).repeat(1, val_m, 1, 1)  # (bs, val_m, N, 2)

    for i in range(val_m):
        if i == 1:
            aug[:, i, :, 0] = 1 - aug[:, i, :, 0]
        elif i == 2:
            aug[:, i, :, 1] = 1 - aug[:, i, :, 1]
        elif i == 3:
            aug[:, i, :, 0] = 1 - aug[:, i, :, 0]
            aug[:, i, :, 1] = 1 - aug[:, i, :, 1]
        elif i == 4:
            aug[:, i, :, 0] = aug[:, 0, :, 1]
            aug[:, i, :, 1] = aug[:, 0, :, 0]
        elif i == 5:
            aug[:, i, :, 0] = 1 - aug[:, 0, :, 1]
            aug[:, i, :, 1] = aug[:, 0, :, 0]
        elif i == 6:
            aug[:, i, :, 0] = aug[:, 0, :, 1]
            aug[:, i, :, 1] = 1 - aug[:, 0, :, 0]
        elif i == 7:
            aug[:, i, :, 0] = 1 - aug[:, 0, :, 1]
            aug[:, i, :, 1] = 1 - aug[:, 0, :, 0]

    return aug.view(bs * val_m, n, 2)


def _parse_kv_line(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s:
        return None
    if ":" in s:
        k, v = s.split(":", 1)
        return k.strip(), v.strip()
    parts = s.split()
    if len(parts) >= 2:
        return parts[0].strip(), " ".join(parts[1:]).strip()
    return None


def read_tsplib_tsp(filepath: str) -> Tuple[str, int, List[List[float]], str]:
    """
    Minimal TSPLIB .tsp parser for NODE_COORD_SECTION (2D) with EUC_2D/CEIL_2D.
    Returns: (name, dimension, locs_sorted_by_id, edge_weight_type)
    """
    name: Optional[str] = None
    dimension: Optional[int] = None
    edge_weight_type: Optional[str] = None
    coords_by_id: List[Tuple[int, float, float]] = []
    started = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if started:
                if line.startswith("EOF"):
                    break
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    idx = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                except ValueError:
                    continue
                coords_by_id.append((idx, x, y))
                continue

            if line.startswith("NODE_COORD_SECTION"):
                started = True
                continue

            kv = _parse_kv_line(line)
            if kv is None:
                continue
            k, v = kv
            k_upper = k.upper()
            if k_upper == "NAME":
                name = v.split()[0]
            elif k_upper == "DIMENSION":
                try:
                    dimension = int(v.split()[0])
                except ValueError:
                    pass
            elif k_upper == "EDGE_WEIGHT_TYPE":
                edge_weight_type = v.split()[0]

    if name is None or dimension is None or edge_weight_type is None:
        raise ValueError(f"Invalid TSPLIB header in {filepath}")
    if edge_weight_type not in {"EUC_2D", "CEIL_2D"}:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE={edge_weight_type} in {filepath}")
    if len(coords_by_id) != dimension:
        raise ValueError(
            f"NODE_COORD_SECTION length mismatch in {filepath}: got {len(coords_by_id)}, expected {dimension}"
        )

    coords_by_id.sort(key=lambda t: t[0])
    locs = [[x, y] for _, x, y in coords_by_id]
    return name, dimension, locs, edge_weight_type


def _tsplib_edge_weight(raw_dist: float, edge_weight_type: str) -> float:
    if edge_weight_type == "CEIL_2D":
        return float(math.ceil(raw_dist))
    if edge_weight_type == "EUC_2D":
        return float(math.floor(raw_dist + 0.5))
    return raw_dist


def tsplib_tour_cost(
    locs: Sequence[Sequence[float]],
    tour: Sequence[int],
    edge_weight_type: str,
) -> float:
    n = len(tour)
    if n <= 1:
        raise ValueError("tour too short")
    if len(set(tour)) != n:
        raise ValueError("tour has duplicate nodes")
    total = 0.0
    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])
        dx = float(locs[a][0]) - float(locs[b][0])
        dy = float(locs[a][1]) - float(locs[b][1])
        dist = math.sqrt(dx * dx + dy * dy)
        total += _tsplib_edge_weight(dist, edge_weight_type)
    return total


def load_optimal_costs_from_70instances_txt(path: str) -> Dict[str, float]:
    """
    Parse `TSPlib_70instances.txt` (one python list literal per line):
      [name, optimal, x1, y1, x2, y2, ...]
    Returns: {name: optimal_cost}
    """
    costs: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = ast.literal_eval(line)
            except Exception:
                continue
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            name = str(rec[0])
            try:
                optimal = float(rec[1])
            except Exception:
                continue
            costs[name] = optimal
    if len(costs) != 70:
        raise ValueError(f"Expected 70 optimal-cost entries, got {len(costs)} from {path}")
    return costs


def load_optimal_costs(optimal_txt: str, optimal_json: str = "") -> Dict[str, float]:
    costs = load_optimal_costs_from_70instances_txt(optimal_txt)
    if not optimal_json:
        return costs
    import json

    with open(optimal_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        try:
            costs[str(k)] = float(v)
        except Exception:
            continue
    return costs


def _find_tsplib_files(root: str) -> Dict[str, str]:
    """
    Recursively map basename -> filepath for `.tsp` files.
    If duplicates exist, the first one discovered is kept.
    """
    out: Dict[str, str] = {}
    for r, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".tsp"):
                continue
            base = fn[:-4]
            out.setdefault(base, os.path.join(r, fn))
    return out


class _NoOpFragmentBuffer:
    """Eval-only: HTSP's RLSolver writes fragments into a replay buffer for training; not needed here."""

    def update_buffer(self, fragments: torch.Tensor) -> None:
        return None


class _EvalRLSolver(RLSolver):
    """
    Wrapper for TSPLIB eval:
    - avoids shape mismatch in `FragmentBuffer.update_buffer(...)` by using a no-op buffer;
    - ensures `group_size <= fragment_len` to prevent shape errors in `Policy_SHPP` when fragment is shorter than `frag_len`.
    """

    def __init__(self, model: torch.nn.Module, max_sample_size: int) -> None:
        super().__init__(model, sample_size=max_sample_size)
        self._max_sample_size = max(2, int(max_sample_size))
        self._noop_buffer = _NoOpFragmentBuffer()

    @staticmethod
    def _repair_local_path(raw_path: np.ndarray, n: int) -> List[int]:
        """
        Ensure a valid permutation of [0..n-1] that starts with 0 and ends with n-1.
        This prevents HTSP from crashing in `LargeState.move_to()` when the lower-level returns
        an order that doesn't respect the intended fragment endpoints.
        """
        flat = np.asarray(raw_path).reshape(-1).tolist()
        seq: List[int] = []
        seen = set()
        for x in flat:
            try:
                xi = int(x)
            except Exception:
                continue
            if 0 <= xi < n and xi not in seen:
                seen.add(xi)
                seq.append(xi)

        if 0 not in seen:
            seq.insert(0, 0)
            seen.add(0)
        # Rotate so 0 is at the front (treat seq as cyclic if needed).
        idx0 = seq.index(0)
        seq = seq[idx0:] + seq[:idx0]

        if (n - 1) in seen and seq[-1] != n - 1:
            # Try reverse orientation first.
            rev = list(reversed(seq))
            idx0 = rev.index(0)
            rev = rev[idx0:] + rev[:idx0]
            if rev[-1] == n - 1:
                seq = rev
            else:
                # Fallback: move n-1 to the end while preserving the rest order.
                seq = [x for x in seq if x != n - 1] + [n - 1]

        if (n - 1) not in seen:
            seq.append(n - 1)
            seen.add(n - 1)

        mid = [x for x in seq if x not in (0, n - 1)]
        mid_set = set(mid)
        missing = [i for i in range(1, n - 1) if i not in mid_set]
        final = [0] + mid + missing + [n - 1]
        # Safety: ensure permutation length exactly n.
        if len(final) != n or len(set(final)) != n:
            final = list(range(n))
        return final

    def solve(self, data: torch.Tensor, fragment: torch.Tensor, frag_buffer: _NoOpFragmentBuffer):  # type: ignore[override]
        # This is a copy of `RLSolver.solve` with eval-only safety repairs.
        device = data.device
        bsz = int(fragment.size(0))
        n = int(fragment.size(1))
        self._sample_size = max(2, min(self._max_sample_size, n))

        node_pos = torch.gather(input=data, index=fragment[..., None].expand(-1, -1, data.shape[-1]), dim=1)
        x = node_pos.to(device=device)

        x1_min = x[..., 0].min(dim=-1, keepdim=True)[0]
        x2_min = x[..., 1].min(dim=-1, keepdim=True)[0]
        x1_max = x[..., 0].max(dim=-1, keepdim=True)[0]
        x2_max = x[..., 1].max(dim=-1, keepdim=True)[0]
        s = 0.9 / torch.maximum(x1_max - x1_min, x2_max - x2_min)
        x_new = torch.empty_like(x)
        x_new[..., 0] = s * (x[..., 0] - x1_min) + 0.05
        x_new[..., 1] = s * (x[..., 1] - x2_min) + 0.05

        source_nodes = torch.tensor([[0]], device=device).expand(bsz, 1)
        target_nodes = torch.tensor([[n - 1]], device=device).expand(bsz, 1)

        lengths, paths = self.low_level_model(
            x_new.float(),
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            val_type="x8Aug_2Traj",
            group_size=self._sample_size,
        )

        # Keep the original length scaling (not used by HTSPEnv.step, but keep it consistent).
        lengths = (lengths / s.squeeze()).detach().cpu().numpy()
        paths_np = paths.detach().cpu().numpy()
        fragment_np = fragment.detach().cpu().numpy()

        if bsz == 1:
            paths_np = [paths_np]

        out_paths: List[List[int]] = []
        for i, raw_path in enumerate(paths_np):
            local_path = self._repair_local_path(raw_path, n)
            out_paths.append([int(fragment_np[i][j]) for j in local_path])

        return out_paths, lengths


@torch.no_grad()
def htsp_infer_tour(
    policy: HTSPPolicy,
    coords_norm_01: torch.Tensor,  # (1, N, 2) in [0,1]
    *,
    device: torch.device,
    k: int,
    frag_len: int,
    max_new_nodes: int,
    max_improvement_step: int,
    low_level_buffer_size: int,
    low_level_sample_size: int,
) -> List[int]:
    n = int(coords_norm_01.size(1))
    frag_len_req = int(frag_len)
    frag_len_eff = min(frag_len_req, n)
    max_new_nodes_eff = min(int(max_new_nodes), frag_len_eff, n)

    # For small instances (n <= frag_len), force full kNN to avoid disconnected kNN components,
    # otherwise `get_fragment_knn()` may fail to collect enough new nodes and produce invalid fragments
    # whose endpoints are not in the current tour.
    if n <= frag_len_req:
        k_eff = n - 1
    else:
        k_eff = min(int(k), n - 1)
    k_eff = max(2, k_eff)

    # Eval-only safety:
    # - some TSPLIB instances produce fragments shorter than `frag_len` (due to directed kNN reachability),
    #   which can break both the replay-buffer write and the lower-level group_size.
    solver = policy.lower_solver
    if isinstance(solver, RLSolver):
        solver = _EvalRLSolver(solver.low_level_model, max_sample_size=int(low_level_sample_size))

    env = HTSPEnv(
        k=k_eff,
        frag_len=frag_len_eff,
        max_new_nodes=max_new_nodes_eff,
        max_improvement_step=max_improvement_step,
        auto_reset=False,
        no_depot=False,
    )

    states = env.reset(coords_norm_01.to(device))
    while not env.done:
        graph_feature = policy.encoder(states)
        actions = policy.actor(graph_feature).detach()
        states, _, _, _ = env.step(actions, solver=solver, frag_buffer=_NoOpFragmentBuffer())

    tour = list(env.states[0].current_tour)
    if len(tour) != n:
        raise RuntimeError(f"HTSP returned tour length {len(tour)} != n={n}")
    return tour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsplib_root",
        type=str,
        default="EasyNCO/data/datasets/tsplib",
        help="Root directory that contains TSPLIB `.tsp` files (will be searched recursively).",
    )
    parser.add_argument(
        "--optimal_txt",
        type=str,
        default="EasyNCO/data/datasets/tsplib/TSPlib_70instances.txt",
        help="Path to `TSPlib_70instances.txt` (used for default 70 instance names + optimal costs).",
    )
    parser.add_argument(
        "--optimal_json",
        type=str,
        default="",
        help="Optional JSON override mapping: {\"berlin52\": 7542, ...}.",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="",
        help="Comma-separated instance names. Empty => evaluate the 70 instances in `--optimal_txt`.",
    )
    parser.add_argument("--min_nodes", type=int, default=1)
    parser.add_argument("--max_nodes", type=int, default=1_000_000)

    # HTSP model/checkpoint
    parser.add_argument(
        "--ckpt",
        type=str,
        default="EasyNCO/pretrained/htsp_tsp1000.ckpt",
        help="HTSP checkpoint (EasyNCO format).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )

    # Evaluation controls
    parser.add_argument(
        "--val_m",
        type=int,
        default=1,
        choices=[1, 4, 6, 8],
        help="Augmentation multiplicity (POMO-style). no_aug uses the first augment; aug uses min over augments.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds, e.g. 0 or 0,1,2,3,4,5,6,7,8,9",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="gap",
        choices=["gap", "cost", "both"],
        help="Output gap/cost/both. Gap is computed from TSPLIB official cost.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output TSV path. Empty => print to stdout only.",
    )

    # HTSP hyperparameters (match `EasyNCO/settings/htsp_settings.yaml` by default)
    parser.add_argument("--k", type=int, default=40, help="kNN size in HTSPEnv.")
    parser.add_argument("--frag_len", type=int, default=200, help="Fragment length for the lower solver.")
    parser.add_argument("--max_new_nodes", type=int, default=190, help="HTSPEnv max_new_nodes.")
    parser.add_argument("--max_improvement_step", type=int, default=0, help="HTSPEnv max_improvement_step.")
    parser.add_argument("--low_level_buffer_size", type=int, default=10000)
    parser.add_argument("--low_level_sample_size", type=int, default=200, help="Lower solver group_size (will be clipped per instance).")

    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = _parse_int_list_arg(args.seeds)

    optimal_costs = load_optimal_costs(args.optimal_txt, args.optimal_json)
    tsplib_files = _find_tsplib_files(args.tsplib_root)

    if args.instances.strip():
        instance_names = [x.strip() for x in args.instances.split(",") if x.strip()]
    else:
        instance_names = sorted(optimal_costs.keys())

    # Build HTSP policy with the same architecture as `EasyNCO/settings/htsp_settings.yaml`.
    model_kwargs = {
        "Encoder": {"input_dim": 8, "embedding_dim": 128},
        "Actor": {"state_dim": 128, "mid_dim": 128, "action_dim": 2, "init_a_std_log": -1},
        "Critic": {"state_dim": 128, "mid_dim": 512, "_action_dim": 2},
        "low_level_type": "pomo",
        "low_level": {
            "low_level_buffer_size": int(args.low_level_buffer_size),
            "frag_len": int(args.frag_len),
            "node_dim": 2,
            "group_size": int(args.low_level_sample_size),
            "update_time": None,
            "lower_path": "/pretrained/htsp_lower_tsp.ckpt",
        },
        "low_env": {
            "k": int(args.k),
            "frag_len": int(args.frag_len),
            "max_new_nodes": int(args.max_new_nodes),
            "max_improvement_step": int(args.max_improvement_step),
            "auto_reset": False,
            "no_depot": False,
        },
        # The following are required by HTSPPolicy.__init__ but are not used for this eval script.
        "target_step": 64,
        "reward_scale": 1.0,
        "gamma": 1.0,
        "experience_items": 6,
        "env_num": 64,
        "grad_method": "norm",
        "lambda_GAE": 0.9,
        "data_augment": False,
        "eval": {"time_limit": 100.0, "batch_size": 16, "improvement_step": 0},
    }

    policy = HTSPPolicy(**model_kwargs)

    # Match EasyNCO's load_model() device convention: list for GPU, 1 for CPU.
    if device.type == "cuda":
        cuda_idx = device.index if device.index is not None else 0
        policy = load_model(policy, args.ckpt, device=[int(cuda_idx)], model_name="htsp")
    else:
        policy = load_model(policy, args.ckpt, device=1, model_name="htsp")

    policy.to(device)
    policy.eval()

    lines: List[str] = []
    if args.report == "both":
        header_fields = [
            "name",
            "n",
            "edge_weight_type",
            "optimal",
            "k",
            "frag_len",
            "max_new_nodes",
            "max_improvement_step",
            "val_m",
            "seeds",
            "no_aug_cost_best",
            "no_aug_cost_avg",
            "aug_cost_best",
            "aug_cost_avg",
            "no_aug_gap_best(%)",
            "no_aug_gap_avg(%)",
            "aug_gap_best(%)",
            "aug_gap_avg(%)",
            "time_sec_total",
        ]
    elif args.report == "cost":
        header_fields = [
            "name",
            "n",
            "edge_weight_type",
            "optimal",
            "k",
            "frag_len",
            "max_new_nodes",
            "max_improvement_step",
            "val_m",
            "seeds",
            "no_aug_cost_best",
            "no_aug_cost_avg",
            "aug_cost_best",
            "aug_cost_avg",
            "time_sec_total",
        ]
    else:
        header_fields = [
            "name",
            "n",
            "edge_weight_type",
            "optimal",
            "k",
            "frag_len",
            "max_new_nodes",
            "max_improvement_step",
            "val_m",
            "seeds",
            "no_aug_gap_best(%)",
            "no_aug_gap_avg(%)",
            "aug_gap_best(%)",
            "aug_gap_avg(%)",
            "time_sec_total",
        ]
    header = "\t".join(header_fields)
    print(header)
    lines.append(header)

    out_f = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")
        out_f.write(header + "\n")
        out_f.flush()

    t_all0 = time.time()
    try:
        for name in instance_names:
            tsp_path = tsplib_files.get(name)
            if tsp_path is None:
                print(f"[skip] missing .tsp for {name}")
                continue

            try:
                file_name, n, locs, edge_weight_type = read_tsplib_tsp(tsp_path)
            except Exception as e:
                print(f"[skip] {tsp_path}: {e}")
                continue
            name = file_name

            if n < args.min_nodes or n > args.max_nodes:
                print(f"[skip] {name} n={n} outside [{args.min_nodes},{args.max_nodes}]")
                continue

            optimal = float(optimal_costs.get(name, 0.0))
            if optimal <= 0:
                print(f"[skip] {name}: missing optimal cost")
                continue

            frag_len_req = int(args.frag_len)
            frag_len_eff = min(frag_len_req, n)
            max_new_nodes_eff = min(int(args.max_new_nodes), frag_len_eff, n)
            k_eff = (n - 1) if n <= frag_len_req else min(int(args.k), n - 1)
            k_eff = max(2, k_eff)

            coords_orig = torch.tensor(locs, dtype=torch.float32)  # (N,2) on CPU
            coords_norm = _normalize_to_unit_board_isotropic(coords_orig).unsqueeze(0)  # (1,N,2)
            coords_aug = _augment_coords_pomo_style(coords_norm, int(args.val_m))  # (val_m, N, 2)

            no_aug_costs: List[float] = []
            aug_costs: List[float] = []

            t0 = time.time()
            for seed in seeds:
                _set_all_seeds(seed)

                per_aug_costs: List[float] = []
                for aug_idx in range(int(args.val_m)):
                    tour = htsp_infer_tour(
                        policy,
                        coords_aug[aug_idx : aug_idx + 1],
                        device=device,
                        k=k_eff,
                        frag_len=frag_len_eff,
                        max_new_nodes=max_new_nodes_eff,
                        max_improvement_step=int(args.max_improvement_step),
                        low_level_buffer_size=int(args.low_level_buffer_size),
                        low_level_sample_size=int(args.low_level_sample_size),
                    )
                    cost = tsplib_tour_cost(locs, tour, edge_weight_type)
                    per_aug_costs.append(float(cost))

                no_aug_costs.append(per_aug_costs[0])
                aug_costs.append(min(per_aug_costs))

            time_used = time.time() - t0

            def _avg(xs: Sequence[float]) -> float:
                return float(sum(xs) / max(len(xs), 1))

            no_aug_best = float(min(no_aug_costs))
            no_aug_avg = _avg(no_aug_costs)
            aug_best = float(min(aug_costs))
            aug_avg = _avg(aug_costs)

            def _gap(cost: float) -> float:
                return 100.0 * ((cost - optimal) / optimal)

            if args.report == "both":
                row_fields = [
                    name,
                    str(n),
                    edge_weight_type,
                    str(int(optimal)),
                    str(k_eff),
                    str(frag_len_eff),
                    str(max_new_nodes_eff),
                    str(args.max_improvement_step),
                    str(args.val_m),
                    ",".join(str(s) for s in seeds),
                    f"{no_aug_best:.6f}",
                    f"{no_aug_avg:.6f}",
                    f"{aug_best:.6f}",
                    f"{aug_avg:.6f}",
                    f"{_gap(no_aug_best):.6f}",
                    f"{_gap(no_aug_avg):.6f}",
                    f"{_gap(aug_best):.6f}",
                    f"{_gap(aug_avg):.6f}",
                    f"{time_used:.3f}",
                ]
            elif args.report == "cost":
                row_fields = [
                    name,
                    str(n),
                    edge_weight_type,
                    str(int(optimal)),
                    str(k_eff),
                    str(frag_len_eff),
                    str(max_new_nodes_eff),
                    str(args.max_improvement_step),
                    str(args.val_m),
                    ",".join(str(s) for s in seeds),
                    f"{no_aug_best:.6f}",
                    f"{no_aug_avg:.6f}",
                    f"{aug_best:.6f}",
                    f"{aug_avg:.6f}",
                    f"{time_used:.3f}",
                ]
            else:
                row_fields = [
                    name,
                    str(n),
                    edge_weight_type,
                    str(int(optimal)),
                    str(k_eff),
                    str(frag_len_eff),
                    str(max_new_nodes_eff),
                    str(args.max_improvement_step),
                    str(args.val_m),
                    ",".join(str(s) for s in seeds),
                    f"{_gap(no_aug_best):.6f}",
                    f"{_gap(no_aug_avg):.6f}",
                    f"{_gap(aug_best):.6f}",
                    f"{_gap(aug_avg):.6f}",
                    f"{time_used:.3f}",
                ]

            line = "\t".join(row_fields)
            print(line)
            lines.append(line)
            if out_f is not None:
                out_f.write(line + "\n")
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()

    total_time = time.time() - t_all0
    print(f"[done] total_time_sec={total_time:.3f}")

    if args.out and not out_f:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
