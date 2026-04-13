#!/usr/bin/env python3
"""
Evaluate DACT on TSPLIB with *official* TSPLIB distance (EUC_2D / CEIL_2D rounding),
so results (cost/gap) are comparable to the TSPLIB-evaluation logs in `final/logs/tsplib_70/*`.

This script does NOT modify the platform code and does not depend on the platform evaluation pipeline.
It:
  1) loads a TSPLIB `.tsp` file (original coordinates + EDGE_WEIGHT_TYPE),
  2) normalizes coordinates to [0,1]^2 (isotropic scaling; preserves aspect ratio) for DACT inference,
  3) runs DACT improvement for `T_max` steps (with restart threshold `P` as Tr),
  4) converts the resulting best tour to a node permutation,
  5) re-computes tour length on original TSPLIB coordinates with TSPLIB rounding,
  6) computes gap w.r.t. a local TSPLIB optimal-cost table (or `--optimal_json`).

Example:
  python 0methods/DACT/eval_tsplib_official.py \
    --tsplib_dir EasyNCO/data/datasets/tsplib/tsplib \
    --instances berlin52 \
    --T_max 3000 --P 10 --val_m 1 --seeds 0

Notes:
  - This script contains a local TSPLIB parser and a local optimal-cost table.
  - If an instance is missing from the built-in optimal table, `gap` will be NaN unless you provide `--optimal_json`
    or you set `--require_optimal` to skip such instances.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from nets.actor_network import Actor
from problems.problem_tsp import TSP, get_real_seq

TSPLIB_OPTIMAL_COST: Dict[str, int] = {
    # Derived from `EasyNCO/data/datasets/tsplib/TSPlib_70instances.txt` (the 70-instance TSPLIB subset used by the platform logs).
    "a280": 2579,
    "berlin52": 7542,
    "bier127": 118282,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "gil262": 2378,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "nrw1379": 56638,
    "p654": 34643,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "st70": 675,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    "vm1084": 239297,
    "vm1748": 336556,
    "fl3795": 28772,
    "fnl4461": 182566,
    "pcb3038": 137694,
}


def _parse_kv_line(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s:
        return None
    # Accept both "KEY: value" and "KEY value" styles.
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
    Returns: (name, dimension, locs, edge_weight_type)
    """
    name: Optional[str] = None
    dimension: Optional[int] = None
    edge_weight_type: Optional[str] = None
    locs: List[List[float]] = []
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
                # Expected: idx x y
                if len(parts) < 3:
                    # Skip malformed line (some TSPLIB files may contain separators/comments).
                    continue
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                except ValueError:
                    continue
                locs.append([x, y])
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
    if len(locs) != dimension:
        raise ValueError(f"NODE_COORD_SECTION length mismatch in {filepath}: got {len(locs)}, expected {dimension}")

    return name, dimension, locs, edge_weight_type


def load_optimal_costs(optimal_json: str = "") -> Dict[str, float]:
    """
    Optional override mapping. If `optimal_json` is provided, it should be a JSON file:
      { "berlin52": 7542, ... }
    """
    costs: Dict[str, float] = {k: float(v) for k, v in TSPLIB_OPTIMAL_COST.items()}
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
    Reproduce `0methods/DACT/agent/ppo.py:PPO.rollout` coordinate augmentations.

    Input:
      coords: (1, N, 2), assumed in [0,1]
    Output:
      (val_m, N, 2)
    """
    assert coords.dim() == 3 and coords.size(0) == 1 and coords.size(2) == 2
    assert val_m in (1, 4, 6, 8), f"val_m must be one of 1/4/6/8, got {val_m}"

    bs, n, _ = coords.size()
    aug = coords.unsqueeze(1).repeat(1, val_m, 1, 1)  # (bs, val_m, N, 2)

    # Note: the original code uses aug[:, 0] as the base for swap transforms (i>=4).
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


def _tsplib_edge_weight(raw_dist: float, edge_weight_type: str) -> float:
    if edge_weight_type == "CEIL_2D":
        return float(math.ceil(raw_dist))
    if edge_weight_type == "EUC_2D":
        # TSPLIB definition: floor(x + 0.5)
        return float(math.floor(raw_dist + 0.5))
    return raw_dist


def tsplib_tour_cost(
    locs: Sequence[Sequence[float]],
    tour: Sequence[int],
    edge_weight_type: str,
) -> float:
    n = len(tour)
    assert n > 1
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        dx = float(locs[a][0]) - float(locs[b][0])
        dy = float(locs[a][1]) - float(locs[b][1])
        raw = math.sqrt(dx * dx + dy * dy)
        total += _tsplib_edge_weight(raw, edge_weight_type)
    return total


@dataclass(frozen=True)
class DACTModelConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    n_heads_actor: int = 4
    n_heads_decoder: int = 4
    n_layers: int = 3
    normalization: str = "layer"
    v_range: float = 6.0


def _load_actor(
    ckpt_path: str,
    device: torch.device,
    seq_length: int,
    cfg: DACTModelConfig,
) -> Actor:
    actor = Actor(
        problem_name="tsp",
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        n_heads_actor=cfg.n_heads_actor,
        n_heads_decoder=cfg.n_heads_decoder,
        n_layers=cfg.n_layers,
        normalization=cfg.normalization,
        v_range=cfg.v_range,
        seq_length=seq_length,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "actor" in ckpt:
        state_dict = ckpt["actor"]
    else:
        state_dict = ckpt
    actor.load_state_dict(state_dict, strict=True)
    actor.to(device)
    actor.eval()
    return actor


def _infer_best_tour_rec(
    actor: Actor,
    coords: torch.Tensor,  # (1, N, 2) on device
    *,
    T_max: int,
    P: int,
    step_method: str,
    init_val_met: str,
) -> torch.Tensor:
    """
    Run DACT improvement on one instance and return the best tour in linked-list (rec) format.
    """
    assert coords.dim() == 3 and coords.size(0) == 1 and coords.size(2) == 2
    n = coords.size(1)

    problem = TSP(
        p_size=n,
        init_val_met=init_val_met,
        with_assert=False,
        step_method=step_method,
        P=P,
        DUMMY_RATE=0,
    )
    problem.eval(perturb=True)

    batch = {"coordinates": coords}
    batch_feature = problem.input_feature_encoding(batch)

    # Initial solution is generated on CPU in upstream code; move to device.
    solutions = problem.get_initial_solutions(batch).to(coords.device).long()
    obj = problem.get_costs(batch, solutions)

    best_solution = solutions.clone()
    exchange = None
    solving_state = torch.zeros((1, 1), device=coords.device, dtype=torch.long)

    for _ in range(T_max):
        exchange = actor(problem, batch_feature, solutions, exchange, do_sample=True)[0]
        solutions, rewards, obj, solving_state = problem.step(
            batch,
            solutions,
            exchange,
            obj,
            solving_state,
            best_solution=best_solution,
        )
        best_solution[rewards > 0] = solutions[rewards > 0]

    return best_solution


def _rec_to_tour(best_solution_rec: torch.Tensor) -> List[int]:
    seq = get_real_seq(best_solution_rec).squeeze(0).detach().cpu().tolist()
    return [int(x) for x in seq]


def _find_tsplib_files(tsplib_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for root, _, files in os.walk(tsplib_dir):
        for f in files:
            if not f.endswith(".tsp"):
                continue
            out[os.path.splitext(f)[0]] = os.path.join(root, f)
    return out


def _parse_int_list_arg(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsplib_dir", type=str, required=True, help="Directory containing TSPLIB .tsp files")
    parser.add_argument(
        "--instances",
        type=str,
        default="",
        help="Comma-separated instance names (without .tsp). Empty => run all .tsp in tsplib_dir",
    )
    parser.add_argument("--min_nodes", type=int, default=0, help="Skip instances with N < min_nodes")
    parser.add_argument("--max_nodes", type=int, default=10**9, help="Skip instances with N > max_nodes")
    parser.add_argument("--T_max", type=int, default=3000, help="DACT inference step limit (T)")
    parser.add_argument("--P", type=int, default=10, help="Restart threshold (Tr) as in DACT paper; TSPLIB uses 10")
    parser.add_argument("--val_m", type=int, default=1, choices=[1, 4, 6, 8], help="Augmentation multiplicity")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds, e.g. 0,1,2,...,9")
    parser.add_argument(
        "--optimal_json",
        type=str,
        default="",
        help="Optional JSON mapping instance name to TSPLIB optimal cost (overrides built-in table).",
    )
    parser.add_argument(
        "--require_optimal",
        action="store_true",
        help="If set, skip instances without known optimal cost (gap becomes undefined).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="both",
        choices=["both", "gap", "cost"],
        help="Output fields: both(cost+gap) / gap(only gap) / cost(only cost)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--tsp50_ckpt",
        type=str,
        default="DACT/pretrained/tsp50-epoch-198.pt",
        help="DACT TSP50 checkpoint",
    )
    parser.add_argument(
        "--tsp100_ckpt",
        type=str,
        default="DACT/pretrained/tsp100-epoch-195.pt",
        help="DACT TSP100 checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output TSV path. Empty => print to stdout only.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = DACTModelConfig()
    seeds = _parse_int_list_arg(args.seeds)
    optimal_costs = load_optimal_costs(args.optimal_json)

    tsplib_files = _find_tsplib_files(args.tsplib_dir)
    if args.instances.strip():
        instance_names = [x.strip() for x in args.instances.split(",") if x.strip()]
    else:
        instance_names = sorted(tsplib_files.keys())

    lines: List[str] = []
    if args.report == "both":
        header_fields = [
            "name",
            "n",
            "edge_weight_type",
            "optimal",
            "T_max",
            "P",
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
    elif args.report == "gap":
        header_fields = [
            "name",
            "n",
            "edge_weight_type",
            "optimal",
            "T_max",
            "P",
            "val_m",
            "seeds",
            "no_aug_gap_best(%)",
            "no_aug_gap_avg(%)",
            "aug_gap_best(%)",
            "aug_gap_avg(%)",
            "time_sec_total",
        ]
    else:  # cost
        header_fields = [
            "name",
            "n",
            "edge_weight_type",
            "optimal",
            "T_max",
            "P",
            "val_m",
            "seeds",
            "no_aug_cost_best",
            "no_aug_cost_avg",
            "aug_cost_best",
            "aug_cost_avg",
            "time_sec_total",
        ]
    header = "\t".join(header_fields)
    lines.append(header)

    # Cache actors by (ckpt_path, n) to avoid reloading for same size.
    actor_cache: Dict[Tuple[str, int], Actor] = {}

    t_all0 = time.time()
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
        if args.require_optimal and optimal <= 0:
            print(f"[skip] {name}: missing optimal cost (use --optimal_json or disable --require_optimal)")
            continue

        # Pick checkpoint following the common benchmark rule:
        # small instances (n<99) use tsp50, otherwise tsp100.
        ckpt_path = args.tsp50_ckpt if n < 99 else args.tsp100_ckpt
        cache_key = (ckpt_path, n)
        if cache_key not in actor_cache:
            actor_cache[cache_key] = _load_actor(ckpt_path, device, n, cfg)
        actor = actor_cache[cache_key]

        # Prepare normalized coords for model (isotropic scaling), then augment.
        coords_orig = torch.tensor(locs, dtype=torch.float32)  # (N,2) on CPU
        coords_norm = _normalize_to_unit_board_isotropic(coords_orig).unsqueeze(0)  # (1,N,2)
        coords_aug = _augment_coords_pomo_style(coords_norm, args.val_m)  # (val_m, N, 2)
        coords_aug = coords_aug.to(device)

        no_aug_costs: List[float] = []
        aug_costs: List[float] = []

        t0 = time.time()
        for seed in seeds:
            _set_all_seeds(seed)

            # Evaluate each augment separately and pick best by *TSPLIB* cost.
            per_aug_costs: List[float] = []
            for aug_idx in range(args.val_m):
                best_rec = _infer_best_tour_rec(
                    actor,
                    coords_aug[aug_idx : aug_idx + 1],
                    T_max=args.T_max,
                    P=args.P,
                    step_method="2_opt",
                    init_val_met="greedy",
                )
                tour = _rec_to_tour(best_rec)
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
            if optimal <= 0:
                return float("nan")
            return 100.0 * ((cost - optimal) / optimal)

        if args.report == "both":
            row_fields = [
                name,
                str(n),
                edge_weight_type,
                str(int(optimal)) if optimal > 0 else "0",
                str(args.T_max),
                str(args.P),
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
        elif args.report == "gap":
            row_fields = [
                name,
                str(n),
                edge_weight_type,
                str(int(optimal)) if optimal > 0 else "0",
                str(args.T_max),
                str(args.P),
                str(args.val_m),
                ",".join(str(s) for s in seeds),
                f"{_gap(no_aug_best):.6f}",
                f"{_gap(no_aug_avg):.6f}",
                f"{_gap(aug_best):.6f}",
                f"{_gap(aug_avg):.6f}",
                f"{time_used:.3f}",
            ]
        else:  # cost
            row_fields = [
                name,
                str(n),
                edge_weight_type,
                str(int(optimal)) if optimal > 0 else "0",
                str(args.T_max),
                str(args.P),
                str(args.val_m),
                ",".join(str(s) for s in seeds),
                f"{no_aug_best:.6f}",
                f"{no_aug_avg:.6f}",
                f"{aug_best:.6f}",
                f"{aug_avg:.6f}",
                f"{time_used:.3f}",
            ]
        line = "\t".join(row_fields)
        print(line)
        lines.append(line)

    total_time = time.time() - t_all0
    print(f"[done] total_time_sec={total_time:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
