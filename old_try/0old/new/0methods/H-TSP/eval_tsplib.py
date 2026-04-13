import argparse
import ast
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra import initialize
from omegaconf import OmegaConf

from h_tsp import HTSP_PPO, RLSolver, VecEnv
from rl_utils import augment_xy_data_by_8_fold


class _NoOpFragmentBuffer:
    def update_buffer(self, fragments: torch.Tensor) -> None:
        return None


@dataclass(frozen=True)
class TSPLIBInstance:
    name: str
    tsp_path: str
    dimension: int
    edge_weight_type: str
    coords_xy: List[Tuple[float, float]]  # len == dimension, 0-based order
    optimal: Optional[int] = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _read_tsplib_70_optimal_list(path: str) -> Dict[str, int]:
    """
    Parse `TSPlib_70instances.txt` style:
      each line is a python list literal: [name, optimal, x1, y1, x2, y2, ...]
    """
    opt: Dict[str, int] = {}
    if not path or not os.path.exists(path):
        return opt
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = ast.literal_eval(line)
            except Exception as e:
                raise ValueError(f"Failed to parse {path}:{line_no}: {e}") from e
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0]).strip()
            if not name:
                continue
            try:
                optimal = int(float(row[1]))
            except Exception:
                continue
            opt[name] = optimal
    return opt


def _find_tsplib_tsp_files(tsplib_root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(tsplib_root):
        for fn in filenames:
            if fn.lower().endswith(".tsp"):
                name = os.path.splitext(fn)[0]
                # keep first occurrence if duplicates exist
                mapping.setdefault(name, os.path.join(dirpath, fn))
    return mapping


def _parse_tsplib_tsp(path: str) -> Tuple[str, int, str, List[Tuple[float, float]]]:
    name: Optional[str] = None
    dimension: Optional[int] = None
    edge_weight_type: Optional[str] = None
    in_coord_section = False
    coords: Dict[int, Tuple[float, float]] = {}

    def _parse_kv(line: str) -> Tuple[str, str]:
        if ":" in line:
            k, v = line.split(":", 1)
            return k.strip().upper(), v.strip()
        parts = line.split()
        if len(parts) >= 2:
            return parts[0].strip().upper(), " ".join(parts[1:]).strip()
        return line.strip().upper(), ""

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper in {"EOF", "END"}:
                break
            if upper.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue
            if not in_coord_section:
                key, value = _parse_kv(line)
                if key == "NAME":
                    name = value.split()[0]
                elif key == "DIMENSION":
                    try:
                        dimension = int(value)
                    except Exception:
                        pass
                elif key == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value.split()[0]
                continue

            # NODE_COORD_SECTION lines: `idx x y`
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                idx = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
            except Exception:
                continue
            coords[idx] = (x, y)

    if not coords:
        raise ValueError(f"No NODE_COORD_SECTION parsed from {path}")
    if dimension is None:
        dimension = max(coords.keys())
    if edge_weight_type is None:
        edge_weight_type = "EUC_2D"
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]

    coord_list: List[Tuple[float, float]] = []
    for i in range(1, dimension + 1):
        if i not in coords:
            raise ValueError(f"Missing coord for node {i} in {path}")
        coord_list.append(coords[i])

    return name, dimension, edge_weight_type, coord_list


def _normalize_xy(coords_xy: Sequence[Tuple[float, float]]) -> np.ndarray:
    xs = np.array([c[0] for c in coords_xy], dtype=np.float64)
    ys = np.array([c[1] for c in coords_xy], dtype=np.float64)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    if x_max > x_min:
        xs = (xs - x_min) / (x_max - x_min)
    else:
        xs = xs - x_min
    if y_max > y_min:
        ys = (ys - y_min) / (y_max - y_min)
    else:
        ys = ys - y_min
    xy = np.stack([xs, ys], axis=-1).astype(np.float32)
    return xy


def _tsplib_distance(edge_weight_type: str, a: Tuple[float, float], b: Tuple[float, float]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    d = math.sqrt(dx * dx + dy * dy)
    t = edge_weight_type.upper()
    if t == "EUC_2D":
        return int(d + 0.5)  # floor(d + 0.5)
    if t == "CEIL_2D":
        return int(math.ceil(d))
    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE={edge_weight_type!r}")


def _tsplib_tour_cost(edge_weight_type: str, coords_xy: Sequence[Tuple[float, float]], tour: Sequence[int]) -> int:
    n = len(coords_xy)
    if len(tour) != n:
        raise ValueError(f"Tour length {len(tour)} != dimension {n}")
    if len(set(tour)) != n:
        raise ValueError("Tour is not a permutation")
    cost = 0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        cost += _tsplib_distance(edge_weight_type, coords_xy[a], coords_xy[b])
    return int(cost)


@torch.no_grad()
def _infer_tours(
    model: HTSP_PPO,
    data_xy_01: torch.Tensor,  # [B, N, 2], in [0, 1]
    *,
    k: int,
    frag_len: int,
    max_new_nodes: int,
    max_improvement_step: int,
    low_level_group_size: int,
) -> List[List[int]]:
    device = next(model.parameters()).device
    vec_env = VecEnv(
        k=k,
        frag_len=frag_len,
        max_new_nodes=max_new_nodes,
        max_improvement_step=max_improvement_step,
    )
    states = vec_env.reset(data_xy_01.to(device))
    vec_env.to_device_(device)

    solver = RLSolver(model.low_level_model, low_level_group_size)
    frag_buffer = _NoOpFragmentBuffer()

    while not vec_env.done:
        actions = model(states).detach()
        states, _, _, _ = vec_env.step(actions, solver, frag_buffer=frag_buffer)

    return [env.state.current_tour for env in vec_env.envs]


def _eval_instance(
    model: HTSP_PPO,
    inst: TSPLIBInstance,
    *,
    k: int,
    frag_len: int,
    max_new_nodes: int,
    max_improvement_step: int,
    val_m: int,
    seeds: Sequence[int],
) -> Tuple[Dict[str, float], float]:
    """
    Returns (metrics, time_sec_total)
    metrics contains:
      no_aug_gap_best, no_aug_gap_avg, aug_gap_best, aug_gap_avg
    """
    if inst.optimal is None:
        raise ValueError(f"Missing optimal cost for instance {inst.name}")

    n = inst.dimension
    frag_len_eff = min(frag_len, n)
    # For small instances where we set `frag_len_eff == n`, we need k-NN to
    # cover all remaining nodes in the very first construction step; otherwise
    # `_extend_fragment` cannot reach `frag_len_eff` and will assert.
    k_eff = (n - 1) if frag_len_eff == n else min(k, max(n - 1, 1))
    max_new_nodes_eff = min(max_new_nodes, n)
    max_improvement_step_eff = max_improvement_step
    if frag_len_eff < 2:
        raise ValueError(f"Invalid frag_len_eff={frag_len_eff} for {inst.name}")
    low_level_group_size = frag_len_eff

    data_np = _normalize_xy(inst.coords_xy)
    data = torch.from_numpy(data_np)[None, ...]  # [1, N, 2]

    start_t = time.time()

    # no augmentation
    no_aug_costs: List[int] = []
    for seed in seeds:
        _set_seed(seed)
        tours = _infer_tours(
            model,
            data,
            k=k_eff,
            frag_len=frag_len_eff,
            max_new_nodes=max_new_nodes_eff,
            max_improvement_step=max_improvement_step_eff,
            low_level_group_size=low_level_group_size,
        )
        cost = _tsplib_tour_cost(inst.edge_weight_type, inst.coords_xy, tours[0])
        no_aug_costs.append(cost)

    no_aug_best = min(no_aug_costs)
    no_aug_avg = float(np.mean(no_aug_costs))

    # augmentation: only support 1 or 8 for now (POMO-style 8-fold)
    if val_m not in (1, 8):
        raise ValueError("--val_m only supports 1 or 8 (8-fold augmentation)")

    if val_m == 1:
        aug_best = no_aug_best
        aug_avg = no_aug_avg
    else:
        aug_costs: List[int] = []
        data_aug = augment_xy_data_by_8_fold(data)  # [8, N, 2]
        for seed in seeds:
            _set_seed(seed)
            tours = _infer_tours(
                model,
                data_aug,
                k=k_eff,
                frag_len=frag_len_eff,
                max_new_nodes=max_new_nodes_eff,
                max_improvement_step=max_improvement_step_eff,
                low_level_group_size=low_level_group_size,
            )
            costs = [
                _tsplib_tour_cost(inst.edge_weight_type, inst.coords_xy, tour)
                for tour in tours
            ]
            aug_costs.append(min(costs))
        aug_best = min(aug_costs)
        aug_avg = float(np.mean(aug_costs))

    total_time = time.time() - start_t

    def _gap(cost: float) -> float:
        return (float(cost) - float(inst.optimal)) / float(inst.optimal) * 100.0

    metrics = {
        "k": float(k_eff),
        "frag_len": float(frag_len_eff),
        "max_new_nodes": float(max_new_nodes_eff),
        "max_improvement_step": float(max_improvement_step_eff),
        "no_aug_gap_best": _gap(no_aug_best),
        "no_aug_gap_avg": _gap(no_aug_avg),
        "aug_gap_best": _gap(aug_best),
        "aug_gap_avg": _gap(aug_avg),
    }
    return metrics, total_time


def _tsv_write_row(fp, header: Sequence[str], row: Dict[str, object]) -> None:
    fp.write("\t".join(header) + "\n")
    fp.write(
        "\t".join("" if row.get(k) is None else str(row.get(k)) for k in header) + "\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate H-TSP (0methods/H-TSP) on TSPLIB with TSPLIB-official gap (%)"
    )
    p.add_argument(
        "--tsplib_root",
        type=str,
        required=True,
        help="Root folder that contains TSPLIB .tsp files (will scan recursively).",
    )
    p.add_argument(
        "--tsplib70_list",
        type=str,
        default=None,
        help="Path to TSPlib_70instances.txt (python list per line). If set and --instances not given, use it as default instance list + optimal.",
    )
    p.add_argument(
        "--instances",
        type=str,
        default="",
        help="Comma-separated instance names (without .tsp). If empty, use --tsplib70_list when available, otherwise evaluate all .tsp under --tsplib_root.",
    )
    p.add_argument(
        "--upper_model",
        type=str,
        required=True,
        help="Upper-level H-TSP lightning checkpoint (e.g., htsp_tsp1000.ckpt).",
    )
    p.add_argument(
        "--lower_model",
        type=str,
        required=True,
        help="Lower-level path solver lightning checkpoint (e.g., htsp_lower_tsp.ckpt).",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--frag_len", type=int, default=200)
    p.add_argument("--max_new_nodes", type=int, default=190)
    p.add_argument("--max_improvement_step", type=int, default=0)
    p.add_argument("--val_m", type=int, default=1, help="1 or 8 (8-fold augmentation).")
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds.")
    p.add_argument("--out", type=str, default="", help="Write TSV results to this file.")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tsplib_root = args.tsplib_root
    tsplib70_list = (
        args.tsplib70_list
        if args.tsplib70_list is not None
        else os.path.join(tsplib_root, "TSPlib_70instances.txt")
    )
    seeds = _parse_csv_int_list(args.seeds) or [0]

    tsp_files = _find_tsplib_tsp_files(tsplib_root)
    optimal_map = _read_tsplib_70_optimal_list(tsplib70_list)

    # instance list
    if args.instances.strip():
        instance_names = [x.strip() for x in args.instances.split(",") if x.strip()]
    elif optimal_map:
        instance_names = list(optimal_map.keys())
    else:
        instance_names = sorted(tsp_files.keys())

    # load model
    ckpt = torch.load(args.upper_model, map_location="cpu")
    with initialize(config_path=os.path.dirname(__file__), version_base="1.1"):
        cfg = OmegaConf.create(ckpt.get("hyper_parameters", {}))
    cfg.low_level_load_path = args.lower_model

    model = HTSP_PPO(cfg)
    incompatible = model.load_state_dict(ckpt.get("state_dict", {}), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "[warn] load_state_dict(strict=False): "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    header = [
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

    out_fp = None
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_fp = open(args.out, "w", encoding="utf-8")
        out_fp.write("\t".join(header) + "\n")
        out_fp.flush()

    print("\t".join(header))

    global_start = time.time()
    for idx, name in enumerate(instance_names, start=1):
        tsp_path = tsp_files.get(name)
        if not tsp_path:
            msg = f"[skip] {name}: .tsp not found under {tsplib_root}"
            if args.continue_on_error:
                print(msg)
                continue
            raise FileNotFoundError(msg)

        try:
            parsed_name, dim, ewt, coords = _parse_tsplib_tsp(tsp_path)
            inst = TSPLIBInstance(
                name=parsed_name,
                tsp_path=tsp_path,
                dimension=dim,
                edge_weight_type=ewt,
                coords_xy=coords,
                optimal=optimal_map.get(name),
            )
            metrics, sec = _eval_instance(
                model,
                inst,
                k=args.k,
                frag_len=args.frag_len,
                max_new_nodes=args.max_new_nodes,
                max_improvement_step=args.max_improvement_step,
                val_m=args.val_m,
                seeds=seeds,
            )
            row = {
                "name": inst.name,
                "n": inst.dimension,
                "edge_weight_type": inst.edge_weight_type,
                "optimal": inst.optimal,
                "k": int(metrics["k"]),
                "frag_len": int(metrics["frag_len"]),
                "max_new_nodes": int(metrics["max_new_nodes"]),
                "max_improvement_step": int(metrics["max_improvement_step"]),
                "val_m": args.val_m,
                "seeds": ",".join(str(s) for s in seeds),
                "no_aug_gap_best(%)": f"{metrics['no_aug_gap_best']:.6f}",
                "no_aug_gap_avg(%)": f"{metrics['no_aug_gap_avg']:.6f}",
                "aug_gap_best(%)": f"{metrics['aug_gap_best']:.6f}",
                "aug_gap_avg(%)": f"{metrics['aug_gap_avg']:.6f}",
                "time_sec_total": f"{sec:.3f}",
            }

            print("\t".join(str(row[h]) for h in header))
            if out_fp is not None:
                out_fp.write("\t".join(str(row[h]) for h in header) + "\n")
                out_fp.flush()
        except Exception as e:
            msg = f"[error] {name} ({idx}/{len(instance_names)}): {e}"
            if args.continue_on_error:
                print(msg)
                continue
            raise

    total = time.time() - global_start
    if out_fp is not None:
        out_fp.close()
    print(f"[done] total_time_sec={total:.3f}")


if __name__ == "__main__":
    main()
