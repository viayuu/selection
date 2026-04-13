from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from easynco_bootstrap import enable_easynco_runtime_patches, ensure_local_easynco

ensure_local_easynco()

from solver_zoo import build_initializer_zoo
from train_tsp_operator_policy import TrainConfig, _batch_to_coords, _build_data_loader, _set_seed
from tsp_utils import validate_tour

DEFAULT_INIT_ZOO = ("lehd", "elg", "difusco", "ins_nearest", "ins_random", "ins_regret")
EPS = 1e-9


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)

    def _emit(msg: str):
        print(msg, flush=True)
        handle.write(msg + "\n")
        handle.flush()

    return _emit, handle


def _resolve_dataset_path(scale: int, dataset_dir: str, dataset_path: str | None) -> str:
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return str(path)

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    matches = sorted(root.glob(f"test_tsp{scale}_*_uniform.pt"))
    if not matches:
        raise FileNotFoundError(f"No uniform TSP dataset found for scale {scale} under {root}")
    if len(matches) > 1:
        names = ", ".join(str(p) for p in matches)
        raise RuntimeError(f"Multiple dataset files matched scale {scale}: {names}")
    return str(matches[0])


def _infer_dataset_size(path: str) -> int:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".pt":
        payload = torch.load(data_path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return int(payload.size(0))
        if isinstance(payload, dict):
            if "node_xy" in payload and isinstance(payload["node_xy"], torch.Tensor):
                return int(payload["node_xy"].size(0))
            if "data" in payload and isinstance(payload["data"], torch.Tensor):
                return int(payload["data"].size(0))
        raise TypeError(f"Unsupported .pt dataset structure in {path}")
    if suffix == ".pkl":
        with data_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, (list, tuple)):
            return int(len(payload))
        if isinstance(payload, dict):
            for key in ("node_xy", "data"):
                value = payload.get(key)
                if hasattr(value, "__len__"):
                    return int(len(value))
        raise TypeError(f"Unsupported .pkl dataset structure in {path}")
    raise ValueError(f"Unsupported dataset suffix: {suffix}")


def _build_config(args: argparse.Namespace) -> tuple[TrainConfig, dict]:
    dataset_path = _resolve_dataset_path(args.problem_size, args.dataset_dir, args.dataset_path)
    dataset_size = _infer_dataset_size(dataset_path)
    target_items = dataset_size if args.max_items is None else min(args.max_items, dataset_size)
    solver_device = args.easynco_solver_device or args.device
    cfg = replace(
        TrainConfig(),
        seed=args.seed,
        device=args.device,
        problem_size=args.problem_size,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        init_zoo=tuple(args.init_zoo),
        easynco_solver_device=solver_device,
        heuristic_init_restarts=args.heuristic_init_restarts,
        heuristic_metric_strategy=args.heuristic_metric_strategy,
        heuristic_metric_p=args.heuristic_metric_p,
        heuristic_regret_k=args.heuristic_regret_k,
        lehd_ckpt=args.lehd_ckpt,
        elg_ckpt=args.elg_ckpt,
        difusco_ckpt=args.difusco_ckpt,
        pomo_ckpt=args.pomo_ckpt,
        am_ckpt=args.am_ckpt,
    )
    meta = {
        "dataset_path": dataset_path,
        "dataset_size": dataset_size,
        "target_items": target_items,
        "max_items": args.max_items,
    }
    return cfg, meta


def _iter_batches(cfg: TrainConfig, dataset_path: str, target_items: int):
    loader = _build_data_loader(
        source="easynco",
        path=dataset_path,
        data_size=target_items,
        batch_size=cfg.batch_size,
        problem_size=cfg.problem_size,
        device=cfg.device,
        distribution="uniform",
        scale_range=None,
    )

    def _iterator():
        seen = 0
        for batch in loader:
            coords = _batch_to_coords(batch, cfg.device)
            remain = target_items - seen
            if remain <= 0:
                break
            if coords.size(0) > remain:
                coords = coords[:remain]
            if coords.numel() == 0:
                continue
            seen += coords.size(0)
            yield coords
            if seen >= target_items:
                break

    return _iterator()


def _write_outputs(output_dir: Path, payload: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Initializer Benchmark")
    lines.append("")
    lines.append(f"- Problem size: `{payload['problem_size']}`")
    lines.append(f"- Dataset: `{payload['dataset_path']}`")
    lines.append(f"- Dataset size: `{payload['dataset_size']}`")
    lines.append(f"- Evaluated items: `{payload['evaluated_items']}`")
    lines.append(f"- Batch size: `{payload['batch_size']}`")
    lines.append(f"- Heuristic restarts: `{payload['heuristic_init_restarts']}`")
    lines.append("")
    lines.append("| Initializer | Avg Init Length | Win Rate | Avg Excess To Best | Avg Init Seconds |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in payload["results"]:
        lines.append(
            f"| `{row['name']}` | `{row['avg_init_length']:.6f}` | `{100.0 * row['win_rate']:.2f}%` | "
            f"`{row['avg_excess_to_best_init']:.6f}` | `{row['avg_init_seconds']:.6f}` |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `avg_init_length` 越低越好。")
    lines.append("- `win_rate` 按实例统计，若与该实例最优初始化长度并列，则记为 win。")
    lines.append("- `avg_excess_to_best_init` 越接近 0 越好。")
    lines.append("- `avg_init_seconds` 用于判断某方法是否靠额外计算预算取得优势。")
    if payload.get("max_items") is not None:
        lines.append("")
        lines.append("## Subset")
        lines.append("")
        lines.append(f"- 这次只评了前 `{payload['evaluated_items']}` 个实例，因为设置了 `--max_items={payload['max_items']}`。")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TSP initializers on a fixed dataset.")
    parser.add_argument("--problem_size", type=int, required=True)
    parser.add_argument("--dataset_dir", type=str, default="EasyNCO/data/datasets/test_dataset_tsp_uniform")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--easynco_solver_device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--init_zoo", nargs="+", default=list(DEFAULT_INIT_ZOO))
    parser.add_argument("--heuristic_init_restarts", type=int, default=4)
    parser.add_argument("--heuristic_metric_strategy", type=str, default="cartesian")
    parser.add_argument("--heuristic_metric_p", type=int, default=2)
    parser.add_argument("--heuristic_regret_k", type=int, default=1)
    parser.add_argument("--pomo_ckpt", type=str, default=None)
    parser.add_argument("--am_ckpt", type=str, default=None)
    parser.add_argument("--lehd_ckpt", type=str, default=None)
    parser.add_argument("--elg_ckpt", type=str, default=None)
    parser.add_argument("--difusco_ckpt", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"0new/operator_policy/benchmarks/init_tsp{args.problem_size}_uniform"
    return args


def main() -> None:
    args = parse_args()
    cfg, meta = _build_config(args)
    _set_seed(cfg.seed)

    output_dir = Path(args.output_dir)
    log, handle = _make_logger(output_dir / "benchmark.log")
    try:
        log(f"[bench] problem_size={cfg.problem_size} dataset={meta['dataset_path']}")
        log(f"[bench] dataset_size={meta['dataset_size']} target_items={meta['target_items']} batch_size={cfg.batch_size}")
        log(f"[bench] init_zoo={','.join(cfg.init_zoo)}")
        log(
            f"[bench] selector_device={cfg.device} easynco_solver_device={cfg.easynco_solver_device} "
            f"heuristic_restarts={cfg.heuristic_init_restarts} metric={cfg.heuristic_metric_strategy} "
            f"p={cfg.heuristic_metric_p} regret_k={cfg.heuristic_regret_k}"
        )

        patched = enable_easynco_runtime_patches(cfg.easynco_solver_device)
        if patched:
            log(f"[bench] enabled_easynco_runtime_patches={','.join(patched)}")

        init_zoo = build_initializer_zoo(cfg, log=log)
        log(f"[bench] initializer_zoo_ready count={len(init_zoo)}")

        stats = {
            solver.name: {
                "length_sum": 0.0,
                "win_count": 0,
                "excess_sum": 0.0,
                "seconds_sum": 0.0,
            }
            for solver in init_zoo
        }

        total_items = 0
        batch_start = time.perf_counter()
        for batch_idx, coords in enumerate(_iter_batches(cfg, meta["dataset_path"], meta["target_items"]), start=1):
            batch_lengths = []
            for solver in init_zoo:
                t0 = time.perf_counter()
                solution = solver.solve(coords)
                elapsed = time.perf_counter() - t0
                if not bool(validate_tour(solution.tour).all().item()):
                    raise RuntimeError(f"[{solver.name}] produced invalid tours during benchmark")
                lengths = solution.length.detach().cpu().to(torch.float64)
                batch_lengths.append(lengths)
                stats[solver.name]["length_sum"] += float(lengths.sum().item())
                stats[solver.name]["seconds_sum"] += float(elapsed)

            stacked = torch.stack(batch_lengths, dim=0)
            best = stacked.min(dim=0).values
            for row_idx, solver in enumerate(init_zoo):
                stats[solver.name]["win_count"] += int(((stacked[row_idx] - best).abs() <= EPS).sum().item())
                stats[solver.name]["excess_sum"] += float((stacked[row_idx] - best).sum().item())

            total_items += coords.size(0)
            if batch_idx == 1 or batch_idx % max(args.log_every_batches, 1) == 0:
                elapsed = time.perf_counter() - batch_start
                current_best = min(
                    (stats[solver.name]["length_sum"] / max(total_items, 1), solver.name) for solver in init_zoo
                )
                log(
                    f"[bench] batch={batch_idx:04d} items={total_items}/{meta['target_items']} "
                    f"best_avg_init={current_best[0]:.6f} ({current_best[1]}) elapsed={elapsed:.2f}s"
                )

        rows = []
        for solver in init_zoo:
            row = {
                "name": solver.name,
                "avg_init_length": stats[solver.name]["length_sum"] / max(total_items, 1),
                "win_rate": stats[solver.name]["win_count"] / max(total_items, 1),
                "avg_excess_to_best_init": stats[solver.name]["excess_sum"] / max(total_items, 1),
                "avg_init_seconds": stats[solver.name]["seconds_sum"] / max(total_items, 1),
            }
            rows.append(row)
        rows.sort(key=lambda r: (r["avg_init_length"], -r["win_rate"], r["avg_init_seconds"], r["name"]))

        payload = {
            "problem_size": cfg.problem_size,
            "dataset_path": meta["dataset_path"],
            "dataset_size": meta["dataset_size"],
            "evaluated_items": total_items,
            "batch_size": cfg.batch_size,
            "heuristic_init_restarts": cfg.heuristic_init_restarts,
            "heuristic_metric_strategy": cfg.heuristic_metric_strategy,
            "heuristic_metric_p": cfg.heuristic_metric_p,
            "heuristic_regret_k": cfg.heuristic_regret_k,
            "max_items": meta["max_items"],
            "results": rows,
        }
        _write_outputs(output_dir, payload)
        best = rows[0] if rows else None
        if best is not None:
            log(
                f"[bench] done best={best['name']} avg_init={best['avg_init_length']:.6f} "
                f"win_rate={100.0 * best['win_rate']:.2f}%"
            )
        log(f"[bench] outputs={output_dir}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        handle.close()


if __name__ == "__main__":
    main()
