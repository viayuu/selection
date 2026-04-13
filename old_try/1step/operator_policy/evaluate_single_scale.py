from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Iterable

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from easynco_bootstrap import enable_easynco_runtime_patches, ensure_local_easynco

ensure_local_easynco()

from env import TSPImprovementEnv
from policy import TSPSelectorPolicy
from solver_zoo import build_initializer_zoo, build_operator_zoo
from trainer import ReinforceTrainer
from train_tsp_operator_policy import TrainConfig, _batch_to_coords, _build_data_loader, _set_seed


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)

    def _emit(msg: str):
        print(msg, flush=True)
        handle.write(msg + "\n")
        handle.flush()

    return _emit, handle


def _load_checkpoint_config(checkpoint: str) -> TrainConfig:
    ckpt = torch.load(checkpoint, map_location="cpu")
    base = TrainConfig()
    if not isinstance(ckpt, dict) or "config" not in ckpt or not isinstance(ckpt["config"], dict):
        return base

    valid_fields = {f.name for f in fields(TrainConfig)}
    payload = {k: v for k, v in ckpt["config"].items() if k in valid_fields}
    for key in ("init_zoo", "operator_zoo", "train_scale_range", "eval_scale_range"):
        if key in payload and isinstance(payload[key], list):
            payload[key] = tuple(payload[key])
    return replace(base, **payload)


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
        raise FileNotFoundError(
            f"No uniform TSP dataset found for scale {scale} under {root}."
        )
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


def _build_eval_config(args: argparse.Namespace) -> tuple[TrainConfig, dict]:
    ckpt_cfg = _load_checkpoint_config(args.checkpoint)
    dataset_path = _resolve_dataset_path(args.problem_size, args.dataset_dir, args.dataset_path)
    dataset_size = _infer_dataset_size(dataset_path)
    target_items = dataset_size if args.max_items is None else min(args.max_items, dataset_size)

    updates = {
        "problem_size": args.problem_size,
        "device": args.device if args.device is not None else ckpt_cfg.device,
        "easynco_solver_device": args.easynco_solver_device if args.easynco_solver_device is not None else ckpt_cfg.easynco_solver_device,
        "dact_solver_device": args.dact_solver_device if args.dact_solver_device is not None else ckpt_cfg.dact_solver_device,
        "eval_batch_size": args.eval_batch_size if args.eval_batch_size is not None else ckpt_cfg.eval_batch_size,
        "rollout_steps": args.rollout_steps if args.rollout_steps is not None else ckpt_cfg.rollout_steps,
        "seed": args.seed if args.seed is not None else ckpt_cfg.seed,
        "output_dir": args.output_dir,
        "load_path": args.checkpoint,
        "eval_only": True,
        "eval_data_source": "easynco",
        "eval_data_path": dataset_path,
        "eval_distribution": "uniform",
        "eval_size": target_items,
        "train_data_source": ckpt_cfg.train_data_source,
        "train_data_path": ckpt_cfg.train_data_path,
    }
    if args.init_zoo:
        updates["init_zoo"] = tuple(args.init_zoo)
    if args.operator_zoo:
        updates["operator_zoo"] = tuple(args.operator_zoo)
    for key in ("pomo_ckpt", "am_ckpt", "lehd_ckpt", "elg_ckpt", "difusco_ckpt", "glop_ckpt", "dact_ckpt", "lih_ckpt"):
        val = getattr(args, key)
        if val is not None:
            updates[key] = val

    cfg = replace(ckpt_cfg, **updates)
    meta = {
        "checkpoint": args.checkpoint,
        "checkpoint_problem_size": ckpt_cfg.problem_size,
        "dataset_path": dataset_path,
        "dataset_size": dataset_size,
        "target_items": target_items,
        "max_items": args.max_items,
        "log_every_batches": args.log_every_batches,
    }
    return cfg, meta


@torch.no_grad()
def _evaluate_with_progress(
    trainer: ReinforceTrainer,
    batches: Iterable[torch.Tensor],
    log,
    target_items: int,
    log_every_batches: int,
    trace_path: str | Path | None,
    init_names,
    op_names,
):
    return trainer.evaluate_batches(
        batches,
        progress_log_fn=log,
        target_items=target_items,
        log_every_batches=log_every_batches,
        trace_path=trace_path,
        init_names=init_names,
        op_names=op_names,
    )


def _make_batches(cfg: TrainConfig, target_items: int):
    loader = _build_data_loader(
        source="easynco",
        path=cfg.eval_data_path,
        data_size=target_items,
        batch_size=cfg.eval_batch_size,
        problem_size=cfg.problem_size,
        device=cfg.device,
        distribution=cfg.eval_distribution,
        scale_range=cfg.eval_scale_range,
    )

    def _iter():
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

    return _iter()


def _write_outputs(output_dir: Path, cfg: TrainConfig, meta: dict, metrics: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Single-scale evaluation")
    lines.append("")
    lines.append(f"- Checkpoint: `{meta['checkpoint']}`")
    lines.append(f"- Problem size: `{cfg.problem_size}`")
    lines.append(f"- Dataset: `{meta['dataset_path']}`")
    lines.append(f"- Dataset size: `{meta['dataset_size']}`")
    lines.append(f"- Evaluated items: `{metrics['num_items']}`")
    lines.append(f"- Eval batches: `{metrics['num_batches']}`")
    lines.append(f"- Rollout steps: `{cfg.rollout_steps}`")
    lines.append(f"- Init zoo: `{', '.join(cfg.init_zoo)}`")
    lines.append(f"- Operator zoo: `{', '.join(cfg.operator_zoo)}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- Init length: `{metrics['init_length']:.6f}`")
    lines.append(f"- Final length: `{metrics['final_length']:.6f}`")
    lines.append(f"- Improvement: `{metrics['improvement']:.6f}`")
    lines.append(f"- Elapsed seconds: `{metrics['elapsed_seconds']:.2f}`")
    lines.append("")
    lines.append("## Action distribution")
    lines.append("")
    for name, value in zip(cfg.init_zoo, metrics["init_distribution"]):
        lines.append(f"- Init `{name}`: `{value:.4f}`")
    for name, value in zip(cfg.operator_zoo, metrics["operator_distribution"]):
        lines.append(f"- Operator `{name}`: `{value:.4f}`")
    if metrics.get("trace_path") is not None:
        lines.append("")
        lines.append("## Detailed Trace")
        lines.append("")
        lines.append(f"- Instance actions: `{metrics['trace_path']}`")
        if metrics.get("trace_meta_path") is not None:
            lines.append(f"- Trace metadata: `{metrics['trace_meta_path']}`")
        if metrics.get("trace_step_summary_path") is not None:
            lines.append(f"- Step-wise operator summary: `{metrics['trace_step_summary_path']}`")

    if meta["max_items"] is not None:
        lines.append("")
        lines.append("## Subset")
        lines.append("")
        lines.append(f"- `--max_items` was set to `{meta['max_items']}`, so only the first `{meta['target_items']}` instances were evaluated.")

    if cfg.problem_size != meta["checkpoint_problem_size"]:
        lines.append("")
        lines.append("## Warning")
        lines.append("")
        lines.append(
            f"- This selector checkpoint was trained with `problem_size={meta['checkpoint_problem_size']}`, but evaluation uses `problem_size={cfg.problem_size}`."
        )
        lines.append("- Results are out-of-distribution and should be interpreted cautiously.")

    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one TSP scale for operator-selection policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--problem_size", type=int, required=True)
    parser.add_argument("--dataset_dir", type=str, default="EasyNCO/data/datasets/test_dataset_tsp_uniform")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--easynco_solver_device", type=str, default=None)
    parser.add_argument("--dact_solver_device", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--init_zoo", nargs="+", default=None)
    parser.add_argument("--operator_zoo", nargs="+", default=None)
    parser.add_argument("--pomo_ckpt", type=str, default=None)
    parser.add_argument("--am_ckpt", type=str, default=None)
    parser.add_argument("--lehd_ckpt", type=str, default=None)
    parser.add_argument("--elg_ckpt", type=str, default=None)
    parser.add_argument("--difusco_ckpt", type=str, default=None)
    parser.add_argument("--glop_ckpt", type=str, default=None)
    parser.add_argument("--dact_ckpt", type=str, default=None)
    parser.add_argument("--lih_ckpt", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_dir = Path(args.checkpoint).resolve().parent
        args.output_dir = str(ckpt_dir / f"eval_tsp{args.problem_size}_uniform")
    return args


def main():
    args = parse_args()
    cfg, meta = _build_eval_config(args)
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    log, handle = _make_logger(output_dir / "eval.log")
    try:
        log(f"[eval] checkpoint={meta['checkpoint']}")
        log(f"[eval] problem_size={cfg.problem_size} dataset={meta['dataset_path']}")
        log(f"[eval] dataset_size={meta['dataset_size']} target_items={meta['target_items']}")
        log(f"[eval] selector_device={cfg.device} easynco_solver_device={cfg.easynco_solver_device} dact_solver_device={cfg.dact_solver_device}")
        log(f"[eval] init_zoo={','.join(cfg.init_zoo)} operator_zoo={','.join(cfg.operator_zoo)}")
        if meta["checkpoint_problem_size"] != cfg.problem_size:
            log(
                f"[warn] checkpoint was trained at problem_size={meta['checkpoint_problem_size']} but is now evaluated at problem_size={cfg.problem_size}"
            )

        patched = enable_easynco_runtime_patches(cfg.easynco_solver_device)
        if patched:
            log(f"[eval] enabled_easynco_runtime_patches={','.join(patched)}")

        t0 = time.perf_counter()
        init_zoo = build_initializer_zoo(cfg, log=log)
        operator_zoo = build_operator_zoo(cfg, log=log)
        log(f"[eval] solver_zoo_ready init={len(init_zoo)} operators={len(operator_zoo)} elapsed={time.perf_counter() - t0:.2f}s")

        policy = TSPSelectorPolicy(
            num_initializers=len(init_zoo),
            num_operators=len(operator_zoo),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            cpe_dim=cfg.cpe_dim,
            dropout=cfg.dropout,
        ).to(cfg.device)
        ckpt = torch.load(cfg.load_path, map_location=cfg.device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict)

        env = TSPImprovementEnv(init_zoo=init_zoo, operator_zoo=operator_zoo, rollout_steps=cfg.rollout_steps)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        trainer = ReinforceTrainer(
            policy=policy,
            env=env,
            optimizer=optimizer,
            rollout_steps=cfg.rollout_steps,
            entropy_coef=cfg.entropy_coef,
            grad_clip=cfg.grad_clip,
            single_sample_baseline_momentum=cfg.single_sample_baseline_momentum,
        )

        log("[eval] data loader ready, starting evaluation")
        batches = _make_batches(cfg, meta["target_items"])
        trace_path = output_dir / "instance_actions.jsonl"
        metrics = _evaluate_with_progress(
            trainer,
            batches,
            log,
            meta["target_items"],
            args.log_every_batches,
            trace_path,
            cfg.init_zoo,
            cfg.operator_zoo,
        )
        if metrics.get("trace_path") is not None:
            log(f"[eval] instance_trace={metrics['trace_path']}")
        if metrics.get("trace_step_summary_path") is not None:
            log(f"[eval] step_summary={metrics['trace_step_summary_path']}")
        _write_outputs(output_dir, cfg, meta, metrics)
        log(f"[eval] done final={metrics['final_length']:.4f} improvement={metrics['improvement']:.4f} elapsed={metrics['elapsed_seconds']:.2f}s")
        log(f"[eval] outputs={output_dir}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    finally:
        handle.close()


if __name__ == "__main__":
    main()
