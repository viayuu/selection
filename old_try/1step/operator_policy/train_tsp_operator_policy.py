from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2024
    device: str = "cpu"
    problem_size: int = 100
    batch_size: int = 32
    train_steps: int = 500
    rollout_steps: int = 20
    lr: float = 1e-4
    entropy_coef: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 100
    save_every: int = 20
    eval_size: int = 128
    eval_batch_size: int = 32
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    cpe_dim: int = 32
    dropout: float = 0.0
    decoder_strategy: str = "greedy"
    init_zoo: tuple[str, ...] = ("lehd", "elg", "difusco")
    operator_zoo: tuple[str, ...] = ("two_opt", "lehd_rrc_step", "dact_2opt_step")
    solver_ckpt_dir: str = "EasyNCO/pretrained"
    output_dir: str = "0new/operator_policy/outputs/default"
    easynco_solver_device: str = "cpu"
    dact_solver_device: str = "cpu"
    two_opt_iterations: int = 1
    glop_pomo_size: int = 2
    glop_aug_factor: int = 4
    glop_revision_iters: int = 1
    heuristic_init_restarts: int = 4
    heuristic_metric_strategy: str = "cartesian"
    heuristic_metric_p: int = 2
    heuristic_regret_k: int = 1
    train_data_source: str = "online"
    train_data_path: str | None = None
    train_data_size: int = 10000
    train_distribution: str = "uniform"
    train_scale_range: tuple[int, int] | None = None
    eval_data_source: str = "online"
    eval_data_path: str | None = None
    eval_distribution: str = "uniform"
    eval_scale_range: tuple[int, int] | None = None
    single_sample_baseline_momentum: float = 0.9
    rollout_log_every: int = 5
    pomo_size: int | None = None
    pomo_ckpt: str | None = None
    am_ckpt: str | None = None
    lehd_ckpt: str | None = None
    elg_ckpt: str | None = None
    difusco_ckpt: str | None = None
    glop_ckpt: str | None = None
    dact_ckpt: str | None = None
    lih_ckpt: str | None = None
    load_path: str | None = None
    eval_only: bool = False


VALID_DATA_SOURCES = {"online", "easynco", "tsplib"}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _make_tsp_batch(batch_size: int, problem_size: int, device: str) -> torch.Tensor:
    return torch.rand((batch_size, problem_size, 2), device=device)



def _make_eval_data(cfg: TrainConfig) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed + 1)
    return torch.rand((cfg.eval_size, cfg.problem_size, 2), generator=g).to(cfg.device)



def _save_checkpoint(policy, cfg: TrainConfig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "config": asdict(cfg)}, output_dir / name)



def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)

    def _emit(msg: str):
        print(msg, flush=True)
        handle.write(msg + "\n")
        handle.flush()

    return _emit, handle


def _append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_distribution(names: tuple[str, ...], values: list[float]) -> str:
    return ", ".join(f"{name}:{value:.2f}" for name, value in zip(names, values))



def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _validate_data_source(source: str, field_name: str) -> str:
    source = source.lower()
    if source not in VALID_DATA_SOURCES:
        raise ValueError(f"{field_name} must be one of {sorted(VALID_DATA_SOURCES)}, got '{source}'")
    return source



def _batch_to_coords(batch, device: str) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        if "data" in batch:
            return batch["data"].to(device)
        if "node_xy" in batch:
            return batch["node_xy"].to(device)
    raise TypeError(f"Unsupported batch type for coordinate extraction: {type(batch)}")



def _build_data_loader(
    source: str,
    path: str | None,
    data_size: int,
    batch_size: int,
    problem_size: int,
    device: str,
    distribution: str,
    scale_range: tuple[int, int] | None,
):
    source = _validate_data_source(source, "data source")
    if source == "online":
        return None

    if source == "easynco":
        from EasyNCO.data.TSPGenerator import TSPGenerator

        kwargs = {}
        if scale_range is not None:
            kwargs["scale_range"] = list(scale_range)
        return TSPGenerator(
            data_size=data_size,
            problem_size=problem_size,
            batch_size=batch_size,
            device=device,
            path=path,
            distribution=distribution,
            **kwargs,
        )

    from torch.utils.data import DataLoader
    from EasyNCO.data.TSPGenerator import tsplib_loader

    dataset = tsplib_loader(device=device, path=path, scale_range=list(scale_range) if scale_range is not None else None)
    return DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None)



def _infinite_coords_iterator(loader, device: str) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield _batch_to_coords(batch, device)



def _evaluate_with_config(
    trainer: ReinforceTrainer,
    cfg: TrainConfig,
    *,
    progress_log_fn=None,
    log_every_batches: int = 0,
    trace_path: str | Path | None = None,
):
    eval_kwargs = {
        "progress_log_fn": progress_log_fn,
        "target_items": cfg.eval_size,
        "log_every_batches": log_every_batches,
        "trace_path": trace_path,
        "init_names": cfg.init_zoo,
        "op_names": cfg.operator_zoo,
    }
    if cfg.eval_data_source == "online":
        eval_coords = _make_eval_data(cfg)
        eval_kwargs["target_items"] = eval_coords.size(0)
        return trainer.evaluate(eval_coords, batch_size=cfg.eval_batch_size, **eval_kwargs)

    loader = _build_data_loader(
        source=cfg.eval_data_source,
        path=cfg.eval_data_path,
        data_size=cfg.eval_size,
        batch_size=cfg.eval_batch_size,
        problem_size=cfg.problem_size,
        device=cfg.device,
        distribution=cfg.eval_distribution,
        scale_range=cfg.eval_scale_range,
    )
    return trainer.evaluate_batches((_batch_to_coords(batch, cfg.device) for batch in loader), **eval_kwargs)



def run(cfg: TrainConfig):
    start_time = time.perf_counter()
    _set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    log, log_handle = _make_logger(output_dir / "run.log")
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"
    latest_metrics_path = output_dir / "latest_metrics.json"

    try:
        log(f"[run] seed={cfg.seed} device={cfg.device} problem_size={cfg.problem_size} batch_size={cfg.batch_size} rollout_steps={cfg.rollout_steps}")
        log(f"[run] train_data={cfg.train_data_source}:{cfg.train_distribution} size={cfg.train_data_size} eval_data={cfg.eval_data_source}:{cfg.eval_distribution} eval_size={cfg.eval_size}")
        log(f"[run] selector_device={cfg.device} easynco_solver_device={cfg.easynco_solver_device} dact_solver_device={cfg.dact_solver_device}")
        log(f"[run] init_zoo={','.join(cfg.init_zoo)} operator_zoo={','.join(cfg.operator_zoo)}")
        log(
            f"[run] heuristic_init_restarts={cfg.heuristic_init_restarts} metric={cfg.heuristic_metric_strategy} "
            f"p={cfg.heuristic_metric_p} regret_k={cfg.heuristic_regret_k}"
        )
        log(f"[run] output_dir={output_dir}")

        patched = enable_easynco_runtime_patches(cfg.easynco_solver_device)
        if patched:
            log(f"[run] enabled_easynco_runtime_patches={','.join(patched)}")

        init_zoo = build_initializer_zoo(cfg)
        operator_zoo = build_operator_zoo(cfg)
        log(f"[run] solver_zoo_ready init={len(init_zoo)} operators={len(operator_zoo)} elapsed={_format_seconds(time.perf_counter() - start_time)}")

        policy = TSPSelectorPolicy(
            num_initializers=len(init_zoo),
            num_operators=len(operator_zoo),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            cpe_dim=cfg.cpe_dim,
            dropout=cfg.dropout,
        ).to(cfg.device)
        if cfg.load_path is not None:
            ckpt = torch.load(cfg.load_path, map_location=cfg.device)
            if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
                ckpt_cfg = ckpt["config"]
                prev_init_zoo = tuple(ckpt_cfg.get("init_zoo", ()))
                prev_operator_zoo = tuple(ckpt_cfg.get("operator_zoo", ()))
                if prev_init_zoo and prev_init_zoo != tuple(cfg.init_zoo):
                    raise ValueError(
                        f"Checkpoint init_zoo={prev_init_zoo} does not match current init_zoo={tuple(cfg.init_zoo)}. "
                        "Please start from scratch when the initializer action space changes."
                    )
                if prev_operator_zoo and prev_operator_zoo != tuple(cfg.operator_zoo):
                    raise ValueError(
                        f"Checkpoint operator_zoo={prev_operator_zoo} does not match current operator_zoo={tuple(cfg.operator_zoo)}. "
                        "Please start from scratch when the operator action space changes."
                    )
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

        if cfg.eval_only:
            log("[eval] eval_only mode: starting evaluation")
            eval_t0 = time.perf_counter()
            eval_trace_dir = output_dir / "eval_traces"
            eval_trace_path = eval_trace_dir / "eval_only_instance_actions.jsonl"
            metrics = _evaluate_with_config(
                trainer,
                cfg,
                progress_log_fn=log,
                log_every_batches=1,
                trace_path=eval_trace_path,
            )
            payload = {
                "mode": "eval_only",
                "elapsed_seconds": time.perf_counter() - eval_t0,
                **metrics,
            }
            _append_jsonl(eval_metrics_path, payload)
            _write_json(latest_metrics_path, {"last_eval": payload})
            log(f"[eval] done elapsed={_format_seconds(payload['elapsed_seconds'])}")
            if metrics.get("trace_path") is not None:
                log(f"[eval] instance_trace={metrics['trace_path']}")
            if metrics.get("trace_step_summary_path") is not None:
                log(f"[eval] step_summary={metrics['trace_step_summary_path']}")
            log(json.dumps(metrics, indent=2, ensure_ascii=False))
            return

        train_iterator = None
        if cfg.train_data_source != "online":
            log("[data] building train loader...")
            data_t0 = time.perf_counter()
            train_loader = _build_data_loader(
                source=cfg.train_data_source,
                path=cfg.train_data_path,
                data_size=cfg.train_data_size,
                batch_size=cfg.batch_size,
                problem_size=cfg.problem_size,
                device=cfg.device,
                distribution=cfg.train_distribution,
                scale_range=cfg.train_scale_range,
            )
            train_iterator = _infinite_coords_iterator(train_loader, cfg.device)
            log(f"[data] train loader ready elapsed={_format_seconds(time.perf_counter() - data_t0)}")

        best_final = float("inf")
        last_eval_payload = None
        log("[run] training started")
        for step in range(1, cfg.train_steps + 1):
            trace_this_step = (step == 1) or (step % cfg.log_every == 0)
            fetch_t0 = time.perf_counter()
            if trace_this_step:
                log(f"[train:prepare] step={step:05d}/{cfg.train_steps:05d} fetching batch")
            coords = _make_tsp_batch(cfg.batch_size, cfg.problem_size, cfg.device) if train_iterator is None else next(train_iterator)
            fetch_seconds = time.perf_counter() - fetch_t0
            if trace_this_step:
                log(
                    f"[train:batch] step={step:05d} shape={tuple(coords.shape)} device={coords.device} fetch_t={_format_seconds(fetch_seconds)}"
                )
            metrics = trainer.train_step(
                coords,
                step_idx=step,
                log_fn=log if trace_this_step else None,
                rollout_log_every=cfg.rollout_log_every,
            )
            train_payload = {
                "step": step,
                "loss": metrics.loss,
                "init_length": metrics.init_length,
                "final_length": metrics.final_length,
                "improvement": metrics.improvement,
                "entropy": metrics.entropy,
                "baseline": metrics.baseline,
                "advantage_mean": metrics.advantage_mean,
                "init_distribution": metrics.init_distribution,
                "operator_distribution": metrics.operator_distribution,
                "select_seconds": metrics.select_seconds,
                "reset_seconds": metrics.reset_seconds,
                "rollout_seconds": metrics.rollout_seconds,
                "optimize_seconds": metrics.optimize_seconds,
                "total_seconds": metrics.total_seconds,
                "fetch_seconds": fetch_seconds,
            }
            _append_jsonl(train_metrics_path, train_payload)
            if trace_this_step:
                log(
                    f"[train] step={step:05d} loss={metrics.loss:.4f} init={metrics.init_length:.4f} "
                    f"final={metrics.final_length:.4f} improve={metrics.improvement:.4f} entropy={metrics.entropy:.4f} "
                    f"baseline={metrics.baseline:.4f} adv_mean={metrics.advantage_mean:.4f} total_t={_format_seconds(metrics.total_seconds)}"
                )
                log(f"[train] step={step:05d} init_dist={_format_distribution(cfg.init_zoo, metrics.init_distribution)}")
                log(f"[train] step={step:05d} op_dist={_format_distribution(cfg.operator_zoo, metrics.operator_distribution)}")
                log(
                    f"[train] step={step:05d} timing select={_format_seconds(metrics.select_seconds)} reset={_format_seconds(metrics.reset_seconds)} "
                    f"rollout={_format_seconds(metrics.rollout_seconds)} optimize={_format_seconds(metrics.optimize_seconds)}"
                )
            if step % max(cfg.save_every, 1) == 0 or step == cfg.train_steps:
                latest_payload = {
                    "step": step,
                    "best_eval_final": (best_final if best_final < float("inf") else None),
                    "last_train": train_payload,
                }
                if last_eval_payload is not None:
                    latest_payload["last_eval"] = last_eval_payload
                _write_json(latest_metrics_path, latest_payload)
                log(f"[save] wrote metrics snapshot -> {latest_metrics_path}")
            if step % cfg.eval_every == 0 or step == cfg.train_steps:
                log(f"[eval] step={step:05d} starting evaluation")
                eval_t0 = time.perf_counter()
                eval_trace_dir = output_dir / "eval_traces"
                eval_trace_path = eval_trace_dir / f"step_{step:05d}_instance_actions.jsonl"
                eval_metrics = _evaluate_with_config(
                    trainer,
                    cfg,
                    progress_log_fn=log,
                    log_every_batches=1,
                    trace_path=eval_trace_path,
                )
                eval_seconds = time.perf_counter() - eval_t0
                last_eval_payload = {
                    "step": step,
                    "elapsed_seconds": eval_seconds,
                    **eval_metrics,
                }
                _append_jsonl(eval_metrics_path, last_eval_payload)
                log(
                    f"[eval] step={step:05d} init={eval_metrics['init_length']:.4f} final={eval_metrics['final_length']:.4f} "
                    f"improve={eval_metrics['improvement']:.4f} elapsed={_format_seconds(eval_seconds)}"
                )
                log(f"[eval] step={step:05d} init_dist={_format_distribution(cfg.init_zoo, eval_metrics['init_distribution'])}")
                log(f"[eval] step={step:05d} op_dist={_format_distribution(cfg.operator_zoo, eval_metrics['operator_distribution'])}")
                if eval_metrics.get("trace_path") is not None:
                    log(f"[eval] step={step:05d} instance_trace={eval_metrics['trace_path']}")
                if eval_metrics.get("trace_step_summary_path") is not None:
                    log(f"[eval] step={step:05d} step_summary={eval_metrics['trace_step_summary_path']}")
                _save_checkpoint(policy, cfg, output_dir, "selector_last.pt")
                log(f"[ckpt] saved {output_dir / 'selector_last.pt'}")
                if eval_metrics["final_length"] < best_final:
                    best_final = eval_metrics["final_length"]
                    _save_checkpoint(policy, cfg, output_dir, "selector_best.pt")
                    log(f"[ckpt] saved new best {output_dir / 'selector_best.pt'} final={best_final:.4f}")
                latest_payload = {
                    "step": step,
                    "best_eval_final": best_final,
                    "last_train": train_payload,
                    "last_eval": last_eval_payload,
                }
                _write_json(latest_metrics_path, latest_payload)
                log(f"[save] wrote eval snapshot -> {latest_metrics_path}")
        log(f"[run] training finished elapsed={_format_seconds(time.perf_counter() - start_time)}")
    finally:
        log_handle.close()



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train TSP operator-selection policy on top of EasyNCO operators.")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--problem_size", type=int, default=TrainConfig.problem_size)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--train_steps", type=int, default=TrainConfig.train_steps)
    parser.add_argument("--rollout_steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--entropy_coef", type=float, default=TrainConfig.entropy_coef)
    parser.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--eval_size", type=int, default=TrainConfig.eval_size)
    parser.add_argument("--eval_batch_size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument("--hidden_dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--num_heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--num_layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--cpe_dim", type=int, default=TrainConfig.cpe_dim)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--decoder_strategy", type=str, default=TrainConfig.decoder_strategy)
    parser.add_argument("--solver_ckpt_dir", type=str, default=TrainConfig.solver_ckpt_dir)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--easynco_solver_device", type=str, default=TrainConfig.easynco_solver_device)
    parser.add_argument("--dact_solver_device", type=str, default=TrainConfig.dact_solver_device)
    parser.add_argument("--two_opt_iterations", type=int, default=TrainConfig.two_opt_iterations)
    parser.add_argument("--glop_pomo_size", type=int, default=TrainConfig.glop_pomo_size)
    parser.add_argument("--glop_aug_factor", type=int, default=TrainConfig.glop_aug_factor)
    parser.add_argument("--glop_revision_iters", type=int, default=TrainConfig.glop_revision_iters)
    parser.add_argument("--heuristic_init_restarts", type=int, default=TrainConfig.heuristic_init_restarts)
    parser.add_argument("--heuristic_metric_strategy", type=str, default=TrainConfig.heuristic_metric_strategy)
    parser.add_argument("--heuristic_metric_p", type=int, default=TrainConfig.heuristic_metric_p)
    parser.add_argument("--heuristic_regret_k", type=int, default=TrainConfig.heuristic_regret_k)
    parser.add_argument("--train_data_source", type=str, default=TrainConfig.train_data_source, choices=sorted(VALID_DATA_SOURCES))
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--train_data_size", type=int, default=TrainConfig.train_data_size)
    parser.add_argument("--train_distribution", type=str, default=TrainConfig.train_distribution)
    parser.add_argument("--train_scale_range", type=int, nargs=2, default=None)
    parser.add_argument("--eval_data_source", type=str, default=TrainConfig.eval_data_source, choices=sorted(VALID_DATA_SOURCES))
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--eval_distribution", type=str, default=TrainConfig.eval_distribution)
    parser.add_argument("--eval_scale_range", type=int, nargs=2, default=None)
    parser.add_argument("--single_sample_baseline_momentum", type=float, default=TrainConfig.single_sample_baseline_momentum)
    parser.add_argument("--rollout_log_every", type=int, default=TrainConfig.rollout_log_every)
    parser.add_argument("--pomo_size", type=int, default=None)
    parser.add_argument("--pomo_ckpt", type=str, default=None)
    parser.add_argument("--am_ckpt", type=str, default=None)
    parser.add_argument("--lehd_ckpt", type=str, default=None)
    parser.add_argument("--elg_ckpt", type=str, default=None)
    parser.add_argument("--difusco_ckpt", type=str, default=None)
    parser.add_argument("--glop_ckpt", type=str, default=None)
    parser.add_argument("--dact_ckpt", type=str, default=None)
    parser.add_argument("--lih_ckpt", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--init_zoo", nargs="+", default=list(TrainConfig.init_zoo))
    parser.add_argument("--operator_zoo", nargs="+", default=list(TrainConfig.operator_zoo))
    args = parser.parse_args()
    return TrainConfig(
        seed=args.seed,
        device=args.device,
        problem_size=args.problem_size,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        eval_size=args.eval_size,
        eval_batch_size=args.eval_batch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        cpe_dim=args.cpe_dim,
        dropout=args.dropout,
        decoder_strategy=args.decoder_strategy,
        init_zoo=tuple(args.init_zoo),
        operator_zoo=tuple(args.operator_zoo),
        solver_ckpt_dir=args.solver_ckpt_dir,
        output_dir=args.output_dir,
        easynco_solver_device=args.easynco_solver_device,
        dact_solver_device=args.dact_solver_device,
        two_opt_iterations=args.two_opt_iterations,
        glop_pomo_size=args.glop_pomo_size,
        glop_aug_factor=args.glop_aug_factor,
        glop_revision_iters=args.glop_revision_iters,
        heuristic_init_restarts=args.heuristic_init_restarts,
        heuristic_metric_strategy=args.heuristic_metric_strategy,
        heuristic_metric_p=args.heuristic_metric_p,
        heuristic_regret_k=args.heuristic_regret_k,
        train_data_source=args.train_data_source,
        train_data_path=args.train_data_path,
        train_data_size=args.train_data_size,
        train_distribution=args.train_distribution,
        train_scale_range=tuple(args.train_scale_range) if args.train_scale_range is not None else None,
        eval_data_source=args.eval_data_source,
        eval_data_path=args.eval_data_path,
        eval_distribution=args.eval_distribution,
        eval_scale_range=tuple(args.eval_scale_range) if args.eval_scale_range is not None else None,
        single_sample_baseline_momentum=args.single_sample_baseline_momentum,
        rollout_log_every=args.rollout_log_every,
        pomo_size=args.pomo_size,
        pomo_ckpt=args.pomo_ckpt,
        am_ckpt=args.am_ckpt,
        lehd_ckpt=args.lehd_ckpt,
        elg_ckpt=args.elg_ckpt,
        difusco_ckpt=args.difusco_ckpt,
        glop_ckpt=args.glop_ckpt,
        dact_ckpt=args.dact_ckpt,
        lih_ckpt=args.lih_ckpt,
        load_path=args.load_path,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    run(parse_args())
