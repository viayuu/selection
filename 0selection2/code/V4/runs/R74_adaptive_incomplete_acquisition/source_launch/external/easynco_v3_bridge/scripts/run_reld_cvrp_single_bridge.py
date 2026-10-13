#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import shutil
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from reld_bridge_common import (
    DATA_ROOT,
    LOG_ROOT,
    current_gpu_snapshot,
    ensure_dir,
    load_pickle,
    atomic_write_json,
    atomic_write_jsonl,
    append_jsonl,
    atomic_write_text,
)


DEFAULT_OUTPUT_ROOT = Path("/public/home/zhoucl/shiys/easynco_v3_bridge/exports/reld_cvrp_single_no_aug_v1")
CVRP_ROOT = Path("/public/home/zhoucl/shiys/reld-nco-main/CVRP")
STATE_DIRNAME = "_bridge_state"
SHARD_DIRNAME = "_shards"
MONITOR_JSON_NAME = "reld_bridge_state.json"
EVENTS_JSONL_NAME = "reld_bridge_events.jsonl"
BATCH_CACHE_JSON_NAME = "batch_size_cache.json"
MANIFEST_JSON_NAME = "task_manifest.json"
RESULT_NAME = "result_RELD_CVRP.txt"
MODEL_KEY = "reld_cvrp"

POLL_SECONDS = 5.0
AUTOSCALE_WINDOW_SECONDS = 300.0
UTIL_SAMPLE_SECONDS = 30.0
UTIL_THRESHOLD = 60.0
NO_OOM_WINDOW_SECONDS = 300.0

TARGET_DATASETS = ["CVRPtrain", "CVRPval", "CVRPtest", "CVRPLIB"]


def state_root(output_root: Path) -> Path:
    return output_root / STATE_DIRNAME


def shard_root(output_root: Path) -> Path:
    return output_root / SHARD_DIRNAME


def monitor_state_path(output_root: Path) -> Path:
    return state_root(output_root) / MONITOR_JSON_NAME


def events_log_path(output_root: Path) -> Path:
    return state_root(output_root) / EVENTS_JSONL_NAME


def batch_cache_path(output_root: Path) -> Path:
    return state_root(output_root) / BATCH_CACHE_JSON_NAME


def task_manifest_path(output_root: Path) -> Path:
    return state_root(output_root) / MANIFEST_JSON_NAME


def copied_dataset_path(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "dataset.pkl"


def result_path(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "results" / RESULT_NAME


def raw_label_path(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "raw_label.pkl"


def shard_file_path(output_root: Path, dataset: str, scale: int) -> Path:
    return shard_root(output_root) / dataset / f"scale_{scale}.jsonl"


def save_pickle_atomic(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        import pickle

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_batch_cache(output_root: Path) -> Dict[str, int]:
    path = batch_cache_path(output_root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_batch_cache(output_root: Path, cache: Dict[str, int]) -> None:
    atomic_write_json(batch_cache_path(output_root), cache)


def default_batch_size(scale: int) -> int:
    if scale <= 100:
        return 128
    if scale <= 200:
        return 32
    if scale <= 500:
        return 8
    return 1


def batch_cache_key(scale: int) -> str:
    return f"{MODEL_KEY}|CVRP|{scale}"


def infer_scale(sample: dict) -> int:
    return int(torch.as_tensor(sample["loc"]).shape[-2])


def group_indices_by_scale(dataset_obj) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset_obj):
        grouped[infer_scale(sample)].append(idx)
    return dict(sorted(grouped.items()))


def collate_batch(dataset_obj, indices: Sequence[int]) -> Tuple[dict, int]:
    samples = [dataset_obj[i] for i in indices]
    scale = infer_scale(samples[0])
    loc = torch.stack([torch.as_tensor(s["loc"], dtype=torch.float32).squeeze(0) for s in samples], dim=0)
    demand = torch.stack([torch.as_tensor(s["demand"], dtype=torch.float32).squeeze(0) for s in samples], dim=0)
    depot = torch.stack([torch.as_tensor(s["depot"], dtype=torch.float32).squeeze(0) for s in samples], dim=0)
    return {"loc": loc, "demand": demand, "depot": depot}, scale


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or ("cuda" in text and "memory" in text)


def instantiate_single_cvrp_model(device: torch.device):
    import sys

    cv_root = str(CVRP_ROOT)
    if cv_root not in sys.path:
        sys.path.insert(0, cv_root)
    from CVRPModel import CVRPModel  # type: ignore

    config = yaml.load((CVRP_ROOT / "config.yml").read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    model_params = dict(config["model_params"])
    checkpoint = torch.load(CVRP_ROOT / "weights" / "ReLD" / "model_epoch_90.pt", map_location=device)
    model = CVRPModel(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, model_params


def run_no_aug_scores(model, batch: dict, scale: int, device: torch.device) -> torch.Tensor:
    import sys

    cv_root = str(CVRP_ROOT)
    if cv_root not in sys.path:
        sys.path.insert(0, cv_root)
    from CVRPEnv import CVRPEnv  # type: ignore

    env = CVRPEnv(multi_width=min(scale, 100), device=device, problem_size=scale)
    model.eval()
    with torch.no_grad():
        env.load_random_problems(batch, aug_factor=1)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, done = env.pre_step()
        while not done:
            cur_dist = env.get_cur_feature()
            selected, _ = model.one_step_rollout(state, cur_dist, eval_type="greedy")
            state, reward, done = env.step(selected)
        no_aug_score = -reward.max(dim=1).values.float()
    return no_aug_score.detach().cpu()


@dataclass(frozen=True)
class Task:
    dataset: str
    scale: int
    indices: Tuple[int, ...]
    expected_lines: int

    @property
    def task_id(self) -> str:
        return f"{self.dataset}|{MODEL_KEY}|{self.scale}"


def serialize_task(task: Task) -> dict:
    return {
        "dataset": task.dataset,
        "problem": "CVRP",
        "model_key": MODEL_KEY,
        "scale": task.scale,
        "indices": list(task.indices),
        "expected_lines": task.expected_lines,
        "task_id": task.task_id,
    }


def deserialize_task(obj: dict) -> Task:
    return Task(
        dataset=obj["dataset"],
        scale=int(obj["scale"]),
        indices=tuple(int(i) for i in obj["indices"]),
        expected_lines=int(obj["expected_lines"]),
    )


def write_task_manifest(output_root: Path, tasks: Sequence[Task], datasets: Sequence[str]) -> None:
    payload = {
        "datasets": list(datasets),
        "models": [MODEL_KEY],
        "tasks": [serialize_task(t) for t in tasks],
        "created_at": time.time(),
    }
    atomic_write_json(task_manifest_path(output_root), payload)


def build_tasks(output_root: Path, datasets: Sequence[str], resume: bool) -> List[Task]:
    tasks: List[Task] = []
    for dataset in datasets:
        dataset_obj = load_pickle(DATA_ROOT / dataset / "dataset.pkl")
        grouped = group_indices_by_scale(dataset_obj)
        for scale, indices in grouped.items():
            shard_path = shard_file_path(output_root, dataset, scale)
            if resume and count_jsonl_lines(shard_path) == len(indices):
                continue
            tasks.append(Task(dataset=dataset, scale=scale, indices=tuple(indices), expected_lines=len(indices)))
    tasks.sort(key=lambda t: (-t.scale, t.dataset))
    return tasks


def merge_outputs(output_root: Path, datasets: Sequence[str]) -> None:
    for dataset in datasets:
        src_dataset = DATA_ROOT / dataset / "dataset.pkl"
        dst_dataset = copied_dataset_path(output_root, dataset)
        ensure_dir(dst_dataset.parent)
        shutil.copy2(src_dataset, dst_dataset)

        dataset_obj = load_pickle(src_dataset)
        n = len(dataset_obj)
        merged: Dict[int, float] = {}
        for shard_path in sorted((shard_root(output_root) / dataset).glob("scale_*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    idx = int(row["global_index"])
                    if idx in merged:
                        raise ValueError(f"Duplicate global_index {idx} in {shard_path}")
                    merged[idx] = float(row["no_aug_score"])
        if sorted(merged.keys()) != list(range(n)):
            raise ValueError(f"{dataset}: merged shard ids are not contiguous 0..{n-1}")

        ordered = [merged[i] for i in range(n)]
        lines = [f"{i},{ordered[i]:.10f}" for i in range(n)]
        ensure_dir((output_root / dataset / "results"))
        atomic_write_text(result_path(output_root, dataset), "\n".join(lines) + "\n")
        raw_label = {
            str(i): {"cost": [ordered[i]], "time": [0.0], "ind": 0, "gap": [0.0]}
            for i in range(n)
        }
        save_pickle_atomic(raw_label_path(output_root, dataset), raw_label)

    manifest = {
        "output_root": str(output_root),
        "datasets": [{"dataset": ds, "methods": ["RELD_CVRP"]} for ds in datasets],
    }
    atomic_write_json(output_root / "export_manifest.json", manifest)


@dataclass
class WorkerHandle:
    worker_id: str
    gpu_id: int
    process: mp.Process
    stop_event: mp.Event
    started_at: float
    is_extra: bool


def worker_process(
    worker_id: str,
    gpu_id: int,
    task_queue: mp.Queue,
    status_queue: mp.Queue,
    stop_event: mp.Event,
    batch_cache,
    output_root_str: str,
) -> None:
    output_root = Path(output_root_str)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    model, _ = instantiate_single_cvrp_model(device)
    dataset_cache: Dict[str, object] = {}

    while True:
        if stop_event.is_set():
            status_queue.put({"event": "worker_exit", "worker_id": worker_id, "gpu_id": gpu_id})
            break
        try:
            payload = task_queue.get(timeout=2.0)
        except queue.Empty:
            continue
        if payload is None:
            status_queue.put({"event": "worker_exit", "worker_id": worker_id, "gpu_id": gpu_id})
            break

        task = deserialize_task(payload)
        shard_path = shard_file_path(output_root, task.dataset, task.scale)
        if count_jsonl_lines(shard_path) == task.expected_lines:
            status_queue.put({"event": "task_skip", "worker_id": worker_id, "gpu_id": gpu_id, "task": serialize_task(task)})
            continue

        status_queue.put({"event": "task_start", "worker_id": worker_id, "gpu_id": gpu_id, "task": serialize_task(task), "timestamp": time.time()})
        started = time.time()
        try:
            dataset_obj = dataset_cache.get(task.dataset)
            if dataset_obj is None:
                dataset_obj = load_pickle(DATA_ROOT / task.dataset / "dataset.pkl")
                dataset_cache[task.dataset] = dataset_obj

            cache_key = batch_cache_key(task.scale)
            current_batch_size = int(batch_cache.get(cache_key, 0) or 0)
            if current_batch_size <= 0:
                current_batch_size = default_batch_size(task.scale)

            records: List[dict] = []
            indices = list(task.indices)
            offset = 0
            while offset < len(indices):
                batch_indices = indices[offset : offset + current_batch_size]
                batch, scale = collate_batch(dataset_obj, batch_indices)
                try:
                    no_aug = run_no_aug_scores(model, batch, scale, device)
                except RuntimeError as exc:
                    if is_oom_error(exc):
                        torch.cuda.empty_cache()
                        if current_batch_size > 1:
                            new_batch_size = max(1, current_batch_size // 2)
                            batch_cache[cache_key] = new_batch_size
                            task_queue.put(serialize_task(task))
                            status_queue.put(
                                {
                                    "event": "oom_requeue",
                                    "worker_id": worker_id,
                                    "gpu_id": gpu_id,
                                    "task": serialize_task(task),
                                    "old_batch_size": current_batch_size,
                                    "new_batch_size": new_batch_size,
                                    "timestamp": time.time(),
                                }
                            )
                            records = []
                            break
                        status_queue.put(
                            {
                                "event": "task_failed",
                                "worker_id": worker_id,
                                "gpu_id": gpu_id,
                                "task": serialize_task(task),
                                "reason": f"OOM even with batch_size=1: {exc}",
                                "timestamp": time.time(),
                            }
                        )
                        records = []
                        break
                    raise

                for global_index, score in zip(batch_indices, no_aug.tolist()):
                    records.append(
                        {
                            "global_index": int(global_index),
                            "dataset": task.dataset,
                            "problem": "CVRP",
                            "model": "RELD_CVRP",
                            "scale": int(task.scale),
                            "no_aug_score": float(score),
                        }
                    )
                offset += len(batch_indices)

            if records:
                atomic_write_jsonl(shard_path, records)
                status_queue.put(
                    {
                        "event": "task_complete",
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "task": serialize_task(task),
                        "lines": len(records),
                        "elapsed_sec": time.time() - started,
                        "batch_size": current_batch_size,
                        "timestamp": time.time(),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            torch.cuda.empty_cache()
            status_queue.put(
                {
                    "event": "task_failed",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "task": serialize_task(task),
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time(),
                }
            )


def spawn_worker(
    ctx,
    workers: Dict[str, WorkerHandle],
    worker_seq: int,
    gpu_id: int,
    is_extra: bool,
    task_queue: mp.Queue,
    status_queue: mp.Queue,
    batch_cache,
    output_root: Path,
) -> int:
    stop_event = ctx.Event()
    worker_id = f"gpu{gpu_id}-{MODEL_KEY}-w{worker_seq}"
    process = ctx.Process(
        target=worker_process,
        args=(worker_id, gpu_id, task_queue, status_queue, stop_event, batch_cache, str(output_root)),
        daemon=True,
    )
    process.start()
    workers[worker_id] = WorkerHandle(
        worker_id=worker_id,
        gpu_id=gpu_id,
        process=process,
        stop_event=stop_event,
        started_at=time.time(),
        is_extra=is_extra,
    )
    return worker_seq + 1


def write_state(
    output_root: Path,
    datasets: List[str],
    pending_count: int,
    completed_by_dataset: Dict[str, int],
    failed_tasks: List[dict],
    workers: Dict[str, WorkerHandle],
    desired_workers: Dict[int, int],
    gpu_history: Dict[int, deque],
    extra_spawn_allowed: Dict[int, int],
    current_gpu_info: List[dict],
) -> None:
    state = {
        "timestamp": time.time(),
        "datasets": datasets,
        "models": [MODEL_KEY],
        "pending_by_model": {MODEL_KEY: pending_count},
        "completed_by_dataset_model": {k: {MODEL_KEY: v} for k, v in completed_by_dataset.items()},
        "failed_tasks": failed_tasks[-50:],
        "workers": {
            worker_id: {
                "gpu_id": handle.gpu_id,
                "model_key": MODEL_KEY,
                "pid": handle.process.pid,
                "alive": handle.process.is_alive(),
                "is_extra": handle.is_extra,
                "started_at": handle.started_at,
            }
            for worker_id, handle in workers.items()
        },
        "desired_workers": desired_workers,
        "gpu_util_history": {str(k): list(v) for k, v in gpu_history.items()},
        "gpu_worker_caps": extra_spawn_allowed,
        "gpu_snapshot": current_gpu_info,
    }
    atomic_write_json(monitor_state_path(output_root), state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dedicated single-task ReLD CVRP label bridge.")
    parser.add_argument("--datasets", default="CVRPtrain,CVRPval,CVRPtest,CVRPLIB")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--max-workers-per-gpu", type=int, default=3)
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(state_root(output_root))
    ensure_dir(LOG_ROOT)

    datasets = parse_csv_list(args.datasets)
    invalid = [d for d in datasets if d not in TARGET_DATASETS]
    if invalid:
        raise SystemExit(f"Unsupported datasets: {invalid}")
    gpus = [int(x) for x in parse_csv_list(args.gpus)]

    if args.merge_only:
        merge_outputs(output_root, datasets)
        return

    all_tasks = build_tasks(output_root, datasets, resume=False)
    pending_tasks = build_tasks(output_root, datasets, resume=args.resume)
    write_task_manifest(output_root, all_tasks, datasets)

    if args.dry_run:
        print(f"Datasets: {len(datasets)}")
        print(f"All tasks: {len(all_tasks)}")
        print(f"Pending tasks: {len(pending_tasks)}")
        for task in pending_tasks[:20]:
            print(json.dumps(serialize_task(task), ensure_ascii=False))
        return

    if not pending_tasks:
        merge_outputs(output_root, datasets)
        return

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    task_queue = ctx.Queue()
    status_queue = ctx.Queue()
    batch_cache = manager.dict(load_batch_cache(output_root))

    for task in pending_tasks:
        task_queue.put(serialize_task(task))

    workers: Dict[str, WorkerHandle] = {}
    desired_workers = {gpu: args.workers_per_gpu for gpu in gpus}
    extra_spawn_allowed = {gpu: args.max_workers_per_gpu for gpu in gpus}
    worker_seq = 1
    for gpu_id in gpus:
        for _ in range(args.workers_per_gpu):
            worker_seq = spawn_worker(ctx, workers, worker_seq, gpu_id, False, task_queue, status_queue, batch_cache, output_root)

    pending_count = len(pending_tasks)
    completed_by_dataset: Dict[str, int] = defaultdict(int)
    for task in all_tasks:
        if count_jsonl_lines(shard_file_path(output_root, task.dataset, task.scale)) == task.expected_lines:
            completed_by_dataset[task.dataset] += 1

    gpu_history: Dict[int, deque] = {gpu: deque(maxlen=max(1, int(AUTOSCALE_WINDOW_SECONDS / UTIL_SAMPLE_SECONDS))) for gpu in gpus}
    last_sample_time = 0.0
    last_oom_time: Dict[int, float] = {gpu: 0.0 for gpu in gpus}
    failed_tasks: List[dict] = []
    active_tasks: Dict[str, dict] = {}

    try:
        while True:
            now = time.time()
            try:
                while True:
                    event = status_queue.get_nowait()
                    append_jsonl(events_log_path(output_root), event)
                    event_type = event["event"]
                    worker_id = event.get("worker_id")
                    if event_type == "task_start":
                        active_tasks[worker_id] = event["task"]
                    elif event_type == "task_complete":
                        active_tasks.pop(worker_id, None)
                        pending_count -= 1
                        task = event["task"]
                        completed_by_dataset[task["dataset"]] += 1
                    elif event_type == "task_skip":
                        active_tasks.pop(worker_id, None)
                        pending_count -= 1
                        task = event["task"]
                        completed_by_dataset[task["dataset"]] += 1
                    elif event_type == "oom_requeue":
                        active_tasks.pop(worker_id, None)
                        gpu_id = int(event["gpu_id"])
                        last_oom_time[gpu_id] = event["timestamp"]
                        save_batch_cache(output_root, dict(batch_cache))
                        if desired_workers[gpu_id] > 1:
                            desired_workers[gpu_id] -= 1
                            candidates = [h for h in workers.values() if h.gpu_id == gpu_id and h.process.is_alive()]
                            if candidates:
                                candidates.sort(key=lambda h: (h.is_extra, h.started_at), reverse=True)
                                candidates[0].stop_event.set()
                    elif event_type == "task_failed":
                        active_tasks.pop(worker_id, None)
                        pending_count -= 1
                        failed_tasks.append(event)
                    elif event_type == "worker_exit":
                        active_tasks.pop(worker_id, None)
            except queue.Empty:
                pass

            dead_workers = [wid for wid, handle in workers.items() if not handle.process.is_alive()]
            for wid in dead_workers:
                handle = workers.pop(wid)
                handle.process.join(timeout=0.1)
                active_task = active_tasks.pop(wid, None)
                if active_task is not None:
                    task_queue.put(active_task)

            if now - last_sample_time >= UTIL_SAMPLE_SECONDS:
                gpu_info = current_gpu_snapshot()
                gpu_info_by_id = {item["index"]: item for item in gpu_info}
                for gpu in gpus:
                    if gpu in gpu_info_by_id:
                        gpu_history[gpu].append(gpu_info_by_id[gpu]["utilization"])
                last_sample_time = now
            else:
                gpu_info = current_gpu_snapshot()

            if args.autoscale:
                alive_by_gpu = Counter(handle.gpu_id for handle in workers.values() if handle.process.is_alive())
                total_pending = max(0, pending_count)
                for gpu in gpus:
                    history = gpu_history[gpu]
                    if len(history) < history.maxlen or total_pending <= 0:
                        continue
                    avg_util = sum(history) / len(history)
                    if avg_util >= UTIL_THRESHOLD:
                        continue
                    if now - last_oom_time[gpu] < NO_OOM_WINDOW_SECONDS:
                        continue
                    if alive_by_gpu[gpu] >= extra_spawn_allowed[gpu]:
                        continue
                    worker_seq = spawn_worker(ctx, workers, worker_seq, gpu, True, task_queue, status_queue, batch_cache, output_root)
                    history.clear()

            write_state(
                output_root=output_root,
                datasets=datasets,
                pending_count=max(0, pending_count),
                completed_by_dataset=dict(completed_by_dataset),
                failed_tasks=failed_tasks,
                workers=workers,
                desired_workers=desired_workers,
                gpu_history=gpu_history,
                extra_spawn_allowed=extra_spawn_allowed,
                current_gpu_info=gpu_info,
            )

            if max(0, pending_count) == 0 and not active_tasks:
                break

            time.sleep(POLL_SECONDS)
    finally:
        save_batch_cache(output_root, dict(batch_cache))
        for handle in workers.values():
            handle.stop_event.set()
        for _ in range(8):
            task_queue.put(None)
        for handle in list(workers.values()):
            handle.process.join(timeout=10)

    merge_outputs(output_root, datasets)


if __name__ == "__main__":
    main()
