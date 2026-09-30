#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import math
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path("/public/home/zhoucl/shiys")
BRIDGE_ROOT = REPO_ROOT / "easynco_v3_bridge"
LOG_ROOT = BRIDGE_ROOT / "logs"
RELD_ROOT = REPO_ROOT / "reld-nco-main"
RELD_MULTI_TASK_ROOT = RELD_ROOT / "Multi-Task"
DEFAULT_OUTPUT_ROOT = BRIDGE_ROOT / "exports" / "reld_nss_like_no_aug_v1"
DATA_ROOT_CANDIDATES = [
    REPO_ROOT / "data",
    REPO_ROOT / "0selection" / "data",
]


def resolve_data_root() -> Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(p) for p in DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Could not find data root. Searched: {searched}")


DATA_ROOT = resolve_data_root()

STATE_DIRNAME = "_bridge_state"
SHARD_DIRNAME = "_shards"
MONITOR_JSON_NAME = "reld_bridge_state.json"
EVENTS_JSONL_NAME = "reld_bridge_events.jsonl"
BATCH_CACHE_JSON_NAME = "batch_size_cache.json"
MANIFEST_JSON_NAME = "task_manifest.json"
MONITOR_MD_NAME = "MONITOR_RELD_LABEL.md"

MODEL_OUTPUT_NAMES = {
    "reld_mtl": "RELD_MTL",
    "reld_moel": "RELD_MOEL",
}

MODEL_SPECS = {
    "reld_mtl": {
        "display_name": "RELD_MTL",
        "model_type": "MTL",
        "checkpoint": RELD_MULTI_TASK_ROOT / "pretrained" / "reld_mtl" / "epoch-5000.pt",
        "num_experts": 4,
        "routing_level": "node",
        "routing_method": "input_choice",
        "norm": "none",
        "norm_loc": "norm_last",
        "expert_loc": ["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"],
        "ffidt": True,
        "eval_type": "argmax",
    },
    "reld_moel": {
        "display_name": "RELD_MOEL",
        "model_type": "MOE_LIGHT",
        "checkpoint": RELD_MULTI_TASK_ROOT / "pretrained" / "reld_moe_light" / "epoch-5000.pt",
        "num_experts": 4,
        "routing_level": "node",
        "routing_method": "input_choice",
        "norm": "none",
        "norm_loc": "norm_last",
        "expert_loc": ["Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Enc5", "Dec"],
        "ffidt": True,
        "eval_type": "argmax",
    },
}

CVRP_DATASETS = ["CVRPtrain", "CVRPval", "CVRPtest", "CVRPLIB"]
MVRP_VARIANTS = [
    "OVRP",
    "VRPB",
    "VRPL",
    "VRPTW",
    "OVRPTW",
    "OVRPB",
    "OVRPL",
    "VRPBL",
    "VRPBTW",
    "VRPLTW",
    "OVRPBL",
    "OVRPBTW",
    "OVRPLTW",
    "VRPBLTW",
    "OVRPBLTW",
]
MVRP_DATASETS = [
    *(f"{variant}{split}" for variant in MVRP_VARIANTS for split in ("train", "val", "test"))
]
ALL_DATASETS = [*CVRP_DATASETS, *MVRP_DATASETS]

SMOKE_DATASETS = ["CVRPval", "OVRPval", "VRPTWval"]

BASE_RESULT_METHODS = ["RELD_MTL", "RELD_MOEL"]


@dataclass(frozen=True)
class Task:
    dataset: str
    problem: str
    model_key: str
    scale: int
    indices: Tuple[int, ...]
    expected_lines: int

    @property
    def task_id(self) -> str:
        return f"{self.dataset}|{self.model_key}|{self.scale}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def monitor_md_path(output_root: Path) -> Path:
    return state_root(output_root) / MONITOR_MD_NAME


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle_atomic(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def dataset_pickle_path(dataset: str) -> Path:
    return DATA_ROOT / dataset / "dataset.pkl"


def result_file_path(output_root: Path, dataset: str, model_key: str) -> Path:
    return output_root / dataset / "results" / f"result_{MODEL_OUTPUT_NAMES[model_key]}.txt"


def raw_label_path(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "raw_label.pkl"


def copied_dataset_path(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "dataset.pkl"


def shard_file_path(output_root: Path, dataset: str, model_key: str, scale: int) -> Path:
    return shard_root(output_root) / dataset / model_key / f"scale_{scale}.jsonl"


def dataset_problem(dataset: str) -> str:
    if dataset.startswith("CVRP"):
        return "CVRP"
    if dataset.startswith("OVRP"):
        # Keep explicit variant match before generic.
        for variant in sorted(MVRP_VARIANTS, key=len, reverse=True):
            if dataset.startswith(variant):
                return variant
    for variant in sorted(MVRP_VARIANTS, key=len, reverse=True):
        if dataset.startswith(variant):
            return variant
    raise KeyError(f"Unsupported dataset: {dataset}")


def dataset_split(dataset: str) -> str:
    if dataset.endswith("train"):
        return "train"
    if dataset.endswith("val"):
        return "val"
    if dataset.endswith("test"):
        return "test"
    if dataset.endswith("LIB"):
        return "LIB"
    raise KeyError(f"Unsupported dataset split: {dataset}")


def list_target_datasets(scope: str = "all_data", datasets: Optional[Sequence[str]] = None) -> List[str]:
    if datasets:
        picked = [name.strip() for name in datasets if name.strip()]
        unknown = [name for name in picked if name not in ALL_DATASETS]
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}")
        return picked
    if scope == "all_data":
        return list(ALL_DATASETS)
    if scope == "smoke":
        return list(SMOKE_DATASETS)
    raise ValueError(f"Unsupported datasets scope: {scope}")


def default_batch_size(scale: int) -> int:
    if scale <= 100:
        return 128
    if scale <= 200:
        return 32
    if scale <= 500:
        return 8
    return 1


def batch_cache_key(model_key: str, problem: str, scale: int) -> str:
    return f"{model_key}|{problem}|{scale}"


def load_batch_cache(output_root: Path) -> Dict[str, int]:
    path = batch_cache_path(output_root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_batch_cache(output_root: Path, cache: Dict[str, int]) -> None:
    atomic_write_json(batch_cache_path(output_root), cache)


def normalize_tensor_like(value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return tensor


def infer_scale_from_sample(dataset: str, sample) -> int:
    problem = dataset_problem(dataset)
    if problem == "CVRP":
        loc = sample["loc"]
        return int(torch.as_tensor(loc).shape[-2])
    if problem in MVRP_VARIANTS:
        node_xy = sample["node_xy"]
        return int(torch.as_tensor(node_xy).shape[-2])
    raise KeyError(dataset)


def group_indices_by_scale(dataset_obj, dataset: str) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset_obj):
        grouped[infer_scale_from_sample(dataset, sample)].append(idx)
    return dict(sorted(grouped.items()))


def collate_problem_batch(dataset: str, dataset_obj, indices: Sequence[int]) -> Tuple[Tuple[torch.Tensor, ...], int]:
    problem = dataset_problem(dataset)
    samples = [dataset_obj[i] for i in indices]
    scale = infer_scale_from_sample(dataset, samples[0])

    if problem == "CVRP":
        depot = []
        loc = []
        demand = []
        for sample in samples:
            depot.append(normalize_tensor_like(sample["depot"]).squeeze(0))
            loc.append(normalize_tensor_like(sample["loc"]).squeeze(0))
            demand.append(normalize_tensor_like(sample["demand"]).squeeze(0))
        return (
            torch.stack(depot, dim=0),
            torch.stack(loc, dim=0),
            torch.stack(demand, dim=0),
        ), scale

    depot_xy = []
    node_xy = []
    node_demand = []
    route_limit = []
    service_time = []
    tw_start = []
    tw_end = []

    for sample in samples:
        depot_xy.append(normalize_tensor_like(sample["depot_xy"]))
        node_xy.append(normalize_tensor_like(sample["node_xy"]))
        node_demand.append(normalize_tensor_like(sample["node_demand"]))
        if "route_limit" in sample:
            route_limit.append(float(sample["route_limit"]))
        if "service_time" in sample:
            service_time.append(normalize_tensor_like(sample["service_time"]))
            tw_start.append(normalize_tensor_like(sample["tw_start"]))
            tw_end.append(normalize_tensor_like(sample["tw_end"]))

    depot_xy_t = torch.stack(depot_xy, dim=0)
    node_xy_t = torch.stack(node_xy, dim=0)
    node_demand_t = torch.stack(node_demand, dim=0)

    if service_time and route_limit:
        return (
            depot_xy_t,
            node_xy_t,
            node_demand_t,
            torch.tensor(route_limit, dtype=torch.float32),
            torch.stack(service_time, dim=0),
            torch.stack(tw_start, dim=0),
            torch.stack(tw_end, dim=0),
        ), scale
    if service_time:
        return (
            depot_xy_t,
            node_xy_t,
            node_demand_t,
            torch.stack(service_time, dim=0),
            torch.stack(tw_start, dim=0),
            torch.stack(tw_end, dim=0),
        ), scale
    if route_limit:
        return (
            depot_xy_t,
            node_xy_t,
            node_demand_t,
            torch.tensor(route_limit, dtype=torch.float32),
        ), scale
    return (
        depot_xy_t,
        node_xy_t,
        node_demand_t,
    ), scale


def import_reld_modules():
    root = str(RELD_MULTI_TASK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    reld_utils = importlib.import_module("utils")
    reld_envs = importlib.import_module("envs")
    reld_models = importlib.import_module("models")
    return reld_utils, reld_envs, reld_models


def model_init_params(spec: Dict[str, object], checkpoint_problem: str) -> Dict[str, object]:
    return {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "num_experts": spec["num_experts"],
        "eval_type": spec["eval_type"],
        "norm": spec["norm"],
        "norm_loc": spec["norm_loc"],
        "expert_loc": spec["expert_loc"],
        "problem": checkpoint_problem,
        "topk": 2,
        "routing_level": spec["routing_level"],
        "routing_method": spec["routing_method"],
        "ffidt": spec["ffidt"],
    }


def instantiate_reld_model(model_key: str, device: torch.device):
    spec = MODEL_SPECS[model_key]
    reld_utils, _, _ = import_reld_modules()
    checkpoint = torch.load(spec["checkpoint"], map_location=device)
    params = model_init_params(spec, checkpoint["problem"])
    model_cls = reld_utils.get_model(spec["model_type"])
    model = model_cls(**params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, params


def get_env_class(problem: str):
    reld_utils, _, _ = import_reld_modules()
    envs = reld_utils.get_env(problem)
    if len(envs) != 1:
        raise ValueError(f"Expected a single env for {problem}, got {envs}")
    return envs[0]


def move_batch_to_device(batch: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(item.to(device) for item in batch)


def _run_no_aug_batch(model, env_class, batch: Tuple[torch.Tensor, ...], scale: int, device: torch.device) -> torch.Tensor:
    batch = move_batch_to_device(batch, device)
    env = env_class(problem_size=scale, pomo_size=scale, device=device)
    batch_size = batch[0].size(0)
    aug_factor = 1
    sample_size = 1

    model.eval()
    with torch.no_grad():
        env.load_problems(batch_size, problems=batch, aug_factor=aug_factor)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

        aug_reward = reward.reshape(aug_factor * sample_size, batch_size, env.pomo_size)
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float()
    return no_aug_score.detach().cpu()


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def current_gpu_snapshot() -> List[dict]:
    try:
        out = os.popen(
            'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
        ).read().strip()
    except OSError:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization": int(parts[2]),
                "memory_used": int(parts[3]),
                "memory_total": int(parts[4]),
            }
        )
    return rows


def dataset_completed_for_model(output_root: Path, dataset: str, model_key: str, expected_instances: int) -> bool:
    result_path = result_file_path(output_root, dataset, model_key)
    if not result_path.is_file():
        return False
    count = 0
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count == expected_instances


def dataset_expected_instances(dataset: str) -> int:
    if dataset.endswith("train"):
        return 10000
    if dataset.endswith("val"):
        return 1000
    if dataset.endswith("test"):
        return 1000
    if dataset == "CVRPLIB":
        return 100
    raise KeyError(dataset)


def build_tasks(output_root: Path, datasets: Sequence[str], model_keys: Sequence[str], resume: bool = True) -> List[Task]:
    tasks: List[Task] = []
    for dataset in datasets:
        dataset_obj = load_pickle(dataset_pickle_path(dataset))
        grouped = group_indices_by_scale(dataset_obj, dataset)
        problem = dataset_problem(dataset)
        for model_key in model_keys:
            for scale, indices in grouped.items():
                shard_path = shard_file_path(output_root, dataset, model_key, scale)
                if resume and count_jsonl_lines(shard_path) == len(indices):
                    continue
                tasks.append(
                    Task(
                        dataset=dataset,
                        problem=problem,
                        model_key=model_key,
                        scale=scale,
                        indices=tuple(indices),
                        expected_lines=len(indices),
                    )
                )
    tasks.sort(key=lambda t: (-t.scale, t.dataset, t.model_key))
    return tasks


def serialize_task(task: Task) -> dict:
    return {
        "dataset": task.dataset,
        "problem": task.problem,
        "model_key": task.model_key,
        "scale": task.scale,
        "indices": list(task.indices),
        "expected_lines": task.expected_lines,
        "task_id": task.task_id,
    }


def deserialize_task(obj: dict) -> Task:
    return Task(
        dataset=obj["dataset"],
        problem=obj["problem"],
        model_key=obj["model_key"],
        scale=int(obj["scale"]),
        indices=tuple(int(i) for i in obj["indices"]),
        expected_lines=int(obj["expected_lines"]),
    )


def write_task_manifest(output_root: Path, tasks: Sequence[Task], datasets: Sequence[str], model_keys: Sequence[str]) -> None:
    payload = {
        "datasets": list(datasets),
        "models": list(model_keys),
        "tasks": [serialize_task(t) for t in tasks],
        "created_at": time.time(),
    }
    atomic_write_json(task_manifest_path(output_root), payload)


def load_task_manifest(output_root: Path) -> dict:
    path = task_manifest_path(output_root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def copy_dataset_pickle(output_root: Path, dataset: str) -> None:
    src = dataset_pickle_path(dataset)
    dst = copied_dataset_path(output_root, dataset)
    if dst.is_file():
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def build_raw_label(result_vectors: Dict[str, List[float]]) -> dict:
    methods = list(result_vectors.keys())
    n = len(next(iter(result_vectors.values())))
    raw = {}
    for idx in range(n):
        costs = [float(result_vectors[m][idx]) for m in methods]
        best = min(range(len(costs)), key=lambda j: costs[j])
        raw[str(idx)] = {
            "cost": costs,
            "time": [0.0 for _ in methods],
            "ind": int(best),
            "gap": [0.0 for _ in methods],
        }
    return raw


def write_export_manifest(output_root: Path, datasets: Sequence[str]) -> None:
    payload = {
        "output_root": str(output_root),
        "datasets": [],
    }
    for dataset in datasets:
        payload["datasets"].append(
            {
                "dataset": dataset,
                "problem": dataset_problem(dataset),
                "num_instances": dataset_expected_instances(dataset),
                "methods": BASE_RESULT_METHODS,
            }
        )
    atomic_write_json(output_root / "export_manifest.json", payload)
