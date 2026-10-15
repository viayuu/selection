"""Original task-specific NSS recipe on the new-only TSP/CVRP release."""

import argparse
import copy
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import pickle
import random
import signal
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, RandomSampler


REPO = Path(__file__).resolve().parents[2]
NSS = REPO / "code/NSS_retrain"
ROOT = REPO / "code/V4/runs/R79_NSS_new_data_batch1024"
DATA = Path("/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014")
GPUS = {
    "TSP": "GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50",
    "CVRP": "GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d",
}
DUAL_RESOURCES = {
    "TSP": dict(job="1475", node="gpu05", gpu_type="4090", uuids=[
        "GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be",
        "GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d",
    ]),
    "CVRP": dict(job="1465", node="gpu03", gpu_type="3090", uuids=[
        GPUS["CVRP"], GPUS["TSP"],
    ]),
}
COUNTS = {"train": 70114, "val": 3895, "test": 3895, "TSPLIB": 49, "CVRPLIB": 100}
SEED, BATCH = 2, 1024


def original_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, NSS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORIGINAL_UTILS = original_module("r79_original_nss_utils", "utils.py")
_previous_utils = sys.modules.get("utils")
sys.modules["utils"] = ORIGINAL_UTILS
try:
    ORIGINAL_DATASET = original_module("r79_original_nss_dataset", "dataset.py")
finally:
    if _previous_utils is None:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = _previous_utils
ORIGINAL_MODEL = original_module("r79_original_nss_model", "model.py")
ORIGINAL_LOSS = original_module("r79_original_nss_loss", "loss.py")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


@contextlib.contextmanager
def file_lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return CPUUnpickler(stream).load()


def source_hashes():
    return {name: sha256(NSS / name) for name in (
        "model.py", "loss.py", "dataset.py", "utils.py", "trainer.py",
        "config_TSP_ours.yml", "config_CVRP_ours.yml",
    )}


def recipe(task):
    original = yaml.safe_load((NSS / f"config_{task}_ours.yml").read_text())
    config = json.loads(json.dumps(original))
    config["seed"] = SEED
    config["train_params"]["train_batch_size"] = BATCH
    p = config["train_params"]
    if (p["num_epochs"] != 50 or p["start_epochs"] != 0 or p["loss"] != "rank"
            or not p["data_aug"] or p["manual_feature"] or p["ns_feature"]
            or p["test_batch_size"] != 8 or p["save_interval"] != 1):
        raise ValueError("The original NSS recipe changed; do not silently change this experiment")
    return original, config


def directory(task, split):
    if split.endswith("LIB"):
        if split != task + "LIB":
            raise ValueError("Mismatched task and original LIB")
        return (NSS / "datasets" / split).resolve()
    return DATA / (task + split)


def cache_identity(task, split):
    d = directory(task, split)
    files = [d / "dataset.pkl", d / "raw_label.pkl"]
    if not split.endswith("LIB"):
        files.extend([d / "metadata.json", d / "input_metadata.npz"])
    return dict(files=[identity(p) for p in files], original_sources=source_hashes(),
                pool=read_json(NSS / "solver_order_ours.json")[task], format=1)


def check_boundary(root, task, split):
    if split == "test" or split.endswith("LIB"):
        lock = read_json(root / "final_test_lock.json")
        record = lock["checkpoints"][task]
        if record["sha256"] != sha256(root / task / "best.pt"):
            raise PermissionError("The frozen VALID-best NSS checkpoint changed")
        if lock["original_sources"] != source_hashes():
            raise PermissionError("Original NSS source changed after checkpoint freeze")


def prepare_cache(root, task, split):
    check_boundary(root, task, split)
    cache = root / "cpu_cache" / (task + split + ".pt")
    stamp = cache.with_suffix(".json")
    expected = cache_identity(task, split)
    if cache.is_file() and stamp.is_file() and read_json(stamp)["identity"] == expected:
        return cache
    if cache.exists() or stamp.exists():
        raise RuntimeError(f"Stale/incomplete cache: {cache}; refusing to replace it silently")
    pool = expected["pool"]
    d = directory(task, split)
    if not split.endswith("LIB"):
        metadata = read_json(d / "metadata.json")
        if metadata["methods"] != pool or metadata["instances"] != COUNTS[split]:
            raise ValueError("New-data NSS candidate order/count disagrees with the original pool")
    else:
        names = [p.stem.removeprefix("result_") for p in sorted((d / "results").glob("result_*.txt"))]
        if names != pool:
            raise ValueError("Original LIB solver order changed")
    started = time.monotonic()
    instances = load_pickle(d / "dataset.pkl")
    raw = load_pickle(d / "raw_label.pkl")
    n = len(instances)
    if n != COUNTS[split] or len(raw) != n:
        raise ValueError("NSS instance/label count mismatch")
    if task == "CVRP":
        instances = [ORIGINAL_UTILS.process_instance_CVRP(x) for x in instances]
    lengths = torch.tensor([x.shape[1] for x in instances], dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    points = torch.cat([x.squeeze(0) for x in instances], dim=0).float().contiguous()
    del instances
    costs = np.asarray([raw[str(i)]["cost"] for i in range(n)], dtype=np.float64)
    gaps = np.asarray([raw[str(i)]["gap"] for i in range(n)], dtype=np.float64)
    times = np.asarray([raw[str(i)]["time"] for i in range(n)], dtype=np.float64)
    winner = np.asarray([raw[str(i)]["ind"] for i in range(n)], dtype=np.int64)
    if (costs.shape != (n, len(pool)) or gaps.shape != costs.shape or times.shape != costs.shape
            or not np.isfinite(costs).all() or not np.isfinite(gaps).all()
            or not np.array_equal(winner, costs.argmin(1))):
        raise ValueError("Invalid original NSS labels")
    del raw
    if not split.endswith("LIB"):
        with np.load(d / "input_metadata.npz", allow_pickle=False) as inputs:
            source_indices = torch.from_numpy(inputs["source_index"].astype(np.int64))
    else:
        source_indices = torch.arange(n)
    value = dict(points=points, offsets=offsets, lengths=lengths, costs=torch.from_numpy(costs),
                 gaps=torch.from_numpy(gaps), times=torch.from_numpy(times), winner=torch.from_numpy(winner),
                 source_indices=source_indices, task=task, split=split, pool=pool, identity=expected)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, cache)
    elapsed = time.monotonic() - started
    write_json(stamp, dict(identity=expected, rows=n, seconds=elapsed, cache=identity(cache),
                          nodes_min=int(lengths.min()), nodes_max=int(lengths.max()),
                          gap_all_zero=bool((gaps == 0).all())))
    print(f"CACHE {task} {split} rows={n} nodes={int(lengths.min())}..{int(lengths.max())} seconds={elapsed:.2f}", flush=True)
    return cache


class NSSDataset(Dataset):
    """Packed CPU instances; the original eight transformations, without 8x RAM."""

    def __init__(self, cache, augment=False):
        self.cache = cache
        self.augment = augment
        self.n = len(cache["winner"])

    def __len__(self):
        return self.n * (8 if self.augment else 1)

    def __getitem__(self, index):
        row, fold = divmod(index, 8) if self.augment else (index, 0)
        c = self.cache
        x = c["points"][int(c["offsets"][row]):int(c["offsets"][row + 1])]
        if self.augment:
            coords = ORIGINAL_UTILS.augment_xy_by_8_fold(x[None, :, :2])[fold]
            x = coords if x.shape[-1] == 2 else torch.cat((coords, x[:, 2:]), dim=-1)
        label = [int(c["winner"][row]), c["costs"][row].tolist(),
                 c["times"][row].tolist(), c["gaps"][row].tolist()]
        return [x, label, index]


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(dataset, batch_size, workers, shuffle=False):
    generator = torch.Generator().manual_seed(SEED)
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        collate_fn=ORIGINAL_DATASET.collate_fn, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0,
                        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(SEED),
                        **({"prefetch_factor": 2} if workers else {}))
    return loader, generator


class ResidentNSSLoader:
    """Same augmented-index sampler and NSS padding, with packed inputs on GPU."""

    def __init__(self, cache, device, batch_size=BATCH, augment=True, shuffle=True, generator=None):
        self.cpu_lengths = cache["lengths"]
        self.n = len(cache["winner"])
        self.size = self.n * (8 if augment else 1)
        self.batch_size, self.augment, self.shuffle = batch_size, augment, shuffle
        self.device = torch.device(device)
        self.generator = generator if generator is not None else torch.Generator().manual_seed(SEED)
        self.resident = {key: cache[key].to(self.device) for key in ("points", "offsets", "lengths", "winner")}
        self.resident.update({key: cache[key].float().to(self.device) for key in ("costs", "gaps", "times")})
        self.resident_bytes = sum(t.numel() * t.element_size() for t in self.resident.values())
        self.epoch_plan, self.next_batch_index = None, 0

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def batch(self, indices):
        cpu_ids = torch.as_tensor(indices, dtype=torch.int64)
        cpu_rows = cpu_ids // 8 if self.augment else cpu_ids
        width = int(self.cpu_lengths[cpu_rows].max())
        ids = cpu_ids.to(self.device)
        rows = ids // 8 if self.augment else ids
        c = self.resident
        columns = torch.arange(width, device=self.device)
        lengths = c["lengths"][rows]
        valid = columns[None] < lengths[:, None]
        positions = c["offsets"][rows, None] + columns[None]
        positions = positions.clamp_max(c["points"].shape[0] - 1)
        x = c["points"][positions]
        if self.augment:
            fold = ids % 8
            xy = torch.where((fold >= 4)[:, None, None], x[:, :, :2].flip(-1), x[:, :, :2])
            flips = torch.stack(((fold & 1) != 0, (fold & 2) != 0), dim=-1)
            xy = torch.where(flips[:, None], 1 - xy, xy)
            x = xy if x.shape[-1] == 2 else torch.cat((xy, x[:, :, 2:]), dim=-1)
        # Original augmentation happens before padding; reflected padding must stay zero.
        x = x.masked_fill(~valid[:, :, None], 0)
        mask = torch.zeros(valid.shape, device=self.device).masked_fill(~valid, float("-inf"))
        return [x, c["winner"][rows], c["costs"][rows], lengths.float(), mask,
                c["gaps"][rows], c["times"][rows], cpu_ids.tolist()]

    def __iter__(self):
        if self.epoch_plan is None:
            sampler = RandomSampler(range(self.size), generator=self.generator) if self.shuffle else range(self.size)
            self.epoch_plan = torch.tensor(list(sampler), dtype=torch.int64)
        for i in range(self.next_batch_index, len(self)):
            indices = self.epoch_plan[i * self.batch_size:(i + 1) * self.batch_size].tolist()
            self.next_batch_index = i + 1
            if i == len(self) - 1:
                self.epoch_plan, self.next_batch_index = None, 0
            yield self.batch(indices)
        self.epoch_plan, self.next_batch_index = None, 0

    def state_dict(self):
        return dict(epoch_plan=self.epoch_plan, next_batch_index=self.next_batch_index,
                    size=self.size, batch_size=self.batch_size)

    def load_state_dict(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Resume sample-plan size/global batch changed")
        plan = state["epoch_plan"]
        if plan is None or plan.shape != (self.size,) or not 0 <= state["next_batch_index"] < len(self):
            raise ValueError("Invalid unfinished epoch plan/cursor")
        self.epoch_plan, self.next_batch_index = plan, state["next_batch_index"]


def gpu_telemetry(task, uuid=None):
    result = subprocess.run([
        "nvidia-smi", "-i", uuid or GPUS[task],
        "--query-gpu=uuid,memory.used,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, timeout=10)
    if result.returncode:
        return dict(error=result.stderr.strip() or result.stdout.strip())
    uuid, used, total, util, memory_util = [x.strip() for x in result.stdout.strip().split(",")]
    return dict(uuid=uuid, nvml_used_mib=float(used), total_mib=float(total),
                gpu_util_pct=float(util), memory_util_pct=float(memory_util), captured_unix=time.time())


def trim_uncheckpointed_updates(run, updates):
    path = run / "updates.jsonl"
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    keep = [line for line in lines if json.loads(line)["update"] <= updates]
    discard = [line for line in lines if json.loads(line)["update"] > updates]
    if discard:
        archive = run / f"uncheckpointed_updates_after_{updates}_{time.time_ns()}.jsonl"
        archive.write_text("\n".join(discard) + "\n")
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(keep) + "\n")
        os.replace(tmp, path)
    return len(discard)


def probe_micro(root, task, model, optimizer, criterion, loader, cache, updates):
    """TRAIN-only execution benchmark; restore weights, Adam and RNG after each trial."""
    run = root / task
    probe_rng = torch.Generator().set_state(loader.generator.get_state())
    sampler = iter(RandomSampler(range(loader.size), generator=probe_rng))
    indices = list(itertools.islice(sampler, BATCH))
    batch = loader.batch(indices)
    cpu_dataset = NSSDataset(cache, augment=True)
    reference = ORIGINAL_DATASET.collate_fn([cpu_dataset[i] for i in indices])
    for i in range(7):
        torch.testing.assert_close(batch[i].cpu(), reference[i], rtol=0, atol=0)
    del reference
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError("Micro changes would affect cross-sample BatchNorm")
    model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    device = next(model.parameters()).device
    trials = []
    baseline_grad, baseline_loss = None, None

    def restore():
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        torch.cuda.empty_cache()

    def trial(requested):
        nonlocal baseline_grad, baseline_loss
        restore()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss, grad, actual, retries = train_update(model, optimizer, criterion, batch, device, requested)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        vector = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
        if baseline_grad is None:
            baseline_grad, baseline_loss = vector.clone(), loss
            relative_grad_error, loss_error = 0.0, 0.0
        else:
            relative_grad_error = ((vector - baseline_grad).norm() / baseline_grad.norm().clamp_min(1e-12)).item()
            loss_error = abs(loss - baseline_loss)
            if relative_grad_error > 5e-4 or loss_error > 3e-5 * max(1, abs(baseline_loss)):
                raise RuntimeError("Larger micro-batch changed the original global-mean gradient unexpectedly")
        value = dict(requested_micro=requested, micro_batch=actual, logical_batch=BATCH,
                     accumulation_steps=math.ceil(BATCH / actual), loss=loss, gradient_norm_sum=grad,
                     relative_gradient_error=relative_grad_error, absolute_loss_error=loss_error,
                     seconds=seconds, samples_per_second=BATCH / seconds, oom_retries=retries,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                     telemetry=gpu_telemetry(task))
        trials.append(value)
        print(f"PERF {task} micro={actual} seconds={seconds:.3f} samples/s={BATCH / seconds:.1f} "
              f"allocated={value['peak_allocated_gib']:.2f} reserved={value['peak_reserved_gib']:.2f}GiB "
              f"gradient_error={relative_grad_error:.3g}", flush=True)
        return value

    model.train()
    before = gpu_telemetry(task)
    try:
        train_update(model, optimizer, criterion, batch, device, 64)
        trial(64)
        larger = trial(128)
        total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if larger["micro_batch"] == 128 and larger["peak_reserved_gib"] < total_gib * 0.6:
            trial(192)
        safe = [t for t in trials if t["micro_batch"] == t["requested_micro"]
                and t["peak_reserved_gib"] < total_gib * 0.85]
        selected = min(safe, key=lambda t: t["seconds"])
        result = dict(task=task, formal_updates_before=updates, formal_updates_after=updates,
                      input_path="gpu_resident", batch_unit="augmented_training_samples",
                      augmentation="8 deterministic XY transformations, not cross-validation",
                      cross_sample_normalization=False, normalization="rezero", dropout=False,
                      same_original_augmentation_and_padding=True,
                      same_cpu_shuffle_rng=True, resident_cache_mib=loader.resident_bytes / 2**20,
                      telemetry_before=before, trials=trials, selected_micro=selected["micro_batch"],
                      optimizer_steps="one per logical batch; isolated probe steps discarded by full state restore",
                      numeric_change="FP32 GEMM/gradient reduction order; no RNG-consuming augmentation or layers")
        write_json(run / "performance_probe.json", result)
        return selected["micro_batch"], result
    finally:
        restore()


def device_padding_concate(self, tensors):
    # Original padding operation, with an explicit device instead of global CUDA defaults.
    lengths = [x.shape[0] for x in tensors]
    max_len = max(lengths)
    mask = tensors[0].new_zeros((len(tensors), max_len))
    padded = []
    for i, x in enumerate(tensors):
        padded.append(torch.nn.functional.pad(x, (0, 0, 0, max_len - lengths[i]))[None])
        mask[i, lengths[i]:max_len] = float("-inf")
    return torch.cat(padded, dim=0), mask


def make_model(config):
    p = dict(config["model_params"], problem_type=config["problem_type"],
             output_dim=config["train_params"]["num_classes"],
             ns_feature=config["train_params"]["ns_feature"])
    model = ORIGINAL_MODEL.Selection_model(**p)
    for block in model.encoder.blocks:
        block.padding_concate = types.MethodType(device_padding_concate, block)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def forward_micro_batch(model, micro_per_gpu):
    devices = len(model.device_ids) if isinstance(model, torch.nn.DataParallel) else 1
    return micro_per_gpu * max(1, devices)


def restore_original_missing_gradients(model):
    names = getattr(model, "original_missing_gradients", ())
    if names:
        parameters = dict(unwrap_model(model).named_parameters())
        for name in names:
            grad = parameters[name].grad
            if grad is not None and torch.count_nonzero(grad).item():
                raise RuntimeError(f"DP gave a nonzero gradient to original unused parameter {name}")
            parameters[name].grad = None


def transfer(batch, device, start=0, end=None):
    return [batch[i][start:end].to(device, non_blocking=True) for i in (0, 2, 3, 4)]


def training_input_ready(tensor, device, input_path):
    if input_path == "gpu_resident":
        return tensor.device == torch.device(device)
    return input_path == "cpu_pinned" and tensor.device.type == "cpu" and tensor.is_pinned()


def accumulate(model, criterion, batch, device, micro, outputs=None):
    n = len(batch[0])
    weighted_loss = torch.zeros((), device=device)
    for start in range(0, n, micro):
        end = min(start + micro, n)
        x, cost, scales, mask = transfer(batch, device, start, end)
        logits = model(x, scales, None, mask)
        loss = criterion(logits, cost)
        weight = (end - start) / n
        (loss * weight).backward()
        weighted_loss += loss.detach() * weight
        if outputs is not None:
            outputs[start:end].copy_(logits.detach())
    return weighted_loss


def accumulate_grouped(model, criterion, batch, device, micro, outputs=None):
    n = len(batch[0])
    lengths = batch[3].detach().to(device="cpu", dtype=torch.int64)
    order = torch.argsort(lengths, stable=True)
    weighted_loss = torch.zeros((), device=device)
    for start in range(0, n, micro):
        cpu_rows = order[start:start + micro]
        width = int(lengths[cpu_rows[-1]])
        rows = cpu_rows.to(batch[0].device)
        x = batch[0][:, :width].index_select(0, rows).to(device, non_blocking=True)
        cost = batch[2].index_select(0, rows).to(device, non_blocking=True)
        scales = batch[3].index_select(0, rows).to(device, non_blocking=True)
        mask = batch[4][:, :width].index_select(0, rows).to(device, non_blocking=True)
        logits = model(x, scales, None, mask)
        loss = criterion(logits, cost)
        weight = len(cpu_rows) / n
        (loss * weight).backward()
        weighted_loss += loss.detach() * weight
        if outputs is not None:
            outputs.index_copy_(0, rows.to(device), logits.detach())
    return weighted_loss


def check_grouped_batch(root, task, model, optimizer, criterion, loader, updates, source_checkpoint,
                        parallel_model=None):
    """One TRAIN batch, two backward paths, zero Adam steps; fixed FP32 gate."""
    tolerances = dict(score_atol=3e-5, score_rtol=3e-5, loss_relative=3e-5,
                      gradient_relative_l2=5e-4, gradient_max_atol=5e-5,
                      gradient_max_rtol=5e-4, argmax_disagreements=0)
    probe_rng = torch.Generator().set_state(loader.generator.get_state())
    sampler = iter(RandomSampler(range(loader.size), generator=probe_rng))
    indices = list(itertools.islice(sampler, BATCH))
    batch = loader.batch(indices)
    if len(batch[0]) != BATCH:
        raise ValueError("Equivalence check requires one original TRAIN logical1024 batch")
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,
                          torch.nn.modules.dropout._DropoutNd)) for m in model.modules()):
        raise RuntimeError("Grouped execution requires no cross-sample normalization or dropout")
    device = next(model.parameters()).device
    model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    devices = parallel_model.device_ids if parallel_model is not None else [device.index]
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    shuffle_state, was_training = loader.generator.get_state(), model.training
    result = dict(task=task, passed=False, tolerances=tolerances,
                  split="TRAIN", logical_batch=BATCH, micro_batch=64,
                  augmented_indices=indices, resume_checkpoint=source_checkpoint,
                  original_sources=source_hashes(), formal_updates_before=updates,
                  formal_updates_after=updates, optimizer_steps=0,
                  reference="original-order micro64, original logical-batch padding",
                  candidate="stable sort by true N within the same logical batch; crop each micro to max true N",
                  weighting="micro sample count / actual logical batch count (1024, tail784)",
                  validation_or_test_access=False)
    if parallel_model is not None:
        if devices != [0, 1] or unwrap_model(parallel_model) is not model:
            raise ValueError("The DP gate requires exactly two replicas of this same original model")
        result.update(reference="single GPU, stable true-N cropped micro64",
                      candidate="DataParallel, cropped micro128 gathered in order; at most64 per GPU",
                      forward_micro_batch=128, replicas=2, resource=DUAL_RESOURCES[task],
                      loss_reduction="original RankingLoss mean on gathered logits; micro count / actual global count")
    previous = [json.loads(line) for line in (root / task / "updates.jsonl").read_text().splitlines()]
    recent = [row for row in previous if row["logical_batch"] == BATCH][-20:]
    result["pre_handoff_recent_formal_updates"] = dict(
        count=len(recent), updates=[row["update"] for row in recent],
        mean_seconds=float(np.mean([row["seconds"] for row in recent])),
        samples_per_second=BATCH / float(np.mean([row["seconds"] for row in recent])))

    def restore():
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        loader.generator.set_state(shuffle_state)
        model.train(was_training)

    def backward(active_model, path, micro):
        restore()
        active_model.train()
        for gpu in devices:
            torch.cuda.synchronize(gpu)
            torch.cuda.reset_peak_memory_stats(gpu)
        outputs = torch.empty((BATCH, batch[2].shape[1]), device=device)
        started = time.monotonic()
        loss = path(active_model, criterion, batch, device, micro, outputs)
        restore_original_missing_gradients(active_model)
        for gpu in devices:
            torch.cuda.synchronize(gpu)
        elapsed = time.monotonic() - started
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        gradients = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]).cpu()
        stats = dict(loss=loss.item(), seconds_forward_backward=elapsed,
                     peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                     per_device_peak=[dict(device=gpu,
                                           allocated_gib=torch.cuda.max_memory_allocated(gpu) / 2**30,
                                           reserved_gib=torch.cuda.max_memory_reserved(gpu) / 2**30)
                                      for gpu in devices],
                     gradient_norm=gradients.double().norm().item(), missing_gradients=missing)
        return outputs.cpu(), gradients, stats

    try:
        ref_outputs, ref_grad, ref = backward(model, accumulate_grouped if parallel_model is not None else accumulate, 64)
        if parallel_model is not None:
            parallel_model.original_missing_gradients = ref["missing_gradients"]
            result["original_missing_gradients"] = ref["missing_gradients"]
            result["unused_gradient_semantics"] = "DP coalescing zeros restored to original None only after exact-zero check; Adam skips them as before"
        new_outputs, new_grad, new = backward(parallel_model if parallel_model is not None else model,
                                              accumulate_grouped, 128 if parallel_model is not None else 64)
        finite = all(torch.isfinite(t).all().item() for t in (ref_outputs, new_outputs, ref_grad, new_grad))
        if finite:
            score_error = (new_outputs.double() - ref_outputs.double()).abs()
            grad_error = (new_grad.double() - ref_grad.double()).abs()
            loss_error = abs(new["loss"] - ref["loss"])
            relative_grad = (grad_error.norm() / ref_grad.double().norm().clamp_min(1e-12)).item()
            max_grad = grad_error.max().item()
            argmax_disagreements = int((ref_outputs.argmax(1) != new_outputs.argmax(1)).sum())
            checks = dict(finite=True,
                          scores_close=torch.allclose(new_outputs, ref_outputs,
                                                      rtol=tolerances["score_rtol"], atol=tolerances["score_atol"]),
                          argmax_equal=argmax_disagreements == 0,
                          loss_close=loss_error <= tolerances["loss_relative"] * max(1, abs(ref["loss"])),
                          gradient_relative_close=relative_grad <= tolerances["gradient_relative_l2"],
                          gradient_max_close=max_grad <= tolerances["gradient_max_atol"]
                          + tolerances["gradient_max_rtol"] * ref_grad.abs().max().item(),
                          gradient_presence_equal=ref["missing_gradients"] == new["missing_gradients"])
            result.update(checks=checks, passed=all(checks.values()),
                          max_absolute_score_error=score_error.max().item(),
                          argmax_disagreements=argmax_disagreements,
                          absolute_weighted_mean_loss_error=loss_error,
                          relative_gradient_l2_error=relative_grad, max_absolute_gradient_error=max_grad,
                          reference_measurement=ref, candidate_measurement=new)
        else:
            result.update(checks=dict(finite=False), error="Nonfinite scores or gradients; candidate path abandoned")
    except torch.cuda.OutOfMemoryError:
        result["error"] = "OOM at fixed64/GPU; candidate path abandoned, no capacity scan"
    finally:
        restore()
        torch.cuda.empty_cache()
        for gpu in devices:
            torch.cuda.reset_peak_memory_stats(gpu)
    result["shuffle_rng_preserved"] = torch.equal(loader.generator.get_state(), shuffle_state)
    name = "parallel_equivalence.json" if parallel_model is not None else "runtime_equivalence.json"
    write_json(root / task / name, result)
    print(f"{'DP_CHECK' if parallel_model is not None else 'GROUPED_CHECK'} {task} "
          f"passed={result['passed']} checkpoint_update={updates} "
          f"score_error={result.get('max_absolute_score_error')} "
          f"argmax_changes={result.get('argmax_disagreements')} "
          f"gradient_error={result.get('relative_gradient_l2_error')}", flush=True)
    return result


def train_update(model, optimizer, criterion, batch, device, micro, grouped=False):
    retries = []
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            path = accumulate_grouped if grouped else accumulate
            loss = path(model, criterion, batch, device, forward_micro_batch(model, micro))
            restore_original_missing_gradients(model)
            break
        except torch.cuda.OutOfMemoryError:
            if isinstance(model, torch.nn.DataParallel) and model.device_ids:
                raise RuntimeError("DP OOM at fixed64/GPU; stop for handoff, do not silently change the DP protocol")
            if micro == 1:
                raise
            retries.append(micro)
            micro = max(1, micro // 2)
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite original NSS RankingLoss")
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    grad = torch.stack(norms).sum()
    if not torch.isfinite(grad):
        raise FloatingPointError("Nonfinite original NSS gradients")
    optimizer.step()
    return loss.item(), grad.item(), micro, retries


def summarize(scores, cache, train_sbs):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    probabilities = scores.softmax(1)
    top = probabilities.topk(min(4, scores.shape[1]), dim=1).indices.numpy()
    predicted = top[:, 0]
    costs = cache["costs"].numpy()
    gaps = cache["gaps"].numpy()
    native = cache["winner"].numpy()
    winner = costs.astype(np.float32).argmin(1)
    row = np.arange(len(costs))
    selected = costs[row, predicted]
    pool = cache["pool"]
    sbs = int(costs.mean(0).argmin())
    metric = torch.from_numpy(gaps if np.any(gaps) else costs).float()
    val_top1 = metric.gather(1, torch.from_numpy(predicted.copy())[:, None]).mean().item()
    fixed = pool.index(train_sbs)
    result = dict(n=len(costs), pool=pool, metric_name="gap" if np.any(gaps) else "cost",
                  top_1=val_top1, mean_cost=float(selected.mean()),
                  mean_oracle_gap_pct=float(((selected / costs.min(1)) - 1).mean() * 100),
                  oracle_cost=float(costs.min(1).mean()),
                  sbs_solver=pool[sbs], sbs_cost=float(costs[:, sbs].mean()),
                  train_sbs_solver=train_sbs, train_sbs_cost=float(costs[:, fixed].mean()),
                  winner_rule="FP32 cost argmin; native FP64 ind additionally reported",
                  fp32_native_winner_disagreements=int((winner != native).sum()),
                  arm_distribution={name: float((predicted == i).mean()) for i, name in enumerate(pool)},
                  methods=[])
    for k in (1, 2, 3):
        result[f"top{k}"] = float((top[:, :k] == winner[:, None]).any(1).mean())
        result[f"top{k}_native_ind"] = float((top[:, :k] == native[:, None]).any(1).mean())
        result[f"sbs_top{k}"] = float((np.argsort(costs.astype(np.float32), axis=1, kind="stable")[:, :k] == sbs).any(1).mean())
        result[f"top_{k}_metric"] = float(metric.gather(1, torch.from_numpy(top[:, :k].copy())).min(1).values.mean())
    for i, name in enumerate(pool):
        result["methods"].append(dict(solver=name, mean_cost=float(costs[:, i].mean()),
                                      top1=float((winner == i).mean()),
                                      selector_better_cost=bool(selected.mean() < costs[:, i].mean()),
                                      selector_better_top1=bool(result["top1"] > (winner == i).mean())))
    result["vs_sbs_pct"] = float((selected.mean() / result["sbs_cost"] - 1) * 100)
    result["vs_oracle_pct"] = float((selected.mean() / result["oracle_cost"] - 1) * 100)
    return result, predicted


@torch.no_grad()
def infer(model, loader, device):
    model = unwrap_model(model)
    model.eval()
    outputs = []
    started = time.monotonic()
    for batch in loader:
        x, _, scales, mask = transfer(batch, device)
        outputs.append(model(x, scales, None, mask).detach().cpu())
    scores = torch.cat(outputs)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite NSS evaluation scores")
    return scores, time.monotonic() - started


def save_predictions(path, scores, cache, predicted):
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, scores=scores.numpy(), pred=predicted,
                            costs=cache["costs"].numpy(), winner=cache["winner"].numpy(),
                            source_indices=cache["source_indices"].numpy(),
                            indices=np.arange(len(predicted)), pool=np.asarray(cache["pool"]))


def prepare(root, workers):
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "prepare.lock"):
        plan = dict(seed=SEED, global_batch=BATCH, augmented_train_rows=COUNTS["train"] * 8,
                    updates_per_epoch=math.ceil(COUNTS["train"] * 8 / BATCH),
                    updates_per_model=math.ceil(COUNTS["train"] * 8 / BATCH) * 50,
                    last_batch_rows=COUNTS["train"] * 8 % BATCH, data_root=str(DATA),
                    original_sources=source_hashes(), tasks={}, cuda_uuids=GPUS,
                    validation_rule="min mean top-1 gap; raw cost only if all gaps are zero; strict improvement",
                    runtime_estimate_hours={"TSP": [15, 30], "CVRP": [18, 40]},
                    runtime_estimate_status="unmeasured 3090 FP32 estimate; replaced by first actual updates")
        for task in GPUS:
            original, effective = recipe(task)
            plan["tasks"][task] = dict(original_config=original, effective_config=effective,
                                      train_sbs=read_json(directory(task, "train") / "metadata.json")["train_sbs_solver"],
                                      lib_root=str(directory(task, task + "LIB")))
            for split in ("train", "val"):
                prepare_cache(root, task, split)
        write_json(root / "protocol.json", plan)
    print(f"PREPARED {root} updates/model={plan['updates_per_model']}", flush=True)


def require_dual_approval():
    if os.environ.get("R79_DP_APPROVED") != "1" or os.environ.get("R79_R77_RELEASED") != "1":
        raise RuntimeError("DP awaits explicit migration license AND confirmed R77 training+TEST resource release")


def require_gpu(task, dual_runtime=False, single4090=False):
    if dual_runtime or single4090:
        require_dual_approval()
        resource = (dict(job="1475", node="gpu05", gpu_type="4090",
                         uuids=[DUAL_RESOURCES["TSP"]["uuids"][0 if task == "TSP" else 1]])
                    if single4090 else DUAL_RESOURCES[task])
        job, node, uuids = resource["job"], resource["node"], resource["uuids"]
    else:
        job, node, uuids = "1465", "gpu03", [GPUS[task]]
    if socket.gethostname().split(".")[0] != node or os.environ.get("SLURM_JOB_ID") != job:
        raise RuntimeError(f"R79 NSS must run through its assigned allocation {job} on {node}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(uuids):
        raise RuntimeError("Wrong GPU UUID; do not use any other GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(uuids):
        raise RuntimeError("Exactly the assigned CUDA GPU(s) must be visible")
    if (dual_runtime or single4090) and any(resource["gpu_type"] not in torch.cuda.get_device_name(i) for i in range(len(uuids))):
        raise RuntimeError("Wrong GPU family for the planned dual runtime")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def train(root, task, micro, workers, resume=False, input_path="cpu_pinned", optimize=False, check_grouped=False,
          dual_runtime=False, check_parallel=False, single4090=False):
    if dual_runtime and (root / "dual_migration_outcome.json").exists():
        raise RuntimeError("Original ReZero forward failed real DP equivalence; use the approved single4090 path")
    device = require_gpu(task, dual_runtime, single4090)
    plan = read_json(root / "protocol.json")
    if plan["original_sources"] != source_hashes():
        raise RuntimeError("Original NSS recipe/source changed after preparation")
    run = root / task
    run.mkdir(parents=True, exist_ok=True)
    with file_lock(run / "task.lock"):
        if (root / "final_test_lock.json").exists() or (run / "training_complete.json").exists():
            raise RuntimeError("Training is complete/frozen; do not retrain after seeing TEST/LIB")
        original, config = recipe(task)
        ORIGINAL_UTILS.seed_everything(SEED)
        train_cache = torch.load(prepare_cache(root, task, "train"), map_location="cpu", weights_only=False)
        val_cache = torch.load(prepare_cache(root, task, "val"), map_location="cpu", weights_only=False)
        shuffle_rng = torch.Generator().manual_seed(SEED)
        if input_path == "gpu_resident":
            loader = ResidentNSSLoader(train_cache, device, generator=shuffle_rng)
        else:
            loader, shuffle_rng = make_loader(NSSDataset(train_cache, augment=True), BATCH, workers, True)
        val_loader, _ = make_loader(NSSDataset(val_cache), 8, workers)
        model = make_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train_params"]["learning_rate"],
                                     weight_decay=float(config["train_params"]["weight_decay"]))
        k = config["train_params"]["num_classes"]
        criterion = ORIGINAL_LOSS.RankingLoss(k, k).to(device)
        train_sbs = plan["tasks"][task]["train_sbs"]
        start_epoch, updates, best, best_epoch = 0, 0, float("inf"), None
        source_checkpoint, discarded_updates = None, 0
        grouped = False
        saved_parallel = False
        saved_dual = False
        resume_epoch_loss, resume_epoch_rows, resume_epoch_seconds = 0.0, 0, 0.0
        if (run / "last.pt").exists():
            if not resume:
                raise RuntimeError("Partial training exists; use --resume to resume at the saved epoch boundary")
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(last["model_state_dict"])
            optimizer.load_state_dict(last["optimizer_state_dict"])
            shuffle_rng.set_state(last["shuffle_rng"])
            torch.set_rng_state(last["torch_rng"])
            torch.cuda.set_rng_state(last["cuda_rng"])
            if dual_runtime and len(last.get("cuda_rng_all") or []) == 2:
                torch.cuda.set_rng_state_all(last["cuda_rng_all"])
            start_epoch, updates, best, best_epoch = last["epoch"] + 1, last["updates"], last["best"], last["best_epoch"]
            if not last.get("cursor", {}).get("epoch_complete", True):
                if not isinstance(loader, ResidentNSSLoader):
                    raise ValueError("Mid-epoch cursor requires the canonical resident sample plan")
                loader.load_state_dict(last["loader_state"])
                start_epoch = last["cursor"]["epoch"]
                resume_epoch_loss, resume_epoch_rows = last["epoch_loss_sum"], last["epoch_rows"]
                resume_epoch_seconds = last["epoch_elapsed_seconds"]
            micro = last["micro_batch"]
            grouped = last.get("execution", {}).get("grouped_by_real_n", False)
            saved_parallel = last.get("execution", {}).get("data_parallel", False)
            saved_dual = last.get("execution", {}).get("dual_runtime", False)
            del last
            source_checkpoint = dict(path=str(run / "last.pt"), sha256=sha256(run / "last.pt"),
                                     saved_update=updates, next_epoch=start_epoch,
                                     next_batch_index=loader.next_batch_index if isinstance(loader, ResidentNSSLoader) else 0)
            discarded_updates = trim_uncheckpointed_updates(run, updates)
        elif (run / "updates.jsonl").exists():
            raise RuntimeError("Interrupted first epoch; refuse silent repetition of formal updates")
        performance = read_json(run / "performance_probe.json") if (run / "performance_probe.json").exists() else None
        if optimize:
            if input_path != "gpu_resident":
                raise ValueError("The performance upgrade requires the resident GPU input path")
            micro, performance = probe_micro(root, task, model, optimizer, criterion, loader, train_cache, updates)
            torch.cuda.reset_peak_memory_stats()
        equivalence = None
        if check_grouped:
            if input_path != "gpu_resident" or not resume or source_checkpoint is None or micro != 64:
                raise ValueError("The one-batch grouped check requires a micro64 GPU-resident checkpoint resume")
            equivalence = check_grouped_batch(root, task, model, optimizer, criterion, loader, updates,
                                             source_checkpoint)
            grouped = equivalence["passed"]
        elif grouped:
            equivalence = read_json(run / "runtime_equivalence.json")
            if not equivalence["passed"] or equivalence["original_sources"] != source_hashes():
                raise RuntimeError("Grouped execution requires the recorded passing original-source gate")
        parallel_equivalence = None
        if dual_runtime:
            if input_path != "gpu_resident" or not grouped or not resume or source_checkpoint is None or micro != 64:
                raise ValueError("Dual runtime requires the existing grouped GPU-resident micro64 checkpoint")
            if check_parallel:
                parallel = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=0)
                parallel_equivalence = check_grouped_batch(root, task, model, optimizer, criterion, loader,
                                                           updates, source_checkpoint, parallel)
                if parallel_equivalence["passed"]:
                    model = parallel
                else:
                    del parallel
            elif saved_parallel:
                parallel_equivalence = read_json(run / "parallel_equivalence.json")
                if not parallel_equivalence["passed"] or parallel_equivalence["original_sources"] != source_hashes():
                    raise RuntimeError("DP resume requires the recorded passing original-source gate")
                model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=0)
                model.original_missing_gradients = parallel_equivalence["original_missing_gradients"]
            elif saved_dual:
                parallel_equivalence = read_json(run / "parallel_equivalence.json")
                if parallel_equivalence["passed"] or parallel_equivalence["original_sources"] != source_hashes():
                    raise RuntimeError("Single-GPU fallback must match its recorded failed DP gate")
            else:
                raise RuntimeError("First dual migration requires --check-parallel; no unvalidated DP run")
        resource = (dict(job="1475", node="gpu05", gpu_type="4090",
                         uuids=[DUAL_RESOURCES["TSP"]["uuids"][0 if task == "TSP" else 1]])
                    if single4090 else DUAL_RESOURCES[task] if dual_runtime
                    else dict(job="1465", node="gpu03", uuids=[GPUS[task]]))
        execution = dict(input_path=input_path, micro_batch=micro, batch_unit="augmented_training_samples",
                         base_train_instances=COUNTS["train"], augmented_rows_per_epoch=COUNTS["train"] * 8,
                         augmentation="8 deterministic XY transforms; not cross-validation",
                         cross_sample_normalization=False, normalization="rezero", dropout=False,
                         grouped_by_real_n=grouped, crop_padding=grouped,
                         numeric_change=("FP32 GEMM shapes and reduction order; stable true-N grouping within unchanged logical batches"
                                         if grouped else "FP32 reduction order only; same augmented-index shuffle RNG"),
                         same_logical_batch_membership=True, actual_batch_mean_weighting=True,
                         optimizer_steps_per_logical_batch=1, tail_batch=plan["last_batch_rows"],
                         runtime_equivalence_path=str(run / "runtime_equivalence.json") if equivalence else None,
                         dual_runtime=dual_runtime, data_parallel=isinstance(model, torch.nn.DataParallel),
                         resource=resource, single4090=single4090,
                         forward_micro_batch=forward_micro_batch(model, micro), micro_batch_per_gpu=micro,
                         loss_reduction="original gathered-logit RankingLoss mean * micro count / actual global count",
                         parallel_equivalence_path=str(run / "parallel_equivalence.json") if parallel_equivalence else None,
                         resume_checkpoint=source_checkpoint, uncheckpointed_updates_archived=discarded_updates,
                         created_unix=time.time())
        execution["sample_plan"] = "canonical CPU RandomSampler materialized once per epoch; identical entries and epoch-end RNG; cursor persisted"
        if dual_runtime or single4090:
            execution["numeric_change"] += "; DP replica gradient summation if enabled; assigned GPU family transition"
        append_json(run / "execution_history.jsonl", execution)
        write_json(run / "config.json", dict(original=original, effective=config, micro_batch=micro,
                                             architecture="original NSS Selection_model", device=str(device),
                                             gpu_uuid=execution["resource"]["uuids"][0], execution=execution, performance_probe=performance,
                                             runtime_equivalence=equivalence,
                                             parallel_equivalence=parallel_equivalence,
                                             parameter_count=sum(p.numel() for p in model.parameters())))
        import wandb
        tracker = wandb.init(project="selector", entity="yjkds-southern-university-of-science-technology",
                             name=f"R79_NSS_{task}_new_data_batch1024_seed2", group="R79_NSS_new_data_batch1024",
                             dir=str(run), mode=os.environ.get("WANDB_MODE", "offline"),
                             config=dict(config, global_batch=BATCH, micro_batch=micro, data_root=str(DATA),
                                         execution=execution))
        launched = time.monotonic()
        pause_requested = False
        previous_handler = signal.getsignal(signal.SIGTERM)

        def request_pause(signum, frame):
            nonlocal pause_requested
            pause_requested = True

        if isinstance(loader, ResidentNSSLoader):
            signal.signal(signal.SIGTERM, request_pause)
        try:
            if not resume:
                scores, elapsed = infer(model, val_loader, device)
                initial, _ = summarize(scores, val_cache, train_sbs)
                write_json(run / "initial_validation.json", dict(initial, seconds=elapsed,
                                                                checkpoint_candidate=False))
                del scores
            for epoch in range(start_epoch, 50):
                model.train()
                epoch_started = time.monotonic() - (resume_epoch_seconds if epoch == start_epoch else 0)
                epoch_loss = resume_epoch_loss if epoch == start_epoch else 0.0
                epoch_rows = resume_epoch_rows if epoch == start_epoch else 0
                start_batch_index = loader.next_batch_index if isinstance(loader, ResidentNSSLoader) else 0
                fetch_started = time.monotonic()
                for batch_index, batch in enumerate(loader, start=start_batch_index):
                    started = time.monotonic()
                    input_seconds = started - fetch_started
                    n = len(batch[0])
                    if n != (BATCH if batch_index < len(loader) - 1 else plan["last_batch_rows"]):
                        raise ValueError("Logical batch shape/count mismatch")
                    loss, grad, micro, retries = train_update(model, optimizer, criterion, batch, device, micro, grouped)
                    updates += 1
                    compute_seconds = time.monotonic() - started
                    seconds = input_seconds + compute_seconds
                    epoch_loss += loss * n
                    epoch_rows += n
                    record = dict(epoch=epoch, update=updates, batch_index=batch_index,
                                  logical_batch=n, micro_batch=micro,
                                  forward_micro_batch=forward_micro_batch(model, micro),
                                  accumulation_steps=math.ceil(n / forward_micro_batch(model, micro)),
                                  loss=loss, gradient_norm_sum=grad, seconds=seconds,
                                  input_seconds=input_seconds, compute_seconds=compute_seconds,
                                  input_path=input_path,
                                  grouped_by_real_n=grouped, crop_padding=grouped,
                                  data_parallel=isinstance(model, torch.nn.DataParallel),
                                  samples_per_second=n / seconds,
                                  peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
                                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                                  allocated_gib=torch.cuda.memory_allocated() / 2**30,
                                  reserved_gib=torch.cuda.memory_reserved() / 2**30,
                                  secondary_peak_cuda_gib=torch.cuda.max_memory_allocated(1) / 2**30 if dual_runtime else None,
                                  secondary_peak_reserved_gib=torch.cuda.max_memory_reserved(1) / 2**30 if dual_runtime else None,
                                  learning_rate=optimizer.param_groups[0]["lr"], oom_retries=retries)
                    append_json(run / "updates.jsonl", record)
                    if updates == 1:
                        if (batch[0].shape[2] != (2 if task == "TSP" else 3)
                                or batch[2].shape != (BATCH, k) or batch[4].shape != batch[0].shape[:2]
                                or not training_input_ready(batch[0], device, input_path)):
                            raise ValueError("The single real NSS smoke batch failed shape/device/pinning checks")
                        write_json(run / "smoke_update.json", dict(record, x_shape=list(batch[0].shape),
                                                                 costs_shape=list(batch[2].shape),
                                                                 input_device=str(batch[0].device),
                                                                 pinned=batch[0].device.type == "cpu" and batch[0].is_pinned(),
                                                                 formal_optimizer_update=True))
                    if updates <= 3 or updates % 20 == 0:
                        eta = (plan["updates_per_model"] - updates) * seconds / 3600
                        print(f"UPDATE {task} {updates}/{plan['updates_per_model']} epoch={epoch} "
                              f"batch={n} micro={micro} loss={loss:.6f} "
                              f"seconds={seconds:.3f} samples/s={n / seconds:.1f} ETA_h={eta:.2f}", flush=True)
                        write_json(run / "status.json", dict(record, state="training", eta_hours=eta))
                        write_json(run / "gpu_telemetry.json", gpu_telemetry(task, execution["resource"]["uuids"][0]))
                        if dual_runtime:
                            write_json(run / "gpu_telemetry_secondary.json", gpu_telemetry(task, DUAL_RESOURCES[task]["uuids"][1]))
                    tracker.log({"train/loss": loss, "train/gradient_norm_sum": grad,
                                 "train/lr": record["learning_rate"], "train/micro_batch": micro,
                                 "speed/samples_per_sec": n / seconds,
                                 "gpu/peak_allocated_gib": record["peak_cuda_gib"]}, step=updates)
                    if pause_requested and batch_index < len(loader) - 1:
                        state = dict(epoch=epoch, updates=updates, best=best, best_epoch=best_epoch,
                                     model_state_dict=unwrap_model(model).state_dict(),
                                     optimizer_state_dict=optimizer.state_dict(), config=config,
                                     original_sources=plan["original_sources"], shuffle_rng=shuffle_rng.get_state(),
                                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                     cuda_rng_all=torch.cuda.get_rng_state_all() if dual_runtime else None,
                                     micro_batch=micro, execution=execution, selection_split="val", selection_metric="gap",
                                     cursor=dict(epoch=epoch, next_batch_index=batch_index + 1, epoch_complete=False),
                                     loader_state=loader.state_dict(), epoch_loss_sum=epoch_loss,
                                     epoch_rows=epoch_rows, epoch_elapsed_seconds=time.monotonic() - epoch_started)
                        torch.save(state, run / "last.pt.tmp")
                        os.replace(run / "last.pt.tmp", run / "last.pt")
                        write_json(run / "status.json", dict(state="paused_at_logical_batch_boundary", updates=updates,
                                                             cursor=state["cursor"], best=best, best_epoch=best_epoch))
                        print(f"SAFE_PAUSE {task} update={updates} cursor={state['cursor']}", flush=True)
                        return
                    fetch_started = time.monotonic()
                scores, val_seconds = infer(model, val_loader, device)
                metrics, predicted = summarize(scores, val_cache, train_sbs)
                improved = metrics["top_1"] < best
                if improved:
                    best, best_epoch = metrics["top_1"], epoch
                state = dict(epoch=epoch, updates=updates, best=best, best_epoch=best_epoch,
                             model_state_dict=unwrap_model(model).state_dict(), optimizer_state_dict=optimizer.state_dict(),
                             config=config, original_sources=plan["original_sources"],
                             shuffle_rng=shuffle_rng.get_state(), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state(), micro_batch=micro,
                             cuda_rng_all=torch.cuda.get_rng_state_all() if dual_runtime else None,
                             execution=execution,
                             cursor=dict(epoch=epoch + 1, next_batch_index=0, epoch_complete=True),
                             selection_split="val", selection_metric=metrics["metric_name"],
                             validation=metrics)
                if improved:
                    torch.save(state, run / "best.pt.tmp")
                    os.replace(run / "best.pt.tmp", run / "best.pt")
                    save_predictions(run / "best_val_predictions.npz", scores, val_cache, predicted)
                    write_json(run / "best_validation.json", dict(metrics, epoch=epoch, updates=updates))
                torch.save(state, run / "last.pt.tmp")
                os.replace(run / "last.pt.tmp", run / "last.pt")
                record = dict(epoch=epoch, updates=updates, train_loss=epoch_loss / epoch_rows,
                              augmented_rows=epoch_rows, seconds=time.monotonic() - epoch_started,
                              validation_seconds=val_seconds, improved=improved, best=best, best_epoch=best_epoch,
                              validation=metrics)
                append_json(run / "epochs.jsonl", record)
                tracker.log({"val/top1": metrics["top1"], "val/top2": metrics["top2"],
                             "val/top3": metrics["top3"], "val/selection_metric": metrics["top_1"],
                             "val/mean_cost": metrics["mean_cost"], "epoch": epoch}, step=updates)
                print(f"VALID {task} epoch={epoch} {metrics['metric_name']}={metrics['top_1']:.6f} "
                      f"best={best:.6f}@{best_epoch} top1={metrics['top1']:.4f}", flush=True)
                del scores, state
                if pause_requested:
                    write_json(run / "status.json", dict(state="paused_at_epoch_boundary", updates=updates,
                                                         best=best, best_epoch=best_epoch))
                    print(f"SAFE_PAUSE {task} update={updates} epoch_complete=True", flush=True)
                    return
            if updates != plan["updates_per_model"]:
                raise RuntimeError("Original 50-epoch NSS update count was not completed")
            write_json(run / "training_complete.json", dict(task=task, updates=updates, epochs=50,
                                                            best_epoch=best_epoch, best_metric=best,
                                                            best_sha256=sha256(run / "best.pt"),
                                                            seconds=time.monotonic() - launched))
            write_json(run / "status.json", dict(state="training_complete", updates=updates, best_epoch=best_epoch))
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            tracker.finish()


def freeze(root):
    with file_lock(root / "freeze.lock"):
        if (root / "final_test_lock.json").exists():
            for task in GPUS:
                check_boundary(root, task, "test")
            return
        plan = read_json(root / "protocol.json")
        records = {}
        validation_records = {}
        for task in GPUS:
            run = root / task
            complete = read_json(run / "training_complete.json")
            best = read_json(run / "best_validation.json")
            if complete["updates"] != plan["updates_per_model"] or best["metric_name"] != "gap":
                raise ValueError("Both models must finish the same original VALID rule before TEST")
            records[task] = dict(path=str(run / "best.pt"), sha256=sha256(run / "best.pt"),
                                 best_epoch=best["epoch"], val_top_1=best["top_1"], criterion=best["metric_name"])
            if records[task]["sha256"] != complete["best_sha256"]:
                raise ValueError("Best checkpoint changed after training")
            predictions = run / "best_val_predictions.npz"
            records[task]["validation_predictions"] = dict(path=str(predictions), sha256=sha256(predictions),
                                                           rows=best["n"], split="val")
            validation_records[task] = dict(
                checkpoint=records[task], best_updates=best["updates"], epoch_index_base=0,
                best_validation=dict(path=str(run / "best_validation.json"),
                                     sha256=sha256(run / "best_validation.json"),
                                     gap=best["top_1"], top1=best["top1"], top1_native_ind=best["top1_native_ind"]),
                pool=best["pool"], train_rows=COUNTS["train"], validation_rows=best["n"],
                train_sbs=plan["tasks"][task]["train_sbs"],
                train_metadata=str(directory(task, "train") / "metadata.json"),
                train_cost_cache=str(root / "cpu_cache" / (task + "train.pt")),
                validation_cache=str(root / "cpu_cache" / (task + "val.pt")))
        created = time.time()
        # Reuse scores saved by the selected epoch's existing VALID pass; no inference.
        write_json(root / "locked_train_val_handoff.json", dict(
            state="locked", allowed_splits=["train", "val"], records=validation_records,
            data_root=str(DATA), original_sources=plan["original_sources"], selected_using="VALID only",
            prediction_fields=["scores", "pred", "winner", "costs", "pool", "indices", "source_indices"],
            winner_rule="winner is original raw-label ind, FP64 cost argmin; pred is original NSS selection",
            row_units="VAL has unaugmented base instances; training batch1024 counts augmented entries",
            test_lib_results_included=False, inference_repeated=False, created_unix=created))
        write_json(root / "final_test_lock.json", dict(checkpoints=records,
                                                      original_sources=plan["original_sources"],
                                                      data_root=str(DATA), selected_using="VALID only",
                                                      locked_train_val_handoff=str(root / "locked_train_val_handoff.json"),
                                                      test_unique_once=True, created_unix=created))
    print("FROZEN both VALID-best checkpoints; TEST/LIB can now open", flush=True)


def evaluate(root, task, workers, dual_runtime=False, single4090=False):
    device = require_gpu(task, dual_runtime, single4090)
    run = root / task
    with file_lock(run / "task.lock"):
        check_boundary(root, task, "test")
        plan = read_json(root / "protocol.json")
        _, config = recipe(task)
        model = make_model(config).to(device)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        frozen = read_json(root / "final_test_lock.json")["checkpoints"][task]
        for split in ("test", task + "LIB"):
            receipt = run / (split + "_once.json")
            if receipt.exists():
                value = read_json(receipt)
                if value["state"] != "complete" or value["checkpoint_sha256"] != frozen["sha256"]:
                    raise RuntimeError("An evaluation was already attempted; do not repeat TEST/LIB inference")
                print(f"REUSE {task} {split} already evaluated exactly once", flush=True)
                continue
            value = dict(state="started", checkpoint_sha256=frozen["sha256"],
                         checkpoint=str(run / "best.pt"), started_unix=time.time())
            with receipt.open("x") as stream:
                json.dump(value, stream)
            try:
                cache = torch.load(prepare_cache(root, task, split), map_location="cpu", weights_only=False)
                loader, _ = make_loader(NSSDataset(cache), 8, workers)
                scores, elapsed = infer(model, loader, device)
                metrics, predicted = summarize(scores, cache, plan["tasks"][task]["train_sbs"])
                predictions = run / (split + "_predictions.npz")
                save_predictions(predictions, scores, cache, predicted)
                metrics.update(task=task, split=split, checkpoint=frozen, inference_seconds=elapsed,
                               original_directory=str(directory(task, split)),
                               time_columns="compatibility placeholders, not measured solver runtime")
                write_json(run / (split + "_metrics.json"), metrics)
                write_json(receipt, dict(value, state="complete", rows=metrics["n"], seconds=elapsed,
                                         predictions_sha256=sha256(predictions), completed_unix=time.time()))
                print(f"EVALUATED {task} {split} n={metrics['n']} top1={metrics['top1']:.4f} "
                      f"mean_cost={metrics['mean_cost']:.6f}", flush=True)
                del loader, cache, scores
            except Exception as exc:
                write_json(receipt, dict(value, state="failed", error=str(exc)))
                raise


def report(root):
    rows = []
    for task in GPUS:
        for split in ("test", task + "LIB"):
            rows.append(read_json(root / task / (split + "_metrics.json")))
    write_json(root / "results.json", rows)
    fields = ["task", "split", "n", "top1", "top2", "top3", "mean_cost", "train_sbs_solver",
              "train_sbs_cost", "sbs_solver", "sbs_cost", "oracle_cost", "mean_oracle_gap_pct"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# R79 Original NSS: New Data", "", "Both task-specific checkpoints were selected using VALID only.",
             "Each new TEST and original LIB was evaluated once with its corresponding frozen checkpoint.", "",
             "| Task/split | n | Top1 | Top2 | Top3 | Mean cost | TRAIN-fixed SBS | Oracle |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['task']}/{r['split']} | {r['n']} | {r['top1']:.4f} | {r['top2']:.4f} | "
                     f"{r['top3']:.4f} | {r['mean_cost']:.6f} | {r['train_sbs_cost']:.6f} | {r['oracle_cost']:.6f} |")
    lines.extend(["", "FP64 raw costs are retained in reports; original FP32 RankingLoss and FP32 winner metrics are unchanged.",
                  "New-data gap is retained-pool Oracle regret, not an optimal-tour gap. Solver time labels are placeholders.",
                  "Post-hoc split SBS is descriptive only; deployment SBS comes from TRAIN.", ""])
    (root / "REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in GPUS:
        run = root / task
        epochs = [json.loads(line) for line in (run / "epochs.jsonl").read_text().splitlines()]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x["epoch"] + 1 for x in epochs], [x["train_loss"] for x in epochs], color="#22754b")
        ax.set(xlabel="Epoch", ylabel="Original NSS ranking loss", title=task)
        fig.tight_layout()
        fig.savefig(run / "loss_curve.png", dpi=160)
        plt.close(fig)
        metrics = [read_json(run / "best_validation.json")]
        metrics[0]["split"] = "val_best"
        metrics += [x for x in rows if x["task"] == task]
        for result in metrics:
            out = run / "figures" / result["split"]
            out.mkdir(parents=True, exist_ok=True)
            names = result["pool"] + ["Selector"]
            for filename, key, ylabel in (
                ("top1_vs_single_methods.png", "top1", "Top-1 accuracy"),
                ("mean_cost_vs_single_methods.png", "mean_cost", "Mean raw cost"),
            ):
                values = [x[key] for x in result["methods"]] + [result[key]]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(names, values, color=["#687f8b"] * (len(names) - 1) + ["#22754b"])
                ax.tick_params(axis="x", labelrotation=55)
                ax.set(ylabel=ylabel, title=task + " " + result["split"])
                fig.tight_layout()
                fig.savefig(out / filename, dpi=160)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(result["pool"], list(result["arm_distribution"].values()), color="#b06e38")
            ax.tick_params(axis="x", labelrotation=55)
            ax.set(ylabel="Selection rate", title=task + " " + result["split"])
            fig.tight_layout()
            fig.savefig(out / "arm_distribution.png", dpi=160)
            plt.close(fig)
    write_json(root / "completion.json", dict(state="complete", tasks=list(GPUS),
                                              test_unique_once=True, completed_unix=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "freeze", "evaluate", "report"], required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--task", choices=list(GPUS))
    parser.add_argument("--micro-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-path", choices=["cpu_pinned", "gpu_resident"], default="cpu_pinned")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--check-grouped", action="store_true")
    parser.add_argument("--dual-runtime", action="store_true")
    parser.add_argument("--check-parallel", action="store_true")
    parser.add_argument("--single4090", action="store_true")
    args = parser.parse_args()
    if args.optimize and args.check_grouped:
        parser.error("The single grouped parity gate must not rerun the micro capacity comparison")
    if args.check_parallel and (not args.dual_runtime or args.stage != "train"):
        parser.error("--check-parallel is only the one-batch gate for dual TRAIN migration")
    if args.dual_runtime and (args.optimize or args.check_grouped):
        parser.error("DP migration must not rerun grouping or micro-capacity searches")
    if args.single4090 and (args.dual_runtime or args.optimize or args.check_parallel or args.check_grouped):
        parser.error("Single4090 fallback does not run another capacity or equivalence search")
    if not 1 <= args.micro_batch <= BATCH or not 0 <= args.workers <= 2 or not 1 <= args.threads <= 4:
        parser.error("Invalid bounded execution resources")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.stage == "prepare":
        prepare(args.root, args.workers)
    elif args.stage == "freeze":
        freeze(args.root)
    elif args.stage == "report":
        report(args.root)
    else:
        if args.task is None:
            parser.error("--task is required for train/evaluate")
        if args.stage == "train":
            train(args.root, args.task, args.micro_batch, args.workers, args.resume, args.input_path,
                  args.optimize, args.check_grouped, args.dual_runtime, args.check_parallel, args.single4090)
        else:
            evaluate(args.root, args.task, args.workers, args.dual_runtime, args.single4090)


if __name__ == "__main__":
    main()
