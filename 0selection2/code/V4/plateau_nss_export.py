"""Bounded CPU export of the two frozen NSS rank checkpoints.

No training, augmentation, prepare_dataset, working-directory assumptions, or
changes to NSS are needed. See nss_export_manifest.json for the export contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import signal
import socket
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "4"

ROOT = Path(__file__).resolve().parents[2]
NSS = ROOT / "code" / "NSS_retrain"
OUTPUT = ROOT / "code" / "V4" / "runs" / "Plateau_diagnosis_R70"
PREDICTIONS = OUTPUT / "nss_predictions"
MANIFEST = OUTPUT / "nss_export_manifest.json"
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(NSS))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import SelectionDataset, collate_fn
from model import Selection_model
from utils import process_instance_CVRP
from unified_selector.registry import GLOBAL_SOLVERS, POOLS, S2I


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = PREDICTIONS / (path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model(problem: str) -> tuple[Selection_model, dict]:
    config_path = NSS / f"config_{problem}_ours.yml"
    saved_path = NSS / "train_logs" / f"config_{problem}_ours_rank_2024" / "config.json"
    checkpoint_path = saved_path.parent / "checkpoint_epoch_best.pt"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    params = dict(config["model_params"], problem_type=problem,
                  output_dim=config["train_params"]["num_classes"],
                  ns_feature=config["train_params"]["ns_feature"])
    if params["output_dim"] != len(POOLS[problem]):
        raise ValueError(f"{problem}: checkpoint output width differs from the registry")
    for key, value in params.items():
        original = saved["train_params"][key] if key == "ns_feature" else saved["model_params"][key]
        if value != original:
            raise ValueError(f"{problem}: current and saved architecture differ at {key}")
    if config["train_params"]["manual_feature"] or params["ns_feature"]:
        raise ValueError("Only the exact requested no-manual-feature rank checkpoints are supported")
    checkpoint_identity = identity(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Selection_model(**params).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    metadata = dict(checkpoint_identity, epoch=int(checkpoint["epoch"]),
                    config=identity(config_path), saved_config=identity(saved_path),
                    model_params=params, state_tensor_count=len(checkpoint["model_state_dict"]),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    pool=list(POOLS[problem]), pool_ids=[S2I[s] for s in POOLS[problem]])
    return model, metadata


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(manifest: dict, manifest_path: Path, event: str, **values) -> None:
    elapsed = time.monotonic() - manifest["_started"]
    record = dict(event=event, utc=utc_now(), elapsed_seconds=elapsed, **values)
    with (PREDICTIONS / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    logging.info("%s", json.dumps(record, allow_nan=False))
    manifest["last_progress"] = record
    manifest["elapsed_seconds"] = elapsed
    write_json(manifest_path, {k: v for k, v in manifest.items() if not k.startswith("_")})


def load_split(problem: str, split: str) -> dict:
    directory = ROOT / "data" / f"{problem}{split}"
    with (directory / "dataset.pkl").open("rb") as stream:
        instances = pickle.load(stream)
    with (directory / "raw_label.pkl").open("rb") as stream:
        labels = pickle.load(stream)
    n = len(instances)
    if set(labels) != {str(i) for i in range(n)}:
        raise ValueError(f"{problem}/{split}: raw labels are not consecutive source indices")
    rows = [labels[str(i)] for i in range(n)]
    pool = list(POOLS[problem])
    result_names = [p.name[len("result_"):-len(".txt")]
                    for p in sorted((directory / "results").glob("result_*.txt"))]
    if result_names != pool:
        raise ValueError(f"{problem}/{split}: result files disagree with the canonical registry pool")
    # Do not turn an already-rounded FP32 cache into apparently raw FP64 costs.
    costs = np.asarray([row["cost"] for row in rows])
    if costs.dtype != np.float64 or costs.shape != (n, len(pool)) or not np.isfinite(costs).all():
        raise ValueError(f"{problem}/{split}: expected finite, original FP64 label costs")
    winner = np.asarray([row["ind"] for row in rows], dtype=np.int64)
    if ((winner < 0) | (winner >= len(pool))).any():
        raise ValueError(f"{problem}/{split}: native winner out of range")
    if problem == "CVRP":
        instances = [process_instance_CVRP(instance) for instance in instances]
    nodes = np.asarray([instance.shape[1] for instance in instances], dtype=np.int64)
    if (nodes <= 1).any() or any(instance.device.type != "cpu" or instance.dtype != torch.float32
                               for instance in instances):
        raise ValueError(f"{problem}/{split}: inputs must be original CPU FP32 node tensors")
    dataset = SelectionDataset(instances, [[row["ind"], row["cost"], row["time"], row["gap"]]
                                           for row in rows], manual_feature=False, data_aug=False)
    sizes, counts = np.unique(nodes, return_counts=True)
    return dict(dataset=dataset, costs=costs, winner=winner, nodes=nodes,
                pool=pool, pool_ids=np.asarray([S2I[s] for s in pool], dtype=np.int64),
                source=dict(dataset=identity(directory / "dataset.pkl"),
                            labels=identity(directory / "raw_label.pkl"), instances=n,
                            cost_dtype=str(costs.dtype), node_min=int(nodes.min()),
                            node_max=int(nodes.max()),
                            node_histogram={str(s): int(c) for s, c in zip(sizes, counts)}))


def metric_summary(logits: np.ndarray, costs: np.ndarray, winner: np.ndarray,
                   costs_fp32: np.ndarray) -> dict:
    scores = torch.from_numpy(logits)
    labels_fp32 = torch.from_numpy(costs_fp32)
    top = scores.topk(min(3, scores.shape[1]), dim=1).indices
    historical_pred = top[:, 0]
    greedy = scores.softmax(dim=1).argmax(dim=1)
    oracle_fp32 = labels_fp32.argmin(dim=1)
    native = torch.from_numpy(winner)
    hits = {f"top{k}": (top[:, :k] == oracle_fp32[:, None]).any(dim=1)
            for k in (1, 2, 3)}
    fp32 = {key: float(value.float().mean()) for key, value in hits.items()}
    fp32.update({f"{key}_count": int(value.sum()) for key, value in hits.items()})
    fp32["mean_cost"] = float(labels_fp32.gather(1, historical_pred[:, None]).mean())
    native_metrics = {f"top{k}": float((top[:, :k] == native[:, None]).any(dim=1).float().mean())
                      for k in (1, 2, 3)}
    native_metrics["greedy_top1"] = float((greedy == native).float().mean())
    picked = costs[np.arange(len(costs)), greedy.numpy()]
    oracle_costs = costs.min(axis=1)
    sbs_idx = int(costs.mean(axis=0).argmin())
    return dict(historical_fp32=fp32, native_winner=native_metrics,
                raw_fp64=dict(mean_cost=float(picked.mean()),
                              oracle_cost=float(oracle_costs.mean()),
                              sbs_cost=float(costs[:, sbs_idx].mean()), sbs_idx=sbs_idx,
                              mean_regret=float((picked - oracle_costs).mean())),
                native_vs_fp32_winner_disagreements=int((native != oracle_fp32).sum()),
                native_vs_fp64_argmin_disagreements=int(np.count_nonzero(winner != costs.argmin(axis=1))),
                greedy_vs_topk_first_disagreements=int((greedy != historical_pred).sum()),
                prediction_counts=np.bincount(greedy.numpy(), minlength=costs.shape[1]).tolist(),
                native_winner_counts=np.bincount(winner, minlength=costs.shape[1]).tolist(),
                fp32_winner_counts=np.bincount(oracle_fp32.numpy(), minlength=costs.shape[1]).tolist())


def save_export(path: Path, arrays: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    # Focused serialization/contract check, not a repository-wide test run.
    with np.load(temporary, allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value, err_msg=key)
        if saved["costs"].dtype != np.float64 or saved["logits"].dtype != np.float32:
            raise ValueError("Export changed raw cost or inference precision")
        n = len(saved["indices"])
        if saved["costs"].shape != saved["logits"].shape or saved["nodes"].shape != (n,):
            raise ValueError("Invalid prediction export shapes")
    temporary.replace(path)
    return identity(path)


def export_split(problem: str, split: str, model: Selection_model, raw: dict,
                 manifest: dict, manifest_path: Path, deadline: float, smoke: bool = False,
                 selection: np.ndarray | None = None) -> dict:
    key = f"{split}/{problem}"
    total_n = len(raw["dataset"])
    target_n = len(selection) if selection is not None else (min(8, total_n) if smoke else total_n)
    dataset = raw["dataset"]
    if smoke:
        dataset = torch.utils.data.Subset(dataset, range(target_n))
    elif selection is not None:
        dataset = torch.utils.data.Subset(dataset, selection.tolist())
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=False,
                        num_workers=0, pin_memory=False, drop_last=False)
    record = dict(status="running", source=raw["source"], expected_instances=total_n,
                  target_instances=target_n, exported_instances=0,
                  source_split="train" if selection is not None else split,
                  is_training_sample=selection is not None,
                  checkpoint_sha256=manifest["checkpoints"][problem]["sha256"])
    manifest["splits"][key] = record
    emit(manifest, manifest_path, "split_started", split=split, problem=problem,
         instances=total_n, target_instances=target_n,
         max_nodes=raw["source"]["node_max"])
    split_started = time.monotonic()
    scores, labels_fp32, source_indices = [], [], []
    inference_seconds = 0.0
    completed = 0
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                if time.monotonic() >= deadline or manifest.get("_stop", False):
                    break
                indices = np.asarray(batch[-1], dtype=np.int64)
                expected = (np.arange(completed, completed + len(indices)) if selection is None else
                            selection[completed:completed + len(indices)])
                np.testing.assert_array_equal(indices, expected)
                np.testing.assert_array_equal(batch[3].numpy().astype(np.int64), raw["nodes"][indices])
                np.testing.assert_array_equal(batch[1].numpy(), raw["winner"][indices])
                np.testing.assert_array_equal(batch[2].numpy(), raw["costs"][indices].astype(np.float32))
                before = time.monotonic()
                logits = model(batch[0], batch[3], None, batch[4])
                inference_seconds += time.monotonic() - before
                if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f"{key}: nonfinite or non-FP32 logits")
                scores.append(logits.cpu().numpy().copy())
                labels_fp32.append(batch[2].numpy().copy())
                source_indices.append(indices)
                completed += len(indices)
                if batch_number in (1, 8) or batch_number % 16 == 0 or completed == target_n:
                    elapsed = time.monotonic() - split_started
                    rate = completed / elapsed
                    record.update(exported_instances=completed, elapsed_seconds=elapsed,
                                  instances_per_second=rate, eta_seconds=(target_n - completed) / rate,
                                  inference_seconds=inference_seconds)
                    event = "throughput_pilot" if batch_number == 8 else "batch_progress"
                    values = dict(split=split, problem=problem, rows=completed, total_rows=target_n,
                                  split_elapsed_seconds=elapsed, sample_throughput=rate,
                                  eta_seconds=record["eta_seconds"])
                    if batch_number == 8:
                        record["pilot"] = dict(sample_rows=completed, elapsed_seconds=elapsed,
                                               instances_per_second=rate,
                                               nominal_10000_train_eta_seconds=10000 / rate)
                        values["nominal_10000_train_eta_seconds"] = 10000 / rate
                    emit(manifest, manifest_path, event, **values)
    finally:
        record.update(exported_instances=completed, elapsed_seconds=time.monotonic() - split_started,
                      inference_seconds=inference_seconds)
        if completed:
            logits = np.concatenate(scores)
            fp32 = np.concatenate(labels_fp32)
            indices = np.concatenate(source_indices)
            costs, winner, nodes = (raw[field][indices] for field in ("costs", "winner", "nodes"))
            pred = torch.from_numpy(logits).softmax(dim=1).argmax(dim=1).numpy()
            top3 = torch.from_numpy(logits).topk(min(3, logits.shape[1]), dim=1).indices.numpy()
            checkpoint = manifest["checkpoints"][problem]
            arrays = dict(indices=indices, logits=logits, costs=costs, winner=winner,
                          pool=np.asarray(raw["pool"]), pool_ids=raw["pool_ids"], nodes=nodes,
                          native_winner=winner, pred=pred, pred_global_id=raw["pool_ids"][pred],
                          winner_global_id=raw["pool_ids"][winner], top3=top3,
                          winner_fp32=fp32.argmin(axis=1), winner_fp64=costs.argmin(axis=1),
                          costs_fp32=fp32, customer_sizes=nodes - (1 if problem == "CVRP" else 0),
                          global_solver_names=np.asarray(GLOBAL_SOLVERS),
                          checkpoint_sha256=np.asarray(checkpoint["sha256"]),
                          checkpoint_path=np.asarray(checkpoint["path"]),
                          checkpoint_epoch=np.asarray(checkpoint["epoch"], dtype=np.int64),
                          problem=np.asarray(problem), split=np.asarray(split))
            subdirectory = "smoke" if smoke else ("partial" if completed != target_n else "")
            path = PREDICTIONS / subdirectory / split / f"{problem}.npz"
            record["output"] = save_export(path, arrays)
            record["metrics"] = metric_summary(logits, costs, winner, fp32)
            record["complete"] = completed == total_n and not smoke and selection is None
            record["sample_complete"] = selection is not None and completed == target_n
            record["status"] = ("smoke_complete" if smoke else
                                ("sample_complete" if record["sample_complete"] else
                                 ("complete" if completed == target_n else "partial")))
            record["instances_per_second"] = completed / record["elapsed_seconds"]
        else:
            record.update(status="not_exported", complete=False)
        emit(manifest, manifest_path, "split_finished", split=split, problem=problem,
             status=record["status"], rows=completed, expected_rows=total_n,
             split_elapsed_seconds=record["elapsed_seconds"])
    return record


def historical_check(manifest: dict) -> dict:
    path = NSS / "results" / "nss_retrain_rank_test_summary.json"
    historical = json.loads(path.read_text(encoding="utf-8"))
    mapping = {"TSP": "test/TSP", "CVRP": "test/CVRP", "TSPLIB": "LIB/TSP", "CVRPLIB": "LIB/CVRP"}
    result = dict(source=identity(path), necessary_not_sufficient=True, tolerance=5e-8,
                  metric_definition="original torch.topk(logits) vs original collator FP32 cost argmin",
                  rows=[], aggregate=None,
                  explanation=[
                      "Native stored winners and raw FP64 argmins are not used for this historical check.",
                      "Any native-vs-FP32 label disagreement is measured separately; rounding can change exact tie winners.",
                      "The exporter is CPU-only; the historical evaluator defaults to cuda:0 with CPU fallback.",
                      "The historical JSON has no checkpoint/data hashes or execution-device record, so it cannot prove artifact identity.",
                      "No parameters, preprocessing, batch composition, or predictions are tuned to match the historical values."
                  ])
    for old in historical["rows"]:
        key = mapping[old["problem"]]
        current = manifest["splits"].get(key, {})
        if not current.get("complete"):
            result["rows"].append(dict(problem=old["problem"], status="unavailable"))
            continue
        metrics = current["metrics"]["historical_fp32"]
        delta = {metric: metrics[metric] - old[metric] for metric in ("top1", "top3")}
        count_delta = {metric: metrics[f"{metric}_count"] - round(old[metric] * old["n"])
                       for metric in ("top1", "top3")}
        matched = current["exported_instances"] == old["n"] and all(abs(v) <= result["tolerance"] for v in delta.values())
        result["rows"].append(dict(problem=old["problem"], status="match" if matched else "mismatch",
                                   n=current["exported_instances"],
                                   historical={k: old[k] for k in ("top1", "top3")},
                                   current={k: metrics[k] for k in ("top1", "top3")},
                                   deltas=delta, count_deltas=count_delta,
                                   native_vs_fp32_winner_disagreements=current["metrics"]["native_vs_fp32_winner_disagreements"],
                                   pick_counts_match=current["metrics"]["prediction_counts"] == [int(old["pick_counts"].get(str(i), 0)) for i in range(len(current["metrics"]["prediction_counts"]))],
                                   oracle_counts_match=current["metrics"]["fp32_winner_counts"] == [int(old["oracle_counts"].get(str(i), 0)) for i in range(len(current["metrics"]["fp32_winner_counts"]))]))
    available = all(row["status"] != "unavailable" for row in result["rows"])
    if available:
        rows = [manifest["splits"][key] for key in mapping.values()]
        n = sum(row["exported_instances"] for row in rows)
        values = {metric: sum(row["metrics"]["historical_fp32"][metric] * row["exported_instances"] for row in rows) / n
                  for metric in ("top1", "top3")}
        delta = {k: values[k] - historical["all"][k] for k in values}
        result["aggregate"] = dict(n=n, current=values,
                                   historical={k: historical["all"][k] for k in values}, deltas=delta,
                                   status="match" if n == historical["all"]["n"] and all(abs(v) <= result["tolerance"] for v in delta.values()) else "mismatch")
    result["status"] = ("incomplete" if not available else
                        ("match" if all(row["status"] == "match" for row in result["rows"]) else "mismatch"))
    if result["status"] == "mismatch":
        result["explanation"].append(
            "Historical FP32-label parity did not pass; treat predictions as a CPU frozen-checkpoint export, not an exact historical reproduction. CPU/CUDA arithmetic and tied pooling topk are plausible but unproven causes; historical per-instance logits and artifact hashes are absent. No native-label substitution explains a mismatch in this already-FP32 comparison.")
    return result


def run_export(args: argparse.Namespace, models: dict, manifest: dict, manifest_path: Path) -> int:
    deadline = manifest["_started"] + args.max_seconds
    priority = [("TSP", "val"), ("CVRP", "val"), ("TSP", "test"),
                ("CVRP", "test"), ("TSP", "LIB"), ("CVRP", "LIB")]
    if args.smoke:
        priority = priority[:2]
    for problem, split in priority:
        if time.monotonic() >= deadline or manifest.get("_stop"):
            break
        raw = load_split(problem, split)
        record = export_split(problem, split, models[problem], raw, manifest, manifest_path, deadline, args.smoke)
        if not args.smoke and not record["complete"]:
            break
    if args.smoke:
        manifest["status"] = "smoke_complete"
        return 0
    heldout_complete = all(manifest["splits"].get(f"{split}/{problem}", {}).get("complete")
                          for problem, split in priority)
    manifest["historical_check"] = historical_check(manifest)
    emit(manifest, manifest_path, "heldout_finished", complete=heldout_complete,
         historical_status=manifest["historical_check"]["status"])
    if not heldout_complete:
        manifest["status"] = "bounded_partial_heldout"
        manifest["train_decision"] = dict(status="skipped", reason="held-out exports did not finish within the bound")
        return 2
    if args.train == "skip" or manifest.get("_stop"):
        manifest["status"] = "complete_heldout_train_skipped"
        manifest["train_decision"] = dict(status="skipped", reason="explicit --train skip or stop signal")
        return 0
    # Only now read train data; use its actual sizes and counts for admission.
    train = {problem: load_split(problem, "train") for problem in ("TSP", "CVRP")}
    estimates = {}
    for problem, raw in train.items():
        references = [manifest["splits"][f"{split}/{problem}"] for split in ("val", "test")]
        rows = sum(reference["exported_instances"] for reference in references)
        seconds = sum(reference["elapsed_seconds"] for reference in references)
        # Attention cost tracks padded batch node count squared, not just rows.
        def attention_work(nodes: np.ndarray) -> int:
            return sum(len(nodes[i:i + 8]) * int(nodes[i:i + 8].max()) ** 2 for i in range(0, len(nodes), 8))
        reference_work = 0
        for split in ("val", "test"):
            with np.load(PREDICTIONS / split / f"{problem}.npz", allow_pickle=False) as saved:
                reference_work += attention_work(saved["nodes"])
        estimate = max(len(raw["dataset"]) * seconds / rows,
                       attention_work(raw["nodes"]) * seconds / reference_work)
        estimates[problem] = dict(instances=len(raw["dataset"]), reference_rows=rows,
                                  measured_instances_per_second=rows / seconds,
                                  estimated_seconds=estimate, conservative_seconds=estimate * 1.5,
                                  source=raw["source"])
    conservative = sum(value["conservative_seconds"] for value in estimates.values())
    remaining = deadline - time.monotonic()
    admitted = conservative + 30 < remaining
    manifest["train_decision"] = dict(status="admitted" if admitted else "skipped",
                                      estimates=estimates, conservative_seconds=conservative,
                                      remaining_seconds=remaining, safety_factor=1.5,
                                      reason="measured full-train estimate fits with reserve" if admitted else
                                      "measured full-train estimate exceeds remaining wall budget")
    emit(manifest, manifest_path, "train_admission", admitted=admitted,
         conservative_eta_seconds=conservative, remaining_seconds=remaining, estimates=estimates)
    if not admitted:
        # Retain every exact train size without presenting uncomputed logits.
        for problem, raw in train.items():
            manifest["train_decision"]["estimates"][problem]["exact_nodes"] = raw["nodes"].tolist()
        manifest["status"] = "complete_heldout_train_skipped"
        return 0
    for problem, raw in train.items():
        record = export_split(problem, "train", models[problem], raw, manifest, manifest_path, deadline)
        if not record["complete"]:
            manifest["status"] = "complete_heldout_partial_train"
            return 0
    manifest["train_decision"]["status"] = "completed"
    manifest["status"] = "complete_all_splits"
    return 0


def stratified_training_indices(nodes: np.ndarray, problem: str, count: int = 2000) -> tuple[np.ndarray, dict]:
    sizes = nodes - (1 if problem == "CVRP" else 0)
    count = min(count, len(nodes))
    edges = np.linspace(int(sizes.min()), int(sizes.max()) + 1, 11)
    bins = np.searchsorted(edges, sizes, side="right") - 1
    population_counts = np.bincount(bins, minlength=10)
    exact_quotas = count * population_counts / len(nodes)
    quotas = np.floor(exact_quotas).astype(np.int64)
    order = sorted(range(10), key=lambda i: (-(exact_quotas[i] - quotas[i]), i))
    for bin_id in order[:count - int(quotas.sum())]:
        quotas[bin_id] += 1
    rng = np.random.Generator(np.random.PCG64(2024))
    selected = [rng.choice(np.flatnonzero(bins == i), size=int(quotas[i]), replace=False)
                for i in range(10) if quotas[i]]
    indices = np.sort(np.concatenate(selected)).astype(np.int64)
    rule = dict(population_instances=len(nodes), sample_instances=len(indices),
                fraction=len(indices) / len(nodes), seed=2024, generator="NumPy PCG64",
                numpy_version=np.__version__, strata=10,
                size_definition="customers (depot excluded)" if problem == "CVRP" else "nodes",
                bin_edges=edges.tolist(), population_counts=population_counts.tolist(),
                sample_quotas=quotas.tolist(),
                allocation="proportional largest remainder; ties by ascending bin id",
                within_stratum="seeded uniform sampling without replacement",
                output_order="ascending original dataset index; original NSS batch size 8",
                selection_uses=["training instance node sizes", "original indices"],
                selection_excludes=["costs", "native winners", "logits", "held-out performance"],
                full_train_available=False)
    return indices, rule


def append_training_samples(args: argparse.Namespace) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    heldout_keys = [f"{split}/{problem}" for split in ("val", "test", "LIB") for problem in ("TSP", "CVRP")]
    if not all(manifest["splits"].get(key, {}).get("complete") for key in heldout_keys):
        logging.error("Training samples require all six completed held-out exports")
        return 2
    if manifest["train_decision"]["status"] != "skipped":
        logging.error("Training samples are only appended when full train was skipped")
        return 2
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_utc"])).total_seconds()
    manifest["_started"] = time.monotonic() - elapsed
    budget = min(args.max_seconds, manifest["runtime"]["max_seconds"])
    deadline = manifest["_started"] + budget
    original_run_seconds = manifest["elapsed_seconds"]
    summary = dict(status="starting", started_utc=utc_now(), original_run_elapsed_seconds=original_run_seconds,
                   overall_budget_seconds=budget, full_train_available=False,
                   command=sys.argv, problems={})
    manifest["train_sampling"] = summary
    manifest["heldout_exporter"] = manifest["exporter"]
    manifest["exporter"] = identity(Path(__file__).resolve())
    manifest["exporter_note"] = "Training-sample support was added after the held-out process exited; held-out NPZ files are unchanged."
    sample_started = time.monotonic()

    def stop(signum, frame):
        manifest["_stop"] = True
        manifest["stop_signal"] = int(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    exit_code = 1
    try:
        estimates = manifest["train_decision"]["estimates"]
        conservative = sum(value["conservative_seconds"] * min(2000, value["instances"]) / value["instances"]
                           for value in estimates.values())
        remaining = deadline - time.monotonic()
        summary.update(conservative_eta_seconds=conservative, remaining_seconds=remaining)
        if conservative + 30 >= remaining:
            summary.update(status="skipped", reason="samples do not fit the unchanged original wall budget")
            manifest["status"] = "complete_heldout_train_skipped"
            exit_code = 0
        else:
            manifest["status"] = "running_train_samples"
            emit(manifest, MANIFEST, "train_sample_admission", admitted=True,
                 conservative_eta_seconds=conservative, remaining_seconds=remaining)
            for problem in ("TSP", "CVRP"):
                if manifest.get("_stop") or time.monotonic() >= deadline:
                    summary["status"] = "partial"
                    break
                raw = load_split(problem, "train")
                indices, rule = stratified_training_indices(raw["nodes"], problem)
                summary["problems"][problem] = rule
                model, _ = load_model(problem)
                record = export_split(problem, "train_sample", model, raw, manifest, MANIFEST,
                                      deadline, selection=indices)
                rule.update(exported_instances=record["exported_instances"],
                            elapsed_seconds=record["elapsed_seconds"],
                            instances_per_second=record.get("instances_per_second"),
                            sample_complete=record.get("sample_complete", False),
                            path=record.get("output", {}).get("path"))
                if not record.get("sample_complete"):
                    summary["status"] = "partial"
                    break
            else:
                summary["status"] = "completed"
            manifest["status"] = ("complete_heldout_train_samples" if summary["status"] == "completed" else
                                  "complete_heldout_partial_train_samples")
            exit_code = 0
    except Exception as error:
        summary.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        manifest["status"] = "complete_heldout_train_sample_failed"
        logging.exception("Optional training sample export failed; held-out exports remain available")
    finally:
        summary.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - sample_started)
        manifest["finished_utc"] = utc_now()
        manifest["exit_code"] = exit_code
        write_json(PREDICTIONS / "train_sample" / "sampling_summary.json", summary)
        emit(manifest, MANIFEST, "train_samples_finished", status=manifest["status"],
             sample_status=summary["status"], sample_elapsed_seconds=summary["elapsed_seconds"],
             exit_code=exit_code)
    print(f"{manifest['status']}; overall_elapsed={manifest['elapsed_seconds']:.2f}s; manifest={MANIFEST}", flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Export the first original batch of each validation split")
    parser.add_argument("--append-train-sample", action="store_true",
                        help="After held-out completion and full-train rejection, sample 2000 train instances per problem within the original budget")
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--train", choices=("auto", "skip"), default="auto")
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=PREDICTIONS / "cpu_export.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    if args.append_train_sample:
        return append_training_samples(args)
    manifest_path = PREDICTIONS / "smoke" / "manifest.json" if args.smoke else MANIFEST
    assumptions = json.loads(MANIFEST.read_text(encoding="utf-8"))["assumptions"]
    manifest = dict(schema_version=1, status="starting", started_utc=utc_now(),
                    _started=time.monotonic(), assumptions=assumptions,
                    runtime=dict(device="cpu", cuda_visible_devices="", torch_threads=4,
                                 interop_threads=1, torch_version=torch.__version__,
                                 python_version=platform.python_version(), host=socket.gethostname(),
                                 pid=os.getpid(), batch_size=8, no_tta=True,
                                 data_aug=False, manual_feature=False, shuffle=False,
                                 num_workers=0, max_seconds=args.max_seconds,
                                 seed=2024, train_policy=args.train, command=sys.argv),
                    contract=dict(output_pattern=str(PREDICTIONS / "{train,val,test,LIB}" / "{TSP,CVRP}.npz"),
                                  costs="raw_label.pkl Python binary64 costs, never FP32-recast",
                                  winner="native stored raw_label.pkl ind; not recomputed from rounded costs",
                                  pred="original Greedy: softmax(logits).argmax", nodes="actual unpadded input node count; includes CVRP depot",
                                  indices="zero-based indices in original dataset.pkl order",
                                  pool_ids="unified_selector.registry.S2I in canonical POOLS order",
                                  historical_metrics="torch.topk(logits) vs original collator FP32 cost argmin",
                                  historical_check="necessary only, not sufficient"),
                    sources={name: identity(NSS / name) for name in
                             ("model.py", "dataset.py", "utils.py", "scripts/evaluate_ours_table.py")},
                    registry=identity(ROOT / "code" / "unified_selector" / "registry.py"),
                    exporter=identity(Path(__file__).resolve()), checkpoints={}, splits={},
                    no_training=True, no_gpu=True, source_code_modified=False)

    def stop(signum, frame):
        manifest["_stop"] = True
        manifest["stop_signal"] = int(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    np.random.seed(2024)
    # Seed the CPU generator without invoking any torch.cuda helper.
    torch.random.default_generator.manual_seed(2024)
    models = {}
    exit_code = 1
    try:
        for problem in ("TSP", "CVRP"):
            models[problem], manifest["checkpoints"][problem] = load_model(problem)
        emit(manifest, manifest_path, "checkpoints_loaded", trainable_parameters=0)
        if args.preflight_only:
            manifest["status"] = "preflight_passed"
            write_json(PREDICTIONS / "preflight.json", {k: v for k, v in manifest.items() if not k.startswith("_")})
            exit_code = 0
        else:
            manifest["status"] = "running"
            exit_code = run_export(args, models, manifest, manifest_path)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = dict(type=type(error).__name__, message=str(error))
        logging.exception("CPU export failed")
    finally:
        manifest["finished_utc"] = utc_now()
        manifest["elapsed_seconds"] = time.monotonic() - manifest["_started"]
        manifest["exit_code"] = exit_code
        manifest["source_identity_unchanged"] = all(sha256(Path(value["path"])) == value["sha256"]
                                                    for value in manifest["sources"].values())
        manifest["checkpoint_identity_unchanged"] = all(sha256(Path(value["path"])) == value["sha256"]
                                                        for value in manifest["checkpoints"].values())
        emit(manifest, manifest_path, "run_finished", status=manifest["status"], exit_code=exit_code)
    print(f"{manifest['status']}; elapsed={manifest['elapsed_seconds']:.2f}s; manifest={manifest_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
