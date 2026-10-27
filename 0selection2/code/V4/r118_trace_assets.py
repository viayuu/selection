"""Finite TRAIN-only native process capture; no selector training or evaluation."""

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

import numpy as np

from . import r114_response_data as common


ROOT = Path(__file__).resolve().parent / "runs/R118_native_trace_assets"
ASSETS = common.WORKSPACE / "selection_annotations/R118_native_trace_seed2_20261027"
VERSION = "r118_realized_train_trace_seed2_v1"
COUNT = 8
METHODS = {p: "LEHD" if p == "TSP" else "MATNET" if p == "ATSP"
           else "RELD_CVRP" if p == "CVRP" else "reld_mt" for p in common.PROBLEMS}
SLOTS = {"A": list(common.PROBLEMS[::2]), "B": list(common.PROBLEMS[1::2])}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def prepare():
    path = ROOT / "capture_plan.json"
    if path.exists():
        return common.read_json(path)
    records = []
    for problem in common.PROBLEMS:
        mapping, metadata = common.read_mapping(common.DATA_ROOT, problem, "train")
        scales = np.asarray([int(x["scale"]) for x in mapping])
        cuts = np.quantile(np.log1p(scales), [.25, .5, .75]).tolist()
        groups = defaultdict(list)
        for row in mapping:
            group = (int(np.searchsorted(cuts, np.log1p(int(row["scale"])), side="right")),
                     row["distribution"], row["capacity_type"])
            groups[group].append(row)
        rng = np.random.default_rng(common.seed_for(2, problem, "r118_input_selection"))
        keys = sorted(groups)
        rng.shuffle(keys)
        for rows in groups.values():
            rng.shuffle(rows)
        selected = []
        while len(selected) < COUNT:
            for key in keys:
                if groups[key]:
                    selected.append(groups[key].pop())
                    if len(selected) == COUNT:
                        break
        method = METHODS[problem]
        pool = common.load_pool(common.DATA_ROOT, problem)
        source_method = next(x["source_method"] for x in pool if x["method"] == method)
        production = common.LEGACY_ROOT / "labels" / problem / source_method
        profile = common.read_json(production / "profile.json")
        directory = common.DATA_ROOT / (problem + "train")
        with (directory / "dataset.pkl").open("rb") as stream:
            items = pickle.load(stream)
        input_path = ASSETS / "inputs" / (problem + ".pkl")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("xb") as stream:
            pickle.dump([items[int(x["local_index"])] for x in selected], stream,
                        protocol=pickle.HIGHEST_PROTOCOL)
        records.append(dict(problem=problem, method=method, source_method=source_method,
            pool=pool, profile_path=str(production / "profile.json"),
            profile_sha256=common.canonical_hash(profile), backend=profile["backend"],
            budget=profile["budget"], production_source_contract=str(production / "dependency_contract.json"),
            dataset_sha256=metadata["files"]["dataset.pkl"]["sha256"],
            train_mapping_sha256=metadata["files"]["index_mapping.csv"]["sha256"],
            input_path=str(input_path), input_sha256=common.file_hash(input_path),
            selection=selected, train_log_size_quartiles=cuts))
        del items
    plan = dict(version=VERSION, split="train", seed=2, data_root=str(common.DATA_ROOT),
        scenario_sha256=common.file_hash(common.DATA_ROOT / "scenario_manifest.json"),
        asset_root=str(ASSETS), count_per_problem=COUNT, planned_observations=COUNT * len(records),
        process_deployments=18, legal_deployments=89, full_candidate_process_coverage=False,
        selector_training=False, labels_changed=False, heldout_labels_read=False,
        execution="original backend, singleton, exact recipe budget, fresh per-instance seed; not historical RNG replay",
        max_array_bytes=384 * 2**20, slots=SLOTS, deployments=records)
    plan["plan_sha256"] = common.canonical_hash(plan)
    common.write_new_json(path, plan)
    return plan


def load_plan():
    plan = common.read_json(ROOT / "capture_plan.json")
    value = dict(plan)
    expected = value.pop("plan_sha256")
    if common.canonical_hash(value) != expected or plan["version"] != VERSION:
        raise ValueError("Frozen TRAIN capture plan differs")
    return plan


def seed_all(torch, value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def run_problem(problem):
    plan = load_plan()
    deployment = next(x for x in plan["deployments"] if x["problem"] == problem)
    directory = ROOT / "capture" / problem
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "attempt.json").exists():
        raise RuntimeError("Existing capture attempt; inspect it, never silently retry")
    save(directory / "attempt.json", dict(state="started", pid=os.getpid(), started=time.time(),
        plan_sha256=plan["plan_sha256"], problem=problem, method=deployment["method"]))
    status = dict(state="running", problem=problem, completed=0, planned=COUNT,
                  pid=os.getpid(), plan_sha256=plan["plan_sha256"])
    try:
        import torch
        if (torch.cuda.device_count() != 1 or "RTX 3090" not in torch.cuda.get_device_name(0)):
            raise RuntimeError("Require one of the two explicitly bound RTX3090s")
        torch.set_num_threads(1)
        overlay = common.LEGACY_ROOT / "python_compat"
        scripts = common.BRIDGE / "scripts"
        sys.path[:0] = [str(overlay), str(scripts)]
        native = importlib.import_module("train_100k_sampling")
        seed_all(torch, common.seed_for(2, problem, "r118_model"))
        backend = native.make_confirmed_backend(problem, deployment["source_method"], "cuda:0", common.LEGACY_ROOT)
        profile = backend.profile()
        if common.canonical_hash(profile) != deployment["profile_sha256"]:
            raise ValueError("Actual backend profile differs; do not substitute current defaults")
        if common.file_hash(deployment["input_path"]) != deployment["input_sha256"]:
            raise ValueError("Selected frozen TRAIN inputs changed")
        with Path(deployment["input_path"]).open("rb") as stream:
            items = pickle.load(stream)
        if problem in ("TSP", "ATSP"):
            from .r118_td_trace import capture_td as capture
        else:
            from .r118_reld_trace import capture_reld as capture
        for position, (row, item) in enumerate(zip(deployment["selection"], items)):
            if problem == "CVRP":
                item = {key: torch.as_tensor(value, dtype=torch.float32, device="cpu") for key, value in item.items()}
            elif problem in ("TSP", "ATSP"):
                item = torch.as_tensor(item, dtype=torch.float32, device="cpu")
            seed = common.seed_for(2, problem, deployment["method"], int(row["source_index"]), "r118_trace")
            seed_all(torch, seed)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                result, arrays, info = capture(backend, [item], max_bytes=plan["max_array_bytes"])
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            if len(result.costs) != 1 or not np.isfinite(result.costs[0]):
                raise ValueError("Native singleton result unavailable")
            retained = sum(np.asarray(x).nbytes for x in arrays.values())
            if retained > plan["max_array_bytes"]:
                raise ValueError("Retained trace exceeds finite per-instance bound")
            target = ASSETS / "traces" / problem / (f"source{int(row['source_index'])}.npz")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            record = dict(version=VERSION, problem=problem, method=deployment["method"],
                split="train", local_index=int(row["local_index"]), source_index=int(row["source_index"]),
                scale=int(row["scale"]), distribution=row["distribution"], capacity_type=row["capacity_type"],
                solver_seed=seed, profile_sha256=deployment["profile_sha256"],
                plan_sha256=plan["plan_sha256"], process=dict(info),
                native_cost=float(result.costs[0]), native_best_tour=common.plain(result.tours[0]),
                native_backend_audit=common.plain(result.audit), native_mask_is_independent_checker=False,
                trace=str(target), trace_sha256=common.file_hash(target),
                compressed_bytes=target.stat().st_size, retained_array_bytes=retained,
                array_shapes={k:list(np.asarray(v).shape) for k,v in arrays.items()},
                capture_seconds=seconds, historical_scalar_reproduction_claim=False,
                original_labels_modified=False, student_or_selector_forward=False)
            save(directory / (f"source{int(row['source_index'])}.json"), record)
            status.update(completed=position + 1, source_index=int(row["source_index"]),
                          last_capture_seconds=seconds, updated=time.time())
            save(directory / "status.json", status)
            print(json.dumps(status), flush=True)
            del arrays
        status.update(state="complete", finished=time.time())
        save(directory / "status.json", status)
        return 0
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), finished=time.time())
        save(directory / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "capture"))
    parser.add_argument("--problem", choices=common.PROBLEMS)
    parser.add_argument("--allow-native-capture", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare()
        print(json.dumps({k:value[k] for k in ("version","planned_observations","process_deployments","selector_training","plan_sha256")}))
        return 0
    if not args.problem or not args.allow_native_capture:
        parser.error("Explicit TRAIN problem and --allow-native-capture required")
    return run_problem(args.problem)


if __name__ == "__main__":
    raise SystemExit(main())
