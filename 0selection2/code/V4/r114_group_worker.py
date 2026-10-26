"""Problem/method-isolated singleton worker; no GPU use without --execute."""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import r114_response_data as data


LAZY_METHODS = {"RouteFinder", "MoSES_RF", "MoSES_CaDA"}
ROUTE_IMPLEMENTATIONS = {
    "original_bq": ("train_100k_backends.py", "BQBackend", "decode(..., make_tours=True)"),
    "original_elg": ("train_100k_backends.py", "OriginalELGBackend", "rollout solutions[augmentation,start]"),
    "original_reld": ("train_100k_backends.py", "RELDBackend", "WitnessEnv.selected_node_list"),
    "original_icam_cvrp": ("train_100k_icam.py", "OriginalICAMCVRPBackend", "env.selected_node_list[augmentation,start]"),
    "original_lehd_cvrp_greedy": ("train_100k_nss_ar.py", "OriginalLEHDCVRPBackend", "student nodes plus depot flags"),
    "original_omni_cvrp_nss_budget": ("train_100k_nss_ar.py", "OriginalOmniCVRPBackend", "env.selected_node_list"),
    "easynco_mvmoe_nss_cvrp_budget": ("train_100k_nss_ar.py", "NSSMVmoeCVRPBackend", "best augmentation/start selected_node_list"),
    "easynco_native_multitask_historical_sampling": ("train_100k_sampling.py", "HistoricalSamplingBackend", "best-start env.selected_node_list"),
    "original_routefinder_family": ("train_100k_backends.py", "RouteFinderBackend", "best_aug_actions; upstream return_actions=True"),
}


def reuse_mode(method):
    return "cold_lazy_subrunner_per_endpoint" if method in LAZY_METHODS else "backend_per_problem_method"


def route_inventory(manifest, lock):
    """Small static source check, not imports, model loading or runtime proof."""
    records, cache = [], {}
    for problem in sorted({t["problem"] for t in manifest["tasks"]}):
        for candidate in manifest["pools"][problem]:
            deployment = lock["deployments"][f"{problem}/{candidate['method']}"]
            implementation = ROUTE_IMPLEMENTATIONS.get(deployment["backend"])
            record = dict(problem=problem, method=candidate["method"], backend=deployment["backend"],
                          runtime_verified=False, evidence="source_only", source_route_path=False)
            if implementation:
                filename, classname, extraction = implementation
                path = str(Path(lock["scripts_root"]) / filename)
                if path not in cache:
                    cache[path] = (Path(path).read_text(), data.file_hash(path))
                source, digest = cache[path]
                tree = ast.parse(source)
                cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname), None)
                run = next((node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run"), None) if cls else None
                has_result = bool(run and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                    node.func.id == "InferenceResult" and len(node.args) >= 2 for node in ast.walk(run)))
                record.update(source=path, class_name=classname, run_line=run.lineno if run else None,
                              extraction=extraction, source_sha256=digest,
                              source_hash_matches=digest == lock["artifacts"].get(path),
                              source_route_path=has_result and digest == lock["artifacts"].get(path),
                              alias_safe=candidate["method"] == "MoSES_RF")
                if candidate["method"] in LAZY_METHODS:
                    repo = data.WORKSPACE / "methods\u8bba\u6587_mineru_output/source_code" / (
                        "routefinder" if candidate["method"] == "RouteFinder" else "moses_vrp")
                    upstream = repo / "test.py"
                    text = upstream.read_text()
                    expected = lock["artifacts"].get(str(upstream))
                    record["actions_requested"] = "return_actions=True" in text
                    record["conditional_action_output"] = True
                    record["source_route_path"] = bool(record["source_route_path"] and record["actions_requested"] and
                        "best_aug_actions" in text and expected == data.file_hash(upstream))
            records.append(record)
    return dict(deployments=len(records), source_route_paths=sum(r["source_route_path"] for r in records),
                runtime_verified=0, records=records,
                conclusion="Source route hooks are not a guarantee that a real deployment exports a valid route")


def reset_rng(torch, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def reset_lazy_runner(backend, method):
    if method not in LAZY_METHODS:
        return
    if not hasattr(backend, "pool") or not hasattr(backend.pool, "_cache") or not hasattr(backend, "validated_buckets"):
        raise ValueError("Unknown lazy backend lifecycle; warm reuse is prohibited")
    # Native lazy construction runs AFTER the endpoint seed, just as cold dispatch.
    backend.pool._cache.clear()
    backend.validated_buckets.clear()


class NativeSession:
    def __init__(self, manifest, lock, problem, method, device):
        sys.path[:0] = [lock["overlay_root"], lock["scripts_root"]]
        import torch
        self.torch, self.device = torch, device
        self.problem, self.method, self.manifest, self.lock = problem, method, manifest, lock
        self.tasks = {t["task_id"]: t for t in manifest["tasks"]}
        self.endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
        torch.set_num_threads(1)
        reset_rng(torch, data.seed_for(2, problem, method, "model"), device)
        factory = importlib.import_module("train_100k_sampling").make_confirmed_backend
        source_method = next(c["source_method"] for c in manifest["pools"][problem] if c["method"] == method)
        self.backend = factory(problem, source_method, device, Path(lock["legacy_root"]))
        self.profile_sha256 = data.canonical_hash(self.backend.profile())
        if self.profile_sha256 != lock["deployments"][f"{problem}/{method}"]["profile_sha256"]:
            raise ValueError("Grouped backend does not match the locked native profile")

    def solve(self, task_id):
        task = self.tasks[task_id]
        if (task["problem"], task["method"]) != (self.problem, self.method):
            raise ValueError("A worker cannot cross a problem/method boundary")
        endpoint = self.endpoints[task["endpoint_id"]]
        if data.file_hash(endpoint["input_path"]) != task["input_sha256"]:
            raise ValueError("Endpoint input changed")
        item = data.read_json(endpoint["input_path"])["item"]
        native = {k: self.torch.tensor(v, dtype=self.torch.float32) for k, v in item.items()} if self.problem == "CVRP" else item
        reset_lazy_runner(self.backend, self.method)
        reset_rng(self.torch, task["solver_seed"], self.device)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        start = time.monotonic()
        result = self.backend.run([native])
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        if len(result.costs) != 1 or not getattr(result, "tours", None) or len(result.tours) != 1:
            raise ValueError("route_unavailable: a real nonempty native tour is mandatory")
        audit = data.check_route(self.problem, item, result.tours[0], result.costs[0])
        receipt = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
        receipt.update(version=self.manifest["version"], lock_sha256=self.lock["lock_sha256"], source="fresh_solver",
            status="ok", cost=float(result.costs[0]), tour=data.plain(result.tours[0]), route_audit=audit,
            profile_sha256=self.profile_sha256, backend_audit=data.plain(result.audit),
            solve_seconds=time.monotonic() - start, batch_size=1, reuse_mode=reuse_mode(self.method))
        return receipt


def same_response(reference, observed):
    if float(reference["cost"]) != float(observed["cost"]) or reference["tour"] != observed["tour"]:
        raise ValueError("cold/group mismatch: exact native cost and route equality required; no tolerance or seed search")


def reference_dispatch(task_id, output, manifest_path, reference_lock_path, timeout):
    command = [sys.executable, "-m", "code.V4.r114_response_data", "dispatch", "--manifest", str(manifest_path),
               "--deployment-lock", str(reference_lock_path), "--task-id", task_id,
               "--device", "cuda:0", "--execute", "--output", str(output)]
    data.write_new_json(output.with_suffix(".command.json"), dict(argv=command))
    with output.with_suffix(".log").open("x") as log:
        completed = subprocess.run(command, cwd=str(data.WORKSPACE / "0selection2"), stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if completed.returncode or not output.exists():
        raise ValueError(f"Cold reference failed for {task_id}, exit={completed.returncode}")
    return data.read_json(output)


def run_group(plan, manifest, lock, job, output, execute=False, device="cuda:0", timeout=3600,
              session_factory=NativeSession, cold_runner=None):
    """Stop on the first unusable response; never fill the unrun tail."""
    if not execute:
        return dict(execution=False, job=job, backend_boundary="problem/method", batch_size=1)
    from . import r114_label_queue as queue
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    if not job["task_ids"] or any((tasks[t]["problem"], tasks[t]["method"]) != (job["problem"], job["method"]) for t in job["task_ids"] + job.get("replay_task_ids", [])):
        raise ValueError("Grouped tasks do not share exactly one problem/method")
    cold_ids = job.get("cold_task_ids", job["task_ids"] if job["phase"] != "label" else [])
    reference_lock = data.read_json(plan["reference_lock_path"]) if cold_ids else None
    references, completed_ids, receipt_hashes = {}, [], {}
    stage, current_task, error = "initialize", None, None
    comparisons = []
    start = time.monotonic()
    def progress(stage, task_id=None):
        queue.atomic_json(output / "progress.json", dict(stage=stage, task_id=task_id,
            started_monotonic=time.monotonic(), job_id=job["job_id"]))
    try:
        if reference_lock:
            stage = "cold_reference"
            for current_task in cold_ids:
                progress(stage, current_task)
                path = output / "cold" / f"{current_task}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                receipt = (cold_runner(current_task) if cold_runner else reference_dispatch(
                    current_task, path, plan["manifest_path"], plan["reference_lock_path"], timeout))
                if cold_runner:
                    data.write_new_json(path, receipt)
                queue.validate_receipt(receipt, tasks[current_task], manifest, reference_lock)
                references[current_task] = receipt
        stage = "grouped_singletons"
        current_task = None
        progress("backend_initialization")
        session = session_factory(manifest, lock, job["problem"], job["method"], device)
        for current_task in job["task_ids"]:
            progress(stage, current_task)
            receipt = session.solve(current_task)
            queue.validate_receipt(receipt, tasks[current_task], manifest, lock)
            if current_task in references:
                same_response(references[current_task], receipt)
            path = output / f"{current_task}.json"
            data.write_new_json(path, receipt)
            completed_ids.append(current_task)
            receipt_hashes[current_task] = data.file_hash(path)
        stage = "replay_after_intervening_instances"
        for current_task in job.get("replay_task_ids", []):
            progress(stage, current_task)
            receipt = session.solve(current_task)
            queue.validate_receipt(receipt, tasks[current_task], manifest, lock)
            baseline = references.get(current_task) or data.read_json(output / f"{current_task}.json")
            same_response(baseline, receipt)
            path = output / "replay" / f"{current_task}.json"
            data.write_new_json(path, receipt)
            comparisons.append(dict(task_id=current_task, sha256=data.file_hash(path)))
    except (Exception, KeyboardInterrupt) as failure:
        error = dict(task_id=current_task, stage=stage, error_type=type(failure).__name__, error=str(failure))
        if current_task and not (output / f"{current_task}.json").exists():
            data.write_new_json(output / f"{current_task}.json", queue.failure_receipt(
                tasks[current_task], manifest, lock, "group_failure", str(failure)))
        data.write_new_json(output / "problem_stop.json", dict(problem=job["problem"], method=job["method"],
            reason=error, stop_problem=True, unrun_tail_not_fabricated=True))
    result = dict(job_id=job["job_id"], problem=job["problem"], method=job["method"], phase=job["phase"],
        status="failed" if error else "passed", executed=True, error=error, stop_problem=bool(error),
        plan_sha256=plan["plan_sha256"], lock_sha256=lock["lock_sha256"],
        receipt_sha256=receipt_hashes, completed_task_ids=completed_ids, expected_task_count=len(job["task_ids"]),
        replay_checks=comparisons, cold_reference_lock_sha256=reference_lock["lock_sha256"] if reference_lock else None,
        cold_reference_sha256={t: data.file_hash(output / "cold" / f"{t}.json") for t in references},
        reuse_mode=reuse_mode(job["method"]), wall_seconds=time.monotonic() - start)
    data.write_new_json(output / "worker_result.json", result)
    return result


def validate_equivalence(plan, job, attempt, manifest, lock):
    """Check real receipts again, not a stored qualification boolean."""
    from . import r114_label_queue as queue
    attempt = Path(attempt)
    result = data.read_json(attempt / "worker_result.json")
    if (result["status"] != "passed" or result["executed"] is not True or
            result["plan_sha256"] != plan["plan_sha256"] or result["lock_sha256"] != lock["lock_sha256"] or
            result["completed_task_ids"] != job["task_ids"] or result["job_id"] != job["job_id"]):
        raise ValueError("Grouped worker is incomplete, failed, or stale")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    if (result.get("problem") != job["problem"] or result.get("method") != job["method"] or
            result.get("expected_task_count") != len(job["task_ids"]) or
            set(result["receipt_sha256"]) != set(job["task_ids"])):
        raise ValueError("Incomplete grouped receipt inventory")
    for task_id in job["task_ids"]:
        path = attempt / f"{task_id}.json"
        if data.file_hash(path) != result["receipt_sha256"][task_id]:
            raise ValueError("Grouped receipt changed")
        queue.validate_receipt(data.read_json(path), tasks[task_id], manifest, lock)
    if job["phase"] == "label":
        return result
    cold_ids = job.get("cold_task_ids", job["task_ids"])
    reference_lock = data.read_json(plan["reference_lock_path"]) if cold_ids else None
    if (result["cold_reference_lock_sha256"] != (reference_lock["lock_sha256"] if reference_lock else None) or
            set(result["cold_reference_sha256"]) != set(cold_ids)):
        raise ValueError("Wrong cold reference source contract")
    for task_id in cold_ids:
        path = attempt / "cold" / f"{task_id}.json"
        if data.file_hash(path) != result["cold_reference_sha256"].get(task_id):
            raise ValueError("Cold reference missing or changed")
        cold, warm = data.read_json(path), data.read_json(attempt / f"{task_id}.json")
        queue.validate_receipt(cold, tasks[task_id], manifest, reference_lock)
        queue.validate_receipt(warm, tasks[task_id], manifest, lock)
        same_response(cold, warm)
    checks = {r["task_id"]: r for r in result["replay_checks"]}
    if set(checks) != set(job["replay_task_ids"]):
        raise ValueError("Missing interleaved replay proof")
    for task_id, proof in checks.items():
        path = attempt / "replay" / f"{task_id}.json"
        if data.file_hash(path) != proof["sha256"]:
            raise ValueError("Replay proof changed")
        receipt = data.read_json(path)
        queue.validate_receipt(receipt, tasks[task_id], manifest, lock)
        baseline = data.read_json(attempt / ("cold" if task_id in cold_ids else "") / f"{task_id}.json")
        same_response(baseline, receipt)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    output = data.cli_output_path(args.output)
    if args.plan:
        from . import r114_label_queue as queue
        plan, manifest, lock = queue.load_plan(args.plan)
        job = next(j for j in plan["runtime_jobs"] + plan["label_jobs"] if j["job_id"] == args.job_id)
        if args.execute:
            if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
                raise ValueError("No explicit physical GPU binding; group solve is not authorized")
            # Queue preflight already checked immutable sources. Recheck metadata here;
            # each solve still hashes its input and verifies the actual native profile.
            data.verify_lock(lock, manifest)
        result = run_group(plan, manifest, lock, job, output, args.execute, timeout=args.timeout_seconds)
        if not args.execute:
            data.write_new_json(output / "dry_run.json", result)
    else:
        result = dict(execution=False, group_boundary="problem/method", batch_size=1,
            task_seed="unchanged, reset for every endpoint", runtime_qualified=False)
        data.write_new_json(output, result)
    print(result.get("status", "dry_run"))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
