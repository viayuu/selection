"""R114-owned waiting queue. Planning never binds or probes a GPU."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from . import r114_response_data as data


BUCKET_METHODS = {"RouteFinder", "MoSES_RF", "MoSES_CaDA"}


def atomic_json(path, value):
    """Mutable progress only; protocol, requests and receipts remain immutable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def supervised_group(command, *, cwd, env, stdout, timeout, progress, stage_limit):
    """A stalled endpoint must not consume an entire production group's budget."""
    started = changed = time.monotonic()
    signature = None
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            if progress.exists():
                current = progress.stat().st_mtime_ns
                if current != signature:
                    signature, changed = current, time.monotonic()
            remaining = timeout - (time.monotonic() - changed)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                return subprocess.CompletedProcess(command, process.wait(timeout=min(1.0, remaining)))
            except subprocess.TimeoutExpired:
                # There are finitely many progress stages in the locked worker.
                if time.monotonic() - started > timeout * stage_limit:
                    raise subprocess.TimeoutExpired(command, timeout)
    finally:
        # Includes cold-reference processes and native subprocess descendants.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def family_id(method):
    return "ReLD" if method in ("RELD_CVRP", "reld_mt") else "MoSES" if method in ("MoSES_RF", "MoSES_CaDA") else method


def deployment_key(task, endpoint):
    bucket = str(50 if endpoint["input_check"]["n"] <= 75 else 100) if task["method"] in BUCKET_METHODS else "shared"
    return f"{task['problem']}/{task['method']}/{bucket}", bucket


def build_plan(manifest_path, lock_path, root):
    manifest, lock = data.read_json(manifest_path), data.read_json(lock_path)
    data.verify_manifest(manifest)
    data.verify_lock(lock, manifest, verify_files=True)
    endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
    tasks = {(t["endpoint_id"], t["method"]): t for t in manifest["tasks"]}
    deployments = {}
    for task in manifest["tasks"]:
        key, bucket = deployment_key(task, endpoints[task["endpoint_id"]])
        if key in deployments:
            continue
        source = lock["deployments"][f"{task['problem']}/{task['method']}"]
        protocol = dict(family_id=family_id(task["method"]), candidate=task["method"],
                        backend=source["backend"], budget=source["budget"],
                        checkpoints=source["checkpoints"], active_bucket=bucket,
                        source_lock_sha256=lock["lock_sha256"])
        choices = []
        for pair in manifest["pairs"]:
            if pair["problem"] != task["problem"] or pair["arm"] != "directed":
                continue
            original = tasks[(pair["original_id"], task["method"])]
            if deployment_key(original, endpoints[pair["original_id"]])[0] == key:
                choices.append(pair)
        preferred = [p for p in choices if p["family"] == ("tw_center" if "TW" in task["problem"] else "demand")]
        choices = sorted(preferred or choices, key=lambda p: (
            endpoints[p["original_id"]]["input_check"]["n"], p["base_id"]))
        chosen = choices[len(choices) // 2] if choices else None
        deployments[key] = dict(deployment_key=key, problem=task["problem"], method=task["method"],
            family_id=family_id(task["method"]), bucket=bucket,
            protocol_id=data.canonical_hash(protocol)[:20], protocol=protocol,
            probe_task_ids=[tasks[(chosen[k], task["method"])]["task_id"]
                            for k in ("original_id", "changed_id")] if chosen else [],
            probe_base_id=chosen["base_id"] if chosen else None,
            missing_reason=None if chosen else "no legal original/changed pair in this exact deployment bucket")
    grouped = defaultdict(list)
    for deployment in deployments.values():
        grouped[deployment["protocol_id"]].append(deployment)
    runtime_jobs, family_job_ids = [], []
    for protocol_id, members in sorted(grouped.items()):
        members.sort(key=lambda d: (not bool(d["probe_task_ids"]), d["deployment_key"]))
        representative = members[0]
        family_job = f"runtime_family__{representative['method']}__{protocol_id}"
        family_job_ids.append(family_job)
        for deployment in members:
            is_representative = deployment is representative
            job_id = family_job if is_representative else "runtime_deployment__" + deployment["deployment_key"].replace("/", "__")
            deployment["gate_job_id"] = job_id
            deployment["family_gate_job_id"] = family_job
            runtime_jobs.append(dict(job_id=job_id, phase="family" if is_representative else "deployment",
                deployment_key=deployment["deployment_key"], family_id=deployment["family_id"],
                protocol_id=protocol_id, requires=[] if is_representative else [family_job],
                task_ids=deployment["probe_task_ids"], missing_reason=deployment["missing_reason"]))
    label_jobs = []
    for task in manifest["tasks"]:
        key, _ = deployment_key(task, endpoints[task["endpoint_id"]])
        label_jobs.append(dict(job_id="label__" + task["task_id"], phase="label", task_ids=[task["task_id"]],
                               deployment_key=key, requires=[deployments[key]["gate_job_id"]]))
    plan = dict(schema_version=1, version=manifest["version"], split="train", queue_root=str(Path(root).resolve()),
        manifest_path=str(Path(manifest_path).resolve()), manifest_sha256=manifest["manifest_sha256"],
        deployment_lock_path=str(Path(lock_path).resolve()), lock_sha256=lock["lock_sha256"],
        queue_source=str(Path(__file__).resolve()), queue_source_sha256=data.file_hash(__file__),
        selection_sha256=manifest["selection_sha256"], gpu_binding=None, execution=False,
        all_family_gate_barrier=True, family_gate_job_ids=family_job_ids,
        deployments=deployments, runtime_jobs=runtime_jobs, label_jobs=label_jobs,
        summary=dict(legal_problem_candidates=len({(t["problem"], t["method"]) for t in manifest["tasks"]}),
            family_ids=len({d["family_id"] for d in deployments.values()}),
            family_protocol_gates=len(family_job_ids), deployment_bucket_gates=len(deployments),
            runtime_jobs=len(runtime_jobs), runtime_candidate_calls=sum(len(j["task_ids"]) for j in runtime_jobs),
            label_jobs=len(label_jobs), unresolved_probe_gates=sum(not j["task_ids"] for j in runtime_jobs)),
        policy=dict(automatic_launch=False, automatic_retry=False, base_replacement=False,
            probe_cost_reuse_as_labels=False, fresh_process_per_task=True,
            immutable_execution_binding=True,
            source="same immutable production source lock for runtime probes and labeling",
            gate_scope="candidate + actual protocol + checkpoint bucket, then exact problem deployment; no inherited missing successes"))
    plan["plan_sha256"] = data.canonical_hash(plan)
    return plan


def build_group_plan(parent_plan_path, root):
    """Metadata-only migration; no input rescan, weights or solver precheck."""
    from . import r114_group_worker as worker
    from . import r114_run_queue as scheduler
    parent = data.read_json(parent_plan_path)
    payload = {k: v for k, v in parent.items() if k != "plan_sha256"}
    if data.canonical_hash(payload) != parent["plan_sha256"]:
        raise ValueError("Parent queue metadata changed")
    if (Path(parent["queue_root"]) / "attempts").exists() or (Path(parent["queue_root"]) / "execution_binding.json").exists():
        raise ValueError("Cannot migrate an executed/bound queue automatically")
    manifest = data.read_json(parent["manifest_path"])
    old_lock = data.read_json(parent["deployment_lock_path"])
    data.verify_lock(old_lock, manifest)
    if (manifest["manifest_sha256"] != parent["manifest_sha256"] or old_lock["lock_sha256"] != parent["lock_sha256"]):
        raise ValueError("Parent queue/source provenance mismatch")
    if data.file_hash(data.__file__) != old_lock["artifacts"].get(str(Path(data.__file__).resolve())):
        raise ValueError("The parent data/seed implementation changed")
    root = Path(root).resolve()
    routes = worker.route_inventory(manifest, old_lock)
    lock = copy.deepcopy(old_lock)
    lock.pop("lock_sha256")
    lock.update(reference_lock_sha256=old_lock["lock_sha256"], execution_revision="problem_method_groups_v2_failfast",
        execution=dict(batch_size=1, fresh_process_per_task=False, fresh_process_per_problem_method=True,
            seed=old_lock["execution"]["seed"], both_endpoints_fresh=True, old_cost_reuse=False,
            lazy_runner_policy="cold reconstruct per endpoint; no RNG stream shortcut",
            qualification="independent cold references plus intervening-base and original replay comparisons"))
    lock["artifacts"][str(Path(__file__).resolve())] = data.file_hash(__file__)
    lock["artifacts"][str(Path(worker.__file__).resolve())] = data.file_hash(worker.__file__)
    lock["artifacts"][str(Path(scheduler.__file__).resolve())] = data.file_hash(scheduler.__file__)
    analysis_source = Path(__file__).with_name("r114_response_analysis.py").resolve()
    lock["artifacts"][str(analysis_source)] = data.file_hash(analysis_source)
    lock["lock_sha256"] = data.canonical_hash(lock)
    lock_path = root / "deployments.grouped.lock.json"
    data.write_new_json(lock_path, lock)
    data.write_new_json(root / "route_sources.json", routes)
    endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    groups = defaultdict(list)
    for task in manifest["tasks"]:
        groups[(task["problem"], task["method"])].append(task["task_id"])
    runtime_jobs, label_jobs = [], []
    for (problem, method), task_ids in sorted(groups.items()):
        old_gates = [j for j in parent["runtime_jobs"] if j["deployment_key"].rsplit("/", 1)[0] == f"{problem}/{method}"]
        probes = list(dict.fromkeys(t for j in old_gates for t in j["task_ids"]))
        replays = list(dict.fromkeys(j["task_ids"][0] for j in old_gates if j["task_ids"]))
        bases = {tasks[t]["base_id"] for t in probes}
        extra = next((t for t in task_ids if endpoints[tasks[t]["endpoint_id"]]["arm"] == "original" and
                      tasks[t]["base_id"] not in bases), None)
        if extra and len(bases) < 2:
            probes.append(extra)
        isolated = len({tasks[t]["base_id"] for t in probes}) >= 2
        runtime_jobs.append(dict(job_id=f"runtime_group__{problem}__{method}",
            phase="family" if any(j["phase"] == "family" for j in old_gates) else "deployment",
            problem=problem, method=method, task_ids=probes, replay_task_ids=replays,
            cold_task_ids=replays[:1] if method not in worker.LAZY_METHODS else [],
            requires=[], reuse_mode=worker.reuse_mode(method),
            requires_group_equivalence=True, missing_reason=None if probes and isolated else "missing independent second-base probe",
            bucket_gates=[j["deployment_key"] for j in old_gates]))
        label_jobs.append(dict(job_id=f"label_group__{problem}__{method}", phase="label", problem=problem,
            method=method, task_ids=task_ids, replay_task_ids=[], requires=[], reuse_mode=worker.reuse_mode(method)))
    for job in label_jobs:
        job["requires"] = [r["job_id"] for r in runtime_jobs if r["problem"] == job["problem"]]
    lazy_tasks = sum(len(j["task_ids"]) for j in label_jobs if j["method"] in worker.LAZY_METHODS)
    eager_groups = sum(j["method"] not in worker.LAZY_METHODS for j in label_jobs)
    probe_calls = sum(len(j["task_ids"]) for j in runtime_jobs)
    reference_calls = sum(len(j["cold_task_ids"]) for j in runtime_jobs)
    replay_calls = sum(len(j["replay_task_ids"]) for j in runtime_jobs)
    plan = dict(schema_version=4, version=manifest["version"], split="train", queue_root=str(root),
        manifest_path=parent["manifest_path"], manifest_sha256=manifest["manifest_sha256"],
        deployment_lock_path=str(lock_path), lock_sha256=lock["lock_sha256"],
        reference_lock_path=parent["deployment_lock_path"], reference_lock_sha256=old_lock["lock_sha256"],
        parent_plan_sha256=parent["plan_sha256"], queue_source=str(Path(__file__).resolve()),
        queue_source_sha256=data.file_hash(__file__), group_worker_source=str(Path(worker.__file__).resolve()),
        group_worker_sha256=data.file_hash(worker.__file__), scheduler_source=str(Path(scheduler.__file__).resolve()),
        scheduler_sha256=data.file_hash(scheduler.__file__), selection_sha256=manifest["selection_sha256"],
        analysis_source=str(analysis_source), analysis_sha256=data.file_hash(analysis_source),
        gpu_binding=None, execution=False, all_family_gate_barrier=False,
        family_gate_job_ids=[j["job_id"] for j in runtime_jobs if j["phase"] == "family"],
        runtime_jobs=runtime_jobs, label_jobs=label_jobs,
        route_source_blocked_problems=sorted({r["problem"] for r in routes["records"] if not r["source_route_path"]}),
        policy=dict(stop_problem_on_first_failure=True, problem_full_pool_runtime_barrier=True,
            automatic_launch=False, automatic_retry=False, immutable_execution_binding=True,
            base_replacement=False, batch_size=1, per_endpoint_seed_reset=True,
            probe_cost_reuse_as_labels=False, no_cross_problem_method_backend=True),
        summary=dict(label_worker_groups=len(label_jobs), label_tasks=sum(len(j["task_ids"]) for j in label_jobs),
            eager_backend_reuse_groups=eager_groups, cold_lazy_groups=len(label_jobs) - eager_groups,
            label_lazy_subrunner_reloads=lazy_tasks, label_neural_loads_upper_estimate=lazy_tasks + eager_groups,
            runtime_worker_groups=len(runtime_jobs), cold_reference_processes=reference_calls,
            grouped_probe_calls=probe_calls + replay_calls,
            total_runtime_candidate_calls=reference_calls + probe_calls + replay_calls,
            source_route_paths=routes["source_route_paths"], runtime_route_verified=0))
    plan["plan_sha256"] = data.canonical_hash(plan)
    validate_group_topology(plan, manifest, lock, old_lock)
    return plan


def validate_group_topology(plan, manifest, lock, reference):
    """Metadata-only checks; do not reload the 512 bases or solver weights."""
    if data.canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Manifest metadata changed")
    data.verify_lock(reference, manifest)
    if (reference["lock_sha256"] != plan["reference_lock_sha256"] or
            lock["reference_lock_sha256"] != reference["lock_sha256"] or
            reference["deployments"] != lock["deployments"] or reference["runtime"] != lock["runtime"] or
            any(lock["artifacts"].get(p) != h for p, h in reference["artifacts"].items()) or
            lock["execution"]["batch_size"] != 1 or lock["execution"]["old_cost_reuse"] is not False):
        raise ValueError("Grouped execution changed the locked native source/budget/seed contract")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
    expected = {(e["endpoint_id"], c["method"]) for e in manifest["endpoints"]
                for c in manifest["pools"][e["problem"]]}
    if (not set(manifest["selected_problems"]) <= set(data.ELIGIBLE) or
            len(tasks) != len(manifest["tasks"]) or len(expected) != len(tasks) or
            {(t["endpoint_id"], t["method"]) for t in tasks.values()} != expected):
        raise ValueError("Only the entire legal pool on applicable TRAIN endpoints is allowed")
    groups = defaultdict(list)
    for task in manifest["tasks"]:
        groups[(task["problem"], task["method"])].append(task["task_id"])
    for jobs in (plan["runtime_jobs"], plan["label_jobs"]):
        if len(jobs) != len(groups) or {(j["problem"], j["method"]) for j in jobs} != set(groups):
            raise ValueError("Missing or duplicate problem/method groups")
    if len({j["job_id"] for j in plan["runtime_jobs"] + plan["label_jobs"]}) != 2 * len(groups):
        raise ValueError("Duplicate queue job identity")
    for job in plan["runtime_jobs"] + plan["label_jobs"]:
        group = groups[(job["problem"], job["method"])]
        if not job["task_ids"] or len(set(job["task_ids"])) != len(job["task_ids"]) or not set(job["task_ids"]) <= set(group):
            raise ValueError("Invalid grouped task boundary")
        if job["phase"] == "label":
            required = {j["job_id"] for j in plan["runtime_jobs"] if j["problem"] == job["problem"]}
            if job["task_ids"] != group or set(job["requires"]) != required:
                raise ValueError("Production requires every legal method's runtime gate for this problem")
        else:
            buckets = {deployment_key(tasks[t], endpoints[tasks[t]["endpoint_id"]])[0] for t in group}
            probes = {deployment_key(tasks[t], endpoints[tasks[t]["endpoint_id"]])[0] for t in job["task_ids"]}
            if (set(job["bucket_gates"]) != buckets or probes != buckets or
                    not set(job["replay_task_ids"]) <= set(job["task_ids"]) or
                    not set(job["cold_task_ids"]) <= set(job["task_ids"])):
                raise ValueError("Missing exact deployment bucket/replay qualification")


def load_plan(path):
    plan = data.read_json(path)
    payload = dict(plan)
    if data.canonical_hash({k: v for k, v in payload.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Queue plan changed")
    if data.file_hash(plan["queue_source"]) != plan["queue_source_sha256"]:
        raise ValueError("Queue source changed; create an independent new queue plan")
    if plan.get("group_worker_source") and data.file_hash(plan["group_worker_source"]) != plan["group_worker_sha256"]:
        raise ValueError("Grouped worker source changed")
    if plan.get("scheduler_source") and data.file_hash(plan["scheduler_source"]) != plan["scheduler_sha256"]:
        raise ValueError("Scheduler source changed")
    if plan.get("analysis_source") and data.file_hash(plan["analysis_source"]) != plan["analysis_sha256"]:
        raise ValueError("Analysis source changed after freeze")
    manifest, lock = data.read_json(plan["manifest_path"]), data.read_json(plan["deployment_lock_path"])
    if (manifest["manifest_sha256"] != plan["manifest_sha256"] or lock["lock_sha256"] != plan["lock_sha256"]):
        raise ValueError("Queue source lock or manifest mismatch")
    data.verify_lock(lock, manifest)
    if plan.get("schema_version", 1) >= 4:
        if Path(path).resolve() != Path(plan["queue_root"]) / "plan.json":
            raise ValueError("Execute the canonical queue plan, not a relocated copy")
        validate_group_topology(plan, manifest, lock, data.read_json(plan["reference_lock_path"]))
    return plan, manifest, lock


def validate_receipt(receipt, task, manifest, lock):
    expected = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
    expected.update(version=manifest["version"], lock_sha256=lock["lock_sha256"], source="fresh_solver", status="ok")
    if any(receipt.get(k) != value for k, value in expected.items()):
        raise ValueError("Missing/failed/stale real solver receipt")
    profile = lock["deployments"][f"{task['problem']}/{task['method']}"]
    if receipt.get("profile_sha256") != profile["profile_sha256"]:
        raise ValueError("Runtime used a different candidate profile")
    endpoint = next(e for e in manifest["endpoints"] if e["endpoint_id"] == task["endpoint_id"])
    if data.file_hash(endpoint["input_path"]) != task["input_sha256"]:
        raise ValueError("Runtime input changed")
    item = data.read_json(endpoint["input_path"])["item"]
    return data.check_route(task["problem"], item, receipt.get("tour"), receipt.get("cost"))


def job_state(plan, job, manifest, lock):
    attempts = sorted((Path(plan["queue_root"]) / "attempts" / job["job_id"]).glob("attempt_*"))
    if not attempts:
        return dict(status="pending" if job["task_ids"] and not job.get("missing_reason") else "blocked", reason=job.get("missing_reason"))
    attempt = attempts[-1]
    if job.get("missing_reason"):
        return dict(status="blocked", reason=job["missing_reason"])
    if not (attempt / "job_result.json").exists():
        return dict(status="incomplete", attempt=str(attempt), reason="attempt has no completion receipt; never assume success")
    binding_path = Path(plan["queue_root"]) / "execution_binding.json"
    try:
        result = data.read_json(attempt / "job_result.json")
        binding = data.read_json(binding_path) if binding_path.exists() else {}
    except (OSError, ValueError, TypeError):
        return dict(status="failed", attempt=str(attempt), reason="Unreadable completion/binding receipt")
    binding_payload = {k: v for k, v in binding.items() if k != "binding_sha256"}
    if (result.get("plan_sha256") != plan["plan_sha256"] or result.get("job_id") != job["job_id"] or
            result.get("status") != "passed" or result.get("executed") is not True or
            binding.get("plan_sha256") != plan["plan_sha256"] or
            data.canonical_hash(binding_payload) != binding.get("binding_sha256") or
            result.get("binding_sha256") != binding.get("binding_sha256") or
            result.get("lock_sha256") != lock["lock_sha256"]):
        return dict(status="failed", attempt=str(attempt), reason=result.get("error", "runtime did not pass"))
    try:
        tasks = {t["task_id"]: t for t in manifest["tasks"]}
        for task_id in job["task_ids"]:
            path = attempt / f"{task_id}.json"
            if data.file_hash(path) != result["receipt_sha256"].get(task_id):
                raise ValueError("Runtime receipt changed")
            validate_receipt(data.read_json(path), tasks[task_id], manifest, lock)
        if plan.get("schema_version", 1) >= 3:
            from . import r114_group_worker as worker
            if data.file_hash(attempt / "worker_result.json") != result.get("worker_result_sha256"):
                raise ValueError("Grouped worker summary missing or changed")
            worker.validate_equivalence(plan, job, attempt, manifest, lock)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(status="failed", attempt=str(attempt), reason=str(error))
    return dict(status="passed", attempt=str(attempt))


def queue_status(plan, manifest, lock):
    runtime = {j["job_id"]: job_state(plan, j, manifest, lock) for j in plan["runtime_jobs"]}
    all_families = all(runtime[j]["status"] == "passed" for j in plan["family_gate_job_ids"])
    labels = {j["job_id"]: job_state(plan, j, manifest, lock) for j in plan["label_jobs"]}
    stopped = set(plan.get("route_source_blocked_problems", []))
    if plan.get("schema_version", 1) >= 3:
        for job in plan["runtime_jobs"] + plan["label_jobs"]:
            state = (labels if job["phase"] == "label" else runtime)[job["job_id"]]
            if state["status"] in ("failed", "incomplete", "blocked"):
                stopped.add(job["problem"])
    ready_labels = [j["job_id"] for j in plan["label_jobs"] if labels[j["job_id"]]["status"] == "pending" and
                    j.get("problem") not in stopped and
                    (all_families or not plan["all_family_gate_barrier"]) and
                    all(runtime[k]["status"] == "passed" for k in j["requires"])]
    binding_path = Path(plan["queue_root"]) / "execution_binding.json"
    return dict(plan_sha256=plan["plan_sha256"], gpu_binding=plan["gpu_binding"], execution=False,
        execution_binding=data.read_json(binding_path) if binding_path.exists() else None,
        stopped_problems=sorted(stopped),
        all_family_protocols_passed=all_families, runtime_states=runtime, label_states=labels,
        summary=dict(runtime_passed=sum(s["status"] == "passed" for s in runtime.values()),
            runtime_failed=sum(s["status"] in ("failed", "incomplete") for s in runtime.values()),
            runtime_total=len(runtime), label_passed=sum(s["status"] == "passed" for s in labels.values()),
            label_failed=sum(s["status"] in ("failed", "incomplete") for s in labels.values()),
            label_total=len(labels), label_ready=len(ready_labels),
            label_endpoint_total=sum(len(j["task_ids"]) for j in plan["label_jobs"])), ready_label_job_ids=ready_labels)


def next_job(plan, manifest, lock, phase="auto", job_id=None, retry_failed=False):
    status = queue_status(plan, manifest, lock)
    runtime = status["runtime_states"]
    jobs = sorted(plan["runtime_jobs"], key=lambda j: (j["phase"] != "family", j["job_id"])) + plan["label_jobs"]
    for job in jobs:
        if job_id is not None and job["job_id"] != job_id:
            continue
        if phase != "auto" and job["phase"] != phase:
            continue
        if job.get("problem") in status["stopped_problems"] and not (job_id is not None and retry_failed):
            continue
        state = (status["label_states"] if job["phase"] == "label" else runtime)[job["job_id"]]["status"]
        eligible = state == "pending" or (job_id is not None and retry_failed and state in ("failed", "incomplete"))
        if not eligible or not job["task_ids"] or job.get("missing_reason"):
            continue
        if job["phase"] == "label" and plan["all_family_gate_barrier"] and not status["all_family_protocols_passed"]:
            continue
        if not all(runtime[k]["status"] == "passed" for k in job["requires"]):
            continue
        return dict(job=job, state=state, executable=False, requires_explicit_binding_and_execute=True)
    return dict(job=None, executable=False, reason="No eligible job; missing/failed gates stay blocked, never defaulted to pass")


def make_binding(plan, host, gpu_uuid, cpu_ids, timeout):
    if (not gpu_uuid.startswith("GPU-") or any(c in gpu_uuid for c in (",", " ", "/", "\n")) or
            not host or not 1 <= len(set(cpu_ids)) <= 2 or min(cpu_ids) < 0 or timeout <= 0):
        raise ValueError("Explicit host, physical GPU UUID, 1-2 CPU IDs and positive timeout required")
    binding = dict(schema_version=1, plan_sha256=plan["plan_sha256"], host=host,
                   gpu_uuid=gpu_uuid, cpu_ids=sorted(set(cpu_ids)), numerical_threads=1,
                   timeout_seconds=timeout, automatic_launch=False)
    binding["binding_sha256"] = data.canonical_hash(binding)
    return binding


def worker_command(plan, binding, task_id, output):
    return ["taskset", "-c", ",".join(map(str, binding["cpu_ids"])), sys.executable,
            "-m", "code.V4.r114_response_data", "dispatch", "--manifest", plan["manifest_path"],
            "--deployment-lock", plan["deployment_lock_path"], "--task-id", task_id,
            "--device", "cuda:0", "--execute", "--output", str(output)]


def failure_receipt(task, manifest, lock, status, error):
    receipt = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
    receipt.update(version=manifest["version"], lock_sha256=lock["lock_sha256"], source="fresh_solver",
                   status=status, error=error, cost=None, tour=None, route_audit=None)
    return receipt


def run_one(plan, manifest, lock, binding, phase="auto", job_id=None, retry_failed=False, execute=False,
            preverified=False):
    if not execute:
        return next_job(plan, manifest, lock, phase, job_id, retry_failed)
    payload = dict(binding or {})
    fingerprint = payload.pop("binding_sha256", None)
    if (data.canonical_hash(payload) != fingerprint or payload.get("plan_sha256") != plan["plan_sha256"] or
            payload.get("host") != socket.gethostname().split(".")[0] or
            not set(payload.get("cpu_ids", [])) <= set(os.sched_getaffinity(0))):
        raise ValueError("Execution needs a valid binding for this host and explicit available CPU IDs")
    root = Path(plan["queue_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue_execution.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = next_job(plan, manifest, lock, phase, job_id, retry_failed)
        if request["job"] is None:
            return request
        if not preverified:
            data.verify_manifest(manifest)
            data.verify_lock(lock, manifest, verify_files=True)
        data.write_new_json(root / "execution_binding.json", binding)
        job = request["job"]
        parent = root / "attempts" / job["job_id"]
        existing = sorted(parent.glob("attempt_*"))
        number = max([int(p.name.split("_")[-1]) for p in existing] or [0]) + 1
        attempt = parent / f"attempt_{number:04d}"
        attempt.mkdir(parents=True)
        data.write_new_json(attempt / "request.json", dict(plan_sha256=plan["plan_sha256"],
            job=job, binding=binding, lock_sha256=lock["lock_sha256"]))
        data.write_new_json(root / "plan.location.json", dict(path=str(Path(plan["queue_root"]) / "plan.json")))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=binding["gpu_uuid"],
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1")
        tasks = {t["task_id"]: t for t in manifest["tasks"]}
        if plan.get("schema_version", 1) >= 3:
            return run_group_process(plan, manifest, lock, binding, job, attempt, environment)
        hashes, errors = {}, []
        started = time.monotonic()
        for task_id in job["task_ids"]:
            receipt_path = attempt / f"{task_id}.json"
            command = worker_command(plan, binding, task_id, receipt_path)
            data.write_new_json(attempt / f"{task_id}.command.json", dict(argv=command))
            try:
                with (attempt / f"{task_id}.log").open("x") as log:
                    completed = subprocess.run(command, cwd=str(data.WORKSPACE / "0selection2"),
                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                        timeout=binding["timeout_seconds"], check=False)
                if not receipt_path.exists():
                    data.write_new_json(receipt_path, failure_receipt(tasks[task_id], manifest, lock,
                        "worker_error", f"Worker exited {completed.returncode} without a receipt"))
                if completed.returncode:
                    raise ValueError(f"Worker exit code {completed.returncode}")
                validate_receipt(data.read_json(receipt_path), tasks[task_id], manifest, lock)
            except (subprocess.TimeoutExpired, ValueError, OSError, TypeError) as error:
                if not receipt_path.exists():
                    data.write_new_json(receipt_path, failure_receipt(tasks[task_id], manifest, lock,
                        "timeout" if isinstance(error, subprocess.TimeoutExpired) else "worker_error", str(error)))
                errors.append(dict(task_id=task_id, error=str(error)))
            if receipt_path.exists():
                hashes[task_id] = data.file_hash(receipt_path)
        result = dict(job_id=job["job_id"], phase=job["phase"], status="failed" if errors else "passed",
            plan_sha256=plan["plan_sha256"], lock_sha256=lock["lock_sha256"], executed=True,
            binding_sha256=binding["binding_sha256"], receipt_sha256=hashes, errors=errors,
            wall_seconds=time.monotonic() - started, probe_costs_reused_as_labels=False)
        data.write_new_json(attempt / "job_result.json", result)
        return result


def run_group_process(plan, manifest, lock, binding, job, attempt, environment):
    command = ["taskset", "-c", ",".join(map(str, binding["cpu_ids"])), sys.executable,
        "-m", "code.V4.r114_group_worker", "--plan", str(Path(plan["queue_root"]) / "plan.json"),
        "--job-id", job["job_id"], "--output", str(attempt), "--execute",
        "--timeout-seconds", str(binding["timeout_seconds"])]
    data.write_new_json(attempt / "group.command.json", dict(argv=command))
    error, hashes = None, {}
    environment = dict(environment)
    stages = 2 * len(job["task_ids"]) + len(job.get("replay_task_ids", [])) + 3
    interrupted = False
    try:
        with (attempt / "group.log").open("x") as log:
            process = supervised_group(command, cwd=str(data.WORKSPACE / "0selection2"),
                env=environment, stdout=log, timeout=binding["timeout_seconds"],
                progress=attempt / "progress.json", stage_limit=stages)
        if process.returncode:
            raise ValueError(f"Grouped worker exit={process.returncode}; stop {job['problem']}")
        from . import r114_group_worker as worker
        worker.validate_equivalence(plan, job, attempt, manifest, lock)
        for task_id in job["task_ids"]:
            path = attempt / f"{task_id}.json"
            validate_receipt(data.read_json(path), next(t for t in manifest["tasks"] if t["task_id"] == task_id), manifest, lock)
            hashes[task_id] = data.file_hash(path)
    except (subprocess.TimeoutExpired, ValueError, OSError, TypeError, KeyError, KeyboardInterrupt) as failure:
        error = str(failure)
        interrupted = isinstance(failure, KeyboardInterrupt)
        data.write_new_json(attempt / "queue_problem_stop.json", dict(problem=job["problem"], method=job["method"],
            stop_problem=True, error=error, unrun_tail_not_fabricated=True))
    worker_summary = attempt / "worker_result.json"
    result = dict(job_id=job["job_id"], phase=job["phase"], status="failed" if error else "passed",
        error=error, problem=job["problem"], method=job["method"], executed=True,
        plan_sha256=plan["plan_sha256"], lock_sha256=lock["lock_sha256"],
        binding_sha256=binding["binding_sha256"], receipt_sha256=hashes,
        worker_result_sha256=data.file_hash(worker_summary) if worker_summary.exists() else None,
        probe_costs_reused_as_labels=False)
    data.write_new_json(attempt / "job_result.json", result)
    if interrupted:
        raise KeyboardInterrupt
    return result


def export_labels(plan, manifest, lock, output):
    status = queue_status(plan, manifest, lock)
    count = 0
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    for job in plan["label_jobs"]:
        if plan.get("schema_version", 1) >= 3 and job["problem"] in status["stopped_problems"]:
            continue  # Preserve raw evidence, but mask the entire unsupported problem in releases.
        state = job_state(plan, job, manifest, lock)
        if state["status"] == "pending":
            continue
        for task_id in job["task_ids"]:
            if plan.get("schema_version", 1) >= 3:
                path = Path(state["attempt"]) / f"{task_id}.json" if state.get("attempt") else None
                if path is None or not path.exists():
                    continue  # Preserve a truly missing unrun tail, not invented receipts.
                receipt = data.read_json(path)
                if receipt.get("status") == "ok":
                    validate_receipt(receipt, tasks[task_id], manifest, lock)
                else:
                    expected = failure_receipt(tasks[task_id], manifest, lock, receipt.get("status"), receipt.get("error"))
                    if any(receipt.get(k) != v for k, v in expected.items()):
                        raise ValueError("Invalid grouped failure provenance or fabricated cost/tour")
            elif state["status"] == "passed":
                receipt = data.read_json(Path(state["attempt"]) / f"{task_id}.json")
            else:
                receipt = failure_receipt(tasks[task_id], manifest, lock, "queue_failure", state.get("reason", state["status"]))
            receipt["queue_plan_sha256"] = plan["plan_sha256"]
            receipt["queue_binding_sha256"] = data.read_json(
                Path(plan["queue_root"]) / "execution_binding.json")["binding_sha256"]
            data.write_new_json(Path(output) / f"{task_id}.json", receipt)
            count += 1
    return dict(exported_label_receipts=count, runtime_probe_costs_exported=0,
        complete_release=not status["stopped_problems"] and status["summary"]["label_passed"] == len(plan["label_jobs"]),
        stopped_problems=status["stopped_problems"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("schema", "plan", "plan-group", "status", "next", "bind", "run-one", "export"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=name in ("schema", "plan", "plan-group", "bind", "export"))
        if name == "plan":
            child.add_argument("--manifest", type=Path, required=True)
            child.add_argument("--deployment-lock", type=Path, required=True)
        elif name == "plan-group":
            child.add_argument("--parent-plan", type=Path, required=True)
        elif name != "schema":
            child.add_argument("--plan", type=Path, required=True)
        if name in ("next", "run-one"):
            child.add_argument("--phase", choices=("auto", "family", "deployment", "label"), default="auto")
            child.add_argument("--job-id")
            child.add_argument("--retry-failed", action="store_true")
        if name == "bind":
            child.add_argument("--host", required=True)
            child.add_argument("--gpu-uuid", required=True)
            child.add_argument("--cpu-ids", required=True)
            child.add_argument("--timeout-seconds", type=int, required=True)
        if name == "run-one":
            child.add_argument("--binding", type=Path)
            child.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output = data.cli_output_path(args.output) if args.output else None
    if args.command == "schema":
        result = dict(schema_version=1, gpu_binding=None, execution=False,
            label_gate="all family protocols and the exact problem/candidate/bucket require real runtime receipts")
    elif args.command == "plan":
        result = build_plan(args.manifest, args.deployment_lock, output.parent)
    elif args.command == "plan-group":
        result = build_group_plan(args.parent_plan, output.parent)
    else:
        plan, manifest, lock = load_plan(args.plan)
        if args.command == "status":
            result = queue_status(plan, manifest, lock)
        elif args.command == "next":
            result = next_job(plan, manifest, lock, args.phase, args.job_id, args.retry_failed)
        elif args.command == "bind":
            result = make_binding(plan, args.host, args.gpu_uuid,
                                  [int(v) for v in args.cpu_ids.split(",")], args.timeout_seconds)
        elif args.command == "run-one":
            result = run_one(plan, manifest, lock, data.read_json(args.binding) if args.binding else None,
                             args.phase, args.job_id, args.retry_failed, args.execute)
        else:
            result = export_labels(plan, manifest, lock, output)
            output = None
    if output:
        data.write_new_json(output, result)
    print(json.dumps(result.get("summary", result), sort_keys=True))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
