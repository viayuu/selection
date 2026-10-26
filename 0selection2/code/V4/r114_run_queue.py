"""Serial R114 scheduler. The resource owner holds the external GPU mutex."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import r114_label_queue as queue
from . import r114_response_data as data


def check_paths(plan, manifest_path, lock_path):
    if (Path(manifest_path).resolve() != Path(plan["manifest_path"]) or
            Path(lock_path).resolve() != Path(plan["deployment_lock_path"])):
        raise ValueError("Use the manifest and grouped deployment lock pinned by this plan")
    if plan.get("schema_version", 1) < 4:
        raise ValueError("This scheduler requires the frozen grouped/fail-fast plan, not the old per-task queue")


def summarize(plan, status):
    states = list(status["runtime_states"].values()) + list(status["label_states"].values())
    complete = bool(states) and all(s["status"] == "passed" for s in states) and not status["stopped_problems"]
    return dict(plan_sha256=plan["plan_sha256"], manifest_sha256=plan["manifest_sha256"],
        lock_sha256=plan["lock_sha256"], status="complete" if complete else "blocked",
        complete=complete, stopped_problems=status["stopped_problems"],
        committed_label_receipts=sum(len(j["task_ids"]) for j in plan["label_jobs"]
            if status["label_states"][j["job_id"]]["status"] == "passed"),
        summary=status["summary"], automatic_retry=False, protocol_changed=False)


def append_event(path, event):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(dict(utc_seconds=time.time(), **event), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_analysis(plan, labels, output, binding):
    command = ["taskset", "-c", ",".join(map(str, binding["cpu_ids"])), sys.executable, "-B", "-m",
        "code.V4.r114_response_analysis", "--manifest", plan["manifest_path"],
        "--labels", str(labels), "--output", str(output)]
    data.write_new_json(output.parent / "analysis.command.json", dict(argv=command, gpu_execution=False))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    with (output.parent / "analysis.log").open("x") as log:
        result = subprocess.run(command, cwd=str(data.WORKSPACE / "0selection2"), env=environment,
            stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
    if result.returncode:
        raise ValueError(f"Read-only response analysis exit={result.returncode}; solver labels remain preserved")
    return dict(status="complete", output=str(output), gpu_execution=False, source_sha256=plan["analysis_sha256"])


def run_queue(plan, manifest, lock, binding=None, *, execute=False, max_jobs=None, finalize=False):
    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive; no unbounded retry loop exists")
    if not execute:
        status = queue.queue_status(plan, manifest, lock)
        return dict(summarize(plan, status), status="dry_run", execution=False,
            next_request=queue.next_job(plan, manifest, lock),
            note="No GPU probe, binding, solver import, state write or mutex acquisition")
    expected = queue.make_binding(plan, socket.gethostname().split(".")[0],
        (binding or {}).get("gpu_uuid", ""), (binding or {}).get("cpu_ids", []),
        (binding or {}).get("timeout_seconds", 0))
    if (binding != expected or not set(binding["cpu_ids"]) <= set(os.sched_getaffinity(0)) or
            os.environ.get("CUDA_VISIBLE_DEVICES") != binding["gpu_uuid"]):
        raise ValueError("Resource owner must export exactly one physical GPU UUID and a matching host/CPU binding")
    root = Path(plan["queue_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler_execution.lock").open("a") as scheduler_lock:
        fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # One full preflight per scheduler invocation. No input conversion or re-selection.
        data.verify_manifest(manifest)
        data.verify_lock(lock, manifest, verify_files=True)
        data.verify_lock(data.read_json(plan["reference_lock_path"]), manifest)
        data.write_new_json(root / "execution_binding.json", binding)
        runs = root / "scheduler_runs"
        runs.mkdir(exist_ok=True)
        number = max([int(p.name.split("_")[-1]) for p in runs.glob("run_*") if p.is_dir()] or [0]) + 1
        run = runs / f"run_{number:04d}"
        run.mkdir()
        data.write_new_json(run / "request.json", dict(plan_sha256=plan["plan_sha256"], binding=binding,
            max_jobs=max_jobs, finalize=finalize, automatic_retry=False))
        count, interrupted, error = 0, False, None
        try:
            # At most one new attempt per planned job. Interrupted/failed attempts stay stopped.
            for _ in range(min(max_jobs or len(plan["runtime_jobs"] + plan["label_jobs"]),
                               len(plan["runtime_jobs"] + plan["label_jobs"]))):
                request = queue.next_job(plan, manifest, lock)
                if request["job"] is None:
                    break
                job = request["job"]
                append_event(run / "events.jsonl", dict(event="job_start", job_id=job["job_id"], phase=job["phase"]))
                result = queue.run_one(plan, manifest, lock, binding, job_id=job["job_id"], execute=True,
                                       preverified=True)
                count += 1
                if result.get("status") not in ("passed", "failed"):
                    raise ValueError("Queue selection changed; stop without guessing or retrying")
                append_event(run / "events.jsonl", dict(event="job_end", job_id=job["job_id"], status=result["status"]))
                queue.atomic_json(root / "scheduler_progress.json", dict(
                    summarize(plan, queue.queue_status(plan, manifest, lock)),
                    status="running", execution=True, jobs_this_invocation=count, run_directory=str(run)))
        except KeyboardInterrupt:
            interrupted, error = True, "Scheduler interrupted; no automatic retry"
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        status = queue.queue_status(plan, manifest, lock)
        result = dict(summarize(plan, status), execution=True, jobs_this_invocation=count,
            interrupted=interrupted, error=error, run_directory=str(run))
        if interrupted or error:
            result.update(status="interrupted" if interrupted else "failed", complete=False)
        elif queue.next_job(plan, manifest, lock)["job"] is not None:
            result.update(status="yielded", complete=False)
        if finalize:
            # Always finalize this invocation, including partial failures/interruption.
            # Each immutable snapshot has its own directory, so resume never overwrites it.
            try:
                result["export"] = queue.export_labels(plan, manifest, lock, run / "label_receipts")
                assembled = data.assemble_labels(manifest, run / "label_receipts", run / "labels.json", lock)
                result["labels"] = dict(path=str(run / "labels.json"), summary=assembled["summary"],
                                        complete=result["complete"], winner_policy=data.WINNER_POLICY)
                if result["complete"]:
                    data.write_new_json(root / "labels.json", assembled)
                result["analysis"] = run_analysis(plan, run / "labels.json", run / "response_analysis", binding)
            except Exception as failure:
                result.update(status="failed", complete=False, assembly_error=f"{type(failure).__name__}: {failure}")
        data.write_new_json(run / "result.json", result)
        queue.atomic_json(root / "scheduler_progress.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--deployment-lock", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--cpu-ids")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan, manifest, lock = queue.load_plan(args.plan)
    check_paths(plan, args.manifest, args.deployment_lock)
    data.cli_output_path(plan["queue_root"])
    binding = None
    if args.execute:
        if not all((args.host, args.gpu_uuid, args.cpu_ids, args.timeout_seconds)):
            parser.error("--execute requires explicit --host, --gpu-uuid, --cpu-ids and --timeout-seconds")
        binding = queue.make_binding(plan, args.host, args.gpu_uuid,
            [int(v) for v in args.cpu_ids.split(",")], args.timeout_seconds)
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
    result = run_queue(plan, manifest, lock, binding, execute=args.execute,
                       max_jobs=args.max_jobs, finalize=args.finalize)
    print(json.dumps(result, sort_keys=True))
    return {"complete": 0, "dry_run": 0, "yielded": 3, "blocked": 2, "interrupted": 130}.get(result["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
