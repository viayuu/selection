"""Stage an isolated CVRP audit repair for the unchanged R114 queue/runner."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import r114_label_queue as queue
from . import r114_response_data as data
from . import r114_tensor_boundary_recovery as boundary


ROOT = boundary.ROOT / "acquisition_v1"
PARENT_MANIFEST = data.PREPARATION / "production_inputs/manifest.json"
PARENT_LOCK = data.PREPARATION / "production_execution/deployments.production.lock.json"
PARENT_PLAN = data.PREPARATION / "grouped_execution/queue_v1/plan.json"
REVISION = "r114_cvrp_host_audit_acquisition_v1_20261026"
CPU_ENV = dict(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def cvrp_counts(manifest):
    endpoints = [row for row in manifest["endpoints"] if row["problem"] == "CVRP"]
    return dict(bases=len({row["base_id"] for row in endpoints}), endpoints=len(endpoints),
                pairs=sum(row["problem"] == "CVRP" for row in manifest["pairs"]),
                methods=len(manifest["pools"]["CVRP"]),
                label_tasks=sum(row["problem"] == "CVRP" for row in manifest["tasks"]))


def subset_manifest(parent):
    if data.canonical_hash({k: v for k, v in parent.items() if k != "manifest_sha256"}) != parent["manifest_sha256"]:
        raise ValueError("Parent manifest metadata changed")
    result = copy.deepcopy(parent)
    result.update(selected_problems=["CVRP"], pools={"CVRP": parent["pools"]["CVRP"]},
                  acquisition_revision=REVISION, parent_manifest_sha256=parent["manifest_sha256"])
    for key in ("endpoints", "tasks", "pairs", "rejections"):
        result[key] = [entry for entry in result[key] if entry["problem"] == "CVRP"]
    for key in ("endpoint", "task", "pair"):
        result[key + "_count"] = len(result[key + "s"])
    result.pop("manifest_sha256")
    result["manifest_sha256"] = data.canonical_hash(result)
    return result


def freeze_scripts(parent):
    original, staged = Path(parent["scripts_root"]), ROOT / "scripts"
    staged.mkdir(parents=True, exist_ok=True)
    bindings = []
    for source in sorted([*original.glob("*.py"), original / "train_100k_environment.json"]):
        digest = data.file_hash(source)
        if parent["artifacts"].get(str(source), digest) != digest:
            raise ValueError(f"Parent source changed: {source}")
        target = staged / source.name
        if source.name == "train_100k_backends.py":
            boundary._write_new_text(target, boundary.revised_source(source.read_text()))
        elif target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Private script link changed: {target}")
        elif target.exists():
            raise ValueError(f"Expected an isolated script symlink: {target}")
        else:
            target.symlink_to(source.resolve())
        bindings.append(dict(path=str(target), resolved_path=str(target.resolve()),
                             sha256=data.file_hash(target), original_path=str(source),
                             original_sha256=digest, symlink=target.is_symlink()))
    data.write_new_json(ROOT / "source_bindings.json", bindings)
    return staged, bindings


def profile_snapshot(scripts, output):
    """Execute real profile() bodies on recipe views, never __init__ or run."""
    scripts, output = Path(scripts).resolve(), Path(output).resolve()
    parent = data.read_json(PARENT_LOCK)
    if scripts not in (Path(parent["scripts_root"]).resolve(), ROOT / "scripts") or not output.is_relative_to(ROOT):
        raise ValueError("Profile preparation is restricted to this private revision")
    sys.path[:0] = [parent["overlay_root"], str(scripts)]
    modules = {name: importlib.import_module("train_100k_" + name)
               for name in ("common", "backends", "sampling", "icam", "nss_ar", "alias_weights")}
    common, native = modules["common"], modules["backends"]
    if common.WORKSPACE != data.WORKSPACE or native.WORKSPACE != data.WORKSPACE:
        raise ValueError("Private scripts changed the true WORKSPACE binding")
    source = native.SOURCE_ROOT
    recipes = {
        "BQ": ("backends", "BQBackend", source / "bq-nco"),
        "ELG": ("backends", "OriginalELGBackend", source / "ELG/CVRP"),
        "ICAM": ("icam", "OriginalICAMCVRPBackend", source / "ICAM/ICAM_CVRP"),
        "LEHD": ("nss_ar", "OriginalLEHDCVRPBackend", source / "NCO_code/single_objective/LEHD/CVRP"),
        "MVMOE": ("nss_ar", "NSSMVmoeCVRPBackend", data.WORKSPACE / "EasyNCO"),
        "MoSES_CaDA": ("backends", "RouteFinderBackend", source / "moses_vrp"),
        "MoSES_RF": ("alias_weights", "AliasSafeMosesBackend", source / "moses_vrp"),
        "OMNI": ("nss_ar", "OriginalOmniCVRPBackend", source / "Omni-VRP/POMO"),
        "RELD_CVRP": ("backends", "RELDBackend", data.WORKSPACE / "reld-nco-main/CVRP"),
        "RouteFinder": ("backends", "RouteFinderBackend", source / "routefinder"),
    }
    profiles = {}
    for method, (module, name, repo) in recipes.items():
        old = data.read_json(parent["deployments"][f"CVRP/{method}"]["profile_path"])
        cls = getattr(modules[module], name)
        view = object.__new__(cls)
        view.problem, view.method, view.repo = "CVRP", method, repo
        for attr in ("parameters", "params", "model_params", "env_params"):
            if attr in old:
                setattr(view, attr, copy.deepcopy(old[attr]))
        if method == "ELG":
            view.params = copy.deepcopy(old["model_params"])
        if "checkpoint" in old:
            view.checkpoint = Path(old["checkpoint"])
        if "checkpoints" in old:
            view.checkpoints = {bucket: Path(entry["path"]) for bucket, entry in old["checkpoints"].items()}
        actual = view.profile()
        # The only changed function is outside these classes. Any profile drift
        # therefore needs a separate explanation, not an accepted replacement hash.
        if actual != old or data.canonical_hash(actual) != parent["deployments"][f"CVRP/{method}"]["profile_sha256"]:
            raise ValueError(f"Actual profile-only source/recipe projection drifted: {method}")
        profiles[method] = actual
    result = dict(profiles=profiles, workspace=str(common.WORKSPACE), source_root=str(source),
                  script_root=str(native.SCRIPT_ROOT), native_qualified=False,
                  evidence="actual profile() methods on frozen recipe views; no constructors or solver calls",
                  modules={name: dict(path=module.__file__, resolved_path=str(Path(module.__file__).resolve()),
                                      sha256=data.file_hash(module.__file__)) for name, module in modules.items()})
    data.write_new_json(output, result)
    return result


def run_snapshot(scripts, output):
    command = [sys.executable, "-B", "-m", "code.V4.r114_cvrp_recovery", "profile-snapshot",
               "--scripts-root", str(scripts), "--output", str(output)]
    completed = subprocess.run(command, cwd=data.WORKSPACE / "0selection2",
                               env=dict(os.environ, **CPU_ENV), text=True, capture_output=True, timeout=600)
    if completed.returncode:
        raise ValueError(f"CPU profile snapshot failed:\n{completed.stdout}{completed.stderr}")
    return data.read_json(output)


def build_inventory(parent, manifest, bindings, snapshot):
    """Rebuild CVRP dependencies and profiles from the observed private sources."""
    scripts, original = ROOT / "scripts", Path(parent["scripts_root"])
    artifacts, cache, records = {}, {}, []

    def observe(path, expected=None):
        path = Path(path).absolute()
        resolved = path.resolve(strict=True)
        if str(resolved) not in cache:
            cache[str(resolved)] = data.file_hash(resolved)
        digest = cache[str(resolved)]
        if expected is not None and digest != expected:
            raise ValueError(f"Dependency/source hash changed: {path}")
        for name in {str(path), str(resolved)}:
            artifacts[name] = dict(path=name, resolved_path=str(resolved), exists=True,
                                   expected_sha256=digest, current_sha256=digest, hash_matches=True)
        return digest

    def remap(path):
        path = Path(path)
        return scripts / path.relative_to(original) if path.is_relative_to(original) else path

    shared = {}
    for entry in bindings:
        shared[entry["path"]] = observe(entry["path"], entry["sha256"])
        shared[entry["resolved_path"]] = observe(entry["resolved_path"], entry["sha256"])
    for path in (Path(boundary.__file__).resolve(), ROOT / "source_bindings.json"):
        shared[str(path)] = observe(path)
    for candidate in manifest["pools"]["CVRP"]:
        method = candidate["method"]
        old_path = Path(parent["deployments"][f"CVRP/{method}"]["profile_path"])
        old_dep_path, old_binding_path = old_path.with_name("dependency_contract.json"), old_path.with_name("profile_path_bindings.json")
        for path in (old_path, old_dep_path, old_binding_path):
            observe(path, parent["artifacts"][str(path)])
        old_dep, old_binding = data.read_json(old_dep_path), data.read_json(old_binding_path)
        profile = snapshot["profiles"][method]
        dependencies = dict(shared)
        for path, expected in old_dep["artifacts"].items():
            observe(path, parent["artifacts"][path])
            current = remap(path)
            dependencies[str(current)] = observe(current, expected if current == Path(path) else None)
            dependencies[str(current.resolve())] = observe(current)
        for path, digest in profile["source_sha256"].items():
            dependencies[path] = observe(path, digest)
        path_bindings = {}
        for field, entry in old_binding["bindings"].items():
            if field in profile:
                path = remap(entry["path"])
                path_bindings[field] = dict(path=str(path), sha256=observe(path, profile[field]))
        for field, filename in (("loader_sha256", "paper_source_routefinder_runner.py"),
                                ("signed_demand_adapter_sha256", "train_100k_routefinder.py")):
            if field in profile:
                path = scripts / filename
                path_bindings[field] = dict(path=str(path), sha256=observe(path, profile[field]))
        for entry in path_bindings.values():
            dependencies[entry["path"]] = entry["sha256"]
        directory = ROOT / "profiles/CVRP" / method
        profile_path, dep_path, binding_path = [directory / name for name in
                                              ("profile.json", "dependency_contract.json", "profile_path_bindings.json")]
        profile_sha = data.canonical_hash(profile)
        data.write_new_json(profile_path, profile)
        dep = dict(schema_version=1, revision=REVISION, profile_sha256=profile_sha, artifacts=dependencies,
                   actual_scale_checkpoints=old_dep.get("actual_scale_checkpoints", {}),
                   parent_contract_path=str(old_dep_path), parent_contract_sha256=data.file_hash(old_dep_path),
                   runtime_qualified=False, evidence_scope="Recomputed CPU sources and recipe views; native gate still required")
        data.write_new_json(dep_path, dep)
        data.write_new_json(binding_path, dict(schema_version=1, revision=REVISION, profile_sha256=profile_sha,
            dependency_contract_sha256=data.file_hash(dep_path), bindings=path_bindings,
            runtime_qualified=False))
        checkpoints = copy.deepcopy(parent["deployments"][f"CVRP/{method}"]["checkpoints"])
        for entry in checkpoints:
            observe(entry["path"], entry["sha256"])
        record = dict(problem="CVRP", **candidate, backend=profile["backend"], budget=profile["budget"],
                      checkpoints=checkpoints, status="pinned_files_match", missing=[], changed=[],
                      profile_sha256=profile_sha, runtime_verified_this_turn=False, intervention_applicable=True)
        for prefix, path in (("profile", profile_path), ("dependency", dep_path), ("bindings", binding_path)):
            record[prefix + "_path"], record[prefix + "_file_sha256"] = str(path), observe(path)
        records.append(record)
    decision_path = Path(parent["legacy_root"]) / "recipe_decisions.json"
    observe(decision_path, parent["artifacts"][str(decision_path)])
    overlay = Path(parent["overlay_root"])
    overlay_artifacts = {str(path.resolve()): observe(path, parent["artifacts"][str(path.resolve())])
                         for path in sorted(overlay.rglob("*")) if path.is_file() and path.suffix in (".py", ".so")}
    inventory = dict(schema_version=1, revision=REVISION, adapter_entry=data.ADAPTER_ENTRY,
        scripts_root=str(scripts), legacy_root=parent["legacy_root"], overlay_root=str(overlay),
        overlay_exists=overlay.is_dir(), overlay_artifacts=overlay_artifacts, artifacts=artifacts,
        hashes_verified=True, records=records, sampling_decision=data.read_json(decision_path)["mvrp_decoder_strategy"],
        runtime=data.runtime_description(overlay), runtime_verified=False, solver_execution=False,
        interpretation="CVRP-only CPU source/recipe preparation, not initialized-backend qualification")
    if inventory["runtime"] != parent["runtime"]:
        raise ValueError("Frozen Python/package runtime changed")
    inventory["inventory_sha256"] = data.canonical_hash(inventory)
    data.verify_inventory(inventory)
    return inventory


def runner_argv(plan, execute=False):
    args = ["--plan", str(ROOT / "queue/plan.json"), "--manifest", plan["manifest_path"],
            "--deployment-lock", plan["deployment_lock_path"]]
    if execute:
        args += ["--host", "OWNER_HOST", "--gpu-uuid", "GPU-OWNER_UUID", "--cpu-ids", "0,1",
                 "--timeout-seconds", "3600", "--execute"]
    return args


def parse_runner_argv(argv):
    """Use the existing runner's real parser without reaching execution code."""
    from . import r114_run_queue as runner
    tree = ast.parse(Path(runner.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = []
    for node in function.body:
        body.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            break
    namespace = dict(argparse=argparse, Path=Path, argv=argv, __doc__=runner.__doc__)
    exec(compile(ast.Module(body=body, type_ignores=[]), "<original-runner-parser-only>", "exec"), namespace)
    return namespace["args"]


def verify():
    plan, manifest, lock = queue.load_plan(ROOT / "queue/plan.json")
    data.verify_manifest(manifest)
    for path in (ROOT / "cold.lock.json", Path(plan["deployment_lock_path"])):
        data.verify_lock(data.read_json(path), manifest, verify_files=True)
    for entry in data.read_json(ROOT / "source_bindings.json"):
        if str(Path(entry["path"]).resolve()) != entry["resolved_path"] or data.file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("Private script source/target binding changed")
    parent = data.read_json(PARENT_MANIFEST)
    if manifest != subset_manifest(parent):
        raise ValueError("Recovery no longer matches the frozen CVRP subset")
    old = data.read_json(PARENT_PLAN)
    for phase in ("runtime_jobs", "label_jobs"):
        previous = {job["job_id"]: job for job in old[phase] if job["problem"] == "CVRP"}
        for job in plan[phase]:
            for field in ("task_ids", "replay_task_ids", "cold_task_ids", "bucket_gates", "reuse_mode", "requires"):
                if job.get(field) != previous[job["job_id"]].get(field):
                    raise ValueError(f"Qualification/production protocol changed: {job['job_id']}/{field}")
    return plan, manifest, lock


def prepare():
    parent = data.read_json(PARENT_LOCK)
    data.verify_lock(parent, data.read_json(PARENT_MANIFEST))
    anchors = data.read_json(boundary.ROOT / "revision.json")["frozen_evidence_sha256"]
    if any(data.file_hash(path) != digest for path, digest in anchors.items()):
        raise ValueError("Original frozen evidence changed")
    manifest = subset_manifest(data.read_json(PARENT_MANIFEST))
    data.verify_manifest(manifest)
    data.write_new_json(ROOT / "manifest.json", manifest)
    scripts, bindings = freeze_scripts(parent)
    run_snapshot(parent["scripts_root"], ROOT / "profiles.original.snapshot.json")
    snapshot = run_snapshot(scripts, ROOT / "profiles.corrected.snapshot.json")
    inventory = build_inventory(parent, manifest, bindings, snapshot)
    data.write_new_json(ROOT / "inventory.json", inventory)
    cold = data.source_lock(inventory, manifest)
    cold["acquisition_revision"] = REVISION
    cold["parent_cold_lock_sha256"] = parent["lock_sha256"]
    cold.pop("lock_sha256")
    cold["lock_sha256"] = data.canonical_hash(cold)
    data.write_new_json(ROOT / "cold.lock.json", cold)
    flat = queue.build_plan(ROOT / "manifest.json", ROOT / "cold.lock.json", ROOT / "cold_queue")
    # Protocol IDs include the new lock hash. Keep the frozen bucket/probe
    # order instead of letting those new IDs reorder qualification calls.
    frozen = data.read_json(PARENT_PLAN)
    buckets = [bucket for job in frozen["runtime_jobs"] if job["problem"] == "CVRP"
               for bucket in job["bucket_gates"]]
    bucket_order = {bucket: index for index, bucket in enumerate(buckets)}
    flat["runtime_jobs"].sort(key=lambda job: bucket_order[job["deployment_key"]])
    flat.pop("plan_sha256")
    flat["plan_sha256"] = data.canonical_hash(flat)
    data.write_new_json(ROOT / "cold_queue/plan.json", flat)
    grouped = queue.build_group_plan(ROOT / "cold_queue/plan.json", ROOT / "queue")
    data.write_new_json(ROOT / "queue/plan.json", grouped)
    plan, _, _ = verify()
    parse_runner_argv(runner_argv(plan, execute=True))
    dry_command = [sys.executable, "-B", "-m", "code.V4.r114_run_queue", *runner_argv(plan)]
    dry = subprocess.run(dry_command, cwd=data.WORKSPACE / "0selection2", env=dict(os.environ, **CPU_ENV),
                         text=True, capture_output=True, timeout=120, check=True)
    dry_result = json.loads(dry.stdout)
    data.write_new_json(ROOT / "runner.dry_run.json", dry_result)
    command = shlex.join(dry_command)
    script = (
        "#!/usr/bin/env bash\nsource /public/home/shiys/miniconda3/etc/profile.d/conda.sh\nconda activate easynco\n"
        "set -euo pipefail\ncd " + shlex.quote(str(data.WORKSPACE / "0selection2")) + "\n"
        ': "${R114_RECOVERY_HOST:?Main must set the container hostname}"\n'
        ': "${R114_RECOVERY_GPU_UUID:?Main must hold the external GPU mutex}"\n'
        ': "${R114_RECOVERY_CPU_IDS:?Main must assign 1-2 CPU IDs}"\n'
        ': "${R114_RECOVERY_TIMEOUT_SECONDS:?Main must supply the unchanged timeout}"\n'
        'export CUDA_VISIBLE_DEVICES="$R114_RECOVERY_GPU_UUID"\nexport PYTHONDONTWRITEBYTECODE=1\n'
        + command + ' --host "$R114_RECOVERY_HOST" --gpu-uuid "$R114_RECOVERY_GPU_UUID"'
        ' --cpu-ids "$R114_RECOVERY_CPU_IDS" --timeout-seconds "$R114_RECOVERY_TIMEOUT_SECONDS" --execute\n'
    )
    boundary._write_new_text(ROOT / "run_native.sh", script)
    subprocess.run(["bash", "-n", str(ROOT / "run_native.sh")], check=True)
    if any(data.file_hash(path) != digest for path, digest in anchors.items()):
        raise ValueError("Original evidence changed during preparation")
    ready = dict(status="CPU_READY_NOT_NATIVELY_QUALIFIED", revision=REVISION, cpu_ready=True,
        native_qualified=False, solver_execution=False, main_ready_unchanged=True,
        scope=dict(cvrp_counts(manifest),
                   endpoint_count_note="Derived solely from frozen parent CVRP rows; no endpoints added"),
        manifest_sha256=manifest["manifest_sha256"], inventory_sha256=inventory["inventory_sha256"],
        cold_lock_sha256=cold["lock_sha256"], grouped_lock_sha256=plan["lock_sha256"],
        plan_sha256=plan["plan_sha256"], plan=str(ROOT / "queue/plan.json"),
        scripts_root=str(scripts), native_cli=str(ROOT / "run_native.sh"), native_cli_executed=False,
        original_runner_parser_checked=True, bash_syntax_checked=True, dry_run_argv=dry_command,
        qualification=plan["summary"], original_evidence_sha256=anchors,
        builder=dict(path=str(Path(__file__).resolve()), sha256=data.file_hash(__file__)),
        profile_evidence=snapshot["evidence"], native_profile_equivalence_still_required=True,
        main_owns="Container, GPU mutex, scheduling after v1 terminal, combined dataset/analysis integration; no --finalize here")
    data.write_new_json(ROOT / "CPU_READY.json", ready)
    return ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify")
    snapshot = sub.add_parser("profile-snapshot")
    snapshot.add_argument("--scripts-root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU preparation/checks require CUDA_VISIBLE_DEVICES=''; this tool never executes solvers")
    if args.command == "prepare":
        result = prepare()
    elif args.command == "profile-snapshot":
        result = profile_snapshot(args.scripts_root, args.output)
    else:
        plan, manifest, _ = verify()
        result = dict(status="CPU_VALIDATED_NOT_NATIVELY_QUALIFIED", plan_sha256=plan["plan_sha256"],
                      endpoint_count=manifest["endpoint_count"], task_count=manifest["task_count"])
    print(json.dumps({k: result[k] for k in ("status", "plan_sha256") if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
