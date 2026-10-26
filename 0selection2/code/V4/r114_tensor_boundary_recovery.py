"""CPU-only preparation of a private audit-boundary revision, never a runner."""

from __future__ import annotations

import argparse
import ast
import difflib
from pathlib import Path

from . import r114_response_data as data


REVISION = "r114_cvrp_audit_tensor_boundary_v1_20261026"
ROOT = data.PREPARATION / "tensor_boundary_recovery_20261026"
EXPECTED_BACKENDS_SHA256 = "46a1dd32b169366673218a6315aee1f44d256393b7bacb54968ec0122500d117"
AUDIT_SIGNATURE = "def audit_tours(problem, items, tours, costs):\n"
AUDIT_INSERTION = (
    '    if problem == "CVRP":\n'
    "        from code.V4.r114_tensor_boundary_recovery import cpu_numpy_view\n"
    "        items = cpu_numpy_view(items)\n"
)


def cpu_numpy_view(value):
    """Copy tensors to host only at the audit boundary; preserve dtype/shape."""
    if isinstance(value, dict):
        return {key: cpu_numpy_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_numpy_view(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_numpy_view(item) for item in value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def revised_source(source):
    """Render exactly one guarded insertion, without importing native solvers."""
    if data.hashlib.sha256(source.encode()).hexdigest() != EXPECTED_BACKENDS_SHA256:
        raise ValueError("Frozen native adapter changed; prepare a different revision")
    if source.count(AUDIT_SIGNATURE) != 1:
        raise ValueError("Expected one native audit entry")
    revised = source.replace(AUDIT_SIGNATURE, AUDIT_SIGNATURE + AUDIT_INSERTION, 1)
    before, after = ast.parse(source), ast.parse(revised)
    audit = next(node for node in after.body if isinstance(node, ast.FunctionDef)
                 and node.name == "audit_tours")
    del audit.body[0]
    if ast.dump(before) != ast.dump(after):
        raise ValueError("Revision changes more than the audit entry")
    compile(revised, "<private-r114-native-adapter-revision>", "exec")
    return revised


def _write_new_text(path, content):
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Immutable recovery artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(content)


def prepare():
    """Stage a patch and evidence; do not create an executable deployment lock."""
    manifest_path = data.PREPARATION / "production_inputs/manifest.json"
    lock_path = data.PREPARATION / "production_execution/deployments.production.lock.json"
    grouped_root = data.PREPARATION / "grouped_execution/queue_v1"
    grouped_lock_path = grouped_root / "deployments.grouped.lock.json"
    manifest, lock = data.read_json(manifest_path), data.read_json(lock_path)
    grouped_lock = data.read_json(grouped_lock_path)
    data.verify_lock(lock, manifest)
    data.verify_lock(grouped_lock, manifest)
    source_path = Path(lock["scripts_root"]) / "train_100k_backends.py"
    if lock["artifacts"].get(str(source_path)) != EXPECTED_BACKENDS_SHA256:
        raise ValueError("Parent lock does not pin the expected native adapter")
    source = source_path.read_text()
    revised = revised_source(source)
    receipt_path = grouped_root / (
        "attempts/runtime_group__CVRP__ICAM/attempt_0001/cold/"
        "CVRP_src53138_original__ICAM.json"
    )
    receipt = data.read_json(receipt_path)
    if (receipt.get("task_id") != "CVRP_src53138_original__ICAM"
            or receipt.get("status") != "error"
            or receipt.get("error_type") != "TypeError"
            or "can't convert cuda:0 device type tensor to numpy" not in receipt.get("error", "")
            or receipt.get("lock_sha256") != lock["lock_sha256"]):
        raise ValueError("Expected original failure receipt is missing or changed")
    pool = manifest["pools"]["CVRP"]
    if {f"CVRP/{entry['method']}" for entry in pool} != {
            key for key in lock["deployments"] if key.startswith("CVRP/")}:
        raise ValueError("Full legal CVRP pool does not match the parent lock")
    patch_path = ROOT / "source_revision/train_100k_backends.py.patch"
    patch = "".join(difflib.unified_diff(
        source.splitlines(keepends=True), revised.splitlines(keepends=True),
        fromfile="a/train_100k_backends.py", tofile="b/train_100k_backends.py",
    ))
    tracked = [
        manifest_path, lock_path, grouped_lock_path, grouped_root / "plan.json",
        receipt_path, receipt_path.with_suffix(".command.json"),
        Path(data.__file__).resolve(), Path(__file__).with_name("r114_group_worker.py"),
        source_path, Path(lock["scripts_root"]) / "train_100k_icam.py",
        Path(lock["scripts_root"]) / "train_100k_nss_ar.py",
        Path(lock["scripts_root"]) / "train_100k_alias_weights.py",
        Path(lock["scripts_root"]) / "train_100k_common.py",
    ]
    observed = {str(path): data.file_hash(path) for path in tracked}
    for path, digest in observed.items():
        expected = grouped_lock["artifacts"].get(path)
        if expected is not None and expected != digest:
            raise ValueError(f"Frozen evidence changed: {path}")
    _write_new_text(patch_path, patch)
    revision = dict(
        schema_version=1, revision=REVISION,
        status="prepared_not_integrated_not_executed", ready_for_scheduling=False,
        solver_execution=False, gpu_probes=False, native_runtime_verified=False,
        full_pool_complete=False, historical_reproduction=False,
        parent_manifest_sha256=manifest["manifest_sha256"],
        parent_cold_lock_sha256=lock["lock_sha256"],
        parent_grouped_lock_sha256=grouped_lock["lock_sha256"],
        original_failure=dict(path=str(receipt_path), sha256=observed[str(receipt_path)],
                              error_type=receipt["error_type"], error=receipt["error"],
                              classification="tensor_to_numpy_interface_error_not_native_paradigm_failure",
                              traceback_available=False, boundary_evidence="frozen_source_trace_and_CPU_tests"),
        patch=dict(path=str(patch_path), sha256=data.file_hash(patch_path),
                   original_source=str(source_path), original_sha256=EXPECTED_BACKENDS_SHA256,
                   revised_sha256=data.hashlib.sha256(revised.encode()).hexdigest()),
        helper=dict(path=str(Path(__file__).resolve()), sha256=data.file_hash(__file__)),
        cvrp_pool=pool,
        cvrp_deployments={key: value for key, value in lock["deployments"].items()
                          if key.startswith("CVRP/")},
        frozen_evidence_sha256=observed,
        invariants=["same manifest, input hashes, tasks, solver seeds and full legal pool",
                    "same solver tensors, default device, weights, precision and budgets",
                    "same route validation, thresholds and cost recomputation",
                    "same cold/grouped qualification and lazy-runner lifecycle",
                    "only detached host views at the CVRP audit boundary"],
        integration_required=[
            "Wait for main to authorize a new source/plan/lock revision; never edit queue_v1",
            "Apply patch only to an isolated complete scripts freeze; pin this helper too",
            "Preserve native path bindings; regenerate all source-dependent profiles and locks",
            "Use the same revised scripts/helper in both cold and grouped processes",
            "Requalify every legal CVRP deployment under existing exact-equivalence gates",
            "Keep v1 errors and receipts; new receipts must carry the new lock provenance",
        ],
    )
    data.write_new_json(ROOT / "revision.json", revision)
    if any(data.file_hash(path) != digest for path, digest in observed.items()):
        raise ValueError("Frozen evidence changed during recovery preparation")
    return revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    revision = prepare()
    print(f"{revision['status']}: {ROOT / 'revision.json'}")


if __name__ == "__main__":
    main()
