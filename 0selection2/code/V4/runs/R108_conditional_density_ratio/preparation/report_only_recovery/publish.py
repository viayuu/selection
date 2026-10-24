"""One explicit, bounded R108 recovery publication under the shared Git lock."""

import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parents[3]
GIT_ROOT = PROJECT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))
from recover import ARMS, digest, read, save


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(args, *, data=None, timeout=120):
    result = subprocess.run(["git", *args], cwd=GIT_ROOT, input=data, text=True,
        capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError("git " + " ".join(args) + ": " + result.stderr.strip())
    return result.stdout.strip()


def gate(goal):
    original = goal.status(persist=False)
    assert not original["errors"]
    assert original["matched_controls"] and original["identical_full_initialization"]
    assert all(a["missing"] == ["same_model_NSS4"] for a in original["arms"])
    locks = {a: read(ROOT / a / "final_test_lock.json") for a in ARMS}
    for arm in ARMS:
        assert goal.nss4_ready(HERE / "normalized" / arm, locks[arm])
    compared = read(HERE / "comparison_artifacts/protocol.json")
    assert compared["experiment"] == "R108" and compared["complete"]
    assert compared["checkpoints"] == locks
    assert compared["training"] is False and compared["inference"] is False
    assert compared["total_parameter_matched"] is True
    assert compared["active_parameter_matched"] is False
    assert compared["equal_flops_claim"] is False and compared["equal_gradient_scale_claim"] is False
    assert compared["identical_complete_fresh_parameters"] is True
    assert compared["priors_sha256"] == read(ROOT / ARMS[0] / "protocol.json")["priors_sha256"]
    for name in ("paired_changes_B_vs_A.csv", "target_recovery_B_vs_A.csv", "target_recovery_summary_B_vs_A.csv"):
        assert (HERE / "comparison_artifacts" / name).stat().st_size > 0
    assert read(HERE / "completion_receipt.json")["complete"]
    assert read(HERE / "compare_receipt.json")["exit_status"] == 0
    for path, expected in read(HERE / "backup_manifest.json")["protected_sha256"].items():
        assert digest(path) == expected, "Frozen file changed: " + path
    return dict(complete=True, kind="additive_recovery_gate", checked_utc=utc(),
        canonical_gate_complete=False, original_status=original,
        all_original_non_NSS_checks_preserved=True,
        original_NSS_gate_passed_on_derived_CSV=True,
        same_lock_original_reporter_comparison_complete=True,
        frozen_sources_protocols_weights_predictions_and_receipts_unchanged=True,
        training=False, inference=False, tuning=False, reporter_source_modified=False)


def bounded_pack(commit, remote):
    config = ["-c", "pack.threads=1", "-c", "pack.windowMemory=64m",
              "-c", "pack.deltaCacheSize=16m"]
    process = subprocess.Popen(["git", *config, "pack-objects", "--stdout", "--revs",
        "--thin", "--delta-base-offset", "--quiet"], cwd=PROJECT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.stdin.write((commit + "\n^" + remote + "\n").encode())
    process.stdin.close()
    count = 0
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        count += len(block)
        if count > 32 * 1024**2:
            process.kill()
            raise RuntimeError("Incremental pack exceeds 32MiB; no push")
    error = process.stderr.read().decode(errors="replace")
    if process.wait(timeout=60):
        raise RuntimeError("Pack preflight failed: " + error)
    return dict(bytes=count, method="streamed and discarded thin incremental pack", threads=1,
                window_memory="64m", delta_cache="16m", network_byte_measurement=False)


def main():
    resume = sys.argv[1:] == ["--resume-pathspec-only"]
    suffix = "_pathfix" if resume else ""
    if resume:
        failed = read(HERE / "publication_failure.json")
        assert failed["new_commit_created"] is False
        assert "did not match any files" in failed["error"]
        assert not git(["diff", "--cached", "--name-only"])
    if (HERE / ("publication_started" + suffix + ".json")).exists():
        raise RuntimeError("Publication already attempted; no automatic retry")
    goal = importlib.import_module("code.V4.r108_goal")
    git_root = Path(git(["rev-parse", "--show-toplevel"]))
    assert git_root == GIT_ROOT
    git_dir = Path(git(["rev-parse", "--absolute-git-dir"]))
    prefix = str(ROOT.relative_to(git_root)) + "/"
    with (git_dir / "codex-publication.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        assert git(["branch", "--show-current"]) == "main"
        assert not git(["diff", "--cached", "--name-only"]), "Existing staged files; leave index unchanged"
        proof = gate(goal)
        save(HERE / ("recovered_gate" + suffix + ".json"), proof)
        if resume:
            save(HERE / "publisher_path_fix.json", dict(
                kind="administrative_git_cwd_only", numerical_logic_changed=False,
                original_source_sha256=digest(HERE / "publish_original_pathspec_failure.py"),
                fixed_source_sha256=digest(Path(__file__)),
                original_failure_preserved=True, original_index_empty=True,
                change="Run root-relative Git pathspecs at the verified Git root; preserve prior receipts with additive suffixes"))
        head = git(["rev-parse", "HEAD"])
        remote_line = git(["ls-remote", "origin", "refs/heads/main"], timeout=45)
        remote = remote_line.split()[0]
        assert remote == head, "Remote/main must equal local HEAD; no fetch/rebase/force"
        candidates = goal.publication_paths("R108", ROOT, goal.LAUNCH, git_root)
        own_sources = set()
        for relative in candidates:
            p = git_root / relative
            if p.parent == PROJECT / "code/V4" and p.name.startswith(("r108_", "test_r108_", "run_v4_r108_")):
                own_sources.add(relative)
        files = {p for p in candidates if p.startswith(prefix)} | own_sources
        files |= {str((HERE / name).relative_to(git_root)) for name in ("recover.py", "publish.py")}
        if resume:
            files.add(str((HERE / "publish_original_pathspec_failure.py").relative_to(git_root)))
        inventory = []
        for relative in sorted(files):
            p = git_root / relative
            assert p.is_file() and not p.is_symlink()
            assert p.suffix in {".json", ".csv", ".pdf", ".md", ".py", ".sh"}
            assert p.stat().st_size <= 32 * 1024**2
            assert relative.startswith(prefix) or relative in own_sources
            inventory.append(dict(path=relative, bytes=p.stat().st_size, sha256=digest(p)))
        assert sum(r["bytes"] for r in inventory) < 64 * 1024**2
        plan = dict(experiment="R108", kind="curated_additive_report_recovery", prepared_utc=utc(),
            authorization="Main explicitly authorizes report-only recovery and completed-version curated GitHub publication",
            parent_commit=head, remote_before=remote, files=inventory, file_count=len(inventory),
            total_asset_bytes=sum(r["bytes"] for r in inventory),
            original_metadata_overwritten=False, one_commit=True, automatic_retry=False,
            source_logic_modified=False, numeric_metrics_modified=False,
            delivery_receipt="Post-push receipt is additive and intentionally uncommitted, following R105 recovery pattern")
        plan_path = HERE / ("publication_plan" + suffix + ".json")
        save(plan_path, plan)
        files.add(str(plan_path.relative_to(git_root)))
        files = sorted(files)
        save(HERE / ("publication_started" + suffix + ".json"), dict(state="staging", started_utc=utc(),
            parent_commit=head, remote_before=remote, global_mutex=str(git_dir / "codex-publication.lock"),
            file_count=len(files), curated_only=True, automatic_retry=False))
        receipt = dict(experiment="R108", kind="explicit_additive_recovery_publication", complete=False,
            started_utc=utc(), parent_commit=head, remote_before=remote,
            global_mutex=str(git_dir / "codex-publication.lock"), curated_only=True,
            canonical_original_gate_unchanged=True, automatic_retry=False, new_commit_created=False)
        try:
            for offset in range(0, len(files), 60):
                git(["add", "-f", "--", *files[offset:offset + 60]])
            staged = git(["diff", "--cached", "--name-only"]).splitlines()
            assert staged and set(staged) <= set(files), "Unexpected staged files; no commit/reset"
            assert git(["rev-parse", "HEAD"]) == head
            for item in inventory:
                assert digest(git_root / item["path"]) == item["sha256"]
            git(["-c", "gc.auto=0", "commit", "-m", "r108: publish completed full18 report-only recovery"])
            commit = git(["rev-parse", "HEAD"])
            receipt.update(commit=commit, new_commit_created=True, changed_path_count=len(staged))
            assert git(["rev-list", "--count", commit, "^" + remote]) == "1"
            objects = git(["rev-list", "--objects", commit, "^" + remote]).splitlines()
            sizes = git(["cat-file", "--batch-check=%(objectsize)"],
                        data="\n".join(x.split()[0] for x in objects) + "\n")
            uncompressed = sum(int(x) for x in sizes.splitlines())
            assert uncompressed < 64 * 1024**2
            receipt["new_object_count"] = len(objects)
            receipt["uncompressed_new_object_bytes"] = uncompressed
            receipt["pack_preflight"] = bounded_pack(commit, remote)
            assert git(["ls-remote", "origin", "refs/heads/main"], timeout=45).split()[0] == remote
            push = git(["-c", "pack.threads=1", "-c", "pack.windowMemory=64m", "-c", "pack.deltaCacheSize=16m",
                "push", "--porcelain", "origin", commit + ":refs/heads/main"], timeout=180)
            after = git(["ls-remote", "origin", "refs/heads/main"], timeout=45).split()[0]
            assert after == commit
            assert git(["rev-parse", "HEAD"]) == commit
            assert not git(["diff", "--cached", "--name-only"])
            gate(goal)
            receipt.update(complete=True, state="complete", exit_status=0, completed_utc=utc(),
                remote_after=after, push_porcelain=push, push_attempts=1, force_push=False,
                history_rewritten=False, unrelated_changes_staged=False,
                verification="Independent post-push git ls-remote origin refs/heads/main",
                actual_receipt_intentionally_uncommitted=True)
            save(HERE / "publication_receipt.json", receipt)
            print(json.dumps(receipt, indent=2), flush=True)
        except BaseException as error:
            receipt.update(state="failed", exit_status=1, completed_utc=utc(),
                error_type=type(error).__name__, error=str(error))
            save(HERE / ("publication_failure" + suffix + ".json"), receipt)
            raise


if __name__ == "__main__":
    main()
