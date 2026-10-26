"""One curated research publication after actual Astra completion and PDF render."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
GIT = Path("/public/home/shiys")
FILES = ("REQUEST.md", "EVIDENCE.md", "ROUTING.json", "REPORT.md", "sources.csv", "directions.csv",
         "REPORT.pdf", "render_notes.json", "DELIVERY.md", "render_pdf.py", "publish_research.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def git(*args):
    process = subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(GIT), *args],
                             capture_output=True, timeout=120)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace")[-2000:])
    return process.stdout.decode().strip()


def publish():
    receipt_path = ROOT / "publication_receipt.json"
    gitdir = Path(git("rev-parse", "--absolute-git-dir"))
    with (gitdir / "codex-publication.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get("state") == "complete":
                return receipt
            raise RuntimeError("Previous publication attempt exists; preserve it and reconcile explicitly")
        if Path(git("rev-parse", "--show-toplevel")).resolve() != GIT or git("diff", "--cached", "--name-only", "-z"):
            raise RuntimeError("Unexpected Git root or unrelated staged changes; leave index untouched")
        routing, rendered = read(ROOT / "ROUTING.json"), read(ROOT / "render_notes.json")
        if routing.get("phase") != "research_complete" or rendered.get("complete") is not True:
            raise ValueError("Require completed research and unchanged-source PDF")
        for name, expected in rendered["inputs"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Scientific handoff changed after rendering: " + name)
        if sha(ROOT / "REPORT.pdf") != rendered["pdf"]["sha256"]:
            raise ValueError("PDF changed after rendering")
        paths = [ROOT / name for name in FILES]
        if any(not p.is_file() or p.is_symlink() or not p.is_relative_to(GIT) for p in paths):
            raise ValueError("Only exact ordinary files under the repository may be published")
        if any(p.stat().st_size > 8 * 2**20 for p in paths) or sum(p.stat().st_size for p in paths) > 32 * 2**20:
            raise ValueError("Bounded research publication is unexpectedly large")
        branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
        remote = git("config", "--get", f"branch.{branch}.remote")
        merge = git("config", "--get", f"branch.{branch}.merge")
        parent = git("rev-parse", "HEAD")
        upstream = git("ls-remote", remote, merge).split()
        if len(upstream) != 2 or upstream[0] != parent:
            raise ValueError("Local HEAD and upstream differ; no force push or unrelated reconciliation")
        relative = [str(p.relative_to(GIT)) for p in paths]
        hashes = {str(p.relative_to(GIT)): sha(p) for p in paths}
        manifest = ROOT / "publication_manifest.json"
        save(manifest, dict(parent=parent, files=hashes, scientific_source_changed=False,
            git_autocrlf="false for this publisher only", single_commit=True,
            active_goal_complete=False))
        relative.append(str(manifest.relative_to(GIT)))
        record = dict(state="prepared", parent=parent, remote=remote, merge=merge, allowlist=relative,
            commit_attempts=0, push_attempts=0, global_mutex=str(gitdir / "codex-publication.lock"))
        with receipt_path.open("x") as stream:
            json.dump(record, stream, indent=2)
        try:
            if any(sha(GIT / name) != expected for name, expected in hashes.items()):
                raise ValueError("Curated input changed before staging")
            git("add", "-f", "--", *relative)
            staged = set(filter(None, git("diff", "--cached", "--name-only", "-z").split("\0")))
            if not staged or staged - set(relative):
                raise ValueError("Staged files exceed the exact research allowlist")
            record.update(state="committing", commit_attempts=1)
            save(receipt_path, record)
            git("commit", "--only", "-m", "Publish Astra major-direction research after R115 and R116", "--", *relative)
            commit = git("rev-parse", "HEAD")
            changed = set(git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines())
            if changed - set(relative) or git("rev-parse", commit + "^") != parent:
                raise ValueError("Committed payload differs from the curated research version")
            record.update(state="pushing", commit=commit, push_attempts=1)
            save(receipt_path, record)
            git("push", "--porcelain", remote, commit + ":" + merge)
            if git("ls-remote", remote, merge).split()[0] != commit:
                raise ValueError("Remote does not contain the completed publication")
            record.update(state="complete", exit_status=0)
            save(receipt_path, record)
            return record
        except BaseException as error:
            record.update(state="failed", error_type=type(error).__name__, error=str(error), exit_status=1)
            save(receipt_path, record)
            raise


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False))
