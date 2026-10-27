"""Saved-only closeout of realized TRAIN assets; never a selector result."""

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import r118_trace_assets as data
from .r112_finish import git, publication_mutex


ROOT = data.ROOT
RECOVERY = ROOT / "recovery_column_indices_v2"
GIT = data.common.WORKSPACE
CODE = ("r118_trace_assets.py", "r118_reld_trace.py", "r118_td_trace.py", "r118_trace_launch.py",
        "r118_trace_recovery.py", "r118_trace_finish.py", "run_v4_r118_trace.sh",
        "run_v4_r118_trace_recovery.sh", "test_r118_reld_trace.py", "test_r118_td_trace.py")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    collections = {"A": ROOT, "B": RECOVERY}
    for arm, root in collections.items():
        worker = data.common.read_json(root / "preparation/resources/capture" / arm / "worker.json")
        if worker.get("state") != "complete" or worker.get("exit_status") != 0:
            raise RuntimeError(f"Await actual completed capture {arm}: {worker.get('state')}")
        if [x["stage"] for x in worker["stages"]] != data.SLOTS[arm]:
            raise ValueError("Capture task coverage differs")
        ready = data.common.read_json(root / "preparation/capture_ready.json")
        for path, expected in ready["sources"].items():
            if data.common.file_hash(path) != expected:
                raise ValueError("Capture source/plan changed: " + path)
    cases, summary = [], []
    inputs = {}
    for problem in data.common.PROBLEMS:
        arm = next(a for a in collections if problem in data.SLOTS[a])
        root = collections[arm]
        plan = data.common.read_json(root / "capture_plan.json")
        deployment = next(x for x in plan["deployments"] if x["problem"] == problem)
        directory = root / "capture" / problem
        status = data.common.read_json(directory / "status.json")
        if status.get("state") != "complete" or status.get("completed") != data.COUNT:
            raise ValueError("All eight real observations required per task")
        split = data.common.DATA_ROOT / (problem + "train")
        metadata = data.common.read_json(split / "metadata.json")
        column = metadata["methods"].index(deployment["method"])
        # Original TRAIN costs are read only now, after input-selected native runs.
        with np.load(split / "targets.npz", allow_pickle=False) as archive:
            old_costs = archive["costs"][:, column]
        rows = []
        for member in deployment["selection"]:
            path = directory / (f"source{int(member['source_index'])}.json")
            record = data.common.read_json(path)
            if (record["profile_sha256"] != deployment["profile_sha256"]
                    or record["local_index"] != int(member["local_index"])
                    or record["plan_sha256"] != plan["plan_sha256"]
                    or data.common.file_hash(record["trace"]) != record["trace_sha256"]):
                raise ValueError("Native trace provenance differs")
            inputs[str(path)] = data.common.file_hash(path)
            with np.load(record["trace"], allow_pickle=False) as trace:
                td = "decoder_probs" in trace.files
                probabilities = trace["decoder_probs" if td else "decoder_probabilities"]
                valid = trace["decoder_valid"]
                finished = trace["pre_finished"]
                active = valid & ~finished
                observed = probabilities[active]
                if not len(observed) or not np.isfinite(observed).all():
                    raise ValueError("Finite genuine nonterminal decoder observations required")
                if td and problem == "TSP":
                    if not np.isin(observed[observed < 0], [-2.]).all():
                        raise ValueError("Unexpected native LEHD sentinel")
                    observed = np.maximum(observed, 0)
                if (observed < 0).any() or not np.allclose(observed.sum(-1), 1, atol=3e-5, rtol=3e-5):
                    raise ValueError("Recorded real decoder probabilities are not normalized")
                tours = trace["final_tours" if td else "all_start_tours"]
                costs = trace["final_costs" if td else "all_start_cost"]
                actions = trace["action"]
                if not td:
                    tours, costs, actions = tours[0], costs[0], actions[:, 0]
                if not np.array_equal(actions.T, tours):
                    raise ValueError("Physical trajectory/action order changed")
                best = int(costs.argmin())
                if (not np.isclose(costs[best], record["native_cost"], rtol=2e-6, atol=5e-5)
                        or tours[best].tolist() != record["native_best_tour"]):
                    raise ValueError("Recorder and original native result differ")
                steps, starts = actions.shape
                probability_rows = int(active.sum())
                role_recorded = td and "initial_col_roles" in trace.files
            old = float(old_costs[record["local_index"]])
            row = dict(problem=problem, method=deployment["method"], split="TRAIN",
                source_index=record["source_index"], local_index=record["local_index"], scale=record["scale"],
                steps=steps, starts=starts, nonterminal_probability_rows=probability_rows,
                original_train_cost=old, captured_native_cost=record["native_cost"],
                native_minus_original=record["native_cost"] - old,
                cost_close=bool(np.isclose(record["native_cost"], old, rtol=2e-6, atol=5e-5)),
                matnet_native_roles_recorded=role_recorded,
                capture_seconds=record["capture_seconds"], compressed_bytes=record["compressed_bytes"],
                trace=record["trace"], trace_sha256=record["trace_sha256"],
                capture_plan_sha256=record["plan_sha256"], collection="initial" if arm == "A" else "explicit_column_index_recovery")
            rows.append(row)
            cases.append(row)
        summary.append(dict(problem=problem, method=deployment["method"], count=len(rows),
            scale_min=min(x["scale"] for x in rows), scale_max=max(x["scale"] for x in rows),
            steps_min=min(x["steps"] for x in rows), steps_max=max(x["steps"] for x in rows),
            starts_min=min(x["starts"] for x in rows), starts_max=max(x["starts"] for x in rows),
            nonterminal_probability_rows=sum(x["nonterminal_probability_rows"] for x in rows),
            cost_close=sum(x["cost_close"] for x in rows),
            max_absolute_cost_difference=max(abs(x["native_minus_original"]) for x in rows),
            capture_seconds=sum(x["capture_seconds"] for x in rows),
            compressed_bytes=sum(x["compressed_bytes"] for x in rows)))
    if len(cases) != 144 or len(summary) != 18:
        raise ValueError("Partial pool cannot be declared a completed asset preparation")
    write_csv(ROOT / "cases.csv", cases)
    write_csv(ROOT / "coverage.csv", summary)
    close_count = sum(x["cost_close"] for x in cases)
    aligned_tasks = sum(x["cost_close"] == data.COUNT for x in summary)
    atsp = next(x for x in summary if x["problem"] == "ATSP")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(ROOT / "RESULTS.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")
        fig.suptitle("R118 | Realized TRAIN Computation Assets (Not A Selector Result)", fontsize=14)
        columns = ["Task", "Native method", "Cases", "Nodes", "Steps", "Starts", "Active prob rows", "Cost close"]
        table = ax.table(colLabels=columns, cellText=[
            [x["problem"], x["method"], x["count"], f"{x['scale_min']}-{x['scale_max']}",
             f"{x['steps_min']}-{x['steps_max']}", f"{x['starts_min']}-{x['starts_max']}",
             x["nonterminal_probability_rows"], f"{x['cost_close']}/8"] for x in summary],
            bbox=[0, .08, 1, .81], cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        fig.text(.03, .93, "18/18 problems; 18/89 deployments. Frozen input-only TRAIN selection, seed2. No holdout answers or label changes.", fontsize=9)
        fig.text(.03, .035, "Actual native recipe; fresh per-source RNG, not historical process replay. Forced distributions excluded; masks are not independent certificates.", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")
        fig.suptitle("What This Preparation Does And Does Not Establish", fontsize=15)
        lines = [
            "Captured: every actual action, node/trajectory identity, real decoder output and pre/post state, not trajectory means.",
            "All native starts retained. TSP/ATSP execute native augmentation budgets but serialize original orientation0 used by the method.",
            "Probabilities and alternative-start histories are observed teacher computation, not prefixes fabricated from the best final tour.",
            "States recoverable from input and full routes are not new independent information. Steps are not independent instances.",
            "One initial CVRP observer call failed on missing POMO_IDX. It is preserved; only uncompleted B was explicitly recovered.",
            "Recovery derives column-index metadata without writing native state; physical start nodes remain actual recorded actions.",
            "The CPU observer bound rose384MiB->1GiB before recovery; inputs, weights, solver budgets and original labels did not change.",
            f"Native cost close to frozen TRAIN: {close_count}/144; fully matching task samples: {aligned_tasks}/18. MATNET: {atsp['cost_close']}/8.",
            "Native MATNET role inputs were recorded, but the cause of its cost difference is not isolated; no label-error claim.",
            "No selector trained, checkpoint selected, TEST/LIB evaluated or zero-selection improvement claimed by this asset release.",
            "No cheap sufficient state, learnability, Markov compression or decision-transfer benefit has been established.",
            "R58 semantics remain deferred. Current legal pools and single reld_mt remain unchanged.",
            "The long-running whole18 improvement goal remains active. New NN training is not automatically approved by capture completion.",
        ]
        for i, text in enumerate(lines):
            fig.text(.05, .90 - .058 * i, text, fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)
    total = sum(x["compressed_bytes"] for x in cases)
    (ROOT / "RESULTS.md").write_text(
        "# R118: TRAIN native computation assets\n\n"
        "## What completed?\n\n"
        f"144 real captures across all18 problems;18/89 deployments, not all-candidate process coverage. "
        f"Compressed trace bytes: {total}. No selector was trained and no labels/pools changed.\n\n"
        "## What failed and was fixed?\n\n"
        "Original B stopped at the first CVRP call because its native inference environment lacks POMO_IDX. "
        "The original failed call/receipt/log and completed A are preserved. Explicit observer-only recovery derives "
        "trajectory column indices without modifying the environment; actual physical starts remain recorded separately. "
        "Its predeclared observer memory bound is1GiB instead of384MiB. No completed A case was rerun.\n\n"
        "## What can be concluded?\n\n"
        f"{close_count}/144 captured costs are close to the original TRAIN scalar under the fixed5e-5+2e-6*abs(cost) tolerance; "
        f"{aligned_tasks}/18 tasks match in all eight sampled rows. ATSP/MATNET is{atsp['cost_close']}/8, with maximum "
        f"absolute difference{atsp['max_absolute_cost_difference']:.9f}. Its native initial roles were recorded in all eight cases, "
        "but this comparison does not isolate RNG from all other lifecycle/numeric effects. Do not claim that these fresh traces "
        "recover the executions that produced the frozen MATNET labels, or use them as an already aligned whole18 process target. "
        "No global accuracy ceiling or label-error diagnosis follows from eight TRAIN samples.\n\n"
        "Real node-resolved decoder competition, all original-orientation trajectories and resources can be collected "
        "from these locked recipes. This is not evidence that a student can predict them, compress them cheaply, "
        "or improve Top1/regret. Final-route-recoverable states are not new independent labels. "
        "cost_close is a small TRAIN reproduction observation, not historical execution identity or a label repair.\n\n"
        "Per-task details: coverage.csv; per-instance TRAIN costs and trace hashes: cases.csv. "
        "RESULTS.pdf is the main report. R58 remains deferred; whole18 improvement goal is not complete.\n")
    complete = dict(complete=True, artifact_kind="TRAIN process assets, not a selector", observations=len(cases),
        problems=18, process_deployments=18, legal_deployments=89, failed_initial_calls=1,
        nonterminal_probability_rows=sum(x["nonterminal_probability_rows"] for x in cases),
        compressed_bytes=total, selector_training=False, heldout_evaluation=False,
        original_labels_changed=False, goal_complete=False, completed_unix=time.time(), inputs=inputs,
        original_train_cost_close=close_count, tasks_with_all_sample_costs_close=aligned_tasks,
        full18_original_label_process_alignment=False,
        outputs={name:data.common.file_hash(ROOT / name) for name in ("RESULTS.md","RESULTS.pdf","cases.csv","coverage.csv")})
    data.save(ROOT / "complete.json", complete)
    return complete


def publish():
    complete = data.common.read_json(ROOT / "complete.json")
    if complete.get("complete") is not True or complete.get("observations") != 144:
        raise ValueError("Require complete18-problem TRAIN asset evidence, not partial status")
    receipt_path = ROOT / "publish_receipt.json"
    with publication_mutex():
        if receipt_path.exists():
            raise RuntimeError("Existing publication attempt; inspect, do not automatically retry")
        if git("diff", "--cached", "--name-only", "-z"):
            raise RuntimeError("Unrelated staged changes; index untouched")
        for name, expected in complete["outputs"].items():
            if data.common.file_hash(ROOT / name) != expected:
                raise ValueError("Completed report changed")
        paths = [data.ROOT.parent.parent / name for name in CODE]
        paths += [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".csv", ".pdf", ".log")]
        paths.append(GIT / "0selection2/docs/experiment_planning/native_trace_tsp_atsp_20261027.json")
        paths = [p for p in dict.fromkeys(paths) if p.name not in ("publish_receipt.json", "publish_manifest.json")]
        if any(p.is_symlink() or p.stat().st_size > 8 * 2**20 for p in paths):
            raise ValueError("Unexpected publication file")
        relative = [str(p.relative_to(GIT)) for p in paths]
        branch = git("symbolic-ref", "--short", "HEAD")
        remote = git("config", "--get", "branch." + branch + ".remote")
        merge = git("config", "--get", "branch." + branch + ".merge")
        parent = git("rev-parse", "HEAD")
        if git("ls-remote", remote, merge).split()[0] != parent:
            raise RuntimeError("Local/remote differ; no force push")
        manifest = dict(parent=parent, data_assets_outside_git=True, goal_complete=False,
            files={str(p.relative_to(GIT)):data.common.file_hash(p) for p in paths})
        data.save(ROOT / "publish_manifest.json", manifest)
        relative.append(str((ROOT / "publish_manifest.json").relative_to(GIT)))
        receipt = dict(state="prepared", parent=parent, commit_attempts=0, push_attempts=0, paths=relative)
        data.save(receipt_path, receipt)
        try:
            git("add", "-f", "--", *relative)
            receipt.update(state="commit_started", commit_attempts=1)
            data.save(receipt_path, receipt)
            git("commit", "--only", "-m", "Add R118 realized TRAIN computation assets and explicit observer recovery", "--", *relative)
            receipt.update(state="push_started", commit=git("rev-parse", "HEAD"), push_attempts=1)
            data.save(receipt_path, receipt)
            git("push", "--porcelain", remote, receipt["commit"] + ":" + merge)
            if git("ls-remote", remote, merge).split()[0] != receipt["commit"]:
                raise RuntimeError("Remote publication not verified")
            receipt.update(state="complete", exit_status=0)
            data.save(receipt_path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(state="failed", error=str(error))
            data.save(receipt_path, receipt)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "publish"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.command]()))
