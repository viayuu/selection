"""Derive a release retaining MOEL as reld_mt, without combining solver costs."""

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS, S2I
from .export_new_split_data import derive_targets, file_hash, train_baselines, utc, write_json, write_pickle
from .new_split_data import scenario_solver_ids
from .r75_data import _CPUUnpickler
from .r75_new_split_trial import tv_pct, write_csv


RETAINED = "RELD_MOEL"
REMOVED = "RELD_MTL"
ALIAS = "reld_mt"
SPLITS = ("train", "val", "test")


def single_reld_pool(source_methods):
    if REMOVED in source_methods and RETAINED not in source_methods:
        raise ValueError("Cannot replace MTL with an unavailable MOEL deployment")
    kept = [slot for slot, name in enumerate(source_methods) if name != REMOVED]
    methods = [ALIAS if source_methods[slot] == RETAINED else source_methods[slot] for slot in kept]
    ids = [S2I[source_methods[slot]] for slot in kept]
    if len(set(methods)) != len(methods) or ids != sorted(ids):
        raise ValueError("Candidate identities must retain their stable source order")
    return methods, kept, ids


def copy_input(source, destination):
    # Reflinks are independent files; unsupported filesystems get a normal copy.
    subprocess.run(["cp", "--reflink=auto", "--preserve=timestamps", str(source), str(destination)], check=True)


def export_split(source, destination, problem, split, methods, kept, ids, baseline, scenario_id):
    metadata = json.loads((source / "metadata.json").read_text())
    with np.load(source / "targets.npz", allow_pickle=False) as old:
        old_costs, old_winner = old["costs"], old["winner"]
        targets = derive_targets(old_costs[:, kept])
        previous_oracle = old["oracle_cost"]
    costs, winners = targets["costs"], targets["winner"]
    if (len(costs) != metadata["instances"] or old_costs.shape[1] != len(metadata["methods"])
            or np.any(targets["oracle_cost"] < previous_oracle)):
        raise ValueError("Parent costs, row count or retained-pool Oracle disagree")
    with (source / "raw_label.pkl").open("rb") as stream:
        original_labels = _CPUUnpickler(stream).load()
    times = np.asarray([original_labels[str(i)]["time"] for i in range(len(costs))], dtype=np.float64)[:, kept]
    if times.shape != costs.shape or not np.isfinite(times).all():
        raise ValueError("Compatibility time columns are not aligned")
    if any(not np.array_equal(original_labels[str(i)]["cost"], old_costs[i]) for i in range(len(costs))):
        raise ValueError("Parent raw labels and target costs disagree")
    del original_labels
    destination.mkdir()
    (destination / "results").mkdir()
    for name in ("dataset.pkl", "input_metadata.npz", "index_mapping.csv"):
        copy_input(source / name, destination / name)
    raw = {}
    for row in range(len(costs)):
        raw[str(row)] = dict(cost=costs[row].tolist(), time=times[row].tolist(), ind=int(winners[row]),
                            gap=targets["gap_pct"][row].tolist(), rank=targets["rank"][row].tolist(),
                            ranking=targets["ranking"][row].tolist(), oracle_cost=float(targets["oracle_cost"][row]),
                            relative_regret=targets["relative_regret"][row].tolist())
    write_pickle(destination / "raw_label.pkl", raw)
    del raw
    targets["sbs_train_cost"] = costs[:, baseline["sbs_slot"]]
    np.savez_compressed(destination / "targets.npz", **targets)
    for slot, name in enumerate(methods):
        with (destination / "results" / ("result_" + name + ".txt")).open("w") as stream:
            for row in range(len(costs)):
                stream.write("{},{:.17g},{:.17g}\n".format(row, costs[row, slot], times[row, slot]))
    counts = np.bincount(winners, minlength=len(methods))
    split_sbs = int(costs.mean(axis=0).argmin())
    summary = dict(
        problem=problem, split=split, instances=len(costs), candidates=len(methods),
        oracle_mean_cost=float(targets["oracle_cost"].mean()),
        oracle_mean_increase_from_parent=float((targets["oracle_cost"] - previous_oracle).mean()),
        train_sbs_solver=baseline["sbs_solver"],
        train_sbs_mean_cost=float(targets["sbs_train_cost"].mean()),
        train_sbs_actual_regret_pct=float(targets["relative_regret"][:, baseline["sbs_slot"]].mean() * 100),
        posthoc_split_sbs_solver=methods[split_sbs], posthoc_split_sbs_mean_cost=float(costs[:, split_sbs].mean()),
        exact_min_cost_ties=int(((costs == targets["oracle_cost"][:, None]).sum(axis=1) > 1).sum()),
        winner_changed_from_parent_pool=int(np.sum(np.asarray(kept)[winners] != old_winner)))
    updated = dict(metadata)
    updated.update(summary, methods=methods, global_solver_ids=ids, scenario_id=scenario_id,
                   solver_aliases={ALIAS: RETAINED} if ALIAS in methods else {},
                   source_methods=metadata["methods"], retained_source_slots=kept,
                   costs_rule="drop RELD_MTL; retain original RELD_MOEL costs exactly; no best-of-two",
                   source_directory=str(source), source_metadata_sha256=file_hash(source / "metadata.json"))
    updated.pop("winner_changed_from_original_pool", None)
    unchanged = {"dataset.pkl", "input_metadata.npz", "index_mapping.csv"}
    updated["files"] = {p.name: (metadata["files"][p.name] if p.name in unchanged else
                                dict(bytes=p.stat().st_size, sha256=file_hash(p)))
                        for p in destination.iterdir() if p.is_file()}
    updated["result_files"] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                               for p in (destination / "results").iterdir()}
    write_json(destination / "metadata.json", updated)
    return summary, counts


def export(source, output):
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    parent = json.loads((source / "scenario_manifest.json").read_text())
    if parent.get("status") != "complete" or parent.get("format_version") != 1:
        raise ValueError("The parent release must be complete")
    if parent.get("solver_aliases"):
        raise ValueError("Expected the original B-filtered release, not another aliased release")
    scenario_solver_ids(parent)
    if output.exists():
        raise FileExistsError("Preserve existing datasets; choose a new output directory")
    output.mkdir()
    for name in ("scenario_manifest.json", "baselines.json"):
        shutil.copy2(source / name, output / ("parent_" + name))
    for name in ("original_split_manifest.json", "split_indices.npz"):
        shutil.copy2(source / name, output / name)
    names = [ALIAS if name == RETAINED else name for name in parent["global_solvers"] if name != REMOVED]
    identities = {name: S2I[RETAINED if name == ALIAS else name] for name in names}
    manifest = dict(parent)
    manifest.update(
        scenario_id="new_only_seed2_B_filtered_MOEL_as_reld_mt_v1", status="building", created_utc=utc(),
        source_data=str(source), parent_scenario_id=parent["scenario_id"],
        parent_manifest_sha256=file_hash(source / "scenario_manifest.json"),
        global_solvers=names, global_solver_ids=identities, model_global_solvers=GLOBAL_SOLVERS,
        solver_aliases={ALIAS: RETAINED}, retained_reld_deployment=RETAINED,
        removed_methods=sorted(set(parent["removed_methods"]) | {REMOVED}),
        reld_removed_from_all_problem_pools=REMOVED,
        cost_combination="none; keep the chosen deployment's unmodified costs",
        reserved_inactive_global_solver_ids=[S2I[REMOVED]],
        instance_splits_changed=False, inputs_modified=False, original_files_modified=False,
        solver_runs=0, fresh_training_required=True, problems={})
    for key in ("completed_utc", "total_winner_changes", "total_problem_solver_combinations", "total_split_directories"):
        manifest.pop(key, None)
    scenario_solver_ids(manifest)
    write_json(output / "scenario_manifest.json", manifest)
    write_json(output / "export_command.json", dict(
        command=[sys.executable, "-m", "code.V4.export_single_reld_data", "--source-dir", str(source),
                 "--output-dir", str(output)], created_utc=utc(), code_sha256=file_hash(Path(__file__))))
    stats, winner_rows, similarities, baselines = [], [], [], {}
    for problem in PROBLEMS:
        original = parent["problems"][problem]
        methods, kept, ids = single_reld_pool(original["methods"])
        with np.load(source / (problem + "train") / "targets.npz", allow_pickle=False) as old:
            baseline = train_baselines(old["costs"][:, kept], methods)
        baselines[problem] = baseline
        fractions = {}
        for split in SPLITS:
            directory = original["splits"][split]
            summary, counts = export_split(source / directory, output / directory, problem, split,
                                           methods, kept, ids, baseline, manifest["scenario_id"])
            stats.append(summary)
            fractions[split] = counts / summary["instances"]
            for slot, name in enumerate(methods):
                winner_rows.append(dict(problem=problem, split=split, solver=name,
                                        instances=summary["instances"], winner_count=int(counts[slot]),
                                        winner_pct=float(fractions[split][slot] * 100)))
            print("EXPORTED {}{} rows={} candidates={} changed_winner={}".format(
                problem, split, summary["instances"], len(methods), summary["winner_changed_from_parent_pool"]), flush=True)
            gc.collect()
        similarities.append(dict(problem=problem,
                                 train_validation_tv_pct=tv_pct(fractions["train"], fractions["val"]),
                                 train_test_tv_pct=tv_pct(fractions["train"], fractions["test"]),
                                 validation_test_tv_pct=tv_pct(fractions["val"], fractions["test"])))
        manifest["problems"][problem] = dict(original, methods=methods, global_solver_ids=ids,
                                              source_methods=original["methods"], retained_source_slots=kept,
                                              removed_methods=[name for name in original["methods"] if name == REMOVED],
                                              solver_aliases={ALIAS: RETAINED} if ALIAS in methods else {},
                                              train_sbs=baseline)
        write_json(output / "scenario_manifest.json", manifest)
    write_json(output / "baselines.json", baselines)
    write_csv(output / "split_statistics.csv", stats)
    write_csv(output / "winner_distribution.csv", winner_rows)
    write_csv(output / "distribution_similarity.csv", similarities)
    manifest.update(status="complete", completed_utc=utc(), total_split_directories=len(stats),
                    total_winner_changes=sum(row["winner_changed_from_parent_pool"] for row in stats),
                    total_problem_solver_combinations=sum(len(record["methods"]) for record in manifest["problems"].values()))
    write_json(output / "scenario_manifest.json", manifest)
    write_json(output / "readiness.json", dict(status="ready", data_root=str(output),
        split_sizes_per_problem=manifest["split_sizes_per_problem"], total_instances=manifest["total_instances"],
        retained_reld_deployment=RETAINED, displayed_name=ALIAS, solvers_run=0,
        labels_recomputed=True, train_sbs_recomputed=True, source_split_indices_unchanged=True,
        loader="code.V4.new_split_data.load_dataset", fresh_training_required=True))
    lines = ["# Single-ReLD Dataset Release", "",
        "Parent: " + str(source), "",
        "RELD_MOEL is retained under the display name reld_mt. RELD_MTL is removed everywhere. "
        "No costs are averaged, minimized across deployments, or newly solved.", "",
        "All 18 tasks retain exactly the parent's row order and split indices: "
        "70,114 train / 3,895 validation / 3,895 test per task; 1,402,272 rows total.", "",
        "The original B-task RF/MoSES exclusion is unchanged. B tasks now have three candidates; "
        "the other seven MVRP tasks have six. TSP, CVRP and ATSP pools are unchanged.", "",
        "Winner, ordinal ranks, Oracle, relative gap and SBS are rebuilt from retained FP64 columns. "
        "SBS is selected by TRAIN mean raw cost only and fixed for validation/test. "
        "Post-hoc split SBS is reported separately, not used as a deployment rule.", "",
        "## Identity and Loading", "",
        "reld_mt resolves to the original MOEL global ID 16. ID 17 (MTL) is inactive; the historical "
        "22-slot model ID space stays compatible without exposing MTL as a legal candidate. "
        "Candidate column order follows metadata/global IDs, NOT alphabetically sorted result filenames.", "",
        "Use code.V4.new_split_data.load_dataset with this explicit data_root. "
        "Do not reuse old R75/new-only caches or loaders that infer columns from sorted filenames. "
        "TEST loading requires the training run's final_test_lock.json as before.", "",
        "```python", "from code.V4.new_split_data import load_dataset, build_batch",
        "dataset = load_dataset('VRPB', 'train', data_root=" + repr(str(output)) + ")",
        "batch = build_batch(dataset, [0, 1], device='cpu')", "```", "",
        "## Files and Boundaries", "",
        "Each split has dataset.pkl, raw_label.pkl, targets.npz, input_metadata.npz, index_mapping.csv, "
        "metadata.json and result_<solver>.txt. The unchanged inputs are independent copies/reflinks, not hard links. "
        "Parent manifests and split_indices.npz preserve provenance. Time columns are source placeholders, "
        "not measured runtimes; gap means retained-pool Oracle gap, not global-optimum gap.", "",
        "The alias does not denote an ensemble or a new architecture. Candidate removal does not prove "
        "constraint equivalence among the remaining methods. Train a fresh model on this scenario, "
        "select using validation only, then evaluate one locked checkpoint on test. Scores are not "
        "directly comparable to older pools or splits.", "",
        "## Candidate Pools", "", "| Problem | Candidates |", "|---|---|"]
    lines.extend("| {} | {} |".format(problem, ", ".join(record["methods"]))
                 for problem, record in manifest["problems"].items())
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "export_code").mkdir()
    for name in ("export_single_reld_data.py", "new_split_data.py", "export_new_split_data.py", "test_single_reld_data.py"):
        shutil.copy2(Path(__file__).parent / name, output / "export_code" / name)
    print(json.dumps(dict(status="complete", data_root=str(output), rows=manifest["total_instances"],
                          problem_solver_combinations=manifest["total_problem_solver_combinations"],
                          winner_changes=manifest["total_winner_changes"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
