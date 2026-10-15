"""Materialize the existing split, removing RF/MoSES only from B tasks."""

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS, S2I
from .r75_data import _CPUUnpickler
from .r75_new_split_trial import tv_pct, write_csv


B_PROBLEMS = frozenset(("VRPB", "OVRPB", "VRPBL", "VRPBTW", "OVRPBL",
                        "OVRPBTW", "VRPBLTW", "OVRPBLTW"))
REMOVED = frozenset(("RouteFinder", "MoSES_RF", "MoSES_CaDA"))
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))
COORD_DIST = {"uniform": 0, "gaussian_mixture": 1, "clustered": 2, "ring": 3,
              "Gaussian": 4, "matrix_closure": 4}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_pickle(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filtered_pool(problem, source_pool):
    methods = [name for name in source_pool if problem not in B_PROBLEMS or name not in REMOVED]
    if methods != sorted(methods) or any(name not in S2I for name in methods):
        raise ValueError("Candidate order must match stable global solver identities")
    if problem in B_PROBLEMS and len(source_pool) - len(methods) != 3:
        raise ValueError("A B task does not contain exactly the three requested removals")
    return methods, [source_pool.index(name) for name in methods]


def derive_targets(costs):
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or not costs.shape[1] or not np.isfinite(costs).all():
        raise ValueError("Each instance needs a finite complete cost vector")
    ranking = np.argsort(costs, axis=1, kind="stable").astype(np.int32)
    rank = np.empty_like(ranking)
    np.put_along_axis(rank, ranking, np.arange(costs.shape[1], dtype=np.int32)[None, :], axis=1)
    winner = ranking[:, 0].astype(np.int64)
    oracle = costs[np.arange(len(costs)), winner]
    if np.any(oracle <= 0):
        raise ValueError("Relative regret requires strictly positive Oracle costs")
    relative = (costs - oracle[:, None]) / oracle[:, None]
    return dict(costs=costs, winner=winner, ranking=ranking, rank=rank,
                oracle_cost=oracle, relative_regret=relative, gap_pct=relative * 100)


def train_baselines(costs, methods):
    targets = derive_targets(costs)
    mean_cost = costs.mean(axis=0)
    mean_regret = targets["relative_regret"].mean(axis=0)
    winners = np.bincount(targets["winner"], minlength=len(methods))
    sbs = int(mean_cost.argmin())
    return dict(
        sbs_solver=methods[sbs], sbs_slot=sbs, selection_split="train",
        selection_rule="minimum train mean raw cost, independently for each problem",
        mean_cost_by_solver=dict(zip(methods, map(float, mean_cost))),
        mean_relative_regret_by_solver=dict(zip(methods, map(float, mean_regret))),
        minimum_mean_relative_regret_solver=methods[int(mean_regret.argmin())],
        majority_winner_solver=methods[int(winners.argmax())])


def input_metadata(metadata):
    n = int(metadata["instances"])
    scale = np.full(n, -1, dtype=np.int32)
    coord_dist = np.full(n, -1, dtype=np.int8)
    distribution = np.empty(n, dtype="U32")
    capacity_type = np.empty(n, dtype="U32")
    for shard in metadata["included_new_shards"]:
        begin, stop = int(shard["start_index"]), min(n, int(shard["start_index"]) + int(shard["count"]))
        if begin >= n:
            break
        scale[begin:stop] = int(shard["scale"])
        coord_dist[begin:stop] = COORD_DIST[shard["distribution"]]
        distribution[begin:stop] = shard["distribution"]
        capacity_type[begin:stop] = shard["capacity_type"]
    if np.any(scale[10000:] < 0) or np.any(coord_dist[10000:] < 0):
        raise ValueError("Missing source input metadata")
    return dict(scale=scale, coord_dist=coord_dist, distribution=distribution, capacity_type=capacity_type)


def export_split(destination, instances, labels, source_costs, indices, kept, methods,
                 problem, split, inputs, baseline):
    destination.mkdir()
    (destination / "results").mkdir()
    targets = derive_targets(source_costs[indices][:, kept])
    costs, winners = targets["costs"], targets["winner"]
    times = np.asarray([labels[str(int(i))]["time"] for i in indices], dtype=np.float64)[:, kept]
    if times.shape != costs.shape or not np.isfinite(times).all():
        raise ValueError("Invalid compatibility time vectors")
    raw = {}
    for row, source_index in enumerate(indices):
        raw[str(row)] = dict(
            cost=costs[row].tolist(), time=times[row].tolist(), ind=int(winners[row]),
            gap=targets["gap_pct"][row].tolist(), rank=targets["rank"][row].tolist(),
            ranking=targets["ranking"][row].tolist(), oracle_cost=float(targets["oracle_cost"][row]),
            relative_regret=targets["relative_regret"][row].tolist())
    write_pickle(destination / "dataset.pkl", [instances[int(i)] for i in indices])
    write_pickle(destination / "raw_label.pkl", raw)
    del raw
    sbs_slot = baseline["sbs_slot"]
    targets["sbs_train_cost"] = costs[:, sbs_slot]
    np.savez_compressed(destination / "targets.npz", **targets)
    np.savez_compressed(destination / "input_metadata.npz", source_index=indices,
                        **{key: value[indices] for key, value in inputs.items()})
    for slot, solver in enumerate(methods):
        with (destination / "results" / ("result_" + solver + ".txt")).open("w") as stream:
            for row in range(len(indices)):
                stream.write("{},{:.17g},{:.17g}\n".format(row, costs[row, slot], times[row, slot]))
    with (destination / "index_mapping.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("local_index", "source_index", "scale", "distribution", "capacity_type"))
        writer.writerows((row, int(index), int(inputs["scale"][index]),
                          str(inputs["distribution"][index]), str(inputs["capacity_type"][index]))
                         for row, index in enumerate(indices))
    sbs_on_split = int(costs.mean(axis=0).argmin())
    counts = np.bincount(winners, minlength=len(methods))
    summary = dict(
        problem=problem, split=split, instances=len(indices), candidates=len(methods),
        oracle_mean_cost=float(targets["oracle_cost"].mean()),
        train_sbs_solver=baseline["sbs_solver"], train_sbs_mean_cost=float(costs[:, sbs_slot].mean()),
        train_sbs_actual_regret_pct=float(targets["relative_regret"][:, sbs_slot].mean() * 100),
        posthoc_split_sbs_solver=methods[sbs_on_split],
        posthoc_split_sbs_mean_cost=float(costs[:, sbs_on_split].mean()),
        exact_min_cost_ties=int(((costs == targets["oracle_cost"][:, None]).sum(axis=1) > 1).sum()),
        winner_changed_from_original_pool=int(np.sum([
            labels[str(int(i))]["ind"] != kept[int(winners[row])]
            for row, i in enumerate(indices)])))
    metadata = dict(summary, methods=methods, global_solver_ids=[S2I[s] for s in methods],
                    cost_dtype="float64", source_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                    rank_definition="zero-based ordinal rank; ranking gives slots from best to worst",
                    tie_break="lowest global solver ID for exactly equal FP64 costs",
                    gap_definition="percent above retained-candidate Oracle, not a known-optimum gap",
                    time_definition="source compatibility placeholders, not measured runtimes")
    metadata["files"] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                         for p in destination.iterdir() if p.is_file()}
    metadata["result_files"] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                                for p in (destination / "results").iterdir()}
    write_json(destination / "metadata.json", metadata)
    return summary, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path(__file__).parent /
                        "runs/R75_large_complete_labels/new_only_split_trial_seed2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split_manifest_path = args.split_dir / "split_manifest.json"
    original = json.loads(split_manifest_path.read_text())
    source = Path(original["source_data"])
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Preserve an existing release; choose a new output directory")
    output.mkdir()
    shutil.copy2(split_manifest_path, output / "original_split_manifest.json")
    shutil.copy2(args.split_dir / "split_indices.npz", output / "split_indices.npz")
    write_json(output / "export_command.json", dict(command=[sys.executable, "-m", "V4.export_new_split_data", *sys.argv[1:]],
               code_sha256=file_hash(Path(__file__)), created_utc=utc()))
    manifest = dict(
        format_version=1, scenario_id="new_only_seed2_B_without_RF_MoSES_v1", status="building",
        created_utc=utc(), seed=original["seed"], source_data=str(source),
        split_sizes_per_problem={disk: original["split_sizes_per_problem"][trial] for trial, disk in SPLITS},
        total_instances=original["total_selected"], global_solvers=GLOBAL_SOLVERS,
        global_solver_ids=dict(S2I), filtered_problems=sorted(B_PROBLEMS), removed_methods=sorted(REMOVED),
        instance_splits_changed=False, inputs_modified=False, solver_runs=0,
        original_files_modified=False, fresh_training_required=True,
        remaining_semantic_limits="Removing RF/MoSES in B tasks does not certify all other constraint implementations as identical",
        problems={})
    write_json(output / "scenario_manifest.json", manifest)
    stats, winner_rows, similarities, baselines = [], [], [], {}
    with np.load(args.split_dir / "split_indices.npz", allow_pickle=False) as saved:
        for problem in PROBLEMS:
            start_record = original["problems"][problem]
            source_directory = source / (problem + "train")
            metadata = json.loads(Path(start_record["source_metadata"]).read_text())
            methods, kept = filtered_pool(problem, metadata["methods"])
            indices = {trial: saved[problem + "_" + trial] for trial, _ in SPLITS}
            for trial, array in indices.items():
                if (len(array) != original["split_sizes_per_problem"][trial]
                        or int(array.min()) < 10000
                        or hashlib.sha256(array.tobytes()).hexdigest() != start_record["index_sha256"][trial]):
                    raise ValueError("The locked split indices changed")
            combined = np.concatenate(list(indices.values()))
            if len(np.unique(combined)) != len(combined):
                raise ValueError("The locked splits overlap")
            print("LOAD " + problem, flush=True)
            with (source_directory / "dataset.pkl").open("rb") as stream:
                instances = _CPUUnpickler(stream).load()
            with (source_directory / "raw_label.pkl").open("rb") as stream:
                labels = _CPUUnpickler(stream).load()
            if len(instances) != metadata["instances"] or len(labels) != len(instances):
                raise ValueError("Frozen source row counts changed")
            source_costs = np.asarray([labels[str(i)]["cost"] for i in range(len(instances))], dtype=np.float64)
            if source_costs.shape != (len(instances), len(metadata["methods"])):
                raise ValueError("Source cost-column shape changed")
            inputs = input_metadata(metadata)
            baseline = train_baselines(source_costs[indices["train"]][:, kept], methods)
            baselines[problem] = baseline
            fractions = {}
            for trial, disk_split in SPLITS:
                summary, counts = export_split(
                    output / (problem + disk_split), instances, labels, source_costs,
                    indices[trial], kept, methods, problem, disk_split, inputs, baseline)
                stats.append(summary)
                fractions[disk_split] = counts / len(indices[trial])
                for slot, solver in enumerate(methods):
                    winner_rows.append(dict(problem=problem, split=disk_split, solver=solver,
                                            instances=len(indices[trial]), winner_count=int(counts[slot]),
                                            winner_pct=float(fractions[disk_split][slot] * 100)))
                print("EXPORTED {}{} rows={} candidates={}".format(problem, disk_split, len(indices[trial]), len(methods)), flush=True)
            similarities.append(dict(problem=problem,
                train_validation_tv_pct=tv_pct(fractions["train"], fractions["val"]),
                train_test_tv_pct=tv_pct(fractions["train"], fractions["test"]),
                validation_test_tv_pct=tv_pct(fractions["val"], fractions["test"])))
            manifest["problems"][problem] = dict(
                methods=methods, global_solver_ids=[S2I[s] for s in methods],
                old_methods=metadata["methods"], retained_source_slots=kept,
                removed_methods=[s for s in metadata["methods"] if s not in methods],
                source_dataset_sha256=metadata["dataset_sha256"], source_labels_sha256=metadata["raw_label_sha256"],
                train_sbs=baseline, splits={disk: problem + disk for _, disk in SPLITS})
            write_json(output / "scenario_manifest.json", manifest)
            del instances, labels, source_costs, inputs
            gc.collect()
    write_json(output / "baselines.json", baselines)
    write_csv(output / "split_statistics.csv", stats)
    write_csv(output / "winner_distribution.csv", winner_rows)
    write_csv(output / "distribution_similarity.csv", similarities)
    manifest.update(status="complete", completed_utc=utc(),
                    total_problem_solver_combinations=sum(len(v["methods"]) for v in manifest["problems"].values()),
                    total_split_directories=len(stats), total_winner_changes=sum(r["winner_changed_from_original_pool"] for r in stats))
    write_json(output / "scenario_manifest.json", manifest)
    lines = ["# New-Only B-Filtered Dataset", "",
        "A materialized, independent dataset release. No original files were overwritten, "
        "no instances were regenerated and no solvers were run.", "",
        "All 18 tasks: train=70,114, val=3,895, test=3,895 per task. Total=1,402,272 instances.", "",
        "The previously locked seed=2 split indices and their order are unchanged. "
        "The first 10,000 old instances in every source task are excluded.", "",
        "Only the eight B tasks remove RouteFinder, MoSES_RF and MoSES_CaDA. "
        "Their remaining ordered pool is MTPOMO / MVMOE / RELD_MOEL / RELD_MTL. "
        "All other tasks retain their original pools. Global solver IDs are unchanged.", "",
        "## File Contract", "",
        "Each <problem>train / <problem>val / <problem>test directory contains:", "",
        "- dataset.pkl: original instance objects, selected without transforming inputs.",
        "- raw_label.pkl: contiguous string IDs; cost, time, ind, gap, rank, ranking, oracle_cost, relative_regret.",
        "- results/result_<solver>.txt: local ID, retained FP64 cost, source time placeholder.",
        "- targets.npz: FP64 costs, winner, rank, ranking, Oracle, relative regret, gap percent, train-SBS per-instance cost.",
        "- input_metadata.npz and index_mapping.csv: source indices, true scale, actual distribution and capacity type.",
        "- metadata.json: candidate order, schema definitions, statistics and file hashes.", "",
        "Ranks are zero-based ordinal ranks. Exactly tied FP64 costs use the lowest global solver ID. "
        "gap is relative to this candidate pool's Oracle in percent, NOT a gap to a globally optimal route. "
        "time retains compatibility placeholders and is NOT a measured runtime.", "",
        "## Baselines", "",
        "baselines.json selects one SBS per problem using TRAIN mean raw cost only. "
        "That solver is fixed for validation and test. split_statistics.csv also reports each split's "
        "post-hoc best fixed method separately; do not use its test identity to choose a deployed policy.", "",
        "## Loading", "",
        "Use code.V4.new_split_data.load_dataset and build_batch. "
        "The historical R75 loader/cache assumes seven candidates in B tasks and must not be reused unchanged. "
        "This loader uses the manifest's name-aligned pools and the actual per-row distribution metadata. "
        "TEST loading still requires the training run's final_test_lock.json; do not create a lock before checkpoint selection.", "",
        "```python",
        "from code.V4.new_split_data import load_dataset, build_batch",
        "dataset = load_dataset('VRPB', split='train', data_root=" + repr(str(output)) + ")",
        "batch = build_batch(dataset, [0, 1], device='cpu')",
        "```", "",
        "## Boundaries", "",
        "Train a fresh model; do not resume a model that previously saw these validation/test instances. "
        "Keep validation-only selection and one locked-checkpoint final test. "
        "This candidate restriction does not prove complete constraint equivalence among the remaining solvers. "
        "Scores from this changed scenario are not directly comparable to the old test benchmark.", "",
        "## Pools", "", "| Problem | Candidates |", "|---|---|"]
    for problem, record in manifest["problems"].items():
        lines.append("| {} | {} |".format(problem, ", ".join(record["methods"])))
    lines += ["", "## Winner Marginal Similarity", "",
        "| Problem | Train/Val TV % | Train/Test TV % | Val/Test TV % |", "|---|---:|---:|---:|"]
    for row in similarities:
        lines.append("| {} | {:.2f} | {:.2f} | {:.2f} |".format(row["problem"], row["train_validation_tv_pct"],
                     row["train_test_tv_pct"], row["validation_test_tv_pct"]))
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="complete", data_root=str(output), rows=manifest["total_instances"],
                          problem_solver_combinations=manifest["total_problem_solver_combinations"],
                          winner_changes=manifest["total_winner_changes"])), flush=True)


if __name__ == "__main__":
    main()
