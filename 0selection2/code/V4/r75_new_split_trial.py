"""Split only fresh, fully labelled instances without using their winners."""

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .registry import PROBLEMS


SPLITS = ("train", "validation", "test")


def proportional_counts(capacities, total, rng):
    capacities = np.asarray(capacities, dtype=np.int64)
    available = int(capacities.sum())
    if total < 0 or total > available or available == 0:
        raise ValueError("Requested allocation exceeds the available instances")
    scaled = capacities * total
    counts = scaled // available
    remainder = total - int(counts.sum())
    order = np.lexsort((rng.random(len(capacities)), -(scaled % available)))
    counts[order[:remainder]] += 1
    if int(counts.sum()) != total or np.any(counts > capacities):
        raise ValueError("Invalid proportional allocation")
    return counts


def new_strata(metadata, old_prefix):
    groups = defaultdict(list)
    stop = int(metadata["instances"])
    cursor = old_prefix
    for shard in metadata["included_new_shards"]:
        begin = int(shard["start_index"])
        if begin >= stop:
            break
        end = min(stop, begin + int(shard["count"]))
        if begin != cursor or end <= begin:
            raise ValueError("New-instance metadata is not a contiguous prefix")
        key = (int(shard["scale"]), shard["distribution"], shard["capacity_type"])
        groups[key].append(np.arange(begin, end, dtype=np.int32))
        cursor = end
    if cursor != stop:
        raise ValueError("Missing generation metadata for new instances")
    return {key: np.concatenate(groups[key]) for key in sorted(groups)}


def make_split(strata, sizes, rng):
    keys = list(strata)
    available = np.asarray([len(strata[key]) for key in keys])
    selected = proportional_counts(available, sum(sizes.values()), rng)
    validation = proportional_counts(selected, sizes["validation"], rng)
    test = proportional_counts(selected - validation, sizes["test"], rng)
    train = selected - validation - test
    counts = dict(train=train, validation=validation, test=test)
    pieces = {split: [] for split in SPLITS}
    for column, key in enumerate(keys):
        indices = rng.permutation(strata[key])[:selected[column]]
        nv, nt = int(validation[column]), int(test[column])
        pieces["validation"].append(indices[:nv])
        pieces["test"].append(indices[nv:nv + nt])
        pieces["train"].append(indices[nv + nt:])
    indices = {
        split: rng.permutation(np.concatenate(pieces[split])).astype(np.int32)
        for split in SPLITS
    }
    combined = np.concatenate(list(indices.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Splits overlap")
    for split in SPLITS:
        if len(indices[split]) != sizes[split]:
            raise ValueError("Incorrect split size")
    return indices, selected, counts


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tv_pct(left, right):
    return float(np.abs(left - right).sum() * 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent /
                        "runs/R75_large_complete_labels/new_only_split_trial_seed2")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--old-prefix", type=int, default=10000)
    parser.add_argument("--train", type=int, default=70114)
    parser.add_argument("--validation", type=int, default=3895)
    parser.add_argument("--test", type=int, default=3895)
    args = parser.parse_args()
    sizes = {split: getattr(args, split) for split in SPLITS}
    if min(sizes.values()) <= 0:
        raise ValueError("All splits must be nonempty")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    winner_rows, summary_rows, stratum_rows, saved_indices = [], [], [], {}
    records = {}
    for problem_id, problem in enumerate(PROBLEMS):
        source = args.data_root / (problem + "train")
        metadata_path = source / "\u5b50\u96c6\u4fe1\u606f.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["original_instances"] != args.old_prefix:
            raise ValueError("Unexpected old-prefix size: " + problem)
        if metadata["index_mapping"] != "local_index == original_global_index":
            raise ValueError("Unsupported source index mapping: " + problem)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, problem_id]))
        strata = new_strata(metadata, args.old_prefix)
        # Lock input-only selection before opening any supervision file.
        indices, selected, counts = make_split(strata, sizes, rng)
        if any(int(v.min()) < args.old_prefix for v in indices.values()):
            raise ValueError("An old instance entered a split")
        with (source / "raw_label.pkl").open("rb") as stream:
            labels = pickle.load(stream)
        n = int(metadata["instances"])
        if len(labels) != n:
            raise ValueError("Label count disagrees with frozen metadata: " + problem)
        pool = metadata["methods"]
        winners = np.fromiter((int(labels[str(i)]["ind"]) for i in range(n)),
                              dtype=np.int64, count=n)
        del labels
        if np.any(winners < 0) or np.any(winners >= len(pool)):
            raise ValueError("Winner index is outside the candidate pool")
        split_counts = {
            split: np.bincount(winners[indices[split]], minlength=len(pool))
            for split in SPLITS
        }
        fractions = {split: split_counts[split] / sizes[split] for split in SPLITS}
        combined_counts = sum(split_counts.values())
        for column, solver in enumerate(pool):
            row = dict(problem=problem, solver=solver,
                       selected_pool_winner_pct=float(combined_counts[column] /
                                                      sum(sizes.values()) * 100))
            for split in SPLITS:
                row[split + "_n"] = sizes[split]
                row[split + "_winner_count"] = int(split_counts[split][column])
                row[split + "_winner_pct"] = float(fractions[split][column] * 100)
            winner_rows.append(row)
        summary = dict(problem=problem, new_available_n=n - args.old_prefix,
                       selected_n=sum(sizes.values()),
                       train_validation_tv_pct=tv_pct(fractions["train"], fractions["validation"]),
                       train_test_tv_pct=tv_pct(fractions["train"], fractions["test"]),
                       validation_test_tv_pct=tv_pct(fractions["validation"], fractions["test"]))
        max_gaps = np.maximum.reduce([
            np.abs(fractions[a] - fractions[b])
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ])
        summary["max_solver_gap_pp"] = float(max_gaps.max() * 100)
        summary["max_gap_solver"] = pool[int(max_gaps.argmax())]
        summary_rows.append(summary)
        for column, key in enumerate(strata):
            row = dict(problem=problem, scale=key[0], distribution=key[1],
                       capacity_type=key[2], available_new_n=len(strata[key]),
                       selected_n=int(selected[column]))
            for split in SPLITS:
                row[split + "_n"] = int(counts[split][column])
            stratum_rows.append(row)
        for split, values in indices.items():
            saved_indices[problem + "_" + split] = values
        records[problem] = dict(
            source_metadata=str(metadata_path), source_dataset_sha256=metadata["dataset_sha256"],
            source_label_sha256=metadata["raw_label_sha256"], methods=pool,
            source_instances=n, excluded_old_instances=args.old_prefix,
            eligible_new_instances=n - args.old_prefix, selected_instances=sum(sizes.values()),
            unused_new_instances=n - args.old_prefix - sum(sizes.values()),
            input_strata=len(strata), split_sizes=sizes,
            index_sha256={split: hashlib.sha256(values.tobytes()).hexdigest()
                          for split, values in indices.items()})
        print(json.dumps(summary), flush=True)
    np.savez_compressed(output / "split_indices.npz", **saved_indices)
    write_csv(output / "winner_distribution.csv", winner_rows)
    write_csv(output / "distribution_similarity.csv", summary_rows)
    write_csv(output / "input_strata.csv", stratum_rows)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        source_data=str(args.data_root), split_sizes_per_problem=sizes,
        total_selected=sum(sizes.values()) * len(PROBLEMS),
        index_format="NPZ keys <problem>_<split>; int32 indices into original snapshot train files",
        selection="Proportional input-only allocation; random selection within exact scale/distribution/capacity strata",
        labels_used_for_selection=False, split_seed_search=False,
        native_winner_policy="Use original raw_label ind, including its existing exact-cost tie convention",
        grouping="Fresh generator samples only; no old-prefix instances or augmented copies are selected",
        source_inputs_loaded=False, model_forwards=0, solver_runs=0,
        diagnostic_only=True, original_files_modified=False, problems=records)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# New-Only Split Trial", "", "Seed: {}. {} instances per task, {} in total.".format(
        args.seed, sum(sizes.values()), manifest["total_selected"]), "",
        "Each task has TRAIN={}, VALIDATION={}, TEST={}. The first {} historical instances are excluded.".format(
            args.train, args.validation, args.test, args.old_prefix), "",
        "Selection uses exact input scale, generation distribution and capacity type. "
        "Winners are only opened after the indices are locked; they do not influence sampling. "
        "There was no split-seed search, solver execution, model inference or training.", "",
        "TV (%) is half the sum of absolute winner-frequency differences. "
        "These are descriptive marginal similarities, not proof of identical joint distributions.", "",
        "## Summary", "",
        "Across all 18 problems, the winner marginals are close under this single fixed input-stratified split.", "",
        "| Comparison | Mean TV % | Maximum TV % |",
        "|---|---:|---:|"]
    for label, field in (("Train / Validation", "train_validation_tv_pct"),
                         ("Train / Test", "train_test_tv_pct"),
                         ("Validation / Test", "validation_test_tv_pct")):
        values = [row[field] for row in summary_rows]
        lines.append("| {} | {:.2f} | {:.2f} |".format(label, float(np.mean(values)), max(values)))
    lines += ["", "The largest individual solver-frequency difference across any two splits is {:.2f} percentage points.".format(
        max(row["max_solver_gap_pp"] for row in summary_rows)), "",
        "Aggregate sizes: TRAIN={}, VALIDATION={}, TEST={}.".format(
            *(sizes[split] * len(PROBLEMS) for split in SPLITS)), "",
        "## Per-Problem Similarity", "",
        "| Problem | Train/Val TV % | Train/Test TV % | Val/Test TV % | Max Method Gap pp |",
        "|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
            row["problem"], row["train_validation_tv_pct"], row["train_test_tv_pct"],
            row["validation_test_tv_pct"], row["max_solver_gap_pp"]))
    for problem in PROBLEMS:
        lines += ["", "## " + problem, "",
                  "| Solver | Train % | Validation % | Test % |", "|---|---:|---:|---:|"]
        for row in winner_rows:
            if row["problem"] == problem:
                lines.append("| {} | {:.2f} | {:.2f} | {:.2f} |".format(
                    row["solver"], *(row[split + "_winner_pct"] for split in SPLITS)))
    lines += ["", "## Boundaries", "",
              "This trial preserves the available new-label scenario; it does not repair solver constraint semantics. "
              "It cannot reconstruct missing large-scale TSP/CVRP tail instances. "
              "Any future model using these heldout splits must be trained from scratch, "
              "not resumed from a model that saw these instances.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print("Saved split indices and distribution reports to " + str(output), flush=True)


if __name__ == "__main__":
    main()
