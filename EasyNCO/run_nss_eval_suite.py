import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime


RELEVANT_PROBLEMS = {"tsp", "cvrp"}
GRAPH_SPLIT_TSP_METHODS = {"difusco", "t2t"}
LIB_DATASET_NAMES = {"TSPLIB", "CVRPLIB"}


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_base_runs(results_root):
    discovered = {}
    for entry in sorted(os.listdir(results_root)):
        full_dir = os.path.join(results_root, entry)
        if not os.path.isdir(full_dir):
            continue
        if "_" not in entry:
            continue
        method, problem = entry.rsplit("_", 1)
        if problem not in RELEVANT_PROBLEMS:
            continue
        hydra_runs = []
        for subdir in sorted(os.listdir(full_dir)):
            override_path = os.path.join(full_dir, subdir, ".hydra", "overrides.yaml")
            if os.path.isfile(override_path):
                hydra_runs.append(override_path)
        if hydra_runs:
            discovered[(method, problem)] = hydra_runs[-1]
    return discovered


def load_override_list(path):
    overrides = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                overrides.append(line[2:])
    return overrides


def override_key(override):
    return override.split("=", 1)[0]


def upsert_override(overrides, new_override):
    key = override_key(new_override)
    for idx, old in enumerate(overrides):
        if override_key(old) == key:
            overrides[idx] = new_override
            return
    overrides.append(new_override)


def ensure_removed(overrides, key):
    return [item for item in overrides if override_key(item) != key]


def count_jsonl_lines(path):
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def clear_partial_outputs(*paths):
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


def load_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset_jobs(manifest):
    jobs = []
    for dataset_name, meta in manifest.items():
        jobs.append(
            {
                "dataset_name": dataset_name,
                "problem": meta["problem"],
                "rel_path": meta["relative_output_path"],
                "num_instances": meta["num_instances"],
                "scale_range": [meta["min_scale"], meta["max_scale"] + 1],
                "fixed_scale_files": meta.get("fixed_scale_files", []),
            }
        )
    return jobs


def is_lib_job(job):
    return job["dataset_name"] in LIB_DATASET_NAMES


def load_existing_status_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_latest_status_map(rows):
    latest = {}
    for row in rows:
        latest[(row["方法"], row["问题"], row["数据集"])] = row
    return latest


def write_status_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "时间",
        "方法",
        "问题",
        "数据集",
        "实例数",
        "状态",
        "结果目录",
        "逐实例结果条数",
        "返回码",
        "备注",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_base_overrides(override_path, cuda):
    overrides = load_override_list(override_path)
    for key in [
        "test_data_path",
        "batch_size",
        "episodes",
        "cuda",
        "dir",
        "varying_data_params.scale_range",
        "varying_data_params.distribution_list",
        "varying_data_params.capacity_range",
    ]:
        overrides = ensure_removed(overrides, key)
    upsert_override(overrides, "batch_size=1")
    upsert_override(overrides, f"cuda={cuda}")
    return overrides


def run_command(command, cwd, log_path=None):
    print("running:", " ".join(command))
    if log_path is None:
        return subprocess.run(command, cwd=cwd)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(command))
        f.write("\n\n")
        f.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )


def select_graph_split_files(fixed_scale_files, instance_count):
    flattened = []
    for scale_entry in fixed_scale_files:
        for idx, global_index in enumerate(scale_entry["global_indices"]):
            flattened.append(
                {
                    "scale": int(scale_entry["scale"]),
                    "relative_path": scale_entry["relative_path"],
                    "global_index": int(global_index),
                    "name": scale_entry["names"][idx],
                    "position": idx,
                }
            )

    flattened.sort(key=lambda item: item["global_index"])
    if instance_count > 0:
        flattened = flattened[:instance_count]

    grouped = defaultdict(
        lambda: {
            "scale": None,
            "relative_path": None,
            "num_instances": 0,
            "global_indices": [],
            "names": [],
            "positions": [],
        }
    )
    for item in flattened:
        key = (item["scale"], item["relative_path"])
        group = grouped[key]
        group["scale"] = item["scale"]
        group["relative_path"] = item["relative_path"]
        group["global_indices"].append(item["global_index"])
        group["names"].append(item["name"])
        group["positions"].append(item["position"])

    selected = []
    for key in sorted(grouped):
        group = grouped[key]
        expected_positions = list(range(len(group["positions"])))
        if group["positions"] != expected_positions:
            raise ValueError(
                f"Selected graph split items are not a prefix of the source scale file: "
                f"scale={group['scale']}, positions={group['positions']}"
            )
        group["num_instances"] = len(group["global_indices"])
        selected.append(group)
    return selected


def run_standard_job(args, overrides, job, run_dir, instance_count):
    instance_jsonl = os.path.join(run_dir, "instance_results.jsonl")
    existing_lines = count_jsonl_lines(instance_jsonl)
    if existing_lines == instance_count and instance_count > 0:
        return True, 0, existing_lines, "instance_results.jsonl 已齐全"
    if 0 < existing_lines < instance_count:
        clear_partial_outputs(
            instance_jsonl,
            os.path.join(run_dir, "eval.log"),
            os.path.join(run_dir, "汇总.json"),
        )

    job_overrides = list(overrides)
    upsert_override(job_overrides, f"episodes={instance_count}")
    upsert_override(job_overrides, f"test_data_path={job['rel_path']}")
    upsert_override(
        job_overrides,
        f"varying_data_params.scale_range=[{job['scale_range'][0]},{job['scale_range'][1]}]",
    )
    upsert_override(job_overrides, "varying_data_params.distribution_list=null")
    if job["problem"] == "cvrp":
        upsert_override(job_overrides, "varying_data_params.capacity_range=null")
    upsert_override(job_overrides, f"dir={run_dir}")

    completed = run_command(
        [sys.executable, "eval.py"] + job_overrides,
        args.easy_root,
        log_path=os.path.join(run_dir, "eval.log"),
    )
    final_lines = count_jsonl_lines(instance_jsonl)
    ok = completed.returncode == 0 and final_lines == instance_count
    return ok, completed.returncode, final_lines, "" if ok else "请查看对应结果目录下 eval.log"


def run_graph_split_tsp_job(args, overrides, job, run_dir, instance_count):
    parent_jsonl = os.path.join(run_dir, "instance_results.jsonl")
    existing_lines = count_jsonl_lines(parent_jsonl)
    if existing_lines == instance_count and instance_count > 0:
        return True, 0, existing_lines, "instance_results.jsonl 已齐全"

    selected_scale_files = select_graph_split_files(job["fixed_scale_files"], instance_count)
    all_records = []
    for scale_entry in selected_scale_files:
        child_dir = os.path.join(run_dir, f"scale_{scale_entry['scale']}")
        child_jsonl = os.path.join(child_dir, "instance_results.jsonl")
        child_count = scale_entry["num_instances"]
        child_lines = count_jsonl_lines(child_jsonl)
        if 0 < child_lines < child_count:
            clear_partial_outputs(
                child_jsonl,
                os.path.join(child_dir, "eval.log"),
            )

        if child_lines != child_count:
            child_overrides = list(overrides)
            upsert_override(child_overrides, f"episodes={child_count}")
            upsert_override(child_overrides, f"test_data_path={scale_entry['relative_path']}")
            upsert_override(child_overrides, f"dir={child_dir}")
            completed = run_command(
                [sys.executable, "eval.py"] + child_overrides,
                args.easy_root,
                log_path=os.path.join(child_dir, "eval.log"),
            )
            child_lines = count_jsonl_lines(child_jsonl)
            if completed.returncode != 0 or child_lines != child_count:
                return False, completed.returncode, count_jsonl_lines(parent_jsonl), "图数据按规模子任务失败，请查看子目录 eval.log"

        child_records = load_jsonl(child_jsonl)
        if len(child_records) != child_count:
            return False, 1, count_jsonl_lines(parent_jsonl), "图数据按规模子任务结果条数不匹配"

        for idx, record in enumerate(child_records):
            record["global_index"] = int(scale_entry["global_indices"][idx])
            record["name"] = scale_entry["names"][idx]
            all_records.append(record)

    all_records.sort(key=lambda item: item["global_index"])
    write_jsonl(parent_jsonl, all_records)

    summary_path = os.path.join(run_dir, "汇总.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        summary = {
            "实例数": len(all_records),
            "aug_score均值": sum(item["aug_score"] for item in all_records) / len(all_records),
            "按规模子任务数": len(selected_scale_files),
        }
        if all("no_aug_score" in item for item in all_records):
            summary["no_aug_score均值"] = (
                sum(item["no_aug_score"] for item in all_records) / len(all_records)
            )
        json.dump(summary, f, ensure_ascii=False, indent=2)
    ok = len(all_records) == instance_count
    return ok, 0 if ok else 1, len(all_records), "" if ok else "聚合后的逐实例结果条数不匹配"


def has_started_outputs(job, run_dir):
    parent_jsonl = os.path.join(run_dir, "instance_results.jsonl")
    if count_jsonl_lines(parent_jsonl) > 0:
        return True

    if job["problem"] == "tsp" and job["fixed_scale_files"]:
        for scale_entry in job["fixed_scale_files"]:
            child_jsonl = os.path.join(
                run_dir,
                f"scale_{scale_entry['scale']}",
                "instance_results.jsonl",
            )
            if count_jsonl_lines(child_jsonl) > 0:
                return True
    return False


def should_skip_unstarted_lib_job(args, latest_status, method, problem, job, run_dir):
    if not args.skip_lib_unstarted:
        return False
    if not is_lib_job(job):
        return False
    if (method, problem, job["dataset_name"]) in latest_status:
        return False
    if has_started_outputs(job, run_dir):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easy-root",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "datasets",
            "nss_varying",
            "nss_manifest.json",
        ),
    )
    parser.add_argument("--cuda", default="[0]")
    parser.add_argument("--methods", default="")
    parser.add_argument("--exclude-methods", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-lib-unstarted", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    jobs = build_dataset_jobs(manifest)
    if args.datasets:
        wanted_datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
        jobs = [job for job in jobs if job["dataset_name"] in wanted_datasets]

    base_runs = discover_base_runs(os.path.join(args.easy_root, "results", "test"))
    if args.methods:
        wanted_methods = {item.strip() for item in args.methods.split(",") if item.strip()}
        base_runs = {
            key: value for key, value in base_runs.items() if key[0] in wanted_methods
        }
    if args.exclude_methods:
        excluded_methods = {
            item.strip() for item in args.exclude_methods.split(",") if item.strip()
        }
        base_runs = {
            key: value for key, value in base_runs.items() if key[0] not in excluded_methods
        }

    status_csv = os.path.join(args.easy_root, "results", "nss_eval", "运行清单.csv")
    status_rows = load_existing_status_rows(status_csv)
    latest_status = build_latest_status_map(status_rows)
    executed = 0

    for (method, problem), override_path in sorted(base_runs.items()):
        for job in jobs:
            if job["problem"] != problem:
                continue

            overrides = build_base_overrides(override_path, args.cuda)
            instance_count = args.sample_size if args.sample_size > 0 else job["num_instances"]
            run_dir = os.path.join(
                args.easy_root,
                "results",
                "nss_eval",
                f"{method}_{problem}",
                job["dataset_name"],
            )
            if should_skip_unstarted_lib_job(
                args=args,
                latest_status=latest_status,
                method=method,
                problem=problem,
                job=job,
                run_dir=run_dir,
            ):
                row = {
                    "时间": datetime.now().isoformat(timespec="seconds"),
                    "方法": method,
                    "问题": problem,
                    "数据集": job["dataset_name"],
                    "实例数": job["num_instances"],
                    "状态": "跳过",
                    "结果目录": run_dir,
                    "逐实例结果条数": 0,
                    "返回码": 0,
                    "备注": "按当前策略跳过尚未开始的lib数据集",
                }
                status_rows.append(row)
                latest_status[(method, problem, job["dataset_name"])] = row
                write_status_csv(status_rows, status_csv)
                print(
                    f"skipping unstarted lib job: {method}_{problem}/{job['dataset_name']}"
                )
                executed += 1
                if args.max_runs and executed >= args.max_runs:
                    return
                continue

            if method in GRAPH_SPLIT_TSP_METHODS and problem == "tsp" and job["fixed_scale_files"]:
                ok, return_code, final_lines, note = run_graph_split_tsp_job(
                    args=args,
                    overrides=overrides,
                    job=job,
                    run_dir=run_dir,
                    instance_count=instance_count,
                )
            else:
                ok, return_code, final_lines, note = run_standard_job(
                    args=args,
                    overrides=overrides,
                    job=job,
                    run_dir=run_dir,
                    instance_count=instance_count,
                )

            status_rows.append(
                {
                    "时间": datetime.now().isoformat(timespec="seconds"),
                    "方法": method,
                    "问题": problem,
                    "数据集": job["dataset_name"],
                    "实例数": instance_count,
                    "状态": "完成" if ok else "失败",
                    "结果目录": run_dir,
                    "逐实例结果条数": final_lines,
                    "返回码": return_code,
                    "备注": note,
                }
            )
            latest_status[(method, problem, job["dataset_name"])] = status_rows[-1]
            write_status_csv(status_rows, status_csv)
            executed += 1

            if return_code != 0 and args.stop_on_error:
                raise SystemExit(return_code)
            if args.max_runs and executed >= args.max_runs:
                return


if __name__ == "__main__":
    main()
