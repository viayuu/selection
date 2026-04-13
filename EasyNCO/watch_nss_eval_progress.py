import csv
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime


EASY_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(EASY_ROOT, "results", "nss_eval")
STATUS_CSV = os.path.join(RESULTS_ROOT, "运行清单.csv")
CTRL_LOG = os.path.join(RESULTS_ROOT, "总控日志.log")
HUMAN_LOG = os.path.join(RESULTS_ROOT, "监控日志.log")
LATEST_JSON = os.path.join(RESULTS_ROOT, "最新监控.json")
PID_FILE = os.path.join(RESULTS_ROOT, "监控进程.pid")
POLICY_FILE = os.path.join(RESULTS_ROOT, "调度策略.json")
MANIFEST_PATH = os.path.join(
    EASY_ROOT, "data", "datasets", "nss_varying", "nss_manifest.json"
)
GRAPH_SPLIT_TSP_METHODS = {"difusco", "t2t"}
RELEVANT_PROBLEMS = {"tsp", "cvrp"}
SLEEP_SECONDS = 60


def count_jsonl_lines(path):
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_policy():
    if not os.path.isfile(POLICY_FILE):
        return {}
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_base_runs(results_root):
    discovered = {}
    if not os.path.isdir(results_root):
        return discovered
    for entry in sorted(os.listdir(results_root)):
        full_dir = os.path.join(results_root, entry)
        if not os.path.isdir(full_dir) or "_" not in entry:
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


def load_latest_status():
    latest = {}
    if not os.path.isfile(STATUS_CSV):
        return latest
    with open(STATUS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            latest[(row["方法"], row["问题"], row["数据集"])] = row
    return latest


def load_active_processes():
    cmd = 'pgrep -af "launch_nss_eval.sh|python run_nss_eval_suite.py|eval.py mode=test"'
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=EASY_ROOT,
    )
    lines = []
    for line in result.stdout.splitlines():
        if "codex-linux-sandbox" in line or "pgrep -af" in line:
            continue
        lines.append(line.strip())
    return lines


def parse_active_eval(active_processes):
    for line in active_processes:
        if " eval.py " not in line and " eval.py" not in line:
            continue
        match = re.search(r"dir=([^ ]+)", line)
        run_dir = match.group(1) if match else ""
        parts = run_dir.split(os.sep) if run_dir else []
        scale_match = re.search(r"/(scale_\d+)$", run_dir)
        scale_name = scale_match.group(1) if scale_match else ""
        if scale_name and len(parts) >= 3:
            method_problem = parts[-3]
            dataset = parts[-2]
        else:
            method_problem = parts[-2] if len(parts) >= 2 else ""
            dataset = parts[-1] if len(parts) >= 1 else ""
        return {
            "命令": line,
            "结果目录": run_dir,
            "方法问题": method_problem,
            "数据集或子任务": dataset,
            "子任务": scale_name,
        }
    return None


def collect_job_progress(method, problem, dataset_name, dataset_meta, latest_status):
    expected_instances = dataset_meta["num_instances"]
    run_dir = os.path.join(RESULTS_ROOT, f"{method}_{problem}", dataset_name)
    key = (method, problem, dataset_name)
    latest_row = latest_status.get(key)

    if method in GRAPH_SPLIT_TSP_METHODS and problem == "tsp" and dataset_meta.get("fixed_scale_files"):
        total_scales = len(dataset_meta["fixed_scale_files"])
        done_scales = 0
        instance_done = 0
        partial_scales = []
        for scale_entry in dataset_meta["fixed_scale_files"]:
            child_jsonl = os.path.join(
                run_dir,
                f"scale_{scale_entry['scale']}",
                "instance_results.jsonl",
            )
            child_count = count_jsonl_lines(child_jsonl)
            instance_done += min(child_count, scale_entry["num_instances"])
            if child_count == scale_entry["num_instances"]:
                done_scales += 1
            elif child_count > 0:
                partial_scales.append(
                    {
                        "scale": scale_entry["scale"],
                        "done": child_count,
                        "expected": scale_entry["num_instances"],
                    }
                )
        parent_jsonl = os.path.join(run_dir, "instance_results.jsonl")
        parent_count = count_jsonl_lines(parent_jsonl)
        return {
            "方法": method,
            "问题": problem,
            "数据集": dataset_name,
            "结果目录": run_dir,
            "预期实例数": expected_instances,
            "已写实例数": instance_done,
            "父结果条数": parent_count,
            "总scale数": total_scales,
            "已完成scale数": done_scales,
            "部分完成scale": partial_scales[:10],
            "最近状态": latest_row["状态"] if latest_row else "未记录",
            "最近备注": latest_row["备注"] if latest_row else "",
        }

    instance_jsonl = os.path.join(run_dir, "instance_results.jsonl")
    return {
        "方法": method,
        "问题": problem,
        "数据集": dataset_name,
        "结果目录": run_dir,
        "预期实例数": expected_instances,
        "已写实例数": count_jsonl_lines(instance_jsonl),
        "最近状态": latest_row["状态"] if latest_row else "未记录",
        "最近备注": latest_row["备注"] if latest_row else "",
    }


def build_snapshot():
    manifest = load_manifest()
    policy = load_policy()
    excluded_methods = set(policy.get("exclude_methods", []))
    base_runs = discover_base_runs(os.path.join(EASY_ROOT, "results", "test"))
    if excluded_methods:
        base_runs = {
            key: value for key, value in base_runs.items() if key[0] not in excluded_methods
        }
    latest_status = load_latest_status()
    active_processes = load_active_processes()
    active_eval = parse_active_eval(active_processes)

    jobs = []
    for (method, problem), _ in sorted(base_runs.items()):
        for dataset_name, meta in manifest.items():
            if meta["problem"] != problem:
                continue
            jobs.append(collect_job_progress(method, problem, dataset_name, meta, latest_status))

    completed_jobs = 0
    running_jobs = 0
    pending_jobs = 0
    skipped_jobs = 0
    for job in jobs:
        if job["最近状态"] == "跳过":
            skipped_jobs += 1
        elif job["已写实例数"] >= job["预期实例数"] and job["预期实例数"] > 0:
            completed_jobs += 1
        elif job["已写实例数"] > 0:
            running_jobs += 1
        else:
            running_jobs += 1 if active_eval and active_eval["结果目录"].startswith(job["结果目录"]) else 0
            if not (active_eval and active_eval["结果目录"].startswith(job["结果目录"])):
                pending_jobs += 1

    status_counter = Counter()
    for row in latest_status.values():
        status_counter[row["状态"]] += 1

    return {
        "时间": datetime.now().isoformat(timespec="seconds"),
        "活跃进程": active_processes,
        "当前评测": active_eval,
        "总任务数": len(jobs),
        "完成任务数": completed_jobs,
        "进行中任务数": running_jobs,
        "待开始任务数": pending_jobs,
        "跳过任务数": skipped_jobs,
        "运行清单状态汇总": dict(status_counter),
        "任务进度": jobs,
    }


def append_human_log(snapshot):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    current = snapshot.get("当前评测") or {}
    line = (
        f"[{snapshot['时间']}] "
        f"完成任务 {snapshot['完成任务数']}/{snapshot['总任务数']} | "
        f"进行中 {snapshot['进行中任务数']} | "
        f"待开始 {snapshot['待开始任务数']} | "
        f"跳过 {snapshot['跳过任务数']} | "
        f"当前评测: {current.get('方法问题', '无')} "
        f"{current.get('数据集或子任务', '')} "
        f"{current.get('子任务', '')}"
    ).strip()
    with open(HUMAN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_snapshot(snapshot):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    while True:
        snapshot = build_snapshot()
        write_snapshot(snapshot)
        append_human_log(snapshot)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
