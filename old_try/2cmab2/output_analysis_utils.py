from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


# 这些 trace 名字和 run_offline.py 的输出保持一致。
TRACE_FILE_NAMES = [
    "selector_train_ucb.jsonl",
    "selector_val_greedy.jsonl",
    "selector_val_ucb.jsonl",
    "selector_test_greedy.jsonl",
    "selector_test_ucb.jsonl",
]


_TIME_LINE_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")


def _safe_float(value):
    """尽量把值转成 float；失败时原样返回。"""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return value


def _format_value(value, digits: int = 6) -> str:
    """把 markdown / csv 里常见的数值格式化得更易读。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.{digits}f}"
    return str(value)


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|")


def rows_to_markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """
    把 list[dict] 渲染成 markdown 表格。

    这样 notebook 不依赖 pandas，也能稳定显示分析结果。
    """
    if not rows:
        return "_空表_"
    if columns is None:
        columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = [_escape_markdown(_format_value(row.get(col))) for col in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_trace_paths(run_dir: Path) -> dict[str, Path]:
    trace_dir = run_dir / "traces"
    paths: dict[str, Path] = {}
    if not trace_dir.exists():
        return paths
    for name in TRACE_FILE_NAMES:
        path = trace_dir / name
        if path.exists():
            paths[name] = path
    return paths


def parse_timed_log(run_log_path: Path) -> list[dict]:
    """
    读取 run log 中形如 [23:07:19] 的日志行，并自动处理跨午夜。
    """
    if not run_log_path.exists():
        return []

    events: list[dict] = []
    day_offset = 0
    prev_seconds = None
    for raw_line in run_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _TIME_LINE_RE.match(raw_line)
        if not match:
            continue
        hh, mm, ss, message = match.groups()
        seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
        if prev_seconds is not None and seconds < prev_seconds:
            day_offset += 24 * 3600
        absolute_seconds = day_offset + seconds
        prev_seconds = seconds
        events.append(
            {
                "clock": f"{hh}:{mm}:{ss}",
                "message": message,
                "absolute_seconds": absolute_seconds,
            }
        )
    return events


def _find_event_time(events: list[dict], keyword: str) -> int | None:
    for event in events:
        if keyword in event["message"]:
            return int(event["absolute_seconds"])
    return None


def _seconds_to_readable(seconds: int | None) -> str:
    if seconds is None:
        return ""
    hours, remain = divmod(int(seconds), 3600)
    minutes, secs = divmod(remain, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def extract_runtime_summary(run_log_path: Path) -> dict:
    """
    从日志里抽取几个阶段的大致耗时。

    这里只做“轻量、稳健”的日志分析，不要求精确到每个内部小步骤。
    """
    events = parse_timed_log(run_log_path)
    if not events:
        return {
            "events": [],
            "durations": {},
            "runtime_rows": [],
        }

    def duration(start_keyword: str, end_keyword: str) -> int | None:
        start = _find_event_time(events, start_keyword)
        end = _find_event_time(events, end_keyword)
        if start is None or end is None or end < start:
            return None
        return end - start

    durations = {
        "dataset_build": duration("开始构建共享数据集", "数据集构建完成"),
        "training": duration("开始训练 epoch", "开始 selector 验证/测试评测"),
        "selector_eval": duration("开始 selector 验证/测试评测", "开始计算 baseline 对比"),
        "baseline_eval": duration("开始计算 baseline 对比", "baseline 对比完成"),
        "write_results": duration("开始写出结果文件到", "结果文件写出完成"),
    }

    runtime_rows = [
        {
            "stage": key,
            "seconds": value if value is not None else "",
            "readable": _seconds_to_readable(value),
        }
        for key, value in durations.items()
    ]
    return {
        "events": events,
        "durations": durations,
        "runtime_rows": runtime_rows,
    }


def flatten_metric_block(block: dict, source: str, split: str) -> list[dict]:
    rows: list[dict] = []
    for problem, metrics in sorted(block.get("by_problem", {}).items()):
        rows.append(
            {
                "source": source,
                "split": split,
                "scope": problem,
                "count": metrics.get("count", ""),
                "mean_reward": metrics.get("mean_reward", ""),
                "top1_accuracy": metrics.get("top1_accuracy", ""),
                "mean_regret": metrics.get("mean_regret", ""),
                "mean_cost": metrics.get("mean_cost", ""),
            }
        )
    return rows


def infer_rank_from_reward(selected_reward: float, feasible_count: int, reward_mode: str) -> float:
    """
    根据 trace 里落下来的 selected_reward 反推出“选中 arm 的平均名次”。

    注意：
    - 这里反推的是 reward 对应的 tie-aware 平均 rank
    - 它适合计算 top2 / top3 这类“前 k 名命中率”
    - 它不一定和 summary.json 里的 top1_accuracy 完全一致
      因为 top1_accuracy 用的是 `selected_arm == best_arm`
    """
    if feasible_count <= 1:
        return 1.0
    reward = float(selected_reward)
    if reward_mode == "linear_zero_one":
        return 1.0 + (1.0 - reward) * (feasible_count - 1)
    if reward_mode == "autosaea":
        return feasible_count + 1.0 - reward * feasible_count
    raise ValueError(f"未知 reward_mode: {reward_mode}")


def compute_topk_rows(trace_rows: list[dict], reward_mode: str, ks: tuple[int, ...] = (1, 2, 3)) -> list[dict]:
    """
    基于 trace 里的 selected_reward 计算 top-k 命中分析。
    """
    enriched = []
    for row in trace_rows:
        feasible_count = len(row.get("feasible_arms", []))
        rank = infer_rank_from_reward(row["selected_reward"], feasible_count, reward_mode=reward_mode)
        enriched.append(
            {
                "problem": row["problem"],
                "rank": rank,
                **{f"top{k}_hit_rate": 1.0 if rank <= float(k) + 1e-12 else 0.0 for k in ks},
            }
        )

    def aggregate(rows: list[dict], scope: str) -> dict:
        payload = {
            "scope": scope,
            "count": len(rows),
            "mean_selected_rank": (sum(item["rank"] for item in rows) / len(rows)) if rows else "",
        }
        for k in ks:
            key = f"top{k}_hit_rate"
            payload[key] = (sum(item[key] for item in rows) / len(rows)) if rows else ""
        return payload

    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        by_problem[row["problem"]].append(row)
    output = []
    for problem, rows in sorted(by_problem.items()):
        output.append(aggregate(rows, problem))
    return output


def compute_selection_distribution_rows(trace_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    统计“每个方法被选了多少次”和“选择集中度”。
    """
    per_arm_rows: list[dict] = []
    summary_rows: list[dict] = []

    grouped: dict[str, list[dict]] = {}
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        by_problem[row["problem"]].append(row)
    for problem, rows in by_problem.items():
        grouped[problem] = rows

    for scope, rows in grouped.items():
        total = len(rows)
        counts = Counter(row["selected_arm_name"] for row in rows)
        probs = [count / total for count in counts.values()] if total else []
        entropy = -sum(p * math.log(p + 1e-12) for p in probs) if probs else 0.0
        effective_arms = math.exp(entropy) if probs else 0.0
        summary_rows.append(
            {
                "scope": scope,
                "count": total,
                "unique_selected_arms": len(counts),
                "entropy": entropy,
                "effective_arms": effective_arms,
            }
        )
        for arm_name, count in counts.most_common():
            per_arm_rows.append(
                {
                    "scope": scope,
                    "arm": arm_name,
                    "count": count,
                    "fraction": count / total if total else "",
                }
            )
    return per_arm_rows, summary_rows


def compute_selected_arm_quality_rows(trace_rows: list[dict]) -> list[dict]:
    """
    统计“模型一旦选到某个 arm，真实效果通常如何”。
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in trace_rows:
        grouped[(row["problem"], row["selected_arm_name"])].append(row)

    rows: list[dict] = []
    for (scope, arm_name), items in sorted(grouped.items()):
        count = len(items)
        rows.append(
            {
                "scope": scope,
                "arm": arm_name,
                "count": count,
                "mean_reward": sum(float(item["selected_reward"]) for item in items) / count,
                "mean_regret": sum(float(item["regret"]) for item in items) / count,
                "top1_rate": sum(float(item["top1"]) for item in items) / count,
                "mean_cost": sum(float(item["selected_cost"]) for item in items) / count,
            }
        )

    # 让每个 scope 下按 count 从高到低排序，更容易读。
    scope_order = {"cvrp": 1, "tsp": 2}
    rows.sort(key=lambda row: (scope_order.get(row["scope"], 99), -int(row["count"]), row["arm"]))
    return rows


def _repo_root() -> Path:
    """返回仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def _resolve_results_dir_from_summary(summary: dict) -> Path:
    """
    从 summary.json 里恢复 results/test 路径。

    注意：
    - 训练时 summary 里可能记录的是远端服务器绝对路径
    - notebook 本地复盘时这个绝对路径未必存在
    - 因此这里优先尝试 summary 里的路径，不存在时自动回退到当前仓库的 EasyNCO/results/test
    """
    config = summary.get("config", {})
    raw = str(config.get("results_dir", "") or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    local_fallback = _repo_root() / "EasyNCO" / "results" / "test"
    return local_fallback


def _resolve_optional_dataset_path(raw_value: object) -> Path | None:
    """
    解析 summary 里记录的数据集路径。

    如果训练时写入的是远端绝对路径，而本地不存在，则尽量映射回当前仓库的
    `EasyNCO/data/datasets/...` 目录。
    """
    raw = str(raw_value or "").strip()
    if not raw:
        return None

    path = Path(raw)
    if path.exists():
        return path

    repo_root = _repo_root()
    datasets_root = repo_root / "EasyNCO" / "data" / "datasets"

    if path.is_absolute():
        parts = list(path.parts)
        if "datasets" in parts:
            idx = parts.index("datasets")
            suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path(path.name)
            candidate = datasets_root / suffix
            return candidate
        return datasets_root / path.name

    return path


def _build_uid_to_sample(summary: dict) -> dict[str, object]:
    """
    按当前 run 的配置重建 joint dataset，并建立 uid -> sample 映射。

    这样分析 notebook 就能回到“完整实例视图”，从而计算：
    - test 集上每个方法自己的 mean cost
    - selector 相比这些方法到底更好还是更差
    """
    try:
        if __package__ in (None, ""):
            from build_dataset import build_joint_dataset  # type: ignore
        else:
            from .build_dataset import build_joint_dataset
    except Exception as exc:  # pragma: no cover - 这里主要是为了 notebook 友好报错
        raise RuntimeError(
            "生成 test 集各方法 mean cost 需要可用的 PyTorch 环境，因为要回溯 EasyNCO 的 .pt 数据集。"
        ) from exc

    config = summary.get("config", {})
    results_dir = _resolve_results_dir_from_summary(summary)
    dataset = build_joint_dataset(
        results_dir=results_dir,
        tsp_dataset_path=_resolve_optional_dataset_path(config.get("tsp_dataset_path")),
        cvrp_dataset_path=_resolve_optional_dataset_path(config.get("cvrp_dataset_path")),
        data_source=str(config.get("data_source", "easynco_results")),
        nss_dataset_root=None if not config.get("nss_dataset_root") else Path(str(config.get("nss_dataset_root"))),
        reward_mode=str(config.get("reward_mode", "linear_zero_one")),
        max_samples_per_problem=int(config.get("max_samples_per_problem", 0) or 0),
        seed=int(config.get("seed", 0) or 0),
        log_fn=None,
        show_progress=False,
    )
    return {sample.uid: sample for sample in dataset.samples}


def compute_method_mean_cost_rows_for_trace(
    trace_rows: list[dict],
    uid_to_sample: dict[str, object],
) -> tuple[list[dict], list[dict]]:
    """
    统计“在当前 trace 对应的 test 集上，每个方法自己的 mean cost”。

    这是对 selector 的一个很有用的横向参照：
    - selector 不是只和 single-best / oracle 比
    - 也能直接看“它比多少个单独方法更好”
    """
    selector_costs_by_scope: dict[str, list[float]] = defaultdict(list)
    method_costs_by_scope_arm: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in trace_rows:
        uid = str(row.get("uid"))
        sample = uid_to_sample.get(uid)
        if sample is None:
            continue
        scope = str(row.get("problem", getattr(sample, "problem", "")))
        selector_costs_by_scope[scope].append(float(row["selected_cost"]))

        costs = getattr(sample, "costs")
        feasible_mask = getattr(sample, "feasible_mask")
        result_records = getattr(sample, "result_records")
        for arm_idx, cost in enumerate(costs):
            if arm_idx >= len(feasible_mask) or float(feasible_mask[arm_idx]) <= 0:
                continue
            value = float(cost)
            if not math.isfinite(value):
                continue
            record = result_records[arm_idx]
            if record is None:
                continue
            arm_name = str(record.get("method") or record.get("name") or f"arm_{arm_idx}")
            method_costs_by_scope_arm[(scope, arm_name)].append(value)

    selector_mean_by_scope = {
        scope: (sum(values) / len(values))
        for scope, values in selector_costs_by_scope.items()
        if values
    }

    method_rows: list[dict] = []
    for (scope, arm_name), values in sorted(method_costs_by_scope_arm.items()):
        if not values:
            continue
        mean_cost = sum(values) / len(values)
        selector_mean_cost = selector_mean_by_scope.get(scope)
        delta_vs_selector = (mean_cost - selector_mean_cost) if selector_mean_cost is not None else ""
        method_rows.append(
            {
                "scope": scope,
                "arm": arm_name,
                "count": len(values),
                "mean_cost": mean_cost,
                "selector_mean_cost": selector_mean_cost if selector_mean_cost is not None else "",
                "delta_vs_selector": delta_vs_selector,
                "selector_better": (selector_mean_cost < mean_cost) if selector_mean_cost is not None else "",
            }
        )

    scope_order = {"cvrp": 1, "tsp": 2}
    method_rows.sort(key=lambda row: (scope_order.get(row["scope"], 99), float(row["mean_cost"]), row["arm"]))

    summary_rows: list[dict] = []
    grouped_method_rows: dict[str, list[dict]] = defaultdict(list)
    for row in method_rows:
        grouped_method_rows[str(row["scope"])].append(row)

    for scope, rows in sorted(grouped_method_rows.items(), key=lambda item: scope_order.get(item[0], 99)):
        selector_mean_cost = selector_mean_by_scope.get(scope)
        if selector_mean_cost is None:
            continue
        better = sum(1 for row in rows if float(row["mean_cost"]) + 1e-12 < selector_mean_cost)
        worse = sum(1 for row in rows if float(row["mean_cost"]) > selector_mean_cost + 1e-12)
        equal = len(rows) - better - worse
        summary_rows.append(
            {
                "scope": scope,
                "selector_mean_cost": selector_mean_cost,
                "n_methods": len(rows),
                "methods_better_than_selector": better,
                "methods_worse_than_selector": worse,
                "methods_equal_to_selector": equal,
                "selector_rank_among_methods": better + 1,
            }
        )

    return method_rows, summary_rows


def compute_best_arm_rows(trace_rows: list[dict]) -> list[dict]:
    """
    统计 oracle 最优 arm 在测试集里的分布。
    """
    grouped: dict[str, list[dict]] = {}
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        by_problem[row["problem"]].append(row)
    for problem, rows in by_problem.items():
        grouped[problem] = rows

    output: list[dict] = []
    for scope, rows in grouped.items():
        total = len(rows)
        counts = Counter(row["best_arm_name"] for row in rows)
        for arm_name, count in counts.most_common():
            output.append(
                {
                    "scope": scope,
                    "arm": arm_name,
                    "oracle_best_count": count,
                    "oracle_best_fraction": count / total if total else "",
                }
            )
    return output


def compute_oracle_coverage_rows(trace_rows: list[dict]) -> list[dict]:
    """
    对比“某个 arm 在 oracle 里最优多少次”和“模型实际选了它多少次”。
    """
    selected_counts = Counter(row["selected_arm_name"] for row in trace_rows)
    best_counts = Counter(row["best_arm_name"] for row in trace_rows)
    all_arms = sorted(set(selected_counts) | set(best_counts))
    rows = []
    for arm_name in all_arms:
        best_count = int(best_counts.get(arm_name, 0))
        selected_count = int(selected_counts.get(arm_name, 0))
        rows.append(
            {
                "arm": arm_name,
                "oracle_best_count": best_count,
                "selected_count": selected_count,
                "selected_minus_oracle": selected_count - best_count,
                "coverage_over_oracle": (selected_count / best_count) if best_count > 0 else "",
            }
        )
    rows.sort(key=lambda row: (-int(row["oracle_best_count"]), row["arm"]))
    return rows


def compare_greedy_ucb(greedy_rows: list[dict], ucb_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    对比测试时 greedy 与 ucb 两种决策模式的差异。
    """
    by_uid: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in greedy_rows:
        by_uid[row["uid"]]["greedy"] = row
    for row in ucb_rows:
        by_uid[row["uid"]]["ucb"] = row

    summary_counter = {
        "total": 0,
        "same_selection": 0,
        "different_selection": 0,
        "ucb_better_when_diff": 0,
        "ucb_worse_when_diff": 0,
        "equal_when_diff": 0,
    }
    by_problem_counter: dict[str, Counter] = defaultdict(Counter)
    example_rows: list[dict] = []

    for uid, item in sorted(by_uid.items()):
        greedy = item.get("greedy")
        ucb = item.get("ucb")
        if greedy is None or ucb is None:
            continue
        summary_counter["total"] += 1
        problem = greedy["problem"]
        if greedy["selected_arm"] == ucb["selected_arm"]:
            summary_counter["same_selection"] += 1
            by_problem_counter[problem]["same_selection"] += 1
            continue

        summary_counter["different_selection"] += 1
        by_problem_counter[problem]["different_selection"] += 1
        greedy_regret = float(greedy["regret"])
        ucb_regret = float(ucb["regret"])
        if ucb_regret < greedy_regret:
            summary_counter["ucb_better_when_diff"] += 1
            by_problem_counter[problem]["ucb_better_when_diff"] += 1
        elif ucb_regret > greedy_regret:
            summary_counter["ucb_worse_when_diff"] += 1
            by_problem_counter[problem]["ucb_worse_when_diff"] += 1
        else:
            summary_counter["equal_when_diff"] += 1
            by_problem_counter[problem]["equal_when_diff"] += 1

        if len(example_rows) < 20:
            example_rows.append(
                {
                    "uid": uid,
                    "problem": problem,
                    "greedy_arm": greedy["selected_arm_name"],
                    "greedy_score": greedy["selected_score"],
                    "greedy_regret": greedy_regret,
                    "ucb_arm": ucb["selected_arm_name"],
                    "ucb_score": ucb["selected_score"],
                    "ucb_regret": ucb_regret,
                    "best_arm": greedy["best_arm_name"],
                    "best_cost": greedy["best_cost"],
                }
            )

    total = max(summary_counter["total"], 1)
    summary_rows = []
    for problem, counter in sorted(by_problem_counter.items()):
        total_problem = counter["same_selection"] + counter["different_selection"]
        summary_rows.append(
            {
                "scope": problem,
                "total": total_problem,
                "same_selection": counter["same_selection"],
                "different_selection": counter["different_selection"],
                "ucb_better_when_diff": counter["ucb_better_when_diff"],
                "ucb_worse_when_diff": counter["ucb_worse_when_diff"],
                "equal_when_diff": counter["equal_when_diff"],
                "different_ratio": (counter["different_selection"] / total_problem) if total_problem else "",
            }
        )
    return summary_rows, example_rows


def build_overview_lines(bundle: dict) -> list[str]:
    summary = bundle["summary"]
    config = summary.get("config", {})
    splits = summary.get("splits", {})
    runtime = bundle.get("runtime", {})
    durations = runtime.get("durations", {})
    artifacts = summary.get("artifacts", {})
    checkpoint_files = artifacts.get("checkpoint_files", [])

    lines = [
        f"- run 目录: `{bundle['run_dir']}`",
        f"- method: `{config.get('method', '')}`",
        f"- reward_mode: `{config.get('reward_mode', '')}`",
        f"- train/val/test: `{splits.get('train', '')} / {splits.get('val', '')} / {splits.get('test', '')}`",
        f"- alpha: `{config.get('alpha', '')}`, hidden_dim: `{config.get('hidden_dim', '')}`, train_every: `{config.get('train_every', '')}`, representation_steps: `{config.get('representation_steps', '')}`",
        f"- representation_buffer_size: `{config.get('representation_buffer_size', '')}`, linear_head_buffer_size: `{config.get('linear_head_buffer_size', '')}`",
        f"- freeze_encoder: `{config.get('freeze_encoder', '')}`",
        f"- checkpoint 文件数: `{len(checkpoint_files)}`",
    ]
    if durations.get("training") is not None:
        lines.append(
            f"- 训练耗时: `{_seconds_to_readable(durations['training'])}`"
        )
    return lines


def build_analysis_bundle(run_dir: str | Path) -> dict:
    """
    读取一个 run 输出目录，生成 notebook / markdown 都能复用的分析数据。
    """
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    run_log_path = run_dir / "run2.log"
    summary = load_json(summary_path)
    reward_mode = summary.get("config", {}).get("reward_mode", "linear_zero_one")

    trace_paths = discover_trace_paths(run_dir)
    traces = {name: load_jsonl(path) for name, path in trace_paths.items()}

    selector_rows: list[dict] = []
    for split_name, block in summary.get("selector", {}).items():
        selector_rows.extend(flatten_metric_block(block, source="selector", split=split_name))

    baseline_rows: list[dict] = []
    for baseline_name, block in summary.get("baselines", {}).items():
        baseline_rows.extend(flatten_metric_block(block, source=baseline_name, split="test_greedy"))

    topk_by_trace: dict[str, list[dict]] = {}
    selection_distribution_by_trace: dict[str, list[dict]] = {}
    selection_summary_by_trace: dict[str, list[dict]] = {}
    selected_arm_quality_by_trace: dict[str, list[dict]] = {}
    test_method_mean_cost_by_trace: dict[str, list[dict]] = {}
    selector_vs_methods_summary_by_trace: dict[str, list[dict]] = {}
    method_mean_cost_error = ""

    for trace_name, rows in traces.items():
        if not rows:
            continue
        topk_by_trace[trace_name] = compute_topk_rows(rows, reward_mode=reward_mode, ks=(1, 2, 3))
        distribution_rows, summary_rows = compute_selection_distribution_rows(rows)
        selection_distribution_by_trace[trace_name] = distribution_rows
        selection_summary_by_trace[trace_name] = summary_rows
        selected_arm_quality_by_trace[trace_name] = compute_selected_arm_quality_rows(rows)

    test_trace_names = [name for name in traces if name.startswith("selector_test_")]
    if test_trace_names:
        try:
            uid_to_sample = _build_uid_to_sample(summary)
            for trace_name in test_trace_names:
                method_rows, summary_rows = compute_method_mean_cost_rows_for_trace(traces[trace_name], uid_to_sample)
                test_method_mean_cost_by_trace[trace_name] = method_rows
                selector_vs_methods_summary_by_trace[trace_name] = summary_rows
        except Exception as exc:
            method_mean_cost_error = str(exc)

    greedy_vs_ucb_summary = []
    greedy_vs_ucb_examples = []
    if "selector_test_greedy.jsonl" in traces and "selector_test_ucb.jsonl" in traces:
        greedy_vs_ucb_summary, greedy_vs_ucb_examples = compare_greedy_ucb(
            traces["selector_test_greedy.jsonl"],
            traces["selector_test_ucb.jsonl"],
        )

    best_arm_rows = compute_best_arm_rows(traces["selector_test_greedy.jsonl"]) if "selector_test_greedy.jsonl" in traces else []
    oracle_coverage_rows = compute_oracle_coverage_rows(traces["selector_test_greedy.jsonl"]) if "selector_test_greedy.jsonl" in traces else []

    runtime = extract_runtime_summary(run_log_path)
    bundle = {
        "run_dir": str(run_dir),
        "summary": summary,
        "reward_mode": reward_mode,
        "trace_paths": {key: str(value) for key, value in trace_paths.items()},
        "runtime": runtime,
        "overview_lines": build_overview_lines({"run_dir": run_dir, "summary": summary, "runtime": runtime}),
        "selector_rows": selector_rows,
        "baseline_rows": baseline_rows,
        "topk_by_trace": topk_by_trace,
        "selection_distribution_by_trace": selection_distribution_by_trace,
        "selection_summary_by_trace": selection_summary_by_trace,
        "selected_arm_quality_by_trace": selected_arm_quality_by_trace,
        "test_method_mean_cost_by_trace": test_method_mean_cost_by_trace,
        "selector_vs_methods_summary_by_trace": selector_vs_methods_summary_by_trace,
        "method_mean_cost_error": method_mean_cost_error,
        "greedy_vs_ucb_summary": greedy_vs_ucb_summary,
        "greedy_vs_ucb_examples": greedy_vs_ucb_examples,
        "best_arm_rows": best_arm_rows,
        "oracle_coverage_rows": oracle_coverage_rows,
    }
    return bundle


def build_markdown_report(bundle: dict) -> str:
    """
    生成一份适合直接保存到 markdown 的文本报告。
    """
    lines = [
        "# 2cmab2 运行结果复盘",
        "",
        "## 基本信息",
        *bundle.get("overview_lines", []),
        "",
        "## Selector 指标",
        rows_to_markdown_table(
            bundle.get("selector_rows", []),
            columns=["source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost"],
        ),
        "",
        "## Baseline 指标",
        rows_to_markdown_table(
            bundle.get("baseline_rows", []),
            columns=["source", "split", "scope", "count", "mean_reward", "top1_accuracy", "mean_regret", "mean_cost"],
        ),
        "",
        "## Top-k 命中分析",
        "",
        "- 这里的 `top1/top2/top3` 不是 summary.json 里原生给出的指标，而是根据 trace 中的 `selected_reward` 反推选中方法的 tie-aware 平均 rank 后得到。",
        "- 因此它更适合回答“选中的方法是不是至少落在真实前 1 / 前 2 / 前 3 名”。",
        "- 它和 `top1_accuracy` 的定义不同：`top1_accuracy` 用的是 `selected_arm == best_arm`。",
        "",
    ]

    for trace_name, rows in bundle.get("topk_by_trace", {}).items():
        lines.extend(
            [
                f"### {trace_name}",
                rows_to_markdown_table(rows, columns=["scope", "count", "mean_selected_rank", "top1_hit_rate", "top2_hit_rate", "top3_hit_rate"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 选臂分布",
        ]
    )
    for trace_name, rows in bundle.get("selection_summary_by_trace", {}).items():
        lines.extend(
            [
                f"### {trace_name} 选择集中度",
                rows_to_markdown_table(rows, columns=["scope", "count", "unique_selected_arms", "entropy", "effective_arms"]),
                "",
            ]
        )
    for trace_name, rows in bundle.get("selection_distribution_by_trace", {}).items():
        lines.extend(
            [
                f"### {trace_name} 各 arm 被选比例",
                rows_to_markdown_table(rows, columns=["scope", "arm", "count", "fraction"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 每个 arm 被选中后的真实效果",
        ]
    )
    for trace_name, rows in bundle.get("selected_arm_quality_by_trace", {}).items():
        lines.extend(
            [
                f"### {trace_name}",
                rows_to_markdown_table(rows, columns=["scope", "arm", "count", "mean_reward", "mean_regret", "top1_rate", "mean_cost"]),
                "",
            ]
        )

    lines.extend(
        [
            "## test 集各方法 mean cost 与 selector 对比",
        ]
    )
    if bundle.get("method_mean_cost_error"):
        lines.extend(
            [
                f"- 该部分生成失败：`{bundle['method_mean_cost_error']}`",
                "",
            ]
        )
    else:
        for trace_name, rows in bundle.get("selector_vs_methods_summary_by_trace", {}).items():
            lines.extend(
                [
                    f"### {trace_name} 汇总",
                    rows_to_markdown_table(
                        rows,
                        columns=[
                            "scope",
                            "selector_mean_cost",
                            "n_methods",
                            "methods_better_than_selector",
                            "methods_worse_than_selector",
                            "methods_equal_to_selector",
                            "selector_rank_among_methods",
                        ],
                    ),
                    "",
                ]
            )
        for trace_name, rows in bundle.get("test_method_mean_cost_by_trace", {}).items():
            lines.extend(
                [
                    f"### {trace_name} 各方法 mean cost",
                    rows_to_markdown_table(
                        rows,
                        columns=["scope", "arm", "count", "mean_cost", "selector_mean_cost", "delta_vs_selector", "selector_better"],
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## greedy 与 ucb 的差异",
            rows_to_markdown_table(
                bundle.get("greedy_vs_ucb_summary", []),
                columns=["scope", "total", "same_selection", "different_selection", "different_ratio", "ucb_better_when_diff", "ucb_worse_when_diff", "equal_when_diff"],
            ),
            "",
            rows_to_markdown_table(
                bundle.get("greedy_vs_ucb_examples", []),
                columns=["uid", "problem", "greedy_arm", "greedy_score", "greedy_regret", "ucb_arm", "ucb_score", "ucb_regret", "best_arm", "best_cost"],
            ),
            "",
            "## oracle 最优 arm 分布",
            rows_to_markdown_table(bundle.get("best_arm_rows", []), columns=["scope", "arm", "oracle_best_count", "oracle_best_fraction"]),
            "",
            "## oracle 覆盖度",
            rows_to_markdown_table(bundle.get("oracle_coverage_rows", []), columns=["arm", "oracle_best_count", "selected_count", "selected_minus_oracle", "coverage_over_oracle"]),
            "",
        ]
    )
    return "\n".join(lines)


def export_analysis_bundle(bundle: dict, output_dir: str | Path) -> dict[str, str]:
    """
    把 notebook 里显示的分析结果同时导出成 markdown / csv / json。
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.md"
    report_path.write_text(build_markdown_report(bundle), encoding="utf-8")

    _write_csv(output_dir / "selector_metrics.csv", bundle.get("selector_rows", []))
    _write_csv(output_dir / "baseline_metrics.csv", bundle.get("baseline_rows", []))
    _write_csv(output_dir / "greedy_vs_ucb_summary.csv", bundle.get("greedy_vs_ucb_summary", []))
    _write_csv(output_dir / "greedy_vs_ucb_examples.csv", bundle.get("greedy_vs_ucb_examples", []))
    _write_csv(output_dir / "oracle_best_distribution.csv", bundle.get("best_arm_rows", []))
    _write_csv(output_dir / "oracle_coverage.csv", bundle.get("oracle_coverage_rows", []))
    _write_csv(output_dir / "runtime_rows.csv", bundle.get("runtime", {}).get("runtime_rows", []))
    if bundle.get("method_mean_cost_error"):
        (output_dir / "method_mean_cost_error.txt").write_text(str(bundle["method_mean_cost_error"]), encoding="utf-8")

    for trace_name, rows in bundle.get("topk_by_trace", {}).items():
        stem = trace_name.replace(".jsonl", "")
        _write_csv(output_dir / f"{stem}_topk.csv", rows)
    for trace_name, rows in bundle.get("selection_distribution_by_trace", {}).items():
        stem = trace_name.replace(".jsonl", "")
        _write_csv(output_dir / f"{stem}_selection_distribution.csv", rows)
    for trace_name, rows in bundle.get("selection_summary_by_trace", {}).items():
        stem = trace_name.replace(".jsonl", "")
        _write_csv(output_dir / f"{stem}_selection_summary.csv", rows)
    for trace_name, rows in bundle.get("selected_arm_quality_by_trace", {}).items():
        stem = trace_name.replace(".jsonl", "")
        _write_csv(output_dir / f"{stem}_selected_arm_quality.csv", rows)
    for trace_name, rows in bundle.get("test_method_mean_cost_by_trace", {}).items():
        stem = trace_name.replace(".jsonl", "")
        _write_csv(output_dir / f"{stem}_method_mean_cost.csv", rows)
    for trace_name, rows in bundle.get("selector_vs_methods_summary_by_trace", {}).items():
        stem = trace_name.replace(".jsonl", "")
        _write_csv(output_dir / f"{stem}_selector_vs_methods_summary.csv", rows)

    overview_json_path = output_dir / "overview.json"
    overview_json_path.write_text(
        json.dumps(
            {
                "run_dir": bundle.get("run_dir"),
                "reward_mode": bundle.get("reward_mode"),
                "trace_paths": bundle.get("trace_paths"),
                "runtime": bundle.get("runtime", {}).get("durations", {}),
                "overview_lines": bundle.get("overview_lines", []),
                "method_mean_cost_error": bundle.get("method_mean_cost_error", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "report_md": str(report_path),
        "overview_json": str(overview_json_path),
    }
