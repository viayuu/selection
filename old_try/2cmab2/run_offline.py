from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm 不是强依赖
    tqdm = None

if __package__ in (None, ""):
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))

    from arm_config import ALL_METHODS, PROBLEM_TO_METHODS, get_arm_space_config  # type: ignore
    from baselines import (  # type: ignore
        OracleSelector,
        RandomSelector,
        SingleBestGlobalSelector,
        SingleBestPerProblemSelector,
    )
    from build_dataset import build_joint_dataset  # type: ignore
    from default_settings import RunOfflineDefaults  # type: ignore
    from linucb import LinUCB  # type: ignore
    from neural_linucb import NeuralLinUCB  # type: ignore
    from neural_ucb_diag import NeuralUCBDiag  # type: ignore
    from output_analysis_utils import infer_rank_from_reward  # type: ignore
else:
    from .arm_config import ALL_METHODS, PROBLEM_TO_METHODS, get_arm_space_config
    from .baselines import OracleSelector, RandomSelector, SingleBestGlobalSelector, SingleBestPerProblemSelector
    from .build_dataset import build_joint_dataset
    from .default_settings import RunOfflineDefaults
    from .linucb import LinUCB
    from .neural_linucb import NeuralLinUCB
    from .neural_ucb_diag import NeuralUCBDiag
    from .output_analysis_utils import infer_rank_from_reward


def _log(message: str, enabled: bool = True) -> None:
    """
    输出一条带时间戳的阶段日志。

    这样在服务器上跑长任务时，即使某个阶段暂时没有最终结果，
    也能知道程序现在到底走到了哪一步。
    """
    if not enabled:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _iter_with_progress(iterable, *, total: int | None = None, desc: str = "", enabled: bool = True):
    """
    给一个可迭代对象包一层进度条。

    - 如果环境里有 tqdm，就显示真实进度条
    - 如果没有 tqdm，或者用户关闭了进度条，就退化为普通迭代
    """
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def _add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """
    为不同 Python 版本统一添加布尔开关参数。

    背景：
    - `argparse.BooleanOptionalAction` 是较新的接口
    - 一些服务器环境里的 Python 版本偏老，没有这个属性

    因此这里做一个兼容层：
    - 新版本 Python：继续使用 `--xxx / --no-xxx`
    - 老版本 Python：手动用 mutually exclusive group 模拟同样的行为
    """
    action_cls = getattr(argparse, "BooleanOptionalAction", None)
    if action_cls is not None:
        parser.add_argument(
            name,
            action=action_cls,
            default=default,
            help=help_text,
        )
        return

    option = name.lstrip("-")
    dest = option.replace("-", "_")
    negative_name = f"--no-{option}"
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(negative_name, dest=dest, action="store_false", help=f"关闭{help_text}")
    parser.set_defaults(**{dest: default})


def subsample_samples_per_problem(samples, max_samples_per_problem: int = 0, seed: int = 0):
    """
    对已经构建好的样本列表按问题做子采样。

    这个函数主要用于：
    - 单元测试
    - 调试时对已有样本对象再做进一步裁剪

    正式运行里更推荐直接在 build_joint_dataset 阶段就限制样本数，
    那样可以避免无谓的特征提取开销。
    """
    if not max_samples_per_problem:
        return list(samples)

    rng = np.random.default_rng(seed)
    by_problem: dict[str, list] = defaultdict(list)
    for sample in samples:
        by_problem[sample.problem].append(sample)

    reduced = []
    for problem in sorted(by_problem):
        items = by_problem[problem]
        if len(items) <= max_samples_per_problem:
            reduced.extend(items)
            continue
        indices = sorted(rng.choice(len(items), size=max_samples_per_problem, replace=False).tolist())
        reduced.extend([items[idx] for idx in indices])
    return reduced


def stratified_split_indices(samples, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 0):
    """
    按问题类型分层切分 train / val / test。

    为什么要分层？
    因为当前数据同时包含 TSP 和 CVRP。
    如果不分层，极端情况下某个 split 里可能几乎只有一种问题，
    这样评测就会失真。
    """
    rng = np.random.default_rng(seed)
    by_problem: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_problem[sample.problem].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for problem, indices in by_problem.items():
        perm = rng.permutation(indices).tolist()
        n = len(perm)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_idx.extend(perm[:n_train])
        val_idx.extend(perm[n_train : n_train + n_val])
        test_idx.extend(perm[n_train + n_val :])

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def _selector_pick(selector, sample, decision_mode: str):
    """
    统一封装不同 selector 的调用方式。

    原因：
    - LinUCB 的输入是 context + mask
    - Neural bandit 的输入是 sample
    - baseline 的接口又更简单
    """
    use_ucb = decision_mode.lower() == "ucb"
    if isinstance(selector, LinUCB):
        return selector.select(sample.linucb_context, mask=sample.feasible_mask, use_ucb=use_ucb)
    try:
        return selector.select(sample, use_ucb=use_ucb)
    except TypeError:
        return selector.select(sample)


def _selector_decision_details(selector, sample, decision_mode: str) -> dict:
    """
    统一拿到一次决策的详细信息。

    如果 selector 自己实现了 decision_details，就优先用它；
    否则退化成“只有 selected_arm 的简单版本”。
    """
    use_ucb = decision_mode.lower() == "ucb"
    if isinstance(selector, LinUCB):
        return selector.decision_details(sample.linucb_context, mask=sample.feasible_mask, use_ucb=use_ucb)
    if hasattr(selector, "decision_details"):
        try:
            return selector.decision_details(sample, use_ucb=use_ucb)
        except TypeError:
            pass

    arm = int(_selector_pick(selector, sample, decision_mode=decision_mode))
    return {
        "selected_arm": arm,
        "selection_reason": "selector_only",
        "use_ucb": bool(use_ucb),
        "feasible_arms": [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)],
        "under_sampled_arms": [],
        "arm_details": [],
    }


def _selector_train_decision_details(selector, sample, decision_mode: str) -> dict:
    """
    训练阶段专用的决策接口。

    和 `_selector_decision_details()` 的区别是：
    - 这里允许 selector 在“完成一次真实选臂”时带上副作用
    - 例如 NeuralUCB 需要在选中 arm 后立即更新不确定性统计量 `U`

    因此：
    - 评测 / trace 复盘继续用无副作用版本
    - 训练主循环用这个有副作用版本
    """
    use_ucb = decision_mode.lower() == "ucb"
    if isinstance(selector, LinUCB):
        return selector.decision_details(sample.linucb_context, mask=sample.feasible_mask, use_ucb=use_ucb)
    if hasattr(selector, "select_with_details"):
        try:
            return selector.select_with_details(sample, use_ucb=use_ucb)
        except TypeError:
            pass
    return _selector_decision_details(selector, sample, decision_mode=decision_mode)


def _safe_number(value):
    """把 numpy / torch 风格的数字安全地转成 JSON 友好的 python 标量。"""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalize_result_record(record: dict | None) -> dict | None:
    """把 sample 里保存的 result record 规范化成 JSON 友好格式。"""
    if record is None:
        return None
    normalized = {}
    for key, value in record.items():
        normalized[key] = _safe_number(value)
    return normalized


def _trace_selected_result(record: dict | None) -> dict | None:
    """
    生成面向 trace 的精简版结果记录。

    用户复盘时最关心的是：
    - 这一步选中的 arm，对应原始结果记录是谁
    - bandit 实验实际采用的是哪一个 score

    因此 trace 里不再把所有可能的 score 字段都原样展开，
    而是只保留“当前真正采用的那一个”。

    例如可能输出成：
    - {"global_index": 6, "name": null, "aug_score": 19.59}
    - {"global_index": 6, "name": null, "score": 19.59}
    - {"global_index": 6, "name": null, "no_aug_score": 19.59}
    """
    normalized = _normalize_result_record(record)
    if normalized is None:
        return None

    score_key = None
    for candidate in ("aug_score", "score", "no_aug_score"):
        if normalized.get(candidate) is not None:
            score_key = candidate
            break
    if score_key is None and normalized.get("picked_score") is not None:
        score_key = "score"

    compact = {
        "global_index": normalized.get("global_index"),
        "name": normalized.get("name"),
    }
    if score_key is not None:
        score_value = normalized.get(score_key)
        if score_value is None:
            score_value = normalized.get("picked_score")
        compact[score_key] = score_value
    return compact


def _materialize_trace_row(sample, decision: dict, arm_names: list[str], stage: str, decision_mode: str, order_index: int, epoch: int | None = None) -> dict:
    """
    把一次 decision_details 转成更适合 jsonl 落盘的一整行记录。
    """
    selected_arm = int(decision["selected_arm"])
    feasible_mask = np.asarray(sample.feasible_mask) > 0
    best_cost = float(np.nanmin(sample.costs[feasible_mask]))
    selected_cost = float(sample.costs[selected_arm])
    selected_reward = float(sample.rewards[selected_arm])
    selected_result = None
    if hasattr(sample, "result_records") and sample.result_records is not None:
        if 0 <= selected_arm < len(sample.result_records):
            selected_result = _trace_selected_result(sample.result_records[selected_arm])
    selected_detail = None
    arm_details = []

    for detail in decision.get("arm_details", []):
        arm_id = int(detail["arm_id"])
        normalized = {
            "arm_id": arm_id,
            "arm_name": arm_names[arm_id],
            "feasible": bool(detail["feasible"]),
            "pull_count": int(detail["pull_count"]),
            "mean": _safe_number(detail["mean"]),
            "bonus": _safe_number(detail["bonus"]),
            "score": _safe_number(detail["score"]),
        }
        arm_details.append(normalized)
        if arm_id == selected_arm:
            selected_detail = normalized

    if selected_detail is None:
        selected_detail = {
            "arm_id": selected_arm,
            "arm_name": arm_names[selected_arm],
            "feasible": True,
            "pull_count": None,
            "mean": None,
            "bonus": None,
            "score": None,
        }

    return {
        "stage": stage,
        "decision_mode": decision_mode,
        "epoch": None if epoch is None else int(epoch),
        "order_index": int(order_index),
        "uid": sample.uid,
        "problem": sample.problem,
        "global_index": int(sample.global_index),
        "selected_arm": selected_arm,
        "selected_arm_name": arm_names[selected_arm],
        "selection_reason": decision.get("selection_reason", ""),
        "selected_mean": selected_detail["mean"],
        "selected_bonus": selected_detail["bonus"],
        "selected_score": selected_detail["score"],
        "selected_reward": selected_reward,
        "selected_cost": selected_cost,
        "selected_result": selected_result,
        "best_arm": int(sample.best_arm),
        "best_arm_name": arm_names[int(sample.best_arm)],
        "best_cost": best_cost,
        "regret": selected_cost - best_cost,
        "top1": 1.0 if selected_arm == int(sample.best_arm) else 0.0,
        "feasible_arms": [int(arm) for arm in decision.get("feasible_arms", [])],
        "under_sampled_arms": [int(arm) for arm in decision.get("under_sampled_arms", [])],
        "arm_details": arm_details,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """把 trace 行写成 jsonl。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class JsonlTraceStreamer:
    """
    实时 jsonl trace 写入器。

    作用：
    - 训练/评测过程中，每产生一条 trace 就立刻写到磁盘
    - 即使程序中途被打断，已经产生的轨迹也不会丢
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._fp = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 行缓冲，尽量做到“写一行就立刻落盘”
            self._fp = path.open("w", encoding="utf-8", buffering=1)

    def write(self, row: dict) -> None:
        if self._fp is None:
            return
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def _periodic_eval_row(global_step: int, epoch: int, decision_mode: str, metrics: dict) -> dict:
    """
    生成训练过程中周期性验证的落盘记录。

    这里单独存一份 jsonl，而不是混进最终 summary：
    - summary 仍然只保留“训练结束后的最终结果”
    - 周期性 val 更像训练曲线 / 监控信息
    """
    return {
        "global_step": int(global_step),
        "epoch": int(epoch) + 1,
        "decision_mode": str(decision_mode),
        "overall": metrics.get("overall", {}),
        "by_problem": metrics.get("by_problem", {}),
        "macro_average": metrics.get("macro_average", {}),
    }


def _maybe_print_trace_row(row: dict, enabled: bool = False, every: int = 1) -> None:
    """
    可选地把每一步详细 trace 直接打印到终端。

    默认不打开，因为全量实验时输出会非常多。
    用户如果明确要“边跑边看每一步选了什么”，可以打开这个选项。
    """
    if not enabled:
        return
    every = max(1, int(every))
    order_index = int(row.get("order_index", 0))
    if order_index % every != 0:
        return
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _record_trace_row(
    row: dict,
    trace_rows: list[dict] | None = None,
    streamer: JsonlTraceStreamer | None = None,
    print_step_json: bool = False,
    step_log_every: int = 1,
) -> None:
    """
    统一处理一条 trace：
    - 放进内存列表
    - 实时追加到 jsonl
    - 需要时打印到终端
    """
    if trace_rows is not None:
        trace_rows.append(row)
    if streamer is not None:
        streamer.write(row)
    _maybe_print_trace_row(row, enabled=print_step_json, every=step_log_every)


def _stage_display_name(stage: str) -> str:
    """把内部 stage 名字转成更易读的中文。"""
    stage_lower = str(stage).lower()
    if stage_lower == "val":
        return "val"
    if stage_lower == "test":
        return "test"
    if stage_lower == "train":
        return "train"
    return stage


def _problem_display_name(problem: str) -> str:
    if str(problem).lower() == "tsp":
        return "TSP"
    if str(problem).lower() == "cvrp":
        return "CVRP"
    if str(problem).upper() == "ALL":
        return "ALL"
    return str(problem).upper()


def _safe_metric_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _format_rate_with_hits(rate: float, hits: int, total: int) -> str:
    return f"{rate:8.4f}  ({hits}/{total})"


def _aggregate_rank_rows(trace_rows: list[dict], reward_mode: str) -> dict[str, dict]:
    """
    基于 trace 计算 tie-aware mean rank / top-k 命中率。

    这里和 analysis notebook 的口径保持一致：
    - rank 由 selected_reward 反推
    - top1/top2/top3 使用 tie-aware 平均 rank
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        grouped[row["problem"]].append(row)

    output: dict[str, dict] = {}
    for scope, rows in [("ALL", trace_rows), *sorted(grouped.items())]:
        if not rows:
            continue
        ranks = []
        for row in rows:
            feasible_count = len(row.get("feasible_arms", []))
            rank = infer_rank_from_reward(row["selected_reward"], feasible_count, reward_mode=reward_mode)
            ranks.append(float(rank))
        total = len(rows)
        top1_hits = sum(1 for rank in ranks if rank <= 1.0 + 1e-12)
        top2_hits = sum(1 for rank in ranks if rank <= 2.0 + 1e-12)
        top3_hits = sum(1 for rank in ranks if rank <= 3.0 + 1e-12)
        output[scope] = {
            "count": total,
            "mean_rank": float(np.mean(ranks)),
            "top1_hit_rate": top1_hits / total,
            "top2_hit_rate": top2_hits / total,
            "top3_hit_rate": top3_hits / total,
            "top1_hits": int(top1_hits),
            "top2_hits": int(top2_hits),
            "top3_hits": int(top3_hits),
            "oracle_cost": float(np.mean([float(row["best_cost"]) for row in rows])),
        }
    return output


def _aggregate_selection_distribution(trace_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        grouped[row["problem"]].append(row)

    output: dict[str, list[dict]] = {}
    for scope, rows in [("ALL", trace_rows), *sorted(grouped.items())]:
        if not rows:
            continue
        total = len(rows)
        counts = defaultdict(int)
        for row in rows:
            counts[(int(row["selected_arm"]), str(row["selected_arm_name"]))] += 1
        scope_rows = []
        for (arm_id, arm_name), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1])):
            scope_rows.append(
                {
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "count": int(count),
                    "fraction": count / total,
                }
            )
        output[scope] = scope_rows
    return output


def _aggregate_selected_arm_quality(trace_rows: list[dict], reward_mode: str) -> dict[str, list[dict]]:
    grouped_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        grouped_by_problem[row["problem"]].append(row)

    output: dict[str, list[dict]] = {}
    for scope, rows in sorted(grouped_by_problem.items()):
        by_arm: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for row in rows:
            by_arm[(int(row["selected_arm"]), str(row["selected_arm_name"]))].append(row)
        scope_rows = []
        for (arm_id, arm_name), arm_rows in sorted(by_arm.items(), key=lambda item: item[0][0]):
            ranks = []
            for row in arm_rows:
                feasible_count = len(row.get("feasible_arms", []))
                rank = infer_rank_from_reward(row["selected_reward"], feasible_count, reward_mode=reward_mode)
                ranks.append(float(rank))
            count = len(arm_rows)
            scope_rows.append(
                {
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "count": count,
                    "mean_cost": float(np.mean([float(row["selected_cost"]) for row in arm_rows])),
                    "mean_reward": float(np.mean([float(row["selected_reward"]) for row in arm_rows])),
                    "top1_rate": float(np.mean([1.0 if rank <= 1.0 + 1e-12 else 0.0 for rank in ranks])),
                    "mean_rank": float(np.mean(ranks)),
                }
            )
        output[scope] = scope_rows
    return output


def _aggregate_oracle_best_distribution(trace_rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        grouped[row["problem"]].append(row)

    output: dict[str, list[dict]] = {}
    for scope, rows in sorted(grouped.items()):
        total = len(rows)
        counts = defaultdict(int)
        for row in rows:
            counts[(int(row["best_arm"]), str(row["best_arm_name"]))] += 1
        scope_rows = []
        for (arm_id, arm_name), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0])):
            scope_rows.append(
                {
                    "arm_id": arm_id,
                    "arm_name": arm_name,
                    "count": int(count),
                    "fraction": count / total,
                }
            )
        output[scope] = scope_rows
    return output


def _log_evaluation_report(
    *,
    selector_name: str,
    decision_mode: str,
    metrics: dict,
    trace_rows: list[dict],
    baselines: dict | None,
    reward_mode: str,
    enabled: bool = True,
) -> None:
    """
    输出一份更适合人眼复盘的 test 评测摘要。

    目的：
    - 跑完 test 后，不需要立刻再开 notebook 才能看大概效果
    - 在 run2.log 里就能直接看到整体质量、baseline 对比、选臂分布等摘要
    """
    if not enabled or not trace_rows:
        return

    rank_stats = _aggregate_rank_rows(trace_rows, reward_mode=reward_mode)
    selection_distribution = _aggregate_selection_distribution(trace_rows)
    selected_arm_quality = _aggregate_selected_arm_quality(trace_rows, reward_mode=reward_mode)
    oracle_distribution = _aggregate_oracle_best_distribution(trace_rows)

    lines = []
    lines.append("=" * 70)
    lines.append(f"  评测结果 | {selector_name} | mode={decision_mode}")
    lines.append("=" * 70)
    lines.append("")

    scope_order = ["ALL", "tsp", "cvrp"]
    for scope in scope_order:
        key = scope if scope == "ALL" else scope
        metric_block = metrics["overall"] if scope == "ALL" else metrics["by_problem"].get(scope)
        rank_block = rank_stats.get("ALL" if scope == "ALL" else scope)
        if metric_block is None or rank_block is None:
            continue
        count = int(rank_block["count"])
        lines.append(f"  {_problem_display_name(scope)} ({count} 实例):")
        lines.append(f"    mean_reward:    {_safe_metric_float(metric_block['mean_reward']):.4f}")
        lines.append(f"    mean_rank:      {float(rank_block['mean_rank']):.2f}")
        lines.append(
            f"    top1_hit_rate:  {_format_rate_with_hits(float(rank_block['top1_hit_rate']), int(rank_block['top1_hits']), count)}"
        )
        lines.append(
            f"    top2_hit_rate:  {_format_rate_with_hits(float(rank_block['top2_hit_rate']), int(rank_block['top2_hits']), count)}"
        )
        lines.append(
            f"    top3_hit_rate:  {_format_rate_with_hits(float(rank_block['top3_hit_rate']), int(rank_block['top3_hits']), count)}"
        )
        lines.append(f"    regret:         {_safe_metric_float(metric_block['mean_regret']):.4f}")
        lines.append(f"    mean_cost:      {_safe_metric_float(metric_block['mean_cost']):.4f}  (模型选择)")
        lines.append(f"    oracle_cost:    {float(rank_block['oracle_cost']):.4f}  (每个实例选最好的)")
        lines.append("")

    if baselines:
        lines.append("=" * 70)
        lines.append("  Baseline 对比 (在同一 test 集上)")
        lines.append("=" * 70)
        lines.append("")
        for scope in ["tsp", "cvrp"]:
            selector_metrics = metrics["by_problem"].get(scope)
            if selector_metrics is None:
                continue
            lines.append(f"  {_problem_display_name(scope)} ({int(selector_metrics['count'])} 实例):")
            oracle_metrics = baselines.get("oracle", {}).get("by_problem", {}).get(scope, {})
            random_metrics = baselines.get("random", {}).get("by_problem", {}).get(scope, {})
            sb_global_metrics = baselines.get("single_best_global", {}).get("by_problem", {}).get(scope, {})
            sb_problem_metrics = baselines.get("single_best_per_problem", {}).get("by_problem", {}).get(scope, {})
            if oracle_metrics:
                lines.append(f"    oracle mean_cost:                {_safe_metric_float(oracle_metrics.get('mean_cost')):.4f}  (每实例选最好的, 上界)")
            lines.append(f"    模型选择 mean_cost:              {_safe_metric_float(selector_metrics.get('mean_cost')):.4f}")
            if sb_problem_metrics:
                lines.append(f"    single_best_per_problem:         {_safe_metric_float(sb_problem_metrics.get('mean_cost')):.4f}")
            if sb_global_metrics:
                lines.append(f"    single_best_global:              {_safe_metric_float(sb_global_metrics.get('mean_cost')):.4f}")
            if random_metrics:
                lines.append(f"    random mean_cost:                {_safe_metric_float(random_metrics.get('mean_cost')):.4f}  (均匀随机的期望)")
            lines.append("")

    lines.append("=" * 70)
    lines.append("  各臂选择分布")
    lines.append("=" * 70)
    lines.append("")
    for scope in ["tsp", "cvrp", "ALL"]:
        rows = selection_distribution.get(scope)
        if not rows:
            continue
        total = sum(int(row["count"]) for row in rows)
        lines.append(f"  {_problem_display_name(scope)} ({total} 实例):")
        for row in rows:
            lines.append(
                f"    {int(row['arm_id']):2d}({str(row['arm_name']):16s}): {int(row['count']):5d} 次 ({float(row['fraction']) * 100:5.1f}%)"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append("  各臂被选后的真实效果")
    lines.append("=" * 70)
    lines.append("")
    for scope in ["tsp", "cvrp"]:
        rows = selected_arm_quality.get(scope)
        if not rows:
            continue
        lines.append(f"  {_problem_display_name(scope)}:")
        lines.append("    arm                   count  mean_cost  mean_reward  top1_rate  mean_rank")
        lines.append("    " + "─" * 68)
        for row in rows:
            label = f"{int(row['arm_id']):2d}({str(row['arm_name']):16s})"
            lines.append(
                f"    {label:22s} {int(row['count']):6d}  {float(row['mean_cost']):9.4f}  {float(row['mean_reward']):11.4f}  {float(row['top1_rate']):9.4f}  {float(row['mean_rank']):9.2f}"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append("  Oracle 最优臂分布 (test 集上哪个方法最常是最优的)")
    lines.append("=" * 70)
    lines.append("")
    for scope in ["tsp", "cvrp"]:
        rows = oracle_distribution.get(scope)
        if not rows:
            continue
        total = sum(int(row["count"]) for row in rows)
        lines.append(f"  {_problem_display_name(scope)} ({total} 实例):")
        for row in rows:
            lines.append(
                f"    {int(row['arm_id']):2d}({str(row['arm_name']):16s}): {int(row['count']):5d} 次 ({float(row['fraction']) * 100:5.1f}%)"
            )
        lines.append("")

    _log("\n" + "\n".join(lines), enabled=enabled)


def _configure_selector_detail_logger(method: str, output_dir: Path | None) -> tuple[logging.Logger | None, Path | None]:
    """
    给需要详细文本日志的 selector 配置一个独立 logger。

    当前主要给：
    - neural_ucb_diag
    - neural_linucb

    使用方式尽量接近旧版 2cmab 的风格：
    - 详细日志始终写文件
    - 不默认刷到终端，避免打爆进度条
    """
    if output_dir is None:
        return None, None

    logger_name = None
    if method == "neural_ucb_diag":
        logger_name = "neural_ucb_diag"
    elif method == "neural_linucb":
        logger_name = "neural_linucb"

    if logger_name is None:
        return None, None

    detail_logger = logging.getLogger(logger_name)
    detail_logger.setLevel(logging.DEBUG)
    detail_logger.handlers.clear()
    detail_logger.propagate = False

    log_path = Path(output_dir) / f"{method}.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    detail_logger.addHandler(handler)
    return detail_logger, log_path


def _log_decision_text(
    detail_logger: logging.Logger | None,
    sample,
    decision: dict,
    arm_names: list[str],
    *,
    stage: str,
    order_index: int,
    epoch: int | None = None,
) -> None:
    """
    把一次 decision 额外写成更易读的文本日志。

    这样除了 jsonl trace 外，还会有一份更接近旧版 `neural_linucb` 的
    “每个 arm 的 mean / bonus / score 表格”文本文件。
    """
    if detail_logger is None:
        return

    selected_arm = int(decision["selected_arm"])
    selected_name = arm_names[selected_arm]
    prefix = f"{stage.upper()}[{order_index}]"
    if epoch is not None:
        prefix = f"epoch={epoch + 1} | {prefix}"

    header = (
        f"{prefix} | {sample.problem.upper()} | "
        f"select arm={selected_arm}({selected_name}) | "
        f"reason={decision.get('selection_reason', '')}"
    )
    lines = [header]
    for arm_detail in decision.get("arm_details", []):
        if not arm_detail.get("feasible", False):
            continue
        arm_id = int(arm_detail["arm_id"])
        marker = " <--" if arm_id == selected_arm else ""
        mean = arm_detail.get("mean")
        bonus = arm_detail.get("bonus")
        score = arm_detail.get("score")
        raw_score = arm_detail.get("raw_score")
        safety_penalty = arm_detail.get("safety_penalty")
        problem_arm_mean_reward = arm_detail.get("problem_arm_mean_reward")
        pull_count = arm_detail.get("pull_count")
        mean_text = "None" if mean is None else f"{float(mean):+.4f}"
        bonus_text = "None" if bonus is None else f"{float(bonus):.4f}"
        score_text = "None" if score is None else f"{float(score):+.4f}"
        raw_score_text = "" if raw_score is None else f" | raw={float(raw_score):+.4f}"
        safety_text = "" if safety_penalty in (None, 0, 0.0) else f" | safety={float(safety_penalty):.4f}"
        problem_reward_text = (
            ""
            if problem_arm_mean_reward is None
            else f" | p_mean_reward={float(problem_arm_mean_reward):.4f}"
        )
        lines.append(
            f"    {arm_id:2d}({arm_names[arm_id]:16s}): "
            f"pulls={int(pull_count):4d} | mean={mean_text} | "
            f"bonus={bonus_text} | score={score_text}{raw_score_text}{safety_text}{problem_reward_text}{marker}"
        )
    detail_logger.info("\n".join(lines))


def _selector_checkpoint_state(selector) -> dict:
    """
    提取 selector 当前可落盘的 checkpoint 状态。

    当前设计目标是：
    - 够你后续做复盘或推理使用
    - 也尽量保留主要 bandit 统计量
    - 但不把特别大的历史数据一股脑全塞进去

    这里暂时没有做“完整恢复训练”的承诺，
    重点是保存“当前训练到这个 epoch 时，selector 学到了什么”。
    """
    if isinstance(selector, LinUCB):
        return {
            "selector_class": selector.__class__.__name__,
            "n_arms": int(selector.n_arms),
            "context_dim": int(selector.context_dim),
            "alpha": float(selector.alpha),
            "initial_pulls": int(selector.initial_pulls),
            "A_inv": selector.A_inv,
            "b": selector.b,
            "arm_counts": selector.arm_counts,
        }

    if isinstance(selector, NeuralLinUCB):
        return {
            "selector_class": selector.__class__.__name__,
            "n_arms": int(selector.n_arms),
            "hidden_dim": int(selector.hidden_dim),
            "alpha": float(selector.alpha),
            "reg": float(selector.reg),
            "train_every": int(selector.train_every),
            "representation_steps": int(selector.representation_steps),
            "representation_buffer_size": int(selector.representation_buffer_size),
            "representation_batch_size": int(selector.representation_batch_size),
            "representation_balance_mode": str(selector.representation_balance_mode),
            "representation_balance_power": float(selector.representation_balance_power),
            "representation_loss_reweight": bool(selector.representation_loss_reweight),
            "linear_head_buffer_size": int(selector.linear_head_buffer_size),
            "initial_pulls": int(selector.initial_pulls),
            "train_epsilon": float(selector.train_epsilon),
            "problem_specific_heads": bool(selector.problem_specific_heads),
            "model_state_dict": selector.state_dict(),
            "optimizer_state_dict": selector.optimizer.state_dict(),
            "problem_head_state": selector.export_problem_head_state(),
            "arm_counts": selector.arm_counts,
            "arm_reward_sums": selector.arm_reward_sums,
            "total_steps": int(selector.total_steps),
            "representation_history_size": len(selector._representation_records()),
            "linear_head_history_size": len(selector._linear_head_records()),
        }

    if isinstance(selector, NeuralUCBDiag):
        return {
            "selector_class": selector.__class__.__name__,
            "n_arms": int(selector.n_arms),
            "nu": float(selector.nu),
            "lamdba": float(selector.lamdba),
            "arm_embed_dim": int(selector.arm_embed_dim),
            "use_arm_onehot": bool(selector.use_arm_onehot),
            "use_arm_bias": bool(selector.use_arm_bias),
            "train_every": int(selector.train_every),
            "train_batch_size": int(selector.train_batch_size),
            "train_steps_per_update": int(selector.train_steps_per_update),
            "initial_pulls": int(selector.initial_pulls),
            "use_problem_arm_safety": bool(selector.use_problem_arm_safety),
            "safety_min_pulls": int(selector.safety_min_pulls),
            "safety_reward_gap": float(selector.safety_reward_gap),
            "safety_penalty_scale": float(selector.safety_penalty_scale),
            "safety_penalty_cap": float(selector.safety_penalty_cap),
            "model_state_dict": selector.state_dict(),
            "optimizer_state_dict": selector.optimizer.state_dict(),
            "U": selector.U.detach().cpu(),
            "arm_counts": selector.arm_counts,
            "arm_reward_sums": selector.arm_reward_sums,
            "problem_arm_counts": dict(selector.problem_arm_counts),
            "problem_arm_reward_sums": dict(selector.problem_arm_reward_sums),
            "total_steps": int(selector.total_steps),
            "history_size": len(selector.history),
        }

    raise TypeError(f"暂不支持保存该 selector 的 checkpoint: {selector.__class__.__name__}")


def _save_selector_checkpoint(
    output_dir: Path,
    selector,
    config: dict,
    epoch: int,
    global_step: int | None,
    arm_names: list[str],
    save_reason: str,
    update_latest_alias: bool = True,
) -> Path:
    """
    保存一个 selector checkpoint。

    文件内容主要包括：
    - 当前 epoch
    - 当前配置
    - 当前动作空间名字
    - selector 的主要参数与 bandit 状态

    当前约定：
    - epoch 周期保存：`checkpoints/epoch_0001.pt`
    - step 周期保存：`checkpoints/step_0001000.pt`
    - 最后一次补保存：`checkpoints/final.pt`
    - 最近一次保存的别名：`checkpoints/latest.pt`
    """
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epoch_one_based = int(epoch) + 1
    global_step_value = None if global_step is None else int(global_step)
    if save_reason == "periodic_epoch":
        checkpoint_path = checkpoint_dir / f"epoch_{epoch_one_based:04d}.pt"
    elif save_reason == "periodic_step":
        if global_step_value is None:
            raise ValueError("step 级 checkpoint 需要传入 global_step")
        checkpoint_path = checkpoint_dir / f"step_{global_step_value:07d}.pt"
    elif save_reason == "best":
        checkpoint_path = checkpoint_dir / "best.pt"
    else:
        checkpoint_path = checkpoint_dir / "final.pt"

    payload = {
        "format_version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": epoch_one_based,
        "global_step": global_step_value,
        "save_reason": save_reason,
        "method": config.get("method"),
        "arm_names": list(arm_names),
        "config": dict(config),
        "selector_state": _selector_checkpoint_state(selector),
    }
    torch.save(payload, checkpoint_path)
    if update_latest_alias:
        torch.save(payload, checkpoint_dir / "latest.pt")
    return checkpoint_path


def _load_selector_checkpoint_state(selector, selector_state: dict) -> None:
    """
    把 checkpoint 里的 selector_state 恢复到当前 selector 对象上。

    这里当前主要用于：
    - 训练结束后恢复 best checkpoint
    - 再用 best 状态做最终 val/test

    注意：
    - 这里只恢复“推理 / 评测所需”的关键状态
    - 不承诺完整恢复 history 后继续训练
    """
    if isinstance(selector, LinUCB):
        selector.A_inv = np.asarray(selector_state["A_inv"], dtype=np.float64)
        selector.b = np.asarray(selector_state["b"], dtype=np.float64)
        selector.arm_counts = np.asarray(selector_state["arm_counts"], dtype=np.int64)
        return

    if isinstance(selector, NeuralLinUCB):
        selector.load_state_dict(selector_state["model_state_dict"])
        if selector_state.get("optimizer_state_dict") is not None:
            selector.optimizer.load_state_dict(selector_state["optimizer_state_dict"])
        selector.problem_specific_heads = bool(
            selector_state.get("problem_specific_heads", getattr(selector, "problem_specific_heads", True))
        )
        if selector_state.get("problem_head_state") is not None:
            selector.load_problem_head_state(selector_state["problem_head_state"])
        else:
            # 兼容旧 checkpoint：把旧版单头复制到当前默认 head 上。
            selector.load_problem_head_state(None)
            legacy_head = {
                "A_inv": np.asarray(selector_state["A_inv"], dtype=np.float64),
                "b": np.asarray(selector_state["b"], dtype=np.float64),
                "theta": np.asarray(selector_state["theta"], dtype=np.float64),
            }
            for key in selector.problem_heads.keys():
                selector.problem_heads[key] = {
                    "A_inv": legacy_head["A_inv"].copy(),
                    "b": legacy_head["b"].copy(),
                    "theta": legacy_head["theta"].copy(),
                }
        selector.arm_counts = np.asarray(selector_state["arm_counts"], dtype=np.int64)
        selector.arm_reward_sums = np.asarray(
            selector_state.get("arm_reward_sums", np.zeros(selector.n_arms, dtype=np.float64)),
            dtype=np.float64,
        )
        selector.total_steps = int(selector_state.get("total_steps", 0))
        selector.representation_history.clear()
        selector.linear_head_history.clear()
        selector.eval()
        return

    if isinstance(selector, NeuralUCBDiag):
        selector.load_state_dict(selector_state["model_state_dict"])
        if selector_state.get("optimizer_state_dict") is not None:
            selector.optimizer.load_state_dict(selector_state["optimizer_state_dict"])
        selector.U = torch.as_tensor(selector_state["U"], dtype=torch.float32, device=selector.device).clone()
        selector.arm_counts = np.asarray(selector_state["arm_counts"], dtype=np.int64)
        selector.arm_reward_sums = np.asarray(
            selector_state.get("arm_reward_sums", np.zeros(selector.n_arms, dtype=np.float64)),
            dtype=np.float64,
        )
        selector.problem_arm_counts = {
            (str(problem), int(arm)): int(count)
            for (problem, arm), count in selector_state.get("problem_arm_counts", {}).items()
        }
        selector.problem_arm_reward_sums = {
            (str(problem), int(arm)): float(reward_sum)
            for (problem, arm), reward_sum in selector_state.get("problem_arm_reward_sums", {}).items()
        }
        selector.total_steps = int(selector_state.get("total_steps", 0))
        selector.history.clear()
        selector.eval()
        return

    raise TypeError(f"暂不支持恢复该 selector 的 checkpoint: {selector.__class__.__name__}")


def _restore_selector_from_checkpoint(checkpoint_path: Path, selector) -> dict:
    """从磁盘读取 checkpoint，并恢复到当前 selector。"""
    payload = torch.load(checkpoint_path, map_location="cpu")
    selector_state = payload["selector_state"]
    _load_selector_checkpoint_state(selector, selector_state)
    return payload


def _extract_nested_metric(metrics: dict, dotted_path: str):
    """
    从嵌套字典里按 `a.b.c` 形式取值。

    例如：
    - dotted_path = "val_greedy.overall.mean_cost"
    - metrics = {"val_greedy": {"overall": {"mean_cost": 12.3}}}
    """
    current = metrics
    for part in str(dotted_path).split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_better_metric(candidate_value, best_value, mode: str) -> bool:
    """
    比较一个新指标是否优于当前 best。

    mode:
    - min: 越小越好
    - max: 越大越好
    """
    if candidate_value is None:
        return False
    try:
        candidate = float(candidate_value)
    except Exception:
        return False
    if not np.isfinite(candidate):
        return False
    if best_value is None:
        return True
    best = float(best_value)
    if str(mode).lower() == "max":
        return candidate > best
    return candidate < best


def evaluate_selector(
    selector,
    samples,
    indices: Iterable[int],
    decision_mode: str = "greedy",
    arm_names: list[str] | None = None,
    trace_rows: list[dict] | None = None,
    trace_streamer: JsonlTraceStreamer | None = None,
    stage: str = "",
    show_progress: bool = False,
    log_progress: bool = False,
    progress_log_every: int = 500,
    print_step_json: bool = False,
    step_log_every: int = 1,
    detail_logger: logging.Logger | None = None,
):
    """
    在指定样本子集上评测一个 selector。

    当前输出四个核心指标：
    - mean_reward
    - top1_accuracy
    - mean_regret
    - mean_cost

    同时分别统计：
    - overall
    - by_problem
    - macro_average
    """
    rows = []
    indices = list(indices)
    total_count = len(indices)
    progress_log_every = max(1, int(progress_log_every))
    if log_progress:
        _log(
            f"在 {_stage_display_name(stage or 'eval')} 集上评测 ({total_count} 实例, mode={decision_mode})...",
            enabled=True,
        )
    progress = _iter_with_progress(
        enumerate(indices),
        total=len(indices),
        desc=f"{stage or 'eval'}:{decision_mode}",
        enabled=show_progress,
    )
    for order_index, idx in progress:
        sample = samples[idx]
        decision = _selector_decision_details(selector, sample, decision_mode=decision_mode)
        if arm_names is not None:
            _log_decision_text(
                detail_logger,
                sample,
                decision,
                arm_names,
                stage=stage or f"selector_{decision_mode}",
                order_index=order_index,
            )
        arm = int(decision["selected_arm"])
        cost = float(sample.costs[arm])
        best_cost = float(np.nanmin(sample.costs[sample.feasible_mask > 0]))
        regret = cost - best_cost
        reward = float(sample.rewards[arm])
        rows.append(
            {
                "problem": sample.problem,
                "reward": reward,
                "cost": cost,
                "regret": regret,
                "top1": 1.0 if arm == sample.best_arm else 0.0,
            }
        )
        if arm_names is not None and (trace_rows is not None or trace_streamer is not None):
            trace_row = _materialize_trace_row(
                sample=sample,
                decision=decision,
                arm_names=arm_names,
                stage=stage or f"selector_{decision_mode}",
                decision_mode=decision_mode,
                order_index=order_index,
            )
            _record_trace_row(
                trace_row,
                trace_rows=trace_rows,
                streamer=trace_streamer,
                print_step_json=print_step_json,
                step_log_every=step_log_every,
            )

        current = order_index + 1
        if log_progress and (current % progress_log_every == 0 or current == total_count):
            _log(f"    已评测 {current}/{total_count}...", enabled=True)

    def _aggregate(part_rows):
        return {
            "count": len(part_rows),
            "mean_reward": float(np.mean([row["reward"] for row in part_rows])) if part_rows else float("nan"),
            "top1_accuracy": float(np.mean([row["top1"] for row in part_rows])) if part_rows else float("nan"),
            "mean_regret": float(np.mean([row["regret"] for row in part_rows])) if part_rows else float("nan"),
            "mean_cost": float(np.mean([row["cost"] for row in part_rows])) if part_rows else float("nan"),
        }

    by_problem = {}
    for problem in sorted({row["problem"] for row in rows}):
        by_problem[problem] = _aggregate([row for row in rows if row["problem"] == problem])

    macro = {
        "mean_reward": float(np.mean([metrics["mean_reward"] for metrics in by_problem.values()])) if by_problem else float("nan"),
        "top1_accuracy": float(np.mean([metrics["top1_accuracy"] for metrics in by_problem.values()])) if by_problem else float("nan"),
        "mean_regret": float(np.mean([metrics["mean_regret"] for metrics in by_problem.values()])) if by_problem else float("nan"),
        "mean_cost": float(np.mean([metrics["mean_cost"] for metrics in by_problem.values()])) if by_problem else float("nan"),
    }

    result = {
        "overall": _aggregate(rows),
        "by_problem": by_problem,
        "macro_average": macro,
    }
    if log_progress:
        overall = result["overall"]
        _log(
            f"{_stage_display_name(stage or 'eval')} 评测完成 | mode={decision_mode} | "
            f"mean_reward={_safe_metric_float(overall['mean_reward']):.6f} | "
            f"top1_accuracy={_safe_metric_float(overall['top1_accuracy']):.6f} | "
            f"mean_regret={_safe_metric_float(overall['mean_regret']):.6f} | "
            f"mean_cost={_safe_metric_float(overall['mean_cost']):.6f}",
            enabled=True,
        )
    return result


def _train_linucb(
    selector: LinUCB,
    samples,
    train_indices,
    epochs: int,
    seed: int,
    arm_names: list[str] | None = None,
    show_progress: bool = False,
    verbose: bool = True,
    trace_streamer: JsonlTraceStreamer | None = None,
    print_step_json: bool = False,
    step_log_every: int = 1,
    epoch_end_callback: Callable[[int, object], None] | None = None,
    step_end_callback: Callable[[int, int, object], None] | None = None,
):
    """LinUCB 的训练主循环。"""
    rng = np.random.default_rng(seed)
    trace_rows: list[dict] = []
    global_step = 0
    for epoch in range(epochs):
        _log(f"开始训练 epoch {epoch + 1}/{epochs} [linucb]", enabled=verbose)
        order = rng.permutation(train_indices).tolist()
        progress = _iter_with_progress(
            enumerate(order),
            total=len(order),
            desc=f"train epoch {epoch + 1}/{epochs}",
            enabled=show_progress,
        )
        for order_index, idx in progress:
            sample = samples[idx]
            decision = selector.decision_details(sample.linucb_context, mask=sample.feasible_mask, use_ucb=True)
            arm = int(decision["selected_arm"])
            if arm_names is not None:
                trace_row = _materialize_trace_row(
                    sample=sample,
                    decision=decision,
                    arm_names=arm_names,
                    stage="train",
                    decision_mode="ucb",
                    order_index=order_index,
                    epoch=epoch,
                )
                _record_trace_row(
                    trace_row,
                    trace_rows=trace_rows,
                    streamer=trace_streamer,
                    print_step_json=print_step_json,
                    step_log_every=step_log_every,
                )
            selector.update(arm, sample.linucb_context, float(sample.rewards[arm]))
            global_step += 1
            if step_end_callback is not None:
                step_end_callback(epoch, global_step, selector)
        _log(f"完成训练 epoch {epoch + 1}/{epochs} [linucb]", enabled=verbose)
        if epoch_end_callback is not None:
            epoch_end_callback(epoch, selector)
    return trace_rows


def _train_sample_bandit(
    selector,
    samples,
    train_indices,
    epochs: int,
    seed: int,
    arm_names: list[str] | None = None,
    show_progress: bool = False,
    verbose: bool = True,
    trace_streamer: JsonlTraceStreamer | None = None,
    print_step_json: bool = False,
    step_log_every: int = 1,
    epoch_end_callback: Callable[[int, object], None] | None = None,
    step_end_callback: Callable[[int, int, object], None] | None = None,
    detail_logger: logging.Logger | None = None,
):
    """Neural bandit 类算法的训练主循环。"""
    rng = np.random.default_rng(seed)
    trace_rows: list[dict] = []
    global_step = 0
    for epoch in range(epochs):
        _log(f"开始训练 epoch {epoch + 1}/{epochs} [{selector.__class__.__name__}]", enabled=verbose)
        order = rng.permutation(train_indices).tolist()
        progress = _iter_with_progress(
            enumerate(order),
            total=len(order),
            desc=f"train epoch {epoch + 1}/{epochs}",
            enabled=show_progress,
        )
        for order_index, idx in progress:
            sample = samples[idx]
            decision = _selector_train_decision_details(selector, sample, decision_mode="ucb")
            arm = int(decision["selected_arm"])
            if arm_names is not None:
                _log_decision_text(
                    detail_logger,
                    sample,
                    decision,
                    arm_names,
                    stage="train",
                    order_index=order_index,
                    epoch=epoch,
                )
            if arm_names is not None:
                trace_row = _materialize_trace_row(
                    sample=sample,
                    decision=decision,
                    arm_names=arm_names,
                    stage="train",
                    decision_mode="ucb",
                    order_index=order_index,
                    epoch=epoch,
                )
                _record_trace_row(
                    trace_row,
                    trace_rows=trace_rows,
                    streamer=trace_streamer,
                    print_step_json=print_step_json,
                    step_log_every=step_log_every,
                )
            selector.update(sample, arm, float(sample.rewards[arm]))
            global_step += 1
            if step_end_callback is not None:
                step_end_callback(epoch, global_step, selector)
        _log(f"完成训练 epoch {epoch + 1}/{epochs} [{selector.__class__.__name__}]", enabled=verbose)
        if epoch_end_callback is not None:
            epoch_end_callback(epoch, selector)
    return trace_rows


def build_selector(args, n_arms: int, arm_names: list[str] | None = None):
    """
    根据命令行参数构造具体算法对象。

    当前支持：
    - linucb
    - neural_linucb
    - neural_ucb_diag
    """
    if args.method == "linucb":
        return LinUCB(
            n_arms=n_arms,
            context_dim=14,
            alpha=args.alpha,
            reg=args.reg,
            initial_pulls=args.initial_pulls,
        )
    if args.method == "neural_linucb":
        selector = NeuralLinUCB(
            n_arms=n_arms,
            hidden_dim=args.hidden_dim,
            alpha=args.alpha,
            reg=args.reg,
            lr=args.lr,
            train_every=args.train_every,
            representation_steps=args.representation_steps,
            representation_buffer_size=args.representation_buffer_size,
            representation_batch_size=args.representation_batch_size,
            representation_balance_mode=args.representation_balance_mode,
            representation_balance_power=args.representation_balance_power,
            representation_loss_reweight=args.representation_loss_reweight,
            linear_head_buffer_size=args.linear_head_buffer_size,
            initial_pulls=args.initial_pulls,
            train_epsilon=args.train_epsilon,
            problem_specific_heads=args.problem_specific_heads,
            arm_names=list(arm_names) if arm_names is not None else list(ALL_METHODS[:n_arms]),
            device=args.device,
        )
        if args.freeze_encoder:
            if hasattr(selector, "freeze_graph_encoder"):
                selector.freeze_graph_encoder()
            else:
                selector.context_encoder.freeze_graph_encoder()
        return selector
    if args.method == "neural_ucb_diag":
        selector = NeuralUCBDiag(
            n_arms=n_arms,
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            use_arm_onehot=args.use_arm_onehot,
            use_arm_bias=args.use_arm_bias,
            lamdba=args.reg,
            nu=args.nu,
            lr=args.lr,
            train_every=args.train_every,
            train_batch_size=args.train_batch_size,
            train_steps_per_update=args.train_steps_per_update,
            initial_pulls=args.initial_pulls,
            use_problem_arm_safety=args.use_problem_arm_safety,
            safety_min_pulls=args.safety_min_pulls,
            safety_reward_gap=args.safety_reward_gap,
            safety_penalty_scale=args.safety_penalty_scale,
            safety_penalty_cap=args.safety_penalty_cap,
            arm_names=list(arm_names) if arm_names is not None else list(ALL_METHODS[:n_arms]),
            device=args.device,
        )
        if args.freeze_encoder:
            if hasattr(selector, "freeze_graph_encoder"):
                selector.freeze_graph_encoder()
            else:
                selector.context_encoder.freeze_graph_encoder()
        return selector
    raise ValueError(f"未知 method: {args.method}")


def run_experiment(args, output_dir: Path | None = None):
    """
    完整实验主流程。

    顺序如下：
    1. 构建共享数据集
    2. 做分层切分
    3. 构造 selector
    4. 在 train 上训练
    5. 在 val / test 上评测
    6. 跑 baseline 对比
    """
    trace_paths = None
    checkpoint_paths: list[str] = []
    latest_saved_epoch: int | None = None
    latest_saved_step: int | None = None
    best_checkpoint_path: str | None = None
    best_checkpoint_metric_value: float | None = None
    best_checkpoint_step: int | None = None
    trace_streamers: dict[str, JsonlTraceStreamer] = {}
    periodic_val_streamer: JsonlTraceStreamer | None = None
    periodic_val_path: Path | None = None
    detail_logger, detail_log_path = _configure_selector_detail_logger(args.method, output_dir)
    if output_dir is not None:
        trace_dir = Path(output_dir) / "traces"
        trace_paths = {
            "selector_train_ucb": trace_dir / "selector_train_ucb.jsonl",
            "selector_val_greedy": trace_dir / "selector_val_greedy.jsonl",
            "selector_test_greedy": trace_dir / "selector_test_greedy.jsonl",
            "selector_val_ucb": trace_dir / "selector_val_ucb.jsonl",
            "selector_test_ucb": trace_dir / "selector_test_ucb.jsonl",
        }
        trace_streamers = {name: JsonlTraceStreamer(path) for name, path in trace_paths.items()}
        periodic_val_path = Path(output_dir) / "periodic_val.jsonl"
        periodic_val_streamer = JsonlTraceStreamer(periodic_val_path)

    arm_names, problem_to_methods = get_arm_space_config(args.data_source)

    _log(
        f"开始构建共享数据集 | data_source={args.data_source} | arm数={len(arm_names)}",
        enabled=not args.quiet,
    )
    dataset = build_joint_dataset(
        results_dir=Path(args.results_dir),
        tsp_dataset_path=Path(args.tsp_dataset_path) if args.tsp_dataset_path else None,
        cvrp_dataset_path=Path(args.cvrp_dataset_path) if args.cvrp_dataset_path else None,
        data_source=args.data_source,
        nss_dataset_root=Path(args.nss_dataset_root) if args.nss_dataset_root else None,
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        reward_mode=args.reward_mode,
        max_samples_per_problem=args.max_samples_per_problem,
        seed=args.seed,
        log_fn=(lambda msg: _log(msg, enabled=not args.quiet)),
        show_progress=not args.disable_progress,
    )
    _log(f"数据集构建完成，样本总数={len(dataset.samples)}", enabled=not args.quiet)

    if dataset.predefined_splits is not None:
        train_idx = list(dataset.predefined_splits.get("train", []))
        val_idx = list(dataset.predefined_splits.get("val", []))
        test_idx = list(dataset.predefined_splits.get("test", []))
        _log(
            "检测到数据源自带 train/val/test split，直接使用预定义划分",
            enabled=not args.quiet,
        )
    else:
        _log("开始做 train/val/test 分层切分", enabled=not args.quiet)
        train_idx, val_idx, test_idx = stratified_split_indices(
            dataset.samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    _log(
        f"数据切分完成: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
        enabled=not args.quiet,
    )

    _log(f"开始构造 selector: {args.method}", enabled=not args.quiet)
    selector = build_selector(args, n_arms=len(dataset.arm_names), arm_names=dataset.arm_names)
    _log(f"selector 构造完成: {selector.__class__.__name__}", enabled=not args.quiet)

    epoch_end_callback = None
    step_end_callbacks: list[Callable[[int, int, object], None]] = []
    if output_dir is not None and int(args.checkpoint_every) > 0:
        _log(
            f"已开启 checkpoint 自动保存: 每 {int(args.checkpoint_every)} 个 epoch 保存一次",
            enabled=not args.quiet,
        )

        def _on_epoch_end(epoch: int, current_selector) -> None:
            nonlocal latest_saved_epoch, latest_saved_step
            epoch_one_based = int(epoch) + 1
            if epoch_one_based % int(args.checkpoint_every) != 0:
                return
            checkpoint_path = _save_selector_checkpoint(
                output_dir=Path(output_dir),
                selector=current_selector,
                config=vars(args),
                epoch=epoch,
                global_step=epoch_one_based * len(train_idx),
                arm_names=dataset.arm_names,
                save_reason="periodic_epoch",
            )
            checkpoint_paths.append(str(checkpoint_path))
            latest_saved_epoch = epoch_one_based
            latest_saved_step = epoch_one_based * len(train_idx)
            _log(f"checkpoint 已保存: {checkpoint_path}", enabled=not args.quiet)

        epoch_end_callback = _on_epoch_end

    if output_dir is not None and int(args.checkpoint_every_steps) > 0:
        _log(
            f"已开启 checkpoint 自动保存: 每 {int(args.checkpoint_every_steps)} 个 step 保存一次",
            enabled=not args.quiet,
        )

        def _on_step_end(epoch: int, global_step: int, current_selector) -> None:
            nonlocal latest_saved_step
            if global_step % int(args.checkpoint_every_steps) != 0:
                return
            checkpoint_path = _save_selector_checkpoint(
                output_dir=Path(output_dir),
                selector=current_selector,
                config=vars(args),
                epoch=epoch,
                global_step=global_step,
                arm_names=dataset.arm_names,
                save_reason="periodic_step",
            )
            checkpoint_paths.append(str(checkpoint_path))
            latest_saved_step = global_step
            _log(f"checkpoint 已保存: {checkpoint_path}", enabled=not args.quiet)

        step_end_callbacks.append(_on_step_end)

    if int(args.eval_every_steps) > 0:
        _log(
            f"已开启周期性验证: 每 {int(args.eval_every_steps)} 个 step 在 val 集评测一次",
            enabled=not args.quiet,
        )

        def _on_step_eval(epoch: int, global_step: int, current_selector) -> None:
            nonlocal best_checkpoint_path, best_checkpoint_metric_value, best_checkpoint_step
            if global_step % int(args.eval_every_steps) != 0:
                return

            periodic_metrics = {
                "val_greedy": evaluate_selector(
                    current_selector,
                    dataset.samples,
                    val_idx,
                    decision_mode="greedy",
                    show_progress=False,
                )
            }
            if args.method in {"linucb", "neural_linucb", "neural_ucb_diag"}:
                periodic_metrics["val_ucb"] = evaluate_selector(
                    current_selector,
                    dataset.samples,
                    val_idx,
                    decision_mode="ucb",
                    show_progress=False,
                )

            greedy_overall = periodic_metrics["val_greedy"]["overall"]
            log_message = (
                f"step={global_step} | periodic val_greedy | "
                f"mean_reward={float(greedy_overall['mean_reward']):.6f} | "
                f"top1={float(greedy_overall['top1_accuracy']):.6f} | "
                f"mean_regret={float(greedy_overall['mean_regret']):.6f} | "
                f"mean_cost={float(greedy_overall['mean_cost']):.6f}"
            )
            if "val_ucb" in periodic_metrics:
                ucb_overall = periodic_metrics["val_ucb"]["overall"]
                log_message += (
                    f" || val_ucb: mean_reward={float(ucb_overall['mean_reward']):.6f} | "
                    f"top1={float(ucb_overall['top1_accuracy']):.6f} | "
                    f"mean_regret={float(ucb_overall['mean_regret']):.6f} | "
                    f"mean_cost={float(ucb_overall['mean_cost']):.6f}"
                )
            _log(log_message, enabled=not args.quiet)

            if periodic_val_streamer is not None:
                periodic_row = {
                    "global_step": int(global_step),
                    "epoch": int(epoch) + 1,
                    "metrics": {
                        mode: _periodic_eval_row(global_step, epoch, mode, metrics)
                        for mode, metrics in periodic_metrics.items()
                    },
                }
                periodic_val_streamer.write(periodic_row)
            else:
                periodic_row = {
                    "global_step": int(global_step),
                    "epoch": int(epoch) + 1,
                    "metrics": {
                        mode: _periodic_eval_row(global_step, epoch, mode, metrics)
                        for mode, metrics in periodic_metrics.items()
                    },
                }

            if output_dir is not None and bool(args.use_best_checkpoint_for_final_eval):
                metric_value = _extract_nested_metric(periodic_row["metrics"], args.best_checkpoint_metric)
                if _is_better_metric(metric_value, best_checkpoint_metric_value, args.best_checkpoint_mode):
                    checkpoint_path = _save_selector_checkpoint(
                        output_dir=Path(output_dir),
                        selector=current_selector,
                        config=vars(args),
                        epoch=epoch,
                        global_step=global_step,
                        arm_names=dataset.arm_names,
                        save_reason="best",
                        update_latest_alias=False,
                    )
                    best_checkpoint_path = str(checkpoint_path)
                    best_checkpoint_metric_value = float(metric_value)
                    best_checkpoint_step = int(global_step)
                    if str(checkpoint_path) not in checkpoint_paths:
                        checkpoint_paths.append(str(checkpoint_path))
                    _log(
                        f"best checkpoint 已刷新: step={global_step} | "
                        f"metric={args.best_checkpoint_metric}={float(metric_value):.6f} | path={checkpoint_path}",
                        enabled=not args.quiet,
                    )

        step_end_callbacks.append(_on_step_eval)

    step_end_callback = None
    if step_end_callbacks:
        def _run_step_end_callbacks(epoch: int, global_step: int, current_selector) -> None:
            for callback in step_end_callbacks:
                callback(epoch, global_step, current_selector)

        step_end_callback = _run_step_end_callbacks

    if isinstance(selector, LinUCB):
        train_trace = _train_linucb(
            selector,
            dataset.samples,
            train_idx,
            epochs=args.epochs,
            seed=args.seed,
            arm_names=dataset.arm_names,
            show_progress=not args.disable_progress,
            verbose=not args.quiet,
            trace_streamer=trace_streamers.get("selector_train_ucb"),
            print_step_json=args.print_step_json,
            step_log_every=args.step_log_every,
            epoch_end_callback=epoch_end_callback,
            step_end_callback=step_end_callback,
        )
    else:
        train_trace = _train_sample_bandit(
            selector,
            dataset.samples,
            train_idx,
            epochs=args.epochs,
            seed=args.seed,
            arm_names=dataset.arm_names,
            show_progress=not args.disable_progress,
            verbose=not args.quiet,
            trace_streamer=trace_streamers.get("selector_train_ucb"),
            print_step_json=args.print_step_json,
            step_log_every=args.step_log_every,
            epoch_end_callback=epoch_end_callback,
            step_end_callback=step_end_callback,
            detail_logger=detail_logger,
        )

    total_train_steps = int(args.epochs) * len(train_idx)
    checkpoint_auto_enabled = int(args.checkpoint_every) > 0 or int(args.checkpoint_every_steps) > 0
    if output_dir is not None and checkpoint_auto_enabled and latest_saved_step != total_train_steps:
        final_checkpoint_path = _save_selector_checkpoint(
            output_dir=Path(output_dir),
            selector=selector,
            config=vars(args),
            epoch=int(args.epochs) - 1,
            global_step=total_train_steps,
            arm_names=dataset.arm_names,
            save_reason="final",
        )
        checkpoint_paths.append(str(final_checkpoint_path))
        _log(f"最终 checkpoint 已保存: {final_checkpoint_path}", enabled=not args.quiet)

    restored_best_for_final_eval = False
    if bool(args.use_best_checkpoint_for_final_eval):
        if best_checkpoint_path is not None:
            _restore_selector_from_checkpoint(Path(best_checkpoint_path), selector)
            restored_best_for_final_eval = True
            _log(
                f"最终评测前已恢复 best checkpoint: step={best_checkpoint_step} | "
                f"metric={args.best_checkpoint_metric}={float(best_checkpoint_metric_value):.6f} | path={best_checkpoint_path}",
                enabled=not args.quiet,
            )
        else:
            _log(
                "未找到可用的 best checkpoint，最终评测继续使用训练结束时的 selector 状态",
                enabled=not args.quiet,
            )

    val_greedy_trace: list[dict] = []
    test_greedy_trace: list[dict] = []
    val_ucb_trace: list[dict] = []
    test_ucb_trace: list[dict] = []

    _log("开始 selector 验证/测试评测", enabled=not args.quiet)
    results = {
        "config": vars(args),
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "checkpoint_selection": {
            "use_best_checkpoint_for_final_eval": bool(args.use_best_checkpoint_for_final_eval),
            "best_checkpoint_metric": str(args.best_checkpoint_metric),
            "best_checkpoint_mode": str(args.best_checkpoint_mode),
            "best_checkpoint_path": best_checkpoint_path,
            "best_checkpoint_step": None if best_checkpoint_step is None else int(best_checkpoint_step),
            "best_checkpoint_metric_value": (
                None if best_checkpoint_metric_value is None else float(best_checkpoint_metric_value)
            ),
            "restored_best_for_final_eval": bool(restored_best_for_final_eval),
        },
        "selector": {
            "val_greedy": evaluate_selector(
                selector,
                dataset.samples,
                val_idx,
                decision_mode="greedy",
                arm_names=dataset.arm_names,
                trace_rows=val_greedy_trace,
                trace_streamer=trace_streamers.get("selector_val_greedy"),
                stage="val",
                show_progress=not args.disable_progress,
                log_progress=not args.quiet,
                progress_log_every=args.eval_log_every,
                print_step_json=args.print_step_json,
                step_log_every=args.step_log_every,
                detail_logger=detail_logger,
            ),
            "test_greedy": evaluate_selector(
                selector,
                dataset.samples,
                test_idx,
                decision_mode="greedy",
                arm_names=dataset.arm_names,
                trace_rows=test_greedy_trace,
                trace_streamer=trace_streamers.get("selector_test_greedy"),
                stage="test",
                show_progress=not args.disable_progress,
                log_progress=not args.quiet,
                progress_log_every=args.eval_log_every,
                print_step_json=args.print_step_json,
                step_log_every=args.step_log_every,
                detail_logger=detail_logger,
            ),
        },
    }

    if args.method in {"linucb", "neural_linucb", "neural_ucb_diag"}:
        results["selector"]["val_ucb"] = evaluate_selector(
            selector,
            dataset.samples,
            val_idx,
            decision_mode="ucb",
            arm_names=dataset.arm_names,
            trace_rows=val_ucb_trace,
            trace_streamer=trace_streamers.get("selector_val_ucb"),
            stage="val",
            show_progress=not args.disable_progress,
            log_progress=not args.quiet,
            progress_log_every=args.eval_log_every,
            print_step_json=args.print_step_json,
            step_log_every=args.step_log_every,
            detail_logger=detail_logger,
        )
        results["selector"]["test_ucb"] = evaluate_selector(
            selector,
            dataset.samples,
            test_idx,
            decision_mode="ucb",
            arm_names=dataset.arm_names,
            trace_rows=test_ucb_trace,
            trace_streamer=trace_streamers.get("selector_test_ucb"),
            stage="test",
            show_progress=not args.disable_progress,
            log_progress=not args.quiet,
            progress_log_every=args.eval_log_every,
            print_step_json=args.print_step_json,
            step_log_every=args.step_log_every,
            detail_logger=detail_logger,
        )
    _log("selector 验证/测试评测完成", enabled=not args.quiet)

    _log("开始计算 baseline 对比", enabled=not args.quiet)
    baselines = {
        "random": RandomSelector(seed=args.seed),
        "single_best_global": SingleBestGlobalSelector.fit(dataset.samples, train_idx, dataset.arm_names),
        "single_best_per_problem": SingleBestPerProblemSelector.fit(dataset.samples, train_idx, dataset.arm_names),
        "oracle": OracleSelector(),
    }
    results["baselines"] = {
        name: evaluate_selector(selector_obj, dataset.samples, test_idx, decision_mode="greedy")
        for name, selector_obj in baselines.items()
    }
    _log("baseline 对比完成", enabled=not args.quiet)

    if not args.quiet:
        _log_evaluation_report(
            selector_name=selector.__class__.__name__,
            decision_mode="greedy",
            metrics=results["selector"]["test_greedy"],
            trace_rows=test_greedy_trace,
            baselines=results["baselines"],
            reward_mode=args.reward_mode,
            enabled=True,
        )
        if "test_ucb" in results["selector"]:
            _log_evaluation_report(
                selector_name=selector.__class__.__name__,
                decision_mode="ucb",
                metrics=results["selector"]["test_ucb"],
                trace_rows=test_ucb_trace,
                baselines=results["baselines"],
                reward_mode=args.reward_mode,
                enabled=True,
            )

    if output_dir is not None:
        _log(f"开始写出结果文件到: {output_dir}", enabled=not args.quiet)
        for streamer in trace_streamers.values():
            streamer.close()
        if periodic_val_streamer is not None:
            periodic_val_streamer.close()
        results["artifacts"] = {
            "trace_dir": str(Path(output_dir) / "traces"),
            "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
            "checkpoint_files": checkpoint_paths,
            "best_checkpoint": best_checkpoint_path,
            "periodic_val": None if periodic_val_path is None else str(periodic_val_path),
            "detail_log": None if detail_log_path is None else str(detail_log_path),
            **{name: str(path) for name, path in trace_paths.items()},
        }
        _log("结果文件写出完成", enabled=not args.quiet)
    return results


def _default_output_dir(method: str) -> Path:
    """默认输出目录：2cmab2/outputs/<timestamp>_<method>"""
    tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return Path(__file__).resolve().parent / "outputs" / f"{tag}_{method}"


def parse_args():
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(
        description="TSP/CVRP 初始化方法选择的共享 contextual bandit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=["linucb", "neural_linucb", "neural_ucb_diag"], default=RunOfflineDefaults.METHOD)
    parser.add_argument("--results-dir", default=RunOfflineDefaults.RESULTS_DIR)
    parser.add_argument(
        "--data-source",
        choices=["easynco_results", "nss"],
        default=RunOfflineDefaults.DATA_SOURCE,
        help="训练/评测使用的数据来源；`nss` 会直接使用 NSS 自带的 train/val/test split。",
    )
    parser.add_argument(
        "--nss-dataset-root",
        default=RunOfflineDefaults.NSS_DATASET_ROOT,
        help="NSS 数据集根目录，仅在 --data-source nss 时生效。",
    )
    parser.add_argument("--tsp-dataset-path", default=RunOfflineDefaults.TSP_DATASET_PATH)
    parser.add_argument("--cvrp-dataset-path", default=RunOfflineDefaults.CVRP_DATASET_PATH)
    parser.add_argument("--reward-mode", choices=["linear_zero_one", "autosaea"], default=RunOfflineDefaults.REWARD_MODE)
    parser.add_argument("--train-ratio", type=float, default=RunOfflineDefaults.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=RunOfflineDefaults.VAL_RATIO)
    parser.add_argument("--epochs", type=int, default=RunOfflineDefaults.EPOCHS)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=RunOfflineDefaults.CHECKPOINT_EVERY,
        help="每隔多少个 epoch 自动保存一次 checkpoint；<=0 表示关闭自动保存。",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=RunOfflineDefaults.CHECKPOINT_EVERY_STEPS,
        help="每隔多少个训练 step 自动保存一次 checkpoint；<=0 表示关闭 step 级自动保存。",
    )
    parser.add_argument("--seed", type=int, default=RunOfflineDefaults.SEED)
    parser.add_argument("--alpha", type=float, default=RunOfflineDefaults.ALPHA)
    parser.add_argument("--nu", type=float, default=RunOfflineDefaults.NU)
    parser.add_argument("--reg", type=float, default=RunOfflineDefaults.REG)
    parser.add_argument("--hidden-dim", type=int, default=RunOfflineDefaults.HIDDEN_DIM)
    parser.add_argument(
        "--arm-embed-dim",
        type=int,
        default=RunOfflineDefaults.ARM_EMBED_DIM,
        help="NeuralUCB-Diag 中 arm embedding 的维度；越大越容易学 arm 相似性，但参数也会稍增。",
    )
    _add_bool_argument(
        parser,
        "--use-arm-onehot",
        default=RunOfflineDefaults.USE_ARM_ONEHOT,
        help_text="NeuralUCB-Diag 是否把 arm one-hot 直接拼到 reward_net 输入，增强不同方法的可区分性。",
    )
    _add_bool_argument(
        parser,
        "--use-arm-bias",
        default=RunOfflineDefaults.USE_ARM_BIAS,
        help_text="NeuralUCB-Diag 是否给每个 arm 增加显式 bias，帮助更快学到方法级基准差异。",
    )
    parser.add_argument("--lr", type=float, default=RunOfflineDefaults.LR)
    parser.add_argument("--initial-pulls", type=int, default=RunOfflineDefaults.INITIAL_PULLS)
    parser.add_argument(
        "--train-every",
        type=int,
        default=RunOfflineDefaults.TRAIN_EVERY,
        help="神经 bandit 每累计多少次 bandit 反馈，就触发一次网络训练；用于 Neural-LinUCB 和 NeuralUCB-Diag。",
    )
    parser.add_argument(
        "--representation-steps",
        type=int,
        default=RunOfflineDefaults.REPRESENTATION_STEPS,
        help="每次触发表示层训练时，最近表示层 buffer 上做多少轮优化。",
    )
    parser.add_argument(
        "--representation-buffer-size",
        type=int,
        default=RunOfflineDefaults.REPRESENTATION_BUFFER_SIZE,
        help="表示层训练使用的 recent window 大小；<=0 表示保留全部历史。",
    )
    parser.add_argument(
        "--representation-batch-size",
        type=int,
        default=RunOfflineDefaults.REPRESENTATION_BATCH_SIZE,
        help="表示层训练的 mini-batch 大小；<=0 表示每轮直接使用整个表示层窗口。",
    )
    parser.add_argument(
        "--representation-balance-mode",
        choices=["none", "arm", "problem_arm"],
        default=RunOfflineDefaults.REPRESENTATION_BALANCE_MODE,
        help="表示层训练时 history 的均衡方式：none / arm / problem_arm。",
    )
    parser.add_argument(
        "--representation-balance-power",
        type=float,
        default=RunOfflineDefaults.REPRESENTATION_BALANCE_POWER,
        help="inverse-frequency 均衡强度；0 表示不做频次修正，越大越强调低频组。",
    )
    _add_bool_argument(
        parser,
        "--representation-loss-reweight",
        default=RunOfflineDefaults.REPRESENTATION_LOSS_REWEIGHT,
        help_text="表示层训练时是否按 inverse-frequency 对 batch loss 再做重加权。",
    )
    parser.add_argument(
        "--linear-head-buffer-size",
        type=int,
        default=RunOfflineDefaults.LINEAR_HEAD_BUFFER_SIZE,
        help="线性头重建使用的窗口大小；<0 表示默认跟随 representation-buffer-size，<=0 表示保留全部历史。",
    )
    parser.add_argument(
        "--train-epsilon",
        type=float,
        default=RunOfflineDefaults.TRAIN_EPSILON,
        help="训练阶段额外的 epsilon 探索概率；只影响 train，不影响 val/test。",
    )
    _add_bool_argument(
        parser,
        "--problem-specific-heads",
        default=RunOfflineDefaults.PROBLEM_SPECIFIC_HEADS,
        help_text="Neural-LinUCB 是否为不同 problem 分别维护独立线性头。",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=RunOfflineDefaults.TRAIN_BATCH_SIZE,
        help="NeuralUCB-Diag 训练 reward_net 时的 mini-batch 大小；<=0 表示每次使用全部 history。",
    )
    parser.add_argument("--train-steps-per-update", type=int, default=RunOfflineDefaults.TRAIN_STEPS_PER_UPDATE)
    _add_bool_argument(
        parser,
        "--use-problem-arm-safety",
        default=RunOfflineDefaults.USE_PROBLEM_ARM_SAFETY,
        help_text="是否启用按问题的经验坏臂惩罚，压低明显高风险 arm 的 score。",
    )
    parser.add_argument(
        "--safety-min-pulls",
        type=int,
        default=RunOfflineDefaults.SAFETY_MIN_PULLS,
        help="某个 (problem, arm) 至少被拉多少次后，才允许根据经验均值对它做 safety 惩罚。",
    )
    parser.add_argument(
        "--safety-reward-gap",
        type=float,
        default=RunOfflineDefaults.SAFETY_REWARD_GAP,
        help="如果当前 arm 的经验均值落后该问题最好 arm 超过这个阈值，就开始惩罚。",
    )
    parser.add_argument(
        "--safety-penalty-scale",
        type=float,
        default=RunOfflineDefaults.SAFETY_PENALTY_SCALE,
        help="坏臂惩罚强度系数；越大越不容易再次选到经验上明显偏差的 arm。",
    )
    parser.add_argument(
        "--safety-penalty-cap",
        type=float,
        default=RunOfflineDefaults.SAFETY_PENALTY_CAP,
        help="单个 arm 的最大 safety 惩罚上限。",
    )
    _add_bool_argument(
        parser,
        "--freeze-encoder",
        default=RunOfflineDefaults.FREEZE_ENCODER,
        help_text="是否冻结底层 NSS 图编码器。",
    )
    parser.add_argument("--max-samples-per-problem", type=int, default=RunOfflineDefaults.MAX_SAMPLES_PER_PROBLEM)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=RunOfflineDefaults.EVAL_EVERY_STEPS,
        help="每隔多少个训练 step 自动在 val 集上评测一次；<=0 表示关闭训练中的周期性验证。",
    )
    parser.add_argument(
        "--eval-log-every",
        type=int,
        default=RunOfflineDefaults.EVAL_LOG_EVERY,
        help="评测阶段每隔多少个实例打印一条进度日志。",
    )
    _add_bool_argument(
        parser,
        "--use-best-checkpoint-for-final-eval",
        default=RunOfflineDefaults.USE_BEST_CHECKPOINT_FOR_FINAL_EVAL,
        help_text="是否根据 periodic val 恢复 best checkpoint，再做最终 val/test 评测。",
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        default=RunOfflineDefaults.BEST_CHECKPOINT_METRIC,
        help="选择 best checkpoint 的指标路径，例如 val_greedy.overall.mean_cost。",
    )
    parser.add_argument(
        "--best-checkpoint-mode",
        choices=["min", "max"],
        default=RunOfflineDefaults.BEST_CHECKPOINT_MODE,
        help="best checkpoint 指标方向：min=越小越好，max=越大越好。",
    )
    parser.add_argument("--device", default=RunOfflineDefaults.DEVICE)
    parser.add_argument("--output-dir", default=RunOfflineDefaults.OUTPUT_DIR)
    _add_bool_argument(
        parser,
        "--disable-progress",
        default=RunOfflineDefaults.DISABLE_PROGRESS,
        help_text="是否关闭 tqdm 进度条显示。",
    )
    _add_bool_argument(
        parser,
        "--quiet",
        default=RunOfflineDefaults.QUIET,
        help_text="是否减少阶段日志输出。",
    )
    _add_bool_argument(
        parser,
        "--print-step-json",
        default=RunOfflineDefaults.PRINT_STEP_JSON,
        help_text="是否把每一步的详细 trace json 直接打印到终端。",
    )
    parser.add_argument("--step-log-every", type=int, default=RunOfflineDefaults.STEP_LOG_EVERY)
    return parser.parse_args()


def main():
    """
    CLI 入口。

    当前默认只把 summary 写成 json。
    后续更完整的表格汇总和可视化，由 analysis.py 负责。
    """
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.method)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"实验开始，输出目录: {output_dir}", enabled=not args.quiet)
    results = run_experiment(args, output_dir=output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results["selector"]["test_greedy"]["overall"], ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
