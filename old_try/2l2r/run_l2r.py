from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

if __package__ in (None, ""):
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))

    from arm_config import ALL_METHODS, PROBLEM_TO_METHODS  # type: ignore
    from baselines import (  # type: ignore
        OracleSelector,
        RandomSelector,
        SingleBestGlobalSelector,
        SingleBestPerProblemSelector,
    )
    from build_dataset import build_joint_dataset  # type: ignore
    from default_settings import RunL2RDefaults  # type: ignore
    from losses import compute_ranking_loss  # type: ignore
    from model import (  # type: ignore
        DefaultGainRanker,
        OneVsDefaultBeatRanker,
        OneVsDefaultCoupledAdvantageRanker,
        OneVsDefaultCoupledSwitchRanker,
        OneVsDefaultGainRanker,
        OneVsDefaultMarginRanker,
        OneVsDefaultSwitchRanker,
        OneVsDefaultTwoStageRanker,
        StaySwitchRanker,
        SupervisedRanker,
    )
    from reward import compute_rank_based_rewards  # noqa: F401  # type: ignore
else:
    from .arm_config import ALL_METHODS, PROBLEM_TO_METHODS
    from .baselines import OracleSelector, RandomSelector, SingleBestGlobalSelector, SingleBestPerProblemSelector
    from .build_dataset import build_joint_dataset
    from .default_settings import RunL2RDefaults
    from .losses import compute_ranking_loss
    from .model import (
        DefaultGainRanker,
        OneVsDefaultBeatRanker,
        OneVsDefaultCoupledAdvantageRanker,
        OneVsDefaultCoupledSwitchRanker,
        OneVsDefaultGainRanker,
        OneVsDefaultMarginRanker,
        OneVsDefaultSwitchRanker,
        OneVsDefaultTwoStageRanker,
        StaySwitchRanker,
        SupervisedRanker,
    )


def _log(message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _iter_with_progress(iterable, *, total: int | None = None, desc: str = "", enabled: bool = True):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def _add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    action_cls = getattr(argparse, "BooleanOptionalAction", None)
    if action_cls is not None:
        parser.add_argument(name, action=action_cls, default=default, help=help_text)
        return

    option = name.lstrip("-")
    dest = option.replace("-", "_")
    negative_name = f"--no-{option}"
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(negative_name, dest=dest, action="store_false", help=f"关闭{help_text}")
    parser.set_defaults(**{dest: default})


def stratified_split_indices(samples, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 0):
    """
    按问题类型做 train / val / test 分层切分。
    """
    rng = np.random.default_rng(seed)
    by_problem: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_problem[sample.problem].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _problem, indices in by_problem.items():
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


def _safe_number(value):
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
    if record is None:
        return None
    return {key: _safe_number(value) for key, value in record.items()}


def _trace_selected_result(record: dict | None) -> dict | None:
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


class JsonlTraceStreamer:
    """
    实时 jsonl 写入器。
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._fp = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
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


def _record_trace_row(
    row: dict,
    trace_rows: list[dict] | None,
    streamer: JsonlTraceStreamer | None,
    print_step_json: bool,
    step_log_every: int,
) -> None:
    if trace_rows is not None:
        trace_rows.append(row)
    if streamer is not None:
        streamer.write(row)
    if print_step_json:
        every = max(1, int(step_log_every))
        order_index = int(row.get("order_index", 0))
        if order_index % every == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)


def _make_batch_tensors(
    batch_samples,
    device: torch.device,
    *,
    single_best_per_problem_arms: dict[str, int] | None = None,
    problem_mean_hardness: dict[str, float] | None = None,
    hard_instance_weight_alpha: float = 0.0,
    hard_instance_weight_cap: float = 0.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    从一批 InstanceSample 构造监督学习所需的目标张量。

    返回：
    - feasible_mask:       [B, A] bool
    - rewards:             [B, A] float
    - costs:               [B, A] float
    - normalized_regrets:  [B, A] float
    - best_arms:           [B] int
    - sample_weights:      [B] float

    其中 normalized regret 定义为：

        (cost - best_cost) / best_cost

    它比当前的 rank reward 更接近我们真正关心的目标：
    “如果这次选错，会在 cost 上吃多大的亏”
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_float = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_weight = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_float, empty_float, empty_float, empty_long, empty_weight, {
            "mean_sample_weight": 0.0,
            "max_sample_weight": 0.0,
            "mean_hardness": 0.0,
        }

    feasible_mask = torch.as_tensor(
        np.stack([sample.feasible_mask > 0 for sample in batch_samples], axis=0),
        dtype=torch.bool,
        device=device,
    )
    rewards = torch.as_tensor(
        np.stack([sample.rewards for sample in batch_samples], axis=0),
        dtype=torch.float32,
        device=device,
    )
    costs = torch.as_tensor(
        np.stack([sample.costs for sample in batch_samples], axis=0),
        dtype=torch.float32,
        device=device,
    )
    best_arms = torch.as_tensor(
        [int(sample.best_arm) for sample in batch_samples],
        dtype=torch.long,
        device=device,
    )

    # 这里显式按实例构造 regret，避免把“排名差 1 名”和“真实 cost 差很多”混为一谈。
    normalized_regrets_np = np.zeros_like(np.stack([sample.costs for sample in batch_samples], axis=0), dtype=np.float32)
    hardness_values: list[float] = []
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    for row_idx, sample in enumerate(batch_samples):
        feasible = np.asarray(sample.feasible_mask) > 0
        valid_costs = np.asarray(sample.costs, dtype=np.float64)[feasible]
        best_cost = float(np.min(valid_costs))
        denom = max(abs(best_cost), eps)
        normalized_regrets_np[row_idx, feasible] = (
            (np.asarray(sample.costs, dtype=np.float64)[feasible] - best_cost) / denom
        ).astype(np.float32)

        hardness = 0.0
        if single_best_per_problem_arms is not None:
            sbp_arm = int(single_best_per_problem_arms[sample.problem])
            hardness = float(normalized_regrets_np[row_idx, sbp_arm])
        hardness_values.append(hardness)

        if hard_instance_weight_alpha > 0.0 and single_best_per_problem_arms is not None:
            reference = 0.0
            if problem_mean_hardness is not None:
                reference = float(problem_mean_hardness.get(sample.problem, 0.0))
            reference = max(reference, eps)
            scaled_hardness = hardness / reference
            sample_weight = 1.0 + float(hard_instance_weight_alpha) * float(scaled_hardness)
            if hard_instance_weight_cap > 0.0:
                sample_weight = min(sample_weight, float(hard_instance_weight_cap))
            sample_weights_np[row_idx] = float(sample_weight)

    normalized_regrets = torch.as_tensor(normalized_regrets_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    extra_stats = {
        "mean_sample_weight": float(np.mean(sample_weights_np)),
        "max_sample_weight": float(np.max(sample_weights_np)),
        "mean_hardness": float(np.mean(hardness_values)) if hardness_values else 0.0,
    }
    return feasible_mask, rewards, costs, normalized_regrets, best_arms, sample_weights, extra_stats


def _make_switch_gate_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    problem_mean_hardness: dict[str, float],
    switch_label_scale: float,
    switch_positive_weight_alpha: float,
    switch_positive_weight_cap: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    为 two-stage default gate 构造监督目标。

    返回：
    - feasible_mask:          [B, A] bool
    - default_arms:           [B] int
    - best_arms:              [B] int
    - default_improvements:   [B] float
    - switch_thresholds:      [B] float
    - sample_weights:         [B] float

    其中：
    - default_improvement 表示“如果坚持默认 arm，会比 oracle 多损失多少 normalized regret”
    - switch_threshold 采用 problem-specific 尺度：
        switch_label_scale * problem_mean_hardness[problem]

    这样做的意图是：
    - TSP / CVRP 的 regret 量级不一样
    - 不能用一个全局绝对阈值粗暴切分
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_long, empty_float, empty_float, empty_float, {
            "switch_positive_rate": 0.0,
            "mean_default_improvement": 0.0,
            "mean_switch_threshold": 0.0,
            "mean_sample_weight": 0.0,
        }

    feasible_mask = torch.as_tensor(
        np.stack([sample.feasible_mask > 0 for sample in batch_samples], axis=0),
        dtype=torch.bool,
        device=device,
    )
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    best_arms_np = np.asarray([int(sample.best_arm) for sample in batch_samples], dtype=np.int64)
    default_improvements_np = np.zeros(len(batch_samples), dtype=np.float32)
    switch_thresholds_np = np.zeros(len(batch_samples), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    positive_count = 0
    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        improvement = (default_cost - best_cost) / max(abs(best_cost), eps)
        threshold = float(switch_label_scale) * max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        default_improvements_np[row_idx] = float(improvement)
        switch_thresholds_np[row_idx] = float(threshold)

        if improvement > threshold + eps and int(best_arms_np[row_idx]) != default_arm:
            positive_count += 1
            if switch_positive_weight_alpha > 0.0:
                weight = 1.0 + float(switch_positive_weight_alpha) * max(0.0, float(improvement / threshold) - 1.0)
                if switch_positive_weight_cap > 0.0:
                    weight = min(weight, float(switch_positive_weight_cap))
                sample_weights_np[row_idx] = float(weight)

    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    best_arms = torch.as_tensor(best_arms_np, dtype=torch.long, device=device)
    default_improvements = torch.as_tensor(default_improvements_np, dtype=torch.float32, device=device)
    switch_thresholds = torch.as_tensor(switch_thresholds_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    extra_stats = {
        "switch_positive_rate": float(positive_count / max(len(batch_samples), 1)),
        "mean_default_improvement": float(np.mean(default_improvements_np)),
        "mean_switch_threshold": float(np.mean(switch_thresholds_np)),
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return (
        feasible_mask,
        default_arms,
        best_arms,
        default_improvements,
        switch_thresholds,
        sample_weights,
        extra_stats,
    )


def _parse_float_grid(raw: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _make_default_gain_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    为 default_gain_regression 构造监督目标。

    target_gain 的定义：
    - target_gain(a) = max(0, (cost_default - cost_a) / |best_cost|)
    - 如果打开 normalize_by_problem_hardness，再除以该问题的 hardness 均值

    含义：
    - 只有当某个 arm 真正优于默认 arm 时，它才得到正目标
    - 比默认 arm 差的动作统一压到 0
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_weight = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_float, empty_float, empty_weight, {
            "positive_gain_rate": 0.0,
            "mean_target_gain": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    feasible_mask_np = np.stack([sample.feasible_mask > 0 for sample in batch_samples], axis=0)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    target_gains_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    positive_gain_weight_np = np.ones((len(batch_samples), n_arms), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = feasible_mask_np[row_idx]
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        scale = 1.0
        if normalize_by_problem_hardness:
            scale = max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        raw_gain = np.zeros(n_arms, dtype=np.float32)
        raw_gain[feasible] = np.maximum(
            0.0,
            (default_cost - np.asarray(sample.costs, dtype=np.float64)[feasible]) / max(abs(best_cost), eps),
        ).astype(np.float32)
        raw_gain = raw_gain / float(scale)
        raw_gain[default_arm] = 0.0

        target_gains_np[row_idx] = raw_gain
        positive_gain_weight_np[row_idx] = 1.0 + float(positive_weight_alpha) * raw_gain

        # 如果这条样本至少存在一个“明显优于默认”的动作，就整体略微加权。
        sample_weights_np[row_idx] = 1.0 + float(np.max(raw_gain))

    feasible_mask = torch.as_tensor(feasible_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    target_gains = torch.as_tensor(target_gains_np, dtype=torch.float32, device=device)
    positive_gain_weight = torch.as_tensor(positive_gain_weight_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    extra_stats = {
        "positive_gain_rate": float(np.mean(target_gains_np > 0)),
        "mean_target_gain": float(np.mean(target_gains_np)),
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return feasible_mask, default_arms, target_gains, positive_gain_weight, sample_weights, extra_stats


def _make_one_vs_default_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    candidate_gain_prior_by_problem: dict[str, dict[int, float]] | None,
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    只为“候选替代臂”构造 default-relative gain 目标。

    与 `_make_default_gain_targets` 的区别是：
    - 非候选 arm 完全不参与监督
    - 每个候选 arm 都是一个独立的 one-vs-default 子任务
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_weight = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_float, empty_float, empty_weight, {
            "positive_gain_rate": 0.0,
            "mean_target_gain": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    target_gains_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    positive_gain_weight_np = np.ones((len(batch_samples), n_arms), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        scale = 1.0
        if normalize_by_problem_hardness:
            scale = max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if not candidate_arms:
            continue
        candidate_mask_np[row_idx, candidate_arms] = True

        max_gain = 0.0
        for arm in candidate_arms:
            gain = max(
                0.0,
                (default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps),
            ) / float(scale)
            prior = 0.0 if candidate_gain_prior_by_problem is None else float(
                candidate_gain_prior_by_problem.get(problem, {}).get(int(arm), 0.0)
            )
            target_gains_np[row_idx, arm] = float(gain - prior)
            positive_gain_weight_np[row_idx, arm] = 1.0 + float(positive_weight_alpha) * float(gain)
            max_gain = max(max_gain, float(gain))
        sample_weights_np[row_idx] = 1.0 + float(max_gain)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    target_gains = torch.as_tensor(target_gains_np, dtype=torch.float32, device=device)
    positive_gain_weight = torch.as_tensor(positive_gain_weight_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    extra_stats = {
        "positive_gain_rate": float(np.mean(target_gains_np[candidate_mask_np] > 0)) if np.any(candidate_mask_np) else 0.0,
        "mean_target_gain": float(np.mean(target_gains_np[candidate_mask_np])) if np.any(candidate_mask_np) else 0.0,
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return candidate_mask, default_arms, target_gains, positive_gain_weight, sample_weights, extra_stats


def _make_one_vs_default_binary_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    构造 one-vs-default beat 二分类目标。

    label(a) = 1:
      该候选 arm 在当前实例上优于默认 arm

    label(a) = 0:
      不优于默认 arm
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_weight = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_float, empty_float, empty_weight, {
            "positive_label_rate": 0.0,
            "mean_target_gain": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    labels_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    positive_weights_np = np.ones((len(batch_samples), n_arms), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    raw_gain_mean = []

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        scale = 1.0
        if normalize_by_problem_hardness:
            scale = max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if not candidate_arms:
            continue
        candidate_mask_np[row_idx, candidate_arms] = True

        max_gain = 0.0
        for arm in candidate_arms:
            gain = max(
                0.0,
                (default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps),
            ) / float(scale)
            raw_gain_mean.append(float(gain))
            labels_np[row_idx, arm] = 1.0 if gain > eps else 0.0
            positive_weights_np[row_idx, arm] = 1.0 + float(positive_weight_alpha) * float(gain)
            max_gain = max(max_gain, float(gain))
        sample_weights_np[row_idx] = 1.0 + float(max_gain)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.float32, device=device)
    positive_weights = torch.as_tensor(positive_weights_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    extra_stats = {
        "positive_label_rate": float(np.mean(labels_np[candidate_mask_np])) if np.any(candidate_mask_np) else 0.0,
        "mean_target_gain": float(np.mean(raw_gain_mean)) if raw_gain_mean else 0.0,
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return candidate_mask, default_arms, labels, positive_weights, sample_weights, extra_stats


def _make_one_vs_default_margin_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    构造 default-vs-candidate 的 signed margin 目标。

    target_margin(a) 定义为：

        (cost_default - cost_a) / |best_cost|

    如果开启按问题 hardness 归一化，再除以 problem_mean_hardness。

    含义：
    - > 0: arm a 比默认 arm 更好，应该推动 score(a) 为正
    - < 0: arm a 比默认 arm 更差，应该推动 score(a) 为负
    - |target_margin| 越大，要求分离越明显
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_weight = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_float, empty_float, empty_weight, {
            "positive_margin_rate": 0.0,
            "mean_abs_margin": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    target_margins_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    entry_weights_np = np.ones((len(batch_samples), n_arms), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        scale = 1.0
        if normalize_by_problem_hardness:
            scale = max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if not candidate_arms:
            continue
        candidate_mask_np[row_idx, candidate_arms] = True

        max_abs_margin = 0.0
        for arm in candidate_arms:
            margin = ((default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps)) / float(scale)
            target_margins_np[row_idx, arm] = float(margin)
            if margin > 0.0:
                entry_weights_np[row_idx, arm] = 1.0 + float(positive_weight_alpha) * float(abs(margin))
            else:
                entry_weights_np[row_idx, arm] = 1.0
            max_abs_margin = max(max_abs_margin, abs(float(margin)))
        sample_weights_np[row_idx] = 1.0 + float(max_abs_margin)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    target_margins = torch.as_tensor(target_margins_np, dtype=torch.float32, device=device)
    entry_weights = torch.as_tensor(entry_weights_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)

    candidate_values = target_margins_np[candidate_mask_np]
    extra_stats = {
        "positive_margin_rate": float(np.mean(candidate_values > 0)) if candidate_values.size > 0 else 0.0,
        "mean_abs_margin": float(np.mean(np.abs(candidate_values))) if candidate_values.size > 0 else 0.0,
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return candidate_mask, default_arms, target_margins, entry_weights, sample_weights, extra_stats


def _make_one_vs_default_hard_rank_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    switch_label_scale: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    构造 one-vs-default hard-rank 目标。

    相比普通的 margin 版本，这里额外生成：
    - hard_sample_mask

    它的语义是：
    - 当前实例上，默认 arm 已经“不够稳”
    - 至少存在一个候选 arm，相对默认 arm 的正 margin 足够大

    只有这类样本，才会额外触发候选臂之间的 pairwise ranking loss。
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0, 0), dtype=torch.float32, device=device)
        empty_weight = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_float, empty_float, empty_weight, empty_bool.new_zeros((0,)), {
            "positive_margin_rate": 0.0,
            "mean_abs_margin": 0.0,
            "hard_sample_rate": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    target_margins_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    entry_weights_np = np.ones((len(batch_samples), n_arms), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)
    hard_sample_mask_np = np.zeros(len(batch_samples), dtype=bool)

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        problem_hardness = max(float(problem_mean_hardness.get(problem, 0.0)), eps)
        scale = 1.0 if not normalize_by_problem_hardness else problem_hardness

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if not candidate_arms:
            continue
        candidate_mask_np[row_idx, candidate_arms] = True

        max_abs_margin = 0.0
        max_positive_margin = 0.0
        for arm in candidate_arms:
            raw_margin = (default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps)
            margin = raw_margin / float(scale)
            target_margins_np[row_idx, arm] = float(margin)
            if margin > 0.0:
                entry_weights_np[row_idx, arm] = 1.0 + float(positive_weight_alpha) * float(abs(margin))
                max_positive_margin = max(max_positive_margin, float(margin))
            else:
                entry_weights_np[row_idx, arm] = 1.0
            max_abs_margin = max(max_abs_margin, abs(float(margin)))

        sample_weights_np[row_idx] = 1.0 + float(max_abs_margin)
        hard_threshold = (float(switch_label_scale) * problem_hardness) / float(scale)
        hard_sample_mask_np[row_idx] = bool(max_positive_margin > hard_threshold + eps)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    target_margins = torch.as_tensor(target_margins_np, dtype=torch.float32, device=device)
    entry_weights = torch.as_tensor(entry_weights_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    hard_sample_mask = torch.as_tensor(hard_sample_mask_np, dtype=torch.bool, device=device)

    candidate_values = target_margins_np[candidate_mask_np]
    extra_stats = {
        "positive_margin_rate": float(np.mean(candidate_values > 0)) if candidate_values.size > 0 else 0.0,
        "mean_abs_margin": float(np.mean(np.abs(candidate_values))) if candidate_values.size > 0 else 0.0,
        "hard_sample_rate": float(np.mean(hard_sample_mask_np)) if hard_sample_mask_np.size > 0 else 0.0,
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return (
        candidate_mask,
        default_arms,
        target_margins,
        entry_weights,
        sample_weights,
        hard_sample_mask,
        extra_stats,
    )


def _make_one_vs_default_switch_rank_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    switch_label_scale: float,
    switch_positive_weight_alpha: float,
    switch_positive_weight_cap: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    构造显式 switch + candidate ranking 的监督目标。

    返回：
    - candidate_mask:            [B, A]
    - default_arms:             [B]
    - target_margins:           [B, A]
    - switch_labels:            [B]
    - switch_positive_weights:  [B]
    - sample_weights:           [B]

    设计上把两个问题拆开：
    1. switch 标签：
       看当前实例上，是否存在候选 arm 的 raw gain
       超过 problem-specific 阈值
    2. candidate ranking：
       用 signed margin 排候选臂
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=device)
        empty_matrix = torch.zeros((0, 0), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_matrix, empty_float, empty_float, empty_float, {
            "switch_positive_rate": 0.0,
            "mean_abs_margin": 0.0,
            "mean_switch_threshold": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    target_margins_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    switch_labels_np = np.zeros(len(batch_samples), dtype=np.float32)
    switch_positive_weights_np = np.ones(len(batch_samples), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)
    switch_thresholds_np = np.zeros(len(batch_samples), dtype=np.float32)

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        problem_hardness = max(float(problem_mean_hardness.get(problem, 0.0)), eps)
        margin_scale = problem_hardness if normalize_by_problem_hardness else 1.0
        switch_threshold = float(switch_label_scale) * problem_hardness
        switch_thresholds_np[row_idx] = switch_threshold

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if not candidate_arms:
            continue
        candidate_mask_np[row_idx, candidate_arms] = True

        max_abs_margin = 0.0
        max_positive_raw_gain = 0.0
        for arm in candidate_arms:
            raw_gain = (default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps)
            normalized_margin = raw_gain / float(margin_scale)
            target_margins_np[row_idx, arm] = float(normalized_margin)
            max_abs_margin = max(max_abs_margin, abs(float(normalized_margin)))
            max_positive_raw_gain = max(max_positive_raw_gain, max(0.0, float(raw_gain)))

        sample_weights_np[row_idx] = 1.0 + float(max_abs_margin)
        if max_positive_raw_gain > switch_threshold + eps:
            switch_labels_np[row_idx] = 1.0
            if switch_positive_weight_alpha > 0.0:
                base = max(float(switch_threshold), eps)
                weight = 1.0 + float(switch_positive_weight_alpha) * max(0.0, float(max_positive_raw_gain / base) - 1.0)
                if switch_positive_weight_cap > 0.0:
                    weight = min(weight, float(switch_positive_weight_cap))
                switch_positive_weights_np[row_idx] = float(weight)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    target_margins = torch.as_tensor(target_margins_np, dtype=torch.float32, device=device)
    switch_labels = torch.as_tensor(switch_labels_np, dtype=torch.float32, device=device)
    switch_positive_weights = torch.as_tensor(switch_positive_weights_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)

    candidate_values = target_margins_np[candidate_mask_np]
    extra_stats = {
        "switch_positive_rate": float(np.mean(switch_labels_np)),
        "mean_abs_margin": float(np.mean(np.abs(candidate_values))) if candidate_values.size > 0 else 0.0,
        "mean_switch_threshold": float(np.mean(switch_thresholds_np)),
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return (
        candidate_mask,
        default_arms,
        target_margins,
        switch_labels,
        switch_positive_weights,
        sample_weights,
        extra_stats,
    )


def _make_one_vs_default_switch_advantage_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    positive_weight_alpha: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    构造 sample-level switch advantage 回归目标。

    switch_target 的定义：
    - 当前样本上，候选臂相对默认臂的最大正收益
    - 若没有候选臂优于默认臂，则为 0

    如果打开按问题 hardness 归一化，则：
    - candidate margin
    - switch target
    都除以 problem_mean_hardness

    这样 TSP 和 CVRP 仍能共用一个训练流程，但量级更可比。
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=device)
        empty_matrix = torch.zeros((0, 0), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_matrix, empty_float, empty_float, empty_float, {
            "switch_positive_rate": 0.0,
            "mean_abs_margin": 0.0,
            "mean_switch_target": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    target_margins_np = np.zeros((len(batch_samples), n_arms), dtype=np.float32)
    switch_targets_np = np.zeros(len(batch_samples), dtype=np.float32)
    switch_positive_weights_np = np.ones(len(batch_samples), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        scale = 1.0
        if normalize_by_problem_hardness:
            scale = max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if not candidate_arms:
            continue
        candidate_mask_np[row_idx, candidate_arms] = True

        max_positive_margin = 0.0
        max_abs_margin = 0.0
        for arm in candidate_arms:
            margin = ((default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps)) / float(scale)
            target_margins_np[row_idx, arm] = float(margin)
            max_abs_margin = max(max_abs_margin, abs(float(margin)))
            if margin > 0.0:
                max_positive_margin = max(max_positive_margin, float(margin))

        switch_targets_np[row_idx] = float(max_positive_margin)
        switch_positive_weights_np[row_idx] = 1.0 + float(positive_weight_alpha) * float(max_positive_margin)
        sample_weights_np[row_idx] = 1.0 + float(max_abs_margin)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    target_margins = torch.as_tensor(target_margins_np, dtype=torch.float32, device=device)
    switch_targets = torch.as_tensor(switch_targets_np, dtype=torch.float32, device=device)
    switch_positive_weights = torch.as_tensor(switch_positive_weights_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)

    candidate_values = target_margins_np[candidate_mask_np]
    extra_stats = {
        "switch_positive_rate": float(np.mean(switch_targets_np > 0.0)),
        "mean_abs_margin": float(np.mean(np.abs(candidate_values))) if candidate_values.size > 0 else 0.0,
        "mean_switch_target": float(np.mean(switch_targets_np)),
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return (
        candidate_mask,
        default_arms,
        target_margins,
        switch_targets,
        switch_positive_weights,
        sample_weights,
        extra_stats,
    )


def _make_one_vs_default_two_stage_targets(
    batch_samples,
    device: torch.device,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    switch_label_scale: float,
    switch_positive_weight_alpha: float,
    switch_positive_weight_cap: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    为候选约束的两阶段 one-vs-default 模型构造监督目标。

    返回：
    - candidate_mask:            [B, A]，只标出候选替代臂
    - default_arms:             [B]
    - candidate_best_arms:      [B]
    - candidate_improvements:   [B]
    - switch_thresholds:        [B]
    - sample_weights:           [B]

    其中：
    - candidate_best_arms 不是 oracle best arm
    - 而是“当前候选集合里，相对默认 arm 收益最大的那个 arm”
    - 第一阶段 switch 标签只看：
      候选集合里是否存在“足够值得切”的替代臂
    """
    if len(batch_samples) == 0:
        empty_bool = torch.zeros((0, 0), dtype=torch.bool, device=device)
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_float = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_bool, empty_long, empty_long, empty_float, empty_float, empty_float, {
            "switch_positive_rate": 0.0,
            "mean_candidate_improvement": 0.0,
            "mean_switch_threshold": 0.0,
            "mean_sample_weight": 0.0,
        }

    n_arms = len(batch_samples[0].costs)
    candidate_mask_np = np.zeros((len(batch_samples), n_arms), dtype=bool)
    default_arms_np = np.asarray(
        [int(default_arms_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
        dtype=np.int64,
    )
    candidate_best_arms_np = default_arms_np.copy()
    candidate_improvements_np = np.zeros(len(batch_samples), dtype=np.float32)
    switch_thresholds_np = np.zeros(len(batch_samples), dtype=np.float32)
    sample_weights_np = np.ones(len(batch_samples), dtype=np.float32)

    positive_count = 0
    for row_idx, sample in enumerate(batch_samples):
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        default_arm = int(default_arms_np[row_idx])
        default_cost = float(sample.costs[default_arm])
        threshold = float(switch_label_scale) * max(float(problem_mean_hardness.get(problem, 0.0)), eps)

        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, []) if feasible[int(arm)]]
        if candidate_arms:
            candidate_mask_np[row_idx, candidate_arms] = True

        best_candidate_arm = default_arm
        best_candidate_gain = 0.0
        for arm in candidate_arms:
            gain = max(
                0.0,
                (default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps),
            )
            if gain > best_candidate_gain + eps:
                best_candidate_gain = float(gain)
                best_candidate_arm = int(arm)

        candidate_best_arms_np[row_idx] = int(best_candidate_arm)
        candidate_improvements_np[row_idx] = float(best_candidate_gain)
        switch_thresholds_np[row_idx] = float(threshold)

        if best_candidate_arm != default_arm and best_candidate_gain > threshold + eps:
            positive_count += 1
            if switch_positive_weight_alpha > 0.0:
                weight = 1.0 + float(switch_positive_weight_alpha) * max(0.0, float(best_candidate_gain / threshold) - 1.0)
                if switch_positive_weight_cap > 0.0:
                    weight = min(weight, float(switch_positive_weight_cap))
                sample_weights_np[row_idx] = float(weight)

    candidate_mask = torch.as_tensor(candidate_mask_np, dtype=torch.bool, device=device)
    default_arms = torch.as_tensor(default_arms_np, dtype=torch.long, device=device)
    candidate_best_arms = torch.as_tensor(candidate_best_arms_np, dtype=torch.long, device=device)
    candidate_improvements = torch.as_tensor(candidate_improvements_np, dtype=torch.float32, device=device)
    switch_thresholds = torch.as_tensor(switch_thresholds_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
    extra_stats = {
        "switch_positive_rate": float(positive_count / max(len(batch_samples), 1)),
        "mean_candidate_improvement": float(np.mean(candidate_improvements_np)),
        "mean_switch_threshold": float(np.mean(switch_thresholds_np)),
        "mean_sample_weight": float(np.mean(sample_weights_np)),
    }
    return (
        candidate_mask,
        default_arms,
        candidate_best_arms,
        candidate_improvements,
        switch_thresholds,
        sample_weights,
        extra_stats,
    )


def _compute_cost_gains_against_baseline(metrics: dict, baseline_metrics: dict) -> dict:
    """
    计算“模型相对某个 baseline 在 mean_cost 上提升了多少”。

    定义：
        gain = baseline_mean_cost - model_mean_cost

    gain > 0 表示模型更好。
    """
    by_problem = {}
    for problem, problem_metrics in metrics.get("by_problem", {}).items():
        baseline_problem_metrics = baseline_metrics.get("by_problem", {}).get(problem)
        if baseline_problem_metrics is None:
            continue
        by_problem[problem] = float(baseline_problem_metrics["mean_cost"]) - float(problem_metrics["mean_cost"])

    macro_gain = float(np.mean(list(by_problem.values()))) if by_problem else float("nan")
    overall_gain = float(baseline_metrics["overall"]["mean_cost"]) - float(metrics["overall"]["mean_cost"])
    return {
        "overall_mean_cost_gain": overall_gain,
        "macro_mean_cost_gain": macro_gain,
        "by_problem": by_problem,
    }


def _sample_hardness_for_selector(sample, arm: int, eps: float = 1e-12) -> float:
    feasible = np.asarray(sample.feasible_mask) > 0
    best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
    denom = max(abs(best_cost), eps)
    return float((float(sample.costs[int(arm)]) - best_cost) / denom)


def _compute_problem_mean_hardness(samples, indices, single_best_per_problem_selector) -> dict[str, float]:
    """
    在训练集上估计每个 problem 的“single_best_per_problem 通常会吃多少 regret”。

    后面做 hard-instance weighting 时，会把每个样本的 hardness
    除以这个均值，避免 TSP/CVRP 因量纲不同导致权重尺度差异过大。
    """
    by_problem: dict[str, list[float]] = defaultdict(list)
    for idx in indices:
        sample = samples[idx]
        arm = int(single_best_per_problem_selector.best_by_problem[sample.problem])
        by_problem[sample.problem].append(_sample_hardness_for_selector(sample, arm))
    return {
        problem: float(np.mean(values)) if values else 0.0
        for problem, values in by_problem.items()
    }


def _compute_one_vs_default_candidate_arms(
    samples,
    indices,
    single_best_per_problem_selector,
    *,
    min_positive_rate: float,
    min_positive_count: int,
    max_candidates: int,
    eps: float = 1e-12,
) -> tuple[dict[str, list[int]], dict[str, list[dict]]]:
    """
    在训练集上自动筛 one-vs-default 的候选替代臂。

    标准：
    1. 某个 arm 必须在足够多样本上优于默认 arm
    2. 同时按“正样本次数 + 正收益总量”排序
    3. 每个问题最多保留前 max_candidates 个

    返回：
    - candidate_arm_by_problem
    - 便于日志展示的详细统计
    """
    by_problem_rows: dict[str, list] = defaultdict(list)
    for idx in indices:
        sample = samples[idx]
        by_problem_rows[str(sample.problem).lower()].append(sample)

    candidate_arm_by_problem: dict[str, list[int]] = {}
    detail_by_problem: dict[str, list[dict]] = {}
    for problem, problem_rows in by_problem_rows.items():
        default_arm = int(single_best_per_problem_selector.best_by_problem[problem])
        positive_count = defaultdict(int)
        positive_gain_sum = defaultdict(float)

        for sample in problem_rows:
            feasible = np.asarray(sample.feasible_mask) > 0
            best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
            default_cost = float(sample.costs[default_arm])
            denom = max(abs(best_cost), eps)
            for arm in np.flatnonzero(feasible):
                arm = int(arm)
                if arm == default_arm:
                    continue
                gain = (default_cost - float(sample.costs[arm])) / denom
                if gain > eps:
                    positive_count[arm] += 1
                    positive_gain_sum[arm] += float(gain)

        detail_rows = []
        total = max(len(problem_rows), 1)
        for arm, count in sorted(positive_count.items(), key=lambda item: (-item[1], item[0])):
            detail_rows.append(
                {
                    "arm_id": int(arm),
                    "positive_count": int(count),
                    "positive_rate": float(count / total),
                    "positive_gain_sum": float(positive_gain_sum[arm]),
                    "avg_positive_gain": float(positive_gain_sum[arm] / max(count, 1)),
                }
            )

        selected = [
            row["arm_id"]
            for row in sorted(
                detail_rows,
                key=lambda row: (-row["positive_count"], -row["positive_gain_sum"], row["arm_id"]),
            )
            if row["positive_count"] >= int(min_positive_count) and row["positive_rate"] >= float(min_positive_rate)
        ][: max(1, int(max_candidates))]

        if not selected and detail_rows:
            selected = [int(detail_rows[0]["arm_id"])]

        candidate_arm_by_problem[problem] = [int(arm) for arm in selected]
        detail_by_problem[problem] = detail_rows

    return candidate_arm_by_problem, detail_by_problem


def _compute_one_vs_default_candidate_gain_priors(
    samples,
    indices,
    *,
    default_arms_by_problem: dict[str, int],
    candidate_arms_by_problem: dict[str, list[int]],
    problem_mean_hardness: dict[str, float],
    normalize_by_problem_hardness: bool,
    eps: float = 1e-12,
) -> dict[str, dict[int, float]]:
    """
    计算 one-vs-default 候选 arm 的平均 gain prior。

    这里的 prior 取的是：
    - 训练集上该候选 arm 相对默认 arm 的平均 normalized gain
    - 如果开启归一化，再除以 problem hardness

    后续网络只学习 residual：
        residual_target = actual_gain - prior
    """
    prior_by_problem: dict[str, dict[int, float]] = {}
    by_problem_rows: dict[str, list] = defaultdict(list)
    for idx in indices:
        sample = samples[idx]
        by_problem_rows[str(sample.problem).lower()].append(sample)

    for problem, problem_rows in by_problem_rows.items():
        default_arm = int(default_arms_by_problem[problem])
        candidate_arms = [int(arm) for arm in candidate_arms_by_problem.get(problem, [])]
        priors: dict[int, float] = {}
        for arm in candidate_arms:
            gains = []
            for sample in problem_rows:
                feasible = np.asarray(sample.feasible_mask) > 0
                if not feasible[arm]:
                    continue
                best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
                default_cost = float(sample.costs[default_arm])
                scale = 1.0
                if normalize_by_problem_hardness:
                    scale = max(float(problem_mean_hardness.get(problem, 0.0)), eps)
                gain = max(
                    0.0,
                    (default_cost - float(sample.costs[arm])) / max(abs(best_cost), eps),
                ) / float(scale)
                gains.append(float(gain))
            priors[int(arm)] = float(np.mean(gains)) if gains else 0.0
        prior_by_problem[problem] = priors
    return prior_by_problem


def _compute_problem_arm_prior_scores(samples, indices, arm_names: list[str], mode: str) -> dict[str, np.ndarray]:
    """
    在训练集上统计 problem-specific 的 arm 先验分数。

    当前实现采用：
    - 对每个 problem / arm 计算平均 normalized regret
    - 再取其相反数作为 prior score
    - 并在每个 problem 内做 max-centering，让最优平均 arm 的 prior = 0

    含义上可以理解为：
    - 先验只表达“这个问题上 arm 的默认强弱顺序”
    - 网络残差再去学习“哪些实例应该偏离这个默认顺序”
    """
    mode = str(mode).lower()
    if mode == "none":
        return {}
    if mode != "problem_arm_avg_neg_regret":
        raise ValueError(f"未知 score_prior_mode: {mode}")

    n_arms = len(arm_names)
    regret_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_arms, dtype=np.float64))
    regret_count: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_arms, dtype=np.int64))

    for idx in indices:
        sample = samples[idx]
        problem = str(sample.problem).lower()
        feasible = np.asarray(sample.feasible_mask) > 0
        best_cost = float(np.nanmin(np.asarray(sample.costs, dtype=np.float64)[feasible]))
        denom = max(abs(best_cost), 1e-12)
        normalized_regrets = np.zeros(n_arms, dtype=np.float64)
        normalized_regrets[feasible] = (np.asarray(sample.costs, dtype=np.float64)[feasible] - best_cost) / denom
        regret_sum[problem][feasible] += normalized_regrets[feasible]
        regret_count[problem][feasible] += 1

    prior_by_problem: dict[str, np.ndarray] = {}
    for problem, summed in regret_sum.items():
        counts = regret_count[problem]
        mean_regret = np.divide(
            summed,
            np.maximum(counts, 1),
            out=np.full(n_arms, np.inf, dtype=np.float64),
            where=counts > 0,
        )
        prior = -mean_regret
        valid = counts > 0
        if np.any(valid):
            prior[valid] = prior[valid] - np.max(prior[valid])
            prior_std = float(np.std(prior[valid]))
            if np.isfinite(prior_std) and prior_std > 1e-8:
                prior[valid] = prior[valid] / prior_std
        prior[~valid] = 0.0
        prior_by_problem[problem] = prior.astype(np.float32)
    return prior_by_problem


def _format_problem_cost_line(metrics: dict) -> str:
    parts = []
    for problem in ("tsp", "cvrp"):
        block = metrics.get("by_problem", {}).get(problem)
        if block is None:
            continue
        parts.append(f"{problem}_cost={float(block['mean_cost']):.6f}")
    return " | ".join(parts)


def _trace_row_from_prediction(
    *,
    sample,
    selected_arm: int,
    arm_scores: np.ndarray,
    arm_names: list[str],
    stage: str,
    order_index: int,
    epoch: int | None = None,
    extra: dict | None = None,
) -> dict:
    selected_record = None
    if hasattr(sample, "result_records") and sample.result_records is not None:
        if 0 <= int(selected_arm) < len(sample.result_records):
            selected_record = _trace_selected_result(sample.result_records[int(selected_arm)])

    feasible_arms = [int(idx) for idx in np.flatnonzero(sample.feasible_mask > 0)]
    best_cost = float(np.nanmin(sample.costs[sample.feasible_mask > 0]))
    selected_cost = float(sample.costs[int(selected_arm)])
    selected_reward = float(sample.rewards[int(selected_arm)])

    arm_details = []
    selected_score = None
    for arm_id in range(len(arm_names)):
        feasible = bool(sample.feasible_mask[arm_id] > 0)
        score_value = None if not feasible else float(arm_scores[arm_id])
        true_reward = None if not feasible else float(sample.rewards[arm_id])
        true_cost = None if not feasible else float(sample.costs[arm_id])
        true_rank = None if not feasible else float(sample.ranks[arm_id])
        if arm_id == int(selected_arm):
            selected_score = score_value
        arm_details.append(
            {
                "arm_id": int(arm_id),
                "arm_name": arm_names[int(arm_id)],
                "feasible": feasible,
                "score": score_value,
                "true_reward": true_reward,
                "true_cost": true_cost,
                "true_rank": true_rank,
            }
        )

    row = {
        "stage": stage,
        "decision_mode": "greedy",
        "epoch": None if epoch is None else int(epoch),
        "order_index": int(order_index),
        "uid": sample.uid,
        "problem": sample.problem,
        "global_index": int(sample.global_index),
        "selected_arm": int(selected_arm),
        "selected_arm_name": arm_names[int(selected_arm)],
        "selection_reason": "max_score",
        "selected_score": _safe_number(selected_score),
        "selected_reward": selected_reward,
        "selected_cost": selected_cost,
        "selected_result": selected_record,
        "best_arm": int(sample.best_arm),
        "best_arm_name": arm_names[int(sample.best_arm)],
        "best_cost": best_cost,
        "regret": selected_cost - best_cost,
        "top1": 1.0 if int(selected_arm) == int(sample.best_arm) else 0.0,
        "feasible_arms": feasible_arms,
        "arm_details": arm_details,
    }
    if extra:
        row.update(extra)
    return row


def _evaluate_pick_results(rows: list[dict]) -> dict:
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
    return {"overall": _aggregate(rows), "by_problem": by_problem, "macro_average": macro}


def _model_candidate_alt_arms(model, sample, feasible_mask: np.ndarray, default_arm: int) -> list[int]:
    """
    返回当前样本在该模型下真正允许考虑的替代臂集合。

    - 普通 DefaultGainRanker: 所有可行非默认 arm
    - OneVsDefaultGainRanker: 只看训练集自动筛出的候选替代 arm
    """
    candidate_method = getattr(model, "candidate_arms_for_problem", None)
    if callable(candidate_method):
        explicit = candidate_method(str(sample.problem).lower())
        if explicit is not None:
            return [int(arm) for arm in explicit if feasible_mask[int(arm)] and int(arm) != int(default_arm)]
    return [int(arm) for arm in np.flatnonzero(feasible_mask) if int(arm) != int(default_arm)]


def evaluate_ranker(
    model: SupervisedRanker,
    samples,
    indices,
    arm_names: list[str],
    *,
    batch_size: int = 64,
    stage: str = "",
    epoch: int | None = None,
    show_progress: bool = False,
    log_progress: bool = False,
    progress_log_every: int = 500,
    trace_rows: list[dict] | None = None,
    trace_streamer: JsonlTraceStreamer | None = None,
    print_step_json: bool = False,
    step_log_every: int = 1,
) -> dict:
    """
    评测 supervised ranker。

    评测逻辑很简单：
    - 对每个样本，给所有 arm 打分
    - 只在 feasible arms 里取分数最大的那个
    """
    model.eval()
    indices = list(indices)
    total_count = len(indices)
    progress_log_every = max(1, int(progress_log_every))
    if log_progress:
        _log(f"在 {stage or 'eval'} 集上评测 ({total_count} 实例)...", enabled=True)

    summary_rows = []
    batch_starts = range(0, total_count, max(1, int(batch_size)))
    progress = _iter_with_progress(
        batch_starts,
        total=(total_count + max(1, int(batch_size)) - 1) // max(1, int(batch_size)),
        desc=f"{stage or 'eval'}",
        enabled=show_progress,
    )

    processed = 0
    for start in progress:
        batch_ids = indices[start : start + max(1, int(batch_size))]
        batch_samples = [samples[idx] for idx in batch_ids]

        with torch.no_grad():
            if isinstance(model, OneVsDefaultSwitchRanker):
                batch_outputs = model.forward_with_details(batch_samples)
                batch_scores = batch_outputs["alt_logits"].detach().cpu().numpy()
                batch_switch_probs = batch_outputs["switch_probs"].detach().cpu().numpy()
                batch_switch_logits = batch_outputs["switch_logits"].detach().cpu().numpy()
                batch_default_arms = batch_outputs["default_arms"].detach().cpu().numpy()
            elif isinstance(model, (StaySwitchRanker, OneVsDefaultTwoStageRanker)):
                batch_outputs = model.forward_with_details(batch_samples)
                batch_scores = batch_outputs["final_scores"].detach().cpu().numpy()
                batch_switch_probs = batch_outputs["switch_probs"].detach().cpu().numpy()
                batch_switch_logits = batch_outputs["switch_logits"].detach().cpu().numpy()
                batch_default_arms = batch_outputs["default_arms"].detach().cpu().numpy()
            elif isinstance(model, DefaultGainRanker):
                batch_outputs = None
                batch_scores = model.score_samples(batch_samples).detach().cpu().numpy()
                batch_switch_probs = None
                batch_switch_logits = None
                batch_default_arms = np.asarray(
                    [int(model.default_arm_by_problem[str(sample.problem).lower()]) for sample in batch_samples],
                    dtype=np.int64,
                )
            else:
                batch_outputs = None
                batch_scores = model.score_samples(batch_samples).detach().cpu().numpy()
                batch_switch_probs = None
                batch_switch_logits = None
                batch_default_arms = None

        for offset, sample in enumerate(batch_samples):
            raw_scores = batch_scores[offset]
            feasible = np.asarray(sample.feasible_mask) > 0
            if isinstance(model, DefaultGainRanker):
                default_arm = int(batch_default_arms[offset])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

                if getattr(model, "output_mode", "gain") == "binary":
                    raw_probs = 1.0 / (1.0 + np.exp(-raw_scores))
                    decision_scores = np.full_like(raw_probs, -np.inf, dtype=np.float64)
                    for arm in candidate_arms:
                        decision_scores[int(arm)] = raw_probs[int(arm)]
                    decision_scores[default_arm] = 0.0

                    alt_scores = decision_scores.copy()
                    alt_scores[default_arm] = -np.inf
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_prob = (
                        float(alt_scores[best_alt_arm]) if np.isfinite(alt_scores[best_alt_arm]) else float("-inf")
                    )
                    gain_threshold = float(model.decision_threshold_for_problem(str(sample.problem).lower()))
                    if np.isfinite(best_alt_prob) and best_alt_prob > gain_threshold:
                        selected_arm = best_alt_arm
                        selection_reason = "beat_prob_switch"
                    else:
                        selected_arm = default_arm
                        selection_reason = "beat_prob_stay"
                    scores_for_trace = decision_scores
                    extra = {
                        "default_arm": default_arm,
                        "default_arm_name": arm_names[default_arm],
                        "candidate_arms": candidate_arms,
                        "candidate_arm_names": [arm_names[int(arm)] for arm in candidate_arms],
                        "gain_threshold": gain_threshold,
                        "best_alt_arm": best_alt_arm,
                        "best_alt_arm_name": arm_names[best_alt_arm],
                        "best_alt_prob": None if not np.isfinite(best_alt_prob) else float(best_alt_prob),
                        "selection_reason": selection_reason,
                        "output_mode": "binary",
                    }
                elif getattr(model, "output_mode", "gain") == "margin":
                    decision_scores = np.full_like(raw_scores, -np.inf, dtype=np.float64)
                    for arm in candidate_arms:
                        decision_scores[int(arm)] = raw_scores[int(arm)]
                    decision_scores[default_arm] = 0.0

                    alt_scores = decision_scores.copy()
                    alt_scores[default_arm] = -np.inf
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_margin = (
                        float(alt_scores[best_alt_arm]) if np.isfinite(alt_scores[best_alt_arm]) else float("-inf")
                    )
                    margin_threshold = 0.0
                    if np.isfinite(best_alt_margin) and best_alt_margin > margin_threshold:
                        selected_arm = best_alt_arm
                        selection_reason = "margin_switch"
                    else:
                        selected_arm = default_arm
                        selection_reason = "margin_stay"
                    scores_for_trace = decision_scores
                    extra = {
                        "default_arm": default_arm,
                        "default_arm_name": arm_names[default_arm],
                        "candidate_arms": candidate_arms,
                        "candidate_arm_names": [arm_names[int(arm)] for arm in candidate_arms],
                        "margin_threshold": margin_threshold,
                        "best_alt_arm": best_alt_arm,
                        "best_alt_arm_name": arm_names[best_alt_arm],
                        "best_alt_margin": None if not np.isfinite(best_alt_margin) else float(best_alt_margin),
                        "selection_reason": selection_reason,
                        "output_mode": "margin",
                    }
                else:
                    decision_scores = np.full_like(raw_scores, -np.inf, dtype=np.float64)
                    for arm in candidate_arms:
                        decision_scores[int(arm)] = raw_scores[int(arm)]
                    decision_scores[default_arm] = 0.0

                    alt_scores = decision_scores.copy()
                    alt_scores[default_arm] = -np.inf
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_gain = (
                        float(alt_scores[best_alt_arm]) if np.isfinite(alt_scores[best_alt_arm]) else float("-inf")
                    )
                    gain_threshold = float(model.decision_threshold_for_problem(str(sample.problem).lower()))
                    if np.isfinite(best_alt_gain) and best_alt_gain > gain_threshold:
                        selected_arm = best_alt_arm
                        selection_reason = "gain_threshold_switch"
                    else:
                        selected_arm = default_arm
                        selection_reason = "gain_threshold_stay"
                    scores_for_trace = decision_scores
                    extra = {
                        "default_arm": default_arm,
                        "default_arm_name": arm_names[default_arm],
                        "candidate_arms": candidate_arms,
                        "candidate_arm_names": [arm_names[int(arm)] for arm in candidate_arms],
                        "gain_threshold": gain_threshold,
                        "best_alt_arm": best_alt_arm,
                        "best_alt_arm_name": arm_names[best_alt_arm],
                        "best_alt_gain": None if not np.isfinite(best_alt_gain) else float(best_alt_gain),
                        "selection_reason": selection_reason,
                        "output_mode": "gain",
                }
            elif isinstance(model, OneVsDefaultSwitchRanker):
                default_arm = int(batch_default_arms[offset])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)
                threshold = float(model.decision_threshold_for_problem(str(sample.problem).lower()))

                decision_scores = np.full_like(raw_scores, -np.inf, dtype=np.float64)
                for arm in candidate_arms:
                    decision_scores[int(arm)] = raw_scores[int(arm)]
                decision_scores[default_arm] = 0.0

                alt_scores = decision_scores.copy()
                alt_scores[default_arm] = -np.inf
                if np.any(np.isfinite(alt_scores)):
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_score = float(alt_scores[best_alt_arm])
                else:
                    best_alt_arm = default_arm
                    best_alt_score = float("-inf")

                if getattr(model, "output_mode", "") == "coupled_advantage_rank":
                    switch_score = float(batch_switch_logits[offset])
                    if np.isfinite(best_alt_score) and switch_score > threshold:
                        selected_arm = best_alt_arm
                        selection_reason = "switch_advantage_on"
                    else:
                        selected_arm = default_arm
                        selection_reason = "switch_advantage_off"
                    extra = {
                        "default_arm": default_arm,
                        "default_arm_name": arm_names[default_arm],
                        "candidate_arms": candidate_arms,
                        "candidate_arm_names": [arm_names[int(arm)] for arm in candidate_arms],
                        "switch_threshold": threshold,
                        "switch_score": switch_score,
                        "best_alt_arm": best_alt_arm,
                        "best_alt_arm_name": None if best_alt_arm is None else arm_names[int(best_alt_arm)],
                        "best_alt_score": None if not np.isfinite(best_alt_score) else float(best_alt_score),
                        "selection_reason": selection_reason,
                        "output_mode": "coupled_advantage_rank",
                    }
                else:
                    switch_prob = float(batch_switch_probs[offset])
                    if np.isfinite(best_alt_score) and switch_prob > threshold:
                        selected_arm = best_alt_arm
                        selection_reason = "switch_gate_on"
                    else:
                        selected_arm = default_arm
                        selection_reason = "switch_gate_off"
                    extra = {
                        "default_arm": default_arm,
                        "default_arm_name": arm_names[default_arm],
                        "candidate_arms": candidate_arms,
                        "candidate_arm_names": [arm_names[int(arm)] for arm in candidate_arms],
                        "switch_threshold": threshold,
                        "switch_prob": switch_prob,
                        "switch_logit": float(batch_switch_logits[offset]),
                        "best_alt_arm": best_alt_arm,
                        "best_alt_arm_name": None if best_alt_arm is None else arm_names[int(best_alt_arm)],
                        "best_alt_score": None if not np.isfinite(best_alt_score) else float(best_alt_score),
                        "selection_reason": selection_reason,
                        "output_mode": getattr(model, "output_mode", "switch_rank"),
                    }
                scores_for_trace = decision_scores
            else:
                scores = raw_scores
                masked_scores = np.full_like(scores, -np.inf, dtype=np.float64)
                masked_scores[feasible] = scores[feasible]
                selected_arm = int(np.argmax(masked_scores))
                scores_for_trace = scores
                extra = None

            cost = float(sample.costs[selected_arm])
            best_cost = float(np.nanmin(sample.costs[feasible]))
            reward = float(sample.rewards[selected_arm])
            summary_rows.append(
                {
                    "problem": sample.problem,
                    "reward": reward,
                    "cost": cost,
                    "regret": cost - best_cost,
                    "top1": 1.0 if selected_arm == int(sample.best_arm) else 0.0,
                }
            )

            gate_extra = None
            if batch_outputs is not None:
                default_arm = int(batch_default_arms[offset])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)
                alt_scores = np.full_like(scores_for_trace, -np.inf, dtype=np.float64)
                for arm in candidate_arms:
                    alt_scores[int(arm)] = scores_for_trace[int(arm)]
                if np.any(np.isfinite(alt_scores)):
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_score = float(alt_scores[best_alt_arm])
                else:
                    best_alt_arm = None
                    best_alt_score = None
                gate_extra = {
                    "default_arm": default_arm,
                    "default_arm_name": arm_names[default_arm],
                    "candidate_arms": candidate_arms,
                    "candidate_arm_names": [arm_names[int(arm)] for arm in candidate_arms],
                    "best_alt_arm": best_alt_arm,
                    "best_alt_arm_name": None if best_alt_arm is None else arm_names[int(best_alt_arm)],
                    "best_alt_score": best_alt_score,
                }
                if isinstance(model, OneVsDefaultSwitchRanker):
                    gate_extra["switch_threshold"] = float(model.decision_threshold_for_problem(str(sample.problem).lower()))
                    gate_extra["selection_reason"] = extra.get("selection_reason")
                    if getattr(model, "output_mode", "") == "coupled_advantage_rank":
                        gate_extra["switch_score"] = float(batch_switch_logits[offset])
                        gate_extra["output_mode"] = "coupled_advantage_rank"
                    else:
                        gate_extra["switch_prob"] = float(batch_switch_probs[offset])
                        gate_extra["switch_logit"] = float(batch_switch_logits[offset])
                        gate_extra["output_mode"] = getattr(model, "output_mode", "switch_rank")
                else:
                    gate_extra["switch_prob"] = float(batch_switch_probs[offset])
                    gate_extra["switch_logit"] = float(batch_switch_logits[offset])

            row = _trace_row_from_prediction(
                sample=sample,
                selected_arm=selected_arm,
                arm_scores=scores_for_trace,
                arm_names=arm_names,
                stage=stage or "eval",
                order_index=processed,
                epoch=epoch,
                extra=extra if batch_outputs is None else gate_extra,
            )
            _record_trace_row(
                row,
                trace_rows=trace_rows,
                streamer=trace_streamer,
                print_step_json=print_step_json,
                step_log_every=step_log_every,
            )
            processed += 1
            if log_progress and (processed % progress_log_every == 0 or processed == total_count):
                _log(f"    已评测 {processed}/{total_count}...", enabled=True)

    result = _evaluate_pick_results(summary_rows)
    if log_progress:
        overall = result["overall"]
        _log(
            f"{stage or 'eval'} 评测完成 | mean_reward={float(overall['mean_reward']):.6f} | "
            f"top1_accuracy={float(overall['top1_accuracy']):.6f} | "
            f"mean_regret={float(overall['mean_regret']):.6f} | "
            f"mean_cost={float(overall['mean_cost']):.6f}",
            enabled=True,
        )
    return result


def evaluate_baseline_selector(selector, samples, indices) -> dict:
    rows = []
    for idx in indices:
        sample = samples[idx]
        arm = int(selector.select(sample))
        feasible = np.asarray(sample.feasible_mask) > 0
        cost = float(sample.costs[arm])
        best_cost = float(np.nanmin(sample.costs[feasible]))
        reward = float(sample.rewards[arm])
        rows.append(
            {
                "problem": sample.problem,
                "reward": reward,
                "cost": cost,
                "regret": cost - best_cost,
                "top1": 1.0 if arm == int(sample.best_arm) else 0.0,
            }
        )
    return _evaluate_pick_results(rows)


def _search_best_gain_threshold(
    model: DefaultGainRanker,
    samples,
    indices,
    *,
    threshold_grid: list[float],
    batch_size: int,
) -> tuple[dict[str, float], dict]:
    """
    在 val 集上为 DefaultGainRanker 搜索一个收益阈值。

    规则：
    - 预测 gain 最大的 alternative arm
    - 如果 max_gain > threshold，则 switch
    - 否则 stay 在默认 arm
    """
    # 先一次性把 val 上的 raw gain 都算出来，避免每个阈值都重复前向。
    cached = []
    model.eval()
    for start in range(0, len(indices), max(1, int(batch_size))):
        batch_ids = indices[start : start + max(1, int(batch_size))]
        batch_samples = [samples[idx] for idx in batch_ids]
        with torch.no_grad():
            batch_scores = model.score_samples(batch_samples).detach().cpu().numpy()
        for offset, sample in enumerate(batch_samples):
            cached.append((sample, batch_scores[offset]))

    # 逐问题独立搜阈值。
    best_threshold_by_problem: dict[str, float] = {}
    cached_by_problem: dict[str, list[tuple[object, np.ndarray]]] = defaultdict(list)
    for sample, raw_scores in cached:
        cached_by_problem[str(sample.problem).lower()].append((sample, raw_scores))

    for problem, problem_cached in cached_by_problem.items():
        best_cost = None
        best_threshold = float(threshold_grid[0]) if threshold_grid else 0.0
        for threshold in threshold_grid:
            costs = []
            for sample, raw_scores in problem_cached:
                feasible = np.asarray(sample.feasible_mask) > 0
                default_arm = int(model.default_arm_by_problem[str(sample.problem).lower()])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

                alt_scores = np.full_like(raw_scores, -np.inf, dtype=np.float64)
                for arm in candidate_arms:
                    alt_scores[int(arm)] = raw_scores[int(arm)]
                best_alt_arm = int(np.argmax(alt_scores))
                best_alt_gain = float(alt_scores[best_alt_arm]) if np.isfinite(alt_scores[best_alt_arm]) else float("-inf")
                selected_arm = best_alt_arm if np.isfinite(best_alt_gain) and best_alt_gain > float(threshold) else default_arm
                costs.append(float(sample.costs[selected_arm]))
            mean_cost = float(np.mean(costs)) if costs else float("inf")
            if best_cost is None or mean_cost < best_cost - 1e-12:
                best_cost = mean_cost
                best_threshold = float(threshold)
        best_threshold_by_problem[problem] = best_threshold

    # 用选出来的 per-problem threshold 再汇总一次整体指标。
    rows = []
    for sample, raw_scores in cached:
        feasible = np.asarray(sample.feasible_mask) > 0
        problem = str(sample.problem).lower()
        default_arm = int(model.default_arm_by_problem[problem])
        threshold = float(best_threshold_by_problem.get(problem, 0.0))
        candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

        alt_scores = np.full_like(raw_scores, -np.inf, dtype=np.float64)
        for arm in candidate_arms:
            alt_scores[int(arm)] = raw_scores[int(arm)]
        best_alt_arm = int(np.argmax(alt_scores))
        best_alt_gain = float(alt_scores[best_alt_arm]) if np.isfinite(alt_scores[best_alt_arm]) else float("-inf")
        selected_arm = best_alt_arm if np.isfinite(best_alt_gain) and best_alt_gain > threshold else default_arm
        cost = float(sample.costs[selected_arm])
        best_cost = float(np.nanmin(sample.costs[feasible]))
        reward = float(sample.rewards[selected_arm])
        rows.append(
            {
                "problem": sample.problem,
                "reward": reward,
                "cost": cost,
                "regret": cost - best_cost,
                "top1": 1.0 if selected_arm == int(sample.best_arm) else 0.0,
            }
        )
    return best_threshold_by_problem, _evaluate_pick_results(rows)


def _search_best_binary_threshold(
    model: DefaultGainRanker,
    samples,
    indices,
    *,
    threshold_grid: list[float],
    batch_size: int,
) -> tuple[dict[str, float], dict]:
    """
    为 beat-default 二分类器按问题搜索最佳概率阈值。
    """
    cached = []
    model.eval()
    for start in range(0, len(indices), max(1, int(batch_size))):
        batch_ids = indices[start : start + max(1, int(batch_size))]
        batch_samples = [samples[idx] for idx in batch_ids]
        with torch.no_grad():
            batch_logits = model.score_samples(batch_samples)
            batch_probs = torch.sigmoid(batch_logits).detach().cpu().numpy()
        for offset, sample in enumerate(batch_samples):
            cached.append((sample, batch_probs[offset]))

    best_threshold_by_problem: dict[str, float] = {}
    cached_by_problem: dict[str, list[tuple[object, np.ndarray]]] = defaultdict(list)
    for sample, raw_probs in cached:
        cached_by_problem[str(sample.problem).lower()].append((sample, raw_probs))

    for problem, problem_cached in cached_by_problem.items():
        best_cost = None
        best_threshold = float(threshold_grid[0]) if threshold_grid else 0.5
        for threshold in threshold_grid:
            costs = []
            for sample, raw_probs in problem_cached:
                feasible = np.asarray(sample.feasible_mask) > 0
                default_arm = int(model.default_arm_by_problem[str(sample.problem).lower()])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

                alt_probs = np.full_like(raw_probs, -np.inf, dtype=np.float64)
                for arm in candidate_arms:
                    alt_probs[int(arm)] = raw_probs[int(arm)]
                best_alt_arm = int(np.argmax(alt_probs))
                best_alt_prob = float(alt_probs[best_alt_arm]) if np.isfinite(alt_probs[best_alt_arm]) else float("-inf")
                selected_arm = best_alt_arm if np.isfinite(best_alt_prob) and best_alt_prob > float(threshold) else default_arm
                costs.append(float(sample.costs[selected_arm]))
            mean_cost = float(np.mean(costs)) if costs else float("inf")
            if best_cost is None or mean_cost < best_cost - 1e-12:
                best_cost = mean_cost
                best_threshold = float(threshold)
        best_threshold_by_problem[problem] = best_threshold

    rows = []
    for sample, raw_probs in cached:
        feasible = np.asarray(sample.feasible_mask) > 0
        problem = str(sample.problem).lower()
        default_arm = int(model.default_arm_by_problem[problem])
        threshold = float(best_threshold_by_problem.get(problem, 0.5))
        candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

        alt_probs = np.full_like(raw_probs, -np.inf, dtype=np.float64)
        for arm in candidate_arms:
            alt_probs[int(arm)] = raw_probs[int(arm)]
        best_alt_arm = int(np.argmax(alt_probs))
        best_alt_prob = float(alt_probs[best_alt_arm]) if np.isfinite(alt_probs[best_alt_arm]) else float("-inf")
        selected_arm = best_alt_arm if np.isfinite(best_alt_prob) and best_alt_prob > threshold else default_arm
        cost = float(sample.costs[selected_arm])
        best_cost = float(np.nanmin(sample.costs[feasible]))
        reward = float(sample.rewards[selected_arm])
        rows.append(
            {
                "problem": sample.problem,
                "reward": reward,
                "cost": cost,
                "regret": cost - best_cost,
                "top1": 1.0 if selected_arm == int(sample.best_arm) else 0.0,
            }
        )
    return best_threshold_by_problem, _evaluate_pick_results(rows)


def _search_best_switch_rank_threshold(
    model: OneVsDefaultSwitchRanker,
    samples,
    indices,
    *,
    threshold_grid: list[float],
    batch_size: int,
) -> tuple[dict[str, float], dict]:
    """
    为显式 switch gate 模型搜索最佳 switch 概率阈值。
    """
    cached = []
    model.eval()
    for start in range(0, len(indices), max(1, int(batch_size))):
        batch_ids = indices[start : start + max(1, int(batch_size))]
        batch_samples = [samples[idx] for idx in batch_ids]
        with torch.no_grad():
            outputs = model.forward_with_details(batch_samples)
            batch_alt_logits = outputs["alt_logits"].detach().cpu().numpy()
            batch_switch_probs = outputs["switch_probs"].detach().cpu().numpy()
        for offset, sample in enumerate(batch_samples):
            cached.append((sample, float(batch_switch_probs[offset]), batch_alt_logits[offset]))

    best_threshold_by_problem: dict[str, float] = {}
    cached_by_problem: dict[str, list[tuple[object, float, np.ndarray]]] = defaultdict(list)
    for sample, switch_prob, alt_logits in cached:
        cached_by_problem[str(sample.problem).lower()].append((sample, switch_prob, alt_logits))

    for problem, problem_cached in cached_by_problem.items():
        best_cost = None
        best_threshold = float(threshold_grid[0]) if threshold_grid else 0.5
        for threshold in threshold_grid:
            costs = []
            for sample, switch_prob, alt_logits in problem_cached:
                feasible = np.asarray(sample.feasible_mask) > 0
                default_arm = int(model.default_arm_by_problem[str(sample.problem).lower()])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

                alt_scores = np.full_like(alt_logits, -np.inf, dtype=np.float64)
                for arm in candidate_arms:
                    alt_scores[int(arm)] = alt_logits[int(arm)]
                if np.any(np.isfinite(alt_scores)):
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_score = float(alt_scores[best_alt_arm])
                else:
                    best_alt_arm = default_arm
                    best_alt_score = float("-inf")
                selected_arm = best_alt_arm if np.isfinite(best_alt_score) and float(switch_prob) > float(threshold) else default_arm
                costs.append(float(sample.costs[selected_arm]))
            mean_cost = float(np.mean(costs)) if costs else float("inf")
            if best_cost is None or mean_cost < best_cost - 1e-12:
                best_cost = mean_cost
                best_threshold = float(threshold)
        best_threshold_by_problem[problem] = best_threshold

    rows = []
    for sample, switch_prob, alt_logits in cached:
        feasible = np.asarray(sample.feasible_mask) > 0
        problem = str(sample.problem).lower()
        default_arm = int(model.default_arm_by_problem[problem])
        threshold = float(best_threshold_by_problem.get(problem, 0.5))
        candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)

        alt_scores = np.full_like(alt_logits, -np.inf, dtype=np.float64)
        for arm in candidate_arms:
            alt_scores[int(arm)] = alt_logits[int(arm)]
        if np.any(np.isfinite(alt_scores)):
            best_alt_arm = int(np.argmax(alt_scores))
            best_alt_score = float(alt_scores[best_alt_arm])
        else:
            best_alt_arm = default_arm
            best_alt_score = float("-inf")
        selected_arm = best_alt_arm if np.isfinite(best_alt_score) and float(switch_prob) > threshold else default_arm
        cost = float(sample.costs[selected_arm])
        best_cost = float(np.nanmin(sample.costs[feasible]))
        reward = float(sample.rewards[selected_arm])
        rows.append(
            {
                "problem": sample.problem,
                "reward": reward,
                "cost": cost,
                "regret": cost - best_cost,
                "top1": 1.0 if selected_arm == int(sample.best_arm) else 0.0,
            }
        )
    return best_threshold_by_problem, _evaluate_pick_results(rows)


def _search_best_switch_advantage_threshold(
    model: OneVsDefaultSwitchRanker,
    samples,
    indices,
    *,
    threshold_grid: list[float],
    batch_size: int,
) -> tuple[dict[str, float], dict]:
    """
    为 switch advantage 回归模型搜索最佳阈值。
    """
    cached = []
    model.eval()
    for start in range(0, len(indices), max(1, int(batch_size))):
        batch_ids = indices[start : start + max(1, int(batch_size))]
        batch_samples = [samples[idx] for idx in batch_ids]
        with torch.no_grad():
            outputs = model.forward_with_details(batch_samples)
            batch_alt_logits = outputs["alt_logits"].detach().cpu().numpy()
            batch_switch_scores = outputs["switch_logits"].detach().cpu().numpy()
        for offset, sample in enumerate(batch_samples):
            cached.append((sample, float(batch_switch_scores[offset]), batch_alt_logits[offset]))

    best_threshold_by_problem: dict[str, float] = {}
    cached_by_problem: dict[str, list[tuple[object, float, np.ndarray]]] = defaultdict(list)
    for sample, switch_score, alt_logits in cached:
        cached_by_problem[str(sample.problem).lower()].append((sample, switch_score, alt_logits))

    for problem, problem_cached in cached_by_problem.items():
        best_cost = None
        best_threshold = float(threshold_grid[0]) if threshold_grid else 0.0
        for threshold in threshold_grid:
            costs = []
            for sample, switch_score, alt_logits in problem_cached:
                feasible = np.asarray(sample.feasible_mask) > 0
                default_arm = int(model.default_arm_by_problem[str(sample.problem).lower()])
                candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)
                alt_scores = np.full_like(alt_logits, -np.inf, dtype=np.float64)
                for arm in candidate_arms:
                    alt_scores[int(arm)] = alt_logits[int(arm)]
                if np.any(np.isfinite(alt_scores)):
                    best_alt_arm = int(np.argmax(alt_scores))
                    best_alt_score = float(alt_scores[best_alt_arm])
                else:
                    best_alt_arm = default_arm
                    best_alt_score = float("-inf")
                selected_arm = best_alt_arm if np.isfinite(best_alt_score) and float(switch_score) > float(threshold) else default_arm
                costs.append(float(sample.costs[selected_arm]))
            mean_cost = float(np.mean(costs)) if costs else float("inf")
            if best_cost is None or mean_cost < best_cost - 1e-12:
                best_cost = mean_cost
                best_threshold = float(threshold)
        best_threshold_by_problem[problem] = best_threshold

    rows = []
    for sample, switch_score, alt_logits in cached:
        feasible = np.asarray(sample.feasible_mask) > 0
        problem = str(sample.problem).lower()
        default_arm = int(model.default_arm_by_problem[problem])
        threshold = float(best_threshold_by_problem.get(problem, 0.0))
        candidate_arms = _model_candidate_alt_arms(model, sample, feasible, default_arm)
        alt_scores = np.full_like(alt_logits, -np.inf, dtype=np.float64)
        for arm in candidate_arms:
            alt_scores[int(arm)] = alt_logits[int(arm)]
        if np.any(np.isfinite(alt_scores)):
            best_alt_arm = int(np.argmax(alt_scores))
            best_alt_score = float(alt_scores[best_alt_arm])
        else:
            best_alt_arm = default_arm
            best_alt_score = float("-inf")
        selected_arm = best_alt_arm if np.isfinite(best_alt_score) and float(switch_score) > threshold else default_arm
        cost = float(sample.costs[selected_arm])
        best_cost = float(np.nanmin(sample.costs[feasible]))
        reward = float(sample.rewards[selected_arm])
        rows.append(
            {
                "problem": sample.problem,
                "reward": reward,
                "cost": cost,
                "regret": cost - best_cost,
                "top1": 1.0 if selected_arm == int(sample.best_arm) else 0.0,
            }
        )
    return best_threshold_by_problem, _evaluate_pick_results(rows)


def infer_rank_from_reward(selected_reward: float, feasible_count: int, reward_mode: str) -> float:
    if feasible_count <= 1:
        return 1.0
    reward = float(selected_reward)
    if reward_mode == "linear_zero_one":
        return 1.0 + (1.0 - reward) * (feasible_count - 1)
    if reward_mode == "autosaea":
        return feasible_count + 1.0 - reward * feasible_count
    raise ValueError(f"未知 reward_mode: {reward_mode}")


def _aggregate_rank_rows(trace_rows: list[dict], reward_mode: str) -> dict[str, dict]:
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


def _problem_display_name(problem: str) -> str:
    if str(problem).lower() == "tsp":
        return "TSP"
    if str(problem).lower() == "cvrp":
        return "CVRP"
    if str(problem).upper() == "ALL":
        return "ALL"
    return str(problem).upper()


def _format_rate_with_hits(rate: float, hits: int, total: int) -> str:
    return f"{rate:8.4f}  ({hits}/{total})"


def _log_evaluation_report(
    *,
    selector_name: str,
    metrics: dict,
    trace_rows: list[dict],
    baselines: dict | None,
    reward_mode: str,
    enabled: bool = True,
) -> None:
    if not enabled or not trace_rows:
        return

    rank_stats = _aggregate_rank_rows(trace_rows, reward_mode=reward_mode)
    selection_distribution = _aggregate_selection_distribution(trace_rows)
    selected_arm_quality = _aggregate_selected_arm_quality(trace_rows, reward_mode=reward_mode)
    oracle_distribution = _aggregate_oracle_best_distribution(trace_rows)

    lines = []
    lines.append("=" * 70)
    lines.append(f"  评测结果 | {selector_name}")
    lines.append("=" * 70)
    lines.append("")

    for scope in ["ALL", "tsp", "cvrp"]:
        metric_block = metrics["overall"] if scope == "ALL" else metrics["by_problem"].get(scope)
        rank_block = rank_stats.get("ALL" if scope == "ALL" else scope)
        if metric_block is None or rank_block is None:
            continue
        count = int(rank_block["count"])
        lines.append(f"  {_problem_display_name(scope)} ({count} 实例):")
        lines.append(f"    mean_reward:    {float(metric_block['mean_reward']):.4f}")
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
        lines.append(f"    regret:         {float(metric_block['mean_regret']):.4f}")
        lines.append(f"    mean_cost:      {float(metric_block['mean_cost']):.4f}  (模型选择)")
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
                lines.append(f"    oracle mean_cost:                {float(oracle_metrics.get('mean_cost')):.4f}  (每实例选最好的, 上界)")
            lines.append(f"    模型选择 mean_cost:              {float(selector_metrics.get('mean_cost')):.4f}")
            if sb_problem_metrics:
                lines.append(f"    single_best_per_problem:         {float(sb_problem_metrics.get('mean_cost')):.4f}")
                lines.append(
                    f"    gain vs single_best_per_problem: "
                    f"{float(sb_problem_metrics.get('mean_cost')) - float(selector_metrics.get('mean_cost')):+.4f}"
                )
            if sb_global_metrics:
                lines.append(f"    single_best_global:              {float(sb_global_metrics.get('mean_cost')):.4f}")
            if random_metrics:
                lines.append(f"    random mean_cost:                {float(random_metrics.get('mean_cost')):.4f}  (均匀随机的期望)")
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


def _save_checkpoint(
    *,
    output_dir: Path,
    filename: str,
    model: SupervisedRanker,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    global_step: int,
    best_val_mean_cost: float | None,
    extra: dict | None = None,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    payload = {
        "format_version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch) + 1,
        "global_step": int(global_step),
        "best_val_mean_cost": best_val_mean_cost,
        "config": dict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, checkpoint_path)
    torch.save(payload, checkpoint_dir / "latest.pt")
    return checkpoint_path


def _load_checkpoint_model(model: SupervisedRanker, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(payload["model_state_dict"])


def _default_output_dir(loss_name: str) -> Path:
    tag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return Path(__file__).resolve().parent / "outputs" / f"{tag}_{loss_name}"


def run_experiment(args, output_dir: Path | None = None):
    output_dir = output_dir or _default_output_dir(args.loss_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = output_dir / "traces"
    train_metrics_streamer = JsonlTraceStreamer(output_dir / "train_metrics.jsonl")
    periodic_val_streamer = JsonlTraceStreamer(output_dir / "periodic_val.jsonl")
    val_trace_streamer = JsonlTraceStreamer(trace_dir / "selector_val_greedy.jsonl")
    test_trace_streamer = JsonlTraceStreamer(trace_dir / "selector_test_greedy.jsonl")

    _log("开始构建共享数据集", enabled=not args.quiet)
    dataset = build_joint_dataset(
        results_dir=Path(args.results_dir),
        tsp_dataset_path=Path(args.tsp_dataset_path) if args.tsp_dataset_path else None,
        cvrp_dataset_path=Path(args.cvrp_dataset_path) if args.cvrp_dataset_path else None,
        arm_names=list(ALL_METHODS),
        problem_to_methods={k: list(v) for k, v in PROBLEM_TO_METHODS.items()},
        reward_mode=args.reward_mode,
        max_samples_per_problem=args.max_samples_per_problem,
        seed=args.seed,
        log_fn=(lambda msg: _log(msg, enabled=not args.quiet)),
        show_progress=not args.disable_progress,
    )
    _log(f"数据集构建完成，样本总数={len(dataset.samples)}", enabled=not args.quiet)

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

    # 先在 train 集上拟合固定策略 baseline。
    # 后面两处都会用到：
    # 1. 训练时构造 hard-instance 权重
    # 2. val 时判断模型是否真的超过了 per-problem 固定策略
    fixed_baseline_selectors = {
        "single_best_global": SingleBestGlobalSelector.fit(dataset.samples, train_idx, dataset.arm_names),
        "single_best_per_problem": SingleBestPerProblemSelector.fit(dataset.samples, train_idx, dataset.arm_names),
    }
    _log(
        "训练集 baseline 拟合完成 | "
        f"single_best_global={dataset.arm_names[int(fixed_baseline_selectors['single_best_global'].arm)]} | "
        f"single_best_per_problem="
        f"{ {problem: dataset.arm_names[int(arm)] for problem, arm in fixed_baseline_selectors['single_best_per_problem'].best_by_problem.items()} }",
        enabled=not args.quiet,
    )
    problem_mean_hardness = _compute_problem_mean_hardness(
        dataset.samples,
        train_idx,
        fixed_baseline_selectors["single_best_per_problem"],
    )
    _log(
        "训练集 hardness 参考均值 | "
        + " | ".join(f"{problem}={value:.6f}" for problem, value in sorted(problem_mean_hardness.items())),
        enabled=not args.quiet,
    )
    one_vs_default_candidates, one_vs_default_candidate_details = _compute_one_vs_default_candidate_arms(
        dataset.samples,
        train_idx,
        fixed_baseline_selectors["single_best_per_problem"],
        min_positive_rate=args.one_vs_default_min_positive_rate,
        min_positive_count=args.one_vs_default_min_positive_count,
        max_candidates=args.one_vs_default_max_candidates,
    )
    one_vs_default_candidate_priors = _compute_one_vs_default_candidate_gain_priors(
        dataset.samples,
        train_idx,
        default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
        candidate_arms_by_problem=one_vs_default_candidates,
        problem_mean_hardness=problem_mean_hardness,
        normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
    )
    for problem in sorted(one_vs_default_candidates):
        candidate_names = [dataset.arm_names[int(arm)] for arm in one_vs_default_candidates[problem]]
        _log(f"{problem} one-vs-default candidates = {candidate_names}", enabled=not args.quiet)
        if one_vs_default_candidate_priors.get(problem):
            pretty = ", ".join(
                f"{dataset.arm_names[int(arm)]}:{float(value):.4f}"
                for arm, value in sorted(one_vs_default_candidate_priors[problem].items(), key=lambda item: -item[1])
            )
            _log(f"{problem} one-vs-default gain priors = {pretty}", enabled=not args.quiet)
    val_reference_baselines = {
        "single_best_global": evaluate_baseline_selector(
            fixed_baseline_selectors["single_best_global"],
            dataset.samples,
            val_idx,
        ),
        "single_best_per_problem": evaluate_baseline_selector(
            fixed_baseline_selectors["single_best_per_problem"],
            dataset.samples,
            val_idx,
        ),
    }
    _log(
        "val 参考 baseline 已计算 | "
        f"sbp_tsp_cost={float(val_reference_baselines['single_best_per_problem']['by_problem']['tsp']['mean_cost']):.6f} | "
        f"sbp_cvrp_cost={float(val_reference_baselines['single_best_per_problem']['by_problem']['cvrp']['mean_cost']):.6f}",
        enabled=not args.quiet,
    )

    if args.loss_name == "two_stage_default_gate":
        model = StaySwitchRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
    elif args.loss_name == "one_vs_default_two_stage":
        model = OneVsDefaultTwoStageRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
    elif args.loss_name == "one_vs_default_gain":
        model = OneVsDefaultGainRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
        model.set_candidate_gain_priors(one_vs_default_candidate_priors)
    elif args.loss_name == "one_vs_default_beat":
        model = OneVsDefaultBeatRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
        model.set_candidate_gain_priors({})
    elif args.loss_name in ("one_vs_default_margin", "one_vs_default_hard_rank"):
        model = OneVsDefaultMarginRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
        model.set_candidate_gain_priors({})
    elif args.loss_name == "one_vs_default_switch_rank":
        model = OneVsDefaultSwitchRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
    elif args.loss_name == "one_vs_default_coupled_switch_rank":
        model = OneVsDefaultCoupledSwitchRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
    elif args.loss_name == "one_vs_default_switch_advantage":
        model = OneVsDefaultCoupledAdvantageRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
        model.set_candidate_arms(one_vs_default_candidates)
    elif args.loss_name == "default_gain_regression":
        model = DefaultGainRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )
        model.set_default_arms(fixed_baseline_selectors["single_best_per_problem"].best_by_problem)
    else:
        model = SupervisedRanker(
            n_arms=len(dataset.arm_names),
            hidden_dim=args.hidden_dim,
            arm_embed_dim=args.arm_embed_dim,
            device=args.device,
        )

    if args.score_prior_mode != "none" and isinstance(model, SupervisedRanker):
        prior_scores = _compute_problem_arm_prior_scores(
            dataset.samples,
            train_idx,
            dataset.arm_names,
            mode=args.score_prior_mode,
        )
        model.set_problem_arm_priors(prior_scores)
        model.zero_initialize_residual_head()
        for problem in sorted(prior_scores):
            top_items = sorted(
                [
                    (dataset.arm_names[arm_id], float(score))
                    for arm_id, score in enumerate(prior_scores[problem])
                    if dataset.arm_names[arm_id] in PROBLEM_TO_METHODS.get(problem, [])
                ],
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            pretty = ", ".join(f"{name}:{score:+.4f}" for name, score in top_items)
            _log(f"{problem} prior top arms | {pretty}", enabled=not args.quiet)
    elif args.score_prior_mode != "none":
        _log("当前 loss 不使用 score prior，已忽略该设置", enabled=not args.quiet)
    if args.freeze_encoder:
        model.freeze_graph_encoder()
    _log(f"模型构造完成: {model.__class__.__name__}", enabled=not args.quiet)
    if isinstance(model, DefaultGainRanker):
        if getattr(model, "output_mode", "gain") == "binary":
            gain_threshold_grid = _parse_float_grid(args.one_vs_default_beat_prob_grid)
        elif getattr(model, "output_mode", "gain") == "margin":
            gain_threshold_grid = []
        else:
            gain_threshold_grid = _parse_float_grid(args.gain_threshold_grid)
    elif isinstance(model, OneVsDefaultSwitchRanker):
        gain_threshold_grid = _parse_float_grid(args.one_vs_default_beat_prob_grid)
    else:
        gain_threshold_grid = []
    if isinstance(model, DefaultGainRanker):
        if getattr(model, "output_mode", "gain") == "margin":
            _log("margin policy 使用固定决策边界: score > 0 才切换", enabled=not args.quiet)
        else:
            threshold_name = "beat prob threshold grid" if getattr(model, "output_mode", "gain") == "binary" else "default gain threshold grid"
            _log(f"{threshold_name} = {gain_threshold_grid}", enabled=not args.quiet)
    elif isinstance(model, OneVsDefaultSwitchRanker):
        if getattr(model, "output_mode", "") == "coupled_advantage_rank":
            gain_threshold_grid = _parse_float_grid(args.gain_threshold_grid)
            _log(f"switch advantage threshold grid = {gain_threshold_grid}", enabled=not args.quiet)
        else:
            _log(f"switch prob threshold grid = {gain_threshold_grid}", enabled=not args.quiet)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)

    best_val_mean_cost: float | None = None
    best_val_selection_metric: float | None = None
    best_checkpoint_path: Path | None = None
    checkpoint_paths: list[str] = []
    global_step = 0
    epochs_without_improvement = 0
    stop_early = False

    _log(
        f"开始监督排序训练: epochs={args.epochs} | batch_size={args.batch_size} | "
        f"loss={args.loss_name} | hard_alpha={args.hard_instance_weight_alpha} | "
        f"prior={args.score_prior_mode} | switch_scale={args.switch_label_scale} | "
        f"select_by={args.model_selection_metric}",
        enabled=not args.quiet,
    )

    for epoch in range(args.epochs):
        if stop_early:
            break

        model.train()
        order = rng.permutation(train_idx).tolist()
        batch_size = max(1, int(args.batch_size))
        batch_starts = list(range(0, len(order), batch_size))
        progress = _iter_with_progress(
            batch_starts,
            total=len(batch_starts),
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            enabled=not args.disable_progress,
        )

        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        epoch_pair_count = 0
        epoch_weight_sum = 0.0

        for batch_no, start in enumerate(progress):
            batch_ids = order[start : start + batch_size]
            batch_samples = [dataset.samples[idx] for idx in batch_ids]

            if args.loss_name == "two_stage_default_gate":
                batch_outputs = model.forward_with_details(batch_samples)
                scores = batch_outputs["final_scores"]
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                target_gains = None
                positive_gain_weight = None
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
                (
                    feasible_mask,
                    default_arms,
                    best_arms,
                    default_improvements,
                    switch_thresholds,
                    sample_weights,
                    batch_target_stats,
                ) = _make_switch_gate_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    problem_mean_hardness=problem_mean_hardness,
                    switch_label_scale=args.switch_label_scale,
                    switch_positive_weight_alpha=args.switch_positive_weight_alpha,
                    switch_positive_weight_cap=args.switch_positive_weight_cap,
                )
            elif args.loss_name == "one_vs_default_two_stage":
                batch_outputs = model.forward_with_details(batch_samples)
                scores = batch_outputs["final_scores"]
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                target_gains = None
                positive_gain_weight = None
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
                (
                    feasible_mask,
                    default_arms,
                    best_arms,
                    default_improvements,
                    switch_thresholds,
                    sample_weights,
                    batch_target_stats,
                ) = _make_one_vs_default_two_stage_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    problem_mean_hardness=problem_mean_hardness,
                    switch_label_scale=args.switch_label_scale,
                    switch_positive_weight_alpha=args.switch_positive_weight_alpha,
                    switch_positive_weight_cap=args.switch_positive_weight_cap,
                )
            elif args.loss_name == "default_gain_regression":
                batch_outputs = None
                scores = model.score_samples(batch_samples)
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    positive_gain_weight,
                    sample_weights,
                    batch_target_stats,
                ) = _make_default_gain_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                )
            elif args.loss_name == "one_vs_default_gain":
                batch_outputs = None
                scores = model.score_samples(batch_samples)
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    positive_gain_weight,
                    sample_weights,
                    batch_target_stats,
                ) = _make_one_vs_default_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    candidate_gain_prior_by_problem=one_vs_default_candidate_priors,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                )
            elif args.loss_name == "one_vs_default_beat":
                batch_outputs = None
                scores = model.score_samples(batch_samples)
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    positive_gain_weight,
                    sample_weights,
                    batch_target_stats,
                ) = _make_one_vs_default_binary_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                )
            elif args.loss_name == "one_vs_default_margin":
                batch_outputs = None
                scores = model.score_samples(batch_samples)
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    positive_gain_weight,
                    sample_weights,
                    batch_target_stats,
                ) = _make_one_vs_default_margin_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                )
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
            elif args.loss_name == "one_vs_default_hard_rank":
                batch_outputs = None
                scores = model.score_samples(batch_samples)
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    positive_gain_weight,
                    sample_weights,
                    hard_sample_mask,
                    batch_target_stats,
                ) = _make_one_vs_default_hard_rank_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                    switch_label_scale=args.switch_label_scale,
                )
                switch_labels = None
                switch_positive_weights = None
            elif args.loss_name in ("one_vs_default_switch_rank", "one_vs_default_coupled_switch_rank"):
                batch_outputs = model.forward_with_details(batch_samples)
                scores = batch_outputs["alt_logits"]
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                hard_sample_mask = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    switch_labels,
                    switch_positive_weights,
                    sample_weights,
                    batch_target_stats,
                ) = _make_one_vs_default_switch_rank_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                    switch_label_scale=args.switch_label_scale,
                    switch_positive_weight_alpha=args.switch_positive_weight_alpha,
                    switch_positive_weight_cap=args.switch_positive_weight_cap,
                )
                positive_gain_weight = None
                switch_targets = None
            elif args.loss_name == "one_vs_default_switch_advantage":
                batch_outputs = model.forward_with_details(batch_samples)
                scores = batch_outputs["alt_logits"]
                rewards = torch.zeros_like(scores)
                normalized_regrets = None
                best_arms = None
                default_improvements = None
                switch_thresholds = None
                hard_sample_mask = None
                switch_labels = None
                (
                    feasible_mask,
                    default_arms,
                    target_gains,
                    switch_targets,
                    switch_positive_weights,
                    sample_weights,
                    batch_target_stats,
                ) = _make_one_vs_default_switch_advantage_targets(
                    batch_samples,
                    model.device,
                    default_arms_by_problem=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    candidate_arms_by_problem=one_vs_default_candidates,
                    problem_mean_hardness=problem_mean_hardness,
                    normalize_by_problem_hardness=args.gain_normalize_by_problem_hardness,
                    positive_weight_alpha=args.gain_positive_weight_alpha,
                )
                positive_gain_weight = None
            else:
                batch_outputs = None
                scores = model.score_samples(batch_samples)
                default_arms = None
                default_improvements = None
                switch_thresholds = None
                target_gains = None
                positive_gain_weight = None
                hard_sample_mask = None
                switch_labels = None
                switch_positive_weights = None
                switch_targets = None
                (
                    feasible_mask,
                    rewards,
                    _costs,
                    normalized_regrets,
                    best_arms,
                    sample_weights,
                    batch_target_stats,
                ) = _make_batch_tensors(
                    batch_samples,
                    model.device,
                    single_best_per_problem_arms=fixed_baseline_selectors["single_best_per_problem"].best_by_problem,
                    problem_mean_hardness=problem_mean_hardness,
                    hard_instance_weight_alpha=args.hard_instance_weight_alpha,
                    hard_instance_weight_cap=args.hard_instance_weight_cap,
                )
            loss, loss_stats = compute_ranking_loss(
                loss_name=args.loss_name,
                scores=scores,
                rewards=rewards,
                feasible_mask=feasible_mask,
                normalized_regrets=normalized_regrets,
                best_arms=best_arms,
                sample_weights=sample_weights,
                switch_logits=None if batch_outputs is None else batch_outputs["switch_logits"],
                alt_logits=None if batch_outputs is None else batch_outputs["alt_logits"],
                default_arms=default_arms,
                default_improvements=default_improvements,
                switch_thresholds=switch_thresholds,
                target_gains=target_gains,
                positive_gain_weight=positive_gain_weight,
                hard_sample_mask=hard_sample_mask,
                switch_labels=switch_labels,
                switch_positive_weights=switch_positive_weights,
                switch_targets=switch_targets,
                zero_gain_weight=args.gain_zero_weight,
                gain_loss_type=args.gain_loss_type,
                pairwise_gap_weight=args.pairwise_gap_weight,
                listwise_target_scale=args.listwise_target_scale,
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer.step()

            global_step += 1
            batch_len = len(batch_samples)
            batch_loss_value = float(loss.detach().cpu().item())
            epoch_loss_sum += batch_loss_value * batch_len
            epoch_sample_count += batch_len
            epoch_pair_count += int(loss_stats.get("pair_count", loss_stats.get("ranking_pair_count", 0)))
            epoch_weight_sum += float(batch_target_stats.get("mean_sample_weight", 1.0)) * batch_len

            train_metrics_streamer.write(
                {
                    "epoch": int(epoch) + 1,
                    "global_step": int(global_step),
                    "batch_index": int(batch_no),
                    "batch_size": int(batch_len),
                    "loss": batch_loss_value,
                    **batch_target_stats,
                    **{key: value for key, value in loss_stats.items() if key != "loss_name"},
                }
            )

            if (batch_no + 1) % max(1, int(args.train_log_every)) == 0 or (batch_no + 1) == len(batch_starts):
                mean_loss_so_far = epoch_loss_sum / max(epoch_sample_count, 1)
                _log(
                    f"epoch={epoch + 1} | step={global_step} | "
                    f"batch={batch_no + 1}/{len(batch_starts)} | "
                    f"loss={batch_loss_value:.6f} | mean_loss={mean_loss_so_far:.6f} | "
                    f"mean_sample_weight={float(batch_target_stats.get('mean_sample_weight', 1.0)):.4f} | "
                    f"{'switch_pos_rate' if args.loss_name in ('two_stage_default_gate', 'one_vs_default_two_stage', 'one_vs_default_switch_rank', 'one_vs_default_coupled_switch_rank', 'one_vs_default_switch_advantage') else ('positive_rate' if args.loss_name in ('default_gain_regression', 'one_vs_default_gain', 'one_vs_default_beat', 'one_vs_default_margin', 'one_vs_default_hard_rank') else 'mean_hardness')}="
                    f"{float(batch_target_stats.get('switch_positive_rate', batch_target_stats.get('positive_gain_rate', batch_target_stats.get('positive_label_rate', batch_target_stats.get('positive_margin_rate', batch_target_stats.get('mean_hardness', 0.0)))))):.6f}"
                    + (
                        f" | hard_rate={float(batch_target_stats.get('hard_sample_rate', 0.0)):.6f}"
                        if args.loss_name == "one_vs_default_hard_rank"
                        else ""
                    ),
                    enabled=not args.quiet,
                )
                if args.loss_name in ("one_vs_default_switch_rank", "one_vs_default_coupled_switch_rank", "one_vs_default_switch_advantage"):
                    _log(
                        f"    gate_loss={float(loss_stats.get('gate_loss', 0.0)):.6f} | "
                        f"rank_loss={float(loss_stats.get('rank_loss', 0.0)):.6f} | "
                        f"{'mean_switch_score' if args.loss_name == 'one_vs_default_switch_advantage' else 'mean_switch_prob'}="
                        f"{float(loss_stats.get('mean_switch_score', loss_stats.get('mean_switch_prob', 0.0))):.6f} | "
                        f"rank_pairs={int(loss_stats.get('ranking_pair_count', 0))}",
                        enabled=not args.quiet,
                    )

            if output_dir is not None and int(args.checkpoint_every_steps) > 0:
                if global_step % int(args.checkpoint_every_steps) == 0:
                    checkpoint_path = _save_checkpoint(
                        output_dir=output_dir,
                        filename=f"step_{global_step:07d}.pt",
                        model=model,
                        optimizer=optimizer,
                        config=vars(args),
                        epoch=epoch,
                        global_step=global_step,
                        best_val_mean_cost=best_val_mean_cost,
                        extra={"save_reason": "periodic_step"},
                    )
                    checkpoint_paths.append(str(checkpoint_path))
                    _log(f"checkpoint 已保存: {checkpoint_path}", enabled=not args.quiet)

            if int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
                if isinstance(model, DefaultGainRanker):
                    if getattr(model, "output_mode", "gain") == "binary":
                        best_threshold_by_problem, periodic_metrics = _search_best_binary_threshold(
                            model,
                            dataset.samples,
                            val_idx,
                            threshold_grid=gain_threshold_grid,
                            batch_size=args.eval_batch_size,
                        )
                        model.set_decision_thresholds(best_threshold_by_problem)
                    elif getattr(model, "output_mode", "gain") == "margin":
                        periodic_metrics = evaluate_ranker(
                            model,
                            dataset.samples,
                            val_idx,
                            dataset.arm_names,
                            batch_size=args.eval_batch_size,
                            stage="val",
                            epoch=epoch,
                            show_progress=False,
                            log_progress=False,
                        )
                    else:
                        best_threshold_by_problem, periodic_metrics = _search_best_gain_threshold(
                            model,
                            dataset.samples,
                            val_idx,
                            threshold_grid=gain_threshold_grid,
                            batch_size=args.eval_batch_size,
                        )
                        model.set_decision_thresholds(best_threshold_by_problem)
                elif isinstance(model, OneVsDefaultSwitchRanker):
                    if getattr(model, "output_mode", "") == "coupled_advantage_rank":
                        best_threshold_by_problem, periodic_metrics = _search_best_switch_advantage_threshold(
                            model,
                            dataset.samples,
                            val_idx,
                            threshold_grid=gain_threshold_grid,
                            batch_size=args.eval_batch_size,
                        )
                    else:
                        best_threshold_by_problem, periodic_metrics = _search_best_switch_rank_threshold(
                            model,
                            dataset.samples,
                            val_idx,
                            threshold_grid=gain_threshold_grid,
                            batch_size=args.eval_batch_size,
                        )
                    model.set_decision_thresholds(best_threshold_by_problem)
                else:
                    periodic_metrics = evaluate_ranker(
                        model,
                        dataset.samples,
                        val_idx,
                        dataset.arm_names,
                        batch_size=args.eval_batch_size,
                        stage="val",
                        epoch=epoch,
                        show_progress=False,
                        log_progress=False,
                    )
                periodic_gains = _compute_cost_gains_against_baseline(
                    periodic_metrics,
                    val_reference_baselines["single_best_per_problem"],
                )
                periodic_val_streamer.write(
                    {
                        "source": "step",
                        "epoch": int(epoch) + 1,
                        "global_step": int(global_step),
                        "gain_threshold": None if not hasattr(model, "decision_threshold_by_problem") else dict(model.decision_threshold_by_problem),
                        "train_mean_loss": epoch_loss_sum / max(epoch_sample_count, 1),
                        "gains_vs_single_best_per_problem": periodic_gains,
                        "metrics": periodic_metrics,
                    }
                )
                _log(
                    f"step={global_step} | periodic val | "
                    f"{_format_problem_cost_line(periodic_metrics)} | "
                    f"gain_vs_sbp_macro={float(periodic_gains['macro_mean_cost_gain']):+.6f}",
                    enabled=not args.quiet,
                )

        epoch_mean_loss = epoch_loss_sum / max(epoch_sample_count, 1)
        _log(
            f"完成 epoch {epoch + 1}/{args.epochs} | train_mean_loss={epoch_mean_loss:.6f} | "
            f"pair_count={epoch_pair_count} | mean_epoch_sample_weight={epoch_weight_sum / max(epoch_sample_count, 1):.4f}",
            enabled=not args.quiet,
        )

        if output_dir is not None and int(args.checkpoint_every_epochs) > 0:
            if (epoch + 1) % int(args.checkpoint_every_epochs) == 0:
                checkpoint_path = _save_checkpoint(
                    output_dir=output_dir,
                    filename=f"epoch_{epoch + 1:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=vars(args),
                    epoch=epoch,
                    global_step=global_step,
                    best_val_mean_cost=best_val_mean_cost,
                    extra={"save_reason": "periodic_epoch"},
                )
                checkpoint_paths.append(str(checkpoint_path))
                _log(f"checkpoint 已保存: {checkpoint_path}", enabled=not args.quiet)

        should_eval_epoch = int(args.eval_every_epochs) > 0 and ((epoch + 1) % int(args.eval_every_epochs) == 0)
        if should_eval_epoch or epoch == args.epochs - 1:
            if isinstance(model, DefaultGainRanker):
                if getattr(model, "output_mode", "gain") == "binary":
                    best_threshold_by_problem, val_metrics = _search_best_binary_threshold(
                        model,
                        dataset.samples,
                        val_idx,
                        threshold_grid=gain_threshold_grid,
                        batch_size=args.eval_batch_size,
                    )
                    model.set_decision_thresholds(best_threshold_by_problem)
                    _log(
                        "val beat 概率阈值搜索完成 | "
                        + " | ".join(f"{problem}={value:.4f}" for problem, value in sorted(model.decision_threshold_by_problem.items())),
                        enabled=not args.quiet,
                    )
                elif getattr(model, "output_mode", "gain") == "margin":
                    val_metrics = evaluate_ranker(
                        model,
                        dataset.samples,
                        val_idx,
                        dataset.arm_names,
                        batch_size=args.eval_batch_size,
                        stage="val",
                        epoch=epoch,
                        show_progress=not args.disable_progress,
                        log_progress=not args.quiet,
                        progress_log_every=args.eval_log_every,
                    )
                else:
                    best_threshold_by_problem, val_metrics = _search_best_gain_threshold(
                        model,
                        dataset.samples,
                        val_idx,
                        threshold_grid=gain_threshold_grid,
                        batch_size=args.eval_batch_size,
                    )
                    model.set_decision_thresholds(best_threshold_by_problem)
                    _log(
                        "val 阈值搜索完成 | "
                        + " | ".join(f"{problem}={value:.4f}" for problem, value in sorted(model.decision_threshold_by_problem.items())),
                        enabled=not args.quiet,
                    )
            elif isinstance(model, OneVsDefaultSwitchRanker):
                if getattr(model, "output_mode", "") == "coupled_advantage_rank":
                    best_threshold_by_problem, val_metrics = _search_best_switch_advantage_threshold(
                        model,
                        dataset.samples,
                        val_idx,
                        threshold_grid=gain_threshold_grid,
                        batch_size=args.eval_batch_size,
                    )
                    label = "val switch advantage 阈值搜索完成"
                else:
                    best_threshold_by_problem, val_metrics = _search_best_switch_rank_threshold(
                        model,
                        dataset.samples,
                        val_idx,
                        threshold_grid=gain_threshold_grid,
                        batch_size=args.eval_batch_size,
                    )
                    label = "val switch 阈值搜索完成"
                model.set_decision_thresholds(best_threshold_by_problem)
                _log(
                    label + " | "
                    + " | ".join(f"{problem}={value:.4f}" for problem, value in sorted(model.decision_threshold_by_problem.items())),
                    enabled=not args.quiet,
                )
            else:
                val_metrics = evaluate_ranker(
                    model,
                    dataset.samples,
                    val_idx,
                    dataset.arm_names,
                    batch_size=args.eval_batch_size,
                    stage="val",
                    epoch=epoch,
                    show_progress=not args.disable_progress,
                    log_progress=not args.quiet,
                    progress_log_every=args.eval_log_every,
                )
            current_val_cost = float(val_metrics["overall"]["mean_cost"])
            val_gains = _compute_cost_gains_against_baseline(
                val_metrics,
                val_reference_baselines["single_best_per_problem"],
            )
            if args.model_selection_metric == "gain_vs_single_best_per_problem":
                current_val_selection_metric = float(val_gains["macro_mean_cost_gain"])
                better = (
                    best_val_selection_metric is None
                    or current_val_selection_metric > float(best_val_selection_metric) + 1e-12
                )
            elif args.model_selection_metric == "overall_mean_cost":
                current_val_selection_metric = -float(current_val_cost)
                better = (
                    best_val_selection_metric is None
                    or current_val_selection_metric > float(best_val_selection_metric) + 1e-12
                )
            else:
                raise ValueError(f"未知 model_selection_metric: {args.model_selection_metric}")
            periodic_val_streamer.write(
                {
                    "source": "epoch",
                    "epoch": int(epoch) + 1,
                    "global_step": int(global_step),
                    "gain_threshold": None if not hasattr(model, "decision_threshold_by_problem") else dict(model.decision_threshold_by_problem),
                    "train_mean_loss": epoch_mean_loss,
                    "gains_vs_single_best_per_problem": val_gains,
                    "metrics": val_metrics,
                }
            )
            _log(
                f"epoch={epoch + 1} | val | {_format_problem_cost_line(val_metrics)} | "
                f"gain_vs_sbp_macro={float(val_gains['macro_mean_cost_gain']):+.6f}",
                enabled=not args.quiet,
            )

            if better:
                best_val_mean_cost = current_val_cost
                best_val_selection_metric = current_val_selection_metric
                best_checkpoint_path = _save_checkpoint(
                    output_dir=output_dir,
                    filename="best.pt",
                    model=model,
                    optimizer=optimizer,
                    config=vars(args),
                    epoch=epoch,
                    global_step=global_step,
                    best_val_mean_cost=best_val_mean_cost,
                    extra={"save_reason": "best_val"},
                )
                checkpoint_paths.append(str(best_checkpoint_path))
                epochs_without_improvement = 0
                _log(
                    f"发现新的 best val | metric={args.model_selection_metric} | "
                    f"selection_score={best_val_selection_metric:.6f} | "
                    f"overall_mean_cost={best_val_mean_cost:.6f} | checkpoint={best_checkpoint_path}",
                    enabled=not args.quiet,
                )
            else:
                epochs_without_improvement += 1
                _log(
                    f"val 未提升 | metric={args.model_selection_metric} | "
                    f"best_selection_score={float(best_val_selection_metric):.6f} | "
                    f"current_selection_score={float(current_val_selection_metric):.6f} | "
                    f"best_cost={float(best_val_mean_cost):.6f} | current_cost={current_val_cost:.6f} | "
                    f"patience={epochs_without_improvement}/{args.early_stopping_patience if args.early_stopping_patience > 0 else 'off'}",
                    enabled=not args.quiet,
                )

            if int(args.early_stopping_patience) > 0 and epochs_without_improvement >= int(args.early_stopping_patience):
                _log("触发早停，结束训练", enabled=not args.quiet)
                stop_early = True

    final_checkpoint_path = _save_checkpoint(
        output_dir=output_dir,
        filename="final.pt",
        model=model,
        optimizer=optimizer,
        config=vars(args),
        epoch=max(0, min(args.epochs - 1, epoch if args.epochs > 0 else 0)),
        global_step=global_step,
        best_val_mean_cost=best_val_mean_cost,
        extra={"save_reason": "final"},
    )
    checkpoint_paths.append(str(final_checkpoint_path))
    _log(f"最终 checkpoint 已保存: {final_checkpoint_path}", enabled=not args.quiet)

    if best_checkpoint_path is not None:
        _log(f"载入 best checkpoint 做最终评测: {best_checkpoint_path}", enabled=not args.quiet)
        _load_checkpoint_model(model, best_checkpoint_path)

    if isinstance(model, DefaultGainRanker):
        if getattr(model, "output_mode", "gain") == "binary":
            best_threshold_by_problem, _ = _search_best_binary_threshold(
                model,
                dataset.samples,
                val_idx,
                threshold_grid=gain_threshold_grid,
                batch_size=args.eval_batch_size,
            )
            model.set_decision_thresholds(best_threshold_by_problem)
            _log(
                "最终评测前重新用 val 搜索 beat 概率阈值 | "
                + " | ".join(f"{problem}={value:.4f}" for problem, value in sorted(model.decision_threshold_by_problem.items())),
                enabled=not args.quiet,
            )
        elif getattr(model, "output_mode", "gain") == "margin":
            model.set_decision_thresholds({"tsp": 0.0, "cvrp": 0.0})
            _log("最终评测使用固定 margin 边界: score > 0 才切换", enabled=not args.quiet)
        else:
            best_threshold_by_problem, _ = _search_best_gain_threshold(
                model,
                dataset.samples,
                val_idx,
                threshold_grid=gain_threshold_grid,
                batch_size=args.eval_batch_size,
            )
            model.set_decision_thresholds(best_threshold_by_problem)
            _log(
                "最终评测前重新用 val 搜索 gain threshold | "
                + " | ".join(f"{problem}={value:.4f}" for problem, value in sorted(model.decision_threshold_by_problem.items())),
                enabled=not args.quiet,
            )
    elif isinstance(model, OneVsDefaultSwitchRanker):
        if getattr(model, "output_mode", "") == "coupled_advantage_rank":
            best_threshold_by_problem, _ = _search_best_switch_advantage_threshold(
                model,
                dataset.samples,
                val_idx,
                threshold_grid=gain_threshold_grid,
                batch_size=args.eval_batch_size,
            )
            label = "最终评测前重新用 val 搜索 switch advantage threshold"
        else:
            best_threshold_by_problem, _ = _search_best_switch_rank_threshold(
                model,
                dataset.samples,
                val_idx,
                threshold_grid=gain_threshold_grid,
                batch_size=args.eval_batch_size,
            )
            label = "最终评测前重新用 val 搜索 switch threshold"
        model.set_decision_thresholds(best_threshold_by_problem)
        _log(
            label + " | "
            + " | ".join(f"{problem}={value:.4f}" for problem, value in sorted(model.decision_threshold_by_problem.items())),
            enabled=not args.quiet,
        )

    val_trace: list[dict] = []
    test_trace: list[dict] = []
    final_results = {
        "config": vars(args),
        "splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "model": {
            "val_greedy": evaluate_ranker(
                model,
                dataset.samples,
                val_idx,
                dataset.arm_names,
                batch_size=args.eval_batch_size,
                stage="val",
                show_progress=not args.disable_progress,
                log_progress=not args.quiet,
                progress_log_every=args.eval_log_every,
                trace_rows=val_trace,
                trace_streamer=val_trace_streamer,
                print_step_json=args.print_step_json,
                step_log_every=args.step_log_every,
            ),
            "test_greedy": evaluate_ranker(
                model,
                dataset.samples,
                test_idx,
                dataset.arm_names,
                batch_size=args.eval_batch_size,
                stage="test",
                show_progress=not args.disable_progress,
                log_progress=not args.quiet,
                progress_log_every=args.eval_log_every,
                trace_rows=test_trace,
                trace_streamer=test_trace_streamer,
                print_step_json=args.print_step_json,
                step_log_every=args.step_log_every,
            ),
        },
    }

    _log("开始计算 baseline 对比", enabled=not args.quiet)
    baselines = {
        "random": RandomSelector(seed=args.seed),
        "single_best_global": fixed_baseline_selectors["single_best_global"],
        "single_best_per_problem": fixed_baseline_selectors["single_best_per_problem"],
        "oracle": OracleSelector(),
    }
    final_results["baselines"] = {
        name: evaluate_baseline_selector(selector_obj, dataset.samples, test_idx)
        for name, selector_obj in baselines.items()
    }
    _log("baseline 对比完成", enabled=not args.quiet)

    final_results["comparisons"] = {
        "val_vs_single_best_per_problem": _compute_cost_gains_against_baseline(
            final_results["model"]["val_greedy"],
            val_reference_baselines["single_best_per_problem"],
        ),
        "test_vs_single_best_per_problem": _compute_cost_gains_against_baseline(
            final_results["model"]["test_greedy"],
            final_results["baselines"]["single_best_per_problem"],
        ),
    }

    if not args.quiet:
        _log_evaluation_report(
            selector_name=model.__class__.__name__,
            metrics=final_results["model"]["test_greedy"],
            trace_rows=test_trace,
            baselines=final_results["baselines"],
            reward_mode=args.reward_mode,
            enabled=True,
        )

    summary = {
        **final_results,
        "one_vs_default": {
            "candidate_arms": one_vs_default_candidates,
            "candidate_details": one_vs_default_candidate_details,
            "candidate_gain_priors": one_vs_default_candidate_priors,
            "decision_threshold": None if not hasattr(model, "decision_threshold_by_problem") else dict(model.decision_threshold_by_problem),
        },
        "best_model": {
            "best_val_mean_cost": best_val_mean_cost,
            "best_val_selection_metric": best_val_selection_metric,
            "model_selection_metric": args.model_selection_metric,
            "best_checkpoint": None if best_checkpoint_path is None else str(best_checkpoint_path),
            "global_step": int(global_step),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "train_metrics": str(output_dir / "train_metrics.jsonl"),
            "periodic_val": str(output_dir / "periodic_val.jsonl"),
            "val_trace": str(trace_dir / "selector_val_greedy.jsonl"),
            "test_trace": str(trace_dir / "selector_test_greedy.jsonl"),
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "checkpoint_files": checkpoint_paths,
            "implementation_doc": str(Path(__file__).resolve().parent / "IMPLEMENTATION.md"),
            "running_doc": str(Path(__file__).resolve().parent / "运行说明.md"),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"summary 已写出: {summary_path}", enabled=not args.quiet)

    train_metrics_streamer.close()
    periodic_val_streamer.close()
    val_trace_streamer.close()
    test_trace_streamer.close()
    return summary, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSP/CVRP 初始化方法选择的 supervised learning-to-rank",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=RunL2RDefaults.RESULTS_DIR)
    parser.add_argument("--tsp-dataset-path", default=RunL2RDefaults.TSP_DATASET_PATH)
    parser.add_argument("--cvrp-dataset-path", default=RunL2RDefaults.CVRP_DATASET_PATH)
    parser.add_argument("--reward-mode", choices=["linear_zero_one", "autosaea"], default=RunL2RDefaults.REWARD_MODE)
    parser.add_argument("--train-ratio", type=float, default=RunL2RDefaults.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=RunL2RDefaults.VAL_RATIO)
    parser.add_argument("--seed", type=int, default=RunL2RDefaults.SEED)
    parser.add_argument("--max-samples-per-problem", type=int, default=RunL2RDefaults.MAX_SAMPLES_PER_PROBLEM)

    parser.add_argument("--hidden-dim", type=int, default=RunL2RDefaults.HIDDEN_DIM)
    parser.add_argument("--arm-embed-dim", type=int, default=RunL2RDefaults.ARM_EMBED_DIM)
    parser.add_argument("--epochs", type=int, default=RunL2RDefaults.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=RunL2RDefaults.BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=RunL2RDefaults.EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=RunL2RDefaults.LR)
    parser.add_argument("--weight-decay", type=float, default=RunL2RDefaults.WEIGHT_DECAY)
    parser.add_argument("--grad-clip-norm", type=float, default=RunL2RDefaults.GRAD_CLIP_NORM)

    parser.add_argument(
        "--loss-name",
        choices=[
            "pairwise_logistic",
            "listwise_ce",
            "best_vs_all_regret",
            "two_stage_default_gate",
            "one_vs_default_two_stage",
            "default_gain_regression",
            "one_vs_default_gain",
            "one_vs_default_beat",
            "one_vs_default_margin",
            "one_vs_default_hard_rank",
            "one_vs_default_switch_rank",
            "one_vs_default_coupled_switch_rank",
            "one_vs_default_switch_advantage",
        ],
        default=RunL2RDefaults.LOSS_NAME,
    )
    _add_bool_argument(parser, "--pairwise-gap-weight", RunL2RDefaults.PAIRWISE_GAP_WEIGHT, "是否按 reward gap 对 pairwise loss 加权")
    parser.add_argument("--listwise-target-scale", type=float, default=RunL2RDefaults.LISTWISE_TARGET_SCALE)
    parser.add_argument("--hard-instance-weight-alpha", type=float, default=RunL2RDefaults.HARD_INSTANCE_WEIGHT_ALPHA)
    parser.add_argument("--hard-instance-weight-cap", type=float, default=RunL2RDefaults.HARD_INSTANCE_WEIGHT_CAP)
    parser.add_argument(
        "--score-prior-mode",
        choices=["none", "problem_arm_avg_neg_regret"],
        default=RunL2RDefaults.SCORE_PRIOR_MODE,
    )
    parser.add_argument("--switch-label-scale", type=float, default=RunL2RDefaults.SWITCH_LABEL_SCALE)
    parser.add_argument(
        "--switch-positive-weight-alpha",
        type=float,
        default=RunL2RDefaults.SWITCH_POSITIVE_WEIGHT_ALPHA,
    )
    parser.add_argument(
        "--switch-positive-weight-cap",
        type=float,
        default=RunL2RDefaults.SWITCH_POSITIVE_WEIGHT_CAP,
    )
    _add_bool_argument(
        parser,
        "--gain-normalize-by-problem-hardness",
        RunL2RDefaults.GAIN_NORMALIZE_BY_PROBLEM_HARDNESS,
        "default gain 回归时是否按问题 hardness 归一化",
    )
    parser.add_argument("--gain-positive-weight-alpha", type=float, default=RunL2RDefaults.GAIN_POSITIVE_WEIGHT_ALPHA)
    parser.add_argument("--gain-zero-weight", type=float, default=RunL2RDefaults.GAIN_ZERO_WEIGHT)
    parser.add_argument("--gain-loss-type", choices=["smooth_l1", "mse"], default=RunL2RDefaults.GAIN_LOSS_TYPE)
    parser.add_argument("--gain-threshold-grid", default=RunL2RDefaults.GAIN_THRESHOLD_GRID)
    parser.add_argument("--one-vs-default-beat-prob-grid", default=RunL2RDefaults.ONE_VS_DEFAULT_BEAT_PROB_GRID)
    parser.add_argument("--one-vs-default-min-positive-rate", type=float, default=RunL2RDefaults.ONE_VS_DEFAULT_MIN_POSITIVE_RATE)
    parser.add_argument("--one-vs-default-min-positive-count", type=int, default=RunL2RDefaults.ONE_VS_DEFAULT_MIN_POSITIVE_COUNT)
    parser.add_argument("--one-vs-default-max-candidates", type=int, default=RunL2RDefaults.ONE_VS_DEFAULT_MAX_CANDIDATES)
    parser.add_argument(
        "--model-selection-metric",
        choices=["overall_mean_cost", "gain_vs_single_best_per_problem"],
        default=RunL2RDefaults.MODEL_SELECTION_METRIC,
    )

    parser.add_argument("--train-log-every", type=int, default=RunL2RDefaults.TRAIN_LOG_EVERY)
    parser.add_argument("--eval-every-epochs", type=int, default=RunL2RDefaults.EVAL_EVERY_EPOCHS)
    parser.add_argument("--eval-every-steps", type=int, default=RunL2RDefaults.EVAL_EVERY_STEPS)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=RunL2RDefaults.CHECKPOINT_EVERY_EPOCHS)
    parser.add_argument("--checkpoint-every-steps", type=int, default=RunL2RDefaults.CHECKPOINT_EVERY_STEPS)
    parser.add_argument("--early-stopping-patience", type=int, default=RunL2RDefaults.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--eval-log-every", type=int, default=RunL2RDefaults.EVAL_LOG_EVERY)

    _add_bool_argument(parser, "--freeze-encoder", RunL2RDefaults.FREEZE_ENCODER, "是否冻结底层 NSS 图编码器")
    parser.add_argument("--device", default=RunL2RDefaults.DEVICE)
    parser.add_argument("--output-dir", default=RunL2RDefaults.OUTPUT_DIR)
    _add_bool_argument(parser, "--disable-progress", RunL2RDefaults.DISABLE_PROGRESS, "是否关闭 tqdm 进度条")
    _add_bool_argument(parser, "--quiet", RunL2RDefaults.QUIET, "是否减少阶段日志输出")
    _add_bool_argument(parser, "--print-step-json", RunL2RDefaults.PRINT_STEP_JSON, "是否把评测 trace 实时打印到终端")
    parser.add_argument("--step-log-every", type=int, default=RunL2RDefaults.STEP_LOG_EVERY)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.loss_name)
    run_experiment(args, output_dir=output_dir)


if __name__ == "__main__":
    main()
