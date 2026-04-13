from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm 不是强依赖
    tqdm = None

if __package__ in (None, ""):
    from arm_config import ALL_METHODS, PROBLEM_TO_METHODS, get_mask, get_problem_onehot  # type: ignore
    from features import extract_manual_feature_bundle  # type: ignore
    from reward import compute_rank_based_rewards  # type: ignore
else:
    from .arm_config import ALL_METHODS, PROBLEM_TO_METHODS, get_mask, get_problem_onehot
    from .features import extract_manual_feature_bundle
    from .reward import compute_rank_based_rewards


@dataclass
class RunInfo:
    """
    记录一个 EasyNCO 测试 run 的最小必要信息。

    我们这里只关心 bandit 数据构建真正需要的字段：
    - problem / method: 属于哪个问题、哪个方法
    - run_dir: 结果目录
    - timestamp: 用于“最新 run”选择
    - test_data_path: 指向原始 .pt 数据集
    - is_init_only: 确认它是不是 NoIteration 的结果
    """

    problem: str
    method: str
    run_dir: Path
    timestamp: str
    test_data_path: str
    is_init_only: bool


@dataclass
class InstanceSample:
    """
    这是 bandit 训练/评测的核心样本结构。

    一条样本对应“一个实例”。
    它同时包含：
    - 原始实例张量 nodes
    - 手工特征
    - LinUCB 上下文
    - feasible mask
    - 所有方法在该实例上的真实 cost
    - 从 cost 转出来的 reward / rank

    也就是说，这个结构已经把
    “实例上下文 + 可行动作集合 + 每个动作的离线表现”
    聚合到了一起。
    """

    uid: str
    problem: str
    global_index: int
    nodes: torch.Tensor
    problem_onehot: np.ndarray
    manual_raw: np.ndarray
    manual_with_scale: np.ndarray
    linucb_context: np.ndarray
    feasible_mask: np.ndarray
    costs: np.ndarray
    rewards: np.ndarray
    ranks: np.ndarray
    best_arm: int
    result_records: list[dict | None]


@dataclass
class JointBanditDataset:
    """最终供实验入口使用的完整数据集对象。"""

    samples: list[InstanceSample]
    arm_names: list[str]
    problem_to_methods: dict[str, list[str]]
    tsp_dataset_path: Path
    cvrp_dataset_path: Path


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    """如果外部传了日志函数，就输出一条日志。"""
    if log_fn is not None:
        log_fn(message)


def _iter_with_progress(iterable, *, total: int | None = None, desc: str = "", enabled: bool = False):
    """可选地给构建数据集阶段加一个进度条。"""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def _read_overrides(path: Path) -> dict[str, str]:
    """读取 Hydra 的 overrides.yaml，提取 key=value。"""
    overrides: dict[str, str] = {}
    if not path.exists():
        return overrides
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        payload = line[2:]
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _read_config_text(path: Path) -> str:
    """读取 Hydra 的 config.yaml 原文，用于做简单的 fallback 检测。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def discover_runs(results_dir: Path, allowed_problems: Iterable[str] = ("tsp", "cvrp")) -> list[RunInfo]:
    """
    扫描 results/test 下的所有 run。

    目录约定：
    results/test/{method}_{problem}/{timestamp_run}/

    这里不会直接筛“最新 run”，而是把候选 run 全部读出来，
    让下一步统一做 latest 选择。
    """
    allowed_problems = set(allowed_problems)
    runs: list[RunInfo] = []
    for run_dir in sorted(results_dir.glob("*/*")):
        if not run_dir.is_dir():
            continue
        overrides = _read_overrides(run_dir / ".hydra" / "overrides.yaml")
        config_text = _read_config_text(run_dir / ".hydra" / "config.yaml")
        parent_name = run_dir.parent.name
        if "_" not in parent_name:
            continue
        fallback_method, fallback_problem = parent_name.rsplit("_", 1)
        problem = overrides.get("problem", fallback_problem).lower()
        method = overrides.get("model", fallback_method).lower()
        if problem not in allowed_problems:
            continue
        runs.append(
            RunInfo(
                problem=problem,
                method=method,
                run_dir=run_dir,
                timestamp=run_dir.name[:19],
                test_data_path=overrides.get("test_data_path", ""),
                is_init_only=(
                    "NoIteration" in overrides.get("settings.iteration._target_", "")
                    or "EasyNCO.neural_solvers.pipeline.NoIteration" in config_text
                ),
            )
        )
    return runs


def select_latest_runs(runs: list[RunInfo]) -> dict[tuple[str, str], RunInfo]:
    """
    每个 (problem, method) 只保留最新 run。

    这一步很重要，因为同一个方法可能被重复评测过多次。
    当前 bandit 数据集必须有一个唯一版本，否则会把不同时间的结果混到一起。
    """
    selected: dict[tuple[str, str], RunInfo] = {}
    for run in sorted(runs, key=lambda item: (item.problem, item.method, item.timestamp, item.run_dir.name)):
        selected[(run.problem, run.method)] = run
    return selected


def _load_result_rows(path: Path) -> dict[int, dict]:
    """
    读取单个方法的 instance_results.jsonl。

    返回：
    - key: global_index
    - value: 该实例这一方法的结果记录

    之所以转成 dict，是因为后面要和 .pt 数据集按 global_index 精确对齐。
    """
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["global_index"])] = row
    return rows


def _pick_score(row: dict) -> float:
    """
    统一选择评价分数口径。

    优先级：
    1. aug_score
    2. score
    3. no_aug_score

    这样可以兼容不同方法导出的略微不同字段。
    """
    for key in ("aug_score", "score", "no_aug_score"):
        if row.get(key) is not None:
            return float(row[key])
    return float("nan")


def _compact_result_record(row: dict, method: str) -> dict:
    """
    从原始 instance_results.jsonl 行里提炼出我们后续复盘真正关心的字段。

    这里保留两类信息：
    1. 原始结果里最重要的字段
    2. 当前 bandit 实验实际采用的统一 score 口径
    """
    return {
        "method": method,
        "global_index": int(row.get("global_index", -1)),
        "name": row.get("name"),
        "aug_score": None if row.get("aug_score") is None else float(row["aug_score"]),
        "score": None if row.get("score") is None else float(row["score"]),
        "no_aug_score": None if row.get("no_aug_score") is None else float(row["no_aug_score"]),
        "picked_score": _pick_score(row),
    }


def _resolve_dataset_path(
    explicit_path: Path | None,
    latest_runs: dict[tuple[str, str], RunInfo],
    problem: str,
    results_dir: Path,
) -> Path:
    """
    确定某个问题对应的原始 .pt 数据集路径。

    逻辑：
    - 如果显式传了路径，就直接用
    - 否则从最新 run 的 test_data_path 里恢复
    """
    if explicit_path is not None:
        return Path(explicit_path)

    run = next((run for (p, _), run in latest_runs.items() if p == problem and run.test_data_path), None)
    if run is None:
        raise ValueError(f"没有找到 {problem} 对应的数据集路径")

    test_data_path = Path(run.test_data_path)
    if test_data_path.is_absolute():
        return test_data_path

    repo_root = results_dir.resolve().parents[2]
    return repo_root / "EasyNCO" / "data" / "datasets" / test_data_path


def _load_problem_instances(problem: str, dataset_path: Path) -> list[torch.Tensor]:
    """
    从 EasyNCO 的 .pt 数据集中恢复实例。

    输出统一成“list[Tensor]”，每个元素对应一个 global_index 的实例。

    - TSP: [n, 2]
    - CVRP: [1+n, 3]，列分别是 x / y / demand
    """
    loaded = torch.load(dataset_path, map_location="cpu")
    if problem == "tsp":
        coords = loaded["node_xy"]
        return [coords[i].to(torch.float32) for i in range(coords.shape[0])]
    if problem == "cvrp":
        depot_xy = loaded["depot_xy"]
        node_xy = loaded["node_xy"]
        node_demand = loaded["node_demand"]
        all_nodes = torch.cat((depot_xy, node_xy), dim=1)
        all_demands = torch.cat((torch.zeros(node_demand.shape[0], 1), node_demand), dim=1)
        merged = torch.cat((all_nodes, all_demands[:, :, None]), dim=2)
        return [merged[i].to(torch.float32) for i in range(merged.shape[0])]
    raise ValueError(f"未知问题类型: {problem}")


def build_joint_dataset(
    results_dir: Path,
    tsp_dataset_path: Path | None = None,
    cvrp_dataset_path: Path | None = None,
    arm_names: list[str] | None = None,
    problem_to_methods: dict[str, list[str]] | None = None,
    reward_mode: str = "linear_zero_one",
    strict_init_only: bool = True,
    max_samples_per_problem: int = 0,
    seed: int = 0,
    log_fn: Callable[[str], None] | None = None,
    show_progress: bool = False,
) -> JointBanditDataset:
    """
    构建最终训练/评测用的共享 bandit 数据集。

    整个流程是：
    1. 扫描 results/test 下所有 run
    2. 对每个 (problem, method) 选最新 run
    3. 从 run 中读出 test_data_path，找到原始 .pt
    4. 读取 .pt 数据集得到实例本身
    5. 读取每个方法的 instance_results.jsonl
    6. 以 global_index 为键，把多个方法的结果聚合到同一个实例上
    7. 提取特征，生成 costs / rewards / ranks / mask

    最终每个 sample 表示一个实例，不再是“一个实例-方法对”。
    """
    arm_names = arm_names or list(ALL_METHODS)
    problem_to_methods = problem_to_methods or {k: list(v) for k, v in PROBLEM_TO_METHODS.items()}

    _log(log_fn, f"扫描结果目录: {results_dir}")
    latest_runs = select_latest_runs(discover_runs(results_dir))
    _log(log_fn, f"已找到最新 runs 数量: {len(latest_runs)}")

    for problem, methods in problem_to_methods.items():
        for method in methods:
            if (problem, method) not in latest_runs:
                raise ValueError(f"缺少结果 run: problem={problem}, method={method}")
            if strict_init_only and not latest_runs[(problem, method)].is_init_only:
                raise ValueError(f"run 不是 initialization-only: problem={problem}, method={method}")

    _log(log_fn, "开始解析原始 .pt 数据集路径")
    tsp_dataset_path = _resolve_dataset_path(tsp_dataset_path, latest_runs, "tsp", results_dir)
    cvrp_dataset_path = _resolve_dataset_path(cvrp_dataset_path, latest_runs, "cvrp", results_dir)
    _log(log_fn, f"TSP 数据集: {tsp_dataset_path}")
    _log(log_fn, f"CVRP 数据集: {cvrp_dataset_path}")
    _log(log_fn, "开始加载原始实例张量")
    problem_instances = {
        "tsp": _load_problem_instances("tsp", tsp_dataset_path),
        "cvrp": _load_problem_instances("cvrp", cvrp_dataset_path),
    }
    _log(
        log_fn,
        f"实例加载完成: tsp={len(problem_instances['tsp'])}, cvrp={len(problem_instances['cvrp'])}",
    )

    samples: list[InstanceSample] = []
    arm_to_idx = {name: idx for idx, name in enumerate(arm_names)}
    rng = np.random.default_rng(seed)

    for problem in ("tsp", "cvrp"):
        methods = problem_to_methods[problem]
        _log(log_fn, f"开始聚合问题 {problem}，候选方法数={len(methods)}")
        # 每个方法先读成 {global_index -> row} 的形式
        per_method_rows = {
            method: _load_result_rows(latest_runs[(problem, method)].run_dir / "instance_results.jsonl")
            for method in methods
        }
        common_indices = None
        for rows in per_method_rows.values():
            indices = set(rows.keys())
            common_indices = indices if common_indices is None else (common_indices & indices)
        common_indices = sorted(common_indices or [])
        # 只保留“所有方法都同时有结果”的实例。
        # 这样后面 reward 的实例内排序才是公平的。
        if max_samples_per_problem and len(common_indices) > max_samples_per_problem:
            chosen = sorted(rng.choice(common_indices, size=max_samples_per_problem, replace=False).tolist())
            common_indices = chosen
        _log(log_fn, f"{problem} 可用实例数={len(common_indices)}")

        progress = _iter_with_progress(
            common_indices,
            total=len(common_indices),
            desc=f"build {problem}",
            enabled=show_progress,
        )
        for global_index in progress:
            nodes = problem_instances[problem][global_index]
            feature_bundle = extract_manual_feature_bundle(problem, nodes)
            problem_onehot = get_problem_onehot(problem)
            feasible_mask = get_mask(problem, arm_names=arm_names, problem_to_methods=problem_to_methods)

            # costs 是统一 arm 空间上的 cost 向量。
            # 当前问题里不可用的方法保留 NaN。
            costs = np.full(len(arm_names), np.nan, dtype=np.float64)
            result_records: list[dict | None] = [None for _ in range(len(arm_names))]
            for method in methods:
                arm_idx = arm_to_idx[method]
                row = per_method_rows[method][global_index]
                costs[arm_idx] = _pick_score(row)
                result_records[arm_idx] = _compact_result_record(row, method=method)

            # reward 和 rank 都是在“实例内方法集合”上算出来的
            rewards, ranks = compute_rank_based_rewards(costs, mask=feasible_mask, mode=reward_mode)
            valid_costs = np.where(np.isfinite(costs), costs, np.inf)
            best_arm = int(np.argmin(valid_costs))

            samples.append(
                InstanceSample(
                    uid=f"{problem}:{global_index}",
                    problem=problem,
                    global_index=int(global_index),
                    nodes=nodes,
                    problem_onehot=problem_onehot,
                    manual_raw=feature_bundle.manual_raw,
                    manual_with_scale=feature_bundle.manual_with_scale,
                    linucb_context=np.concatenate((problem_onehot, feature_bundle.manual_with_scale), axis=0).astype(np.float32),
                    feasible_mask=feasible_mask,
                    costs=costs,
                    rewards=rewards,
                    ranks=ranks,
                    best_arm=best_arm,
                    result_records=result_records,
                )
            )
        _log(log_fn, f"问题 {problem} 聚合完成，当前累计样本数={len(samples)}")

    samples.sort(key=lambda sample: (sample.problem, sample.global_index))
    _log(log_fn, f"全部样本构建完成，最终样本数={len(samples)}")
    return JointBanditDataset(
        samples=samples,
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        tsp_dataset_path=Path(tsp_dataset_path),
        cvrp_dataset_path=Path(cvrp_dataset_path),
    )
