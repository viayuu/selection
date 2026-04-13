from __future__ import annotations

import json
import pickle
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
    from arm_config import ALL_METHODS, PROBLEM_TO_METHODS, get_arm_space_config, get_mask, get_problem_onehot  # type: ignore
    from features import extract_manual_feature_bundle  # type: ignore
    from reward import compute_rank_based_rewards  # type: ignore
else:
    from .arm_config import ALL_METHODS, PROBLEM_TO_METHODS, get_arm_space_config, get_mask, get_problem_onehot
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
    source_split: str | None = None


@dataclass
class JointBanditDataset:
    """最终供实验入口使用的完整数据集对象。"""

    samples: list[InstanceSample]
    arm_names: list[str]
    problem_to_methods: dict[str, list[str]]
    tsp_dataset_path: Path | None
    cvrp_dataset_path: Path | None
    predefined_splits: dict[str, list[int]] | None = None
    data_source: str = "easynco_results"


# NSS 数据集里各问题对应的固定方法顺序。
# 这个顺序必须与 raw_label.pkl 中的 cost / time / ind 一致。
NSS_METHOD_FILE_NAMES = {
    "bq": "bq",
    "elg": "ELG",
    "lehd": "LEHD",
    "t2t": "T2T",
    "t2t500": "T2T500",
    "difusco": "DIFUSCO",
    "difusco500": "DIFUSCO500",
    "omni": "Omni",
    "mvmoe": "MVMoE",
}


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


def _load_pickle(path: Path):
    """读取 NSS 的 `.pkl` 文件。"""
    with path.open("rb") as f:
        return pickle.load(f)


def _load_nss_raw_labels(path: Path) -> dict[str, dict]:
    """
    读取 NSS 的 `raw_label.pkl`。

    其字段结构来自原仓库 `process_raw_label.py`：
    - cost: 当前实例上各方法的 cost，顺序固定
    - time: 当前实例上各方法的耗时，顺序固定
    - ind: 该实例上 cost 最小的方法下标
    - gap: 相对 opt 的 gap（有些 split 可能全为 0）
    """
    loaded = _load_pickle(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"raw_label.pkl 格式异常: {path}")
    return loaded


def _normalize_nss_instance(problem: str, instance) -> torch.Tensor:
    """
    把 NSS 原始实例统一成当前 2cmab2 使用的节点张量格式。

    输出约定：
    - TSP: [n, 2]
    - CVRP: [1+n, 3]，列分别是 x / y / demand
    """
    problem = problem.lower()
    if problem == "tsp":
        if not torch.is_tensor(instance):
            raise ValueError(f"TSP NSS 实例格式异常: {type(instance)}")
        tensor = instance.to(torch.float32)
        if tensor.dim() == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2 or tensor.size(-1) != 2:
            raise ValueError(f"TSP NSS 实例维度异常: {tuple(tensor.shape)}")
        return tensor

    if problem == "cvrp":
        if not isinstance(instance, dict):
            raise ValueError(f"CVRP NSS 实例格式异常: {type(instance)}")
        depot = instance["depot"].to(torch.float32)
        loc = instance["loc"].to(torch.float32)
        demand = instance["demand"].to(torch.float32)
        if depot.dim() == 3 and depot.size(0) == 1:
            depot = depot.squeeze(0)
        if loc.dim() == 3 and loc.size(0) == 1:
            loc = loc.squeeze(0)
        if demand.dim() == 2 and demand.size(0) == 1:
            demand = demand.squeeze(0)
        xy = torch.cat((depot, loc), dim=0)
        all_demand = torch.cat((torch.zeros(1, dtype=demand.dtype), demand), dim=0)
        merged = torch.cat((xy, all_demand[:, None]), dim=1)
        if merged.dim() != 2 or merged.size(-1) != 3:
            raise ValueError(f"CVRP NSS 实例维度异常: {tuple(merged.shape)}")
        return merged

    raise ValueError(f"未知问题类型: {problem}")


def _make_nss_result_record(
    *,
    method: str,
    global_index: int,
    split_name: str,
    cost: float,
    time_cost: float,
    gap: float,
) -> dict:
    """
    为 NSS 数据构造一个与当前 trace 兼容的结果记录。

    NSS 没有 EasyNCO 那种 `aug_score / no_aug_score` 概念，
    所以这里统一落在 `score` 字段上。
    """
    return {
        "method": method,
        "global_index": int(global_index),
        "name": None,
        "split": split_name,
        "score": float(cost),
        "picked_score": float(cost),
        "time": float(time_cost),
        "gap": float(gap),
    }


def _sample_indices_for_limit(
    count: int,
    *,
    max_samples: int,
    rng: np.random.Generator,
) -> list[int]:
    """对一个 split 内的样本做可选裁剪。"""
    indices = list(range(count))
    if max_samples > 0 and count > max_samples:
        indices = sorted(rng.choice(count, size=max_samples, replace=False).tolist())
    return indices


def _build_nss_joint_dataset(
    *,
    nss_dataset_root: Path,
    arm_names: list[str],
    problem_to_methods: dict[str, list[str]],
    reward_mode: str,
    max_samples_per_problem: int,
    seed: int,
    log_fn: Callable[[str], None] | None,
    show_progress: bool,
) -> JointBanditDataset:
    """
    从 NSS 原始 `datasets/{TSP,CVRP}{train,val,test}` 直接构造 bandit 数据集。

    与 EasyNCO 结果目录分支相比，这里有两个关键不同点：
    1. train/val/test split 已经由 NSS 提前固定好，不能再随机重切
    2. 方法空间不是原来的 12 臂，而是 NSS 自带的 TSP 7 臂 / CVRP 5 臂
    """
    nss_dataset_root = Path(nss_dataset_root)
    rng = np.random.default_rng(seed)
    arm_to_idx = {name: idx for idx, name in enumerate(arm_names)}

    samples: list[InstanceSample] = []
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    split_suffix = {"train": "train", "val": "val", "test": "test"}
    problem_prefix = {"tsp": "TSP", "cvrp": "CVRP"}

    _log(log_fn, f"使用 NSS 数据源: {nss_dataset_root}")
    for split_name in ("train", "val", "test"):
        _log(log_fn, f"开始构建 NSS {split_name} split")
        for problem in ("tsp", "cvrp"):
            split_dir = nss_dataset_root / f"{problem_prefix[problem]}{split_suffix[split_name]}"
            dataset_path = split_dir / "dataset.pkl"
            raw_label_path = split_dir / "raw_label.pkl"
            if not dataset_path.exists():
                raise ValueError(f"NSS 数据集不存在: {dataset_path}")
            if not raw_label_path.exists():
                raise ValueError(f"NSS 标签不存在: {raw_label_path}")

            raw_instances = _load_pickle(dataset_path)
            raw_labels = _load_nss_raw_labels(raw_label_path)
            methods = list(problem_to_methods[problem])
            for method in methods:
                result_file_name = NSS_METHOD_FILE_NAMES.get(method)
                if result_file_name is None:
                    raise ValueError(f"NSS 方法缺少结果文件名映射: {method}")
                result_file = split_dir / "results" / f"result_{result_file_name}.txt"
                if not result_file.exists():
                    raise ValueError(f"NSS 结果文件不存在: {result_file}")
            selected_indices = _sample_indices_for_limit(
                len(raw_instances),
                max_samples=max_samples_per_problem,
                rng=rng,
            )
            _log(
                log_fn,
                f"NSS split={split_name} problem={problem} 原始样本数={len(raw_instances)}，实际使用={len(selected_indices)}",
            )
            progress = _iter_with_progress(
                selected_indices,
                total=len(selected_indices),
                desc=f"nss {split_name} {problem}",
                enabled=show_progress,
            )
            for local_index in progress:
                label_key = str(local_index)
                if label_key not in raw_labels:
                    raise ValueError(f"NSS raw_label 缺少索引 {label_key}: {raw_label_path}")
                label = raw_labels[label_key]
                label_costs = list(label.get("cost", []))
                label_times = list(label.get("time", []))
                label_gaps = list(label.get("gap", []))
                if len(label_costs) != len(methods):
                    raise ValueError(
                        f"NSS 标签维度与方法数不一致: split={split_name}, problem={problem}, "
                        f"costs={len(label_costs)}, methods={len(methods)}"
                    )
                if len(label_times) != len(methods):
                    raise ValueError(
                        f"NSS 耗时维度与方法数不一致: split={split_name}, problem={problem}, "
                        f"time={len(label_times)}, methods={len(methods)}"
                    )
                if len(label_gaps) != len(methods):
                    # 有些旧文件 gap 可能缺失；这里退化为 0，不阻断训练。
                    label_gaps = [0.0 for _ in methods]

                nodes = _normalize_nss_instance(problem, raw_instances[local_index])
                feature_bundle = extract_manual_feature_bundle(problem, nodes)
                problem_onehot = get_problem_onehot(problem)
                feasible_mask = get_mask(problem, arm_names=arm_names, problem_to_methods=problem_to_methods)

                costs = np.full(len(arm_names), np.nan, dtype=np.float64)
                result_records: list[dict | None] = [None for _ in range(len(arm_names))]
                for method_index, method in enumerate(methods):
                    arm_idx = arm_to_idx[method]
                    cost = float(label_costs[method_index])
                    time_cost = float(label_times[method_index])
                    gap = float(label_gaps[method_index])
                    costs[arm_idx] = cost
                    result_records[arm_idx] = _make_nss_result_record(
                        method=method,
                        global_index=int(local_index),
                        split_name=split_name,
                        cost=cost,
                        time_cost=time_cost,
                        gap=gap,
                    )

                rewards, ranks = compute_rank_based_rewards(costs, mask=feasible_mask, mode=reward_mode)
                best_method_index = int(label.get("ind", int(np.nanargmin(np.where(np.isfinite(costs), costs, np.nan)))))
                if not 0 <= best_method_index < len(methods):
                    raise ValueError(
                        f"NSS ind 越界: split={split_name}, problem={problem}, ind={best_method_index}, methods={len(methods)}"
                    )
                best_arm = arm_to_idx[methods[best_method_index]]

                sample = InstanceSample(
                    uid=f"{problem}:{split_name}:{local_index}",
                    problem=problem,
                    global_index=int(local_index),
                    nodes=nodes,
                    problem_onehot=problem_onehot,
                    manual_raw=feature_bundle.manual_raw,
                    manual_with_scale=feature_bundle.manual_with_scale,
                    linucb_context=np.concatenate((problem_onehot, feature_bundle.manual_with_scale), axis=0).astype(np.float32),
                    feasible_mask=feasible_mask,
                    costs=costs,
                    rewards=rewards,
                    ranks=ranks,
                    best_arm=int(best_arm),
                    result_records=result_records,
                    source_split=split_name,
                )
                split_indices[split_name].append(len(samples))
                samples.append(sample)

    _log(
        log_fn,
        "NSS 数据集构建完成: "
        f"train={len(split_indices['train'])}, val={len(split_indices['val'])}, test={len(split_indices['test'])}",
    )
    return JointBanditDataset(
        samples=samples,
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        tsp_dataset_path=nss_dataset_root / "TSPtrain" / "dataset.pkl",
        cvrp_dataset_path=nss_dataset_root / "CVRPtrain" / "dataset.pkl",
        predefined_splits=split_indices,
        data_source="nss",
    )


def build_nss_benchmark_dataset(
    *,
    nss_dataset_root: Path,
    arm_names: list[str],
    problem_to_methods: dict[str, list[str]],
    reward_mode: str,
    log_fn: Callable[[str], None] | None = None,
    show_progress: bool = False,
) -> JointBanditDataset:
    """
    读取 NSS 提供的 TSPLIB / CVRPLIB 基准数据集。

    与 `train/val/test` 合成数据不同，这里没有训练划分，
    只有两个“最终测试 benchmark”：
    - `TSPLIB`
    - `CVRPLIB`

    因此这里统一把它们放进一个 `test` split 中，便于后续直接做：
    - selector benchmark 测试
    - benchmark 上的 oracle / single-best / 单方法对比
    """
    nss_dataset_root = Path(nss_dataset_root)
    arm_to_idx = {name: idx for idx, name in enumerate(arm_names)}

    benchmark_spec = [
        ("TSPLIB", "tsp"),
        ("CVRPLIB", "cvrp"),
    ]

    samples: list[InstanceSample] = []
    split_indices: dict[str, list[int]] = {"test": []}

    _log(log_fn, f"使用 NSS benchmark 数据源: {nss_dataset_root}")
    for benchmark_name, problem in benchmark_spec:
        split_dir = nss_dataset_root / benchmark_name
        dataset_path = split_dir / "dataset.pkl"
        raw_label_path = split_dir / "raw_label.pkl"
        if not dataset_path.exists():
            raise ValueError(f"NSS benchmark 数据集不存在: {dataset_path}")
        if not raw_label_path.exists():
            raise ValueError(f"NSS benchmark 标签不存在: {raw_label_path}")

        raw_instances = _load_pickle(dataset_path)
        raw_labels = _load_nss_raw_labels(raw_label_path)
        methods = list(problem_to_methods[problem])

        _log(
            log_fn,
            f"NSS benchmark={benchmark_name} problem={problem} 样本数={len(raw_instances)}",
        )

        progress = _iter_with_progress(
            range(len(raw_instances)),
            total=len(raw_instances),
            desc=f"benchmark {benchmark_name}",
            enabled=show_progress,
        )
        for local_index in progress:
            label_key = str(local_index)
            if label_key not in raw_labels:
                raise ValueError(f"NSS benchmark raw_label 缺少索引 {label_key}: {raw_label_path}")
            label = raw_labels[label_key]
            label_costs = list(label.get("cost", []))
            label_times = list(label.get("time", []))
            label_gaps = list(label.get("gap", []))
            if len(label_costs) != len(methods):
                raise ValueError(
                    f"NSS benchmark 标签维度与方法数不一致: benchmark={benchmark_name}, problem={problem}, "
                    f"costs={len(label_costs)}, methods={len(methods)}"
                )
            if len(label_times) != len(methods):
                raise ValueError(
                    f"NSS benchmark 耗时维度与方法数不一致: benchmark={benchmark_name}, problem={problem}, "
                    f"time={len(label_times)}, methods={len(methods)}"
                )
            if len(label_gaps) != len(methods):
                label_gaps = [0.0 for _ in methods]

            nodes = _normalize_nss_instance(problem, raw_instances[local_index])
            feature_bundle = extract_manual_feature_bundle(problem, nodes)
            problem_onehot = get_problem_onehot(problem)
            feasible_mask = get_mask(problem, arm_names=arm_names, problem_to_methods=problem_to_methods)

            costs = np.full(len(arm_names), np.nan, dtype=np.float64)
            result_records: list[dict | None] = [None for _ in range(len(arm_names))]
            for method_index, method in enumerate(methods):
                arm_idx = arm_to_idx[method]
                cost = float(label_costs[method_index])
                time_cost = float(label_times[method_index])
                gap = float(label_gaps[method_index])
                costs[arm_idx] = cost
                result_records[arm_idx] = _make_nss_result_record(
                    method=method,
                    global_index=int(local_index),
                    split_name=benchmark_name.lower(),
                    cost=cost,
                    time_cost=time_cost,
                    gap=gap,
                )

            rewards, ranks = compute_rank_based_rewards(costs, mask=feasible_mask, mode=reward_mode)
            best_method_index = int(label.get("ind", int(np.nanargmin(np.where(np.isfinite(costs), costs, np.nan)))))
            if not 0 <= best_method_index < len(methods):
                raise ValueError(
                    f"NSS benchmark ind 越界: benchmark={benchmark_name}, problem={problem}, "
                    f"ind={best_method_index}, methods={len(methods)}"
                )
            best_arm = arm_to_idx[methods[best_method_index]]

            sample = InstanceSample(
                uid=f"{problem}:{benchmark_name.lower()}:{local_index}",
                problem=problem,
                global_index=int(local_index),
                nodes=nodes,
                problem_onehot=problem_onehot,
                manual_raw=feature_bundle.manual_raw,
                manual_with_scale=feature_bundle.manual_with_scale,
                linucb_context=np.concatenate((problem_onehot, feature_bundle.manual_with_scale), axis=0).astype(np.float32),
                feasible_mask=feasible_mask,
                costs=costs,
                rewards=rewards,
                ranks=ranks,
                best_arm=int(best_arm),
                result_records=result_records,
                source_split=benchmark_name.lower(),
            )
            split_indices["test"].append(len(samples))
            samples.append(sample)

    _log(
        log_fn,
        "NSS benchmark 数据集构建完成: "
        f"TSPLIB+CVRPLIB 总样本数={len(split_indices['test'])}",
    )
    return JointBanditDataset(
        samples=samples,
        arm_names=arm_names,
        problem_to_methods=problem_to_methods,
        tsp_dataset_path=nss_dataset_root / "TSPLIB" / "dataset.pkl",
        cvrp_dataset_path=nss_dataset_root / "CVRPLIB" / "dataset.pkl",
        predefined_splits=split_indices,
        data_source="nss_benchmark",
    )


def build_joint_dataset(
    results_dir: Path,
    tsp_dataset_path: Path | None = None,
    cvrp_dataset_path: Path | None = None,
    data_source: str = "easynco_results",
    nss_dataset_root: Path | None = None,
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

    当前支持两种数据源：

    1. `easynco_results`
       - 扫描 EasyNCO 的 `results/test`
       - 再回溯对应 `.pt` 数据集
       - 最后按实例聚合成共享 bandit 样本

    2. `nss`
       - 直接读取 `9nss论文/neural-solver-selection/datasets`
       - 使用其自带的 `train / val / test` split
       - 不再额外随机切分

    最终每个 sample 表示一个实例，不再是“一个实例-方法对”。
    """
    normalized_source = str(data_source).strip().lower()
    default_arm_names, default_problem_to_methods = get_arm_space_config(normalized_source)
    arm_names = arm_names or default_arm_names
    problem_to_methods = problem_to_methods or default_problem_to_methods
    if normalized_source == "nss":
        dataset_root = Path(nss_dataset_root) if nss_dataset_root is not None else Path(
            __file__
        ).resolve().parents[1] / "9nss论文" / "neural-solver-selection" / "datasets"
        return _build_nss_joint_dataset(
            nss_dataset_root=dataset_root,
            arm_names=arm_names,
            problem_to_methods=problem_to_methods,
            reward_mode=reward_mode,
            max_samples_per_problem=max_samples_per_problem,
            seed=seed,
            log_fn=log_fn,
            show_progress=show_progress,
        )
    if normalized_source != "easynco_results":
        raise ValueError(f"未知数据源: {data_source}")

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
        predefined_splits=None,
        data_source="easynco_results",
    )
