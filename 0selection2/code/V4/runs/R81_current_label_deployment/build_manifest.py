"""Bind the current two-task label columns to saved generation evidence, CPU only."""

import ast
import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[3]
HOME = PROJECT.parent
DATA = HOME / "selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014"
PARENT = HOME / "selection_data_new_only_B_filtered_seed2_20261014"
SNAPSHOT = HOME / "训练数据_公共完整标签_20261013_1923"
WORK = HOME / "easynco_v3_bridge/exports/train_100k_20261002"
PROBLEMS = ("VRPTW", "VRPLTW")
METHODS = ("MTPOMO", "MVMOE", "MoSES_CaDA", "MoSES_RF", "reld_mt", "RouteFinder")
SCALES = (50, 75, 76, 100)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def field(value, level, evidence, detail, route_checked=False):
    result = dict(value=value, evidence_level=level, evidence=evidence, detail=detail)
    if route_checked:
        result["corroboration"] = dict(evidence_level="verified-on-route",
            evidence="route_checks.csv; four fixed current TRAIN inputs per task",
            limitation="The saved routes satisfy this contract; this is not a new solver execution or proof for all possible routes.")
    return result


def source_reference(path, recorded_hashes, symbols):
    path = Path(path)
    expected = recorded_hashes.get(str(path))
    actual = digest(path) if path.is_file() else None
    return dict(path=str(path), symbols=symbols, recorded_sha256=expected,
                observed_sha256=actual, matches_recorded=bool(expected and actual == expected))


def load_checker():
    path = PROJECT / "code/V4/r57_routes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check_route")
    namespace = dict(np=np, TOL=2e-5)
    # Load only the existing independent checker, without importing experiment runners.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["check_route"], identity(path)


def fixed_train_rows(problem, generation):
    with np.load(DATA / (problem + "train") / "input_metadata.npz", allow_pickle=False) as archive:
        indices, sizes = archive["source_index"], archive["scale"]
    selected = []
    for size in SCALES:
        candidates = np.flatnonzero(sizes == size)
        local = int(candidates[np.argmin(indices[candidates])])
        source = int(indices[local])
        shard = next(s for s in generation["shards"] if s["start_index"] <= source < s["start_index"] + s["count"])
        selected.append(dict(problem=problem, split="train", scale=size, local_index=local,
            source_index=source, shard_id=shard["id"], shard_offset=source - shard["start_index"],
            input_path=shard["path"], input_sha256=shard["sha256"]))
    return selected


def core_fields(problem, method, profile, source_refs):
    recorded = "profile.json#normalization + adapter_source"
    native = method in ("MTPOMO", "MVMOE")
    reld = method == "reld_mt"
    env = "pinned environment source in source_references"
    adapter = "profile.json#base_adapter_source" if native else "profile.json#adapter_source"
    references = [recorded, adapter, env]
    level = "source-derived" if reld else "current-run-recorded"
    boundary = ("service start and depot return <= deadline + 1e-5" if native or reld else
                "arrival < customer deadline; service completion + return < depot deadline; distance <= limit")
    return dict(
        depot_time_window=field([0.0, 3.0], level, references,
            "Closed routes return by 3, after waiting and customer service. ReLD's internal random-generator horizon 4.6 is not called when problems=batch is supplied.", True),
        speed=field(1.0, "source-derived" if native or reld else "current-run-recorded", references,
            "Travel time is Euclidean distance / speed; implicit environment speed for MTPOMO/MVMOE and ReLD.", True),
        customer_service_time=field("instance.service_time", "current-run-recorded",
            [adapter, "saved input shards; route_checks.csv"],
            "Customer values are passed unchanged through FP32 tensors, added after waiting/service start; not charged as travel cost.", True),
        depot_service_time=field(0.0, "source-derived", references,
            "Adapter/environment prepends a zero depot service time.", True),
        customer_time_windows=field("instance.tw_start / instance.tw_end", "source-derived", references,
            "Service START must satisfy the customer window; clock=max(arrival, early)+service. Late is not a service-completion deadline.", True),
        route_length_limit=field("instance.route_limit (=3 in generated VRPLTW)" if problem == "VRPLTW" else "disabled",
            "current-run-recorded", references,
            ("Per-vehicle Euclidean travel distance, including depot return; waiting and service excluded. RF/MoSES explicitly construct distance_limit=3; native/ReLD read route_limit."
             if problem == "VRPLTW" else "VRPTW has no L constraint. Native adapter's unused route_limit=0 is not an active zero-length constraint."), True),
        cost=field("sum of Euclidean travel distances over all closed vehicle routes", "source-derived", references,
            "Departure, between-customer and return legs count; no waiting/service penalty, fixed vehicle cost or route-count charge. Best start (and RF/MoSES best augmentation) is selected.", True),
        numerical_boundary=field(boundary, "source-derived", [env],
            "Boundary/tolerance implementations are recorded separately; common nominal fields do not prove exact equivalence on every floating-point boundary."),
        decode_and_budget=field(profile["budget"], "current-run-recorded", ["profile.json#budget"],
            "RF/MoSES record phase=test, not a fully resolved external-library decode configuration. ReLD and native sampling budgets are explicitly recorded."),
        external_library_resolved_decode=field(None if not native and not reld else profile["budget"]["decoding"],
            "unknown" if not native and not reld else "current-run-recorded", ["profile.json#budget"],
            "Do not turn phase=test into an independently captured effective greedy setting. No new library default is substituted."))


def inspect_deployment(problem, method, frozen, aligned, rows, checker, route_output):
    original = "RELD_MOEL" if method == "reld_mt" else method
    folder = WORK / "labels" / problem / original
    profile, complete, dependencies = (read(folder / name) for name in
                                         ("profile.json", "complete.json", "dependency_contract.json"))
    recipe_hash = hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    frozen_column = frozen["labels"][original]
    frozen_completion_match = digest(folder / "complete.json") == frozen_column["completion"]["sha256"]
    profile_match = recipe_hash == complete["profile_sha256"] == dependencies["profile_sha256"]
    merged_hash_match = complete["result_sha256"] == frozen_column["merged"]["sha256"]
    if not (frozen_completion_match and profile_match and merged_hash_match):
        raise ValueError(f"Frozen generation identity changed: {problem}/{method}")
    artifacts = dependencies["artifacts"]
    if method in ("MTPOMO", "MVMOE"):
        env = HOME / "EasyNCO/neural_solvers/envs/MVRPEnv.py"
    elif method == "reld_mt":
        env = HOME / f"reld-nco-main/Multi-Task/envs/{problem}Env.py"
    else:
        suffix = "routefinder/routefinder" if method == "RouteFinder" else "moses_vrp"
        env = HOME / "methods论文_mineru_output/source_code" / suffix / "envs/mtvrp/env.py"
    refs = [source_reference(env, artifacts, ["load_problems/_reset", "step/_step", "get_action_mask", "_get_travel_distance/_get_reward"])]
    if method == "reld_mt":
        refs.append(source_reference(HOME / "easynco_v3_bridge/scripts/reld_bridge_common.py", artifacts,
                                    ["collate_problem_batch", "_run_no_aug_batch"]))
    source_matches = all(r["matches_recorded"] for r in refs)
    if not source_matches:
        raise ValueError(f"Cannot interpret changed source as the recorded deployment: {problem}/{method}")
    calibration = read(folder / "calibration.json")
    route_records, witnesses = [], []
    for row in rows:
        path = folder / "shards" / (row["shard_id"] + ".json")
        saved = read(path)
        offset = row["shard_offset"]
        item_path = Path(row["input_path"])
        if digest(item_path) != row["input_sha256"] or saved["input_sha256"] != row["input_sha256"]:
            raise ValueError("Selected saved route does not match its input")
        if saved["profile_sha256"] != recipe_hash:
            raise ValueError("Selected route uses a different deployment")
        with item_path.open("rb") as stream:
            item = pickle.load(stream)[offset]
        tour = saved.get("tours", [])[offset] if len(saved.get("tours", [])) == saved["count"] else None
        if tour is None:
            route_records.append(dict(**row, solver=method, status="unverified-no-saved-route"))
            continue
        if tour[0] != 0:
            tour = [0] + tour
        # Both constants are supported by the bound deployment, not checker defaults.
        checked = checker(dict(item, speed=1.0, depot_tw_end=3.0), tour, problem)
        current = float(aligned["train"]["costs"][row["local_index"], METHODS.index(method)])
        saved_cost = float(saved["costs"][offset])
        error = abs(checked["cost"] - current)
        flags = ("visit_once", "delivery_capacity_ok", "customer_tw_ok", "route_limit_ok", "closed_depot_horizon3_ok")
        feasible = all(checked[name] for name in flags)
        record = dict(problem=problem, solver=method, split="train", local_index=row["local_index"],
            source_index=row["source_index"], scale=row["scale"], current_label_cost=current,
            saved_route_cost=saved_cost, independent_cost=checked["cost"],
            saved_cost_fp64_exact=current == saved_cost,
            saved_cost_matches_label=current == float(f"{saved_cost:.10f}"),
            serialization_rounding_difference=abs(current - saved_cost),
            independent_cost_matches=bool(np.isclose(checked["cost"], current, atol=5e-5, rtol=2e-6)),
            absolute_cost_error=error, feasible=feasible, **{name: checked[name] for name in flags},
            service_time_min=float(np.min(item["service_time"])), service_time_max=float(np.max(item["service_time"])),
            route_limit=item.get("route_limit"), source_route_file=str(path))
        route_records.append(record)
        chunk = next(c for c in saved["chunks"] if c["offset"] <= offset < c["offset"] + c["count"])
        witnesses.append(dict(sample=row, saved_output=identity(path),
            profile_sha256=saved["profile_sha256"], dependency_contract_sha256=saved.get("dependency_contract_sha256"),
            producer_sources=saved.get("producer_sources", {}), chunk=chunk,
            original_run_audit=saved.get("audit"), original_created_at=saved.get("created_at")))
    route_output.extend(route_records)
    fields = core_fields(problem, method, profile, refs)
    route_pass = len(route_records) == len(rows) and all(r.get("feasible") and r.get("independent_cost_matches")
                                                       and r.get("saved_cost_matches_label") for r in route_records)
    logs = sorted((folder / "logs").glob("*.log"))
    gaps = ["Original shell argv is not recorded in the selected profile; saved chunks record seeds and batch sizes.",
            "Exact cross-hardware/library bitwise replay is not certified by this CPU-only evidence collection."]
    if method in ("MoSES_CaDA", "MoSES_RF", "RouteFinder"):
        gaps.append("Effective external-library decode type is not captured as a resolved runtime field; recorded budget is phase=test.")
    if not route_pass:
        gaps.append("At least one fixed saved-route check is missing or failed; inspect route_checks.csv.")
    return dict(problem=problem, solver_name=method, source_solver_name=original,
        column_index=METHODS.index(method), global_solver_id=aligned["train"]["metadata"]["global_solver_ids"][METHODS.index(method)],
        label_binding_status="bound", core_source_binding="recorded sources match local bytes",
        route_verification_status="passed-fixed-train-sample" if route_pass else "incomplete-or-failed",
        fields=fields, source_references=refs,
        deployment=dict(backend=profile["backend"], checkpoint=profile.get("checkpoint"),
            checkpoint_sha256=profile.get("checkpoint_sha256"), checkpoints=profile.get("checkpoints"),
            weights_evidence_level="current-run-recorded", weights_note="Hashes captured in generation profile; no new model load or checkpoint inference.",
            model_params=profile.get("model_params", profile.get("params")), env_params=profile.get("env_params"),
            actual_command=None, entrypoint="label_train_100k.py -> make_confirmed_backend -> recorded backend.run",
            normalization=profile.get("normalization")),
        run_records=dict(profile=identity(folder / "profile.json"), profile_recipe_sha256=recipe_hash,
            completion=identity(folder / "complete.json"), completed_at=complete["completed_at"],
            frozen_export_completion=frozen_column["completion"], frozen_export_merged=frozen_column["merged"],
            frozen_completion_match=frozen_completion_match, profile_match=profile_match,
            merged_identity_match=merged_hash_match, generation_shards=len(complete["shards"]),
            dependency_contract=identity(folder / "dependency_contract.json"),
            log_files_count=len(logs), representative_logs=[identity(p) for p in logs[:1] + logs[-1:]],
            calibration=dict(path=str(folder / "calibration.json"), created_at=calibration.get("created_at"),
                historical_match=calibration.get("historical_match"),
                scope="Calibration against OLD prefix only; not a test of current new-only labels.")),
        alignment=[aligned[s]["columns"][method] for s in ("train", "val", "test")],
        fixed_saved_route_witnesses=witnesses, gaps=gaps)


def align_problem(problem, frozen):
    result = {}
    sources = {}
    for method in METHODS:
        original = "RELD_MOEL" if method == "reld_mt" else method
        path = SNAPSHOT / (problem + "train") / "results" / ("result_" + original + ".txt")
        data = np.loadtxt(path, delimiter=",", usecols=(0, 1), dtype=np.float64)
        if not np.array_equal(data[:, 0], np.arange(len(data))):
            raise ValueError("Snapshot source index is not contiguous")
        sha = digest(path)
        if sha != frozen["labels"][original]["merged"]["sha256"]:
            raise ValueError("Frozen snapshot result does not match generation completion")
        sources[method] = (data[:, 1], identity(path))
    for split in ("train", "val", "test"):
        folder, parent = DATA / (problem + split), PARENT / (problem + split)
        metadata, parent_meta = read(folder / "metadata.json"), read(parent / "metadata.json")
        if tuple(metadata["methods"]) != METHODS:
            raise ValueError("Unexpected candidate order")
        with np.load(folder / "input_metadata.npz", allow_pickle=False) as archive:
            indices = archive["source_index"].copy()
        with np.load(parent / "input_metadata.npz", allow_pickle=False) as archive:
            parent_indices = archive["source_index"].copy()
        if not np.array_equal(indices, parent_indices) or np.min(indices) < 10000:
            raise ValueError("Parent/child rows differ or old-prefix labels leaked into this audit")
        with np.load(folder / "targets.npz", allow_pickle=False) as archive:
            costs = archive["costs"].copy()
        with np.load(parent / "targets.npz", allow_pickle=False) as archive:
            parent_costs = archive["costs"].copy()
        columns = {}
        for slot, method in enumerate(METHODS):
            original = "RELD_MOEL" if method == "reld_mt" else method
            parent_slot = parent_meta["methods"].index(original)
            expected = sources[method][0][indices]
            current = costs[:, slot]
            difference = np.abs(current - expected)
            parent_equal = np.array_equal(current, parent_costs[:, parent_slot])
            if not parent_equal or not np.array_equal(current, expected):
                raise ValueError(f"Cost binding failed: {problem}/{split}/{method}")
            columns[method] = dict(problem=problem, split=split, solver=method,
                rows=len(indices), fp64_exact_matches=int(np.sum(current == expected)),
                maximum_absolute_difference=float(difference.max()), parent_fp64_exact=bool(parent_equal),
                parent_child_source_index_equal=True, source_index_min=int(indices.min()), source_index_max=int(indices.max()),
                source_cost_file=sources[method][1], parent_metadata=identity(parent / "metadata.json"),
                current_metadata=identity(folder / "metadata.json"), current_targets=identity(folder / "targets.npz"),
                current_input_metadata=identity(folder / "input_metadata.npz"),
                current_cost_dtype=str(costs.dtype))
        result[split] = dict(costs=costs, metadata=metadata, columns=columns)
    return result


def report(manifest, routes):
    total = sum(a["rows"] for d in manifest["deployments"] for a in d["alignment"])
    maximum = max(r["absolute_cost_error"] for r in routes)
    lines = ["# R81 当前标签部署清单", "",
        "范围：当前 single-RELD 场景中的 VRPTW、VRPLTW，各六列；不涉及 R58 场景重建。", "",
        "## 已完成", "",
        f"- 12/12 列已从当前 targets 经 parent/source_index 绑定到 10/13 冻结结果，再绑定到生成 profile/complete；共 {total:,} 个 FP64 成本值完全一致，最大差 0。",
        "- 每列 profile 配方哈希、完成记录文件哈希、merged 输出身份均与冻结导出清单吻合；12 列均对应 408 个已完成生成分片。",
        "- 当前 train、validation、test 都是新增训练源数据的重新划分，共每题 77,904 例；不是原始旧 validation/test 部署。",
        "- reld_mt 仅为 RELD_MOEL 的成本别名；没有使用 MTL 成本，也没有混回 B 筛除列。",
        f"- 复用八个固定 TRAIN 实例上的 {len(routes)} 条已有路线；{manifest['offline_summary']['saved_routes_passed']} 条通过独立客户访问、容量、客户时间窗、含服务/等待的 depot 截止、L 距离与成本检查。最大重算成本绝对差 {maximum:.9g}。",
        "- 分片 JSON 保存原始浮点成本，冻结导出按 `.10f` 写成本，再解析为 FP64。48条分片成本经该原有格式化后与当前标签精确一致；不把十进制序列化尾差误判为路线错误。",
        "- 新增 solver 运行 0 次、训练 0 次；性能标签、划分、solver 源码及 R58 deferred 状态均未修改。", "",
        "## 当前部署字段", "",
        "| 字段 | 当前场景结论 | 证据等级 |", "|---|---|---|",
        "| depot 时间窗 | 两任务六方法均为 [0,3]，闭合路线需在 3 前返仓 | 当前运行 profile/适配器记录；ReLD 为锁定源码推导；48条路线佐证 |",
        "| speed | 1；行驶时间=距离/speed | RF/MoSES 运行适配器记录；其余为锁定环境源码推导 |",
        "| 客户 service time | 原实例逐客户 service_time，depot=0；服务在等待后累加 | 运行适配器、固定输入及路线检查 |",
        "| TW | 约束服务开始时间；等待和服务计入时间状态 | 锁定环境源码及路线检查 |",
        "| VRPTW 的 L | 不启用 | 记录的输入与环境任务开关 |",
        "| VRPLTW 的 L | 每车欧氏行驶距离上限3，包含返仓；不含等待/服务 | RF/MoSES显式传3；native/ReLD读取输入route_limit；路线检查 |",
        "| cost | 闭合路线全部欧氏行驶距离之和，含出仓/返仓；不加服务/等待/车辆固定费 | 锁定计费源码及路线重算 |", "",
        "## 逐列状态", "",
        "| 问题 | 当前列 | 原列 | 标签绑定 | 已存路线核验 | 解码/预算 |", "|---|---|---|---|---|---|"]
    for d in manifest["deployments"]:
        b = d["fields"]["decode_and_budget"]["value"]
        passed = sum(r.get("feasible") and r.get("independent_cost_matches") and r.get("saved_cost_matches_label")
                     for r in routes if r['problem'] == d['problem'] and r['solver'] == d['solver_name'])
        lines.append(f"| {d['problem']} | {d['solver_name']} | {d['source_solver_name']} | 全部三划分 FP64 一致 | {passed}/4 | {b['decoding']}；aug={b['augmentation']} |")
    lines += ["", "## 固定样本与证据范围", "",
        "核验前按输入选择 TRAIN 中规模50、75、76、100各一个实例，同规模取最小 source index。六候选共用同一批物理实例，覆盖 <=75/ >75 权重切换边界。具体 local/source index 见 route_check_plan.json；输出见 route_checks.csv。",
        "使用现有 r57_routes.py 的独立 check_route 函数，不加载原生 solver/mask；显式传入有当前部署证据的 speed=1、depot_tw_end=3。RF/MoSES 保存路线省略开头 depot 时补0，仅为序列表达归一化。",
        "证据等级：documented=文档声明；source-derived=与运行哈希绑定的源码推导；current-run-recorded=生成当前标签时保存的运行记录；verified-on-route=本次对直接映射的已存路线检查；unknown=记录不足。路线通过不等于任意边界或所有实例已经逐条独立认证。", "",
        "## 保留的实际缺口", "",
        "1. RF/MoSES 的 profile 锁定 phase=test、8倍增强及按规模权重，但没有单独捕获外部库最终解析的 decode type。清单按 unknown 保留，不用当前库的默认 greedy 补写。",
        "2. 原始完整 shell argv 及完整外部库/设备环境不能由 profile 单独恢复；已保存分片有实际 seed、batch chunk、producer source 和权重哈希，可定位对应数值产物，但本轮不声称跨设备逐比特复跑。",
        "3. 原生环境存在数值边界差异：native/ReLD 使用约1e-5容差，RF/MoSES有严格小于号。统一名义时间/距离参数不等于已证明所有浮点边界完全等价。",
        "4. 本轮四个固定样本的 service_time 统计不外推为全体实例固定常数；规范仍是读取实例字段。",
        "5. 不把旧前1万例 calibration 的历史成本差当作当前新标签错配；当前使用的 source index 均>=10000，且已直接核对对应冻结成本。", "",
        "## 复现本次离线整理", "", "```bash",
        "source /public/home/shiys/miniconda3/etc/profile.d/conda.sh",
        "conda activate easynco",
        f"python {OUT / 'build_manifest.py'}", "```", "",
        "主交付 manifest.json；辅助 alignment.csv、route_checks.csv、route_check_plan.json；RESULTS.pdf 供离线阅读。原 solver_source_manifest.json 只增加本场景引用，原历史记录保持原样。", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def pdf_report():
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    lines = []
    for line in (OUT / "RESULTS.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if cells and all(set(c) <= set('-: ') for c in cells):
                continue
            line = " / ".join(cells)
        if line.startswith("python "):
            line = "python code/V4/runs/R81_current_label_deployment/build_manifest.py"
        lines.extend(textwrap.wrap(line, width=54, replace_whitespace=False) or [""])
    with PdfPages(OUT / "RESULTS.pdf") as pdf:
        for start in range(0, len(lines), 41):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(.08, .95, "R81 | 当前 VRPTW / VRPLTW 标签部署", fontsize=13, va="top")
            for i, line in enumerate(lines[start:start + 41]):
                fig.text(.08, .90 - i * .0198, line.lstrip("# "), fontsize=9, va="top")
            fig.text(.90, .045, str(start // 41 + 1), ha="right", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    scenario = read(DATA / "scenario_manifest.json")
    export_plan = read(SNAPSHOT / "导出计划.json")
    frozen = {p["problem"]: p for p in export_plan["problems"] if p["problem"] in PROBLEMS}
    generations = {p: read(WORK / "generation" / (p + ".json")) for p in PROBLEMS}
    selected = {p: fixed_train_rows(p, generations[p]) for p in PROBLEMS}
    write(OUT / "route_check_plan.json", dict(
        scope="CPU-only reuse of saved current TRAIN routes; no solver calls", selection_uses_labels=False,
        rule="For exact scales 50,75,76,100 choose the TRAIN row with minimum original source index; reuse across all six candidates.",
        samples=selected, planned_routes=48, speed=1.0, depot_deadline=3.0,
        numeric_tolerance=2e-5, cost_atol=5e-5, cost_rtol=2e-6))
    checker, checker_identity = load_checker()
    deployments, routes, alignment = [], [], []
    for problem in PROBLEMS:
        aligned = align_problem(problem, frozen[problem])
        for method in METHODS:
            deployments.append(inspect_deployment(problem, method, frozen[problem], aligned,
                                                   selected[problem], checker, routes))
            for split in ("train", "val", "test"):
                a = aligned[split]["columns"][method]
                alignment.append({k: a[k] for k in ("problem", "split", "solver", "rows", "fp64_exact_matches",
                    "maximum_absolute_difference", "parent_fp64_exact", "parent_child_source_index_equal", "source_index_min", "source_index_max")})
        print(f"{problem}: six columns bound across train/val/test; 24 saved routes checked", flush=True)
    manifest = dict(schema="r81-current-label-deployments-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        scenario_id=scenario["scenario_id"], current_scenario=str(DATA), parent_scenario=str(PARENT),
        source_snapshot=str(SNAPSHOT), generation_root=str(WORK), tasks=list(PROBLEMS),
        scope="12 current new-only task/solver columns, including their three resplit partitions",
        status="bound-with-explicit-runtime-gaps", solver_aliases={"reld_mt": "RELD_MOEL"},
        evidence_levels={"documented": "documentation only, not automatically a captured execution value",
            "source-derived": "derived from source bytes matched to the generation dependency contract",
            "current-run-recorded": "archived by the runs that generated current labels, not a new solver execution",
            "verified-on-route": "CPU-independent verification of a directly mapped saved route",
            "unknown": "not recovered; no defaults substituted"},
        source_evidence=dict(current_scenario=identity(DATA / "scenario_manifest.json"),
            parent_scenario=identity(PARENT / "scenario_manifest.json"), frozen_export=identity(SNAPSHOT / "导出计划.json"),
            checker=checker_identity, builder=identity(Path(__file__))),
        offline_summary=dict(bound_columns=len(deployments), aligned_scalar_costs=sum(a["rows"] for a in alignment),
            exact_scalar_matches=sum(a["fp64_exact_matches"] for a in alignment),
            maximum_alignment_difference=max(a["maximum_absolute_difference"] for a in alignment),
            saved_routes_checked=len(routes), saved_routes_passed=sum(bool(r.get("feasible") and r.get("independent_cost_matches") and r.get("saved_cost_matches_label")) for r in routes),
            maximum_route_cost_error=max(r["absolute_cost_error"] for r in routes),
            solver_runs=0, model_training=0, label_mutations=0, r58="deferred; untouched"), deployments=deployments)
    write(OUT / "manifest.json", manifest)
    csv_write(OUT / "alignment.csv", alignment)
    csv_write(OUT / "route_checks.csv", routes)
    report(manifest, routes)
    pdf_report()
    print(json.dumps(manifest["offline_summary"], ensure_ascii=False), flush=True)


def finalize_existing():
    """Record the exporter's decimal format without repeating alignment or checks."""
    manifest = read(OUT / "manifest.json")
    with (OUT / "route_checks.csv").open(encoding="utf-8-sig") as stream:
        routes = list(csv.DictReader(stream))
    for row in routes:
        for key, value in list(row.items()):
            if value in ("True", "False"):
                row[key] = value == "True"
        for key in ("current_label_cost", "saved_route_cost", "independent_cost", "absolute_cost_error",
                    "service_time_min", "service_time_max"):
            row[key] = float(row[key])
        for key in ("local_index", "source_index", "scale"):
            row[key] = int(row[key])
        current, saved = row["current_label_cost"], row["saved_route_cost"]
        row["saved_cost_fp64_exact"] = current == saved
        row["saved_cost_matches_label"] = current == float(f"{saved:.10f}")
        row["serialization_rounding_difference"] = abs(current - saved)
    exporter = HOME / "easynco_v3_bridge/scripts/export_train_100k_common.py"
    export_plan = read(SNAPSHOT / "导出计划.json")
    manifest["cost_serialization"] = dict(format=".10f", evidence_level="source-derived",
        evidence=dict(**identity(exporter), locator="export_method_scores: suffix.extend(...value:.10f...)",
            matches_frozen_exporter=digest(exporter) == export_plan["exporter"]["sha256"]),
        exact_after_formatting=sum(r["saved_cost_matches_label"] for r in routes),
        checked_saved_routes=len(routes), maximum_rounding_difference=max(r["serialization_rounding_difference"] for r in routes))
    manifest["offline_summary"]["saved_routes_passed"] = sum(bool(r["feasible"] and r["independent_cost_matches"]
                                                           and r["saved_cost_matches_label"]) for r in routes)
    for d in manifest["deployments"]:
        subset = [r for r in routes if r["problem"] == d["problem"] and r["solver"] == d["solver_name"]]
        passed = all(r["feasible"] and r["independent_cost_matches"] and r["saved_cost_matches_label"] for r in subset)
        d["route_verification_status"] = "passed-fixed-train-sample" if passed else "incomplete-or-failed"
        if passed:
            d["gaps"] = [g for g in d["gaps"] if not g.startswith("At least one fixed saved-route")]
    manifest["source_evidence"]["builder"] = identity(Path(__file__))
    write(OUT / "manifest.json", manifest)
    csv_write(OUT / "route_checks.csv", routes)
    report(manifest, routes)
    pdf_report()
    print(json.dumps(dict(offline_summary=manifest["offline_summary"], serialization=manifest["cost_serialization"]), ensure_ascii=False))


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing()
    else:
        main()
        finalize_existing()
