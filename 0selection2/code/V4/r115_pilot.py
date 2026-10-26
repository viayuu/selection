"""Finite TRAIN-only, base-disjoint response-learning pilot; no solver or GPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


ROOT = Path(__file__).resolve().parent / "runs/R115_cross_base_response"
SOURCE = Path(__file__).resolve().parent / "runs/R114_paired_responses/preparation"
LABELS = SOURCE / "finish/tensor_boundary_combined_20261026/combined_labels.json"
MANIFEST = SOURCE / "production_inputs/manifest.json"
RESEARCH = Path(__file__).resolve().parents[2] / "docs/research/post_major_paradigm_results_20261026/REPORT.md"
MODELS = ("ridge", "zero", "train_mean")
SIGNAL = (("CVRP", "demand"), ("VRPL", "demand"))
GUARDS = (("OVRPL", "demand"), ("OVRPTW", "tw_center"))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Existing lock differs; use a new version: {path}")
    else:
        save(path, value)


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_folds(endpoints, seed=2):
    """Balance true sizes without inspecting costs; all endpoints share base fold."""
    groups = defaultdict(dict)
    for row in endpoints:
        groups[row["problem"]][row["base_id"]] = int(row["input_check"]["n"])
    folds = {}
    for problem, sizes in sorted(groups.items()):
        rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{problem}".encode()).hexdigest()[:8], 16))
        bases = sorted(sizes)
        rng.shuffle(bases)
        bases.sort(key=sizes.__getitem__)
        if len(bases) % 2:
            raise ValueError("This fixed pilot requires an even base count per problem")
        for start in range(0, len(bases), 2):
            first = int(rng.integers(2))
            for offset, base in enumerate(bases[start:start + 2]):
                folds[base] = (first + offset) % 2
    return folds


def centered_response(before, after, scale):
    change = np.asarray(after, np.float64) - np.asarray(before, np.float64)
    return (change - change.mean(-1, keepdims=True)) / scale


def ridge_fit_predict(x_train, y_train, x_query, alpha=1.0):
    """Standardize exclusively on fit rows; solve the small dual ridge system."""
    center = x_train.mean(0)
    spread = x_train.std(0)
    active = spread > 1e-10
    train = (x_train[:, active] - center[active]) / spread[active]
    query = (x_query[:, active] - center[active]) / spread[active]
    y_center = y_train.mean(0)
    system = train @ train.T + alpha * np.eye(len(train))
    dual = np.linalg.solve(system, y_train - y_center)
    predicted = query @ train.T @ dual + y_center
    # The target is centered across candidates; remove numerical roundoff only.
    predicted -= predicted.mean(1, keepdims=True)
    return predicted, dict(active_columns=int(active.sum()), train_rows=len(train), alpha=alpha)


def gate(comparison):
    groups = defaultdict(dict)
    for row in comparison:
        groups[row["problem"], row["family"]][row["model"]] = row["mae_pct_train_scale"]
    checks = []
    for key in (*SIGNAL, *GUARDS):
        values = groups[key]
        for control in MODELS[1:]:
            ratio = values["ridge"] / values[control] if values[control] > 0 else None
            threshold = .90 if key in SIGNAL else 1.10
            checks.append(dict(problem=key[0], family=key[1], control=control,
                               ridge_to_control_mae_ratio=ratio, maximum_ratio=threshold,
                               passed=bool(ratio is not None and ratio <= threshold)))
    return dict(passed=all(row["passed"] for row in checks), checks=checks,
                guard_interpretation="No >10% harm versus EITHER zero or fold-mean control",
                authority="Predeclared engineering screen; not statistical significance or a selector result")


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest, labels = read(MANIFEST), read(LABELS)
    if manifest["split"] != "train" or labels["summary"]["full_pool_pairs"] != len(labels["pairs"]):
        raise ValueError("Use only complete, corrected TRAIN response pairs")
    if {r["endpoint_id"] for r in labels["endpoints"]} != {r["endpoint_id"] for r in manifest["endpoints"]}:
        raise ValueError("Corrected labels and original input IDs disagree")
    folds = make_folds(manifest["endpoints"])
    fold_rows = []
    originals = {r["base_id"]: r for r in manifest["endpoints"] if r["arm"] == "original"}
    for base, fold in sorted(folds.items()):
        row = originals[base]
        fold_rows.append(dict(base_id=base, problem=row["problem"], fold=fold,
                              n=int(row["input_check"]["n"]), source_index=row["source_index"]))
    immutable(root / "fold_manifest.json", dict(seed=2, folds=2, rows=fold_rows,
        rule="Within each problem, sort true n with seeded ties; adjacent base pairs split to opposite folds",
        same_base_all_families_and_arms_together=True, membership_changed=False))
    config = dict(version="R115_cross_base_response", seed=2,
        scope="TRAIN-only two-fold cross-base response pilot; not a selector training/test",
        source_labels=dict(path=str(LABELS), sha256=sha(LABELS)),
        source_manifest=dict(path=str(MANIFEST), sha256=sha(MANIFEST)),
        research=dict(path=str(RESEARCH), sha256=sha(RESEARCH)),
        folds_sha256=sha(root / "fold_manifest.json"),
        feature=dict(source="Existing input-only R92 357 statistics", columns=714,
            representation="concat(phi(original), phi(changed)-phi(original)); no arm/family/row-ID feature",
            geometry="Existing CPU local geometry; demand/TW-to-geometry correlations retained",
            coord_dist="fixed missing=0; never infer generator distribution from an index"),
        predictor=dict(type="multioutput linear ridge", alpha=1.0, intercept=True,
            sharing="One identical formula fit per problem/family and training fold; arms pooled",
            preprocessing="Training-fold means/std only; discard std <=1e-10; no tuning"),
        target="(cost_after-cost_before minus its candidate mean) / fixed TRAIN-fold problem scale",
        scale="Mean of original endpoint costs over ALL candidates and unique training-fold bases of that problem",
        controls=["zero response", "training-fold problem/family average response, arms pooled"],
        aggregation="Mean over candidates, then both arms, then independent base; no endpoint counted as a new base",
        order_metric="Sign agreement of predicted vs true candidate-pair RESPONSE differences, excluding true exact ties; NOT selection accuracy",
        signal_conditions=[list(x) for x in SIGNAL], guard_conditions=[list(x) for x in GUARDS],
        proceed_rule="Both signal conditions: MAE <=90% of BOTH controls; both guards <=110% of BOTH controls",
        budget=dict(cpu_core_hours_max=8, wall_seconds_max=14400, cpu_threads=2,
                    gpu_hours=0, new_labels=0, validation_reads=0, test_reads=0, lib_reads=0),
        source_files={str(Path(__file__).resolve()): sha(__file__),
                      str(Path(__file__).with_name("r115_features.py")): sha(Path(__file__).with_name("r115_features.py")),
                      str(Path(__file__).with_name("r92_tree_features.py")): sha(Path(__file__).with_name("r92_tree_features.py")),
                      str(Path(__file__).with_name("local_geometry.py")): sha(Path(__file__).with_name("local_geometry.py"))})
    immutable(root / "config.json", config)
    save(root / "progress.json", dict(state="prepared", bases=len(folds), pairs=len(labels["pairs"]),
                                      endpoints=len(labels["endpoints"]), config_sha256=sha(root / "config.json")))


def process_cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def run(root):
    from .r115_features import native_phi
    import torch

    torch.set_num_threads(2)
    config = read(root / "config.json")
    if (root / "complete.json").exists():
        raise ValueError("Completed finite pilot is immutable; do not refit it")
    for source in (config["source_labels"], config["source_manifest"]):
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Source changed after protocol lock")
    for path, expected in config["source_files"].items():
        if sha(path) != expected:
            raise ValueError(f"Code changed after protocol lock: {path}")
    folds = {r["base_id"]: r["fold"] for r in read(root / "fold_manifest.json")["rows"]}
    manifest, labels = read(MANIFEST), read(LABELS)
    endpoint_inputs = {row["endpoint_id"]: row for row in manifest["endpoints"]}
    endpoint_labels = {row["endpoint_id"]: row for row in labels["endpoints"]}
    pairs = sorted(labels["pairs"], key=lambda row: row["pair_id"])
    started, cpu_started = time.monotonic(), process_cpu()

    def status(phase, **kwargs):
        elapsed, cpu = time.monotonic() - started, process_cpu() - cpu_started
        save(root / "progress.json", dict(state="running", phase=phase, elapsed_seconds=elapsed,
                                          cpu_core_hours=cpu / 3600, **kwargs))
        if elapsed > config["budget"]["wall_seconds_max"] or cpu / 3600 > config["budget"]["cpu_core_hours_max"]:
            raise TimeoutError("Predeclared finite CPU budget exhausted; no automatic expansion")

    cache_path = root / "features.npz"
    endpoint_ids = sorted(endpoint_inputs)
    if cache_path.exists():
        raise ValueError("Unexpected partial feature cache; preserve it and inspect explicit recovery, do not overwrite")
    phi = {}
    for index, endpoint_id in enumerate(endpoint_ids):
        row = endpoint_inputs[endpoint_id]
        if sha(row["input_path"]) != row["input_sha256"]:
            raise ValueError("Input endpoint file changed")
        payload = read(row["input_path"])
        if payload["split"] != "train" or payload["base_id"] != row["base_id"]:
            raise ValueError("Input identity or split mismatch")
        phi[endpoint_id] = native_phi(row["problem"], payload["item"]).astype(np.float64)
        if index % 32 == 0:
            status("input_features", endpoints=index + 1, total_endpoints=len(endpoint_ids))
    np.savez_compressed(cache_path, endpoint_ids=np.asarray(endpoint_ids),
                        phi=np.stack([phi[key] for key in endpoint_ids]))
    status("cross_base_fitting", endpoints=len(phi))
    x = np.stack([np.concatenate((phi[row["original_id"]], phi[row["changed_id"]] - phi[row["original_id"]]))
                  for row in pairs])
    if x.shape[1] != config["feature"]["columns"] or not np.isfinite(x).all():
        raise ValueError("Input-only representation shape/values invalid")
    groups = defaultdict(list)
    for index, pair in enumerate(pairs):
        if pair["base_id"] not in folds:
            raise ValueError("Pair base absent from frozen fold assignment")
        old, new = endpoint_labels[pair["original_id"]], endpoint_labels[pair["changed_id"]]
        if old["methods"] != new["methods"] or not all(old["valid_mask"]) or not all(new["valid_mask"]):
            raise ValueError("Partial or changed candidate pool cannot define a response target")
        groups[pair["problem"], pair["family"]].append(index)
    predictions, fit_receipts = [], []
    for (problem, family), indices in sorted(groups.items()):
        for fold in range(2):
            train = [i for i in indices if folds[pairs[i]["base_id"]] != fold]
            query = [i for i in indices if folds[pairs[i]["base_id"]] == fold]
            train_bases = {pairs[i]["base_id"] for i in train}
            query_bases = {pairs[i]["base_id"] for i in query}
            if not train_bases or train_bases & query_bases or len(train_bases) != len(query_bases):
                raise ValueError("Cross-base fold leakage or empty/unbalanced fold")
            original_rows = [r for r in labels["endpoints"] if r["base_id"] in train_bases and
                              r["endpoint_id"] == r["problem"] + "_src" + str(r["source_index"]) + "_original"]
            if len(original_rows) != len(train_bases):
                raise ValueError("Scale must use each training original exactly once")
            scale = float(np.mean([np.mean(r["costs_fp64"]) for r in original_rows]))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("TRAIN-fold fixed scale must be finite and positive")

            def targets(selection):
                return np.stack([centered_response(endpoint_labels[pairs[i]["original_id"]]["costs_fp64"],
                                                   endpoint_labels[pairs[i]["changed_id"]]["costs_fp64"], scale)
                                 for i in selection])

            y_train, y_query = targets(train), targets(query)
            predicted, receipt = ridge_fit_predict(x[train], y_train, x[query], config["predictor"]["alpha"])
            receipt.update(problem=problem, family=family, query_fold=fold, scale=scale,
                           train_bases=len(train_bases), query_bases=len(query_bases), query_rows=len(query))
            fit_receipts.append(receipt)
            estimates = dict(ridge=predicted, zero=np.zeros_like(predicted),
                             train_mean=np.broadcast_to(y_train.mean(0), predicted.shape))
            for row_number, index in enumerate(query):
                pair = pairs[index]
                for slot, method in enumerate(pair["methods"]):
                    record = {key: pair[key] for key in ("pair_id", "base_id", "problem", "family", "arm")}
                    record.update(query_fold=fold, method=method, train_only_scale=scale,
                                  target_delta=float(y_query[row_number, slot]))
                    record.update({model: float(estimates[model][row_number, slot]) for model in MODELS})
                    predictions.append(record)
        status("cross_base_fitting", groups_completed=len(fit_receipts) // 2, total_groups=len(groups))
    write_csv(root / "predictions.csv", predictions)
    save(root / "fit_receipts.json", fit_receipts)
    comparison, by_arm = summaries(predictions)
    write_csv(root / "comparison.csv", comparison)
    write_csv(root / "by_arm.csv", by_arm)
    result = gate(comparison)
    result.update(bases=len(folds), pairs=len(pairs), endpoints=len(phi), conditions=len(groups),
                  config_sha256=sha(root / "config.json"),
                  cpu_core_hours=(process_cpu() - cpu_started) / 3600,
                  elapsed_seconds=time.monotonic() - started,
                  training_validation_test_membership_unchanged=True, new_selector=False,
                  neural_model_evaluated=False, solver_runs=0, gpu_hours=0)
    save(root / "decision.json", result)
    render(root, comparison, result)
    save(root / "complete.json", dict(state="complete", **result))
    save(root / "progress.json", dict(state="complete", passed=result["passed"],
                                      cpu_core_hours=result["cpu_core_hours"], elapsed_seconds=result["elapsed_seconds"]))
    print(json.dumps(result, ensure_ascii=False), flush=True)


def summaries(predictions):
    pair_rows = defaultdict(list)
    for row in predictions:
        pair_rows[row["pair_id"]].append(row)
    records = []
    for rows in pair_rows.values():
        target = np.asarray([r["target_delta"] for r in rows])
        upper = np.triu_indices(len(target), 1)
        difference = (target[:, None] - target[None, :])[upper]
        valid = np.abs(difference) > 1e-12
        for model in MODELS:
            predicted = np.asarray([r[model] for r in rows])
            estimated = (predicted[:, None] - predicted[None, :])[upper]
            row = {key: rows[0][key] for key in ("pair_id", "base_id", "problem", "family", "arm")}
            row.update(model=model, mae_pct_train_scale=float(np.abs(target - predicted).mean() * 100),
                       response_order_accuracy_pct=float((np.sign(estimated[valid]) == np.sign(difference[valid])).mean() * 100)
                           if valid.any() else None,
                       response_order_coverage_pct=float((np.abs(estimated[valid]) > 1e-12).mean() * 100)
                           if valid.any() else None)
            records.append(row)

    def aggregate(with_arm):
        groups = defaultdict(list)
        keys = ("problem", "family", "arm", "model") if with_arm else ("problem", "family", "model")
        for row in records:
            groups[tuple(row[key] for key in keys)].append(row)
        output = []
        for group, rows in sorted(groups.items()):
            bases = defaultdict(list)
            for row in rows:
                bases[row["base_id"]].append(row)
            result = dict(zip(keys, group))
            result.update(bases=len(bases), pairs=len(rows))
            for metric in ("mae_pct_train_scale", "response_order_accuracy_pct", "response_order_coverage_pct"):
                values = [np.mean([r[metric] for r in items if r[metric] is not None])
                          for items in bases.values() if any(r[metric] is not None for r in items)]
                result[metric] = float(np.mean(values)) if values else None
            output.append(result)
        return output

    return aggregate(False), aggregate(True)


def render(root, comparison, decision):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager

    font = Path("/public/home/shiys/.cache/selector_fonts/20261026/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(), "axes.unicode_minus": False})
    groups = defaultdict(dict)
    for row in comparison:
        groups[row["problem"], row["family"]][row["model"]] = row
    lines = ["# R115：现有干预响应能否跨基础实例预测？", "",
        f"结论：{'通过预声明的继续门槛' if decision['passed'] else '未通过预声明的继续门槛，不自动扩大响应标签/神经训练'}。", "",
        "只读取 R114 已修正的 TRAIN 数据：512 个独立基础实例、2,048 个端点、1,536 个响应对、16 题/24 条件。",
        "每题按真实规模相邻分组，两折各16个基础实例；同源的全部 family、定向/随机端点不跨折。",
        "固定 Ridge(alpha=1)，输入为已有357维统计的原点及端点差；标准化、固定成本尺度与均值对照均只拟合训练折。",
        "", "| 条件 | Ridge MAE | 零响应 MAE | 训练折均值 MAE | 相对零/均值改善 |", "|---|---:|---:|---:|---:|"]
    table = []
    for key, values in sorted(groups.items()):
        ridge, zero, avg = [values[m]["mae_pct_train_scale"] for m in MODELS]
        improve = [(1 - ridge / baseline) * 100 if baseline else float("nan") for baseline in (zero, avg)]
        label = key[0] + "/" + key[1]
        lines.append(f"| {label} | {ridge:.5f} | {zero:.5f} | {avg:.5f} | {improve[0]:+.1f}% / {improve[1]:+.1f}% |")
        table.append([label, f"{ridge:.4f}", f"{zero:.4f}", f"{avg:.4f}", f"{improve[0]:+.1f}% / {improve[1]:+.1f}%"])
    lines.extend(["", "MAE 单位：固定训练折成本尺度的百分比；不是逐实例 actual regret，也不是 Top1。",
        "候选次序指标是成本响应差的正负次序，不读取原点真实成本作为预测锚点，不是最终方法选择准确率。",
        "门槛：CVRP-demand、VRPL-demand 相对两种对照均降低至少10%；OVRPL-demand、OVRPTW-tw_center 对任一对照不恶化超过10%。",
        "", "## 结论边界", "这一小样本、固定统计特征的线性检验不能证明任意神经响应模型都无效，也不能证明标签随机或准确率上限。",
        "原 R114 定向/随机干预幅度未严格匹配；本轮不将其响应大小差解释为因果效果。",
        "没有新增 solver 求解或标签，没有读取 validation/TEST/LIB，没有训练或发布新的完整选择器。",
        f"实际耗时 {decision['elapsed_seconds']:.1f} 秒，CPU {decision['cpu_core_hours']:.4f} core-hours；GPU 0。",
        "完整逐方法预测见 predictions.csv；分臂结果见 by_arm.csv，全部预声明检查见 decision.json。"])
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    with PdfPages(root / "RESULTS.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(12.8, 15))
        ax.axis("off")
        ax.set_title("R115 | TRAIN 内跨基础实例响应学习", fontsize=19, pad=30)
        text = ("通过预声明门槛" if decision["passed"] else "未通过预声明门槛：本版止步，不自动追加响应训练")
        ax.text(0, .99, text + "\n512 bases · 16题 · 两折不跨base · Ridge α=1\nMAE单位：固定训练折成本尺度的百分比；不是选择regret", transform=ax.transAxes, fontsize=12, va="top")
        tab = ax.table(cellText=table, colLabels=["条件", "Ridge", "零响应", "折内均值", "相对两对照改善"],
                       bbox=[0, .11, 1, .79], colWidths=[.27, .13, .13, .13, .34], cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#dce5e8")
            cell.set_facecolor("#e2edef" if r == 0 else "#f2f6f6" if r % 2 else "white")
        ax.text(0, .07, "同源端点不独立计数；全部24条件均保留。未读取VAL/TEST/LIB，0新增标签，0 GPU。\n阴性结果仅限制这套小样本、固定统计特征的线性预测器，不宣称准确率上限。", transform=ax.transAxes, fontsize=11, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, key in zip(axes.flat, (*SIGNAL, *GUARDS)):
            values = groups[key]
            ax.bar(["Ridge", "零响应", "训练折均值"], [values[m]["mae_pct_train_scale"] for m in MODELS], color=["#177f88", "#bfcdd2", "#c54d59"])
            ax.set_title(key[0] + " / " + key[1])
            ax.set_ylabel("MAE (% TRAIN固定尺度)")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("四个预声明条件：两个正信号条件 + 两个反例保护条件", fontsize=15)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root)
    else:
        try:
            run(args.root)
        except Exception as exc:
            save(args.root / "failure.json", dict(state="failed", error_type=type(exc).__name__, error=str(exc)))
            raise


if __name__ == "__main__":
    main()
