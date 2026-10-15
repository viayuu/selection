"""Incremental, CPU-only comparison of completed NSS VAL predictions with R77B."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from .r76_probability_diagnostics import file_hash, read_json, score_ranks, write_rows


V4 = Path(__file__).resolve().parent
NSS = V4 / "runs/R79_NSS_new_data_batch1024"
B = V4 / "runs/R77_sample_size_batch1024"
DATA = Path("/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014")
OUT = V4 / "runs/R78_zero_selection_goal/review-stage"
STEM = "NSS_TRAIN_VAL_COMPARISON"


def analyze(task):
    complete_path = NSS / task / "training_complete.json"
    complete = read_json(complete_path)
    if complete["epochs"] != 50:
        raise ValueError("Only a completed 50-epoch task may be summarized")
    manifest = read_json(B / "selected_train_val_predictions/manifest.json")
    b_lock = manifest["arms"]["B"]
    nss_path = NSS / task / "best_val_predictions.npz"
    b_path = Path(b_lock["files"]["val"]["path"])
    with np.load(nss_path, allow_pickle=False) as saved:
        nss = {key: saved[key] for key in ("pool", "pred", "costs", "winner", "source_indices", "indices")}
    with np.load(b_path, allow_pickle=False) as saved:
        raw = {key: saved[task + "__" + key] for key in
               ("pool", "pool_ids", "logits", "costs", "winner", "source_indices", "indices")}
    with np.load(DATA / (task + "val/input_metadata.npz"), allow_pickle=False) as saved:
        snapshot_rows = saved["source_index"]
    pool, b_pool = nss["pool"].tolist(), raw["pool"].tolist()
    if len(set(pool)) != len(pool) or set(pool) != set(b_pool):
        raise ValueError("VAL candidate names must agree exactly")
    n_order = np.argsort(nss["source_indices"], kind="stable")
    b_sources = snapshot_rows[raw["source_indices"]]
    b_order = np.argsort(b_sources, kind="stable")
    n = len(n_order)
    if (n != 3895 or len(np.unique(nss["source_indices"])) != n
            or not np.array_equal(nss["source_indices"][n_order], b_sources[b_order])
            or not np.array_equal(nss["indices"], np.arange(n))
            or not np.array_equal(raw["indices"], np.arange(n))):
        raise ValueError("The two saved VAL instance populations must align")
    columns = [b_pool.index(name) for name in pool]
    b_to_nss = np.asarray([pool.index(name) for name in b_pool])
    costs, winner = nss["costs"][n_order], nss["winner"][n_order]
    if (costs.dtype != np.float64
            or not np.array_equal(costs, raw["costs"][b_order][:, columns])
            or not np.array_equal(winner, b_to_nss[raw["winner"][b_order]])):
        raise ValueError("Aligned FP64 costs and native winner identities must agree")
    b_pred, _ = score_ranks(raw["logits"], raw["pool_ids"], ~np.isneginf(raw["logits"]))
    predictions = dict(R77B=b_to_nss[b_pred[b_order]], NSS=nss["pred"][n_order])
    stamp = read_json(NSS / "cpu_cache" / (task + "train.json"))
    if stamp["rows"] != 70114 or stamp["identity"]["pool"] != pool:
        raise ValueError("Shared TRAIN population/candidate identities must agree")
    label_path = DATA / (task + "train/raw_label.pkl")
    if str(label_path) not in {record["path"] for record in stamp["identity"]["files"]}:
        raise ValueError("The NSS TRAIN label cache must refer to the same release")
    with (B / "train_val_selection_diagnosis/B_train_val_native_supervision.csv").open(newline="") as stream:
        training = {row["native_winner_name"]: row for row in csv.DictReader(stream)
                    if row["problem"] == task}
    oracle, rows = costs.min(1), np.arange(n)
    fp32_winner = costs.astype(np.float32).argmin(1)
    methods, metrics = [], {}
    zero_b = [i for i in range(len(pool)) if not np.any(predictions["R77B"] == i) and np.any(winner == i)]
    cohort = np.isin(winner, zero_b)
    for model, predicted in predictions.items():
        selected_cost = costs[rows, predicted]
        correct = predicted == winner
        counts = np.bincount(predicted, minlength=len(pool))
        zero = (counts == 0) & (np.bincount(winner, minlength=len(pool)) > 0)
        metrics[model] = dict(
            n=n, correct_count=int(correct.sum()), top1_native_pct=float(100 * correct.mean()),
            top1_fp32_pct=float(100 * np.mean(predicted == fp32_winner)),
            native_fp32_winner_disagreements=int((winner != fp32_winner).sum()),
            mean_cost=float(selected_cost.mean()),
            actual_regret_pct=float(100 * np.mean((selected_cost - oracle) / oracle)),
            zero_method_pairs=int(zero.sum()), zero_winner_count=int(np.isin(winner, np.flatnonzero(zero)).sum()),
            zero_winner_pct=float(100 * np.isin(winner, np.flatnonzero(zero)).mean()),
            b_zero_cohort_correct_count=int((correct & cohort).sum()),
            b_zero_cohort_actual_regret_pct=float(100 * np.mean((selected_cost[cohort] - oracle[cohort]) / oracle[cohort]))
                if cohort.any() else None,
        )
        for slot, method in enumerate(pool):
            native_count = int((winner == slot).sum())
            if int(training[method]["val_true_winner_count"]) != native_count:
                raise ValueError("Existing native-winner table must describe these VAL labels")
            methods.append(dict(
                task=task, status="task_complete_global_freeze_pending", split="val", model=model,
                method=method, n=n, train_instances=70114,
                train_native_winner_count=int(training[method]["train_true_winner_count"]),
                train_selected_count=int(training[method]["train_selected_count"]) if model == "R77B" else "",
                native_winner_count=native_count, selected_count=int(counts[slot]),
                selected_pct=float(100 * counts[slot] / n),
                correct_selected_count=int((correct & (winner == slot)).sum()),
                winner_recall_pct=float(100 * (correct & (winner == slot)).sum() / native_count) if native_count else "",
                zero_with_native_winner=bool(zero[slot]),
                near_zero_with_native_winner=bool(0 < 100 * counts[slot] / n <= .1 and native_count > 0),
                b_zero_on_val=slot in zero_b,
            ))
    print(json.dumps(dict(task=task, metrics=metrics, methods=methods), indent=2))
    return dict(
        task=task, n=n, metrics=metrics, methods=methods,
        nss_best_epoch_zero_based=int(complete["best_epoch"]),
        nss_best_epoch_one_based=int(complete["best_epoch"]) + 1,
        nss_completed_updates=int(complete["updates"]), nss_checkpoint_sha256=complete["best_sha256"],
        b_selected_epoch=b_lock["evaluation_point"], b_selected_update=b_lock["selected_update"],
        b_checkpoint_sha256=b_lock["checkpoint_sha256"],
        nss_predictions=str(nss_path), nss_predictions_sha256=file_hash(nss_path),
        nss_completion=str(complete_path), b_predictions=str(b_path),
        train_supervision_table=str(B / "train_val_selection_diagnosis/B_train_val_native_supervision.csv"),
        b_zero_cohort_winner_count=int(cohort.sum()),
        b_zero_methods=[pool[i] for i in zero_b],
    )


def render(state):
    text = ["# NSS TRAIN/VAL 与 R77B：已完成任务比较", "",
            "执行与结果检查：Sol；仅本地 CPU 读取已有缓存，不交 Astra 复核。当前逐任务完成摘要，不替代待 CVRP 完成后统一生成的正式 locked_train_val_handoff.json。", "",
            "TRAIN 边界：共享 70,114 原始实例/题，方法真实 winner 计数复用 R77B 已有 native 表，并核对 NSS TRAIN cache stamp 的数据目录、方法名和行数。cpu_cache/*train.pt 是成本/native winner/输入缓存，不是模型 TRAIN 预测；NSS TRAIN 选择分布不可用，CSV 对应字段留空，不填0。", ""]
    csv_rows = []
    for task in ("TSP", "CVRP"):
        if task not in state["tasks"]:
            text.extend([f"## {task}：PENDING", "", "尚未纳入完成缓存；不读取仍在更新的预测，不重放 VAL。自然完成后仅追加该任务，保留已完成统计。", ""])
            csv_rows.append(dict(task=task, status="pending", split="val", model="NSS", method="__ALL__"))
            continue
        record = state["tasks"][task]
        b, nss = record["metrics"]["R77B"], record["metrics"]["NSS"]
        text.extend([f"## {task}：完成，统一 freeze 待定", "",
                     f"NSS：50轮/{record['nss_completed_updates']:,} updates，最佳第{record['nss_best_epoch_one_based']}轮（JSON零基 {record['nss_best_epoch_zero_based']}），模型 SHA `{record['nss_checkpoint_sha256']}`。R77B：ep{record['b_selected_epoch']}/update{record['b_selected_update']:,}。同一 {record['n']:,} VAL 行已映射到原始快照 source_index，候选名称、FP64成本、native winner逐行一致。", "",
                     "| VAL指标 | R77B | NSS | NSS−B |", "| --- | ---: | ---: | ---: |"])
        for title, field, digits in (("Top1 native (%)", "top1_native_pct", 6), ("Top1 FP32标签 (%)", "top1_fp32_pct", 6),
                                     ("Mean cost", "mean_cost", 9), ("Actual regret (%)", "actual_regret_pct", 9),
                                     ("零选择且真实获胜的方法对", "zero_method_pairs", 0),
                                     ("上述方法覆盖native winner数", "zero_winner_count", 0)):
            text.append(f"| {title} | {b[field]:.{digits}f} | {nss[field]:.{digits}f} | {nss[field] - b[field]:+.{digits}f} |")
        text.extend(["", f"FP32 argmin 与 native winner 有 {nss['native_fp32_winner_disagreements']} 行不同；主比较使用 native Top1，不混用两种标签。Actual regret = mean((所选FP64成本 / 同池FP64最小成本 − 1)×100)。", "",
                     "| 方法 | TRAIN真实winner | B TRAIN选择 | VAL真实winner | B VAL选择/命中 | NSS VAL选择/命中 |", "| --- | ---: | ---: | ---: | --- | --- |"])
        b_methods = {row["method"]: row for row in record["methods"] if row["model"] == "R77B"}
        nss_methods = {row["method"]: row for row in record["methods"] if row["model"] == "NSS"}
        for method, row in b_methods.items():
            other = nss_methods[method]
            text.append(f"| {method} | {row['train_native_winner_count']:,} | {row['train_selected_count']:,} | {row['native_winner_count']:,} | {row['selected_count']:,}/{row['correct_selected_count']:,} | {other['selected_count']:,}/{other['correct_selected_count']:,} |")
        still_zero = [name for name in record["b_zero_methods"] if nss_methods[name]["selected_count"] == 0]
        text.extend(["", f"B 的 VAL 零选择真实获胜方法：{', '.join(record['b_zero_methods'])}；其中 NSS 仍零选：{', '.join(still_zero) or '无'}。这组 native winner 共 {record['b_zero_cohort_winner_count']:,} 行，B/NSS 命中分别 {b['b_zero_cohort_correct_count']}/{nss['b_zero_cohort_correct_count']}。覆盖改善只能以这些行的实际命中与成本判断，非零选择本身不构成修复。", "",
                     f"结果判断：NSS−B 的 native Top1 为 {nss['top1_native_pct'] - b['top1_native_pct']:+.6f} 个百分点（正确实例数差 {nss['correct_count'] - b['correct_count']:+d}）；平均成本差 {nss['mean_cost'] - b['mean_cost']:+.9f}，actual regret 差 {nss['actual_regret_pct'] - b['actual_regret_pct']:+.9f} 个百分点。Top1 与成本的方向分别报告，不能互相替代。", "",
                     f"在 B 零类真实winner这一相同子集，B/NSS actual regret 为 {b['b_zero_cohort_actual_regret_pct']:.6f}% / {nss['b_zero_cohort_actual_regret_pct']:.6f}%（均未命中的类别不因此变成覆盖修复）。" if record['b_zero_cohort_winner_count'] else "B没有零选择且真实获胜的方法子集。", "",
                     f"来源：[NSS现成VAL预测]({record['nss_predictions']})、[完成记录]({record['nss_completion']})、[R77B锁定VAL]({record['b_predictions']})。", ""])
        csv_rows.extend(record["methods"])
        for model, metrics in record["metrics"].items():
            csv_rows.append(dict(task=task, status="task_complete_global_freeze_pending", split="val", model=model,
                                 method="__ALL__", **metrics))
    text.extend(["## 结论边界", "", "这是各自按VAL选定模型在同一VAL上的描述比较，不是独立泛化验证或单变量因果结论；不能将方法非零选择频数等同修复，也不计算标签后择优联合成绩。没有读取 TEST/LIB、没有新推理/GPU/训练/tests/profile/模型修改/来源追踪。CVRP完成前不汇总两任务整体成绩。", ""])
    fields = list(dict.fromkeys(key for row in csv_rows for key in row))
    write_rows(OUT / (STEM + ".csv"), csv_rows, fields=fields)
    (OUT / (STEM + ".md")).write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("TSP", "CVRP"), required=True)
    args = parser.parse_args()
    path = OUT / (STEM + ".json")
    state = read_json(path) if path.exists() else dict(tasks={}, test_lib_read=False, inference_performed=False)
    if args.task in state["tasks"]:
        raise SystemExit(f"{args.task} already summarized; refusing to recompute completed statistics")
    state["tasks"][args.task] = analyze(args.task)
    OUT.mkdir(parents=True, exist_ok=True)
    render(state)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
