"""Read locked R77 TRAIN/VAL NPZ and reuse R76 probability diagnostics only."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from . import r76_probability_diagnostics as base


ROOT = base.RUN_ROOT.parent / "R77_sample_size_batch1024"
ARMS = {"R77_A_10k_bs1024": ("A_10k_seed2", 10000),
        "R77_B_full_bs1024": ("B_full_seed2", 70114)}
BASELINE = "R76_full_bs9856"
NEAR_ZERO_PCT = .1
METRICS = ("mean_cost", "oracle_cost", "vs_sbs_pct", "actual_regret_pct")
TABLES = ("method_summary", "problem_summary", "missed_candidate_summary",
          "missed_selection_frequencies", "calibration_summary", "calibration_bins",
          "class_calibration", "class_calibration_bins", "split_summary")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    def parse(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def tag_rows(rows, model, info):
    return [dict(model=model, logical_batch_size=info["logical_batch_size"],
                 selected_epoch=info["selected_epoch"], selected_update=info.get("selected_update"),
                 checkpoint_sha256=info["checkpoint_sha256"], **row) for row in rows]


def add_cost_metrics(tables, metrics):
    for row in tables["problem_summary"]:
        original = metrics["per_problem"][row["problem"]]
        for key in METRICS:
            row[key] = original[key]
        row["top2_pct"], row["top3_pct"] = 100 * original["top2"], 100 * original["top3"]
        row["ce"] = next(r["ce"] for r in tables["calibration_summary"] if r["problem"] == row["problem"])
    aggregate = tables["split_summary"][0]
    for key in METRICS:
        aggregate["macro_" + key] = metrics["macro"].get(
            "macro_" + key, float(np.mean([r[key] for r in metrics["per_problem"].values()])))


def load_baseline(old_root):
    research = old_root / "missed_winner_probabilities/research_train_val"
    lock = base.read_json(old_root / "final_test_lock.json")
    config = base.read_json(old_root / "protocol.json")
    source_protocol = base.read_json(research / "protocol.json")
    if config["logical_global_batch"] != 9856 or source_protocol["near_zero_selected_pct"] != NEAR_ZERO_PCT:
        raise ValueError("The baseline must be the existing bs9856, same descriptive near-zero definition")
    if source_protocol["checkpoint_sha256"] != lock["checkpoint_sha256"]:
        raise ValueError("Existing R76 research tables do not match the locked checkpoint")
    info = dict(state="ready", logical_batch_size=9856, train_rows_per_problem=70114,
                val_rows_per_problem=3895, selected_epoch=lock["epoch"], selected_update=None,
                checkpoint_sha256=lock["checkpoint_sha256"], data_root=config["data_root"])
    result, paths = {}, [old_root / "final_test_lock.json", old_root / "protocol.json", research / "protocol.json"]
    for split in ("train", "val"):
        directory = old_root / "evaluation" / split / ("selected_best" if split == "train" else f"epoch{lock['epoch']:03d}")
        metrics_path = directory / "metrics.json"
        metrics = base.read_json(metrics_path)
        tables = {}
        for name in TABLES:
            path = research / f"{name}_{split}.csv"
            tables[name] = read_csv(path)
            paths.append(path)
        add_cost_metrics(tables, metrics)
        result[split] = tables
        info[split + "_source"] = str(directory)
        paths.append(metrics_path)
    return dict(info=info, splits=result, sources=paths)


def locked_sources(root, model):
    arm, train_n = ARMS[model]
    directory = root / arm
    required = [directory / name for name in
                ("training_complete.json", "final_test_lock.json", "final_metrics.json", "protocol.json")]
    if not all(path.is_file() for path in required):
        return None
    complete, lock, final, config = (base.read_json(path) for path in required)
    # Check 1: completed fixed budget and a matching immutable VAL-selected checkpoint.
    if (complete["successful_updates"] != 49680 or config["max_successful_updates"] != 49680
            or config["global_batch_size"] != 1024 or config["train_rows_per_problem"] != train_n
            or not lock["selected_without_test"] or lock["arm"] != arm[0]
            or lock["selected_update"] != final["selected_update"] or lock["epoch"] != final["epoch"]
            or complete["selected_best"]["checkpoint_sha256"] != lock["checkpoint_sha256"]
            or final["checkpoint_sha256"] != lock["checkpoint_sha256"]
            or base.file_hash(directory / "best.pt") != lock["checkpoint_sha256"]):
        raise ValueError(f"{model}: inconsistent completed training / checkpoint lock")
    paths, sources = {}, required + [directory / "best.pt"]
    for split in ("train", "val"):
        expected_n = train_n if split == "train" else 3895
        if (set(final[split]["per_problem"]) != set(base.PROBLEMS)
                or any(r["n"] != expected_n for r in final[split]["per_problem"].values())):
            raise ValueError(f"{model}/{split}: wrong evaluated population")
        # A best at an every-five TRAIN point is already saved under its epoch tag.
        candidates = (["selected_best", f"epoch{lock['epoch']:03d}"] if split == "train"
                      else [f"epoch{lock['epoch']:03d}"])
        chosen = next((directory / "evaluation" / split / tag for tag in candidates
                       if (directory / "evaluation" / split / tag / "metrics.json").is_file()
                       and list((directory / "evaluation" / split / tag).glob("rank*.npz"))), None)
        if chosen is None:
            return None
        paths[split] = chosen
    info = dict(state="ready", logical_batch_size=1024, train_rows_per_problem=train_n,
                val_rows_per_problem=3895, selected_epoch=lock["epoch"], selected_update=lock["selected_update"],
                checkpoint_sha256=lock["checkpoint_sha256"], data_root=config["data_root"],
                train_source=str(paths["train"]), val_source=str(paths["val"]))
    return dict(info=info, paths=paths, final=final, sources=sources)


def check_source_rows(root, model, split, prediction_paths):
    arm, train_n = ARMS[model]
    parts = {}
    for path in prediction_paths:
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if key.endswith("__logits"):
                    prefix = key[:-len("logits")]
                    problem = prefix[:-2]
                    parts.setdefault(problem, []).append((saved[prefix + "indices"], saved[prefix + "source_indices"]))
    # Check 2: local rows map to exactly the actual arm's TRAIN subset or full held-out rows.
    for problem, chunks in parts.items():
        local = np.concatenate([value[0] for value in chunks])
        source = np.concatenate([value[1] for value in chunks])[np.argsort(local, kind="stable")]
        expected = (np.load(root / "indices/A" / f"{problem}.npy", allow_pickle=False)
                    if split == "train" and arm.startswith("A")
                    else np.arange(train_n if split == "train" else 3895))
        if not np.array_equal(source, expected):
            raise ValueError(f"{model}/{split}/{problem}: source rows differ from actual training population")


def diagnose(predictions, metrics, split):
    methods, candidates, flows, problems = [], [], [], []
    for problem, value in predictions.items():
        m, c, f = base.method_diagnostics(value, NEAR_ZERO_PCT)
        methods.extend(m)
        candidates.extend(c)
        flows.extend(f)
        zero = [r for r in m if r["zero_selected_with_winner"]]
        near = [r for r in m if r["near_zero_selected_with_winner"]]
        missed = ~value["correct"]
        problems.append(dict(problem=problem, split=split, instances=value["n"], candidates=value["k"],
                             top1_pct=100 * float(value["correct"].mean()), missed_count=int(missed.sum()),
                             zero_method_count=len(zero), zero_winner_count=sum(r["true_winner_count"] for r in zero),
                             zero_winner_pct=100 * sum(r["true_winner_count"] for r in zero) / value["n"],
                             near_zero_method_count=len(near), zero_methods=";".join(r["solver"] for r in zero),
                             costloss_sum=float(value["costloss"].sum(dtype=np.float64)),
                             mean_costloss=float(value["costloss"].mean()),
                             relative_costloss_pct=float(np.nanmean(value["relative_costloss_pct"])),
                             zero_costloss_share_pct=sum(r["problem_costloss_share_pct"] for r in zero),
                             native_winner_nonminimum_count=int((value["winner_cost"] != value["oracle_cost"]).sum()),
                             missed_cost_tie_count=int((missed & (value["costloss"] == 0)).sum()),
                             negative_costloss_count=int((value["costloss"] < 0).sum())))
    calibration, bins, classes, class_bins, aggregate = base.calibration_reports(predictions, split)
    zero = [r for r in methods if r["zero_selected_with_winner"]]
    aggregate.update(missed_count=sum(r["missed_count"] for r in problems), zero_method_count=len(zero),
                     zero_winner_count=sum(r["true_winner_count"] for r in zero),
                     near_zero_method_count=sum(r["near_zero_selected_with_winner"] for r in methods),
                     zero_winner_pct=100 * sum(r["true_winner_count"] for r in zero) / aggregate["instances"])
    tables = dict(method_summary=methods, problem_summary=problems, missed_candidate_summary=candidates,
                  missed_selection_frequencies=flows, calibration_summary=calibration, calibration_bins=bins,
                  class_calibration=classes, class_calibration_bins=class_bins, split_summary=[aggregate])
    add_cost_metrics(tables, metrics)
    return tables


def load_arm(root, model, locked):
    splits, sources = {}, list(locked["sources"])
    for split in ("train", "val"):
        predictions, paths = base.load_shards(locked["paths"][split], split, locked["final"][split]["per_problem"])
        check_source_rows(root, model, split, paths)
        splits[split] = diagnose(predictions, locked["final"][split], split)
        sources.extend(paths + [locked["paths"][split] / "metrics.json"])
        del predictions
    return dict(info=locked["info"], splits=splits, sources=sources)


def comparisons(rows, reference_keys, fields):
    baseline = {tuple(r[key] for key in reference_keys): r for r in rows if r["model"] == BASELINE}
    result = []
    for row in rows:
        reference = baseline[tuple(row[key] for key in reference_keys)]
        record = dict(row)
        for field in fields:
            current, previous = row[field], reference[field]
            record["r76_" + field] = previous
            record["delta_vs_r76_" + field] = current - previous if current is not None and previous is not None else None
        result.append(record)
    return result


def native_b_handoff(output, combined):
    model = "R77_B_full_bs1024"
    rows = [r for r in combined["method_summary"] if r["model"] == model]
    if not rows:
        return ""
    lookup = {(r["split"], r["problem"], r["solver"]): r for r in rows}
    fields = ("instances", "true_winner_count", "true_winner_pct", "selected_count", "selected_pct",
              "correct_selected_count", "missed_winner_count", "zero_selected_with_winner",
              "near_zero_selected_with_winner", "winner_probability_mean", "winner_probability_median",
              "winner_probability_q05", "winner_probability_q95", "winner_rank_mean", "winner_rank_median",
              "winner_ce_mean", "mean_costloss_contribution", "problem_costloss_share_pct",
              "relative_costloss_contribution_pp")
    paired = []
    for row in rows:
        if row["split"] != "train":
            continue
        record = dict(problem=row["problem"], native_winner_name=row["solver"],
                      global_solver_id=row["solver_id"], selected_epoch=row["selected_epoch"],
                      selected_update=row["selected_update"])
        for split in ("train", "val"):
            value = lookup[split, row["problem"], row["solver"]]
            record.update({split + "_" + key: value[key] for key in fields})
        paired.append(record)
    zero = [r for r in rows if r["zero_selected_with_winner"]]
    base.write_rows(output / "B_zero_methods.csv", zero, list(rows[0]))
    base.write_rows(output / "B_train_val_native_supervision.csv", paired)
    zero_paired = [r for r in paired if r["train_zero_selected_with_winner"] or r["val_zero_selected_with_winner"]]
    base.write_rows(output / "B_train_val_zero_native_names.csv", zero_paired, list(paired[0]))
    counts = {r["split"]: r for r in combined["split_summary"] if r["model"] == model}
    shared = sum(r["train_zero_selected_with_winner"] and r["val_zero_selected_with_winner"] for r in paired)
    lines = ["", "## B最终native监督交接", "",
             "B唯一锁定为epoch28/update34,776；A为epoch13/update16,146。R76、A、B的TRAIN/VAL全部18题已经ready，不含任何R77 TEST数字。Astra请先使用这个完整基础作papers/ideas分析，再讨论goal4；这里不创建或启动目标4训练。", "",
             "| B split | zero方法-问题对 | 覆盖native winner样本数 | 占全部样本 | near-zero对 | Top1 | CE | actual regret |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("train", "val"):
        r = counts[split]
        lines.append(f"| {split} | {r['zero_method_count']} | {r['zero_winner_count']:,} / {r['instances']:,} | "
                     f"{r['zero_winner_pct']:.4f}% | {r['near_zero_method_count']} | {r['top1_pct']:.4f}% | "
                     f"{r['macro_ce']:.6f} | {r['macro_actual_regret_pct']:.6f}% |")
    lehd = lookup["val", "TSP", "LEHD"]
    mat = lookup["val", "ATSP", "MATPOENET"]
    lines.extend(("", f"TRAIN/VAL共同zero且真实获胜的方法对为{shared}个，任一split为zero的名称对并集为{len(zero_paired)}个。",
                  f"观察1：B的TSP/LEHD在VAL有{lehd['true_winner_count']}个native winner、模型选{lehd['selected_count']}次，"
                  f"条件平均概率{100*lehd['winner_probability_mean']:.4f}%、平均排名{lehd['winner_rank_mean']:.4f}，"
                  f"贡献本题{lehd['problem_costloss_share_pct']:.4f}%成本损失。零选择不等于没有概率质量。",
                  f"观察2：ATSP/MATPOENET在B VAL有{mat['true_winner_count']}个native winner、模型选{mat['selected_count']}次，"
                  f"条件平均概率{100*mat['winner_probability_mean']:.4f}%、平均排名{mat['winner_rank_mean']:.4f}，"
                  f"贡献本题{mat['problem_costloss_share_pct']:.4f}%成本损失。这只是标签条件观测，不指定修正规则。",
                  "`B_train_val_native_supervision.csv`按问题/native方法名与global ID对齐全部候选，含TRAIN/VAL真实winner与模型选择的分别计数、概率分位数、排名、native CE和成本贡献；两种频数不能互换。",
                  "`B_train_val_zero_native_names.csv`给出任一split剩余zero的方法名及两侧完整条件统计；`B_zero_methods.csv`保留两个split各自的完整zero行。native标签沿用保存winner，不使用FP32 argmin重标。"))
    for split in ("train", "val"):
        lines.extend(("", f"### B {split.upper()}：全部剩余zero真实winner方法", "",
                      "| 问题 | native方法名 | truewinner数/比例 | 模型选择数 | 条件平均P | 条件平均rank | 本题成本损失贡献 |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"))
        for r in zero:
            if r["split"] == split:
                lines.append(f"| {r['problem']} | {r['solver']} | {r['true_winner_count']:,} / {r['true_winner_pct']:.4f}% | "
                             f"{r['selected_count']} | {100*r['winner_probability_mean']:.4f}% | {r['winner_rank_mean']:.4f} | "
                             f"{r['problem_costloss_share_pct']:.4f}% |")
    return "\n".join(lines) + "\n"


def write_reports(output, results, readiness):
    combined = {name: [] for name in TABLES}
    for model, value in results.items():
        for split, tables in value["splits"].items():
            for name, rows in tables.items():
                tagged = tag_rows(rows, model, value["info"])
                combined[name].extend(tagged)
                directory = output / model
                directory.mkdir(exist_ok=True)
                base.write_rows(directory / f"{name}_{split}.csv", tagged)
    for name, rows in combined.items():
        base.write_rows(output / ("summary.csv" if name == "split_summary" else name + ".csv"), rows)
    methods = combined["method_summary"]
    for name, key in (("zero_methods", "zero_selected_with_winner"),
                      ("near_zero_methods", "near_zero_selected_with_winner")):
        base.write_rows(output / (name + ".csv"), [r for r in methods if r[key]], list(methods[0]))
    class_lookup = {(r["model"], r["split"], r["problem"], r["solver"]): r for r in combined["class_calibration"]}
    near_classes = [dict(r, **{key: class_lookup[r["model"], r["split"], r["problem"], r["solver"]][key]
                              for key in ("mean_probability", "true_winner_fraction", "classwise_ece20", "binary_brier")})
                    for r in methods if r["zero_selected_with_winner"] or r["near_zero_selected_with_winner"]]
    base.write_rows(output / "zero_or_near_zero_classes.csv", near_classes)
    summary = comparisons(combined["split_summary"], ("split",),
                          ("top1_pct", "macro_ce", "macro_actual_regret_pct", "macro_vs_sbs_pct",
                           "zero_method_count", "near_zero_method_count", "zero_winner_pct"))
    for row in summary:
        row["ce_reduction_vs_r76_pct"] = 100 * (row["r76_macro_ce"] - row["macro_ce"]) / row["r76_macro_ce"]
    base.write_rows(output / "comparison_vs_r76.csv", summary)
    method_comparison = comparisons(methods, ("split", "problem", "solver"),
                                    ("selected_pct", "true_winner_pct", "winner_probability_mean", "winner_rank_mean",
                                     "winner_ce_mean", "mean_costloss_contribution", "problem_costloss_share_pct",
                                     "relative_costloss_contribution_pp"))
    base.write_rows(output / "method_comparison_vs_r76.csv", method_comparison)
    problem_comparison = comparisons(combined["problem_summary"], ("split", "problem"),
                                     ("top1_pct", "ce", "mean_cost", "actual_regret_pct", "vs_sbs_pct",
                                      "zero_method_count", "zero_winner_pct"))
    base.write_rows(output / "problem_comparison_vs_r76.csv", problem_comparison)
    base.write_json(output / "source_readiness.json", dict(updated_utc=utc(), models=readiness,
                                                          complete=all(v["state"] == "ready" for v in readiness.values()),
                                                          r77_test_read=False))
    lines = ["# R78 研究输入：R77 / R76 TRAIN与VAL选择诊断", "",
             "接收方：Astra，仅用于 papers/ideas 与 goal4 方案讨论。本脚本不创建 goal、不做代码重审、不决定或启动新训练。",
             "**严格隔离：没有读取任何 R77 TEST 预测、TEST metrics、TEST结果或含TEST数字的status文件。** `final_test_lock.json` 仅作为VAL选权重的元数据；本文不含TEST数字。",
             "## 来源状态", "", "| 模型 | 状态 | 实际TRAIN/题 | VAL/题 | selected epoch / update |",
             "| --- | --- | ---: | ---: | --- |"]
    for model, info in readiness.items():
        point = f"{info['selected_epoch']} / {info.get('selected_update') or 'n/a'}" if info["state"] == "ready" else "未锁定，不读取临时best"
        lines.append(f"| {model} | {info['state']} | {info['train_rows_per_problem']:,} | 3,895 | {point} |")
    lines.extend(("", "## 核心数字", "",
                  "| 模型/split | 实例数 | Top1 | 宏CE | 宏actual regret | zero方法-问题对 | zero获胜比例 | near-zero对 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
    for r in combined["split_summary"]:
        lines.append(f"| {r['model']}/{r['split']} | {r['instances']:,} | {r['top1_pct']:.4f}% | {r['macro_ce']:.6f} | "
                     f"{r['macro_actual_regret_pct']:.6f}% | {r['zero_method_count']} | {r['zero_winner_pct']:.4f}% | {r['near_zero_method_count']} |")
    lines.extend(("", "## 口径与限制", "",
                  "A的TRAIN分母只有实际训练的10,000例/题，不重新推理其余60,114例；B和R76的TRAIN是70,114例/题。跨模型优先比较比率，而不是把winner样本数差当改进。VAL是相同的3,895例/题。",
                  "zero为selected_count=0且native truewinner>0；near-zero为0<selected_pct≤0.1%且truewinner>0，固定描述性阈值，不拟合规则。模型选择频数与真实winner频数分别保留。",
                  "复用R76 reader：最终maximin logits的masked FP32 softmax，温度1；排名/选法用原logits，并列按global solver ID。native标签不由FP32成本argmin替代。成本保留FP64。",
                  "条件概率/排名/CE在本方法是真实winner时计算；另有漏选条件统计及5/25/50/75/95分位数。Δcost=所选成本−native winner成本，mean_costloss_contribution=该方法漏选Δcost总和/本题N，costloss share分母为本题全部损失；不同题原始成本不相加。actual regret是所选相对FP64 Oracle的逐样本百分比均值。",
                  "ECE为固定20等宽bins，Brier为多类平方误差和；不拟合温度。R77与R76同时改变batch、update轴LR/预算；A还改变TRAIN样本和几何stats，不能从此表声称单一batch变量的因果改进。",
                  "实现角色：Solmax；same-family provisional，不声称独立外部审查。只有锁定/预算一致性与真实row映射两项输入guard，未运行额外tests、prep、GPU、training或profile。",
                  "## 完整文件", "",
                  "`zero_methods.csv`：完整零选择但真实获胜的方法-问题对表；`near_zero_methods.csv`与`zero_or_near_zero_classes.csv`：稀少选择类及class calibration。",
                  "`method_summary.csv`：全部方法的真实/选择频数、winner条件概率/排名/CE/成本贡献；`missed_candidate_summary.csv`与`missed_selection_frequencies.csv`：漏选候选概率及选法流向。",
                  "`summary.csv`、`problem_summary.csv`、`calibration_summary.csv`：整体与逐题CE/Top1/成本regret；`comparison_vs_r76.csv`、`method_comparison_vs_r76.csv`、`problem_comparison_vs_r76.csv`：与oldbs9856的同口径并列表及差值。",
                  "`source_readiness.json`与`protocol.json`：准确源路径、checkpoint与输入hash。各模型独立目录保留TRAIN/VAL原表；R76历史输出只读，未覆盖。"))
    text = "\n".join(lines) + "\n" + native_b_handoff(output, combined)
    (output / "README.md").write_text(text, encoding="utf-8")
    (output / "RESEARCH_HANDOFF.md").write_text(text, encoding="utf-8")
    print("Snapshot: " + "; ".join(f"{r['model']}/{r['split']}: zero={r['zero_method_count']}, "
                                  f"Top1={r['top1_pct']:.4f}%, CE={r['macro_ce']:.6f}" for r in combined["split_summary"]), flush=True)


def refresh_handoff(root, output):
    state = base.read_json(output / "source_readiness.json")
    readiness, results = state["models"], {}
    tags = {"model", "logical_batch_size", "selected_epoch", "selected_update", "checkpoint_sha256"}
    for model, info in readiness.items():
        if info["state"] != "ready":
            continue
        splits = {}
        for split in ("train", "val"):
            splits[split] = {name: [{k: v for k, v in row.items() if k not in tags}
                                   for row in read_csv(output / model / f"{name}_{split}.csv")]
                             for name in TABLES}
        results[model] = dict(info=info, splits=splits)
    write_reports(output, results, readiness)
    manifest_path = root / "selected_train_val_predictions/manifest.json"
    manifest = base.read_json(manifest_path)
    reference = dict(path=str(manifest_path), complete=manifest["complete"], input_splits=manifest["input_splits"],
                     test_inputs_read=manifest["test_inputs_read"], arms={arm: dict(
                         ready=value["ready"], selected_update=value["selected_update"], epoch=value["evaluation_point"],
                         checkpoint_sha256=value["checkpoint_sha256"],
                         files={split: value["files"][split] for split in ("train", "val")})
                         for arm, value in manifest["arms"].items()})
    base.write_json(output / "final_train_val_manifest_reference.json", reference)
    note = ("\n## 最终manifest\n\n"
            f"源manifest：`{manifest_path}`，complete=true、input_splits=train/val。"
            "`final_train_val_manifest_reference.json`保留两臂确切TRAIN/VAL路径、每题行数及锁定身份。"
            "本次交接刷新仅读取已有CSV与manifest，没有重新读NPZ、核验旧R76标签/CE或执行推理/tests/modelaudit。\n")
    for filename in ("README.md", "RESEARCH_HANDOFF.md"):
        path = output / filename
        path.write_text(path.read_text(encoding="utf-8") + note, encoding="utf-8")
    protocol = base.read_json(output / "protocol.json")
    protocol["handoff_refresh"] = dict(updated_utc=utc(), script_sha256=base.file_hash(Path(__file__)),
                                       input="existing generated TRAIN/VAL CSVs and final TRAIN/VAL manifest only",
                                       r77_test_read=False, new_forward_passes=0, extra_tests=0, model_audits=0,
                                       all18_ready=state["complete"], manifest_reference=reference)
    base.write_json(output / "protocol.json", protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    parser.add_argument("--r76-root", type=Path, default=base.RUN_ROOT)
    parser.add_argument("--wait", action="store_true", help="Wait on final TRAIN/VAL files only, no GPU/job control")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--refresh-handoff", action="store_true", help="Use existing CSVs/manifest only; no NPZ or model reads")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("Poll interval must be at least five seconds")
    root, old_root = args.run_root.resolve(), args.r76_root.resolve()
    output = root / "train_val_selection_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    if args.refresh_handoff:
        refresh_handoff(root, output)
        return
    baseline = load_baseline(old_root)
    preserved = {str(path): base.file_hash(path) for path in baseline["sources"]}
    results = {BASELINE: baseline}
    while True:
        changed = not (output / "source_readiness.json").exists()
        readiness = {BASELINE: baseline["info"]}
        for model, (_, size) in ARMS.items():
            if model not in results:
                locked = locked_sources(root, model)
                if locked:
                    if locked["info"]["data_root"] != baseline["info"]["data_root"]:
                        raise ValueError("R76/R77 must use the same new-data release")
                    results[model] = load_arm(root, model, locked)
                    changed = True
            readiness[model] = results[model]["info"] if model in results else dict(
                state="waiting_for_final_train_val", train_rows_per_problem=size, val_rows_per_problem=3895)
        if changed:
            write_reports(output, results, readiness)
            if any(base.file_hash(Path(path)) != digest for path, digest in preserved.items()):
                raise RuntimeError("Existing R76 TRAIN/VAL source tables changed during analysis")
            protocol = dict(created_utc=utc(), complete=len(results) == 3, r77_test_read=False,
                            purpose="Astra-only TRAIN/VAL evidence for R78/goal4 discussion; not a training decision",
                            implementation_role="Solmax; same-family provisional", new_forward_passes=0,
                            training_runs=0, gpu_used=False, extra_tests=0, profiles=0,
                            near_zero_selected_pct=NEAR_ZERO_PCT, calibration_bins=20,
                            script_sha256=base.file_hash(Path(__file__)),
                            reused_reader_sha256=base.file_hash(Path(base.__file__)),
                            baseline_sources_unchanged=True,
                            sources={model: [dict(path=str(path), sha256=base.file_hash(path)) for path in value["sources"]]
                                     for model, value in results.items()})
            base.write_json(output / "protocol.json", protocol)
        if len(results) == 3 or not args.wait:
            print(f"Source state: {[ (model, info['state']) for model, info in readiness.items() ]}", flush=True)
            return
        print(f"{utc()} Waiting for locked B TRAIN/VAL; no TEST reads or inference", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
