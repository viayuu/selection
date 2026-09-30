"""R38 cost attribution and frozen-versus-full geometry probe comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS
from .geometry_probe import METRICS, PROB_METRICS, ROOT as R37, metrics_from_logits
from .multitask_probe import dump, state_hash
from .report_geometry_probe import load_history, verify
from .tsp_learnability import BASE, GAP_LABELS, gap_and_regret, raw_tsp, size_labels, write_csv


ROOT = Path("code/V4/runs/R38_tsp_frozen_geometry")
SIZE_RULES = Path("code/V4/runs/R36_tsp_learnability/size_prior_rules.json")
CATEGORIES = ("corrected", "harmed", "wrong_switched", "wrong_unchanged", "correct_unchanged")
LABELS = dict(corrected="原错→新对", harmed="原对→新错", wrong_switched="两者都错，换方法",
              wrong_unchanged="两者都错，未换方法", correct_unchanged="两者都对，未换方法")


def read_prediction(path, raw):
    with np.load(path) as data:
        result = {k: data[k].copy() for k in data.files}
    np.testing.assert_array_equal(result["indices"], np.arange(len(raw["winner"])))
    np.testing.assert_array_equal(result["winner"], raw["winner"])
    np.testing.assert_array_equal(result["pool_ids"], [GLOBAL_SOLVERS.index(s) for s in POOLS["TSP"]])
    np.testing.assert_array_equal(result["pred"], result["logits"].argmax(1))
    assert np.isfinite(result["logits"]).all()
    return result


def probabilities(logits):
    shifted = logits.astype(np.float64) - logits.max(1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(1, keepdims=True)


def decision_changes(base, new, raw, cuts, seed):
    costs, winner = raw["costs"], raw["winner"]
    old, pred = base["pred"], new["pred"]
    n = len(pred)
    index = np.arange(n)
    gap, old_regret = gap_and_regret(costs, old)
    _, new_regret = gap_and_regret(costs, pred)
    delta_cost = costs[index, pred] - costs[index, old]
    delta_regret = new_regret - old_regret
    old_correct, new_correct = old == winner, pred == winner
    masks = dict(corrected=~old_correct & new_correct, harmed=old_correct & ~new_correct,
                 wrong_switched=~old_correct & ~new_correct & (old != pred),
                 wrong_unchanged=~old_correct & ~new_correct & (old == pred),
                 correct_unchanged=old_correct & new_correct)
    np.testing.assert_array_equal(sum(m.astype(int) for m in masks.values()), np.ones(n))
    gap_groups = np.searchsorted([0, .01, .1, .5], gap, side="left")
    size_groups = np.searchsorted(cuts, raw["nodes"], side="left")
    buckets = [("ALL", "ALL", np.ones(n, dtype=bool))]
    buckets.extend(("winner_gap", label, gap_groups == g) for g, label in enumerate(GAP_LABELS))
    buckets.extend(("size", label, size_groups == g) for g, label in enumerate(size_labels(cuts)))
    old_prob, new_prob = probabilities(base["logits"]), probabilities(new["logits"])
    cost_regret = (costs - costs.min(1)[:, None]) / costs.min(1)[:, None] * 100
    expected = [(p * cost_regret).sum(1) for p in (old_prob, new_prob)]
    rows = []
    for dimension, label, bucket in buckets:
        for category, selection in masks.items():
            mask = selection & bucket
            count = int(mask.sum())
            row = dict(seed=seed, dimension=dimension, bucket=label, category=category, n=count,
                       net_correct=int((new_correct[mask].astype(int) - old_correct[mask]).sum()),
                       sum_cost_change=float(delta_cost[mask].sum()),
                       contribution_mean_cost=float(delta_cost[mask].sum() / n),
                       sum_actual_regret_change_pct=float(delta_regret[mask].sum()),
                       contribution_actual_regret_pct=float(delta_regret[mask].sum() / n),
                       sum_expected_regret_change_pct=float((expected[1][mask] - expected[0][mask]).sum()),
                       lower_cost_count=int((delta_cost[mask] < 0).sum()),
                       higher_cost_count=int((delta_cost[mask] > 0).sum()))
            for prefix, p, choice in (("base", old_prob, old), ("new", new_prob, pred)):
                row[prefix + "_mean_winner_probability"] = float(p[index[mask], winner[mask]].mean()) if count else None
                row[prefix + "_mean_chosen_probability"] = float(p[index[mask], choice[mask]].mean()) if count else None
            rows.append(row)
    transitions = []
    for before, after in sorted(set(zip(old.tolist(), pred.tolist()))):
        if before == after:
            continue
        mask = (old == before) & (pred == after)
        transitions.append(dict(seed=seed, old_solver=POOLS["TSP"][before], new_solver=POOLS["TSP"][after],
                                n=int(mask.sum()), net_correct=int((new_correct[mask].astype(int) - old_correct[mask]).sum()),
                                sum_cost_change=float(delta_cost[mask].sum()),
                                contribution_mean_cost=float(delta_cost[mask].sum() / n),
                                contribution_actual_regret_pct=float(delta_regret[mask].sum() / n)))
    total = [r for r in rows if r["dimension"] == "ALL"]
    np.testing.assert_allclose(sum(r["contribution_mean_cost"] for r in total), delta_cost.mean(), rtol=0, atol=1e-14)
    np.testing.assert_allclose(sum(r["contribution_actual_regret_pct"] for r in total), delta_regret.mean(), rtol=0, atol=1e-12)
    assert sum(r["net_correct"] for r in total) == int(new_correct.sum() - old_correct.sum())
    codes = np.empty(n, dtype="<U20")
    for name, mask in masks.items():
        codes[mask] = name
    instances = dict(indices=index, old=old, new=pred, native_winner=winner, nodes=raw["nodes"],
                     category=codes, winner_gap_pct=gap, gap_bucket=gap_groups, size_bucket=size_groups,
                     cost_change=delta_cost, actual_regret_change_pct=delta_regret,
                     base_winner_probability=old_prob[index, winner], new_winner_probability=new_prob[index, winner],
                     base_chosen_probability=old_prob[index, old], new_chosen_probability=new_prob[index, pred],
                     base_expected_regret_pct=expected[0], new_expected_regret_pct=expected[1])
    return rows, transitions, instances


def attribute_existing(root, raw):
    cuts = np.array(json.loads(SIZE_RULES.read_text())["boundaries"])
    base = read_prediction(R37 / "baseline_val.npz", raw)
    results = json.loads((R37 / "results.json").read_text())
    rows, transitions, summaries = [], [], []
    for result in results:
        if result["mode"] != "real":
            continue
        new = read_prediction(R37 / result["name"] / "predictions_val_u1000.npz", raw)
        changes, pairs, instances = decision_changes(base, new, raw, cuts, result["seed"])
        rows.extend(changes)
        transitions.extend(pairs)
        np.savez_compressed(root / f"R37B_decision_changes_seed{result['seed']}.npz", **instances)
        for name, prediction in (("R34", base), ("R37B", new)):
            m, _ = metrics_from_logits(torch.from_numpy(prediction["logits"]), raw)
            summaries.append(dict(model=name, seed=result["seed"], **{k: m[k] for k in METRICS + PROB_METRICS}))
    write_csv(root / "R37_decision_changes.csv", rows)
    write_csv(root / "R37_solver_transitions.csv", transitions)
    write_csv(root / "R37_probability_costs.csv", summaries)
    dump(root / "size_boundaries.json", dict(boundaries=cuts.tolist(), source=str(SIZE_RULES),
                                           gap_labels=list(GAP_LABELS), test_read=False))
    text = ["# R37B 相对 R34：验证集决策成本归因", "",
            "仅复用保存的验证预测；成本直接从 raw_label.pkl 读取为 FP64。严格正确按 native ind。",
            "以下为三个续训 seed 的均值，数量可以是小数。Δcost=新选择成本−R34选择成本，负数表示节省。",
            "regret 改变量按每个实例各自的 oracle 成本归一化；不是由平均 cost 反推。", "",
            "| 决策变化 | 数量 | 净正确数 | 成本变化合计 | 对整体 mean_cost 的贡献 | 对整体 actual regret 的贡献(pp) |",
            "|---|---:|---:|---:|---:|---:|"]
    aggregate = []
    for category in CATEGORIES:
        selected = [r for r in rows if r["dimension"] == "ALL" and r["category"] == category]
        mean = {k: float(np.mean([r[k] for r in selected])) for k in
                ("n", "net_correct", "sum_cost_change", "contribution_mean_cost", "contribution_actual_regret_pct")}
        aggregate.append(dict(category=category, **mean))
        text.append(f"| {LABELS[category]} | {mean['n']:.2f} | {mean['net_correct']:+.2f} | {mean['sum_cost_change']:+.6f} | "
                    f"{mean['contribution_mean_cost']:+.6f} | {mean['contribution_actual_regret_pct']:+.6f} |")
    net_correct = sum(r["net_correct"] for r in aggregate)
    cost_change = sum(r["contribution_mean_cost"] for r in aggregate)
    regret_change = sum(r["contribution_actual_regret_pct"] for r in aggregate)
    text.extend(["", f"核算合计：净增加正确选择 {net_correct:.2f} 例，mean_cost 变化 {cost_change:+.9f}，actual regret 变化 {regret_change:+.9f} pp。",
                 "这同时计入所有纠错、改错、两者都错的切换，不把小 winner gap 直接等同于小切换代价。", "",
                 "## 概率分配与硬选择", "", "| 模型 | CE | argmax actual regret | softmax expected regret | mean_cost | softmax expected cost |",
                 "|---|---:|---:|---:|---:|---:|"])
    for name in ("R34", "R37B"):
        values = [s for s in summaries if s["model"] == name]
        mean = {k: np.mean([s[k] for s in values]) for k in ("ce", "actual_regret_pct", "expected_regret_pct", "mean_cost", "expected_mean_cost")}
        text.append(f"| {name} | {mean['ce']:.6f} | {mean['actual_regret_pct']:.6f}% | {mean['expected_regret_pct']:.6f}% | "
                    f"{mean['mean_cost']:.6f} | {mean['expected_mean_cost']:.6f} |")
    corrected, harmed = aggregate[:2]
    saved = -corrected["sum_cost_change"]
    added = harmed["sum_cost_change"]
    text.extend(["", f"纠错节省成本合计 {saved:.6f}，新改错增加 {added:.6f}。"
                 f"每例纠错平均节省 {saved / corrected['n']:.6f}，每例改错平均增加 {added / harmed['n']:.6f}。"
                 "因此，较少的新错误也可以抵消较多的纠错收益；这里应比较实际切换代价，而不是只看 winner gap 或净正确数。",
                 "本轮期望成本/regret下降而硬选择成本/regret未降，支持检查概率分配到argmax决策的目标差异；不能据此认定risk权重太小，也不能直接认定错误越来越自信。", ""])
    text.append("期望regret诊断使用原始FP64成本且不截断；训练risk仍按R34用FP32成本，并保留逐实例clamp(max=1)。二者的数值口径不完全相同，未改训练目标。")
    text.extend(["", "## 原方法→新方法：成本增加最大的切换", "",
                 "按跨 seed 的整体 mean_cost 贡献排序；这是原始验证实例上的描述性核算，不是新增独立样本。", "",
                 "| 原方法 | 新方法 | 平均切换数 | 净正确数 | Δmean_cost | Δactual regret(pp) |",
                 "|---|---|---:|---:|---:|---:|"])
    pairs = []
    for before, after in set((p["old_solver"], p["new_solver"]) for p in transitions):
        selected = [p for p in transitions if (p["old_solver"], p["new_solver"]) == (before, after)]
        row = dict(old_solver=before, new_solver=after,
                   **{k: sum(r[k] for r in selected) / 3 for k in
                      ("n", "net_correct", "contribution_mean_cost", "contribution_actual_regret_pct")})
        pairs.append(row)
    pairs.sort(key=lambda p: p["contribution_mean_cost"], reverse=True)
    write_csv(root / "R37_solver_transitions_mean.csv", pairs)
    for p in pairs[:10]:
        text.append(f"| {p['old_solver']} | {p['new_solver']} | {p['n']:.2f} | {p['net_correct']:+.2f} | "
                    f"{p['contribution_mean_cost']:+.6f} | {p['contribution_actual_regret_pct']:+.6f} |")
    for dimension in ("winner_gap", "size"):
        text.extend(["", f"## R36 固定分桶：{dimension}", "",
                     "| 桶 | 纠正数 | 新改错数 | 两错切换数 | 净正确数 | Δmean_cost整体贡献 | Δactual regret整体贡献(pp) |",
                     "|---|---:|---:|---:|---:|---:|---:|"])
        for bucket in dict.fromkeys(r["bucket"] for r in rows if r["dimension"] == dimension):
            selected = [r for r in rows if r["dimension"] == dimension and r["bucket"] == bucket]
            counts = {c: sum(r["n"] for r in selected if r["category"] == c) / 3 for c in CATEGORIES}
            delta = {k: sum(r[k] for r in selected) / 3 for k in
                     ("net_correct", "contribution_mean_cost", "contribution_actual_regret_pct")}
            text.append(f"| {bucket} | {counts['corrected']:.2f} | {counts['harmed']:.2f} | {counts['wrong_switched']:.2f} | "
                        f"{delta['net_correct']:+.2f} | {delta['contribution_mean_cost']:+.6f} | {delta['contribution_actual_regret_pct']:+.6f} |")
    text.extend(["", "每桶的成本贡献都除以完整验证集1000例，故互斥桶的贡献可以相加。逐 seed、逐类别的概率统计见 R37_decision_changes.csv；逐实例完整结果见 R37B_decision_changes_seed*.npz。", ""])
    (root / "R37_cost_attribution.md").write_text("\n".join(text))
    dump(root / "R37_cost_attribution.json", dict(categories=aggregate, net_correct=net_correct,
                                                mean_cost_change=cost_change, actual_regret_change_pct=regret_change,
                                                total_accounting_verified=True, test_read=False))
    print(f"[R37 attribution] net_correct={net_correct:.2f} mean_cost_delta={cost_change:+.9f} regret_delta={regret_change:+.9f} pp", flush=True)


def histories(root, results, raw):
    enriched = {}
    for result in results:
        rows = load_history(root, result)
        for row in rows:
            for split in ("train", "val"):
                saved = read_prediction(root / result["name"] / f"predictions_{split}_u{row['updates']:04d}.npz", raw[split])
                metrics, _ = metrics_from_logits(torch.from_numpy(saved["logits"]), raw[split])
                row[split].update({k: metrics[k] for k in PROB_METRICS})
        enriched[result["name"]] = rows
    return enriched


def bootstrap(root, results, raw):
    costs, winner = raw["costs"], raw["winner"]
    def outcomes(prediction):
        pred = prediction["pred"]
        probability = probabilities(prediction["logits"])
        regret_matrix = (costs - costs.min(1)[:, None]) / costs.min(1)[:, None] * 100
        return np.stack([(pred == winner).astype(float), costs[np.arange(len(pred)), pred],
                         gap_and_regret(costs, pred)[1], (probability * regret_matrix).sum(1)], 1)
    values = {}
    for label, directory, models in (("R38A", root, [r for r in results if r["mode"] == "zero"]),
                                     ("R38B", root, [r for r in results if r["mode"] == "real"]),
                                     ("R37B", R37, [r for r in json.loads((R37 / "results.json").read_text()) if r["mode"] == "real"])):
        values[label] = np.mean([outcomes(read_prediction(directory / r["name"] / "predictions_val_u1000.npz", raw)) for r in models], 0)
    values["R34"] = outcomes(read_prediction(root / "baseline_val.npz", raw))
    indices = np.random.default_rng(38).integers(0, len(winner), size=(5000, len(winner)))
    rows = []
    for reference in ("R38A", "R34", "R37B"):
        delta = values["R38B"] - values[reference]
        low, high = np.quantile(delta[indices].mean(1), [.025, .975], axis=0)
        for j, metric in enumerate(("top1", "mean_cost", "actual_regret_pct", "expected_regret_pct")):
            rows.append(dict(comparison=f"R38B - {reference}", metric=metric,
                             mean_difference=float(delta[:, j].mean()), ci95_low=float(low[j]), ci95_high=float(high[j])))
    write_csv(root / "paired_bootstrap.csv", rows)
    return rows


def compare(root, raw):
    results = json.loads((root / "results.json").read_text())
    reference = json.loads((R37 / "results.json").read_text())
    verify(root, results, raw)
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    base_hash = state_hash(base["model"])
    frozen_checks = []
    for result in results:
        directory = root / result["name"]
        config = json.loads((directory / "args.json").read_text())
        checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        original = {k: v for k, v in checkpoint["model"].items() if "geometry_residual." not in k}
        assert config["freeze_base"] and state_hash(original) == base_hash == config["original_state_initial_hash"]
        assert all(row["original_state_hash"] == base_hash for row in load_history(root, result))
        assert len(checkpoint["optimizer"]["param_groups"]) == 1
        assert checkpoint["optimizer"]["param_groups"][0]["name"] == "geometry"
        assert len(checkpoint["optimizer"]["state"]) == 4
        full_config = json.loads((R37 / result["name"] / "args.json").read_text())
        for key in ("model_initial_hash", "new_branch_initial_hash", "batch_hash", "dropout_rng_initial_hash", "geometry_stats", "loss"):
            assert config[key] == full_config[key], key
        frozen_trace = [json.loads(line) for line in (directory / "update_trace.jsonl").read_text().splitlines()]
        full_trace = [json.loads(line) for line in (R37 / result["name"] / "update_trace.jsonl").read_text().splitlines()]
        for a, b in zip(frozen_trace, full_trace):
            assert all(a[k] == b[k] for k in ("updates", "batch_hash", "dropout_rng_before", "dropout_rng_after", "lr_factor"))
        assert len(frozen_trace) == len(full_trace) == 1000
        frozen_checks.append(dict(run=result["name"], original_parameters_and_buffers_bitwise_unchanged=True,
                                  only_geometry_in_optimizer=True, paired_with_R37_batch_dropout=True))
    dump(root / "frozen_verification.json", dict(checks=frozen_checks, test_read=False))
    own = histories(root, results, raw)
    old = histories(R37, reference, raw)
    baseline = json.loads((root / "baseline.json").read_text())
    intervals = bootstrap(root, results, raw["val"])
    cuts = np.array(json.loads(SIZE_RULES.read_text())["boundaries"])
    base_prediction = read_prediction(root / "baseline_val.npz", raw["val"])
    decision_rows, transition_rows = [], []
    for result in results:
        if result["mode"] == "real":
            prediction = read_prediction(root / result["name"] / "predictions_val_u1000.npz", raw["val"])
            changes, transitions, instances = decision_changes(base_prediction, prediction, raw["val"], cuts, result["seed"])
            decision_rows.extend(changes)
            transition_rows.extend(transitions)
            np.savez_compressed(root / f"R38B_decision_changes_seed{result['seed']}.npz", **instances)
    write_csv(root / "R38_decision_changes.csv", decision_rows)
    write_csv(root / "R38_solver_transitions.csv", transition_rows)
    metrics = METRICS + PROB_METRICS
    rows = []
    for study, models, all_history in (("R38", results, own), ("R37", reference, old)):
        for result in models:
            history = all_history[result["name"]]
            for point, selection in (("final", history[-1:]), ("last5", history[-5:])):
                for split in ("train", "val"):
                    rows.append(dict(model=study + ("A" if result["mode"] == "zero" else "B"), seed=result["seed"],
                                     summary=point, split=split,
                                     **{k: float(np.mean([r[split][k] for r in selection])) for k in metrics}))
    write_csv(root / "comparison.csv", rows)
    differences = []
    for point in ("final", "last5"):
        for left, right in (("R38B", "R38A"), ("R38B", "R34"), ("R38B", "R37B"), ("R38A", "R37A")):
            for seed in (2, 3, 4):
                a = next(r for r in rows if r["model"] == left and r["seed"] == seed and r["summary"] == point and r["split"] == "val")
                b = baseline["val"] if right == "R34" else next(
                    r for r in rows if r["model"] == right and r["seed"] == seed and r["summary"] == point and r["split"] == "val")
                differences.append(dict(comparison=left + " - " + right, seed=seed, summary=point,
                                        **{k: a[k] - b[k] for k in metrics}))
    write_csv(root / "paired_differences.csv", differences)
    text = ["# R38：冻结原模型的 TSP 几何残差对照", "",
            "R34 仍为18任务主基线；本轮只做 TSP train/val，未读取 test，不从 R37 最终权重继续。",
            "先完成 [R37 成本归因](R37_cost_attribution.md)，再从同一 R34 第50轮 best.pt 跑 R38A/B × seed2/3/4。",
            "R38A 的12维几何标准化后置零；R38B 使用真实几何；原参数全部冻结，仅21→64→128残差 MLP更新。",
            "原网络训练前向未使用 no_grad，保持 train() 的原 dropout0.1，梯度穿过冻结网络回传新分支。eval() 仅用于评估。",
            "其余与 R37 一致：fresh AdamW、分支LR2e-4、WD1e-4、batch640、1000次成功更新、warmup50后cosine至10%、AMP fp16/SDPA、clip1、winner-cost/native ind、关闭增强、自然分布采样。",
            "每50次评估完整 train/val；采用1000步和最后五点（800/850/900/950/1000），不是挑最高点。",
            "同一起点的续训重复，不是三次从头训练。A/B 以及同 seed 同输入的 R37/R38，批次与 dropout随机流逐更新核验一致。", ""]
    for point in ("final", "last5"):
        text.extend([f"## {point}：三 seed 均值", "",
                     "| 模型 | split | CE | Top1 | mean_cost | argmax actual regret | softmax expected regret | gap>0.1 Top1 | gap>0.5 Top1 |",
                     "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for split in ("train", "val"):
            for name in ("R34", "R37A", "R37B", "R38A", "R38B"):
                if name == "R34":
                    mean = baseline[split]
                else:
                    selected = [r for r in rows if r["model"] == name and r["summary"] == point and r["split"] == split]
                    mean = {k: np.mean([r[k] for r in selected]) for k in metrics}
                text.append(f"| {name} | {split} | {mean['ce']:.6f} | {mean['top1']:.2%} | {mean['mean_cost']:.6f} | "
                            f"{mean['actual_regret_pct']:.6f}% | {mean['expected_regret_pct']:.6f}% | "
                            f"{mean['gap_gt_0_1_top1']:.2%} | {mean['gap_gt_0_5_top1']:.2%} |")
    text.extend(["", "## 各 seed 验证结果", "", "| 模型 | seed | summary | Top1 | mean_cost | actual regret | expected regret |",
                 "|---|---:|---|---:|---:|---:|---:|"])
    for row in rows:
        if row["split"] == "val" and row["model"].startswith("R38"):
            text.append(f"| {row['model']} | {row['seed']} | {row['summary']} | {row['top1']:.2%} | {row['mean_cost']:.6f} | "
                        f"{row['actual_regret_pct']:.6f}% | {row['expected_regret_pct']:.6f}% |")
    text.extend(["", "## 成对验证差值：三 seed 均值", "",
                 "Top1单位为百分点，成本/regret越负越好；R38B−R37B直接比较只训练分支与全参数训练。", "",
                 "| 对照 | summary | ΔTop1(pp) | Δmean_cost | Δactual regret(pp) | Δexpected regret(pp) |",
                 "|---|---|---:|---:|---:|---:|"])
    for point in ("final", "last5"):
        for name in dict.fromkeys(r["comparison"] for r in differences):
            selected = [r for r in differences if r["comparison"] == name and r["summary"] == point]
            d = {k: np.mean([r[k] for r in selected]) for k in metrics}
            text.append(f"| {name} | {point} | {100*d['top1']:+.3f} | {d['mean_cost']:+.6f} | "
                        f"{d['actual_regret_pct']:+.6f} | {d['expected_regret_pct']:+.6f} |")
    text.extend(["", "## R38B−R34 的最终决策成本分解", "",
                 "三 seed 均值；贡献均除以完整验证集1000例，负成本差表示节省。", "",
                 "| 决策变化 | 数量 | 净正确数 | Δmean_cost贡献 | Δactual regret贡献(pp) |",
                 "|---|---:|---:|---:|---:|"])
    for category in CATEGORIES:
        selected = [r for r in decision_rows if r["dimension"] == "ALL" and r["category"] == category]
        d = {k: np.mean([r[k] for r in selected]) for k in
             ("n", "net_correct", "contribution_mean_cost", "contribution_actual_regret_pct")}
        text.append(f"| {LABELS[category]} | {d['n']:.2f} | {d['net_correct']:+.2f} | "
                    f"{d['contribution_mean_cost']:+.6f} | {d['contribution_actual_regret_pct']:+.6f} |")
    text.extend(["", "## R36 高损失区域：Top1 与成本一起看", "",
                 "下面两个累计区域相互重叠，不能将样本数或损失贡献相加；分桶仍由原始FP64成本固定。", "",
                 "| 模型 | summary | gap>0.1% Top1 (763例) | gap>0.1% actual regret | gap>0.5% Top1 (289例) | gap>0.5% actual regret |",
                 "|---|---|---:|---:|---:|---:|"])
    for point in ("final", "last5"):
        for name in ("R34", "R37B", "R38B"):
            selected = [r for r in rows if r["model"] == name and r["summary"] == point and r["split"] == "val"]
            d = baseline["val"] if name == "R34" else {k: np.mean([r[k] for r in selected]) for k in metrics}
            text.append(f"| {name} | {point} | {d['gap_gt_0_1_top1']:.2%} | {d['gap_gt_0_1_actual_regret_pct']:.6f}% | "
                        f"{d['gap_gt_0_5_top1']:.2%} | {d['gap_gt_0_5_actual_regret_pct']:.6f}% |")
    text.extend(["", "## 第1000步：成对实例 bootstrap", "",
                 "重采样共同1000个验证实例5000次，保留同一实例上的三个 seed；不把3000个预测当独立样本。只描述本验证集实例抽样不确定性，不覆盖新seed或新分布。",
                 "Top1 单位为百分点，regret单位也为百分点；cost为原始单位。负成本差表示改善。", "",
                 "| 对照 | 指标 | 均值差 | 95%区间 |", "|---|---|---:|---|"])
    for interval in intervals:
        scale = 100 if interval["metric"] == "top1" else 1
        text.append(f"| {interval['comparison']} | {interval['metric']} | {scale*interval['mean_difference']:+.6f} | "
                    f"[{scale*interval['ci95_low']:+.6f}, {scale*interval['ci95_high']:+.6f}] |")
    cost_ci = next(r for r in intervals if r["comparison"] == "R38B - R34" and r["metric"] == "mean_cost")
    regret_ci = next(r for r in intervals if r["comparison"] == "R38B - R34" and r["metric"] == "actual_regret_pct")
    all_seeds_safe = all(r["mean_cost"] < baseline["val"]["mean_cost"] and r["actual_regret_pct"] < baseline["val"]["actual_regret_pct"]
                         for r in rows if r["model"] == "R38B" and r["split"] == "val")
    supported = cost_ci["ci95_high"] < 0 and regret_ci["ci95_high"] < 0 and all_seeds_safe
    text.extend(["", "## 判断", "",
                 "R38B 相对 R34 的成本/regret改善通过本轮严格检验，可考虑只将此修改带回18任务受控复验。" if supported else
                 "尚未同时满足‘R38B 每seed的最终点和末五点成本/regret均低于R34，且最终成本/regret差的95%区间上界小于0’。不能宣布可信超越R34，不立即扩展18任务。",
                 "冻结对照区分了原参数更新的影响，但不能自动把所有剩余误差归因于几何不足。softmax期望regret和argmax实际regret分别报告，不能相互替代。", "",
                 "期望regret诊断由原始FP64成本计算且不截断；训练risk沿用R34的FP32、clamp(max=1)与cost_scale，未更改监督目标。", "",
                 "## 核验与文件", "",
                 "frozen_verification.json：原参数及buffer前后逐位一致，optimizer只含新分支，R37/R38的初始权重、批次、dropout和LR计划对齐。",
                 "prediction_replay.json：全部评估重算和AMP成功更新/Adam步数核验。geometry_stats.json复用R37训练节点的统计，保存为checkpoint buffer。",
                 "每实验目录有 args.json、train.log、history.json、last.pt、batch_indices.pt、update_trace.jsonl 和逐实例预测；W&B离线记录。", ""])
    summary = {name: {k: float(np.mean([r[k] for r in rows if r["model"] == name and r["summary"] == "final" and r["split"] == "val"]))
                      for k in metrics} for name in ("R37A", "R37B", "R38A", "R38B")}
    old_harm = next(r for r in json.loads((root / "R37_cost_attribution.json").read_text())["categories"] if r["category"] == "harmed")
    new_harm = [r for r in decision_rows if r["dimension"] == "ALL" and r["category"] == "harmed"]
    harm_count = float(np.mean([r["n"] for r in new_harm]))
    interpretation = ["### 具体解读", "",
                      f"1. 几何增量仍在：R38B相对冻结冗余输入R38A，最终Top1提高{100*(summary['R38B']['top1']-summary['R38A']['top1']):.3f}个百分点。相对R34的Top1区间为正，但这只是本验证集上的成对实例bootstrap证据。",
                      f"2. 冻结改善了验证CE：R38B为{summary['R38B']['ce']:.6f}，R37B为{summary['R37B']['ce']:.6f}，R34为{baseline['val']['ce']:.6f}。R38A也没有复现R37A明显的验证退化。支持减少整体续训造成的泛化损害，而非否定几何特征。",
                      f"3. 新改错数量由R37B平均{old_harm['n']:.2f}例降为R38B的{harm_count:.2f}例；但这些剩余错误仍有较高代价。冻结减少破坏，不等于已解决成本表现。",
                      f"4. 期望与硬选择没有同步：R38B期望regret为{summary['R38B']['expected_regret_pct']:.6f}%，低于R34的{baseline['val']['expected_regret_pct']:.6f}%，但高于R37B的{summary['R37B']['expected_regret_pct']:.6f}%。更强的期望成本优化不自动产生更好的argmax成本，不能只看risk曲线或直接放大risk权重。",
                      "5. R38完成的是冻结对照，不是18任务的新主模型。成本区间仍包含0，继续保留R34；后续优先检查剩余高代价新错误的原方法→新方法和训练样本覆盖，不立即增加几何字段或扩展任务。",
                      f"6. 最大gap区域仍未解决：gap>0.5%的最终actual regret从R34的{baseline['val']['gap_gt_0_5_actual_regret_pct']:.6f}%变为R38B的{summary['R38B']['gap_gt_0_5_actual_regret_pct']:.6f}%，即使该区域Top1上升，也不能宣称其选择代价改善。这里同样是点估计，不能把小幅正差直接当作确定退步。",
                      "本轮没有读取test。反复使用同一验证集的诊断及模型设计过程不由上述bootstrap区间覆盖；不能将此区间当作独立测试收益保证。", ""]
    position = text.index("## 核验与文件")
    text[position:position] = interpretation
    text.extend(["checkpoint_replay.json：最终checkpoint的GPU重放及在线/缓存几何计算一致性检查。",
                 "source_before.tar.gz是修改前备份，source.tar.gz是启动时源码，source_final.tar.gz含最终分析脚本。", ""])
    plot(root, own, old, results, reference, baseline)
    for metric in ("top1", "ce", "mean_cost", "actual_regret_pct", "expected_regret_pct", "gap_gt_0_5_top1"):
        text.extend([f"![{metric}](comparison_{metric}.png)", ""])
    (root / "comparison.md").write_text("\n".join(text))
    top1_ci = next(r for r in intervals if r["comparison"] == "R38B - R34" and r["metric"] == "top1")
    dump(root / "conclusion.json", dict(credible_cost_regret_improvement_vs_R34=supported,
                                       all_seeds_final_last5_cost_regret_better_than_R34=all_seeds_safe,
                                       top1_ci_excludes_zero_vs_R34=top1_ci["ci95_low"] > 0,
                                       final_harmed_count_mean=harm_count,
                                       main_baseline="R34", test_read=False))
    print("[R38 comparison complete]", root, flush=True)


def plot(root, own, old, results, reference, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("R37A full", "zero", reference, old, "#a5a5a5", "--"),
              ("R37B full", "real", reference, old, "#907b43", "--"),
              ("R38A frozen", "zero", results, own, "#c35c30", "-"),
              ("R38B frozen", "real", results, own, "#137d9b", "-")]
    for metric in METRICS + PROB_METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for axis, split in zip(axes, ("train", "val")):
            for name, mode, models, history, color, style in groups:
                selected = [history[r["name"]] for r in models if r["mode"] == mode]
                x = [h["updates"] for h in selected[0]]
                values = np.array([[h[split][metric] for h in rows] for rows in selected])
                if name.startswith("R38"):
                    for curve in values:
                        axis.plot(x, curve, color=color, alpha=.18, linewidth=1)
                axis.plot(x, values.mean(0), color=color, linestyle=style, label=name, linewidth=2)
            axis.axhline(baseline[split][metric], color="black", linestyle=":", label="R34 frozen reference")
            axis.axvspan(800, 1000, color="gray", alpha=.05)
            axis.set(xlabel="Successful TSP updates", ylabel=metric, title="TSP " + split)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.savefig(root / f"comparison_{metric}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = {s: raw_tsp(s) for s in ("train", "val")}
    attribute_existing(ROOT, raw["val"])
    if not args.attribution_only:
        compare(ROOT, raw)


if __name__ == "__main__":
    main()
