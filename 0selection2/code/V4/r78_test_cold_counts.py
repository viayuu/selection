"""Post-hoc delivery counts for two completed TEST caches; no fitting or inference."""

import csv
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .r76_probability_diagnostics import read_json, score_ranks, write_rows


V4 = Path(__file__).resolve().parent
OLD = V4 / "runs/R76_R45A_single_reld_mt"
NEW = V4 / "runs/R77_sample_size_batch1024/B_full_seed2"
OUT = V4 / "runs/R78_zero_selection_goal/final_delivery"
STEM = "R76_R77_TEST_COLD_COUNTS"
P = len(PROBLEMS)


def load_predictions(root):
    directory = root / "evaluation/test/selected_best"
    expected = read_json(directory / "metrics.json")["per_problem"]
    parts = {}
    paths = sorted(directory.glob("rank*.npz"))
    if not paths or set(expected) != set(PROBLEMS):
        raise ValueError("Complete saved TEST predictions for all 18 tasks are required")
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if key.endswith("__logits"):
                    problem = key.removesuffix("__logits")
                    parts.setdefault(problem, []).append({field: saved[problem + "__" + field]
                        for field in ("indices", "logits", "costs", "winner", "pool", "pool_ids", "nodes")})
    if set(parts) != set(PROBLEMS):
        raise ValueError("TEST prediction task coverage is incomplete")
    result = {}
    for problem in PROBLEMS:
        chunks = parts[problem]
        for chunk in chunks:
            for field in ("pool", "pool_ids"):
                if not np.array_equal(chunk[field], chunks[0][field]):
                    raise ValueError(f"{problem}: candidate identities differ across shards")
        value = {field: np.concatenate([chunk[field] for chunk in chunks])
                 for field in ("indices", "logits", "costs", "winner", "nodes")}
        value.update(pool=chunks[0]["pool"], pool_ids=chunks[0]["pool_ids"])
        order = np.argsort(value["indices"], kind="stable")
        n = len(order)
        if (not np.array_equal(value["indices"][order], np.arange(n))
                or n != expected[problem]["n"] or n != 3895):
            raise ValueError(f"{problem}: native prediction rows are missing or duplicated")
        for field in ("indices", "logits", "costs", "winner", "nodes"):
            value[field] = value[field][order]
        if value["costs"].dtype != np.float64:
            raise ValueError("Saved original FP64 costs are required")
        selected, _ = score_ranks(value["logits"], value["pool_ids"], ~np.isneginf(value["logits"]))
        value["selected"] = selected
        if not np.allclose(np.bincount(selected, minlength=len(value["pool"])) / n,
                           expected[problem]["pick_dist"], rtol=0, atol=1e-12):
            raise ValueError(f"{problem}: recovered choices differ from the delivered metrics")
        result[problem] = value
    return result, paths


def mean_or_blank(values):
    return float(np.mean(values)) if len(values) else ""


def fixed_stats(mask, a_regret, b_regret, a_correct, b_correct, n):
    count = int(mask.sum())
    a_sum, b_sum = float(a_regret[mask].sum()), float(b_regret[mask].sum())
    return dict(
        fixed_winner_rows=count, fixed_r76_correct_hits=int(a_correct[mask].sum()),
        fixed_b_correct_hits=int(b_correct[mask].sum()), fixed_b_missed_rows=int((~b_correct[mask]).sum()),
        fixed_b_recovery_pct=float(100 * b_correct[mask].mean()) if count else "",
        fixed_r76_regret_conditional_pct=mean_or_blank(a_regret[mask]),
        fixed_b_regret_conditional_pct=mean_or_blank(b_regret[mask]),
        fixed_regret_delta_conditional_pp=mean_or_blank((b_regret - a_regret)[mask]),
        fixed_r76_regret_all18_contribution_pp=a_sum / (P * n),
        fixed_b_regret_all18_contribution_pp=b_sum / (P * n),
        fixed_regret_delta_all18_contribution_pp=(b_sum - a_sum) / (P * n),
        fixed_b_correct_all18_top1_contribution_pp=float(100 * b_correct[mask].sum() / (P * n)),
    )


def analyze():
    with (OLD / "missed_winner_probabilities/zero_methods_test.csv").open(newline="") as stream:
        fixed = {(row["problem"], row["solver"]): int(row["true_winner_count"])
                 for row in csv.DictReader(stream)}
    if len(fixed) != 52 or sum(fixed.values()) != 14077:
        raise ValueError("The original fixed cohort must remain 52 pairs / 14077 native-winner rows")
    old, old_paths = load_predictions(OLD)
    new, new_paths = load_predictions(NEW)
    methods, tasks, observed_fixed = [], [], {}
    for problem in PROBLEMS:
        a, b = old[problem], new[problem]
        pool, b_pool = a["pool"].tolist(), b["pool"].tolist()
        if len(set(pool)) != len(pool) or set(pool) != set(b_pool):
            raise ValueError(f"{problem}: candidate names do not align")
        columns = [b_pool.index(name) for name in pool]
        b_to_a = np.asarray([pool.index(name) for name in b_pool])
        for field in ("indices", "nodes"):
            if not np.array_equal(a[field], b[field]):
                raise ValueError(f"{problem}: {field} do not align")
        if (not np.array_equal(a["costs"], b["costs"][:, columns])
                or not np.array_equal(a["winner"], b_to_a[b["winner"]])
                or not np.array_equal(a["pool_ids"], b["pool_ids"][columns])):
            raise ValueError(f"{problem}: FP64 costs / native winners / solver IDs do not align")
        n, winner = len(a["winner"]), a["winner"]
        ap, bp = a["selected"], b_to_a[b["selected"]]
        ac, bc = ap == winner, bp == winner
        oracle = a["costs"].min(1)
        a_cost, b_cost = a["costs"][np.arange(n), ap], a["costs"][np.arange(n), bp]
        ar, br = 100 * (a_cost - oracle) / oracle, 100 * (b_cost - oracle) / oracle
        a_count, b_count = (np.bincount(pred, minlength=len(pool)) for pred in (ap, bp))
        cohort = np.zeros(n, dtype=bool)
        local = []
        for slot, method in enumerate(pool):
            won = winner == slot
            wins = int(won.sum())
            frozen = (problem, method) in fixed
            a_zero, b_zero = bool(a_count[slot] == 0 and wins), bool(b_count[slot] == 0 and wins)
            if a_zero:
                observed_fixed[(problem, method)] = wins
            if frozen and (not a_zero or wins != fixed[(problem, method)]):
                raise ValueError("The frozen R76 zero cohort no longer describes these saved rows")
            mask = won & frozen
            cohort |= mask
            hits = int((won & bc).sum())
            row = dict(
                row_type="method", problem=problem, method=method, instances=n,
                true_winner_count=wins, r76_selected_count=int(a_count[slot]), b_selected_count=int(b_count[slot]),
                r76_correct_hits=int((won & ac).sum()), b_correct_hits=hits,
                r76_fixed_zero_pair_count=int(frozen), b_zero_pair_count=int(b_zero),
                b_zero_winner_rows=wins if b_zero else 0,
                b_new_zero_outside_fixed_pair_count=int(b_zero and not frozen),
                b_new_zero_outside_fixed_winner_rows=wins if b_zero and not frozen else 0,
                fixed_b_selected_pair_count=int(frozen and b_count[slot] > 0),
                fixed_b_hit_pair_count=int(frozen and hits > 0),
                fixed_b_selected_nohit_pair_count=int(frozen and b_count[slot] > 0 and hits == 0),
                fixed_b_still_zero_pair_count=int(frozen and b_count[slot] == 0),
                **fixed_stats(mask, ar, br, ac, bc, n))
            methods.append(row)
            local.append(row)
        task = dict(row_type="task", problem=problem, method="__ALL__", instances=n,
                    true_winner_count=n, r76_selected_count=n, b_selected_count=n,
                    r76_correct_hits=int(ac.sum()), b_correct_hits=int(bc.sum()))
        for key in ("r76_fixed_zero_pair_count", "b_zero_pair_count", "b_zero_winner_rows",
                    "b_new_zero_outside_fixed_pair_count", "b_new_zero_outside_fixed_winner_rows",
                    "fixed_b_selected_pair_count", "fixed_b_hit_pair_count", "fixed_b_selected_nohit_pair_count",
                    "fixed_b_still_zero_pair_count"):
            task[key] = sum(row[key] for row in local)
        task.update(fixed_stats(cohort, ar, br, ac, bc, n),
                    r76_top1_pct=float(100 * ac.mean()), b_top1_pct=float(100 * bc.mean()),
                    top1_delta_pp=float(100 * (bc.astype(float) - ac).mean()),
                    r76_mean_cost=float(a_cost.mean()), b_mean_cost=float(b_cost.mean()),
                    r76_regret_pct=float(ar.mean()), b_regret_pct=float(br.mean()),
                    regret_delta_pp=float((br - ar).mean()))
        tasks.append(task)
    if observed_fixed != fixed:
        raise ValueError("The saved R76 fixed category set is inconsistent with the prediction cache")
    total = dict(row_type="all18", problem="ALL18", method="__ALL__")
    for key in tasks[0]:
        if key in total:
            continue
        values = [row[key] for row in tasks]
        if key.endswith("_count") or key.endswith("_hits") or key.endswith("_rows") or key == "instances":
            total[key] = sum(values)
        elif key in ("fixed_b_recovery_pct", "fixed_r76_regret_conditional_pct",
                     "fixed_b_regret_conditional_pct", "fixed_regret_delta_conditional_pp"):
            continue
        elif "all18_" in key:
            total[key] = float(sum(values))
        else:
            total[key] = float(np.mean(values))
    cold_n = total["fixed_winner_rows"]
    total["fixed_b_recovery_pct"] = 100 * total["fixed_b_correct_hits"] / cold_n
    for key in ("fixed_r76_regret_conditional_pct", "fixed_b_regret_conditional_pct",
                "fixed_regret_delta_conditional_pp"):
        total[key] = sum(row[key] * row["fixed_winner_rows"] for row in tasks if row["fixed_winner_rows"]) / cold_n
    total["outside_fixed_regret_delta_all18_contribution_pp"] = (
        total["regret_delta_pp"] - total["fixed_regret_delta_all18_contribution_pp"])
    if total["fixed_r76_correct_hits"] != 0 or cold_n != 14077:
        raise ValueError("The fixed R76 missed-winner row population changed")
    return methods, tasks, total, old_paths, new_paths


def report(methods, tasks, total, old_paths, new_paths):
    a_lock, b_lock = (read_json(root / "final_test_lock.json") for root in (OLD, NEW))
    t = total
    still_zero_fixed_rows = sum(row["fixed_winner_rows"] for row in methods
                                if row["fixed_b_still_zero_pair_count"])
    text = ["# R76 / R77B TEST 冷选真实纠正：完成版本交付", "",
            "**仅事后描述已经完成的两个版本，不是调参、规则评估或训练依据。没有读取或修改 round006，其协议/14904预算/VAL gate保持不变；没有向Newton训练owner发送本统计。**", "",
            f"R76锁定 ep{a_lock['epoch']}，SHA `{a_lock['checkpoint_sha256']}`；R77B锁定 ep{b_lock['epoch']}/update{b_lock['selected_update']:,}，SHA `{b_lock['checkpoint_sha256']}`。仅复用已保存TEST logits/原始FP64成本/native winner；按原生行号、候选名称/global ID对齐，节点数、成本矩阵和winner身份一致；并列选法沿用global ID规则，无模型加载、推理或重跑TEST。", "",
            "## 核心结果", "",
            f"固定R76旧集合：**52个task×method、14,077个native winner实例**。B在这些完全相同的实例上真实命中 **{t['fixed_b_correct_hits']:,} / 14,077 = {t['fixed_b_recovery_pct']:.4f}%**，R76为0；仍漏选 **{t['fixed_b_missed_rows']:,}** 行。命中严格要求最终所选方法等于保存的native winner，成本并列不算身份命中。", "",
            f"52个旧类别中，B在 {t['fixed_b_selected_pair_count']} 类出现非零选择，其中 {t['fixed_b_hit_pair_count']} 类至少命中1次；{t['fixed_b_selected_nohit_pair_count']} 类虽选过却零命中，{t['fixed_b_still_zero_pair_count']} 类仍选0次。不得用“选过一次”把整类winner都算作修复。", "",
            f"B自身zero-with-winner：**{t['b_zero_pair_count']}个类别 / {t['b_zero_winner_rows']:,}个winner行**，占全部 {t['instances']:,} 行的 {100 * t['b_zero_winner_rows'] / t['instances']:.4f}%；其中旧集合外新出现zero为 {t['b_new_zero_outside_fixed_pair_count']} 类 / {t['b_new_zero_outside_fixed_winner_rows']} 行。", "",
            f"按动态zero类别计数已消失的旧winner行有 {t['fixed_winner_rows'] - still_zero_fixed_rows:,}，但其中 **{t['fixed_b_missed_rows'] - still_zero_fixed_rows:,} 行仍未命中**；这些类别出现非零选择不等于整类winner已被修复。", "",
            "## 分母与成本变化", "",
            "逐行FP64相对regret定义为 r=100×(所选成本−同池最小成本)/同池最小成本；保留native winner身份，不用FP32成本重标。每题N=3,895，P=18，总N=70,110。固定子集S只由R76旧zero类别及native winner决定，不随B选法改变。", "",
            "| 指标/分母 | R76 | R77B | B−R76 |", "| --- | ---: | ---: | ---: |"]
    for name, a, b, digits in (
        ("全18等权Top1 (%)", t["r76_top1_pct"], t["b_top1_pct"], 6),
        ("全18等权mean cost", t["r76_mean_cost"], t["b_mean_cost"], 9),
        ("全18等权regret (%)", t["r76_regret_pct"], t["b_regret_pct"], 9),
        ("固定S对全18regret贡献 (pp)", t["fixed_r76_regret_all18_contribution_pp"], t["fixed_b_regret_all18_contribution_pp"], 9),
        ("仅固定14,077行conditional regret (%)", t["fixed_r76_regret_conditional_pct"], t["fixed_b_regret_conditional_pct"], 9)):
        text.append(f"| {name} | {a:.{digits}f} | {b:.{digits}f} | {b - a:+.{digits}f} |")
    text.extend(["", "全18等权regret = (1/18)Σp mean_i(r_pi)。固定S的全18贡献 = (1/18)Σp Σ(i∈S_p)r_pi/N_p，**不是**在子集内重新求均值；conditional = Σ(p,i∈S_p)r_pi/14,077，按子集实例汇总，不把各题子集均值简单等权平均。CSV的每题/每方法all18_contribution也已含1/18，ALL18行是这些贡献求和。", "",
                 f"固定S真实纠正对应全18Top1的毛贡献为 **+{t['fixed_b_correct_all18_top1_contribution_pp']:.6f}个百分点**，不能替代整体净Top1变化 {t['top1_delta_pp']:+.6f}个百分点。全18regret净变化 {t['regret_delta_pp']:+.9f}pp = 固定S贡献变化 {t['fixed_regret_delta_all18_contribution_pp']:+.9f}pp + 其余实例贡献变化 {t['outside_fixed_regret_delta_all18_contribution_pp']:+.9f}pp。原始成本跨题尺度不同，主成本结论使用上述逐行相对regret。", "",
                 "## 18题固定实例与B自身zero", "",
                 "| 问题 | 旧zero类 | 固定winner行 | B真实纠正 | 固定召回率 | B自身zero类/行 | 旧S regret R76/B (%) |",
                 "| --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in tasks:
        cond = f"{row['fixed_r76_regret_conditional_pct']:.6f}/{row['fixed_b_regret_conditional_pct']:.6f}" if row["fixed_winner_rows"] else "n/a"
        recovery = f"{row['fixed_b_recovery_pct']:.4f}%" if row["fixed_winner_rows"] else "n/a"
        text.append(f"| {row['problem']} | {row['r76_fixed_zero_pair_count']} | {row['fixed_winner_rows']:,} | {row['fixed_b_correct_hits']:,} | {recovery} | {row['b_zero_pair_count']}/{row['b_zero_winner_rows']} | {cond} |")
    text.extend(["", "## 旧52类完整纠正表", "",
                 "B选择数是本题全部3,895行中的选择次数；B正确数只计该方法为native winner的原有固定行。CSV还包含全部候选（非仅旧zero）的两模型选择数、真实winner数及正确命中数。", "",
                 "| 问题 | 方法 | 固定winner数 | B选择数 | B正确数 | 固定召回率 | conditional regret R76/B (%) |",
                 "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for row in methods:
        if row["r76_fixed_zero_pair_count"]:
            text.append(f"| {row['problem']} | {row['method']} | {row['true_winner_count']:,} | {row['b_selected_count']:,} | {row['b_correct_hits']:,} | {row['fixed_b_recovery_pct']:.4f}% | {row['fixed_r76_regret_conditional_pct']:.6f}/{row['fixed_b_regret_conditional_pct']:.6f} |")
    text.extend(["", "## 文件与隔离", "",
                 f"- [完整CSV]({OUT / (STEM + '.csv')})：`method`行覆盖全部{len(methods)}个候选组合；`task`行18题；`all18`行完整等权结果。固定字段只统计旧S，非旧zero方法的固定字段为0或空，不代表其整体未被命中。",
                 f"- [固定旧zero来源]({OLD / 'missed_winner_probabilities/zero_methods_test.csv'})：只读，保持52类/14,077行。",
                 f"- [R76已交付TEST预测目录]({old_paths[0].parent})：{', '.join(path.name for path in old_paths)}。",
                 f"- [R77B已交付TEST预测]({new_paths[0]})。两模型的原有metrics仅用于确认行数与保存选法一致。",
                 f"- [CPU脚本]({Path(__file__).resolve()})：`PYTHONDONTWRITEBYTECODE=1 python -B -m code.V4.r78_test_cold_counts`（仓库根目录/easynco）。",
                 "- 本次只写该final_delivery目录和上述小脚本；未读未交付训练结果，未访问round006/STATUS/旧数据源或solver；无概率全矩阵、温度/规则拟合、GPU、前向、训练或Git提交。单种子完成版本对比，不作单变量因果或未来泛化结论。", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = methods + tasks + [total]
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    write_rows(OUT / (STEM + ".csv"), all_rows, fields=fields)
    (OUT / (STEM + ".md")).write_text("\n".join(text), encoding="utf-8")


def main():
    methods, tasks, total, old_paths, new_paths = analyze()
    report(methods, tasks, total, old_paths, new_paths)
    import json
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
