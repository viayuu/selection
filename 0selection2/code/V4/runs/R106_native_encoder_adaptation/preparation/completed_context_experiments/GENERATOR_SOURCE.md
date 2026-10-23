# Report Generator Source

This archive preserves the local `build_report.py` verbatim for the whole-version
publication, whose preparation-artifact filter includes Markdown but not extra
Python files. It is reporting code only, not part of the frozen training or
inference source closure. See `METHODS.md` for inputs, scope and the normal local
command. No model, label, prediction, solver or GPU is executed by this report.

```python
"""CPU-only interpretation of existing locked, aggregate result artifacts."""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUNS = OUT.parents[2]
ROOT = RUNS.parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
FP = FontProperties(fname=str(FONT))
plt.rcParams.update({"font.family": FP.get_name(), "pdf.fonttype": 42, "axes.unicode_minus": False})

SPECS = [
    ("R89B", "R89_optimizer_geometry/B_muon_hybrid", "Muon历史基线"),
    ("R94A", "R94_pretrained_node_fusion/A", "冻结随机routing"),
    ("R94B", "R94_pretrained_node_fusion/B", "冻结预训练routing"),
    ("R99A", "R99_global_competition_input/preparation/postprocess_recovery/report_view/A", "row-softmax输入"),
    ("R99B", "R99_global_competition_input/preparation/postprocess_recovery/report_view/B", "Sinkhorn输入"),
    ("R101A", "R101_hint_pretraining/A", "随机hint后reset"),
    ("R101B", "R101_hint_pretraining/B", "预训练hint后reset"),
    ("R102A", "R102_full_type_pretrained_fusion/preparation/postprocess_recovery/report_view/A", "预训练routing+随机额外类型"),
    ("R102B", "R102_full_type_pretrained_fusion/preparation/postprocess_recovery/report_view/B", "预训练routing+真实额外预训练"),
    ("R104A", "R104_capacity_scaling/A_width256", "加宽256"),
    ("R104B", "R104_capacity_scaling/B_depth8", "加深8/4层"),
]
PAIRS = {
    "R94": "R94_pretrained_node_fusion",
    "R99": "R99_global_competition_input/preparation/postprocess_recovery/report_view",
    "R101": "R101_hint_pretraining",
    "R102": "R102_full_type_pretrained_fusion/preparation/postprocess_recovery/report_view",
    "R104": "R104_capacity_scaling",
}
READS = set()


def rel(path):
    return str(path.relative_to(ROOT))


def load_csv(path):
    READS.add(rel(path))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    READS.add(rel(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value):
    return float(value) if value not in (None, "", "NA") else None


def fmt(value, digits=3, signed=False):
    value = number(value)
    return "NA" if value is None else format(value, f"{'+' if signed else ''}.{digits}f")


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="NA")
        writer.writeheader()
        writer.writerows({key: "NA" if value in (None, "") else value for key, value in row.items()} for row in rows)


models, metrics, recovery, references, groups, nss, pairs = {}, [], [], [], [], [], []
for name, directory, mechanism in SPECS:
    folder = RUNS / directory / "full_model_statistics"
    protocol = load_json(folder / "protocol.json")
    lock = protocol["checkpoint_lock"]
    rows = load_csv(folder / "metrics_by_problem.csv")
    main_has_train = any(row["split"] == "TRAIN" for row in rows)
    if not main_has_train:
        rows += load_csv(folder / "train_metrics_by_problem.csv")
    selected = []
    train_verified = True
    for row in rows:
        if row["split"] not in ("TRAIN", "VAL", "TEST"):
            continue
        if row["split"] == "TRAIN":
            source = row.get("metric_source", "")
            train_verified = (
                row.get("checkpoint_sha256") == lock.get("checkpoint_sha256")
                and ("/selected_best/" in source or f"/update{lock['selected_update']:07d}/" in source)
            )
            if not train_verified:
                continue
        source_file = "train_metrics_by_problem.csv" if row["split"] == "TRAIN" and not main_has_train else "metrics_by_problem.csv"
        selected.append(dict(row, model_id=name, source_csv=rel(folder / source_file)))
    metrics.extend(selected)
    summary_file = folder / "target_recovery_summary.csv"
    if not summary_file.exists():
        summary_file = folder / "target_recovery_summary_vs_R84A.csv"
    summaries = load_csv(summary_file)
    group_file = folder / "target_recovery.csv"
    if not group_file.exists():
        group_file = folder / "target_recovery_vs_R84A.csv"
    raw_groups = load_csv(group_file)
    for split in ("TRAIN", "VAL", "TEST"):
        for subset, category in (("all36", None), ("strict_zero29", "zero"), ("near_zero7", "near_zero")):
            sub = [row for row in raw_groups if row["split"] == split and (category is None or row["target_category"] == category)]
            totals = {key: sum(int(row[key]) for row in sub) for key in ("winner_count", "correct_count", "selected_count", "false_pick_count")}
            recovery.append(dict(model_id=name, split=split, subset=subset, target_groups=len(sub), **totals,
                                 recall_pct=100 * totals["correct_count"] / totals["winner_count"] if totals["winner_count"] else None,
                                 precision_pct=100 * totals["correct_count"] / totals["selected_count"] if totals["selected_count"] else None,
                                 groups_with_correct=sum(int(row["correct_count"]) > 0 for row in sub),
                                 groups_with_test_or_split_winners=sum(int(row["winner_count"]) > 0 for row in sub),
                                 source_csv=rel(group_file)))
    groups.extend(dict(row, model_id=name, source_csv=rel(group_file)) for row in raw_groups if row["split"] == "TEST")
    files = [summary_file] + [folder / f"target_recovery_summary_vs_{ref}.csv" for ref in ("R84A", "R89B", "R94B")]
    seen = set()
    for file in files:
        if not file.exists():
            continue
        for row in load_csv(file):
            ref = row["baseline"].replace(" historical", "")
            if (ref, row["split"]) in seen:
                continue
            seen.add((ref, row["split"]))
            pair_file = folder / f"paired_changes_vs_{ref}.csv"
            pair_rows = load_csv(pair_file) if pair_file.exists() else []
            paired = next((x for x in pair_rows if x["split"] == row["split"] and x["problem"] == "ALL"), {})
            net_cost = paired.get("mean_cost_delta", row.get("full18_net_mean_cost_delta"))
            references.append(dict(row, model_id=name, baseline=ref, net_rescues=int(row["rescued"]) - int(row["lost"]),
                                   full18_net_mean_cost_delta=net_cost, source_csv=rel(file),
                                   net_cost_source_csv=rel(pair_file) if paired else rel(file) if net_cost is not None else "NA"))
    nss_file = folder / "nss_comparison.csv"
    if nss_file.exists():
        nss.extend(dict(row, model_id=name, data_version=selected[0]["data_version"], candidate_pool="same per-problem legal pool as metrics_by_problem.csv", source_csv=rel(nss_file)) for row in load_csv(nss_file))
    models[name] = dict(folder=folder, protocol=protocol, selected=selected, mechanism=mechanism, train_verified=train_verified, summaries=summaries)

for experiment, directory in PAIRS.items():
    file = RUNS / directory / "comparison_artifacts/paired_changes_B_vs_A.csv"
    pairs.extend(dict(row, experiment=experiment, matched_control=(experiment != "R104"), source_csv=rel(file)) for row in load_csv(file) if row["problem"] == "ALL")

baseline = {(row["split"], row["problem"]): row for row in models["R89B"]["selected"]}
for row in metrics:
    base = baseline.get((row["split"], row["problem"]), {})
    for key in ("top1_pct", "actual_regret_pct", "mean_cost"):
        row[f"delta_{key}_vs_R89B"] = float(row[key]) - float(base[key]) if key in base else None

overall = [row for row in metrics if row["problem"] == "ALL"]
write_csv("overall_summary.csv", overall)
write_csv("metrics_by_problem.csv", metrics)
write_csv("recovery_summary.csv", recovery)
write_csv("recovery_by_reference.csv", references)
write_csv("target_groups_test.csv", groups)
write_csv("matched_ab.csv", pairs)
write_csv("nss_comparison.csv", nss)


def metric(name, split="TEST"):
    return next((row for row in overall if row["model_id"] == name and row["split"] == split), {})


def rec(name, subset="all36"):
    return next(row for row in recovery if row["model_id"] == name and row["split"] == "TEST" and row["subset"] == subset)


def pair(experiment, split):
    return next(row for row in pairs if row["experiment"] == experiment and row["split"] == split)


overall_headers = ["模型 / 方向", "TRAIN Top1%", "VAL Top1%", "TEST Top1%", "TEST差pp", "VAL regret%", "TEST regret%", "TEST mean_cost"]
overall_rows = []
recovery_headers = ["模型", "36组正确", "召回%", "误选", "precision%", "严格零组正确", "召回%", "误选", "precision%"]
recovery_rows = []
for name, _, mechanism in SPECS:
    train, val, test = metric(name, "TRAIN"), metric(name, "VAL"), metric(name)
    overall_rows.append([f"{name} {mechanism}", fmt(train.get("top1_pct")), fmt(val.get("top1_pct")), fmt(test.get("top1_pct")),
                         fmt(test.get("delta_top1_pct_vs_R89B"), signed=True), fmt(val.get("actual_regret_pct"), 4),
                         fmt(test.get("actual_regret_pct"), 4), fmt(test.get("mean_cost"), 7)])
    a, z = rec(name), rec(name, "strict_zero29")
    recovery_rows.append([name, a["correct_count"], fmt(a["recall_pct"], 2), a["false_pick_count"], fmt(a["precision_pct"], 2),
                          z["correct_count"], fmt(z["recall_pct"], 2), z["false_pick_count"], fmt(z["precision_pct"], 2)])

reference_headers = ["模型", "明确参考", "rescued", "lost", "净恢复", "new_false", "目标regret差pp", "全18净cost差"]
reference_rows = []
for name, _, _ in SPECS[1:]:
    ref = "R94B" if name.startswith("R102") else "R89B"
    row = next((x for x in references if x["model_id"] == name and x["baseline"] == ref and x["split"] == "TEST"), {})
    reference_rows.append([name, ref, row.get("rescued", "NA"), row.get("lost", "NA"), row.get("net_rescues", "NA"),
                           row.get("new_false_pick_count", "NA"), fmt(row.get("target_regret_delta_pp"), 5, True),
                           fmt(row.get("full18_net_mean_cost_delta"), 7, True)])

pair_headers = ["B 相对 A", "TEST改变", "纠正", "损害", "净正确", "VAL Top1差pp", "TEST Top1差pp", "VAL regret差pp", "TEST regret差pp"]
pair_rows = []
for experiment in PAIRS:
    v, t = pair(experiment, "VAL"), pair(experiment, "TEST")
    pair_rows.append([experiment + (" 非严格对照" if experiment == "R104" else " matched"), t["changed"], t["corrected"], t["harmed"],
                      fmt(t["net_correct"], 0, True), fmt(v["top1_delta_pp"], signed=True), fmt(t["top1_delta_pp"], signed=True),
                      fmt(v["actual_regret_delta_pp"], 5, True), fmt(t["actual_regret_delta_pp"], 5, True)])

focus_source = OUT / "target_groups_test.csv"
focus_candidates = [row for row in load_csv(focus_source) if row["model_id"] == "R102A" and row["split"] == "TEST"
                    and int(row["winner_count"]) > 0 and int(row["correct_count"]) == 0]
focus = sorted(focus_candidates, key=lambda row: (-int(row["winner_count"]), row["problem"], row["solver"]))[:5]
write_csv("unrecovered_focus.csv", [dict(
    rank=rank, model_id=row["model_id"], split=row["split"], problem=row["problem"], solver=row["solver"],
    winner_count=row["winner_count"], selected_count=row["selected_count"], correct_count=row["correct_count"],
    false_pick_count=row["false_pick_count"], target_category=row["target_category"],
    category_basis="R84A locked VAL historical list; not current prediction category",
    source_csv=rel(focus_source), original_source_csv=row["source_csv"]
) for rank, row in enumerate(focus, 1)])
group_headers = ["问题 / 方法", "TEST winner", "R102A选择", "正确 / 误选"]
group_rows = [[f"{row['problem']} / {row['solver']}", row["winner_count"], row["selected_count"],
               f"{row['correct_count']} / {row['false_pick_count']}"] for row in focus]

PAGE1_NOTES = [
    "结论：R94B、R102A 相对 R89B 有小幅净收益，但不是大幅突破；R99、R104 均未超过基线。",
    "R94B：VAL/TEST Top1 +0.180/+0.158pp；TEST regret -0.01053pp（相对下降约2.70%）。",
    "R102A：VAL/TEST Top1 +0.053/+0.170pp；TEST regret -0.01120pp（相对下降约2.88%）。",
    "但 R102A 比 R94B：VAL Top1 -0.127pp、TEST仅 +0.012pp（10例），regret仅 -0.00067pp；额外净收益很小。",
    "R101B 比 R89B：TEST仅 +0.015pp（12例），regret反增 +0.00137pp；不能称为整体改善。",
]
PAGE2_NOTES = [
    "36组分母4,598；严格零29组分母2,731。另7组是近零，不得把36组都称为严格零选择。",
    "严格零来自 R84A 锁定VAL的选择=0；并非本模型/TEST选择=0。召回=正确/winner；precision=正确/选择。",
    "R102A最好也仅135/4,598（2.94%），仍漏4,463；严格零69/2,731（2.53%），同时误选59。",
]
PAGE3_NOTES = [
    "R94：纠正>损害，VAL/TEST Top1与regret同向改善，支持这次routing预训练对照有正收益，但幅度有限。",
    "R99：2,056次改变仅净对26例，VAL反向且TEST regret更差；R101的matched收益较小，未胜R89B整体质量。",
    "R102：A/B共同使用预训练routing；B新增真实TSP/ATSP预训练后，TEST净损295例，目标组也14救回/57丢失。",
    "旧zero清单命中仅5组：VRPBL/MTPOMO29、ATSP/MATPOENET26、OVRPB/MTPOMO11，另两组2和1。",
    "zero是R84A-VAL旧类别，非当前选择0；27组有TEST winner、22组仍0命中，另2组无winner；LEHD选1次全错不算恢复。",
    "边界：R101迁移encoder后重置下游优化器/随机流，不等同冻结native；R104宽/深不是严格单变量A/B。",
    "这些阴性结果不证明标签随机、无实例信息或达到上限；单seed与研发用TEST不足以证明可复现/OOD迁移。",
]


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


md = ["# 已完成模型：近期结果解释", "", "仅解释既有锁定证据；核心R99/R101/R102/R104，R89B与R94为参照。R105/R106仍训练，排除结果，不预测结论。", "",
      "## 1. 哪个方向有真实净收益？", "", "NEW90k full18；TRAIN 81,000/题，VAL与TEST各4,500/题（各81,000）。每版本唯一VAL锁定checkpoint与推理规则；TRAIN仅使用该selected checkpoint保存统计。", "",
      md_table(overall_headers, overall_rows), "", *PAGE1_NOTES, "",
      "ALL为18题等权；跨题raw mean_cost仅描述，不能单凭其差值评价净收益。TEST差相对R89B，pp=百分点；regret为逐实例(actual cost/oracle-1)的题内均值再等题平均，不是vs_Oracle均值比。", "",
      "## 2. 固定36组与真正零选择组救回多少？", "", *PAGE2_NOTES, "", md_table(recovery_headers, recovery_rows), "",
      "### 相对各自明确参考的增减（TEST，36组）", "", md_table(reference_headers, reference_rows), "",
      "目标regret差是固定目标winner子集对full18 actual regret的贡献差，不是目标子集自身的条件均值；全18净cost差取同参考paired ALL mean_cost_delta，负值更好，跨题raw cost仍只描述。R102的参考为R94B，不冒充R89B；未提供的R102-vs-R89B逐实例rescued/lost为NA，不推算。全部R84A共同参考及可用成本项保留于CSV。", "",
      "## 3. 真正学到区分，还是仅改变选择？", "", "所有纠正/损害计数直接复用已有paired_changes；未读取预测数组或重开推理。", "", md_table(pair_headers, pair_rows), "", *PAGE3_NOTES[:3], "",
      "### R102A最大仍未恢复组", "", "来源：target_groups_test.csv，筛选固定36组中TEST winner>0且正确=0的R102A组，按winner降序取前5；同表另存unrecovered_focus.csv。", "",
      md_table(group_headers, group_rows), "", *PAGE3_NOTES[3:], "",
      "### 证据边界", "", "R99/R102是已保存cached-feature TEST效能，不是bitwise online效能认证。原timing失败保留；R102原metadata/额外strict-timing失败也保留，本报告不改其状态。cached selector耗时不代表端到端推理，本文不作速度结论。", "",
      "全部winner为native/FP64 ind，不混FP32 argmin。合法池维持NEW90k规则（B题删除3方法、移除RELD_MTL、保留单一reld_mt）；Oracle为逐实例合法最低成本，SBS为冻结81k TRAIN拟合。完整每题Top1/mean_cost/SBS/Oracle/vs_SBS/vs_Oracle/actual regret及selected TRAIN、VAL、TEST见metrics_by_problem.csv；同bank NSS四数据集结果见nss_comparison.csv（TSPLIB49/CVRPLIB100）。", "",
      "单seed历史版本比较不作因果或显著性结论；TEST已参与研发，不称盲测。预训练外部数据/重叠和计算预算仍有来源限制。不给新实验、loss或协议建议，不将R106论文DEFER当成实测证据。来源及复现命令见inputs_provenance.json与METHODS.md。", ""]
(OUT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")


def page(number_, title, subtitle):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(0.045, 0.956, "已完成模型 / NEW90k full18 / 结果解释", fontproperties=FP, fontsize=10, color="#36756e")
    fig.text(0.045, 0.906, title, fontproperties=FP, fontsize=19, color="#18332f")
    fig.text(0.045, 0.870, subtitle, fontproperties=FP, fontsize=9.1, color="#4b5855")
    fig.text(0.045, 0.027, "只读小型CSV/JSON汇总；无训练、推理、weights、预测数组、solver或GPU操作。", fontproperties=FP, fontsize=8, color="#6a716e")
    fig.text(0.955, 0.027, f"{number_} / 3", ha="right", fontproperties=FP, fontsize=8, color="#6a716e")
    return fig


def table(fig, headers, rows, bottom, height, widths, font=9, highlight=()):
    ax = fig.add_axes([0.045, bottom, 0.91, height])
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colWidths=widths, bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(font)
    for (i, j), cell in tab.get_celld().items():
        cell.get_text().set_fontproperties(FP)
        cell.get_text().set_fontsize(font)
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#dbe3df")
        cell.PAD = 0.025
        if i == 0:
            cell.set_facecolor("#234f48")
            cell.get_text().set_color("white")
        else:
            cell.set_facecolor("#e5f1e9" if i - 1 in highlight else "#f3f6f4" if i % 2 == 0 else "white")
        if j == 0:
            cell.get_text().set_ha("left")
    return tab


def lines(fig, texts, top, size=10, step=0.032, color="#263c36"):
    for i, text in enumerate(texts):
        fig.text(0.049, top - i * step, text, fontproperties=FP, fontsize=size, color=color)


with PdfPages(OUT / "RECENT_RESULTS.pdf", metadata={"Title": "NEW90k full18 已完成模型近期结果", "Author": "CPU result sidecar", "Subject": "Locked aggregate evidence only"}) as pdf:
    fig = page(1, "1. 哪个方向有真实净收益？", "11个唯一VAL锁定模型；TRAIN为selected checkpoint；VAL/TEST各81,000例；TEST差相对R89B。")
    table(fig, overall_headers, overall_rows, .427, .410, [.285, .092, .092, .092, .080, .087, .087, .125], font=8.8, highlight=(2, 7))
    lines(fig, PAGE1_NOTES, .386, size=10.2, step=.040)
    lines(fig, ["口径：ALL=18题等权；TEST mean_cost为跨题raw cost，仅作描述。pp=百分点，regret越低越好。",
                "actual regret=逐实例(cost/oracle-1)的题内均值再等题平均；不是vs_Oracle的均值比。",
                "数据：每题TRAIN 81,000、VAL 4,500、TEST 4,500；native/FP64 ind；原合法池、TRAIN拟合SBS不变。",
                "范围：单seed历史比较不因果；TEST已参与研发，不称盲测。R105/R106仍训练，未列入实测结果。"], .152, size=8.9, step=.028)
    pdf.savefig(fig)
    plt.close(fig)

    fig = page(2, "2. 真正漏选被救回多少？", "固定36组=29个严格零组+7个近零组（按R84A锁定VAL定义）；下表均为TEST逐实例正确，不是选择频数。")
    table(fig, recovery_headers, recovery_rows, .546, .291, [.12, .11, .09, .08, .12, .15, .09, .08, .12], font=8.6, highlight=(2, 7))
    lines(fig, ["分母：36组4,598个winner；严格零组2,731。R102A命中135/69，误选117/59；仍漏4,463/2,662。"], .517, size=9.2)
    fig.text(.045, .478, "同参考增减：只在标明的参考内解读，不将R94B冒充R89B", fontproperties=FP, fontsize=11, color="#234f48")
    table(fig, reference_headers, reference_rows, .190, .266, [.12, .13, .10, .08, .09, .11, .19, .18], font=8.5, highlight=(1, 6))
    lines(fig, ["rescued/lost/new_false均相对明确参考；净恢复=rescued-lost。全18净cost差来自同参考paired ALL，负值更好。",
                "目标regret差=固定目标winner子集对full18 regret的贡献差，不是目标子集的条件均值；跨题raw cost仅描述。",
                "召回=正确/winner；precision=正确/(正确+误选)。严格零定义在R84A VAL，不等于本模型TEST选择为0。",
                "R102-vs-R89B逐实例rescued/lost未提供，记NA；共同R84A及其它现有成本字段完整保留在CSV。"], .151, size=8.7, step=.028)
    pdf.savefig(fig)
    plt.close(fig)

    fig = page(3, "3. 学到可迁移区分，还是仅改变选择？", "直接复用已有matched A/B corrected / harmed / regret；B相对A，负regret差为改善。R104单列非严格对照。")
    table(fig, pair_headers, pair_rows, .652, .185, [.17, .09, .07, .07, .075, .125, .125, .1375, .1375], font=8.4)
    lines(fig, PAGE3_NOTES[:3], .614, size=9.1, step=.032)
    fig.text(.045, .499, "R102A最大仍未恢复组：来源target_groups_test.csv，正确=0，按TEST winner降序取前5", fontproperties=FP, fontsize=10.5, color="#234f48")
    table(fig, group_headers, group_rows, .293, .181, [.46, .18, .18, .18], font=8.6)
    lines(fig, PAGE3_NOTES[3:], .256, size=8.6, step=.030)
    lines(fig, ["边界：R99/R102保留cached TEST结果，不是bitwise online认证；原timing失败及R102 metadata/额外strict-timing失败保留。",
                "不作端到端耗时结论；外部预训练数据/重叠与预算有来源限制。每题完整指标、NSS四数据集及成本项见CSV。",
                "R102A并非全随机；R106论文DEFER不是实测支持。本文仅解释完成证据，不提出新实验/loss，不改协议。"], .137, size=8.5, step=.029)
    pdf.savefig(fig)
    plt.close(fig)

provenance = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "CPU-only existing small aggregate CSV/JSON and existing RESULTS descriptions; no weights or prediction arrays read",
    "data_root": models["R89B"]["protocol"]["data_root"],
    "winner_rule": models["R89B"]["protocol"]["winner_rule"],
    "target_definition": models["R89B"]["protocol"]["target_identity"]["identity"]["rule"],
    "target_test_counts": {"all36": 4598, "strict_zero29": 2731, "near_zero7": 1867},
    "aggregation": "equal problem weights; cross-problem raw cost descriptive only",
    "unavailable": {"R102_vs_R89B_instance_rescued_lost_new_false": "NA; no such paired input, no prediction reconstruction"},
    "excluded_unfinished": ["R105A", "R105B", "R106A", "R106B"],
    "models": [{"model_id": name, "report_directory": rel(info["folder"]), "mechanism": info["mechanism"],
                "selected_update": info["protocol"]["checkpoint_lock"]["selected_update"], "selected_train_verified_from_report_metadata": info["train_verified"],
                "checkpoint_rule": info["protocol"]["checkpoint_lock"]["rule"],
                "online_bitwise_claim": False if name.startswith(("R99", "R102")) else "not assessed"} for name, info in models.items()],
    "input_files": sorted(READS),
    "unrecovered_focus": {"source_csv": rel(focus_source), "output_csv": "unrecovered_focus.csv", "model_id": "R102A",
                          "filter": "fixed36 TEST groups with winner_count > 0 and correct_count == 0",
                          "order": "winner_count descending, then problem and solver", "top_n": 5,
                          "category_basis": "R84A locked VAL historical list; not current prediction category"},
    "descriptive_sources": ["code/V4/runs/R94_pretrained_node_fusion/RESULTS.md", "code/V4/runs/R101_hint_pretraining/RESULTS.md",
                            "code/V4/runs/R99_global_competition_input/preparation/postprocess_recovery/report_view/RESULTS.md",
                            "code/V4/runs/R102_full_type_pretrained_fusion/preparation/postprocess_recovery/report_view/RESULTS.md"],
    "font": str(FONT),
}
(OUT / "inputs_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pdf": str(OUT / "RECENT_RESULTS.pdf"), "pages": 3, "models": len(models), "csv_rows": {
    "overall": len(overall), "metrics": len(metrics), "recovery": len(recovery), "references": len(references), "target_groups_test": len(groups), "matched_ab": len(pairs), "nss": len(nss), "unrecovered_focus": len(focus)},
    "selected_train_verified": {name: info["train_verified"] for name, info in models.items()}}, ensure_ascii=False, indent=2))
```

