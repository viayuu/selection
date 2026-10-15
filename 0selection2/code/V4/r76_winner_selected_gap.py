"""Paired winner/selected gaps from three existing R76 CSVs, without inference."""

import argparse
import html
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/r76_winner_selected_gap_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "runs/R76_R45A_single_reld_mt/missed_winner_probabilities"
OUT = ROOT / "winner_selected_gap_report"
FIG = OUT / "figures"
AGG = dict(
    count=("index", "size"), winner_p_mean_pct=("winner_p_pct", "mean"),
    selected_p_mean_pct=("selected_p_pct", "mean"), delta_p_mean_pp=("delta_p_pp", "mean"),
    delta_p_median_pp=("delta_p_pp", "median"), delta_p_p90_pp=("delta_p_pp", lambda x: x.quantile(.9)),
    cost_gap_mean_pct=("native_cost_gap_pct", "mean"), cost_gap_median_pct=("native_cost_gap_pct", "median"),
    cost_gap_p90_pct=("native_cost_gap_pct", lambda x: x.quantile(.9)),
    actual_regret_mean_pct=("actual_regret_pct", "mean"),
    full18_regret_contribution_pp=("full18_regret_contribution_pp", "sum"),
    rank_distance_mean=("rank_distance", "mean"),
)


def save(frame, name):
    frame.to_csv(OUT / name, index=False, float_format="%.17g")


def read(name):
    return pd.read_csv(OUT / name, float_precision="round_trip")


def configure_plot():
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#bfc7ce",
                         "text.color": "#17252b", "axes.labelcolor": "#17252b"})


def paired_plot(frame, title, filename, pair=False):
    frame = frame.sort_values("full18_regret_contribution_pp", ascending=False)
    height = max(5.4, 2.5 + .53 * len(frame))
    fig, (prob, cost) = plt.subplots(1, 2, figsize=(14, height), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=.35 if pair else .18, right=.96, top=.82 if len(frame) < 8 else .88,
                        bottom=.21 if len(frame) < 8 else .11, wspace=.25)
    fig.suptitle(title, fontsize=15, y=.975)
    if frame.empty:
        prob.text(.5, .5, "此范围没有错例", ha="center", transform=prob.transAxes)
        cost.set_axis_off()
    else:
        y = np.arange(len(frame))
        labels = [f"{r.problem} / {r.true_winner} → {r.selected_solver}  (n={r.count:,})" if pair
                  else f"{r.true_winner}  (n={r.count:,})" for r in frame.itertuples()]
        for i, row in enumerate(frame.itertuples()):
            prob.plot([row.winner_p_mean_pct, row.selected_p_mean_pct], [i, i], color="#c0c8cb", lw=3, zorder=1)
            prob.text(row.winner_p_mean_pct, i - .18, f"{row.winner_p_mean_pct:.1f}", color="#167c69", fontsize=10, ha="center")
            prob.text(row.selected_p_mean_pct, i + .28, f"{row.selected_p_mean_pct:.1f}", color="#c84b43", fontsize=10, ha="center")
            prob.text(1.025, i, f"{row.delta_p_mean_pp:+.1f}pp", transform=prob.get_yaxis_transform(),
                      fontsize=10, ha="left", va="center", color="#273842", clip_on=False)
        prob.text(1.025, 1.045, "Δp均值", transform=prob.transAxes, fontsize=10, ha="left", clip_on=False)
        prob.scatter(frame.winner_p_mean_pct, y, color="#167c69", s=45, label="真实winner平均概率", zorder=3)
        prob.scatter(frame.selected_p_mean_pct, y, color="#c84b43", s=45, label="同一错例实际selected平均概率", zorder=3)
        prob.set_yticks(y, labels, fontsize=11 if pair else 13)
        prob.set_xlim(-2, 102)
        prob.set_ylim(len(frame) - .5, -.55)
        prob.set_xlabel("概率 (%)，未校准；端点差为 Δp (pp)")
        prob.grid(axis="x", color="#e5eaed", lw=.7)
        prob.set_axisbelow(True)
        prob.legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, fontsize=10)
        cost.barh(y, frame.cost_gap_mean_pct, height=.43, color="#397eae", label="误选多花成本均值")
        cost.scatter(frame.cost_gap_median_pct, y, marker="|", s=150, color="#1e2930", linewidths=2, label="误选多花成本中位数")
        for i, row in enumerate(frame.itertuples()):
            cost.annotate(f"{row.cost_gap_mean_pct:.3f}%", (row.cost_gap_mean_pct, i), xytext=(5, 0),
                          textcoords="offset points", va="center", fontsize=11)
        maximum = max(.2, float(frame.cost_gap_mean_pct.max()), float(frame.cost_gap_median_pct.max()))
        cost.set_xlim(min(0., float(frame.cost_gap_mean_pct.min())) * 1.2, maximum * 1.4 + .12)
        cost.set_ylim(len(frame) - .5, -.55)
        cost.set_yticks([])
        cost.set_xlabel("误选多花成本 (%)\n相对真实winner成本增加", fontsize=10)
        cost.legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, fontsize=10)
        cost.grid(axis="x", color="#e5eaed", lw=.7)
        cost.set_axisbelow(True)
    fig.text(.02, .018, "每行概率双方来自完全相同的错例子集；按全18题等权actual regret贡献排序，不跨题相加raw成本。", fontsize=10)
    fig.savefig(FIG / filename, dpi=170, facecolor="white")
    plt.close(fig)


def scope_stats(frame, name):
    row = dict(scope=name, count=len(frame))
    for prefix, column in (("delta_p", "delta_p_pp"), ("cost_gap", "native_cost_gap_pct")):
        for label, value in (("mean", frame[column].mean()), ("median", frame[column].median()),
                             ("p90", frame[column].quantile(.9))):
            row[prefix + "_" + label] = float(value)
    row.update(winner_p_mean_pct=float(frame.winner_p_pct.mean()), selected_p_mean_pct=float(frame.selected_p_pct.mean()),
               actual_regret_mean_pct=float(frame.actual_regret_pct.mean()),
               full18_regret_contribution_pp=float(frame.full18_regret_contribution_pp.sum()),
               rank2_pct=float(100 * frame.winner_rank.eq(2).mean()),
               rank3_pct=float(100 * frame.winner_rank.eq(3).mean()),
               rank_gt3_pct=float(100 * frame.winner_rank.gt(3).mean()),
               selected_native_cost_ties=int(frame.selected_native_cost_tie.sum()),
               native_nonminimum=int(frame.native_nonminimum.sum()), negative_cost_gaps=int(frame.native_cost_gap_pct.lt(0).sum()))
    return row


def bucket_table(frame, column, edges, labels, scope, total_contribution, exact_zero=False):
    values = frame[column]
    records = []
    categories = [("negative (<0)", values.lt(0))]
    if exact_zero:
        categories.append(("exact=0", values.eq(0)))
    for index, label in enumerate(labels):
        low, high = edges[index], edges[index + 1]
        mask = (values.ge(low) if index == 0 and not exact_zero else values.gt(low)) & values.le(high)
        categories.append((label, mask))
    categories.append((f">{edges[-1]:g}", values.gt(edges[-1])))
    for label, mask in categories:
        contribution = float(frame.loc[mask, "full18_regret_contribution_pp"].sum())
        records.append(dict(scope=scope, bin=label, count=int(mask.sum()),
                            missed_share_pct=float(100 * mask.mean()),
                            full18_regret_contribution_pp=contribution,
                            full18_regret_share_pct=100 * contribution / total_contribution if total_contribution else 0.))
    return pd.DataFrame(records)


def summary(samples, winners, pairs, scopes, probability_bins, cost_bins, problems, example):
    text = ["# R76 真实winner与实际selected：成对差距", "",
            "仅描述已有TEST错例，不调参、不训练、不推理。默认原52个zero-with-winner组；概率未经校准，不是实际胜率。每个selected概率取该例实际选中的方法，同一子集严格配对，不借用其它样本的BQ/T2T等全组均值。", "",
            "## 先看数字", "",
            "| 范围 | 错例数 | Δp均值/中位/P90 (pp) | literal成本差均值/中位/P90 (%) | 全18actual regret贡献 (pp) |",
            "| --- | ---: | --- | --- | ---: |"]
    for r in scopes.itertuples():
        text.append(f"| {'全部错例' if r.scope == 'all' else '原zero组'} | {r.count:,} | {r.delta_p_mean:.3f}/{r.delta_p_median:.3f}/{r.delta_p_p90:.3f} | {r.cost_gap_mean:.4f}/{r.cost_gap_median:.4f}/{r.cost_gap_p90:.4f} | {r.full18_regret_contribution_pp:.9f} |")
    text.extend(["", "Δp=100×(p_selected−p_nativewinner)，排名距离=winner_rank−1；literal成本差=100×(c_selected−c_nativewinner)/c_nativewinner；actual regret=100×(c_selected−c_oracle)/c_oracle。两种成本口径分别保留，不改native身份。", "",
                 f"分母来自problem_summary：{len(problems)}题，各题{int(problems.instances.min()):,}例，共{int(problems.instances.sum()):,}例。每例贡献=actual_regret_pct/(18×本题全体N)，再加总；**不是**错例actual regret均值，**不是**对各题raw成本求和。各桶错例占比以该scope错例数为分母，贡献及贡献占比仍用全18口径。", "",
                 "## TSP / LEHD：同一批实际漏选", ""])
    lehd = winners[(winners.problem == "TSP") & (winners.true_winner == "LEHD")].iloc[0]
    text.extend([f"{int(lehd['count']):,}个LEHD真实获胜错例：native平均p **{lehd.winner_p_mean_pct:.4f}%**，实际selected平均p **{lehd.selected_p_mean_pct:.4f}%**；Δp均值/中位/P90 **{lehd.delta_p_mean_pp:.4f}/{lehd.delta_p_median_pp:.4f}/{lehd.delta_p_p90_pp:.4f}pp**。literal成本差均值/中位/P90 **{lehd.cost_gap_mean_pct:.6f}/{lehd.cost_gap_median_pct:.6f}/{lehd.cost_gap_p90_pct:.6f}%**。", "",
                 "| 实际selected | 次数 | native均p (%) | selected均p (%) | Δp均值 (pp) | literal成本差均值 (%) |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for r in pairs[(pairs.problem == "TSP") & (pairs.true_winner == "LEHD")].itertuples():
        text.append(f"| {r.selected_solver} | {r.count:,} | {r.winner_p_mean_pct:.4f} | {r.selected_p_mean_pct:.4f} | {r.delta_p_mean_pp:.4f} | {r.cost_gap_mean_pct:.6f} |")
    text.extend(["", "## 最主要原zero成对错误", "",
                 "| 问题 | native → 实际selected | 次数 | 均p native/selected (%) | Δp均值 (pp) | 成本差均值 (%) | 全18贡献 (pp) |",
                 "| --- | --- | ---: | --- | ---: | ---: | ---: |"])
    for r in pairs[pairs.zero_group].head(8).itertuples():
        text.append(f"| {r.problem} | {r.true_winner} → {r.selected_solver} | {r.count:,} | {r.winner_p_mean_pct:.2f}/{r.selected_p_mean_pct:.2f} | {r.delta_p_mean_pp:.2f} | {r.cost_gap_mean_pct:.4f} | {r.full18_regret_contribution_pp:.6f} |")
    text.extend(["", "## 分桶与排名", "",
                 "概率差固定[0,1,2,5,10,20,40,100]pp，首桶含0，后续右闭；成本差负值、exact=0、(0,.1]、(.1,.5]、(.5,1]、(1,2]、(2,5]、>5%分开。完整桶计数、错例占比及全18贡献见两个bins CSV和离线HTML。", "",
                 "| 范围 | winner排第2 | 第3 | >3 | selected/native成本并列 | native成本非最小 | literal负差 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in scopes.itertuples():
        text.append(f"| {r.scope} | {r.rank2_pct:.2f}% | {r.rank3_pct:.2f}% | {r.rank_gt3_pct:.2f}% | {r.selected_native_cost_ties} | {r.native_nonminimum} | {r.negative_cost_gaps} |")
    text.extend(["", "并列仅指该错例selected成本与native成本精确相等；未读取全池成本，不能据此枚举其它候选的并列。异常行保留在cost_exceptions.csv，不改native标签。", "",
                 "## 最少核对与文件", "",
                 f"条数21426、zero条数14077及52组与原CSV一致；两个scope的分桶次数/贡献合计一致；仅检查给定TSP index1算例：Δp={example.delta_p_pp:.12f}pp，排名距离={int(example.rank_distance)}，literal成本差={example.native_cost_gap_pct:.12f}%，actual regret={example.actual_regret_pct:.12f}%。未重复概率/校准分析或扩展来源审计。", "",
                 "- sample_gaps.csv：逐例成对差值、两种成本口径和全体分母贡献。",
                 "- by_winner.csv / by_pair.csv：紧凑聚合，zero_group明确区分原zero范围与其余错例。",
                 "- scope_summary.csv / probability_gap_bins.csv / cost_gap_bins.csv / cost_exceptions.csv：全局/zero分位数、分桶和成本例外。",
                 "- figures/global_top18_zero_pairs.png、figures/TSP_zero_winners.png：优先两图；共18题小图及index.html由finish阶段从聚合CSV生成。",
                 "- 输入仅missed_samples.csv、method_summary_test.csv、problem_summary_test.csv；无NPZ、训练、GPU、新infer、Git或goal操作。", ""])
    (OUT / "SUMMARY.md").write_text("\n".join(text), encoding="utf-8")


def core():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "missed_samples.csv", float_precision="round_trip")
    problems = pd.read_csv(ROOT / "problem_summary_test.csv", float_precision="round_trip")
    methods = pd.read_csv(ROOT / "method_summary_test.csv", usecols=["problem", "solver", "true_winner_count", "selected_count", "zero_selected_with_winner"])
    samples = raw[["problem", "index", "nodes", "true_winner", "selected_solver"]].copy()
    samples["zero_group"] = raw.zero_selected_winner
    samples["winner_p_pct"], samples["selected_p_pct"] = raw.winner_probability * 100, raw.selected_probability * 100
    samples["delta_p_pp"] = 100 * (raw.selected_probability - raw.winner_probability)
    samples["winner_rank"], samples["rank_distance"] = raw.winner_rank, raw.winner_rank - 1
    for column in ("winner_cost", "selected_cost", "oracle_cost"):
        samples[column] = raw[column]
    samples["native_cost_gap_pct"] = 100 * (raw.selected_cost - raw.winner_cost) / raw.winner_cost
    samples["actual_regret_pct"] = 100 * (raw.selected_cost - raw.oracle_cost) / raw.oracle_cost
    counts = problems.set_index("problem").instances
    samples["full18_regret_contribution_pp"] = samples.actual_regret_pct / (len(problems) * samples.problem.map(counts))
    samples["selected_native_cost_tie"] = raw.selected_cost.eq(raw.winner_cost)
    samples["native_nonminimum"] = raw.winner_cost.gt(raw.oracle_cost)
    group = ["problem", "true_winner", "zero_group"]
    winners = samples.groupby(group, sort=False).agg(**AGG).reset_index().sort_values("full18_regret_contribution_pp", ascending=False)
    pairs = samples.groupby(group + ["selected_solver"], sort=False).agg(**AGG).reset_index().sort_values("full18_regret_contribution_pp", ascending=False)
    scopes, prob_bins, cost_bins = [], [], []
    full_contribution = float(samples.full18_regret_contribution_pp.sum())
    for name, frame in (("all", samples), ("zero", samples[samples.zero_group])):
        scopes.append(scope_stats(frame, name))
        prob_bins.append(bucket_table(frame, "delta_p_pp", [0, 1, 2, 5, 10, 20, 40, 100],
                                     ["[0,1]", "(1,2]", "(2,5]", "(5,10]", "(10,20]", "(20,40]", "(40,100]"], name, full_contribution))
        cost_bins.append(bucket_table(frame, "native_cost_gap_pct", [0, .1, .5, 1, 2, 5],
                                     ["(0,.1]", "(.1,.5]", "(.5,1]", "(1,2]", "(2,5]"], name, full_contribution, exact_zero=True))
    scopes, prob_bins, cost_bins = pd.DataFrame(scopes), pd.concat(prob_bins), pd.concat(cost_bins)
    assert len(samples) == int(problems.missed_count.sum()) == 21426
    assert int(samples.zero_group.sum()) == int(problems.zero_winner_count.sum()) == 14077
    assert len(winners[winners.zero_group]) == int(methods.zero_selected_with_winner.sum()) == 52
    for table in (prob_bins, cost_bins):
        for r in scopes.itertuples():
            subset = table[table.scope == r.scope]
            assert int(subset["count"].sum()) == r.count
            assert np.isclose(subset.full18_regret_contribution_pp.sum(), r.full18_regret_contribution_pp, rtol=0, atol=1e-12)
    example = samples[(samples.problem == "TSP") & samples["index"].eq(1)].iloc[0]
    assert np.isclose(example.delta_p_pp, 16.15767478942871, rtol=0, atol=1e-12)
    assert example.rank_distance == 3 and np.isclose(example.native_cost_gap_pct, .20079361960522374, rtol=0, atol=1e-12)
    for frame, name in ((samples, "sample_gaps.csv"), (winners, "by_winner.csv"), (pairs, "by_pair.csv"),
                        (scopes, "scope_summary.csv"), (prob_bins, "probability_gap_bins.csv"), (cost_bins, "cost_gap_bins.csv")):
        save(frame, name)
    exceptions = samples[samples.selected_native_cost_tie | samples.native_nonminimum | samples.native_cost_gap_pct.lt(0)]
    save(exceptions, "cost_exceptions.csv")
    summary(samples, winners, pairs, scopes, prob_bins, cost_bins, problems, example)
    configure_plot()
    paired_plot(pairs[pairs.zero_group].head(18), "R76 原零选择组：贡献最大的18个真实winner → 实际selected", "global_top18_zero_pairs.png", pair=True)
    paired_plot(winners[(winners.problem == "TSP") & winners.zero_group], "TSP：真实winner与实际selected的成对差距（原零选择组）", "TSP_zero_winners.png")
    lehd = winners[(winners.problem == "TSP") & (winners.true_winner == "LEHD")]
    print(json.dumps(dict(scopes=scopes.to_dict("records"), tsp_lehd=lehd.to_dict("records")), ensure_ascii=False, indent=2))


def finish():
    winners, pairs, scopes = read("by_winner.csv"), read("by_pair.csv"), read("scope_summary.csv")
    probability_bins, cost_bins = read("probability_gap_bins.csv"), read("cost_gap_bins.csv")
    problems = winners.problem.drop_duplicates().tolist()
    configure_plot()
    for problem in problems:
        filename = problem + "_zero_winners.png"
        paired_plot(winners[(winners.problem == problem) & winners.zero_group],
                    f"{problem}：真实winner与实际selected的成对差距（原零选择组）", filename)
    paired_plot(pairs[pairs.zero_group].head(18), "R76 原零选择组：贡献最大的18个真实winner → 实际selected", "global_top18_zero_pairs.png", pair=True)
    paired_plot(pairs.head(18), "R76 全部错例：贡献最大的18个真实winner → 实际selected", "global_top18_all_pairs.png", pair=True)
    payload = dict(pairs=json.loads(pairs.to_json(orient="records", force_ascii=False, double_precision=15)))
    cards = "".join(f"<tr><td>{'全部错例' if r.scope == 'all' else '原zero组'}</td><td>{r.count:,}</td><td>{r.delta_p_mean:.3f} / {r.delta_p_median:.3f} / {r.delta_p_p90:.3f}</td><td>{r.cost_gap_mean:.4f} / {r.cost_gap_median:.4f} / {r.cost_gap_p90:.4f}</td><td>{r.full18_regret_contribution_pp:.6f}</td></tr>" for r in scopes.itertuples())
    options = "".join(f'<option value="{html.escape(p)}">{html.escape(p)}</option>' for p in problems)
    gallery = " ".join(f'<a href="figures/{p}_zero_winners.png">{p}</a>' for p in problems)
    bins_html = []
    for title, table, unit in (("概率差固定分桶", probability_bins, "pp"), ("误选多花成本固定分桶", cost_bins, "%")):
        body = "".join(f"<tr><td>{r.scope}</td><td>{html.escape(r.bin)} {unit}</td><td>{r.count:,}</td><td>{r.missed_share_pct:.2f}%</td><td>{r.full18_regret_contribution_pp:.6f}</td></tr>" for r in table.itertuples())
        bins_html.append(f"<section><h2>{title}</h2><div class=scroll><table><thead><tr><th>范围</th><th>桶</th><th>错例数</th><th>范围内错例占比</th><th>全18 regret贡献 (pp)</th></tr></thead><tbody>{body}</tbody></table></div></section>")
    document = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R76 成对差距报告</title>
<style>*{box-sizing:border-box;letter-spacing:0}body{margin:0;color:#17252b;background:#f7f9fa;font:14px/1.6 "Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:1480px;margin:auto;padding:24px}header,section{padding:14px 0 24px;border-bottom:1px solid #d7dfe2}h1{font-size:26px;line-height:1.35;margin:0 0 8px}h2{font-size:18px;margin:12px 0}p{margin:8px 0;color:#45545c}a{color:#176b74}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border-bottom:1px solid #e1e7e9;padding:9px 11px;text-align:right;white-space:nowrap}th{background:#eef3f5;position:sticky;top:0}th:first-child,td:first-child,td:nth-child(2){text-align:left}figure{margin:14px 0}img{display:block;width:100%;height:auto;background:white}figcaption{font-size:12px;color:#51616b}.scroll{overflow:auto}.pairs{max-height:620px}select{font:inherit;padding:6px;border:1px solid #a9b9bf;background:white;border-radius:3px;max-width:100%}.filters{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:12px 0}label{display:flex;align-items:center;gap:8px}.tasklinks{display:flex;flex-wrap:wrap;gap:10px 20px}.note{border-left:3px solid #397eae;padding-left:12px}@media(max-width:600px){main{padding:14px}h1{font-size:23px}th,td{padding:8px;font-size:12px}}</style>
<main><header><h1>R76 真实winner与实际selected</h1><p>已保存TEST错例 · 默认原52个零选择组 · 概率未经校准，不是实际胜率</p><p class=note>每个概率端点来自同一批错例实际选中的方法。误选多花成本表示相对真实winner成本增加；全池actual regret仍以Oracle成本为基准。全18贡献使用每题全部3,895例的分母，不用错例均值冒充整体regret。</p><p><a href="SUMMARY.md">中文摘要</a> · <a href="by_pair.csv">成对CSV</a> · <a href="sample_gaps.csv">逐例CSV</a> · <a href="cost_exceptions.csv">并列/非最小native例外</a></p></header>
<section><h2>全局与原zero范围</h2><div class=scroll><table><thead><tr><th>范围</th><th>错例数</th><th>Δp 均值/中位/P90 (pp)</th><th>误选多花成本 均值/中位/P90 (%)</th><th>全18贡献 (pp)</th></tr></thead><tbody>__CARDS__</tbody></table></div></section>
<section><h2>同一子集的概率与成本</h2><figure><img id=globalplot src="figures/global_top18_zero_pairs.png" alt="按全18相对regret贡献排序的18个原zero成对错误"><figcaption>概率：绿色为native winner，红色为每例实际selected，右侧标注平均Δp。成本：蓝条为误选多花成本均值，黑刻度为中位数。排序为全18actual regret贡献。</figcaption></figure><figure><img src="figures/TSP_zero_winners.png" alt="TSP五个原zero组的概率与成本成对差距"><figcaption>TSP：固定原5个zero-with-winner组，同一错例子集配对。</figcaption></figure><nav class=tasklinks>__GALLERY__</nav></section>
<section><h2>实际选择去向</h2><div class=filters><label>范围<select id=scope><option value=zero>原零选择组</option><option value=all>全部错例</option></select></label><label>问题<select id=problem><option value=all>全部问题</option>__OPTIONS__</select></label><span id=count></span></div><div class="scroll pairs"><table><thead><tr><th>问题</th><th>native → 实际selected</th><th>次数</th><th>均p native → selected (%)</th><th>Δp 均值/中位/P90 (pp)</th><th>误选多花成本 均值/中位 (%)</th><th>全18贡献 (pp)</th></tr></thead><tbody id=pairbody></tbody></table></div></section>
__BINS__<footer><p>误选多花成本=100×(selected成本−native winner成本)/native winner成本；全池actual regret=100×(selected成本−Oracle成本)/Oracle成本。原输入仅三份CSV；成本按round-trip FP64解析，原生标签不改。成本并列仅指保存的selected/native成本相等，负值与非最小native另列。错例的条件均值与全体分母贡献分开；本报告不作训练或规则依据。</p></footer></main>
<script type="application/json" id=data>__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),scope=document.getElementById('scope'),problem=document.getElementById('problem');
function render(){const rows=data.pairs.filter(r=>(scope.value==='all'||r.zero_group)&&(problem.value==='all'||r.problem===problem.value));const body=document.getElementById('pairbody');body.replaceChildren();for(const r of rows){const tr=document.createElement('tr');const values=[r.problem,r.true_winner+' → '+r.selected_solver,r.count.toLocaleString(),r.winner_p_mean_pct.toFixed(2)+' → '+r.selected_p_mean_pct.toFixed(2),[r.delta_p_mean_pp,r.delta_p_median_pp,r.delta_p_p90_pp].map(x=>x.toFixed(2)).join(' / '),r.cost_gap_mean_pct.toFixed(4)+' / '+r.cost_gap_median_pct.toFixed(4),r.full18_regret_contribution_pp.toFixed(6)];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}document.getElementById('count').textContent=rows.length+' 组 · '+rows.reduce((a,r)=>a+r.count,0).toLocaleString()+' 错例';const image=document.getElementById('globalplot');image.src='figures/global_top18_'+(scope.value==='zero'?'zero':'all')+'_pairs.png';image.alt=scope.value==='zero'?'原zero成对差距':'全部错例成对差距'}scope.addEventListener('change',render);problem.addEventListener('change',render);render();
</script></html>'''
    for marker, value in (("__CARDS__", cards), ("__GALLERY__", gallery), ("__OPTIONS__", options),
                          ("__BINS__", "".join(bins_html)), ("__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))):
        document = document.replace(marker, value)
    (OUT / "index.html").write_text(document, encoding="utf-8")
    print(f"Offline HTML: {OUT / 'index.html'}; {len(problems)} task figures; existing core aggregates reused.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "finish", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("core", "all"):
        core()
    if args.stage in ("finish", "all"):
        finish()


if __name__ == "__main__":
    main()
