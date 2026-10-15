"""Read historical JSON/CSV only; write a three-page retrospective, never run models."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
FONT = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
HANDOFF = OUT / "rounds001_003_handoff.json"
TITLE = "R78 | \u771f\u5b9e\u4f1a\u8d62\u4f46\u6a21\u578b\u96f6\u9009\u62e9\uff1a\u516d\u8f6e\u5b9e\u9a8c\u56de\u987e"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(record):
    record = record.get("macro", record)
    return {"top1_pct": record["macro_top1"] * 100,
            "regret_pct": record["macro_actual_regret_pct"], "mean_cost": record["macro_mean_cost"]}


def changes(rows):
    rows = [r for r in rows if r.get("split", "val") == "val"]
    return {key: sum(int(r[key]) for r in rows)
            for key in ("changed_predictions", "corrected", "harmed", "wrong_to_wrong")}


def recent_rounds():
    a, b, c = [read_json(ROOT / f"round00{i}/results.json") for i in (4, 5, 6)]
    receipt4 = read_json(ROOT / "round004/final_receipt.json")
    receipt6 = read_json(ROOT / "round006/execution_receipt.json")
    change6 = read_json(ROOT / "round006/change_summary.json")["val"]
    records = []
    for number, result, base, fitted, cold, budget, mechanism, short in [
        (4, a, a["reports"]["val"]["B"], a["reports"]["val"]["fitted"],
         [a["cold_methods"]["val"][key] for key in ("B", "fitted")],
         "24,840 CPU updates; 1,380/task; batch 1,024; fit 21.50s",
         "Frozen B cost head: g=1-p+Dp; FP64 relative-cost regression + fixed unit prior; no backbone update.",
         "Frozen B\nprobability-to-cost head"),
        (5, b, b["B"], b["ensemble"], [b["cold_methods"][key] for key in ("B", "ensemble")],
         "0 updates; one cached VAL screen, 70,110 rows; CPU 0.290s",
         "Fixed 0.5/0.5 average of A/B T=1 probabilities; original preprocessing; no training or weight search.",
         "Fixed A/B\nprobability average"),
        (6, c, c["B"]["val"], c["fitted"]["val"], [],
         "14,904 GPU updates; 828/task; batch 1,024; single 4090; 4,858.43s",
         "Original two-view loss + 0.10 cost-profile neighborhood representation loss; unique final14904 endpoint.",
         "Cost-profile\nneighborhood loss"),
    ]:
        record = dict(round=f"{number:03}", baseline="R77B ep28/u34776", mechanism=mechanism,
                      mechanism_short=short, actual_budget=budget, successful_updates={4: 24840, 5: 0, 6: 14904}[number],
                      baseline_val=metrics(base), candidate_val=metrics(fitted),
                      zero_pairs_base=cold[0]["zero_with_winner_pairs"] if cold else change6["zero_winner_method_pairs_B"],
                      zero_pairs_candidate=cold[1]["zero_with_winner_pairs"] if cold else change6["zero_winner_method_pairs_candidate"],
                      zero_winner_rows_base=cold[0]["zero_winner_count"] if cold else 2648,
                      zero_winner_rows_candidate=cold[1]["zero_winner_count"] if cold else None,
                      status="closed negative; all 3 gates failed", final_action="retain whole R77B", test_run=False,
                      metric_split="VAL", counts_split="VAL", counts_rows_per_task=3895,
                      source_results=str(ROOT / f"round00{number}/results.json"),
                      source_protocol=str(ROOT / f"round00{number}" / ("prepared.json" if number == 6 else "protocol.json")))
        if number in (4, 5):
            record.update(changes(read_csv(ROOT / f"round00{number}/decision_changes.csv")))
        else:
            record.update({k: change6[k] for k in ("corrected", "harmed", "wrong_to_wrong")})
            record["changed_predictions"] = change6["changed"]
            counts = read_csv(ROOT / "round006/native_method_counts.csv")
            record["zero_winner_rows_candidate"] = sum(int(r["winners"]) for r in counts
                if r["split"] == "val" and int(r["winners"]) > 0 and int(r["fitted_selected"]) == 0)
        records.append(record)
    records[0]["runtime_seconds"] = receipt4["TRAIN_fitting_elapsed_seconds"]
    records[1]["runtime_seconds"] = b["runtime_seconds"]
    records[2]["runtime_seconds"] = receipt6["elapsed_seconds"]
    records[2]["historical_control_val"] = metrics(c["historical_control_final14904_val"])
    return records


def load_records():
    early = read_json(HANDOFF)
    if isinstance(early, dict):
        early = early["rounds"]
    indexed = {int(r["round"]): r for r in early}
    records = []
    for number in (1, 2, 3):
        record = indexed.get(number, dict(round=f"{number:03}", baseline="pending handoff", mechanism="Pending first-three-round handoff",
            mechanism_short="Pending handoff", successful_updates=None, actual_budget="Pending handoff",
            baseline_val={}, candidate_val={}, status="inconclusive; convergence not certified" if number < 3 else "pending handoff",
            final_action="no accepted candidate", test_run=False, source_results="pending handoff", source_protocol="pending handoff"))
        record["round"] = f"{number:03}"
        if number < 3 and "inconclusive" not in record["status"].lower():
            raise ValueError("Rounds001/002 must retain their inconclusive status")
        for key in ("source_results", "source_protocol", "source_reference"):
            if key in record:
                record[key] = str((ROOT / record[key]).resolve())
        records.append(record)
    return records + recent_rounds()


def fmt(value, digits=6):
    return "not recorded" if value is None else f"{value:.{digits}f}"


def transition(record, key, digits=6):
    if record["round"] == "002":
        return "VAL not evaluated"
    if "matched_control_val" in record:
        return (f"B {fmt(record['baseline_val'].get(key), digits)}\n"
                f"C {fmt(record['matched_control_val'].get(key), digits)}\n"
                f"T {fmt(record['candidate_val'].get(key), digits)}")
    return f"{fmt(record['baseline_val'].get(key), digits)}\n-> {fmt(record['candidate_val'].get(key), digits)}"


def page(title, number):
    fig = plt.figure(figsize=(16.54, 11.69))
    fig.text(.045, .95, title, fontsize=21, va="top")
    fig.text(.045, .902, "Read-only history | original metrics and actual candidates, not retained fallback metrics", fontsize=11)
    fig.text(.045, .025, "No training / inference / experiment rerun / literature search / goal operation / Git commit", fontsize=9)
    fig.text(.955, .025, str(number), ha="right", fontsize=9)
    return fig


def table(fig, headers, rows, bbox, widths, font=10):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    t = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc="left", colLoc="center", bbox=bbox)
    t.auto_set_font_size(False)
    t.set_fontsize(font)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#d3dcdf")
        cell.set_linewidth(.5)
        cell.PAD = .07
        if r == 0:
            cell.set_facecolor("#e8eef0")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f7f7")


def make_pages(records):
    first = page(TITLE, 1)
    first.text(.045, .85, "\u7ed3\u8bba\uff1a\u516d\u8f6e\u5747\u672a\u4ea7\u751f\u901a\u8fc7\u5f53\u8f6e gate \u7684\u6539\u8fdb\uff1b\u5f53\u524d\u4fdd\u7559 R77B\uff0c\u603b\u76ee\u6807\u5c1a\u672a\u8fbe\u6210\u3002", fontsize=13)
    first.text(.045, .812, "\u4e0b\u8868\u4e3a\u6bcf\u8f6e\u539f\u57fa\u7ebf -> \u5b9e\u9645\u5019\u9009\uff1b\u57fa\u7ebf\u4e0d\u540c\uff0c\u4e0d\u662f\u540c\u6761\u4ef6\u6392\u884c\u699c\u3002Top1 \u4e0e regret \u5355\u4f4d\u5747\u4e3a %\u3002", fontsize=12)
    status = {"001": "inconclusive\n1/18 converged\ncost gate failed", "002": "inconclusive\n14/18 converged\nVAL not evaluated",
              "003": "closed negative\n0/6 endpoint gates\nfinite budget only"}
    rows = [[r["round"], r["baseline"].replace(" ep", "\nep"), r.get("mechanism_short", r["mechanism"]),
             "not recorded" if r["successful_updates"] is None else f"{r['successful_updates']:,}",
             transition(r, "top1_pct"), transition(r, "regret_pct"), transition(r, "mean_cost", 9),
             status.get(r["round"], r["status"].replace("; ", "\n")) + "\nTEST: " + ("yes" if r["test_run"] else "no")]
            for r in records]
    table(first, ["Round", "Original baseline", "Mechanism", "Actual\nupdates", "VAL Top1 (%)\nbase -> actual",
                  "VAL regret (%)\nbase -> actual", "VAL mean_cost\nbase -> actual", "Status / TEST"], rows,
          [.045, .295, .91, .48], [.05, .115, .185, .075, .12, .12, .145, .19], font=10)
    notes = [
        "001/002: inconclusive; \u672a\u8bc1\u660e\u5df2\u6536\u655b\u3002\u4e0d\u80fd\u5c06\u6709\u9650\u8ba1\u7b97\u7684\u5931\u8d25\u7b49\u540c\u4e8e\u673a\u5236\u88ab\u5426\u8bc1\u3002",
        "004/005/006 \u4e3b\u8868\u4f7f\u7528 fitted / ensemble / fitted\uff0c\u4e0d\u4f7f\u7528\u672a\u53d8\u7684 fallback_B \u6307\u6807\u3002",
             "VAL ALL \u4e3a 18 \u9898\u7b49\u6743\uff0c\u6bcf\u9898 3,895 \u4f8b\uff08\u5171 70,110\uff09\u3002R2 \u6ca1\u6709\u5019\u9009 VAL\uff1bR3 \u5217 B / control / treatment \u6700\u7ec8\u5b9e\u9645\u7aef\u70b9\u3002",
        "\u5173\u95ed\u67d0\u8f6e\u8bd5\u9a8c\u4e0d\u7b49\u4e8e\u957f\u671f\u76ee\u6807\u5b8c\u6210\u3002\u672c\u6b21\u4ec5\u56de\u987e\uff0c\u4e0d\u63d0\u51fa\u6216\u542f\u52a8\u65b0\u8bd5\u9a8c\u3002",
    ]
    for i, text in enumerate(notes):
        first.text(.045, .235 - i * .043, text, fontsize=11)

    second = page("\u9884\u7b97\u3001\u96f6\u9009\u62e9\u7ec4\u4e0e\u9010\u4f8b\u7ea0\u6b63", 2)
    rows = []
    for r in records:
        pair = lambda key: "not recorded" if r.get(key) is None else str(r[key])
        rows.append([r["round"], f"{r['counts_split']}\n{r['counts_rows_per_task']:,}/task",
                     pair("zero_pairs_base") + " -> " + pair("zero_pairs_candidate"),
                     pair("zero_winner_rows_base") + " -> " + pair("zero_winner_rows_candidate"),
                     pair("changed_predictions"), pair("corrected"), pair("harmed"), pair("wrong_to_wrong")])
    table(second, ["Round", "Count split\n18 tasks", "Zero/winner\nmethod pairs", "Winner rows in\ndynamic zero groups", "Changed", "Corrected", "Harmed", "Wrong->wrong"],
          rows, [.045, .635, .91, .22], [.055, .16, .15, .195, .11, .11, .11, .11], font=10.2)
    second.text(.045, .600, "zero-with-winner = native winner \u6570 > 0 \u4e14\u9009\u62e9\u6570 = 0\uff1b\u7ec4\u5408\u6570\u51cf\u5c11\u4e0d\u7b49\u4e8e\u539f\u6f0f\u9009\u884c\u88ab\u7ea0\u6b63\u3002R2 \u8ba1\u6570\u4e3a TRAIN\uff0c\u4e0d\u662f VAL\u3002", fontsize=10.8)
    second.text(.045, .565, "R3 \u8ba1\u6570\u4ec5\u4e3a +4968\uff08control/treatment \u5747 27\uff09\uff1b\u6700\u7ec8 fallback \u53d8\u5316\u8868\u7684\u5168\u96f6\u4e0d\u662f\u5b9e\u9645\u5019\u9009\u672a\u53d8\u3002\u7f3a\u5931\u8ba1\u6570\u4e0d\u8865\u9020\u3002", fontsize=10.8)
    for i, r in enumerate(records):
        second.text(.045, .515 - i * .031, f"{r['round']} | {r['actual_budget']}", fontsize=10.2)
    second.text(.045, .310, "R2 | TRAIN ONLY: 70,114/task, 18 tasks (1,262,052 total)", fontsize=10.8)
    train2 = records[1]
    train_rows = [[label, f"{train2[key]['top1_pct']:.6f}", f"{train2[key]['regret_pct']:.9f}", f"{train2[key]['mean_cost']:.9f}"]
                  for label, key in (("R76", "baseline_train"), ("Fitted", "candidate_train"))]
    table(second, ["TRAIN rule", "Top1 (%)", "regret (%)", "mean_cost"], train_rows,
          [.045, .175, .44, .115], [.16, .23, .26, .35], font=10.3)
    second.text(.045, .140, "R2 b* TRAIN reference: Top1=40.849505409%,", fontsize=10)
    second.text(.045, .110, "regret=2.826224389%; only 27 predictions differ; not deployed.", fontsize=10)
    second.text(.515, .310, "R3 | actual VAL endpoints (C / T); n=70,110 each", fontsize=10.8)
    endpoints = records[2]["val_endpoints"]
    endpoint_rows = []
    for update in (4968, 9936, 14904):
        control, treatment = [next(e for e in endpoints if e["update"] == update and e["arm"] == arm) for arm in ("control", "treatment")]
        endpoint_rows.append([f"+{update:,}", f"{control['top1_pct']:.6f}", f"{control['regret_pct']:.9f}",
                              f"{treatment['top1_pct']:.6f}", f"{treatment['regret_pct']:.9f}"])
    table(second, ["Update", "C Top1 (%)", "C regret (%)", "T Top1 (%)", "T regret (%)"], endpoint_rows,
          [.515, .135, .44, .155], [.13, .20, .235, .20, .235], font=9.1)
    second.text(.515, .100, "Both arms: every endpoint cost/regret worse than B; 0/6 passed.", fontsize=9.7)
    second.text(.045, .060, "005: 0.290s = cached processing, not deployment latency. Lower CE / fewer zero groups do not override cost gates.", fontsize=10.5)

    third = page("\u8fb9\u754c\u4e0e\u5df2\u6709\u5b8c\u6574\u7248\u672c\u53c2\u7167", 3)
    third.text(.045, .843, "R76 -> R77B | \u53ea\u5bf9\u5df2\u5b8c\u6210 TEST \u7684\u4e8b\u540e\u63cf\u8ff0\uff0c\u4e0d\u662f\u67d0\u8f6e R78 \u5e72\u9884\u7684\u6210\u7ee9", fontsize=14)
    table(third, ["Existing TEST comparison", "R76", "R77B"], [
        ["Zero-with-winner task-method groups", "52", "26"],
        ["Fixed 14,077 original missed winners: true hits", "0 / 14,077", "703 / 14,077 (4.9940%)"],
        ["TSP / LEHD: selected / native winner support", "0 / 637", "0 / 637"],
        ["ALL18 Top1 (%)", "69.439452", "71.225218"],
        ["ALL18 actual regret (%)", "0.464143599", "0.394014312"],
    ], [.045, .56, .91, .245], [.53, .20, .27], font=12)
    paragraphs = [
        "52 \u7c7b\u4e2d 26 \u7c7b\u51fa\u73b0\u975e\u96f6\u9009\u62e9\uff0c\u4e0d\u4ee3\u8868\u8fd9\u4e9b\u7c7b\u7684 winner \u5df2\u5168\u4fee\u590d\uff1b\u56fa\u5b9a\u539f\u96c6\u5408\u4ecd\u6f0f 13,374 \u4f8b\u3002",
        "R76 / R77B \u540c\u65f6\u6539\u53d8\u4e86\u8bad\u7ec3\u6279\u5927\u5c0f\u3001\u9884\u7b97\u7b49\u6761\u4ef6\uff0c\u4e0d\u80fd\u5f52\u56e0\u4e3a batch \u5355\u56e0\u7d20\uff1b\u4e5f\u4e0d\u8bc1\u660e R78 \u7684\u4fee\u590d\u6210\u529f\u3002",
        "004 \u51cf\u5c11 zero \u7ec4\u5408\u4f46 TRAIN / VAL \u6210\u672c\u4e0e Top1 \u5747\u53d8\u5dee\uff1b005 \u964d\u4f4e CE \u4f46\u4e09\u9879 gate \u5747\u5931\u8d25\u3002",
        "006 \u4ec5\u4e3a\u4e00\u4e2a\u79cd\u5b50\u7684\u6709\u9650\u8bd5\u9a8c\uff1b\u4e0e B \u6bd4\u8f83\u540c\u65f6\u5305\u542b\u989d\u5916\u8bad\u7ec3\u4e0e\u65b0\u76ee\u6807\uff0c\u4e0d\u4f5c\u552f\u4e00\u673a\u5236\u5f52\u56e0\u3002",
    ]
    for i, text in enumerate(paragraphs):
        third.text(.045, .515 - i * .047, text, fontsize=11.3)
    control = records[-1]["historical_control_val"]
    third.text(.045, .300, "006 \u53e6\u6709\u5386\u53f2 final14904 control\uff08\u4e0d\u662f fallback\uff09: "
               f"VAL Top1={control['top1_pct']:.6f}%, regret={control['regret_pct']:.9f}%, cost={control['mean_cost']:.9f}.", fontsize=11)
    third.text(.045, .260, "\u8be5 control \u6765\u81ea R3\uff0c\u4e0d\u662f R6 \u65b0\u9c9c\u968f\u673a\u5bf9\u7167\uff1bTRAIN \u6307\u6807\u672a\u8bb0\u5f55\u3002\u5404\u8f6e\u590d\u7528 VAL\uff0c\u4e0d\u662f\u72ec\u7acb\u786e\u8ba4\u3002", fontsize=11)
    third.text(.045, .200, "\u6765\u6e90\uff1around001-006 \u539f\u7ed3\u679c/\u534f\u8bae\u3001GOAL.md \u5f53\u524d\u7ed3\u6848\u6458\u8981\u3001final_delivery/USER_TASKS_SUMMARY.md\u3002", fontsize=10.7)
    third.text(.045, .160, "\u5df2\u6709 TEST \u53c2\u7167\uff1afinal_delivery/R76_R77_TEST_COLD_COUNTS.md\u3002\u53ef\u8ffd\u6eaf\u6765\u6e90\u8def\u5f84\u4e0e\u5b8c\u6574\u7cbe\u5ea6\u5728\u540c\u76ee\u5f55 CSV / Markdown\u3002", fontsize=10.7)
    third.text(.045, .110, "\u6240\u6709\u6570\u5b57\u4ec5\u6574\u7406\u73b0\u6210\u7ed3\u679c\uff0c\u4e0d\u52a0\u8f7d\u6a21\u578b\u6216\u9884\u6d4b\u77e9\u9635\uff0c\u4e0d\u66f4\u6539\u539f\u62a5\u544a\u3001goal\u3001\u9884\u7b97\u6216\u89c4\u5219\u3002", fontsize=11)
    return [first, second, third]


def main():
    records = load_records()
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
                         "pdf.fonttype": 42, "axes.unicode_minus": False, "text.color": "#242d31"})
    rows = []
    for r in records:
        row = {key: json.dumps(value) if isinstance(value, list) else value
               for key, value in r.items() if not isinstance(value, dict)}
        row.pop("mechanism_short", None)
        for rule in ("baseline_val", "candidate_val", "matched_control_val", "baseline_train", "candidate_train"):
            row.update({rule + "_" + key: r.get(rule, {}).get(key) for key in ("top1_pct", "regret_pct", "mean_cost")})
        row.update(val_rows_per_task=3895, val_rows_total=70110, train_rows_per_task=70114, train_rows_total=1262052)
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "round_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with PdfPages(OUT / "ZERO_SELECTION_RETROSPECTIVE.pdf", metadata={"Title": TITLE, "Subject": "Existing result retrospective only"}) as pdf:
        for fig in make_pages(records):
            pdf.savefig(fig)
            plt.close(fig)
    text = ["# " + TITLE, "", "[PDF](ZERO_SELECTION_RETROSPECTIVE.pdf) | [CSV](round_summary.csv)", "",
            "\u5404\u8f6e\u57fa\u7ebf\u4e0d\u540c\uff0c\u975e\u540c\u6761\u4ef6\u6392\u884c\u699c\u3002\u5019\u9009\u5747\u6307\u5b9e\u9645\u7ed3\u679c\uff0c\u4e0d\u662f fallback\u3002",
            "", "| Round | Baseline | VAL Top1 baseline -> actual (%) | VAL regret baseline -> actual (%) | VAL cost baseline -> actual | Status |", "|---|---|---:|---:|---:|---|"]
    for r in records:
        vals = [transition(r, key, 9 if key == "mean_cost" else 6).replace("\n", " ") for key in ("top1_pct", "regret_pct", "mean_cost")]
        text.append(f"| {r['round']} | {r['baseline']} | {' | '.join(vals)} | {r['status']} |")
    for r in records:
        text.extend(["", "## Round " + r["round"], r["mechanism"], r["actual_budget"],
                     f"{r['counts_split']} ({r['counts_rows_per_task']:,}/task, 18 tasks) zero/winner pairs: {r.get('zero_pairs_base', 'not recorded')} -> {r.get('zero_pairs_candidate', 'not recorded')}; "
                     f"corrected/harmed/wrong-to-wrong: {r.get('corrected', 'not recorded')}/{r.get('harmed', 'not recorded')}/{r.get('wrong_to_wrong', 'not recorded')}.",
                     f"TEST: {r['test_run']}; final: {r['final_action']}.",
                     f"[Results]({r['source_results']}) | [Protocol]({r['source_protocol']})"])
        if r["round"] == "002":
            text.extend(["", "TRAIN only; candidate VAL was not evaluated because 18/18 convergence was required.",
                         f"TRAIN R76 -> fitted Top1: {r['baseline_train']['top1_pct']:.6f}% -> {r['candidate_train']['top1_pct']:.6f}%; "
                         f"regret: {r['baseline_train']['regret_pct']:.9f}% -> {r['candidate_train']['regret_pct']:.9f}%; "
                         f"mean_cost: {r['baseline_train']['mean_cost']:.9f} -> {r['candidate_train']['mean_cost']:.9f}.",
                         "Analytical b* reference: TRAIN Top1=40.849505409%, regret=2.826224389%; 27 prediction disagreements with fitted; not deployed."])
        if r["round"] == "003":
            text.extend(["", "Actual VAL endpoints; every cost/regret worse than locked B, 0/6 gates passed. Final fallback zeros are not actual trajectory changes.",
                         "| Update | Arm | VAL Top1 (%) | VAL regret (%) | VAL mean_cost |", "|---:|---|---:|---:|---:|"])
            for e in r["val_endpoints"]:
                text.append(f"| {e['update']} | {e['arm']} | {e['top1_pct']:.9f} | {e['regret_pct']:.9f} | {e['mean_cost']:.12f} |")
    text.extend(["", "## Existing R76 / R77B TEST reference", "52 -> 26 zero-selected winner-method groups; only 703 / 14,077 (4.9940%) fixed original missed-winner rows truly corrected, with 13,374 still missed. TSP/LEHD remains 0/637.",
                 "R76/R77 changed batch and training conditions jointly; this is not a batch-only causal result or an accepted R78 intervention.",
                 f"[Existing source]({ROOT / 'final_delivery/R76_R77_TEST_COLD_COUNTS.md'})", "",
                 "Rounds001/002 remain inconclusive, with convergence not certified. No accepted R78 candidate; retain B; the long-term goal is unresolved.",
                 "Read historical JSON/CSV/Markdown only. No model/NPZ loading, training, inference, experiment reruns, literature search, goal changes, report edits or commits.", ""])
    (OUT / "RETROSPECTIVE.md").write_text("\n".join(text), encoding="utf-8")
    (OUT / "historical_facts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "pages": 3, "early_round_handoff_present": HANDOFF.is_file()}, indent=2))


if __name__ == "__main__":
    main()
