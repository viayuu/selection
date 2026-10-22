"""CPU-only analysis of completed R94B scalar artifacts; no runtime imports.

Writes only B/ROUND_ANALYSIS.md, B/ROUND_ANALYSIS.pdf and
B/round_analysis_by_problem.csv. Never opens predictions, weights or live A.
"""

import ast
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
V4 = ROOT / "code/V4"
RUN = V4 / "runs/R94_pretrained_node_fusion/B"
STATS = RUN / "full_model_statistics"
REFERENCES = {
    "R89B": V4 / "runs/R89_optimizer_geometry/B_muon_hybrid",
    "R84A": V4 / "runs/R84_new90k_attention/A_baseline",
}
METRICS = (
    "top1_pct", "top2_pct", "top3_pct", "mean_cost", "sbs_cost",
    "oracle_cost", "vs_sbs_pct", "vs_oracle_pct", "actual_regret_pct",
)
PAIR_COUNTS = (
    "corrected", "harmed", "net_correct", "changed",
    "both_wrong_cheaper", "both_wrong_dearer",
)
PAIR_DELTAS = ("top1_delta_pp", "mean_cost_delta", "actual_regret_delta_pp")
TARGET_COUNTS = (
    "winner_count", "baseline_missed", "correct_count", "selected_count",
    "rescued", "lost", "false_pick_count", "new_false_pick_count",
)
TARGET_COSTS = (
    "new_false_pick_regret_contribution_pp", "target_regret_delta_pp",
)
OUTPUT_NAMES = ("ROUND_ANALYSIS.md", "ROUND_ANALYSIS.pdf", "round_analysis_by_problem.csv")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right):
    assert math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10), (left, right)


def timing_code_check(protocol):
    names = (
        "r94_experiment.py", "r94_model.py", "r94_features.py",
        "r84_experiment.py", "pairwise_objective.py",
    )
    hashes = {name: digest(V4 / name) for name in names}
    assert all(hashes[n] == protocol["source_sha256"][n] for n in names)
    namespace = {"DONOR_FIELDS": ("nco_nodes", "nco_node_mask")}
    for name in ("pairwise_objective.py", "r94_model.py"):
        tree = ast.parse((V4 / name).read_text())
        nodes = [
            node for node in tree.body
            if (isinstance(node, ast.FunctionDef)
                and node.name in ("inputs_only", "inputs_with_donor"))
            or (name == "pairwise_objective.py" and isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "LABEL_KEYS" for t in node.targets))
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
    batch = {key: object() for key in (
        "node", "node_mask", "nco_nodes", "nco_node_mask", "ind", "costs",
    )}
    raw = namespace["inputs_only"](batch)
    cached = namespace["inputs_with_donor"](batch)
    assert raw.keys() == cached.keys()
    assert all(raw[key] is cached[key] for key in raw)
    assert raw["nco_nodes"] is batch["nco_nodes"]
    assert "ind" not in raw and "costs" not in raw
    return hashes


def target_aggregate(rows, task_count):
    result = {key: sum(int(row[key]) for row in rows) for key in TARGET_COUNTS}
    result.update({key: sum(float(row[key]) for row in rows) / task_count for key in TARGET_COSTS})
    result["groups"] = len(rows)
    result["groups_with_winners"] = sum(int(row["winner_count"]) > 0 for row in rows)
    result["groups_with_correct"] = sum(int(row["correct_count"]) > 0 for row in rows)
    result["net_recovered"] = result["rescued"] - result["lost"]
    for field, numerator, denominator in (
        ("recall_pct", "correct_count", "winner_count"),
        ("precision_pct", "correct_count", "selected_count"),
        ("rescued_of_baseline_misses_pct", "rescued", "baseline_missed"),
    ):
        result[field] = 100 * result[numerator] / result[denominator] if result[denominator] else None
    return result


def collect():
    protocol = read_json(RUN / "protocol.json")
    lock = read_json(RUN / "final_test_lock.json")
    report_protocol = read_json(STATS / "protocol.json")
    assert lock["selected_update"] == 54720
    assert lock["selected_without_test"] and not lock["per_problem_assembly"]
    assert report_protocol["checkpoint_lock"] == lock
    controller = read_json(V4 / "runs/R94_launch/B/controller.json")
    worker = read_json(V4 / "runs/R94_launch/B/worker.json")
    for receipt in (controller, worker):
        assert receipt["state"] == "complete" and receipt["exit_status"] == 0
        assert all(stage["exit_status"] == 0 for stage in receipt["stages"])
    assert read_json(RUN / "postprocess_status.json")["complete"]
    assert read_json(RUN / "test_results.json")["complete"]
    assert read_json(RUN / "lib_stage_complete.json")["complete"]
    assert read_json(RUN / "training_complete.json")["successful_updates"] == 57600
    tasks = list(protocol["candidate_pools"])
    assert len(tasks) == 18
    metrics = {
        (row["split"], row["problem"]): row
        for name in ("train_metrics_by_problem.csv", "metrics_by_problem.csv")
        for row in read_csv(STATS / name)
    }
    for row in metrics.values():
        assert row["checkpoint_sha256"] == lock["checkpoint_sha256"]
    for split in ("TRAIN", "VAL", "TEST"):
        for task in tasks:
            assert int(metrics[split, task]["n"]) == (81000 if split == "TRAIN" else 4500)
        for field in METRICS:
            close(metrics[split, "ALL"][field], mean(float(metrics[split, t][field]) for t in tasks))
    close(metrics["VAL", "ALL"]["top1_pct"], lock["val"]["macro_top1"] * 100)
    close(metrics["VAL", "ALL"]["mean_cost"], lock["val"]["macro_mean_cost"])
    close(metrics["VAL", "ALL"]["actual_regret_pct"], lock["val"]["macro_actual_regret_pct"])
    pairs, targets, refs = {}, {}, {}
    for name, path in REFERENCES.items():
        ref_protocol = read_json(path / "protocol.json")
        for field in (
            "data_root", "preparation_sha256", "candidate_pools", "candidate_pool_ids",
            "winner_rule", "baselines", "rows_per_problem", "global_batch_size", "seed",
        ):
            assert ref_protocol[field] == protocol[field], (name, field)
        refs[name] = {(r["split"], r["problem"]): r for r in read_csv(path / "full_model_statistics/metrics_by_problem.csv")}
        pairs[name] = {(r["split"], r["problem"]): r for r in read_csv(STATS / f"paired_changes_vs_{name}.csv")}
        targets[name] = read_csv(STATS / f"target_recovery_vs_{name}.csv")
        summaries = read_csv(STATS / f"target_recovery_summary_vs_{name}.csv")
        for split in ("TRAIN", "VAL", "TEST"):
            rr = [r for r in targets[name] if r["split"] == split]
            assert len(rr) == 36
            assert len({(r["problem"], r["solver"]) for r in rr}) == 36
            summary = next(r for r in summaries if r["split"] == split)
            computed = target_aggregate(rr, 18)
            for field in TARGET_COUNTS + TARGET_COSTS:
                close(summary[field], computed[field])
            pair = pairs[name][split, "ALL"]
            assert int(pair["corrected"]) - int(pair["harmed"]) == int(pair["net_correct"])
            close(pair["top1_delta_pp"], int(pair["net_correct"]) / int(pair["n"]) * 100)
            for field in PAIR_COUNTS:
                assert int(pair[field]) == sum(int(pairs[name][split, t][field]) for t in tasks)
            for field in PAIR_DELTAS:
                close(pair[field], mean(float(pairs[name][split, t][field]) for t in tasks))
            if split != "TRAIN":
                for delta, metric in zip(PAIR_DELTAS, ("top1_pct", "mean_cost", "actual_regret_pct")):
                    close(pair[delta], float(metrics[split, "ALL"][metric]) - float(refs[name][split, "ALL"][metric]))
        for lib, task in (("TSPLIB", "TSP"), ("CVRPLIB", "CVRP")):
            assert (lib, task) in refs[name]
    hashes = timing_code_check(protocol)
    timing = read_csv(STATS / "inference_timing.csv")
    assert len(timing) == 18
    return dict(protocol=protocol, lock=lock, metrics=metrics, pairs=pairs, targets=targets,
                tasks=tasks, refs=refs, timing=timing, hashes=hashes,
                nss=read_csv(STATS / "nss_comparison.csv"), report_protocol=report_protocol)


def make_rows(data):
    result = []
    tasks = data["tasks"]
    for split in ("TRAIN", "VAL", "TEST", "TSPLIB", "CVRPLIB"):
        if split in ("TSPLIB", "CVRPLIB"):
            scopes = [("TSP" if split == "TSPLIB" else "CVRP", ["TSP" if split == "TSPLIB" else "CVRP"])]
        else:
            scopes = [(t, [t]) for t in tasks] + [
                ("ALL", tasks), ("COVERED16", [t for t in tasks if t not in ("TSP", "ATSP")]),
                ("RAW2", ["TSP", "ATSP"]),
            ]
        for scope, members in scopes:
            originals = [data["metrics"][split, t] for t in members]
            row = dict(model="R94B", split=split, problem=scope, n=sum(int(r["n"]) for r in originals),
                       task_count=len(members), donor_coverage="raw_only" if all(t in ("TSP", "ATSP") for t in members)
                       else "covered" if all(t not in ("TSP", "ATSP") for t in members) else "mixed",
                       aggregation="equal_task_mean" if len(members) > 1 else "single_problem",
                       selected_update=54720, checkpoint_sha256=data["lock"]["checkpoint_sha256"],
                       data_version=Path(data["protocol"]["data_root"]).name if split in ("TRAIN", "VAL", "TEST")
                       else "original_native_LIB", winner_rule=data["protocol"]["winner_rule"],
                       sbs_fit_split="frozen_FULL_TRAIN_81000_per_problem",
                       candidate_pool=json.dumps({t: data["protocol"]["candidate_pools"][t] for t in members}),
                       cost_scope="cross_problem_raw_cost_descriptive_only" if len(members) > 1 else "within_problem",
                       primary_reference="R89B", secondary_reference="R84A", matched_AB_claim=False,
                       timing_donor_plus_selector_valid=False, timing_evidence="cached_nco_fields_retained; online_donor_bypassed",
                       metrics_source=str(STATS / ("train_metrics_by_problem.csv" if split == "TRAIN" else "metrics_by_problem.csv")))
            row.update({field: mean(float(r[field]) for r in originals) for field in METRICS})
            for name in REFERENCES:
                prefix = name.lower() + "_"
                if split in ("TSPLIB", "CVRPLIB"):
                    baseline = data["refs"][name][split, members[0]]
                    for delta, metric in zip(PAIR_DELTAS, ("top1_pct", "mean_cost", "actual_regret_pct")):
                        row[prefix + delta] = row[metric] - float(baseline[metric])
                    continue
                pp = [data["pairs"][name][split, t] for t in members]
                for field in PAIR_COUNTS:
                    row[prefix + field] = sum(int(r[field]) for r in pp)
                for field in PAIR_DELTAS:
                    row[prefix + field] = mean(float(r[field]) for r in pp)
                tt = [r for r in data["targets"][name] if r["split"] == split and r["problem"] in members]
                row.update({prefix + "target_" + k: v for k, v in target_aggregate(tt, len(members)).items()})
                row[prefix + "paired_source"] = str(STATS / f"paired_changes_vs_{name}.csv")
                row[prefix + "recovery_source"] = str(STATS / f"target_recovery_vs_{name}.csv")
            result.append(row)
    assert len(result) == 65
    return result


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def f(value, digits=6, signed=False):
    return format(float(value), ("+" if signed else "") + f".{digits}f")


def write_reports(data, rows):
    # This generator deliberately has no configurable output root or live-run glob.
    paths = {name: RUN / name for name in OUTPUT_NAMES}
    assert all(p.parent == RUN for p in paths.values())
    index = {(r["split"], r["problem"]): r for r in rows}
    overall = [index[s, "ALL"] for s in ("TRAIN", "VAL", "TEST")]
    main_val, main_test = index["VAL", "ALL"], index["TEST", "ALL"]
    stats_table = [[r["split"], f(r["top1_pct"]), f(r["mean_cost"], 9), f(r["sbs_cost"], 9),
                    f(r["oracle_cost"], 9), f(r["vs_sbs_pct"]), f(r["vs_oracle_pct"]), f(r["actual_regret_pct"])]
                   for r in overall]
    comparison_table, recovery_table = [], []
    for name in REFERENCES:
        for split in ("VAL", "TEST"):
            r = index[split, "ALL"]
            p = name.lower() + "_"
            comparison_table.append([name, split, f(r[p + "top1_delta_pp"], signed=True),
                                     f(r[p + "mean_cost_delta"], 9, True), f(r[p + "actual_regret_delta_pp"], signed=True)])
            recovery_table.append([name, split, r[p + "target_rescued"], r[p + "target_lost"],
                                   r[p + "target_net_recovered"], r[p + "target_new_false_pick_count"],
                                   f(r[p + "target_target_regret_delta_pp"], signed=True),
                                   f(r[p + "target_new_false_pick_regret_contribution_pp"])])
    gap = overall[0]["top1_pct"] - main_val["top1_pct"]
    regret_gap = main_val["actual_regret_pct"] - overall[0]["actual_regret_pct"]
    nss_table = [[r["dataset"], r["n"], f(r["current_top1_native_pct"], 4), f(r["nss_top1_native_pct"], 4),
                  f(r["current_mean_cost"], 6), f(r["nss_mean_cost"], 6)] for r in data["nss"]]
    covered_val, covered_test = index["VAL", "COVERED16"], index["TEST", "COVERED16"]
    raw_test = index["TEST", "RAW2"]
    delta_times = [float(r["donor_plus_selector_ms_per_instance"]) - float(r["cache_hit_selector_ms_per_instance"]) for r in data["timing"]]
    timing_range = f"{min(delta_times):.6f} to {max(delta_times):.6f} ms/instance"
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    analysis = f"""# R94B Completed-Round Analysis
Results-analysis role; generated {timestamp}. No external model/reviewer was invoked; this session has no verified gpt-6.1-sol routing, and no Astra was used.

## Verdict And Lock
Small positive historical-reference result, not a demonstrated pretrained-donor effect or automatic promotion. B's actual controller and worker both returned 0; full18 TRAIN/VAL/TEST, LIB, PDFs and timing stages completed. Timing completion is not timing validity (see below). R94A was still running at the request snapshot (~46k/57.6k); A was not read or evaluated here. No matched A/B claim is made.

Unique selected update **54,720/57,600**, epoch 38; strict minimum full18 VAL macro_vs_sbs_pct, first tie wins. Checkpoint `{data['lock']['checkpoint_sha256']}`. No TEST/LIB reselection, per-problem stitching, calibration, new training, GPU use, Git or publishing in this analysis.

Same frozen `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018` as R89B (main) and R84A (secondary): 81,000/4,500/4,500 per task, seed2, logical batch1024, same preparation hash, native/FP64-ind winners and legal pools. Eight B tasks retain only MTPOMO/MVMOE/reld_mt; RELD_MTL is removed, RELD_MOEL is aliased to reld_mt without minima merging. TEST partly informed prior R&D; not a new blind test. External solver pretraining adds knowledge/compute, and its original instance manifest/overlap is unavailable.

## Locked Full18 Metrics
{md_table(['Split', 'Top1 %', 'Mean cost', 'SBS cost', 'Oracle cost', 'vs SBS %', 'vs Oracle %', 'Actual regret %'], stats_table)}
ALL is the equal-weight mean of 18 task metrics; cross-task raw cost is descriptive, not a pooled performance objective. SBS was fitted on frozen full TRAIN; Oracle is the per-instance minimum over legal candidates. vs_SBS/vs_Oracle are task-level ratios of means, not mean per-instance actual regret.

## Historical Comparisons
{md_table(['Reference', 'Split', 'dTop1 pp', 'dMean cost', 'dActual regret pp'], comparison_table)}
R89B TEST: 1,756 corrected / 1,628 harmed = **+128** native correct, with 4,869 changed predictions. Top1 improves on 14/18 tasks, cost on 14/18, regret on 13/18. VRPBL has the largest Top1 gain (+0.9556 pp); VRPL regresses (-0.6444 pp, cost +0.000678), while OVRPTW/OVRPL/OVRPLTW improve Top1 but worsen cost and regret. Against R84A TEST: 1,913 corrected / 1,636 harmed = +277. These are single-seed historical comparisons, not statistical significance or a matched mechanism test.

## Fixed36 True Recovery
Targets remain the R84A locked-VAL zero/near-zero list (chosen <=0.1%, native winners >0); no TEST-based target reselection. Full18 TEST: 117/4,598 true target winners recovered (recall 2.5446%), 230 target selections, 113 false picks (precision 50.8696%); 12/36 groups have any correct pick, and 34/36 have TEST winner support. VAL: 126/4,625 correct, 238 selections, 112 false picks, recall 2.7243%, precision 52.9412%. TRAIN: 2,730/83,279 correct (3.2781%), 1,526 false picks.

{md_table(['Reference', 'Split', 'Rescued', 'Lost', 'Net', 'New false', 'Target-winner dRegret pp', 'New-false regret burden pp'], recovery_table)}
Rescued/lost are paired native-correct changes within the fixed target winner groups, not mere method-selection counts. 81/86 TEST rescues vs R89B occur in OVRPB, VRPBL, OVRPBL and OVRPLTW. Target-winner dRegret is the change on target-winner rows, normalized by all rows per task then macro-averaged. New-false burden is current regret on newly false-selected rows, not their baseline-relative change. These two subset quantities must not be added/subtracted into a total net regret; full18 dActual regret above is the net cost-quality evidence.

## Coverage And Fitting
Donor features cover CVRP + 15 MVRP tasks only; TSP/ATSP are raw-only. Versus R89B, covered16 TEST dTop1={f(covered_test['r89b_top1_delta_pp'], signed=True)} pp, dCost={f(covered_test['r89b_mean_cost_delta'], 9, True)}, dRegret={f(covered_test['r89b_actual_regret_delta_pp'], signed=True)} pp; raw2 dTop1={f(raw_test['r89b_top1_delta_pp'], signed=True)} pp, dRegret={f(raw_test['r89b_actual_regret_delta_pp'], signed=True)} pp. Covered16 accounts for 86 rescued / 12 lost vs raw2 0 / 2. Covered16 VAL cost actually worsens by {f(covered_val['r89b_mean_cost_delta'], 9, True)} despite full18 VAL cost improving slightly. Raw-only changes can arise from shared training/checkpoint differences, not direct donor execution.

Selected full-TRAIN Top1 71.947874% vs VAL 71.428395%: TRAIN-VAL gap **{gap:.6f} pp**, with VAL-TRAIN actual regret +{regret_gap:.6f} pp. This is the full new90k cohort, not an old tiny training subset. It does not support a large tiny-cohort memorization story, establish a regularization cause, or show broad rare-winner recovery.

## LIB And Reused NSS
{md_table(['Dataset', 'N', 'R94B Top1 %', 'NSS Top1 %', 'R94B cost', 'NSS cost'], nss_table)}
Same R94B checkpoint throughout. NSS is reused R79 historical weights/predictions, NEW4500 TEST inference plus original LIB, not NEW90k retraining. Dataset/pool identities are retained in the existing NSS CSV. TSPLIB vs R89B: -2.0408 pp Top1, +0.010187 cost, +0.150375 pp regret; CVRPLIB: -1.0000 pp Top1, -0.097105 cost, -0.016568 pp regret. Against R84A, TSPLIB Top1 rises +4.0816 pp but cost worsens +0.003230; CVRPLIB Top1 rises +1 pp and cost falls -0.109740. LIB is mixed and small, not uniform transfer superiority. NSS cost is lower on both synthetic TEST tasks despite R94B's higher Top1; CVRPLIB also favors NSS on Top1 and cost.

## Timing Finding And Decision Boundary
**Code-supported diagnostic call-path/labeling error:** `r94_experiment.py:327-334` obtains `FrozenFeatureCache.batch` and calls `engine.inputs_only(batch)` for the nominal online branch. `r94_features.py:414-435` injects `nco_nodes`/`nco_node_mask`; `r84_experiment.py:26` imports `pairwise_objective.inputs_only`, whose lines 10-14 remove supervision but retain those fields. `r94_model.py:191-212` preserves them and calls the donor only under `if not cached`. Thus covered-task nominal donor+selector calls skip the online donor; TSP/ATSP skip it by design. The five inspected source hashes match the recorded protocol. A CPU-only extracted-helper sentinel check confirms raw/cached keys and objects are identical, without importing torch or a model.

Reported column differences span {timing_range}; they are not measured donor overhead. Cached GPU-resident selector timing remains a narrow diagnostic (128 fixed rows, batch timing/128, one warmup, three repetitions; excludes I/O/cache preparation), not independent-instance or full-system latency. `includes_native_encoder=true` / `donor_plus_selector_measured=true` are unsupported by this call path. No online donor latency is established. This finding does not invalidate the saved predictive metrics; no runtime or timing receipt was changed or rerun.

Retain the completed B as a small historical gain with limited true recovery, mixed LIB and unresolved online latency. Defer the matched pretrained-vs-random-donor conclusion until R94A completes under its existing lock. No automatic promotion, new variant, new split or additional run is authorized by this report.

Sources: B `protocol.json`, `final_test_lock.json`, `full_model_statistics/metrics_by_problem.csv`, `train_metrics_by_problem.csv`, paired/recovery CSVs vs R89B/R84A, `nss_comparison.csv`, `inference_timing.csv`; B actual controller/worker receipts; historical scalar CSVs and protocols. No raw predictions, neural forward passes or live A reads. Existing selection/learning-curve PDFs remain unchanged. Companion CSV: 65 rows (18 tasks + ALL/COVERED16/RAW2 for TRAIN/VAL/TEST; 2 LIB rows), with full metrics, both references, recovery and provenance. Missing LIB paired/target counts are blank, not zero.\n"""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with paths["round_analysis_by_problem.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    paths["ROUND_ANALYSIS.md"].write_text(analysis, encoding="utf-8")
    render_pdf(paths["ROUND_ANALYSIS.pdf"], data, index, stats_table, comparison_table,
               recovery_table, nss_table, gap, regret_gap, timing_range)
    return paths


def render_pdf(path, data, index, stats_table, comparison_table, recovery_table,
               nss_table, gap, regret_gap, timing_range):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 8.5})
    ink, teal, muted = "#172A38", "#087F8C", "#52626F"

    def page(title, number):
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(.065, .955, title, fontsize=17, weight="bold", color=ink, va="top")
        fig.text(.065, .928, "Locked R94B | seed 2 | native/FP64 winners | frozen NEW90k18", color=muted, fontsize=9)
        fig.add_artist(plt.Line2D([.065, .935], [.913, .913], color=teal, lw=2, transform=fig.transFigure))
        fig.text(.065, .027, "Source: completed scalar artifacts only. No GPU, training, live A edits, Git or publication.", fontsize=7, color=muted)
        fig.text(.935, .027, f"{number} / 2", fontsize=8, ha="right", color=muted)
        return fig

    def heading(fig, y, text):
        fig.text(.065, y, text, fontsize=10.4, weight="bold", color=teal, va="top")
        return y - .023

    def paragraph(fig, y, text, size=8.6, width=109):
        import textwrap
        lines = []
        for line in text.split("\n"):
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False) or [""])
        step = size * 1.32 / (11.69 * 72)
        for line in lines:
            fig.text(.065, y, line, va="top", fontsize=size, color=ink)
            y -= step
        return y - .008

    def table(fig, y, headers, values, widths=None, size=8, row_height=.021):
        height = row_height * (len(values) + 1)
        ax = fig.add_axes([.065, y - height, .87, height])
        ax.axis("off")
        t = ax.table(cellText=values, colLabels=headers, colWidths=widths,
                     cellLoc="right", colLoc="right", bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(size)
        for (r, c), cell in t.get_celld().items():
            cell.set_linewidth(.35)
            cell.set_edgecolor("#DDE5EA")
            cell.PAD = .08
            if r == 0:
                cell.set_facecolor(ink)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#EFF5F7" if r % 2 else "white")
                cell.set_text_props(color=ink)
            if c == 0:
                cell.get_text().set_ha("left")
        return y - height - .013

    with PdfPages(path, metadata={"Title": "R94B Completed-Round Analysis", "Author": "Codex results analysis"}) as pdf:
        fig = page("R94B | Small Gain, Limited Recovery", 1)
        y = heading(fig, .894, "01  Completed and locked, not a matched A/B verdict")
        y = paragraph(fig, y,
            "Actual B controller/worker rc=0; full18 TRAIN/VAL/TEST, LIB and PDFs complete. Selected update "
            "54,720/57,600 (epoch 38), strict minimum full18 VAL macro_vs_sbs_pct; one checkpoint/rule for all datasets. "
            "R94A was still running at the request snapshot (~46k/57.6k); not read here. No matched donor-effect claim.")
        y = paragraph(fig, y,
            "Same frozen NEW90k18 as R89B (main) / R84A (secondary): 81,000/4,500/4,500 per task; seed2, batch1024; "
            "preparation hash, native/FP64-ind labels, TRAIN SBS and legal pools verified equal. Eight B tasks keep "
            "MTPOMO/MVMOE/reld_mt only; no RELD minima merging. TEST partly informed prior R&D, not a new blind test.")
        y = heading(fig, y, "02  Locked full18 metrics")
        compact_stats = [[r[0], f(r[1], 4), f(r[2], 6), f(r[3], 6), f(r[4], 6), f(r[5], 4), f(r[6], 4), f(r[7], 4)] for r in stats_table]
        y = table(fig, y, ["Split", "Top1 %", "Cost", "SBS", "Oracle", "vsSBS %", "vsOrcl %", "Regret %"],
                  compact_stats, [.10, .11, .14, .14, .14, .12, .13, .12], size=7.8)
        y = paragraph(fig, y,
            "ALL = equal-task mean, not pooled raw-cost evidence. SBS is fitted on frozen full TRAIN; Oracle is the "
            "legal per-instance minimum. Ratios of means (vsSBS/vsOracle) differ from mean per-instance actual regret.", size=8)
        y = heading(fig, y, "03  Historical references: positive, but small and uneven")
        y = table(fig, y, ["Ref", "Split", "dTop1 pp", "dCost", "dRegret pp"], comparison_table,
                  [.14, .13, .22, .28, .23], size=8.3)
        y = paragraph(fig, y,
            "R89B TEST: 1,756 corrected - 1,628 harmed = +128 correct; 4,869 predictions changed. Top1 and cost "
            "improve on 14/18 tasks; regret improves on 13/18. R84A TEST: +277 net correct. Single-seed historical "
            "comparisons do not establish statistical significance or isolate donor pretraining.")
        y = heading(fig, y, "04  Fixed36: count true recovery and its price")
        y = paragraph(fig, y,
            "Fixed R84A VAL zero/near-zero groups (chosen <=0.1%, native support >0); no target reselection. "
            "TEST: 117/4,598 true target winners correct (2.5446% recall); 230 selections / 113 false picks "
            "(50.8696% precision). VAL: 126/4,625 correct (2.7243%), 112 false picks. Only 12/36 TEST groups "
            "have any correct pick; 34/36 have winner support.")
        y = table(fig, y, ["Ref", "Split", "Rescue", "Lost", "Net", "New FP", "dTarget R", "New-FP R"],
                  [[*r[:6], f(r[6], 6, True), f(r[7], 6)] for r in recovery_table],
                  [.12, .10, .10, .08, .09, .11, .20, .20], size=7.9)
        y = paragraph(fig, y,
            "81/86 TEST rescues vs R89B occur in just four tasks: OVRPB, VRPBL, OVRPBL, OVRPLTW. "
            "dTarget R is target-winner regret change, normalized by all task rows and macro-averaged (pp). "
            "New-FP R is current regret burden on newly false-selected rows, not a baseline-relative change. "
            "Do not add/subtract these subsets; total net dRegret is in section 03.", size=8.1)
        y = heading(fig, y, "Decision boundary")
        y = paragraph(fig, y,
            "Retain B as a small historical gain with limited true recovery. No automatic promotion/variant. "
            "Wait for the existing R94A completion before a pretrained-vs-random-donor conclusion. External "
            "pretraining adds compute/knowledge; its original instance manifest and overlap remain unknown.")
        assert y > .055, ("page1 overflow", y)
        fig.text(.065, .054, "Checkpoint SHA256: " + data["lock"]["checkpoint_sha256"], fontsize=6.3, color=muted)
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("R94B | Where the Changes Occur", 2)
        y = heading(fig, .894, "05  All18 TEST changes vs main R89B; secondary R84A shown at right")
        task_values = []
        for task in data["tasks"]:
            r = index["TEST", task]
            task_values.append([
                task + ("*" if task in ("TSP", "ATSP") else ""), f(r["top1_pct"], 3),
                f(r["r89b_top1_delta_pp"], 4, True),
                f(r["r89b_mean_cost_delta"], 6, True) if abs(r["r89b_mean_cost_delta"]) >= 5e-7
                else format(r["r89b_mean_cost_delta"], "+.2e"),
                f(r["r89b_actual_regret_delta_pp"], 5, True),
                f"{r['r89b_target_rescued']}/{r['r89b_target_lost']}/{r['r89b_target_new_false_pick_count']}",
                f(r["r84a_top1_delta_pp"], 4, True),
            ])
        y = table(fig, y, ["Task", "Top1 %", "dTop1 pp", "dCost", "dRegret pp", "R/L/newFP", "dTop1 R84"],
                  task_values, [.18, .13, .14, .16, .16, .13, .14], size=7.8, row_height=.016)
        y = paragraph(fig, y,
            "* TSP/ATSP are raw-only. VRPL loses Top1/cost; OVRPTW, OVRPL and OVRPLTW gain Top1 but worsen "
            "cost/regret. Full per-task metrics, SBS/Oracle, both references and recovery losses: "
            "round_analysis_by_problem.csv.", size=8, width=116)
        y = heading(fig, y, "06  Coverage and full-TRAIN fitting, not an old tiny-cohort story")
        y = paragraph(fig, y,
            "Covered16 TEST vs R89B: dTop1 +0.166667 pp, dCost -0.001376926, dRegret -0.011536 pp; "
            "86 rescued / 12 lost. Raw2: dTop1 +0.088889 pp, dRegret -0.002487 pp; 0 rescued / 2 lost. "
            "Covered16 VAL cost worsens +0.000082329 despite the tiny full18 VAL cost gain. Raw-only "
            "changes are not direct donor evidence.", size=8.2, width=114)
        y = paragraph(fig, y,
            f"Selected full TRAIN vs VAL Top1: 71.947874% vs 71.428395%, gap {gap:.6f} pp; "
            f"VAL-TRAIN regret +{regret_gap:.6f} pp. TRAIN target recall is only 3.2781% vs VAL 2.7243%. "
            "No large tiny-subset memorization claim or regularization-cause inference is supported.", size=8.2, width=114)
        y = heading(fig, y, "07  LIB / reused NSS: mixed, not uniform transfer superiority")
        y = table(fig, y, ["Dataset", "N", "B Top1 %", "NSS Top1 %", "B cost", "NSS cost"], nss_table,
                  [.18, .07, .18, .19, .19, .19], size=7.8, row_height=.018)
        y = paragraph(fig, y,
            "NSS = reused R79 historical weights/predictions, NEW4500 TEST plus original LIB; no NEW90k retrain. "
            "Same B checkpoint/pools throughout. TSPLIB vs R89B: Top1 -2.0408 pp, cost +0.010187; CVRPLIB: "
            "Top1 -1 pp, cost -0.097105. Vs R84A, TSPLIB Top1 +4.0816 pp but cost +0.003230; CVRPLIB "
            "Top1 +1 pp and cost -0.109740. NSS has lower cost on both synthetic TEST tasks and CVRPLIB.", size=8, width=116)
        y = heading(fig, y, "08  Timing call-path error; saved predictive metrics not implicated")
        y = paragraph(fig, y,
            "r94_experiment.py:327-334 reads FrozenFeatureCache.batch, then engine.inputs_only(batch). "
            "r94_features.py:414-435 injects nco_nodes/nco_node_mask; pairwise_objective.py:10-14 removes "
            "labels but retains both. r94_model.py:191-212 executes the donor only when those fields are "
            "absent. Thus the nominal donor+selector path uses injected cached features on covered tasks; "
            "TSP/ATSP never use a donor. All five inspected source hashes match the recorded protocol; a "
            "CPU-only dictionary-helper check confirms identical raw/cached keys and objects.", size=8.1, width=115)
        y = paragraph(fig, y,
            f"Reported differences ({timing_range}) are NOT measured donor overhead. Online donor latency "
            "is unestablished; includes_native_encoder / donor_plus_selector_measured metadata is unsupported. "
            "Only GPU-resident cached-selector batch timing remains: 128 rows, one warmup, three repetitions, "
            "excluding I/O/cache preparation. No end-to-end claim, rerun, runtime fix or result change here.", size=8.1, width=115)
        assert y > .048, ("page2 overflow", y)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    data = collect()
    rows = make_rows(data)
    outputs = write_reports(data, rows)
    print(json.dumps({"rows": len(rows), "pdf_pages": 2,
                      "outputs": {k: str(v) for k, v in outputs.items()},
                      "source_hashes_match": True, "timing_raw_retains_cached_features": True}))


if __name__ == "__main__":
    main()
