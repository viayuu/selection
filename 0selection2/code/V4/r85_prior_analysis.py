"""CPU-only analysis of existing locked R77/R80-R84 predictions."""

import argparse
import csv
import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "code/V4/runs"
OUTPUT = RUNS / "R85_zero_selection_research/prior_experiments"
SPECS = (
    ("R77B", "R77_sample_size_batch1024/B_full_seed2", "Original R77 objective"),
    ("R80A", "R80_training_and_code_ablation/A_ce_only", "CE only"),
    ("R80B", "R80_training_and_code_ablation/B_batch512", "Batch 512; matched sample exposures"),
    ("R80C", "R80_training_and_code_ablation/C_lr3e4", "Peak learning rate 3e-4"),
    ("R80D", "R80_training_and_code_ablation/D_code_embedding", "Four-view source embedding"),
    ("R81B", "R81_method_conditioned_auc/B_cross_instance_rank_seed2", "Method-conditioned cross-instance AUC"),
    ("R82B", "R82_context_matched_ranking/B_context_matched_rank_seed2", "Context-matched regret ranking"),
    ("R83", "R83_model_only_source_embedding/model_only_seed2", "Model-only source embedding"),
    ("R84A", "R84_new90k_attention/A_baseline", "NEW90k matched baseline"),
    ("R84B", "R84_new90k_attention/B_attention_direct", "Direct attention selection"),
    ("R84C", "R84_new90k_attention/C_focal", "Original head plus focal penalty"),
    ("R84D", "R84_new90k_attention/D_attention_focal", "Direct attention plus focal penalty"),
)
METRICS = ("top1", "ce", "mean_cost", "actual_regret_pct", "vs_sbs_pct", "vs_oracle_pct")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_path(run, split, lock):
    directory = run / "evaluation" / split.lower()
    options = (directory / "selected_best/rank000.npz",
               directory / ("update%07d/rank000.npz" % lock["selected_update"]),
               directory / ("epoch%03d/rank000.npz" % lock["epoch"]))
    return next((path for path in options if path.is_file()), None)


def metric_summary(result):
    return {key: result["macro"].get("macro_" + key) for key in METRICS}


def saved_zero_summary(result):
    rows = []
    for problem, saved in result["per_problem"].items():
        n = saved["n"]
        for solver, share in zip(saved["pool"], saved["pick_dist"]):
            support = int(round(saved["method_top1"][solver] * n))
            chosen = int(round(share * n))
            if support and chosen == 0:
                rows.append(dict(problem=problem, solver=solver, winner_count=support))
    return dict(groups=len(rows), winner_mass=sum(row["winner_count"] for row in rows), methods=rows)


def load_runs():
    runs = []
    for model, relative, description in SPECS:
        path = RUNS / relative
        config = read_json(path / "protocol.json")
        lock = read_json(path / "final_test_lock.json")
        final = read_json(path / "final_metrics.json")
        test = read_json(path / "test_results.json")
        if not test.get("complete") or not lock.get("selected_without_test") or lock.get("per_problem_assembly"):
            raise ValueError("Incomplete/unlocked run: " + model)
        for result in (final, test):
            if result["checkpoint_sha256"] != lock["checkpoint_sha256"] or result["selected_update"] != lock["selected_update"]:
                raise ValueError("Saved metrics are not from the locked model: " + model)
        results = dict(TRAIN=final.get("train"), VAL=final["val"], TEST=test["result"])
        group = "NEW90k" if "new90k" in config["data_root"] else "OLD77904"
        baseline = "R84A" if group == "NEW90k" else "R77B"
        runs.append(dict(model=model, path=path, description=description, config=config, lock=lock,
                         results=results, group=group, baseline=baseline))
    return runs


def handoff(runs, output):
    rows = []
    for run in runs:
        row = dict(model=run["model"], dataset_group=run["group"], baseline=run["baseline"],
                   description=run["description"], data_root=run["config"]["data_root"],
                   selected_update=run["lock"]["selected_update"],
                   checkpoint_sha256=run["lock"]["checkpoint_sha256"], splits={})
        for split, result in run["results"].items():
            zero = saved_zero_summary(result) if result else {}
            row["splits"][split] = (dict(metrics=metric_summary(result),
                                       zero_selection={k: zero[k] for k in ("groups", "winner_mass")},
                                       prediction_path=str(selected_path(run["path"], split, run["lock"]) or "MISSING"))
                                       if result else dict(status="missing_saved_train_metrics"))
        rows.append(row)
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage="locked_metrics_handoff",
                    previous_goal_turn="progress: NEW90k release, training, locked TEST and reports completed",
                    goal_complete=False, analysis_backend="current Codex execution; no Astra used for results",
                    scope="R80 A/B/C/D, R81B, R82B, R83, R84 A/B/C/D; R77B historical reference",
                    winner_rule="saved native/FP64 ind; never FP32 argmin",
                    comparisons="Within dataset only; OLD77904 vs NEW90k raw costs are not controlled comparisons",
                    source="final_metrics.json and final_test_lock.json; never latest validation",
                    key_evidence=[
                        "All R84 arms have TRAIN Top1 around 71.2-71.3%, close to locked VAL; zero selection already exists in TRAIN.",
                        "R84A/B TRAIN: 26 zero-with-winner groups / 25,627 winner instances; R84C: 23 / 18,259; R84D: 23 / 15,936.",
                        "R84 focal/direct-attention variants do not improve locked VAL Top1 together with cost and regret.",
                        "Next action should address TRAIN method discrimination/argmax recall; generic overfitting is not established by these locked models.",
                    ], runs=rows)
    write_json(output / "EVIDENCE_HANDOFF.json", evidence)
    return evidence


def auc(scores, truth):
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(scores, method="average")
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def load_split(run, split):
    from .r77_best_model_report import load_predictions
    path = selected_path(run["path"], split, run["lock"])
    result = run["results"][split]
    if path is None:
        return None
    if result is None:
        raise ValueError("Saved prediction without associated metrics: " + str(path))
    identity = result.get("identity")
    test_identity = run["results"]["TEST"].get("identity")
    if identity and test_identity and identity["model_state_sha256"] != test_identity["model_state_sha256"]:
        raise ValueError("TRAIN/VAL/TEST parameter state mismatch: " + run["model"])
    predictions = {}
    with np.load(path, allow_pickle=False) as archive:
        for problem, saved in result["per_problem"].items():
            data = load_predictions(archive, path, problem + "__")
            np.testing.assert_allclose(data.top1_pct, saved["top1"] * 100, atol=1e-10, rtol=0)
            np.testing.assert_allclose(data.mean_cost, saved["mean_cost"], atol=1e-10, rtol=0)
            scores = archive[problem + "__logits"].astype(np.float64)
            centered = scores - scores.max(axis=1, keepdims=True)
            probabilities = np.exp(centered)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            order = np.lexsort((np.broadcast_to(data.pool_ids, scores.shape), -scores), axis=1)
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(np.arange(len(data.pool)), order.shape), axis=1)
            release_key = problem + "__release_source_indices"
            predictions[problem] = dict(data=data, scores=scores, probabilities=probabilities,
                                        ranks=ranks + 1,
                                        release_indices=archive[release_key] if release_key in archive else None)
    return predictions


def metric_rows(run, split):
    result = run["results"][split]
    if result is None:
        return []
    rows = []
    for problem, saved in result["per_problem"].items():
        rows.append(dict(model=run["model"], dataset_group=run["group"], split=split, problem=problem,
                         n=saved["n"], native_correct=saved.get("native_correct", int(round(saved["top1"] * saved["n"]))),
                         top1_pct=100 * saved["top1"], ce=saved["ce"], mean_cost=saved["mean_cost"],
                         actual_regret_pct=saved["actual_regret_pct"], sbs_cost=saved["sbs_cost"],
                         oracle_cost=saved["oracle_cost"], vs_sbs_pct=saved["vs_sbs_pct"],
                         vs_oracle_pct=saved["vs_oracle_pct"], sbs_name=saved["sbs_name"],
                         sbs_selection_split="TRAIN", winner_rule="saved native/FP64 ind",
                         checkpoint_sha256=run["lock"]["checkpoint_sha256"],
                         selected_update=run["lock"]["selected_update"]))
    macro = dict(rows[0], problem="ALL", n=sum(r["n"] for r in rows),
                 native_correct=sum(r["native_correct"] for r in rows), sbs_name="per-problem TRAIN SBS")
    for name in ("top1_pct", "ce", "mean_cost", "actual_regret_pct", "sbs_cost", "oracle_cost", "vs_sbs_pct", "vs_oracle_pct"):
        macro[name] = float(np.mean([r[name] for r in rows]))
    for name in METRICS:
        expected = result["macro"]["macro_" + name] * (100 if name == "top1" else 1)
        np.testing.assert_allclose(macro["top1_pct" if name == "top1" else name], expected, rtol=0, atol=1e-9)
    return rows + [macro]


def method_rows(run, split, predictions):
    rows = []
    for problem, packed in predictions.items():
        data, scores = packed["data"], packed["scores"]
        index = np.arange(data.n)
        oracle = data.costs.min(axis=1)
        chosen_regret = (data.costs[index, data.pred] - oracle) / oracle
        for slot, solver in enumerate(data.pool):
            truth, selected = data.winner == slot, data.pred == slot
            support, count = int(truth.sum()), int(selected.sum())
            correct = int((truth & selected).sum())
            other = np.max(np.where(np.arange(len(data.pool))[None, :] == slot, -np.inf, scores), axis=1)
            rows.append(dict(model=run["model"], dataset_group=run["group"], split=split,
                             problem=problem, solver=solver, n=data.n, winner_count=support,
                             selected_count=count, correct_count=correct, missed_winner_count=support-correct,
                             winner_share_pct=100 * support/data.n, selected_share_pct=100 * count/data.n,
                             recall_pct=100 * correct/support if support else None,
                             precision_pct=100 * correct/count if count else None,
                             false_pick_count=count-correct,
                             false_pick_cost_increase_sum=float((data.costs[index, data.pred]-oracle)[selected & ~truth].sum()),
                             false_pick_regret_contribution_pp=100 * float(chosen_regret[selected & ~truth].sum()/data.n),
                             missed_winner_regret_contribution_pp=100 * float(chosen_regret[truth & ~selected].sum()/data.n),
                             mean_softmax_probability=float(packed["probabilities"][:, slot].mean()),
                             winner_mean_probability=float(packed["probabilities"][truth, slot].mean()) if support else None,
                             nonwinner_mean_probability=float(packed["probabilities"][~truth, slot].mean()) if (~truth).any() else None,
                             winner_mean_rank=float(packed["ranks"][truth, slot].mean()) if support else None,
                             winner_top2_count=int((truth & (packed["ranks"][:, slot] <= 2)).sum()),
                             winner_top3_count=int((truth & (packed["ranks"][:, slot] <= 3)).sum()),
                             winner_mean_margin=float((scores[truth, slot]-other[truth]).mean()) if support else None,
                             decision_logit_auc=auc(scores[:, slot], truth),
                             method_probability_auc=auc(packed["probabilities"][:, slot], truth),
                             probability_note="softmax(saved decision logits), temperature 1; not calibrated posterior",
                             prediction_path=str(data.path)))
    return rows


def check_pair(old, new):
    if set(old) != set(new):
        raise ValueError("Different task populations")
    for problem in old:
        left, right = old[problem]["data"], new[problem]["data"]
        if left.pool != right.pool:
            raise ValueError("Different pools: " + problem)
        for name in ("costs", "winner", "indices", "pool_ids"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        for name in ("source_indices",):
            if getattr(left, name) is not None and getattr(right, name) is not None:
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        a, b = old[problem]["release_indices"], new[problem]["release_indices"]
        if a is not None and b is not None:
            np.testing.assert_array_equal(a, b)


def paired_rows(model, baseline, split, old, new):
    check_pair(old, new)
    rows = []
    for problem in old:
        left, right = old[problem]["data"], new[problem]["data"]
        old_ok, new_ok = left.pred == left.winner, right.pred == right.winner
        index, oracle = np.arange(left.n), left.costs.min(axis=1)
        delta = right.costs[index, right.pred] - left.costs[index, left.pred]
        both_wrong = ~old_ok & ~new_ok
        rows.append(dict(model=model, baseline=baseline, split=split, problem=problem, n=left.n,
                         changed=int((left.pred != right.pred).sum()), corrected=int((~old_ok & new_ok).sum()),
                         harmed=int((old_ok & ~new_ok).sum()), net_correct=int(new_ok.sum()-old_ok.sum()),
                         both_wrong_cheaper=int((both_wrong & (delta < 0)).sum()),
                         both_wrong_dearer=int((both_wrong & (delta > 0)).sum()),
                         mean_cost_delta=float(delta.mean()),
                         actual_regret_delta_pp=100 * float((delta/oracle).mean())))
    macro = dict(rows[0], problem="ALL", n=sum(r["n"] for r in rows))
    for key in ("changed", "corrected", "harmed", "net_correct", "both_wrong_cheaper", "both_wrong_dearer"):
        macro[key] = sum(r[key] for r in rows)
    for key in ("mean_cost_delta", "actual_regret_delta_pp"):
        macro[key] = float(np.mean([r[key] for r in rows]))
    return rows + [macro]


def target_rows(run, split, baseline_predictions, predictions, targets, definition):
    check_pair(baseline_predictions, predictions)
    rows = []
    for problem, solver in targets:
        old, new = baseline_predictions[problem]["data"], predictions[problem]["data"]
        slot, index = new.pool.index(solver), np.arange(new.n)
        truth, selected, old_selected = new.winner == slot, new.pred == slot, old.pred == slot
        oracle = new.costs.min(axis=1)
        old_regret = (old.costs[index, old.pred] - oracle)/oracle
        new_regret = (new.costs[index, new.pred] - oracle)/oracle
        new_false = selected & ~truth & ~old_selected
        rows.append(dict(model=run["model"], baseline=run["baseline"], dataset_group=run["group"],
                         split=split, target_definition=definition, problem=problem, solver=solver,
                         n=new.n, winner_count=int(truth.sum()), baseline_correct=int((truth & old_selected).sum()),
                         baseline_missed=int((truth & ~old_selected).sum()), selected_count=int(selected.sum()),
                         correct_count=int((truth & selected).sum()), rescued=int((truth & ~old_selected & selected).sum()),
                         lost=int((truth & old_selected & ~selected).sum()), false_pick_count=int((selected & ~truth).sum()),
                         new_false_pick_count=int(new_false.sum()),
                         new_false_pick_regret_contribution_pp=100 * float(new_regret[new_false].sum()/new.n),
                         target_winner_regret_delta_pp=100 * float((new_regret-old_regret)[truth].sum()/new.n)))
    if not rows:
        return [], None
    total = dict(rows[0], problem="ALL", solver="fixed target list", target_groups=len(rows))
    for key in ("n", "winner_count", "baseline_correct", "baseline_missed", "selected_count", "correct_count", "rescued", "lost", "false_pick_count", "new_false_pick_count"):
        total[key] = sum(row[key] for row in rows)
    for key in ("new_false_pick_regret_contribution_pp", "target_winner_regret_delta_pp"):
        total[key] = sum(row[key] for row in rows)/18
    total.update(recall_pct=100 * total["correct_count"]/total["winner_count"] if total["winner_count"] else None,
                 precision_pct=100 * total["correct_count"]/total["selected_count"] if total["selected_count"] else None,
                 rescued_of_baseline_misses_pct=100 * total["rescued"]/total["baseline_missed"] if total["baseline_missed"] else None,
                 n_note="sum of method-specific populations; overlapping instances, not independent samples")
    return rows, total


def analyze(runs, output):
    metric_table, methods, summaries, pairs, target_details, target_totals = [], [], [], [], [], []
    predictions, target_sets, fixed_targets, history_rows, train_evidence = {}, {}, [], [], []
    baselines = [run for run in runs if run["model"] in ("R77B", "R84A")]
    for run in baselines:
        predictions[run["model"], "VAL"] = load_split(run, "VAL")
        rows = method_rows(run, "VAL", predictions[run["model"], "VAL"])
        for definition, cutoff in (("strict_zero_VAL", 0.0), ("near_zero_le_0.1pct_VAL", 0.1)):
            targets = [(r["problem"], r["solver"]) for r in rows if r["winner_count"] > 0 and r["selected_share_pct"] <= cutoff]
            target_sets[run["group"], definition] = targets
            fixed_targets.extend(dict(baseline=run["model"], dataset_group=run["group"], definition=definition,
                                      problem=r["problem"], solver=r["solver"], VAL_winners=r["winner_count"],
                                      VAL_selected=r["selected_count"], VAL_correct=r["correct_count"])
                                 for r in rows if (r["problem"], r["solver"]) in targets)
    # Retain only the two baseline TRAIN archives; each variant TRAIN is released after use.
    for run in runs:
        print("Analyzing " + run["model"], flush=True)
        for split in ("TRAIN", "VAL", "TEST"):
            result = run["results"][split]
            split_predictions = predictions.get((run["model"], split))
            if split_predictions is None:
                split_predictions = load_split(run, split)
            table = metric_rows(run, split)
            metric_table.extend(table)
            if split_predictions is None:
                summary = dict(next(row for row in table if row["problem"] == "ALL"),
                               zero_with_winner_groups=None, zero_winner_mass=None,
                               prediction_status="MISSING; no zero-selection inference") if table else None
                if summary:
                    summaries.append(summary)
                continue
            observations = method_rows(run, split, split_predictions)
            methods.extend(observations)
            zero = [row for row in observations if row["winner_count"] and row["selected_count"] == 0]
            summaries.append(dict(next(row for row in table if row["problem"] == "ALL"),
                                  zero_with_winner_groups=len(zero), zero_winner_mass=sum(r["winner_count"] for r in zero),
                                  prediction_status="same-lock saved predictions"))
            summaries[-1]["total_missed_winner_count"] = summaries[-1]["n"] - summaries[-1]["native_correct"]
            if run["model"] != run["baseline"]:
                baseline_predictions = predictions[run["baseline"], split]
                pairs.extend(paired_rows(run["model"], run["baseline"], split, baseline_predictions, split_predictions))
            else:
                baseline_predictions = split_predictions
            for definition in ("strict_zero_VAL", "near_zero_le_0.1pct_VAL"):
                details, total = target_rows(run, split, baseline_predictions, split_predictions,
                                             target_sets[run["group"], definition], definition)
                target_details.extend(details)
                if total:
                    target_totals.append(total)
            if split != "TRAIN" or run["model"] == run["baseline"]:
                predictions[run["model"], split] = split_predictions
        history = read_json(run["path"] / "history.json")
        for row in history:
            for split in ("train", "val"):
                result = row.get(split)
                if result:
                    history_rows.append(dict(model=run["model"], dataset_group=run["group"], split=split.upper(),
                                             update=row["successful_updates"],
                                             selected=row["successful_updates"] == run["lock"]["selected_update"],
                                             **metric_summary(result)))
        last = history[-1]
        train = run["results"]["TRAIN"]
        val = run["results"]["VAL"]
        train_evidence.append(dict(model=run["model"], dataset_group=run["group"],
                                   selected_update=run["lock"]["selected_update"],
                                   selected_train_top1_pct=100 * train["macro"]["macro_top1"] if train else None,
                                   selected_val_top1_pct=100 * val["macro"]["macro_top1"],
                                   train_val_gap_pp=100 * (train["macro"]["macro_top1"]-val["macro"]["macro_top1"]) if train else None,
                                   last_update=last["successful_updates"],
                                   last_saved_train_top1_pct=100 * last["train"]["macro"]["macro_top1"] if last.get("train") else None,
                                   last_val_top1_pct=100 * last["val"]["macro"]["macro_top1"],
                                   last_saved_train_update=last["successful_updates"] if last.get("train") else None,
                                   selected_train_prediction_path=str(selected_path(run["path"], "TRAIN", run["lock"]) or "MISSING")))
    for model, reference in (("R83", "R80D"), ("R84D", "R84B"), ("R84D", "R84C")):
        for split in ("VAL", "TEST"):
            pairs.extend(paired_rows(model, reference, split, predictions[reference, split], predictions[model, split]))
    summary_index = {(row["model"], row["split"]): row for row in summaries}
    run_index = {run["model"]: run for run in runs}
    for row in summaries:
        baseline = run_index[row["model"]]["baseline"]
        reference = summary_index[baseline, row["split"]]
        row["baseline"] = baseline
        row["top1_delta_pp"] = row["top1_pct"] - reference["top1_pct"]
        row["mean_cost_delta"] = row["mean_cost"] - reference["mean_cost"]
        row["actual_regret_delta_pp"] = row["actual_regret_pct"] - reference["actual_regret_pct"]
    zero_methods = [row for row in methods if row["winner_count"] and row["selected_count"] == 0]
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "metrics_by_problem.csv", metric_table)
    write_csv(output / "method_diagnostics.csv", methods)
    write_csv(output / "zero_selection_methods.csv", zero_methods)
    write_csv(output / "fixed_targets.csv", fixed_targets)
    write_csv(output / "target_recovery.csv", target_totals)
    write_csv(output / "target_recovery_by_method.csv", target_details)
    write_csv(output / "paired_changes.csv", [row for row in pairs if row["problem"] == "ALL"])
    write_csv(output / "paired_changes_by_problem.csv", pairs)
    write_csv(output / "train_validation_evidence.csv", train_evidence)
    write_csv(output / "learning_curves.csv", history_rows)
    result = dict(summary=summaries, methods=methods, targets=target_totals, pairs=pairs,
                  fixed_targets=fixed_targets, history=history_rows, train_evidence=train_evidence)
    write_json(output / "analysis_data.json", result)
    return result


def fmt(value, digits=4):
    return "MISSING" if value is None else f"{value:.{digits}f}"


def report(runs, data, output):
    by_key = {(r["model"], r["split"]): r for r in data["summary"]}
    evidence = {r["model"]: r for r in data["train_evidence"]}
    targets = {(r["model"], r["split"], r["target_definition"]): r for r in data["targets"]}
    facts = [
        "R80-R83 belong to OLD77904 (70,114/3,895/3,895 per task); R84 belongs to NEW90k (81,000/4,500/4,500). Compare within each release. Raw costs across releases are not a controlled gain.",
        "The current locked R84 models have TRAIN Top1 around 71.2-71.3%, close to VAL. Zero selection is already present in TRAIN. These models do not show the near-100% TRAIN vs 50% holdout behavior of earlier binary readouts.",
        "R84A/B TRAIN have 26 zero-with-winner groups / 25,627 winners. Focal C/D leave 23 groups / 18,259 and 15,936 winners. Reducing zero groups does not establish correct recovery or system improvement.",
        "All three R84 variants have lower locked VAL Top1 and higher mean_cost/actual regret than A. Focal alone has slightly higher TEST Top1 but also higher TEST cost/regret; TEST does not override the failed VAL acceptance gate.",
        "R81 global AUC, R82 matched ranking, R83 source filtering and the R80 optimizer/loss changes do not provide a clear system-level breakthrough. Same-method ranking evidence must not be confused with becoming the correct cross-method argmax.",
        "Next action: prioritize method-specific TRAIN winner discrimination and the gap between ranking/probability support and final argmax. A generic longer-training/overfitting narrative is not justified for these locked R84 models. Any new proposal must improve VAL Top1 and reduce missed winner mass without increasing cost.",
        "All runs use seed 2 only; no multi-seed significance claim. Native/FP64 winner ties follow the saved label. False picks can have zero extra cost if tied. All18 metrics are equal task means; mixed raw mean_cost is descriptive.",
        "TEST has informed earlier research. This retrospective analysis does not select new checkpoints, thresholds, temperatures or per-task model mixtures. No GPU inference or new training was used.",
    ]
    lines = ["# R80-R84: locked-model evidence for zero-selection research", "",
             "Status: prior analysis complete; the broader optimization goal remains active.",
             "Execution: CPU-only result analysis; no Astra result-review call. Previous goal turn: progress.", ""]
    for group, heading in (("OLD77904", "Original 77,904-example release"), ("NEW90k", "NEW90k matched release")):
        lines += ["## " + heading, "", "| Model | TRAIN Top1 % | VAL Top1 % | TEST Top1 % | TEST mean_cost | TEST actual regret % | VAL zero groups / winners |", "|---|---:|---:|---:|---:|---:|---:|"]
        for run in runs:
            if run["group"] != group:
                continue
            model = run["model"]
            train, val, test = [by_key[model, split] for split in ("TRAIN", "VAL", "TEST")]
            lines.append(f"| {model} | {fmt(train['top1_pct'])} | {fmt(val['top1_pct'])} | {fmt(test['top1_pct'])} | {fmt(test['mean_cost'],9)} | {fmt(test['actual_regret_pct'],6)} | {val['zero_with_winner_groups']} / {val['zero_winner_mass']:,} |")
        lines.append("")
    lines += ["## What changes the next action", ""] + [f"{i+1}. {fact}" for i, fact in enumerate(facts)] + ["", "## Fixed target recovery", "", "Fixed targets are baseline locked VAL methods with native winners and chosen share <=0.1%. This descriptive list never enters sampling/checkpoint selection.", "", "| Model | VAL target correct / winners | Rescued / lost vs baseline | New false picks | TEST target correct / winners |", "|---|---:|---:|---:|---:|"]
    for run in runs:
        model = run["model"]
        a, b = [targets[model, split, "near_zero_le_0.1pct_VAL"] for split in ("VAL", "TEST")]
        lines.append(f"| {model} | {a['correct_count']} / {a['winner_count']} | {a['rescued']} / {a['lost']} | {a['new_false_pick_count']} | {b['correct_count']} / {b['winner_count']} |")
    focus = [row for row in data["methods"] if row["model"] in ("R84A", "R84C")
             and row["split"] in ("TRAIN", "VAL")
             and (row["problem"], row["solver"]) in (("TSP", "LEHD"), ("TSP", "DIFUSCO"), ("ATSP", "MATPOENET"))]
    lines += ["", "## TRAIN discrimination examples", "",
              "| Model | Split | Problem / method | Winners | Selected | Correct | Recall % | Winner mean predicted rank |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in focus:
        lines.append(f"| {r['model']} | {r['split']} | {r['problem']} / {r['solver']} | {r['winner_count']} | {r['selected_count']} | {r['correct_count']} | {fmt(r['recall_pct'])} | {fmt(r['winner_mean_rank'])} |")
    lines += ["", "This shows method suppression already on TRAIN, not that all instance information is absent. AUC is only a mechanism metric; raw decision-logit AUC can depend on instance score offsets, so `method_probability_auc` is separately reported from temperature-1 softmax.",
              "The zero-winner mass is not total missed-winner mass: after a single choice, an entire method leaves the strict-zero list. `total_missed_winner_count` in summary.csv measures all errors; fixed-target rescues measure actual improvement.",
              "", "## Files and definitions", "", "- `PRIOR_EXPERIMENTS.pdf`: concise authoritative report.", "- `summary.csv`, `metrics_by_problem.csv`: locked ALL / every-task metrics with TRAIN-fitted SBS and Oracle.", "- `zero_selection_methods.csv`, `method_diagnostics.csv`: actual counts from saved TRAIN/VAL/TEST predictions; probabilities are temperature-1 softmax of saved decision logits, not calibrated probabilities.", "- `fixed_targets.csv`, `target_recovery*.csv`: fixed baseline target sets, recall, precision, real rescues, lost winners, new false picks and regret effects.", "- `paired_changes*.csv`: same-instance corrections/harms and both-wrong cost changes; no uncontrolled cross-release pairing.", "- `train_validation_evidence.csv`, `learning_curves.csv`: selected models and saved training trajectories. Missing predictions remain explicitly missing; absent artifacts never imply zero selections.", "- `EVIDENCE_HANDOFF.json`, `analysis_data.json`, `protocol.json`: machine evidence and exact source/checkpoint provenance.", "", "R80A is CE-only; it is not the original-objective control for R81/R82. R77B is a historical same-data reference. R80B uses batch512 and twice the updates to match sample exposures. R83 vs R80D isolates source-scope changes only to the extent recorded in their protocols. R84 A/B/C/D is the matched new-release comparison.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42})

    def page(title, subtitle):
        fig = plt.figure(figsize=(13.8, 9.7), facecolor="white")
        fig.text(.045, .95, title, fontsize=20, weight="bold", color="#064b50")
        fig.text(.045, .918, subtitle, fontsize=10, color="#40545a")
        return fig

    def table(fig, rows, headers, box, sizes=None, fontsize=9):
        axis = fig.add_axes(box)
        axis.axis("off")
        rendered = axis.table(cellText=rows, colLabels=headers, colWidths=sizes,
                              loc="upper left", cellLoc="left", colLoc="left", bbox=[0,0,1,1])
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(fontsize)
        for (row, column), cell in rendered.get_celld().items():
            cell.set_edgecolor("#dce5e7")
            if row == 0:
                cell.set_facecolor("#deedef")
                cell.set_text_props(weight="bold", color="#064b50")
            elif row % 2 == 0:
                cell.set_facecolor("#f4f7f8")

    with PdfPages(output / "PRIOR_EXPERIMENTS.pdf", metadata={"Title":"R80-R84 locked prior experiments", "Author":"CPU result analysis"}) as pdf:
        for group, title in (("OLD77904", "R80-R83 | Original release"), ("NEW90k", "R84 | Matched NEW90k experiments")):
            chosen = [r for r in runs if r["group"] == group]
            fig = page(title, "One locked VAL checkpoint per run. Native/FP64 winner. ALL = equal weight across 18 tasks.")
            rows = []
            for run in chosen:
                model=run["model"]
                train,val,test=[by_key[model, split] for split in ("TRAIN","VAL","TEST")]
                rows.append([model,fmt(train["top1_pct"]),fmt(val["top1_pct"]),fmt(test["top1_pct"]),
                             fmt(test["mean_cost"],6),fmt(test["actual_regret_pct"],6),
                             f"{val['zero_with_winner_groups']} / {val['zero_winner_mass']:,}"])
            table(fig, rows, ["Model","TRAIN Top1 %","VAL Top1 %","TEST Top1 %","TEST cost","TEST regret %","VAL zero / winners"],
                  [.045,.40,.91,.46],fontsize=10)
            desc="\n".join(f"{r['model']}: {r['description']}" for r in chosen)
            fig.text(.05,.35,desc,fontsize=10,va="top",linespacing=1.7)
            fig.text(.05,.065,"Raw costs cannot be compared across these two data releases. TRAIN-fitted SBS and Oracle are in metrics_by_problem.csv.",fontsize=9)
            pdf.savefig(fig);plt.close(fig)
        fig=page("Zero selection already appears on TRAIN", "Counts below come from actual saved predictions of the selected model, including R84D's matching update0043200 archive.")
        rows=[]
        for run in runs:
            model=run["model"];tr,va,te=[by_key[model,s] for s in ("TRAIN","VAL","TEST")]
            rows.append([model,f"{tr['zero_with_winner_groups']} / {tr['zero_winner_mass']:,}",
                         f"{va['zero_with_winner_groups']} / {va['zero_winner_mass']:,}",
                         f"{te['zero_with_winner_groups']} / {te['zero_winner_mass']:,}",
                         fmt(evidence[model]["train_val_gap_pp"])])
        table(fig,rows,["Model","TRAIN zero / winners","VAL zero / winners","TEST zero / winners","TRAIN-VAL Top1 pp"],
              [.045,.25,.91,.62],fontsize=10)
        fig.text(.05,.18,"A zero group means (problem, solver) with native winner_count > 0 and selected_count = 0.\nWinner mass counts unique instances within that split. A method can leave the zero list after one incorrect pick.\nThese small TRAIN-VAL gaps do not prove the representation is sufficient or that the task has reached an upper bound.",va="top",fontsize=10,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
        fig=page("Fixed target recovery | Correct winners, not selection quotas", "Targets frozen from each release baseline's locked VAL: selected share <=0.1%, native winners >0.")
        rows=[]
        for run in runs:
            model=run["model"];r=targets[model,"VAL","near_zero_le_0.1pct_VAL"]
            rows.append([model,f"{r['correct_count']} / {r['winner_count']}",fmt(r["recall_pct"],3),
                         f"{r['rescued']} / {r['lost']}",str(r["new_false_pick_count"]),fmt(r["new_false_pick_regret_contribution_pp"],6)])
        table(fig,rows,["Model","VAL correct / winners","Recall %","Rescued / lost","New false picks","False-pick regret pp"],
              [.045,.25,.91,.62],fontsize=10)
        fig.text(.05,.18,"Old and NEW90k target sets differ and must not be directly pooled.\nRescue = old missed winner now correctly selected. New false pick = newly selected target on a nonwinner.\nNo variant passes by merely shrinking the number of zero-selected methods.",va="top",fontsize=10,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
        fig=page("R84 matched changes versus A", "VAL determines acceptance; TEST is the final report for the already locked version.")
        rows=[]
        for split in ("VAL","TEST"):
            for model in ("R84B","R84C","R84D"):
                p=next(r for r in data["pairs"] if r["model"]==model and r["baseline"]=="R84A" and r["split"]==split and r["problem"]=="ALL")
                delta=by_key[model,split]["top1_pct"]-by_key["R84A",split]["top1_pct"]
                rows.append([split,model,fmt(delta,5),f"{p['corrected']} / {p['harmed']}",
                             f"{p['both_wrong_cheaper']} / {p['both_wrong_dearer']}",fmt(p["mean_cost_delta"],9),fmt(p["actual_regret_delta_pp"],6)])
        table(fig,rows,["Split","Model","Top1 delta pp","Corrected / harmed","Both wrong cheaper / dearer","Cost delta","Regret delta pp"],
              [.045,.45,.91,.40],fontsize=9)
        fig.text(.05,.36,"R84 all three variants fail the joint VAL gate: Top1 higher, mean_cost not higher, regret not higher.\nR84C's +0.03580pp TEST Top1 comes with higher TEST cost and regret.\nDo not promote a model or adjust its checkpoint/rule using TEST.",va="top",fontsize=11,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
        fig=page("TRAIN winner discrimination | Selected checkpoints", "Large positive support still produces almost no correct argmax choices; this is observable before testing generalization.")
        rows=[[r["model"],r["split"],r["problem"]+" / "+r["solver"],str(r["winner_count"]),str(r["selected_count"]),
               str(r["correct_count"]),fmt(r["recall_pct"],4),fmt(r["winner_mean_rank"],3)] for r in focus]
        table(fig,rows,["Model","Split","Method","Winners","Chosen","Correct","Recall %","Winner rank"],
              [.045,.25,.91,.62],sizes=[.08,.075,.24,.10,.09,.09,.10,.125],fontsize=9)
        fig.text(.05,.18,"R84A LEHD: 14,279 TRAIN winners but only 7 correct choices. ATSP MATPOENET: 6,476 winners, zero choices.\nThis supports investigating TRAIN discrimination / decision competition. It does not prove an accuracy ceiling.\nZero-list winner mass is not the same as all errors: one incorrect choice can remove a whole group from that list.",va="top",fontsize=10,linespacing=1.8)
        pdf.savefig(fig);plt.close(fig)
        fig=page("R84 optimization trajectories", "All saved history shown for diagnosis; markers identify each model's previously locked checkpoint.")
        colors={"R84A":"#157d86","R84B":"#d15d65","R84C":"#416fa3","R84D":"#87803d"}
        for j,(split,key,title,mult) in enumerate((("TRAIN","top1","TRAIN Top1 (%)",100),("VAL","top1","VAL Top1 (%)",100),("TRAIN","ce","TRAIN CE",1),("VAL","ce","VAL CE",1))):
            ax=fig.add_axes([.08+(j%2)*.46,.52-(j//2)*.36,.39,.28])
            for model,color in colors.items():
                points=[r for r in data["history"] if r["model"]==model and r["split"]==split]
                ax.plot([r["update"] for r in points],[r[key]*mult for r in points],color=color,label=model,linewidth=1.5)
                selected=by_key[model,"VAL"]["selected_update"]
                value=by_key[model,split]["top1_pct" if key=="top1" else "ce"]
                ax.scatter([selected],[value],color=color,s=22,zorder=4)
            ax.set_title(title,fontsize=11);ax.set_xlabel("Successful updates");ax.grid(alpha=.18)
            if j==0:ax.legend(ncol=2,fontsize=8)
        fig.text(.05,.07,"Locked TRAIN/VAL are close; final epochs remain around 71%. R84 does not reproduce severe training memorization seen in older binary-only experiments.",fontsize=10)
        pdf.savefig(fig);plt.close(fig)
        fig=page("Evidence and implications", "No new training / GPU inference / checkpoint selection. Single seed, same saved decision rules.")
        y=.86
        for i,fact in enumerate(facts):
            wrapped=textwrap.fill(f"{i+1}. {fact}",width=125)
            fig.text(.05,y,wrapped,fontsize=9.5,va="top",linespacing=1.5)
            y-=.028*(wrapped.count("\n")+1)+.027
        pdf.savefig(fig);plt.close(fig)
    protocol=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),cpu_only=True,
                  GPU_inference=False,training=False,checkpoint_reselection=False,Astra_result_review=False,
                  actual_model_identifier="not exposed by this tool session; no backend identity claim",
                  previous_goal_turn="progress",goal_complete=False,
                  source_files=[dict(model=r["model"],data_root=r["config"]["data_root"],
                                     run=str(r["path"]),checkpoint_sha256=r["lock"]["checkpoint_sha256"],
                                     selected_update=r["lock"]["selected_update"],
                                     metrics=[str(r["path"]/name) for name in ("protocol.json","final_test_lock.json","final_metrics.json","test_results.json")],
                                     predictions={s:str(selected_path(r["path"],s,r["lock"]) or "MISSING") for s in ("TRAIN","VAL","TEST")}) for r in runs],
                  findings=facts,pdf="PRIOR_EXPERIMENTS.pdf",pages=8)
    write_json(output / "protocol.json",protocol)
    handoff_data=read_json(output/"EVIDENCE_HANDOFF.json")
    handoff_data.update(stage="complete_saved_prediction_analysis",report_complete=True,
                        summary_csv=str(output/"summary.csv"),pdf=str(output/"PRIOR_EXPERIMENTS.pdf"),findings=facts,
                        target_VAL_recovery=[{key: row[key] for key in ("model", "baseline", "target_groups", "winner_count", "correct_count", "rescued", "lost", "new_false_pick_count", "rescued_of_baseline_misses_pct")}
                                             for row in data["targets"] if row["split"] == "VAL" and row["target_definition"] == "near_zero_le_0.1pct_VAL"],
                        TRAIN_examples=[{key: row[key] for key in ("model", "problem", "solver", "winner_count", "selected_count", "correct_count", "recall_pct")}
                                        for row in focus if row["split"] == "TRAIN"])
    write_json(output/"EVIDENCE_HANDOFF.json",handoff_data)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--handoff-only", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Regenerate reports from this script's saved analysis_data.json")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    handoff(runs, args.output)
    if not args.handoff_only:
        data = read_json(args.output / "analysis_data.json") if args.render_only else analyze(runs, args.output)
        report(runs, data, args.output)
    print(str(args.output / "EVIDENCE_HANDOFF.json"), flush=True)


if __name__ == "__main__":
    main()
