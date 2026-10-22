"""Bounded CPU-only synthesis of completed, locked NEW90k experiments.

No experiment/runtime imports, checkpoint loading, inference, training or Git.
Only VAL NPZ files are opened; TEST metrics/recovery reuse existing CSVs.
Run with --self-test for tiny synthetic tests, or --output for the supplement.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import resource
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "code/V4/runs"
OUTPUT = RUNS / "R101_hint_pretraining/preparation/completed_context_experiments"
DATA = Path("/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018")
TARGET_PATH = RUNS / "R84_new90k_attention/comparison_artifacts/target_list.json"
FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
MODELS = {
    "R89B": "R89_optimizer_geometry/B_muon_hybrid",
    "R84A": "R84_new90k_attention/A_baseline",
    "R94A": "R94_pretrained_node_fusion/A",
    "R94B": "R94_pretrained_node_fusion/B",
    "R95A": "R95_adaptive_example_sampling/A",
    "R95B": "R95_adaptive_example_sampling/B",
    "R97": "R97_tabular_in_context",
    "R98": "R98_performance_context",
    "R100": "R100_pairwise_context",
}
RECENT = tuple(name for name in MODELS if name not in ("R89B", "R84A"))
METRICS = ("top1_pct", "mean_cost", "sbs_cost", "oracle_cost", "vs_sbs_pct", "vs_oracle_pct", "actual_regret_pct")
COUNTS = ("winner_count", "correct_count", "selected_count", "baseline_missed", "rescued", "lost", "false_pick_count", "new_false_pick_count")
CONTRIBUTIONS = (
    "new_false_pick_regret_contribution_pp", "target_regret_delta_pp",
    "new_false_pick_cost_contribution", "new_false_pick_cost_delta_contribution",
    "target_mean_cost_delta_contribution",
)
LABEL = "saved native/FP64 ind; never substituted by FP32 argmin"
BOUNDARY = (
    "VAL-only answer-aware offline diagnostic; not a deployable ensemble, "
    "Bayes ceiling, checkpoint/rule selection or per-problem model stitching."
)


def number(value):
    return None if value is None or value == "" else float(value)


def close(left, right, label):
    if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Inconsistent saved evidence: {label}: {left} != {right}")


def pct(count, denominator):
    return 100.0 * count / denominator if denominator and count is not None else None


def total_optional(rows, key, divisor=1):
    values = [number(row.get(key)) for row in rows]
    return sum(values) / divisor if all(value is not None for value in values) else None


def write_csv(path, rows):
    if not rows:
        rows = [{"status": "unavailable", "reason": "See sources.json"}]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Sources:
    """Stat receipts, not another pass hashing large inputs."""

    def __init__(self):
        self.paths = {}

    def record(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        receipt = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if str(path) in self.paths and self.paths[str(path)] != receipt:
            raise ValueError(f"Input changed during synthesis: {path}")
        self.paths[str(path)] = receipt
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())

    def csv(self, path):
        with self.record(path).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def finish(self):
        for receipt in list(self.paths.values()):
            self.record(receipt["path"])
        return list(self.paths.values())


def catalog(sources):
    protocols, records = {}, []
    for name, relative in MODELS.items():
        root = RUNS / relative
        protocol = sources.json(root / "protocol.json")
        lock = sources.json(root / "final_test_lock.json")
        if (Path(protocol["data_root"]).resolve() != DATA.resolve()
                or protocol["rows_per_problem"] != {"train": 81000, "val": 4500, "test": 4500}
                or lock.get("selected_without_test") is not True
                or lock.get("per_problem_assembly") is not False):
            raise ValueError(f"Not a locked same-data full18 model: {name}")
        if name in ("R97", "R98", "R100"):
            receipt = sources.json(root / "run_complete.json")
            if not receipt.get("complete") or not receipt.get("full18_VAL") or not receipt.get("full18_TEST"):
                raise ValueError(f"Incomplete context bank: {name}")
        else:
            if not sources.json(root / "postprocess_status.json").get("complete"):
                raise ValueError(f"Incomplete postprocessing: {name}")
            receipt = sources.json(root / "training_complete.json")
            if receipt["successful_updates"] != protocol["max_successful_updates"]:
                raise ValueError(f"Incomplete training: {name}")
        protocols[name] = protocol
        baseline = protocols["R89B"]
        for key in ("candidate_pools", "candidate_pool_ids"):
            if protocol[key] != baseline[key]:
                raise ValueError(f"Different legal candidate pools: {name}/{key}")
        publication = {}
        if (root / "github_publication.json").exists():
            publication = sources.json(root / "github_publication.json")
        expected = {"R98": "07c3adea3", "R100": "6a808ab3b"}.get(name)
        if expected and not publication.get("commit", "").startswith(expected):
            raise ValueError(f"Wrong completed publication receipt: {name}")
        records.append(dict(model=name, run=str(root), data_root=str(DATA),
            selected_update=lock.get("selected_update"),
            selection_rule=lock.get("rule", lock.get("selection_rule")),
            checkpoint_sha256=lock["checkpoint_sha256"], state_hash=lock["state_hash"],
            optimizer_updates=protocol.get("max_successful_updates", protocol.get("optimizer_updates")),
            logical_batch=protocol.get("global_batch_size"),
            total_sample_exposures=protocol.get("total_sample_exposures"),
            public_pretraining_matched=protocol.get("pretraining_budget_matched"),
            historical_train_context=protocol.get("context_rows"),
            donor_variant=protocol.get("frozen_donor", {}).get("variant"),
            donor_pretraining_source=protocol.get("frozen_donor", {}).get("pretraining_source"),
            published_commit=publication.get("commit"), winner_rule=LABEL,
            val_prediction=str(root / "evaluation/val/selected_best/rank000.npz")))
    return protocols, records


def metrics(sources, protocols):
    rows, indexed = [], {}
    problems = list(protocols["R89B"]["candidate_pools"])
    if len(problems) != 18:
        raise ValueError("Require the complete 18-problem scope")
    for name, relative in MODELS.items():
        path = RUNS / relative / "full_model_statistics/metrics_by_problem.csv"
        selected = [row for row in sources.csv(path)
                    if row["split"].upper() in ("VAL", "TEST") and row["problem"] in ("ALL", *problems)]
        if len(selected) != 38:
            raise ValueError(f"Missing ALL/per-problem VAL/TEST metrics: {name}")
        for row in selected:
            item = dict(model=name, split=row["split"].upper(), problem=row["problem"], n=int(row["n"]),
                        **{key: float(row[key]) for key in METRICS}, source=str(path),
                        winner_rule=LABEL, data_version=DATA.name,
                        candidate_pool="per-problem frozen legal pools", aggregation="equal weight across 18 problems" if row["problem"] == "ALL" else "single problem",
                        sbs_selection_split=row.get("sbs_selection_split", "FULL TRAIN"))
            indexed[name, item["split"], item["problem"]] = item
            rows.append(item)
        for split in ("VAL", "TEST"):
            all_row = indexed[name, split, "ALL"]
            if all_row["n"] != 81000:
                raise ValueError("Expected 81000 heldout rows, 4500 per problem")
            for field in METRICS:
                close(all_row[field], mean(indexed[name, split, p][field] for p in problems), name + "/" + field)
    for row in rows:
        for reference in ("R89B", "R84A"):
            old = indexed[reference, row["split"], row["problem"]]
            for key, suffix in (("top1_pct", "top1_delta_pp"), ("mean_cost", "mean_cost_delta"), ("actual_regret_pct", "actual_regret_delta_pp")):
                row[f"vs_{reference}_{suffix}"] = row[key] - old[key]
    return problems, rows, indexed


def normalized_target(row, name, baseline, source, target_lookup):
    problem, solver = row["problem"], row.get("solver", row.get("method"))
    if (problem, solver) not in target_lookup:
        raise ValueError("Existing comparison contains a non-Fixed36 group")
    result = dict(model=name, baseline=baseline, split=row["split"].upper(), problem=problem,
                  solver=solver, category=target_lookup[problem, solver]["category"], n=4500,
                  donor_covers_R94B=problem not in ("TSP", "ATSP"), source=str(source))
    for field in COUNTS:
        result[field] = int(row[field]) if row.get(field) not in (None, "") else None
    for field in CONTRIBUTIONS:
        result[field] = number(row.get(field))
    if result["baseline_missed"] is None:
        result["baseline_missed"] = result["winner_count"] - (result["correct_count"] - result["rescued"] + result["lost"])
    result["recall_pct"] = pct(result["correct_count"], result["winner_count"])
    return result


def existing_targets(sources, targets):
    lookup = {(row["problem"], row["method"]): row for row in targets}
    rows = {}
    for name in ("R94A", "R94B", "R95A", "R95B", "R97", "R98", "R100", "R89B"):
        stats = RUNS / MODELS[name] / "full_model_statistics"
        paths = [(baseline, stats / f"target_recovery_vs_{baseline}.csv") for baseline in ("R89B", "R84A")]
        if name in ("R97", "R98", "R100", "R89B"):
            paths = [(None, stats / "target_recovery.csv")]
        for baseline, path in paths:
            for row in sources.csv(path):
                if row["split"].upper() not in ("VAL", "TEST"):
                    continue
                reference = baseline or ("R84A" if name == "R89B" else row["baseline"])
                item = normalized_target(row, name, reference, path, lookup)
                rows[name, reference, item["split"], item["problem"], item["solver"]] = item
    path = RUNS / "R84_new90k_attention/comparison_artifacts/method_precision_recall.csv"
    for original in sources.csv(path):
        key = original["problem"], original["method"]
        if original["arm"] != "A_baseline" or key not in lookup:
            continue
        row = dict(split=original["split"], problem=key[0], solver=key[1],
            winner_count=original["winner_n"], correct_count=original["true_positive_n"],
            selected_count=original["selected_n"], false_pick_count=original["false_picks_n"],
            baseline_missed=int(original["winner_n"]) - int(original["true_positive_n"]),
            rescued=0, lost=0, new_false_pick_count=0,
            **{field: 0.0 for field in CONTRIBUTIONS})
        item = normalized_target(row, "R84A", "R84A", path, lookup)
        rows[("R84A", "R84A", item["split"], *key)] = item
        old = rows[("R89B", "R84A", item["split"], *key)]
        reverse = dict(item, baseline="R89B", rescued=old["lost"], lost=old["rescued"],
            baseline_missed=item["winner_count"] - old["correct_count"],
            new_false_pick_count=item["false_pick_count"] - (old["false_pick_count"] - old["new_false_pick_count"]),
            target_regret_delta_pp=-old["target_regret_delta_pp"],
            new_false_pick_regret_contribution_pp=None,
            new_false_pick_cost_contribution=None, new_false_pick_cost_delta_contribution=None,
            target_mean_cost_delta_contribution=None,
            source=str(path) + " + reversed counts from " + old["source"])
        rows[("R84A", "R89B", item["split"], *key)] = reverse
        self_row = dict(old, baseline="R89B", rescued=0, lost=0,
                        baseline_missed=old["winner_count"] - old["correct_count"],
                        new_false_pick_count=0, **{field: 0.0 for field in CONTRIBUTIONS})
        rows[("R89B", "R89B", item["split"], *key)] = self_row
    return rows


def stable_top1(scores, ids, ascending=False):
    scores, ids = np.asarray(scores), np.asarray(ids)
    if (scores.ndim != 2 or ids.shape != (scores.shape[1],)
            or len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all()):
        raise ValueError("Need finite full legal-pool scores and unique global IDs")
    order = np.argsort(ids, kind="stable")
    values = scores if ascending else -scores
    return order[np.argmin(values[:, order], axis=1)]


def load_val(archive, model, problem):
    fields = ("indices", "source_indices", "release_source_indices", "nodes", "pool", "pool_ids", "costs", "winner", "pred")
    data = {key: archive[problem + "__" + key] for key in fields}
    field = "expected_gaps" if model == "R98" else "mean_opponent_win" if model == "R100" else "logits"
    scores = archive[problem + "__" + field]
    if not np.array_equal(stable_top1(scores, data["pool_ids"], model == "R98"), data["pred"]):
        raise ValueError(f"Saved prediction disagrees with its locked score rule: {model}/{problem}")
    costs, winner = data["costs"], data["winner"]
    if (costs.dtype != np.float64 or costs.shape[0] != 4500 or scores.shape != costs.shape
            or not np.isfinite(costs).all() or (costs <= 0).any()
            or winner.shape != (4500,) or winner.dtype.kind not in "iu"
            or (winner < 0).any() or (winner >= costs.shape[1]).any()
            or not np.array_equal(data["indices"], np.arange(4500))
            or not np.array_equal(costs[np.arange(4500), winner], costs.min(1))):
        raise ValueError(f"Invalid native/FP64 validation arrays: {model}/{problem}")
    return data


def target_stats(data, pred, reference, target, model, baseline):
    slot = data["pool"].tolist().index(target["method"])
    truth, picks, old = data["winner"] == slot, pred == slot, reference == slot
    selected, before = data["costs"][np.arange(4500), pred], data["costs"][np.arange(4500), reference]
    oracle = data["costs"].min(1)
    regret, delta = (selected - oracle) / oracle, selected - before
    new_false = picks & ~truth & ~old
    return dict(model=model, baseline=baseline, split="VAL", problem=target["problem"], solver=target["method"],
        category=target["category"], n=4500, donor_covers_R94B=target["problem"] not in ("TSP", "ATSP"),
        winner_count=int(truth.sum()), correct_count=int((truth & picks).sum()),
        selected_count=int(picks.sum()), baseline_missed=int((truth & ~old).sum()),
        rescued=int((truth & ~old & picks).sum()), lost=int((truth & old & ~picks).sum()),
        false_pick_count=int((picks & ~truth).sum()), new_false_pick_count=int(new_false.sum()),
        recall_pct=pct(int((truth & picks).sum()), int(truth.sum())),
        new_false_pick_regret_contribution_pp=float(100 * regret[new_false].sum() / 4500),
        target_regret_delta_pp=float(100 * (delta / oracle)[truth].sum() / 4500),
        new_false_pick_cost_contribution=float(selected[new_false].sum() / 4500),
        new_false_pick_cost_delta_contribution=float(delta[new_false].sum() / 4500),
        target_mean_cost_delta_contribution=float(delta[truth].sum() / 4500),
        source="saved VAL NPZ; exact identity + locked full-score Top1 checked")


def coverage(correct, mask):
    reference = correct[0, mask]
    recent = correct[2:, mask]
    misses = ~reference
    rescue = recent & misses[None]
    union = rescue.any(0)
    result = dict(n=int(mask.sum()), reference_correct=int(reference.sum()), reference_misses=int(misses.sum()),
        rescued_reference_misses=int(union.sum()), common_misses=int((misses & ~union).sum()),
        reference_plus_recent_union_correct=int((reference | recent.any(0)).sum()),
        reference_plus_recent_plus_R84A_union_correct=int(correct[:, mask].any(0).sum()))
    result["rescued_of_reference_misses_pct"] = pct(result["rescued_reference_misses"], result["reference_misses"])
    result["reference_plus_recent_union_pct"] = pct(result["reference_plus_recent_union_correct"], result["n"])
    result["R84A_only_additional"] = result["reference_plus_recent_plus_R84A_union_correct"] - result["reference_plus_recent_union_correct"]
    models = []
    for i, name in enumerate(RECENT):
        models.append(dict(model=name, reference_misses=result["reference_misses"],
            rescued=int(rescue[i].sum()), unique_rescue_among_recent=int((rescue[i] & (rescue.sum(0) == 1)).sum()),
            lost_reference_correct=int((reference & ~recent[i]).sum())))
    pairs = []
    for i, j in itertools.combinations(range(len(RECENT)), 2):
        shared, both = int((rescue[i] & rescue[j]).sum()), int((rescue[i] | rescue[j]).sum())
        pairs.append(dict(first=RECENT[i], second=RECENT[j], first_rescues=int(rescue[i].sum()),
            second_rescues=int(rescue[j].sum()), shared_rescues=shared, union_rescues=both,
            first_only=int((rescue[i] & ~rescue[j]).sum()), second_only=int((rescue[j] & ~rescue[i]).sum()),
            rescue_jaccard_pct=pct(shared, both)))
    return result, models, pairs


def validation(sources, problems, targets, stored_targets, metric_index):
    coverage_rows, model_rows, pair_rows, target_coverage = [], [], [], []
    generated, matrices, masks = {}, [], []
    with ExitStack() as stack:
        archives = {}
        for model, relative in MODELS.items():
            path = (RUNS / relative / "evaluation/val/selected_best/rank000.npz").resolve()
            if "/evaluation/val/" not in str(path):
                raise PermissionError("NPZ reads are restricted to VAL")
            archives[model] = stack.enter_context(np.load(sources.record(path), allow_pickle=False))
        for problem in problems:
            reference, predictions = None, []
            for model, archive in archives.items():
                data = load_val(archive, model, problem)
                if reference is None:
                    reference = data
                elif any(not np.array_equal(data[field], reference[field]) for field in reference if field != "pred"):
                    raise ValueError(f"Different native VAL row/candidate identity: {model}/{problem}")
                pred = data["pred"]
                predictions.append(pred)
                selected = data["costs"][np.arange(4500), pred]
                oracle = data["costs"].min(1)
                row = metric_index[model, "VAL", problem]
                for key, value in (("top1_pct", 100 * (pred == data["winner"]).mean()),
                                   ("mean_cost", selected.mean()),
                                   ("actual_regret_pct", 100 * ((selected - oracle) / oracle).mean())):
                    close(value, row[key], model + "/VAL/" + problem + "/" + key)
            correct = np.stack(predictions) == reference["winner"][None]
            target_mask = np.zeros(4500, dtype=bool)
            for target in (t for t in targets if t["problem"] == problem):
                slot = reference["pool"].tolist().index(target["method"])
                mask = reference["winner"] == slot
                target_mask |= mask
                info, _, _ = coverage(correct, mask)
                target_coverage.append(dict(problem=problem, solver=target["method"], **info))
                for i, model in enumerate(MODELS):
                    for j, baseline in enumerate(("R89B", "R84A")):
                        item = target_stats(reference, predictions[i], predictions[j], target, model, baseline)
                        key = model, baseline, "VAL", problem, target["method"]
                        if key in stored_targets:
                            saved = stored_targets[key]
                            for field in (*COUNTS, *CONTRIBUTIONS[:2]):
                                if saved.get(field) is not None:
                                    close(item[field], saved[field], str(key) + "/" + field)
                            # Keep existing comparison values; supplement only missing raw-cost fields.
                            item.update({field: value for field, value in saved.items()
                                         if value is not None and field not in CONTRIBUTIONS[2:]})
                        generated[key] = item
            matrices.append(correct)
            masks.append(target_mask)
            for scope, mask in (("full18", np.ones(4500, dtype=bool)), ("Fixed36", target_mask)):
                info, individual, _ = coverage(correct, mask)
                coverage_rows.append(dict(problem=problem, scope=scope, **info))
                model_rows.extend(dict(problem=problem, scope=scope, **item) for item in individual)
    correct, mask = np.concatenate(matrices, axis=1), np.concatenate(masks)
    for scope, selected in (("full18", np.ones(correct.shape[1], dtype=bool)), ("Fixed36", mask)):
        info, individual, pairs = coverage(correct, selected)
        coverage_rows.append(dict(problem="ALL", scope=scope, **info))
        model_rows.extend(dict(problem="ALL", scope=scope, **item) for item in individual)
        pair_rows.extend(dict(problem="ALL", scope=scope, **item) for item in pairs)
    return generated, coverage_rows, model_rows, pair_rows, target_coverage


def aggregate_targets(rows, task_count=18):
    result = {field: total_optional(rows, field) for field in COUNTS}
    result.update({field: total_optional(rows, field, task_count) for field in CONTRIBUTIONS})
    for field in COUNTS:
        if result[field] is not None:
            result[field] = int(result[field])
    result.update(target_groups=len(rows), groups_with_genuine_recall=(sum(row["correct_count"] > 0 for row in rows)
        if all(row["correct_count"] is not None for row in rows) else None),
        net_recovery=(result["rescued"] - result["lost"] if result["rescued"] is not None and result["lost"] is not None else None),
        recall_pct=pct(result["correct_count"], result["winner_count"]),
        precision_pct=pct(result["correct_count"], result["selected_count"]))
    return result


def target_summaries(rows):
    output = []
    for model, baseline, split in itertools.product(MODELS, ("R89B", "R84A"), ("VAL", "TEST")):
        selected = [r for r in rows if (r["model"], r["baseline"], r["split"]) == (model, baseline, split)]
        if len(selected) != 36:
            raise ValueError(f"Incomplete fixed target rows: {model}/{baseline}/{split}")
        output.append(dict(model=model, baseline=baseline, split=split, **aggregate_targets(selected)))
    return output


def r94_domains(rows):
    output = []
    for split, domain in itertools.product(("VAL", "TEST"), ("TSP", "CVRP", "ATSP", "15MVRP", "ALL")):
        selected = [row for row in rows if row["model"] == "R94B" and row["baseline"] == "R89B" and row["split"] == split
                    and (domain == "ALL" or (row["problem"] == domain if domain != "15MVRP" else row["problem"] not in ("TSP", "CVRP", "ATSP")))]
        count = 18 if domain == "ALL" else 15 if domain == "15MVRP" else 1
        result = aggregate_targets(selected, count)
        output.append(dict(model="R94B", baseline="R89B", split=split, domain=domain, donor_path="mixed" if domain == "ALL" else "raw" if domain in ("TSP", "ATSP") else "covered",
                           domain_problem_count=count, **result,
                           all18_new_false_pick_regret_contribution_pp=total_optional(selected, CONTRIBUTIONS[0], 18),
                           all18_target_regret_delta_pp=total_optional(selected, CONTRIBUTIONS[1], 18)))
    return output


def fmt(value, decimals=4, signed=False):
    if value is None or value == "":
        return "缺失"
    if isinstance(value, str):
        return value
    return format(value, f"{'+' if signed else ''}.{decimals}f")


def count_ratio(numerator, denominator):
    return "缺失" if numerator is None or denominator is None else f"{numerator}/{denominator}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(str(v) for v in row) + " |" for row in rows)])


def report_content(metric_index, summaries, domains, target_rows, overlaps, mechanism_rows, pairs, val_status):
    metric_tables, target_tables = {}, {}
    for split in ("VAL", "TEST"):
        metric_tables[split] = [[model, fmt((r := metric_index[model, split, "ALL"])["top1_pct"], 6),
            fmt(r["vs_R89B_top1_delta_pp"], 6, True), fmt(r["mean_cost"], 9), fmt(r["vs_R89B_mean_cost_delta"], 9, True),
            fmt(r["actual_regret_pct"], 6), fmt(r["vs_R89B_actual_regret_delta_pp"], 6, True)] for model in MODELS]
        target_tables[split] = []
        for model in MODELS:
            row = next(r for r in summaries if (r["model"], r["baseline"], r["split"]) == (model, "R89B", split))
            target_tables[split].append([model, count_ratio(row["correct_count"], row["winner_count"]), fmt(row["recall_pct"]),
                count_ratio(row["rescued"], row["lost"]), row["new_false_pick_count"] if row["new_false_pick_count"] is not None else "缺失",
                fmt(row[CONTRIBUTIONS[0]], 6), fmt(row[CONTRIBUTIONS[1]], 6, True)])
    r = metric_index["R94B", "TEST", "ALL"]
    domain = {r["domain"]: r for r in domains if r["split"] == "TEST"}
    selected = [row for row in target_rows if (row["model"], row["baseline"], row["split"]) == ("R94B", "R89B", "TEST")]
    related = sum(row["rescued"] for row in selected if row["solver"] in ("MTPOMO", "MVMOE"))
    total = domain["ALL"]
    conclusions = ['R94B TEST 固定目标净救回：' + '；'.join(f'{key} {domain[key]["rescued"]}-{domain[key]["lost"]}={domain[key]["net_recovery"]:+d}' for key in ("TSP", "CVRP", "ATSP", "15MVRP")) + f'。{related}/{total["rescued"]}救回来自MTPOMO/MVMOE；TSP/ATSP是raw路径，不能外推为全域donor收益。']
    conclusions.append(f'R94B TEST ALL：Top1 {r["vs_R89B_top1_delta_pp"]:+.6f}pp，成本 {r["vs_R89B_mean_cost_delta"]:+.9f}，regret {r["vs_R89B_actual_regret_delta_pp"]:+.6f}pp。固定目标新增误选{total["new_false_pick_count"]}例，代价{total[CONTRIBUTIONS[0]]:+.6f}pp；单seed、与R89B非匹配对照，不作因果结论。')
    conclusions.append('R97 TEST Top1 略升但成本/regret 更差；R98 与 R100 的 Top1 下降而部分成本指标改善，均不是整体胜出。R95 自适应采样在本协议下未改善相对 R89B 的三项 ALL 指标。')
    if val_status["available"]:
        full = next(r for r in overlaps if r["problem"] == "ALL" and r["scope"] == "full18")
        fixed = next(r for r in overlaps if r["problem"] == "ALL" and r["scope"] == "Fixed36")
        own = next(r for r in mechanism_rows if (r["model"], r["scope"], r["problem"]) == ("R94B", "Fixed36", "ALL"))
        icl = [r["rescue_jaccard_pct"] for r in pairs if r["scope"] == "Fixed36" and r["first"] in ("R97", "R98") and r["second"] in ("R98", "R100")]
        conclusions.append(f'VAL 救回并集：全域 {full["rescued_reference_misses"]}/{full["reference_misses"]}，Fixed36 {fixed["rescued_reference_misses"]}/{fixed["reference_misses"]}（仍漏{fixed["common_misses"]}）。R94B目标救回{own["rescued"]}中{own["unique_rescue_among_recent"]}例独有；三种ICL目标救回Jaccard={min(icl):.2f}%-{max(icl):.2f}%，高度重叠。')
    else:
        conclusions.append('VAL 互补性不可用：' + val_status["reason"] + '；未做推理或进一步审计。')
    conclusions.append('下一步应区分 donor 覆盖域与 raw 域，并优先检验真实目标救回是否抵消丢失与误选代价；保持单一完整模型、预先固定 VAL 选型。公共预训练与上下文读取开销不能折算成零预算，历史 TEST 不是盲测。')
    return metric_tables, target_tables, conclusions


def create_report(output, problems, metric_index, summaries, domain_rows, target_rows, overlaps, mechanism_rows, pairs, definition, records, val_status):
    os.environ["MPLCONFIGDIR"] = str(output / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages

    if not FONT.is_file():
        raise FileNotFoundError("Required installed CJK font is unavailable")
    font_manager.fontManager.addfont(str(FONT))
    font = font_manager.FontProperties(fname=str(FONT))
    matplotlib.rcParams.update({"font.family": font.get_name(), "pdf.fonttype": 42, "axes.unicode_minus": False})
    metric_tables, target_tables, conclusions = report_content(metric_index, summaries, domain_rows, target_rows, overlaps, mechanism_rows, pairs, val_status)
    mh = ["模型", "Top1(%)", "差(pp)", "mean_cost", "成本差", "regret(%)", "差(pp)"]
    th = ["模型", "真中 / 真赢家", "召回(%)", "救回 / 丢失", "新增误选", "误选代价(pp)", "目标regret差(pp)"]
    dh = ["划分", "域 / 路径", "目标组", "真中 / 真赢家", "救回 / 丢失", "新增误选", "误选代价(pp)", "目标regret差(pp)"]
    dtable = [[r["split"], r["domain"] + "/" + r["donor_path"], r["target_groups"], f'{r["correct_count"]}/{r["winner_count"]}',
        f'{r["rescued"]}/{r["lost"]}', r["new_false_pick_count"], fmt(r[CONTRIBUTIONS[0]], 6), fmt(r[CONTRIBUTIONS[1]], 6, True)]
        for r in domain_rows if r["domain"] != "ALL"]
    indexed = {(r["model"], r["baseline"], r["split"], r["problem"], r["solver"]): r for r in target_rows}
    details = []
    for target in definition:
        p, solver = target["problem"], target["method"]
        val, test = (indexed["R94B", "R89B", split, p, solver] for split in ("VAL", "TEST"))
        details.append([p + "/" + solver, "raw" if p in ("TSP", "ATSP") else "donor",
            f'{val["correct_count"]}/{val["winner_count"]}', f'{val["rescued"]}/{val["lost"]}', val["new_false_pick_count"], fmt(val[CONTRIBUTIONS[1]], 5, True),
            f'{test["correct_count"]}/{test["winner_count"]}', f'{test["rescued"]}/{test["lost"]}', test["new_false_pick_count"], fmt(test[CONTRIBUTIONS[1]], 5, True)])
    detail_headers = ["固定问题 / 目标方法", "路径", "VAL真中/n", "救/失", "新误选", "目标差(pp)", "TEST真中/n", "救/失", "新误选", "目标差(pp)"]
    budget_rows = []
    mechanisms = {"R89B": "Muon 混合优化", "R84A": "原始基准", "R94A": "固定随机 donor", "R94B": "预训练 ReLD donor", "R95A": "均匀 IID 替换采样", "R95B": "校正自适应采样", "R97": "冻结 TabICL winner 分类", "R98": "冻结 TabICL gap 回归", "R100": "冻结 TabICL 二元比较"}
    for r in records:
        budget_rows.append([r["model"], mechanisms[r["model"]], r["optimizer_updates"],
            r["selected_update"] if r["selected_update"] is not None else "预先固定上下文 bank",
            "未匹配外部预训练" if r["model"] in ("R94B", "R97", "R98", "R100") else "本地 57600 x 1024"])
    cli = ("eval \"$(conda shell.bash hook)\"\nconda activate easynco\n"
           "python -B code/V4/r101_completed_context_synthesis.py \\\n"
           "  --output code/V4/runs/R101_hint_pretraining/preparation/completed_context_experiments\n"
           "python -B code/V4/r101_completed_context_synthesis.py --self-test")
    notes = [
        f"数据：{DATA}。18题各 TRAIN/VAL/TEST=81000/4500/4500，VAL/TEST 各81000。原始10000例不纳入。",
        "候选池：8个带B问题删除 RouteFinder、MoSES_RF、MoSES_CaDA；删除 RELD_MTL；原 RELD_MOEL 单独命名 reld_mt，不取两种ReLD最低值。",
        "标签：保存的 native/FP64 ind。FP64 成本仅验证 winner 是合法池最低值；不以 FP32 argmin 替代，不把选择比例当准确率。",
        "ALL：逐问题等权；mean_cost 是18题均值的描述性均值，不能用混合 raw cost 掩盖单题差异。各题 SBS 在完整 TRAIN 拟合；Oracle 为逐例合法池最低成本。",
        "actual regret = 100*mean((selected-Oracle)/Oracle)；vs_SBS/vs_Oracle 为各题均值比再等权，不等于 actual regret。CSV 保留原始数值精度。",
        "每个版本使用已锁定唯一 checkpoint/bank 与原推理规则。本次不依 TEST/LIB 选型，不做逐题模型拼接，不做 ensemble、不画额外分布。历史 TEST 已参与研发，不称盲测。",
        "固定目标来自 R84A 锁定 VAL：选择比例<=0.1%且 native winner>0，共36个问题-方法组。比例仅定义历史名单，不是本轮配额或成功指标。",
        "救回/丢失以该目标方法的真实 winner 为集合；新增误选=本模型选该目标、真实非该目标、参考未选该目标。代价为这些误选的 regret 贡献，而非全部误选都更贵。",
        "目标 regret 差在真实目标 winner 上累加，以所在问题全部4500例归一化；ALL 再除18。域表按该域1题或15题等权，未出现目标的域内问题仍占分母。",
        "TEST 只复用已有 CSV；R84A 相对 R89B 的新增误选 regret 无法从既有表反推，显式缺失；不读取 TEST NPZ 补齐。VAL 另给新增误选 raw-cost 及相对参考的 raw-cost 贡献。",
        "R94B donor 只覆盖 CVRP+15MVRP；TSP/ATSP 是 raw 路径。预训练来源/训练实例重叠记录不完整。R97/98/100 用全部历史 TRAIN 上下文，无本地SGD，但公共预训练预算未匹配。",
        BOUNDARY,
    ]
    md = ["# 已完成上下文与预训练实验补充分析", "", "## 口径", *["- " + n for n in notes], "", "## ALL 指标与 R89B 差值"]
    for split in ("VAL", "TEST"):
        md.extend(["", "### " + split, md_table(mh, metric_tables[split])])
    md.append("\n## Fixed36 真实召回与误选代价")
    for split in ("VAL", "TEST"):
        md.extend(["", "### " + split, md_table(th, target_tables[split])])
    md.extend(["", "### R94B 收益位置", md_table(dh, dtable), "", "### 固定36项与 R94B 方法级结果", md_table(detail_headers, details)])
    overlap_table, model_table = [], []
    oh = ["问题", "N", "R89错", "救回并集", "共同错", "目标N", "目标错", "目标救回", "目标共同错"]
    uh = ["模型", "全域救回", "独有救回", "全域丢失", "目标救回", "目标独有", "目标丢失"]
    if val_status["available"]:
        oi = {(r["problem"], r["scope"]): r for r in overlaps}
        for p in ("ALL", *problems):
            a, b = oi[p, "full18"], oi[p, "Fixed36"]
            overlap_table.append([p, a["n"], a["reference_misses"], a["rescued_reference_misses"], a["common_misses"], b["n"], b["reference_misses"], b["rescued_reference_misses"], b["common_misses"]])
        for model in RECENT:
            a, b = (next(r for r in mechanism_rows if r["model"] == model and r["problem"] == "ALL" and r["scope"] == scope) for scope in ("full18", "Fixed36"))
            model_table.append([model, a["rescued"], a["unique_rescue_among_recent"], a["lost_reference_correct"], b["rescued"], b["unique_rescue_among_recent"], b["lost_reference_correct"]])
        md.extend(["", "## VAL 离线互补性", BOUNDARY,
            "七个近期模型=R94A/B、R95A/B、R97、R98、R100；主并集=R89B+七模型。R84A不混入主并集，其额外覆盖单列CSV。独有救回是七模型中仅该模型命中的R89B错误。",
            md_table(oh, overlap_table), "", md_table(uh, model_table), "",
            "逐对重叠见 val_pairwise_rescue_overlap.csv：仅在 R89B 原错集合上计算救回交集/并集/Jaccard；不是全体正确预测的相似度。",
            md_table(["模型对", "范围", "共享救回", "并集", "Jaccard(%)"], [[r["first"]+"/"+r["second"], r["scope"], r["shared_rescues"], r["union_rescues"], fmt(r["rescue_jaccard_pct"])] for r in pairs])])
    else:
        md.extend(["", "## VAL 离线互补性", "不可用：" + val_status["reason"]])
    md.extend(["", "## 结论与边界", *[f"{i}. {s}" for i, s in enumerate(conclusions, 1)], "", "## 预算与来源", md_table(["模型", "机制", "本地更新", "锁定更新/规则", "预算边界"], budget_rows), "",
        "原始来源逐文件见 sources.json，模型根目录与锁定 SHA 见 model_sources.csv；metrics_by_problem.csv、现有 per-model target_recovery CSV 优先。未重复计算大输入 hash，现存 SHA 仅作为来源声明而非本次重验。",
        "R98 发布凭据 07c3adea389d1926a9625598afeb73cab3f67b1d；R100 发布凭据 6a808ab3b056fce467eecac1d59d5d62914b8227。已完成 CPU 报告恢复，本次仅读取恢复后完整统计，不重新恢复或发布。",
        "CPU affinity最多2核，OMP/BLAS各2线程；逐问题读取VAL数组，不读取权重/训练缓存；无GPU/SSH/Slurm/训练/推理/Git。未改运行中模块、launch specs、核心报告或master state。", "", "## 精确 CLI", "从仓库根目录执行；脚本内固定 CPU2/OMP2。", "```bash", cli, "```", ""])
    (output / "ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")

    def page(title, page_number):
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.035, .947, title, fontsize=17, fontproperties=font, color="#123a48")
        fig.text(.035, .912, "冻结 NEW90k | native/FP64 ind | 主参考 R89B，次参考 R84A | 仅已完成版本", fontsize=9, fontproperties=font)
        fig.text(.965, .018, f"R101 报告补充 | {page_number}/5", fontsize=8, ha="right", fontproperties=font)
        return fig

    def text(fig, x, y, string, size=9):
        fig.text(x, y, string, fontsize=size, fontproperties=font, va="top", linespacing=1.45)

    def table(fig, headers, cells, bounds, size=8.5, widths=None):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        tab = ax.table(cellText=cells, colLabels=headers, bbox=[0, 0, 1, 1], cellLoc="center", colWidths=widths)
        tab.auto_set_font_size(False)
        for (row, _), cell in tab.get_celld().items():
            cell.get_text().set_fontproperties(font)
            cell.get_text().set_fontsize(size)
            cell.set_linewidth(.35)
            cell.set_edgecolor("#d4dfe2")
            cell.set_facecolor("#e0edf0" if row == 0 else "#f4f7f8" if row % 2 else "white")
            if row == 0:
                cell.get_text().set_color("#123a48")

    with PdfPages(output / "ANALYSIS.pdf", metadata={"Title": "Completed NEW90k context experiments", "Author": "CPU-only reporting supplement"}) as pdf:
        fig = page("1. ALL：Top1、成本与 regret 不合并成单一胜负", 1)
        text(fig, .035, .872, "VAL：各题4500例，共81000；差值均相对 R89B。", 10)
        table(fig, mh, metric_tables["VAL"], [.035, .565, .93, .275], 9)
        text(fig, .035, .53, "TEST：同一已锁定版本；只读取现成指标，不据此重新选择。", 10)
        table(fig, mh, metric_tables["TEST"], [.035, .222, .93, .275], 9)
        text(fig, .035, .19, "ALL按18题等权；跨问题 raw mean_cost 仅描述。逐题完整 Top1/cost/SBS/Oracle/vs_SBS/vs_Oracle/regret 在CSV。\n"
             "actual regret=逐例相对 Oracle 的均值(%)，与均值比不同；SBS 固定于完整 TRAIN，Oracle 为逐例合法候选最低成本。\n"
             "本页显示6/9位小数，CSV保留源值精度。TEST历史成员已参与研发，不是全新盲测。\n"
             "数据：selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018；每题 TRAIN/VAL/TEST=81000/4500/4500。\n"
             "池：八个带B问题删除 RouteFinder/MoSES_RF/MoSES_CaDA；删除 RELD_MTL；原 RELD_MOEL 单独命名 reld_mt。", 8.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("2. Fixed36：真实 winner 救回在哪里，付出什么代价？", 2)
        text(fig, .035, .875, "VAL（上）/ TEST（下）：固定 R84A VAL 名单；救回、丢失和新增误选均相对 R89B。", 9)
        table(fig, th, target_tables["VAL"], [.035, .627, .93, .218], 8.5)
        table(fig, th, target_tables["TEST"], [.035, .38, .93, .218], 8.5)
        text(fig, .035, .353, "R94B 按域拆解：只有 CVRP+15MVRP 被 donor 直接覆盖，TSP/ATSP 保持 raw 路径。", 10)
        table(fig, dh, dtable, [.035, .132, .93, .19], 8)
        text(fig, .035, .107, "误选代价=新增误选的 actual-regret 贡献；目标差=真实目标winner上的 regret 净变化，不是配额。\n"
             "上两表贡献按18题归一；域表按域内1或15题归一，不可直接求和。R84A TEST新增误选代价缺失，未读取TEST预测补齐。", 8.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("3. VAL：近期机制救回不同错误，还是重复命中？", 3)
        if val_status["available"]:
            table(fig, oh, overlap_table, [.035, .36, .93, .50], 8.5)
            a, b = oi["ALL", "full18"], oi["ALL", "Fixed36"]
            text(fig, .035, .331,
                 f'离线命中并集：全域 {a["reference_plus_recent_union_correct"]}/{a["n"]}={a["reference_plus_recent_union_pct"]:.6f}%；目标 {b["reference_plus_recent_union_correct"]}/{b["n"]}={b["reference_plus_recent_union_pct"]:.6f}%。\n'
                 "主并集=R89B+七个近期完整模型；共同错=R89B原错且七模型均错。目标只含固定36项的真实winner。\n"
                 "独有救回=七模型中仅本模型正确的R89B错例，不保证预测时可识别。逐对Jaccard、R84A额外覆盖见CSV。", 8.2)
            table(fig, uh, model_table, [.035, .101, .93, .163], 8.5)
        else:
            text(fig, .035, .82, "不可用：" + val_status["reason"], 11)
        text(fig, .035, .075, "仅离线诊断，使用真实答案判断覆盖：不是可部署 ensemble、不是 Bayes ceiling，也不构成任何模型拼接结果。", 9)
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("4. 固定36项：R94B 方法级救回 / 丢失与路径边界", 4)
        text(fig, .035, .878, "n=该方法真实winner数；真中/n=真正召回；新误选不是命中。完整代价、次参考对照与并集覆盖另存CSV。", 9)
        table(fig, detail_headers, details, [.035, .095, .93, .75], 7.9,
              [.24, .05, .1, .06, .065, .085, .1, .06, .065, .085])
        text(fig, .035, .071, "固定名单不随本轮结果改变；目标差(pp)以对应问题全部4500例归一。raw域变化不能直接证明 donor 直接迁移。", 8.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("5. 结论、预算边界与可复现来源", 5)
        for i, sentence in enumerate(conclusions):
            # Deliberate short lines, rather than shrinking Chinese into unreadable footnotes.
            lines, current, width = [], "", 0
            for char in sentence:
                step = 1 if ord(char) > 127 else .52
                if width + step > 88:
                    lines.append(current)
                    current, width = "", 0
                current, width = current + char, width + step
            lines.append(current)
            text(fig, .035, .873 - i * .074, str(i + 1) + ". " + "\n".join(lines), 9)
        table(fig, ["模型", "机制", "本地更新", "锁定更新 / bank", "预算边界"], budget_rows, [.035, .309, .93, .175], 8)
        text(fig, .035, .28, "来源：各已完成模型 full_model_statistics/metrics_by_problem.csv 与 target_recovery*.csv；仅VAL补充保存分数。\n"
             "精确根目录、锁定SHA、逐文件来源见 model_sources.csv / sources.json。R98=07c3adea3，R100=6a808ab3b。\n"
             "R94B外部donor训练清单/重叠未知；R97/98/100为公共冻结预训练+历史TRAIN上下文，0本地SGD不等于0计算。", 8.5)
        text(fig, .035, .178, "仓库根目录 CLI（脚本内固定 CPU2/OMP2；全程 easynco）：\n"
             'eval "$(conda shell.bash hook)"\nconda activate easynco\n'
             "python -B code/V4/r101_completed_context_synthesis.py \\\n"
             "  --output code/V4/runs/R101_hint_pretraining/preparation/completed_context_experiments\n"
             "python -B code/V4/r101_completed_context_synthesis.py --self-test", 8)
        pdf.savefig(fig)
        plt.close(fig)
    return conclusions


def run(output):
    output = Path(output).resolve()
    if output != OUTPUT.resolve():
        raise PermissionError("Output is restricted to the authorized completed_context_experiments directory")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    output.mkdir(parents=True, exist_ok=True)
    sources = Sources()
    protocols, records = catalog(sources)
    problems, metric_rows, metric_index = metrics(sources, protocols)
    target_doc = sources.json(TARGET_PATH)
    targets = target_doc["targets"]
    target_pairs = {(r["problem"], r["method"]) for r in targets}
    if (len(targets) != 36 or len(target_pairs) != 36 or Path(target_doc["identity"]["data_root"]) != DATA
            or target_doc["identity"]["baseline_checkpoint_sha256"] != records[1]["checkpoint_sha256"]):
        raise ValueError("Different fixed R84A VAL target36 identity")
    stored = existing_targets(sources, targets)
    status = dict(available=True, reason="exact native row/pool/cost/winner equality and locked score Top1 checked")
    try:
        generated, overlap, mechanisms, pairs, target_coverage = validation(sources, problems, targets, stored, metric_index)
        stored.update(generated)
    except (OSError, KeyError, ValueError) as error:
        # Bounded failure: retain completed CSV evidence, never infer or start a recovery audit.
        status = dict(available=False, reason=type(error).__name__ + ": " + str(error))
        overlap, mechanisms, pairs, target_coverage = [], [], [], []
        # Context-model VAL recovery has no published CSV; mark unavailable instead of inventing rows.
        for model, baseline, target in itertools.product(MODELS, ("R89B", "R84A"), targets):
            key = model, baseline, "VAL", target["problem"], target["method"]
            if key not in stored:
                stored[key] = dict(model=model, baseline=baseline, split="VAL", problem=target["problem"],
                    solver=target["method"], category=target["category"], n=4500, source=status["reason"],
                    donor_covers_R94B=target["problem"] not in ("TSP", "ATSP"),
                    **{field: None for field in (*COUNTS, *CONTRIBUTIONS)}, recall_pct=None)
    target_rows = list(stored.values())
    summaries = target_summaries(target_rows)
    domains = r94_domains(target_rows)
    source_receipts = sources.finish()
    for name, rows in {
        "all_metrics.csv": [r for r in metric_rows if r["problem"] == "ALL"],
        "metrics_by_problem.csv": metric_rows, "model_sources.csv": records,
        "fixed36_definition.csv": targets, "fixed36_detail.csv": target_rows,
        "fixed36_summary.csv": summaries, "r94b_rescue_by_domain.csv": domains,
        "r94b_target_method_rows.csv": [r for r in target_rows if r["model"] == "R94B" and r["baseline"] == "R89B"],
        "val_coverage_by_problem.csv": overlap, "val_mechanism_coverage.csv": mechanisms,
        "val_pairwise_rescue_overlap.csv": pairs, "val_fixed36_coverage.csv": target_coverage,
    }.items():
        write_csv(output / name, rows)
    conclusions = create_report(output, problems, metric_index, summaries, domains, target_rows, overlap, mechanisms, pairs, targets, records, status)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), script=str(Path(__file__).resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), output=str(output),
        data_root=str(DATA), label_rule=LABEL, target_identity=target_doc["identity"],
        validation_alignment=status, boundary=BOUNDARY, recent_models=RECENT,
        main_reference="R89B", secondary_reference="R84A", test_npz_read=False,
        gpu_used=False, fitting=False, inference=False, runtime_modules_imported=False,
        model_or_rule_selection=False, input_hash_policy="no large input hashing; source stat receipts; old hashes are declarations",
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        threads=2, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cjk_font=str(FONT), inputs=source_receipts, conclusions=conclusions)
    (output / "sources.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validation_alignment": status, "peak_rss_kib": receipt["peak_rss_kib"], "conclusions": conclusions}, ensure_ascii=False, indent=2))


def self_test():
    import unittest

    class TinyTests(unittest.TestCase):
        def test_stable_full_scores(self):
            np.testing.assert_array_equal(stable_top1([[1., 1., 0.]], [9, 2, 3]), [1])
            np.testing.assert_array_equal(stable_top1([[1., 1., 2.]], [9, 2, 3], True), [1])
            with self.assertRaises(ValueError):
                stable_top1([[np.nan]], [1])

        def test_rescues_are_not_selection_counts(self):
            correct = np.zeros((9, 5), dtype=bool)
            correct[0, 0] = True
            correct[2, [0, 1, 2]] = True
            correct[3, [1, 3]] = True
            info, models, pairs = coverage(correct, np.ones(5, dtype=bool))
            self.assertEqual((info["reference_misses"], info["rescued_reference_misses"], info["common_misses"]), (4, 3, 1))
            self.assertEqual((models[0]["rescued"], models[0]["unique_rescue_among_recent"]), (2, 1))
            self.assertEqual((pairs[0]["shared_rescues"], pairs[0]["union_rescues"]), (1, 3))
            close(pairs[0]["rescue_jaccard_pct"], 100 / 3, "tiny Jaccard")

        def test_empty_target_scope(self):
            info, _, pairs = coverage(np.zeros((9, 3), dtype=bool), np.zeros(3, dtype=bool))
            self.assertIsNone(info["reference_plus_recent_union_pct"])
            self.assertIsNone(pairs[0]["rescue_jaccard_pct"])

        def test_equal_problem_contribution_and_pooled_recall(self):
            rows = [{**{key: 0 for key in COUNTS}, **{key: 0.18 for key in CONTRIBUTIONS}, "winner_count": 10, "correct_count": 1, "selected_count": 2}]
            result = aggregate_targets(rows)
            self.assertEqual(result["recall_pct"], 10)
            close(result[CONTRIBUTIONS[0]], .01, "divide by all18, not target groups")

        def test_missing_cost_not_zero(self):
            self.assertIsNone(total_optional([{"cost": 0}, {"cost": None}], "cost"))

        def test_unavailable_alignment_retains_missing_values(self):
            row = {key: None for key in (*COUNTS, *CONTRIBUTIONS)}
            result = aggregate_targets([row])
            self.assertIsNone(result["recall_pct"])
            self.assertIsNone(result["net_recovery"])
            self.assertEqual(count_ratio(None, None), "缺失")

        def test_output_scope(self):
            with self.assertRaises(PermissionError):
                run(ROOT / "unauthorized")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TinyTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.output)


if __name__ == "__main__":
    main()
