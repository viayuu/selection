"""Describe locked R77 A/B VAL differences; no new selector or inference."""

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .r76_probability_diagnostics import score_ranks, write_rows


V4 = Path(__file__).resolve().parent
MANIFEST = V4 / "runs/R77_sample_size_batch1024/selected_train_val_predictions/manifest.json"
OUTPUT = V4 / "runs/R78_zero_selection_goal/review-stage/R77_AB_VAL_COMPLEMENTARITY.csv"


def load_val(arm):
    result = {}
    with np.load(arm["files"]["val"]["path"], allow_pickle=False) as saved:
        for problem in PROBLEMS:
            prefix = problem + "__"
            source = saved[prefix + "source_indices"]
            rows = np.argsort(source, kind="stable")
            ids = saved[prefix + "pool_ids"]
            columns = np.argsort(ids, kind="stable")
            scores = saved[prefix + "logits"][rows]
            costs = saved[prefix + "costs"][rows]
            winner = saved[prefix + "winner"][rows]
            selected, _ = score_ranks(scores, ids, ~np.isneginf(scores))
            if len(rows) != 3895 or len(np.unique(source)) != len(rows):
                raise ValueError(f"{problem}: expected 3895 distinct VAL rows")
            result[problem] = dict(
                source=source[rows], nodes=saved[prefix + "nodes"][rows],
                pool=saved[prefix + "pool"][columns], ids=ids[columns],
                costs=costs[:, columns], winner=ids[winner], selected=ids[selected],
                selected_cost=costs[np.arange(len(rows)), selected].astype(np.float64),
            )
    return result


def describe(problem, a, b):
    for field in ("source", "nodes", "pool", "ids", "costs", "winner"):
        if not np.array_equal(a[field], b[field]):
            raise ValueError(f"{problem}: A/B {field} must align exactly")
    n = len(a["source"])
    ac, bc = a["selected"] == a["winner"], b["selected"] == b["winner"]
    different = a["selected"] != b["selected"]
    wrong = ~ac & ~bc
    delta = b["selected_cost"] - a["selected_cost"]
    oracle = a["costs"].min(1)
    a_regret = 100 * np.mean((a["selected_cost"] - oracle) / oracle)
    b_regret = 100 * np.mean((b["selected_cost"] - oracle) / oracle)
    row = dict(
        problem=problem, split="val", instances=n,
        different_count=int(different.sum()), different_pct=100 * different.mean(),
        both_correct_count=int((ac & bc).sum()),
        a_only_correct_count=int((ac & ~bc).sum()),
        b_only_correct_count=int((~ac & bc).sum()),
        both_wrong_count=int(wrong.sum()),
        both_wrong_different_count=int((wrong & different).sum()),
        both_wrong_same_count=int((wrong & ~different).sum()),
        both_wrong_a_cheaper_count=int((wrong & (delta > 0)).sum()),
        both_wrong_b_cheaper_count=int((wrong & (delta < 0)).sum()),
        both_wrong_cost_tie_count=int((wrong & (delta == 0)).sum()),
        a_top1_pct=100 * ac.mean(), b_top1_pct=100 * bc.mean(),
        a_mean_cost=float(a["selected_cost"].mean()),
        b_mean_cost=float(b["selected_cost"].mean()),
        mean_cost_delta_b_minus_a=float(delta.mean()),
        a_actual_regret_pct=float(a_regret), b_actual_regret_pct=float(b_regret),
        regret_delta_b_minus_a_pp=float(b_regret - a_regret),
    )
    return row


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise ValueError("Both locked arms must be ready")
    for name, lock in (("A", (13, 16146)), ("B", (28, 34776))):
        arm = manifest["arms"][name]
        if (arm["evaluation_point"], arm["selected_update"]) != lock:
            raise ValueError(f"Unexpected {name} lock")
    a, b = (load_val(manifest["arms"][name]) for name in ("A", "B"))
    rows = [describe(problem, a[problem], b[problem]) for problem in PROBLEMS]
    macro = dict(problem="ALL18", split="val")
    for field in rows[0]:
        if field not in macro:
            values = [row[field] for row in rows]
            macro[field] = sum(values) if field == "instances" or field.endswith("_count") else float(np.mean(values))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT, rows + [macro], fields=list(rows[0]))
    print(json.dumps(dict(manifest=str(MANIFEST), csv=str(OUTPUT),
                          descriptive_only=True, test_read=False, macro=macro), indent=2))


if __name__ == "__main__":
    main()
