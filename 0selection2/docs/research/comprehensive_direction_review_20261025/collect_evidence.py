"""Read existing small reports into a research index; no model/data execution."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
RUNS = REPO / "code/V4/runs"


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return str(path.relative_to(REPO))


def main():
    snapshot = datetime.now(timezone.utc).isoformat()
    inventory, metrics, states = [], [], []
    for root in sorted(RUNS.iterdir()):
        match = re.match(r"R(\d+)(?:_|$)", root.name)
        if not match or not root.is_dir():
            continue
        version = int(match[1])
        reports = [p for p in (root / "RESULTS.md", root / "REPORT.md",
                   root / "README.md", root / "PLAN.md", root / "MODEL_PREP.md",
                   root / "comparison_artifacts/RESULTS.md") if p.is_file()]
        reports += sorted(root.glob("*/full_model_statistics/RESULTS.md"))
        code = [p for p in (REPO / f"code/V4/r{version}_model.py",
                REPO / f"code/V4/r{version}_experiment.py",
                REPO / f"code/V4/R{version}_README.md") if p.is_file()]
        inventory.append(dict(version=f"R{version}", run_root=relative(root),
                              reports=";".join(map(relative, reports)),
                              code_entry_points=";".join(map(relative, code))))
        for path in sorted(root.glob("*/status.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            if version >= 103:
                states.append(dict(version=f"R{version}", arm=path.parent.name,
                    phase=state.get("phase"),
                    successful_updates=state.get("successful_updates", state.get("step")),
                    selected_update=state.get("selected_update"),
                    updated_utc=state.get("updated_utc"), source=relative(path)))
        files = sorted(root.glob("*/full_model_statistics/metrics_by_problem.csv"))
        if (root / "full_model_statistics/metrics_by_problem.csv").is_file():
            files.append(root / "full_model_statistics/metrics_by_problem.csv")
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("problem") != "ALL":
                        continue
                    record = {key: row.get(key, "") for key in (
                        "model", "split", "problem", "n", "top1_pct", "mean_cost",
                        "sbs_cost", "oracle_cost", "vs_sbs_pct", "actual_regret_pct",
                        "aggregation", "winner_rule", "data_version", "checkpoint_sha256")}
                    record.update(version=f"R{version}", run_root=relative(root),
                                  source=relative(path))
                    metrics.append(record)
    write_csv("artifact_index.csv", inventory)
    write_csv("reported_all_metrics.csv", metrics)
    write_csv("running_snapshot.csv", states)
    manifest = dict(snapshot_utc=snapshot,
        scope="Existing report/status CSV and JSON only; no raw arrays, weights, inference or training",
        inventory_rows=len(inventory), reported_metric_rows=len(metrics),
        distinct_version_numbers=len({row["version"] for row in inventory}),
        limitations=["Presence of a file is not independent certification of a completed experiment.",
            "Rows copy report values without re-evaluation or cross-era causal claims.",
            "Running statuses are a dated snapshot, not final results.",
            "Early experiment variants also appear in prior review history CSVs."])
    (OUT / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
