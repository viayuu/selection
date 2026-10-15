"""Count native winners and redistribute winners of one removed solver."""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import markdown
import numpy as np

from code.unified_selector.registry import DATA_ROOT, GLOBAL_SOLVERS, POOLS, PROBLEMS


SCENARIOS = (
    ("original", None),
    ("without_RELD_MTL", "RELD_MTL"),
    ("without_RELD_MOEL", "RELD_MOEL"),
)
SPLITS = (("train", 10000), ("test", 1000))
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs/Winner_distribution_solver_removal"


def after_removal(costs, winner, pool, removed):
    result = winner.copy()
    if removed not in pool:
        return result
    deleted = pool.index(removed)
    affected = winner == deleted
    remaining = np.flatnonzero(np.arange(len(pool)) != deleted)
    result[affected] = remaining[costs[affected][:, remaining].argmin(axis=1)]
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_html(source):
    body = markdown.markdown(source.read_text(encoding="utf-8"), extensions=["tables", "toc"])
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>方法Top1分布</title><style>
body{font:16px/1.7 "Noto Sans CJK SC","Microsoft YaHei",sans-serif;
margin:0;color:#202528;background:#fafafa}
main{max-width:1200px;margin:auto;padding:28px 20px 60px}
h1{font-size:28px}h2{font-size:21px;margin-top:32px}h3{font-size:18px}
table{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:14px}
th,td{border:1px solid #d8dddf;padding:8px 12px;white-space:nowrap}
th{background:#edf1f2}a{color:#13676d}
img{display:block;max-width:100%;height:auto;margin:18px 0}
</style></head><body><main>""" + body + "</main></body></html>\n"
    source.with_suffix(".html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows, transfers, sources = [], [], []
    for problem in PROBLEMS:
        pool = list(POOLS[problem])
        for split, expected_n in SPLITS:
            directory = DATA_ROOT / (problem + split)
            path = directory / "raw_label.pkl"
            payload = path.read_bytes()
            labels = pickle.loads(payload)
            records = [labels[str(i)] for i in range(len(labels))]
            if len(records) != expected_n:
                raise ValueError(f"{problem}/{split}: expected {expected_n}, got {len(records)}")
            names = [p.stem[len("result_"):] for p in sorted((directory / "results").glob("result_*.txt"))]
            if names and names != pool:
                raise ValueError(f"{problem}/{split}: solver names do not match registry")
            costs = np.asarray([record["cost"] for record in records], dtype=np.float64)
            winner = np.asarray([record["ind"] for record in records], dtype=np.int64)
            if costs.shape != (expected_n, len(pool)) or not np.isfinite(costs).all():
                raise ValueError(f"{problem}/{split}: invalid costs")
            if (winner < 0).any() or (winner >= len(pool)).any():
                raise ValueError(f"{problem}/{split}: invalid native winners")
            minimum = costs.min(axis=1)
            sources.append(dict(
                problem=problem, split=split, n=expected_n, path=str(path),
                sha256=hashlib.sha256(payload).hexdigest(), pool=pool,
                exact_minimum_tie_rows=int(((costs == minimum[:, None]).sum(1) > 1).sum()),
                native_nonminimum_rows=int((costs[np.arange(expected_n), winner] != minimum).sum()),
            ))
            for scenario, removed in SCENARIOS:
                selected = winner if removed is None else after_removal(costs, winner, pool, removed)
                counts = np.bincount(selected, minlength=len(pool))
                if int(counts.sum()) != expected_n:
                    raise AssertionError("Winner counts do not sum to the split size")
                for solver, count in zip(pool, counts):
                    active = solver != removed
                    rows.append(dict(
                        problem=problem, split=split, scenario=scenario,
                        removed_solver=removed or "", solver=solver,
                        candidate_active=active, instances=expected_n,
                        eligible_instances=expected_n if active else 0,
                        winner_count=int(count), winner_pct=100 * int(count) / expected_n,
                    ))
                if removed in pool:
                    affected = winner == pool.index(removed)
                    moved = np.bincount(selected[affected], minlength=len(pool))
                    affected_n = int(affected.sum())
                    if counts[pool.index(removed)] != 0:
                        raise AssertionError("A removed method is still a winner")
                    for solver, count in zip(pool, moved):
                        if solver == removed:
                            continue
                        transfers.append(dict(
                            problem=problem, split=split, removed_solver=removed,
                            new_winner=solver, removed_winner_count=affected_n,
                            transferred_count=int(count),
                            share_of_removed_winners_pct=100 * int(count) / affected_n if affected_n else 0.,
                            share_of_all_instances_pct=100 * int(count) / expected_n,
                        ))
            print(f"{problem}/{split}: {expected_n} instances", flush=True)

    detail = list(rows)
    for scope, problems in (("ALL18", PROBLEMS), ("MVRP15", PROBLEMS[3:])):
        solvers = [solver for solver in GLOBAL_SOLVERS if any(solver in POOLS[p] for p in problems)]
        for split, _ in SPLITS:
            for scenario, removed in SCENARIOS:
                subset = [r for r in detail if r["problem"] in problems and r["split"] == split and r["scenario"] == scenario]
                n = sum(next(r["instances"] for r in subset if r["problem"] == p) for p in problems)
                for solver in solvers:
                    solver_rows = [r for r in subset if r["solver"] == solver]
                    count = sum(r["winner_count"] for r in solver_rows)
                    rows.append(dict(
                        problem=scope, split=split, scenario=scenario,
                        removed_solver=removed or "", solver=solver,
                        candidate_active=solver != removed, instances=n,
                        eligible_instances=sum(r["eligible_instances"] for r in solver_rows),
                        winner_count=count, winner_pct=100 * count / n,
                    ))

    lookup = {(r["problem"], r["split"], r["scenario"], r["solver"]): r for r in rows}
    comparison = []
    for problem in ["ALL18", "MVRP15", *PROBLEMS]:
        solvers = [s for s in GLOBAL_SOLVERS if (problem, "train", "original", s) in lookup]
        for solver in solvers:
            record = dict(problem=problem, solver=solver)
            for scenario, _ in SCENARIOS:
                for split, _ in SPLITS:
                    row = lookup[problem, split, scenario, solver]
                    record[f"{scenario}_{split}_count"] = row["winner_count"]
                    record[f"{scenario}_{split}_pct"] = row["winner_pct"]
            comparison.append(record)

    def distribution_table(problem):
        lines = [
            "| 方法 | 原训练 | 原测试 | 删除MTL训练 | 删除MTL测试 | 删除MOEL训练 | 删除MOEL测试 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for record in comparison:
            if record["problem"] != problem:
                continue
            cells = [record["solver"]]
            for scenario, removed in SCENARIOS:
                for split, _ in SPLITS:
                    if record["solver"] == removed:
                        cells.append("已删除")
                    else:
                        count = record[f"{scenario}_{split}_count"]
                        pct = record[f"{scenario}_{split}_pct"]
                        cells.append(f"{count:,} ({pct:.2f}%)")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    transfer_lines = [
        "| 删除的方法 | 新winner | 训练转移数 | 占该方法原训练winner | 测试转移数 | 占该方法原测试winner |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for removed in ("RELD_MTL", "RELD_MOEL"):
        new_solvers = [s for s in POOLS[PROBLEMS[3]] if s != removed]
        for solver in new_solvers:
            cells = [removed, solver]
            for split, _ in SPLITS:
                subset = [r for r in transfers if r["removed_solver"] == removed and r["split"] == split and r["new_winner"] == solver]
                moved = sum(r["transferred_count"] for r in subset)
                n = sum(r["removed_winner_count"] for r in subset)
                cells.extend((f"{moved:,}", f"{100 * moved / n:.2f}%" if n else "0.00%"))
            transfer_lines.append("| " + " | ".join(cells) + " |")

    report = """# 各方法Top1分布与删除单一方法后的变化

Top1分布指方法成为真实winner的次数，不是选择模型的预测准确率。每个问题train为10,000例、test为1,000例；不使用validation。

原候选池使用原标签的native winner。每次只删除一个指定方法；原winner未被删除的实例保持不变，被删除的winner按剩余FP64成本重新分配，完全并列按原成本列顺序确定。

## 15类MVRP汇总

训练共150,000例，测试共15,000例。单元格为“数量（比例）”。

""" + distribution_table("MVRP15") + "\n\n## 被删除方法的winner流向\n\n" + "\n".join(transfer_lines)
    report += "\n\n## 全18任务汇总\n\n训练共180,000例，测试共18,000例。方法适用的候选池不同，这里是全部标签中的份额，不是各方法在其可用任务上的胜率。\n\n"
    report += distribution_table("ALL18")
    report += """

## 说明

- 只删除RELD_MTL和只删除RELD_MOEL是两次独立统计，不是连续删除。
- TSP、CVRP、ATSP的候选池中没有这两个方法，三种情况下的分布完全不变。
- 原实例、成本、winner文件均未修改；没有训练模型或重新运行solver。
- 带B问题仍沿用原成本场景，本次不修复其已知约束语义问题。
- [全部18问题逐项表](by_problem.html) / [Markdown](by_problem.md)
- [宽表CSV](comparison.csv) / [原始长表CSV](winner_distribution.csv) / [winner流向CSV](winner_transfers.csv)
"""
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    sections = ["# 全部18问题的Top1分布", "每个问题训练10,000例、测试1,000例；单元格为数量（比例）。[返回汇总](RESULTS.html)。"]
    for problem in PROBLEMS:
        sections.extend((f"## {problem}", distribution_table(problem)))
    (output / "by_problem.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    write_csv(output / "winner_distribution.csv", rows)
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "winner_transfers.csv", transfers)
    protocol = dict(
        data_root=str(DATA_ROOT), scenarios=[dict(name=n, removed_solver=s) for n, s in SCENARIOS],
        baseline="original native ind", precision="FP64 for redistributed winners",
        tie_rule="original cost-column order for winners reassigned after removal",
        unchanged_rule="retain native winner unless that winner is the removed method",
        sources=sources,
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_html(output / "RESULTS.md")
    render_html(output / "by_problem.md")
    print(f"Output: {output}", flush=True)
    print(distribution_table("MVRP15"), flush=True)


if __name__ == "__main__":
    main()
