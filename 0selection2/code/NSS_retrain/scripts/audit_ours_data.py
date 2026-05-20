from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def solver_order(split_dir: Path) -> list[str]:
    result_dir = split_dir / "results"
    order = []
    for path in sorted(result_dir.glob("result_*.txt")):
        name = path.name
        if name.startswith("result_"):
            name = name[len("result_") :]
        if name.endswith(".txt"):
            name = name[: -len(".txt")]
        order.append(name)
    return order


def audit_split(root: Path, split: str, expected_solvers: list[str]) -> dict:
    split_dir = root / split
    with open(split_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(split_dir / "raw_label.pkl", "rb") as f:
        labels = pickle.load(f)

    bad_len = 0
    bad_ind = 0
    for i in range(len(dataset)):
        label = labels[str(i)]
        costs = list(label["cost"])
        if len(costs) != len(expected_solvers):
            bad_len += 1
            continue
        if int(label["ind"]) != min(range(len(costs)), key=costs.__getitem__):
            bad_ind += 1

    return {
        "split": split,
        "num_instances": len(dataset),
        "num_solvers": len(expected_solvers),
        "bad_cost_len": bad_len,
        "bad_argmin_label": bad_ind,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--problems", default="TSP,CVRP")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--output", default="solver_order_ours.json")
    args = parser.parse_args()

    root = Path(args.data_root)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_orders: dict[str, list[str]] = {}

    for problem in problems:
        train_split = root / f"{problem}train"
        order = solver_order(train_split)
        if not order:
            raise RuntimeError(f"No result_*.txt files found in {train_split / 'results'}")
        all_orders[problem] = order
        print(f"[{problem}] solver order ({len(order)}): {', '.join(order)}")
        for split in split_names:
            report = audit_split(root, f"{problem}{split}", order)
            print(
                f"  {report['split']}: n={report['num_instances']} "
                f"solvers={report['num_solvers']} bad_len={report['bad_cost_len']} "
                f"bad_argmin={report['bad_argmin_label']}"
            )
            if report["bad_cost_len"] or report["bad_argmin_label"]:
                raise RuntimeError(f"Audit failed for {report['split']}: {report}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_orders, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
