"""Freeze validation selection, then evaluate the chosen model and references."""
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from code.V4.evaluate import load_model, table_line
from code.V4.train import evaluate, configure_torch
from code.unified_selector.registry import PROBLEMS, POOLS, DATA_ROOT
from code.unified_selector.analyze import fig_top1_vs_methods, fig_mean_cost_vs_methods, fig_arm_distribution

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checkpoint(name, source):
    destination = ROOT / "checkpoints" / f"{name}.pt"
    destination.parent.mkdir(exist_ok=True)
    sha256 = file_hash(source)
    if not destination.exists():
        shutil.copy2(source, destination)
    if file_hash(destination) != sha256:
        raise RuntimeError(f"Checkpoint identity changed: {source}; use a new study directory")
    return destination, sha256


def enrich(rows, split):
    for p, row in rows.items():
        with (DATA_ROOT / f"{p}{split}" / "raw_label.pkl").open("rb") as f:
            labels = pickle.load(f)
        costs = np.asarray([labels[str(i)]["cost"] for i in range(len(labels))], dtype=np.float64)
        winners = np.asarray([labels[str(i)]["ind"] for i in range(len(labels))])
        freq = np.bincount(winners, minlength=costs.shape[1]) / len(costs)
        means = costs.mean(0); sbs = int(means.argmin()); names = POOLS[p]
        row.update(problem=p, sbs_cost=row["sbs"], oracle_cost=row["vbs"], vbs_mean=row["vbs"],
                   vs_oracle_pct=(row["mean_cost"] / row["vbs"] - 1) * 100,
                   sbs_name=names[sbs], pool_names=names,
                   method_top1=dict(zip(names, freq.tolist())), method_mean=dict(zip(names, means.tolist())),
                   arm_distribution=dict(zip(names, row["pick_dist"])),
                   beat_top1=[s for i, s in enumerate(names) if row["top1"] > freq[i]],
                   lose_top1=[s for i, s in enumerate(names) if row["top1"] <= freq[i]],
                   beat_mean_cost=[s for i, s in enumerate(names) if row["mean_cost"] < means[i]],
                   lose_mean_cost=[s for i, s in enumerate(names) if row["mean_cost"] >= means[i]])
        rank = np.argsort(costs, axis=1, kind="stable")
        row["sbs_topk"] = [float((rank[:, :k] == sbs).any(1).mean()) for k in (1, 2, 3)]
    return rows


def all_row(rows):
    keys = ("top1", "top2", "top3", "mean_cost", "sbs_cost", "oracle_cost", "vs_sbs_pct", "vs_oracle_pct", "ce")
    return dict(problem="ALL", **{k: float(np.mean([r[k] for r in rows.values()])) for k in keys})


def main():
    torch.set_num_threads(1); configure_torch()
    if (ROOT / "exit_code").read_text().strip() != "0":
        raise RuntimeError("The training queue must finish successfully before test evaluation")
    recipes = []
    for mode in ("original", "ce", "winner_cost"):
        run = ROOT.parent / f"R34_{mode}_seed2_4090"
        if not (run / "finished_at.txt").exists():
            raise RuntimeError(f"Training is unfinished: {run}")
        history = json.loads((run / "history.json").read_text())
        if history[-1]["updates"] < 12000:
            raise RuntimeError(f"Training budget not met: {run}")
        frozen, sha256 = freeze_checkpoint("R34_" + mode, run / "best.pt")
        checkpoint = torch.load(frozen, map_location="cpu", weights_only=False)
        m = checkpoint["macro"]
        score = m["macro_top1"] - .25 * max(0., m["macro_vs_sbs_pct"])
        recipes.append(dict(mode=mode, run=str(run), epoch=checkpoint["epoch"], validation=m, score=score,
                            epochs_run=len(history), updates=history[-1]["updates"],
                            checkpoint_updates=checkpoint["successful_updates"], checkpoint=str(frozen), sha256=sha256))
    chosen = max(recipes, key=lambda r: r["score"])
    selection = dict(recipes=recipes, selected=chosen, criterion="max validation top1 - 0.25*max(0,vs_sbs_pct)")
    selection_path = ROOT / "validation_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise RuntimeError("Validation selection is frozen; use a new study directory for new experiments")
    else:
        dump(selection_path, selection)
    print("VALIDATION SELECTION", chosen, flush=True)
    names = [("R34_" + chosen["mode"], Path(chosen["run"])),
             ("R32e", ROOT.parent / "R32e_solver_features_seed2_4090"),
             ("R33a", ROOT.parent / "R33a_dualstream_seed2_4090")]
    comparisons = {}
    for name, run in names:
        frozen, sha256 = freeze_checkpoint(name, run / "best.pt")
        model, ckpt = load_model(frozen, "cuda:0")
        model.params["native_winner"] = True
        comparisons[name] = {}
        for split in ("val", "test"):
            predictions = {}

            def collect_predictions(module, inputs, output):
                problem = PROBLEMS[inputs[0]["problem_id"]]
                predictions.setdefault(problem, []).append(output["logits"].detach().float().cpu().numpy())

            hook = model.register_forward_hook(collect_predictions)
            try:
                rows, macro = evaluate(model, PROBLEMS, split, 640, 0, "cuda:0", "cuda:0")
            finally:
                hook.remove()
            if name.startswith("R34") and split == "val":
                for metric in ("macro_top1", "macro_top2", "macro_top3"):
                    if abs(macro[metric] - ckpt["macro"][metric]) > 1e-12:
                        raise RuntimeError(f"Validation replay mismatch: {name}/{metric}")
            np.savez_compressed(ROOT / f"{name}_{split}_logits.npz",
                                **{p: np.concatenate(v) for p, v in predictions.items()})
            rows = enrich(rows, split)
            payload = dict(ckpt=str(frozen), sha256=sha256, epoch=ckpt["epoch"], split=split,
                           label_mode="native winner", all=all_row(rows), per_problem=rows, macro=macro)
            dump(ROOT / f"{name}_{split}.json", payload)
            comparisons[name][split] = payload["all"]
            print(name, split, payload["all"], flush=True)
            if name.startswith("R34"):
                dump(run / f"analysis_{split}.json", payload)
                for key, func in [("top1_vs_single_methods", fig_top1_vs_methods),
                                  ("mean_cost_vs_single_methods", fig_mean_cost_vs_methods),
                                  ("arm_distribution", fig_arm_distribution)]:
                    func(rows, run / f"{split}_{key}.png")
                lines = [f"# {name}: {split}", "", f"Validation-selected checkpoint: epoch {ckpt['epoch'] + 1}.",
                         "Native winner labels; ALL percentages are means of per-problem percentages.",
                         "SBS is the best fixed solver on this evaluated split (a hindsight comparator). Oracle is pool VBS, not the proven routing optimum.", "",
                         "| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |",
                         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", table_line(payload["all"], True)]
                lines += [table_line(rows[p]) for p in PROBLEMS]
                lines += ["", "## Arm Distribution", "", "| Problem | Selection frequencies |", "| --- | --- |"]
                lines += [f"| {p} | " + "; ".join(f"{s}: {v:.2%}" for s, v in rows[p]["arm_distribution"].items()) + " |" for p in PROBLEMS]
                (run / f"{split}_summary.md").write_text("\n".join(lines) + "\n")
        del model
    dump(ROOT / "comparison.json", comparisons)
    manifest = {f"{p}/{split}": dict(pool=POOLS[p], sha256=hashlib.sha256((DATA_ROOT / f"{p}{split}/raw_label.pkl").read_bytes()).hexdigest())
                for p in PROBLEMS for split in ("train", "val", "test")}
    dump(ROOT / "label_manifest.json", manifest)
    print("EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
