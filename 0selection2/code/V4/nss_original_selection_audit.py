"""Count greedy selections from released NSS weights on the original datasets."""

import csv
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "literature" / "nss\u4ee3\u7801" / "neural-solver-selection"
OUTPUT = Path(__file__).resolve().parent / "runs" / "NSS_original_selection_audit"
POOLS = {
    "TSP": ["BQ", "ELG", "LEHD", "T2T", "T2T500", "DIFUSCO", "DIFUSCO500"],
    "CVRP": ["BQ", "ELG", "LEHD", "OMNI", "MVMOE"],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(task, dataset):
    folder = SOURCE / "datasets" / dataset
    with (folder / "dataset.pkl").open("rb") as handle:
        instances = pickle.load(handle)
    with (folder / "raw_label.pkl").open("rb") as handle:
        labels = pickle.load(handle)
    if task == "CVRP":
        converted = []
        for instance in instances:
            xy = torch.cat((instance["depot"], instance["loc"]), dim=1)
            demand = torch.cat((torch.zeros(1, 1), instance["demand"]), dim=1)
            converted.append(torch.cat((xy, demand[:, :, None]), dim=2))
        instances = converted
    winner = np.array([labels[str(i)]["ind"] for i in range(len(instances))], dtype=np.int64)
    costs = np.array([labels[str(i)]["cost"] for i in range(len(instances))], dtype=np.float64)
    return instances, winner, costs, folder


def run():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("original_nss_model_audit", SOURCE / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], {}
    for task, pool in POOLS.items():
        run_dir = SOURCE / "train_logs" / f"config_{task}.yml_rank_2024"
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert not config["train_params"]["ns_feature"]
        assert not config["train_params"]["manual_feature"]
        params = dict(config["model_params"], ns_feature=False)
        assert params["output_dim"] == len(pool)
        torch.manual_seed(config["seed"])
        model = module.Selection_model(**params)
        checkpoint_path = run_dir / "checkpoint_epoch_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        identity = dict(checkpoint=str(checkpoint_path), sha256=sha256(checkpoint_path),
                        config=str(config_path), saved_seed=config["seed"], epoch=checkpoint.get("epoch"),
                        pool=pool, model_source_sha256=sha256(SOURCE / "model.py"))
        for dataset in (task + "test", task + "LIB"):
            instances, winner, costs, folder = load_data(task, dataset)
            outputs = []
            batch_size = config["train_params"]["test_batch_size"]
            with torch.inference_mode():
                for start in range(0, len(instances), batch_size):
                    samples = [x.squeeze(0) for x in instances[start:start + batch_size]]
                    lengths = torch.tensor([len(x) for x in samples], dtype=torch.float32)
                    size = int(lengths.max())
                    points = torch.stack([F.pad(x, (0, 0, 0, size - len(x))) for x in samples])
                    mask = torch.zeros(len(samples), size)
                    for i, sample in enumerate(samples):
                        mask[i, len(sample):] = float("-inf")
                    outputs.append(model(points, lengths, None, mask).cpu())
                    if start % 200 == 0:
                        print(f"{dataset}: {min(start + batch_size, len(instances))}/{len(instances)}", flush=True)
            scores = torch.cat(outputs)
            probabilities = scores.softmax(1)
            # Match the released trainer's Greedy/Top-1 selection procedure.
            pred = probabilities.topk(1, dim=1, largest=True).indices[:, 0].numpy()
            argmax = scores.argmax(1).numpy()
            selected = np.bincount(pred, minlength=len(pool))
            won = np.bincount(winner, minlength=len(pool))
            correct = np.bincount(pred[pred == winner], minlength=len(pool))
            np.savez_compressed(OUTPUT / f"{dataset}_predictions.npz", pred=pred, winner=winner,
                                logits=scores.numpy(), costs=costs, pool=np.array(pool))
            missing = []
            for slot, solver in enumerate(pool):
                missed = bool(won[slot] > 0 and selected[slot] == 0)
                rows.append(dict(dataset=dataset, n=len(instances), solver=solver,
                    true_winner_count=int(won[slot]), selected_count=int(selected[slot]),
                    correct_selected_count=int(correct[slot]), missed_all_winners=missed,
                    top1_native_pct=100 * float((pred == winner).mean())))
                if missed:
                    missing.append(f"{solver}(winners={won[slot]})")
            provenance[dataset] = dict(identity, dataset=str(folder),
                dataset_sha256=sha256(folder / "dataset.pkl"), labels_sha256=sha256(folder / "raw_label.pkl"),
                n=len(instances), top1_native_pct=100 * float((pred == winner).mean()),
                greedy_vs_argmax_differences=int((pred != argmax).sum()),
                score_max_tie_rows=int(((probabilities == probabilities.max(1, keepdim=True).values).sum(1) > 1).sum()),
                cost_argmin_vs_saved_ind_differences=int((costs.argmin(1) != winner).sum()))
            print(f"{dataset}: top1={100 * (pred == winner).mean():.2f}%; "
                  f"zero-selected winning solvers={missing}", flush=True)
    with (OUTPUT / "selection_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(execution="CPU FP32 eval-only; released original checkpoints; no retraining",
            scope="One publicly released ranking checkpoint per problem, not all five paper seeds or CE models",
            selection="softmax(scores).topk(1), as in the original trainer; no calibration or rule changes",
            labels="saved original raw_label.pkl ind; separate FP64 argmin difference recorded",
            batch="Original saved test_batch_size (8)", datasets=provenance), handle, indent=2)


if __name__ == "__main__":
    run()
