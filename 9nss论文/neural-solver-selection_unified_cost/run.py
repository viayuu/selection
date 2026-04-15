import argparse
import json
import os

import yaml

from dataset import (
    DEFAULT_NODE_FEATURE_DIM,
    DEFAULT_PROBLEM_FEATURE_DIM,
    DEFAULT_INSTANCE_STATS_DIM,
    UnifiedSelectionDataset,
    build_dataset_specs_from_export_root,
    build_problem_pool,
    build_solver_features,
    build_solver_pool,
    normalize_dataset_specs,
)
from model import UnifiedSelectionModel
from trainer import Trainer
from utils import csv_logger, make_run_dir, seed_everything


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.load(config_file.read(), Loader=yaml.FullLoader)


def summarize_specs(specs):
    rows = []
    for spec in specs:
        rows.append(
            {
                "name": spec["name"],
                "problem_type": spec["problem_type"],
                "dataset_dir": spec["dataset_dir"],
                "solver_names": spec["solver_names"],
                "num_selected": len(spec["sample_indices"]) if spec.get("sample_indices") is not None else spec.get("max_samples"),
            }
        )
    return rows


def resolve_dataset_specs(config_dir, data_params):
    if data_params.get("export_root") is not None:
        export_root = os.path.join(config_dir, data_params["export_root"])
        return build_dataset_specs_from_export_root(export_root, data_params)

    train_specs = normalize_dataset_specs(data_params["train_datasets"], config_dir)
    val_specs = normalize_dataset_specs(data_params["val_datasets"], config_dir)
    test_specs = normalize_dataset_specs(data_params.get("test_datasets", []), config_dir)
    benchmark_specs = normalize_dataset_specs(data_params.get("benchmark_datasets", []), config_dir)
    return train_specs, val_specs, test_specs, benchmark_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified cost-only selector training")
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_name)
    config_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    seed = args.seed if args.seed is not None else config["seed"]
    seed_everything(seed)

    train_specs, val_specs, test_specs, benchmark_specs = resolve_dataset_specs(config_dir, config["data_params"])
    all_specs = train_specs + val_specs + test_specs + benchmark_specs
    solver_pool = config["data_params"].get("solver_pool") or build_solver_pool(all_specs)
    problem_pool = build_problem_pool(all_specs)
    solver_features = build_solver_features(solver_pool, all_specs, problem_pool)

    print("Resolved problems:", problem_pool, flush=True)
    print("Resolved solver pool:", solver_pool, flush=True)
    print(
        "Spec counts:",
        {
            "train": len(train_specs),
            "val": len(val_specs),
            "test": len(test_specs),
            "benchmark": len(benchmark_specs),
        },
        flush=True,
    )

    node_feature_dim = config["data_params"].get("node_feature_dim", DEFAULT_NODE_FEATURE_DIM)
    train_dataset = UnifiedSelectionDataset(
        train_specs,
        solver_pool=solver_pool,
        problem_pool=problem_pool,
        node_feature_dim=node_feature_dim,
        data_aug=config["train_params"].get("data_aug", False),
    )
    val_dataset = UnifiedSelectionDataset(
        val_specs,
        solver_pool=solver_pool,
        problem_pool=problem_pool,
        node_feature_dim=node_feature_dim,
        data_aug=False,
    )
    print(
        "Dataset sizes:",
        {
            "train": len(train_dataset),
            "val": len(val_dataset),
        },
        flush=True,
    )
    config["model_params"]["node_dim"] = node_feature_dim
    config["model_params"]["num_solvers"] = len(solver_pool)
    config["model_params"]["num_problems"] = len(problem_pool)
    config["model_params"]["problem_feature_dim"] = len(train_specs[0]["problem_features"]) if train_specs else DEFAULT_PROBLEM_FEATURE_DIM
    config["model_params"]["instance_stats_dim"] = config["data_params"].get(
        "instance_stats_dim", DEFAULT_INSTANCE_STATS_DIM
    )
    config["model_params"]["solver_feature_dim"] = int(solver_features.size(1))
    config["model_params"]["pooling"] = config["model_params"].get("pooling", True)
    config["model_params"]["output_dim"] = len(solver_pool)

    model = UnifiedSelectionModel(solver_features=solver_features, **config["model_params"])

    run_dir = make_run_dir(
        os.path.join(config_dir, "train_logs"),
        config.get("name", os.path.splitext(os.path.basename(config_path))[0]),
    )
    logger = csv_logger(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2)
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "solver_pool": solver_pool,
                "problem_pool": problem_pool,
                "solver_features": solver_features.tolist(),
                "train_specs": summarize_specs(train_specs),
                "val_specs": summarize_specs(val_specs),
                "test_specs": summarize_specs(test_specs),
                "benchmark_specs": summarize_specs(benchmark_specs),
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": None,
                "benchmark_size": None,
            },
            file_obj,
            indent=2,
        )

    cuda_device_num = config.get("cuda_device_num", -1) if args.gpu_id is None else args.gpu_id
    trainer = Trainer(
        model=model,
        logger=logger,
        cuda_device_num=cuda_device_num,
        train_params=config["train_params"],
        solver_pool=solver_pool,
        problem_pool=problem_pool,
    )

    results = trainer.run(
        train_dataset,
        val_dataset,
        run_dir,
        load_path=args.load,
        test_dataset=None,
        benchmark_dataset=None,
    )

    del train_dataset
    del val_dataset

    if len(test_specs) > 0:
        test_dataset = UnifiedSelectionDataset(
            test_specs,
            solver_pool=solver_pool,
            problem_pool=problem_pool,
            node_feature_dim=node_feature_dim,
            data_aug=False,
        )
        results["test"] = trainer.evaluate_dataset(test_dataset, "test", os.path.join(run_dir, "analysis_test"))
        del test_dataset

    if len(benchmark_specs) > 0:
        benchmark_dataset = UnifiedSelectionDataset(
            benchmark_specs,
            solver_pool=solver_pool,
            problem_pool=problem_pool,
            node_feature_dim=node_feature_dim,
            data_aug=False,
        )
        results["benchmark"] = trainer.evaluate_dataset(
            benchmark_dataset, "benchmark", os.path.join(run_dir, "analysis_benchmark")
        )
        del benchmark_dataset

    print("Run directory:", run_dir)
    for split_name, split_metrics in results.items():
        if not isinstance(split_metrics, dict):
            continue
        keys = [
            key
            for key in split_metrics.keys()
            if key.endswith("top_1_cost")
            or key.endswith("acc")
            or key.endswith("analysis_dir")
            or key.endswith("macro_problem_oracle_ratio")
        ]
        summary = {key: split_metrics[key] for key in keys}
        print(f"{split_name}: {summary}")
