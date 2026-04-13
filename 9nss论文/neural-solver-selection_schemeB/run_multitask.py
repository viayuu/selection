import argparse
import json
import os

import yaml

from multitask_common import load_multitask_dataset, load_problem_only_dataset, seed_everything
from multitask_model import MultiTaskSelectionModel
from multitask_trainer import MultiTaskTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train/eval multitask NSS selector")
    parser.add_argument("--config_name", type=str, default="config_multitask.yml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    return parser.parse_args()


def build_log_dir(args, config):
    if args.load is not None:
        return os.path.join("train_logs", args.load)
    run_name = f"{os.path.splitext(args.config_name)[0]}_{config['train_params']['loss']}_{config['seed']}"
    return os.path.join("train_logs", run_name)


def build_eval_dataset(test_file: str):
    if test_file == "ALLVAL":
        return load_multitask_dataset("val", data_aug=False)
    if test_file == "ALLTEST":
        return load_multitask_dataset("test", data_aug=False)
    if test_file == "ALLLIB":
        return load_multitask_dataset("lib", data_aug=False)
    if test_file == "TSPtest":
        return load_problem_only_dataset("TSP", "test")
    if test_file == "CVRPtest":
        return load_problem_only_dataset("CVRP", "test")
    if test_file == "TSPLIB":
        return load_problem_only_dataset("TSP", "lib")
    if test_file == "CVRPLIB":
        return load_problem_only_dataset("CVRP", "lib")
    raise ValueError(f"Unsupported test_file: {test_file}")


def main():
    args = parse_args()

    if args.load is not None:
        with open(os.path.join("train_logs", args.load, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        with open(args.config_name, "r", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.loss is not None:
        config["train_params"]["loss"] = args.loss
    seed_everything(config["seed"])

    cuda_device_num = config["cuda_device_num"] if args.gpu_id is None else args.gpu_id
    log_dir = build_log_dir(args, config)
    os.makedirs(log_dir, exist_ok=True)

    if args.load is None and not os.path.exists(os.path.join(log_dir, "config.json")):
        with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    model = MultiTaskSelectionModel(**config["model_params"])
    trainer = MultiTaskTrainer(model=model, train_params=config["train_params"], log_dir=log_dir, cuda_device_num=cuda_device_num)

    if args.test_file is None:
        train_dataset = load_multitask_dataset("train", data_aug=config["train_params"]["data_aug"])
        val_dataset = load_multitask_dataset("val", data_aug=False)
        load_path = None
        if args.load is not None:
            load_path = os.path.join("train_logs", args.load, "checkpoint_best.pt")
        trainer.run(train_dataset, val_dataset, load_path=load_path)
    else:
        if args.load is None:
            raise ValueError("--test_file requires --load to specify a trained checkpoint directory.")
        checkpoint_path = os.path.join("train_logs", args.load, "checkpoint_best.pt")
        trainer.load_checkpoint(checkpoint_path)
        eval_dataset = build_eval_dataset(args.test_file)
        results = trainer.evaluate(eval_dataset, split_name=args.test_file)
        result_name = args.exp_name if args.exp_name is not None else args.test_file
        with open(os.path.join(log_dir, f"{result_name}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(results)


if __name__ == "__main__":
    main()
