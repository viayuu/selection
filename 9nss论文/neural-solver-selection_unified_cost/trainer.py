import csv
import json
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import collate_fn
from loss import MaskedCrossEntropyLoss, MaskedRankingLoss


class Trainer:
    def __init__(self, model, logger, cuda_device_num, train_params, solver_pool, problem_pool):
        super().__init__()
        self.train_params = train_params
        self.solver_pool = solver_pool
        self.problem_pool = problem_pool

        if cuda_device_num == -1 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device("cuda", cuda_device_num)

        self.model = model.to(self.device)
        self.logger = logger
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_params["learning_rate"],
            weight_decay=float(self.train_params["weight_decay"]),
        )

        if self.train_params["loss"] == "rank":
            self.criterion = MaskedRankingLoss(rank_steps=self.train_params.get("rank_steps"))
        elif self.train_params["loss"] == "CE":
            self.criterion = MaskedCrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss: {self.train_params['loss']}")

    def run(self, train_dataset, val_dataset, log_dir, load_path=None, test_dataset=None, benchmark_dataset=None):
        train_sampler = self._build_train_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_params["train_batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_params["test_batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )
        test_loader = (
            DataLoader(
                test_dataset,
                batch_size=self.train_params["test_batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                **self._loader_kwargs(),
            )
            if test_dataset is not None
            else None
        )
        benchmark_loader = (
            DataLoader(
                benchmark_dataset,
                batch_size=self.train_params["test_batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                **self._loader_kwargs(),
            )
            if benchmark_dataset is not None
            else None
        )

        start_epoch = 0
        best_metric = None
        if load_path is not None:
            checkpoint_path = os.path.join("train_logs", load_path, "checkpoint_epoch_best.pt")
            checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            start_epoch = int(checkpoint_dict.get("epoch", -1)) + 1
            best_metric = checkpoint_dict.get("selection_value")
            print(f"Checkpoint is loaded from {checkpoint_path}")
            torch.save(checkpoint_dict, os.path.join(log_dir, "checkpoint_epoch_best.pt"))
            if checkpoint_dict.get("epoch") is not None:
                torch.save(checkpoint_dict, os.path.join(log_dir, f"checkpoint_epoch_{int(checkpoint_dict['epoch']):03d}.pt"))
                with open(os.path.join(log_dir, "best_checkpoint_info.json"), "w", encoding="utf-8") as file_obj:
                    json.dump(
                        {
                            "epoch": int(checkpoint_dict["epoch"]),
                            "selection_metric": checkpoint_dict.get("selection_metric"),
                            "selection_value": checkpoint_dict.get("selection_value"),
                            "val_results": checkpoint_dict.get("val_results"),
                        },
                        file_obj,
                        indent=2,
                    )

        last_val_results = None
        if self.train_params["num_epochs"] == 0:
            last_val_results = self.evaluate(val_loader, "val", os.path.join(log_dir, "analysis_val"))
            self.logger.write(last_val_results)
            return self._evaluate_final_splits(log_dir, val_loader, test_loader, benchmark_loader)

        for epoch in range(start_epoch, self.train_params["num_epochs"]):
            print(f"Training epoch {epoch + 1}/{self.train_params['num_epochs']}")
            train_results = self.train_one_epoch(train_loader)
            val_results = self.evaluate(val_loader, "val", None)
            last_val_results = val_results
            log_dict = {"epoch": epoch, **train_results, **val_results}
            self.logger.write(log_dict)

            selection_metric = self.train_params.get("selection_metric", "val_top_1_cost")
            current_metric = val_results[selection_metric]
            checkpoint_dict = {
                "epoch": epoch,
                "selection_metric": selection_metric,
                "selection_value": current_metric,
                "val_results": val_results,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.train_params.get("save_every_epoch", True):
                torch.save(checkpoint_dict, os.path.join(log_dir, f"checkpoint_epoch_{epoch:03d}.pt"))
            if best_metric is None or current_metric < best_metric:
                best_metric = current_metric
                torch.save(checkpoint_dict, os.path.join(log_dir, "checkpoint_epoch_best.pt"))
                with open(os.path.join(log_dir, "best_checkpoint_info.json"), "w", encoding="utf-8") as file_obj:
                    json.dump(
                        {
                            "epoch": epoch,
                            "selection_metric": selection_metric,
                            "selection_value": current_metric,
                            "val_results": val_results,
                        },
                        file_obj,
                        indent=2,
                    )

        best_checkpoint_path = os.path.join(log_dir, "checkpoint_epoch_best.pt")
        if os.path.exists(best_checkpoint_path):
            checkpoint_dict = torch.load(best_checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])

        final_results = self._evaluate_final_splits(log_dir, val_loader, test_loader, benchmark_loader)
        if last_val_results is not None:
            final_results["last_val_metrics"] = last_val_results
        return final_results

    def _evaluate_final_splits(self, log_dir, val_loader, test_loader, benchmark_loader):
        outputs = {}
        outputs["val"] = self.evaluate(val_loader, "val", os.path.join(log_dir, "analysis_val"))
        if test_loader is not None:
            outputs["test"] = self.evaluate(test_loader, "test", os.path.join(log_dir, "analysis_test"))
        if benchmark_loader is not None:
            outputs["benchmark"] = self.evaluate(
                benchmark_loader, "benchmark", os.path.join(log_dir, "analysis_benchmark")
            )
        return outputs

    def evaluate_dataset(self, dataset, split_name, analysis_dir):
        data_loader = DataLoader(
            dataset,
            batch_size=self.train_params["test_batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )
        return self.evaluate(data_loader, split_name, analysis_dir)

    def _loader_kwargs(self):
        num_workers = int(self.train_params.get("num_workers", 0))
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(self.train_params.get("pin_memory", True)),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = int(self.train_params.get("prefetch_factor", 2))
        return kwargs

    def _build_train_sampler(self, train_dataset):
        if not self.train_params.get("problem_balanced_sampling", False):
            return None

        problem_ids = []
        for entry in train_dataset.entries:
            spec = train_dataset.spec_cache[entry["spec_id"]]["spec"]
            problem_ids.append(int(train_dataset.problem_to_idx[spec["problem_type"]]))
        counts = {}
        for problem_id in problem_ids:
            counts[problem_id] = counts.get(problem_id, 0) + 1

        weights = torch.tensor([1.0 / counts[problem_id] for problem_id in problem_ids], dtype=torch.double)
        epoch_multiplier = float(self.train_params.get("train_epoch_size_multiplier", 1.0))
        num_samples = max(1, int(len(problem_ids) * epoch_multiplier))
        return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in tqdm(train_loader):
            batch = self._to_device(batch)
            logits = self.model(
                batch["nodes"],
                batch["scales"],
                batch["node_mask"],
                batch["problem_ids"],
                batch["problem_features"],
                batch["feasible_mask"],
                batch["instance_stats"],
            )
            loss = self._compute_loss(batch, logits)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        return {"train_loss": total_loss / max(total_batches, 1)}

    @torch.no_grad()
    def evaluate(self, data_loader, split_name, analysis_dir=None):
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0.0
        total_instances = 0

        all_costs = []
        all_masks = []
        all_problem_ids = []
        all_problem_names = []
        all_logits = []
        all_labels = []

        for batch in data_loader:
            batch = self._to_device(batch)
            logits = self.model(
                batch["nodes"],
                batch["scales"],
                batch["node_mask"],
                batch["problem_ids"],
                batch["problem_features"],
                batch["feasible_mask"],
                batch["instance_stats"],
            )
            loss = self._compute_loss(batch, logits)
            preds = torch.argmax(logits, dim=1)

            total_loss += float(loss.item())
            total_batches += 1
            total_correct += float((preds == batch["labels"]).sum().item())
            total_instances += int(batch["labels"].size(0))
            all_costs.append(batch["costs"].detach().cpu())
            all_masks.append(batch["feasible_mask"].detach().cpu())
            all_problem_ids.append(batch["problem_ids"].detach().cpu())
            all_problem_names.extend(batch["problem_names"])
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())

        costs = torch.cat(all_costs, dim=0)
        feasible_mask = torch.cat(all_masks, dim=0)
        problem_ids = torch.cat(all_problem_ids, dim=0)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = torch.argmax(logits.masked_fill(~feasible_mask, -1e9), dim=1)
        oracle_cost = costs.masked_fill(~feasible_mask, float("inf")).min(dim=1).values

        prefix = f"{split_name}_"
        metrics = {
            f"{prefix}loss": total_loss / max(total_batches, 1),
            f"{prefix}acc": total_correct / max(total_instances, 1),
            f"{prefix}oracle_cost": float(oracle_cost.mean().item()),
            f"{prefix}single_best_problem_cost": float(
                self._single_best_problem_cost(costs, feasible_mask, problem_ids).mean().item()
            ),
        }

        max_topk = min(self.train_params.get("eval_top_k", 4), costs.size(1))
        masked_logits = logits.masked_fill(~feasible_mask, -1e9)
        for k in range(1, max_topk + 1):
            topk_idx = torch.topk(masked_logits, k=k, dim=1).indices
            topk_cost = costs.gather(1, topk_idx).min(dim=1).values
            metrics[f"{prefix}top_{k}_cost"] = float(topk_cost.mean().item())
            metrics[f"{prefix}top_{k}_ratio"] = float((topk_cost / oracle_cost).mean().item())

        analysis_payload = self._build_analysis_payload(
            costs=costs,
            feasible_mask=feasible_mask,
            logits=masked_logits,
            labels=labels,
            preds=preds,
            problem_ids=problem_ids,
            problem_names=all_problem_names,
        )
        metrics[f"{prefix}macro_problem_top1_accuracy"] = analysis_payload["macro_average"]["selector_top1_accuracy"]
        metrics[f"{prefix}macro_problem_oracle_ratio"] = analysis_payload["macro_average"]["selector_oracle_ratio"]
        metrics[f"{prefix}macro_problem_gap_vs_best_single"] = analysis_payload["macro_average"][
            "selector_gap_vs_best_single"
        ]
        if analysis_dir is not None:
            self._write_analysis(analysis_dir, analysis_payload)

        if analysis_dir is not None:
            metrics[f"{split_name}_analysis_dir"] = analysis_dir
        return metrics

    def _single_best_problem_cost(self, costs, feasible_mask, problem_ids):
        selected_costs = torch.zeros(costs.size(0), dtype=torch.float32)
        for problem_id in problem_ids.unique():
            problem_mask = problem_ids == problem_id
            problem_costs = costs[problem_mask]
            problem_feasible = feasible_mask[problem_mask]
            masked_problem_costs = problem_costs.masked_fill(~problem_feasible, float("inf"))
            mean_costs = masked_problem_costs.mean(dim=0)
            best_solver = torch.argmin(mean_costs)
            selected_costs[problem_mask] = problem_costs[:, best_solver]
        return selected_costs

    def _build_analysis_payload(self, costs, feasible_mask, logits, labels, preds, problem_ids, problem_names):
        oracle_labels = torch.argmin(costs.masked_fill(~feasible_mask, float("inf")), dim=1)
        selector_costs = costs.gather(1, preds[:, None]).squeeze(1)
        oracle_costs = costs.gather(1, oracle_labels[:, None]).squeeze(1)
        single_best_problem_costs = self._single_best_problem_cost(costs, feasible_mask, problem_ids)

        per_problem_summary = []
        single_method_compare_all = []
        arm_distribution_rows = []
        by_problem = {}

        unique_ids = problem_ids.unique(sorted=True)
        for pid in unique_ids.tolist():
            mask = problem_ids == pid
            first_index = int(mask.nonzero(as_tuple=False)[0].item())
            problem_name = problem_names[first_index]
            p_costs = costs[mask]
            p_mask = feasible_mask[mask]
            p_labels = labels[mask]
            p_preds = preds[mask]
            p_selector_costs = selector_costs[mask]
            p_oracle_costs = oracle_costs[mask]
            p_single_best_problem_costs = single_best_problem_costs[mask]

            selector_mean_cost = float(p_selector_costs.mean().item())
            selector_top1_accuracy = float((p_preds == p_labels).float().mean().item())

            row_bundle = []
            for solver_idx, solver_name in enumerate(self.solver_pool):
                solver_available = p_mask[:, solver_idx]
                if not solver_available.any():
                    continue
                solver_cost = float(p_costs[solver_available, solver_idx].mean().item())
                solver_top1 = float((p_labels == solver_idx).float().mean().item())
                row = {
                    "problem": problem_name.lower(),
                    "arm_id": solver_idx,
                    "arm_name": solver_name,
                    "mean_cost": solver_cost,
                    "top1_accuracy": solver_top1,
                    "selector_mean_cost": selector_mean_cost,
                    "selector_top1_accuracy": selector_top1_accuracy,
                    "selector_beats_cost": selector_mean_cost < solver_cost,
                    "selector_beats_top1": selector_top1_accuracy > solver_top1,
                }
                row_bundle.append(row)
                single_method_compare_all.append(row)

            pred_counts = torch.bincount(p_preds, minlength=len(self.solver_pool)).tolist()
            for solver_idx, count in enumerate(pred_counts):
                if count == 0:
                    continue
                arm_distribution_rows.append(
                    {
                        "problem": problem_name.lower(),
                        "arm_id": solver_idx,
                        "arm_name": self.solver_pool[solver_idx],
                        "count": int(count),
                        "fraction": float(count / max(len(p_preds), 1)),
                    }
                )

            row_bundle.sort(key=lambda item: item["mean_cost"])
            best_single = row_bundle[0] if row_bundle else None
            best_single_mean_cost = None if best_single is None else best_single["mean_cost"]
            best_single_top1_accuracy = None if best_single is None else best_single["top1_accuracy"]
            selector_oracle_ratio = selector_mean_cost / max(float(p_oracle_costs.mean().item()), 1e-12)
            best_single_oracle_ratio = None
            selector_gap_vs_best_single = None
            selector_top1_gain_vs_best_single = None
            if best_single_mean_cost is not None:
                best_single_oracle_ratio = best_single_mean_cost / max(float(p_oracle_costs.mean().item()), 1e-12)
                selector_gap_vs_best_single = selector_mean_cost - best_single_mean_cost
                selector_top1_gain_vs_best_single = selector_top1_accuracy - best_single_top1_accuracy
            by_problem[problem_name.lower()] = {
                "count": int(mask.sum().item()),
                "selector_mean_cost": selector_mean_cost,
                "selector_top1_accuracy": selector_top1_accuracy,
                "oracle_mean_cost": float(p_oracle_costs.mean().item()),
                "selector_oracle_ratio": selector_oracle_ratio,
                "single_best_per_problem_mean_cost": float(p_single_best_problem_costs.mean().item()),
                "best_single_method_by_cost": best_single["arm_name"] if best_single else None,
                "best_single_mean_cost": best_single_mean_cost,
                "best_single_top1_accuracy": best_single_top1_accuracy,
                "best_single_oracle_ratio": best_single_oracle_ratio,
                "selector_gap_vs_best_single": selector_gap_vs_best_single,
                "selector_top1_gain_vs_best_single": selector_top1_gain_vs_best_single,
                "single_method_rows": row_bundle,
            }
            per_problem_summary.append(
                {
                    "problem": problem_name.lower(),
                    "count": int(mask.sum().item()),
                    "selector_mean_cost": selector_mean_cost,
                    "selector_top1_accuracy": selector_top1_accuracy,
                    "oracle_mean_cost": float(p_oracle_costs.mean().item()),
                    "selector_oracle_ratio": selector_oracle_ratio,
                    "single_best_per_problem_mean_cost": float(p_single_best_problem_costs.mean().item()),
                    "best_single_method_by_cost": best_single["arm_name"] if best_single else None,
                    "best_single_mean_cost": best_single_mean_cost,
                    "best_single_top1_accuracy": best_single_top1_accuracy,
                    "best_single_oracle_ratio": best_single_oracle_ratio,
                    "selector_gap_vs_best_single": selector_gap_vs_best_single,
                    "selector_top1_gain_vs_best_single": selector_top1_gain_vs_best_single,
                    "selector_beats_best_single_cost": False
                    if selector_gap_vs_best_single is None
                    else selector_gap_vs_best_single < 0.0,
                    "selector_beats_best_single_top1": False
                    if selector_top1_gain_vs_best_single is None
                    else selector_top1_gain_vs_best_single > 0.0,
                }
            )

        macro_avg = {
            "selector_mean_cost": float(
                sum(item["selector_mean_cost"] for item in per_problem_summary) / max(len(per_problem_summary), 1)
            ),
            "selector_top1_accuracy": float(
                sum(item["selector_top1_accuracy"] for item in per_problem_summary) / max(len(per_problem_summary), 1)
            ),
            "selector_oracle_ratio": float(
                sum(item["selector_oracle_ratio"] for item in per_problem_summary) / max(len(per_problem_summary), 1)
            ),
            "selector_gap_vs_best_single": float(
                sum(item["selector_gap_vs_best_single"] for item in per_problem_summary if item["selector_gap_vs_best_single"] is not None)
                / max(sum(1 for item in per_problem_summary if item["selector_gap_vs_best_single"] is not None), 1)
            ),
        }

        return {
            "overall": {
                "count": int(costs.size(0)),
                "selector_mean_cost": float(selector_costs.mean().item()),
                "selector_top1_accuracy": float((preds == labels).float().mean().item()),
                "oracle_mean_cost": float(oracle_costs.mean().item()),
                "single_best_per_problem_mean_cost": float(single_best_problem_costs.mean().item()),
            },
            "macro_average": macro_avg,
            "by_problem": by_problem,
            "per_problem_summary_rows": per_problem_summary,
            "single_method_compare_all": single_method_compare_all,
            "arm_distribution_rows": arm_distribution_rows,
        }

    def _write_analysis(self, analysis_dir, analysis_payload):
        os.makedirs(analysis_dir, exist_ok=True)

        with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as file_obj:
            json.dump(analysis_payload, file_obj, indent=2)

        with open(os.path.join(analysis_dir, "summary.txt"), "w", encoding="utf-8") as file_obj:
            file_obj.write(
                "overall selector top1_accuracy: {:.6f}\n".format(
                    analysis_payload["overall"]["selector_top1_accuracy"]
                )
            )
            file_obj.write(
                "overall selector mean_cost: {:.6f}\n".format(
                    analysis_payload["overall"]["selector_mean_cost"]
                )
            )
            file_obj.write(
                "overall single_best_per_problem mean_cost: {:.6f}\n".format(
                    analysis_payload["overall"]["single_best_per_problem_mean_cost"]
                )
            )
            file_obj.write(
                "macro selector top1_accuracy: {:.6f}\n".format(
                    analysis_payload["macro_average"]["selector_top1_accuracy"]
                )
            )
            file_obj.write(
                "macro selector mean_cost: {:.6f}\n\n".format(
                    analysis_payload["macro_average"]["selector_mean_cost"]
                )
            )
            file_obj.write(
                "macro selector oracle_ratio: {:.6f}\n".format(
                    analysis_payload["macro_average"]["selector_oracle_ratio"]
                )
            )
            file_obj.write(
                "macro selector gap_vs_best_single: {:.6f}\n\n".format(
                    analysis_payload["macro_average"]["selector_gap_vs_best_single"]
                )
            )
            for row in analysis_payload["per_problem_summary_rows"]:
                file_obj.write(
                    "{} top1_accuracy: {:.6f}, mean_cost: {:.6f}, oracle_mean_cost: {:.6f}, best_single_mean_cost: {:.6f}, oracle_ratio: {:.6f}\n".format(
                        row["problem"],
                        row["selector_top1_accuracy"],
                        row["selector_mean_cost"],
                        row["oracle_mean_cost"],
                        row["best_single_mean_cost"],
                        row["selector_oracle_ratio"],
                    )
                )

        self._write_csv(
            os.path.join(analysis_dir, "per_problem_summary.csv"),
            analysis_payload["per_problem_summary_rows"],
        )
        self._write_csv(
            os.path.join(analysis_dir, "top1_vs_baselines_by_problem.csv"),
            analysis_payload["per_problem_summary_rows"],
        )
        self._write_csv(
            os.path.join(analysis_dir, "single_method_compare_all.csv"),
            analysis_payload["single_method_compare_all"],
        )
        self._write_csv(
            os.path.join(analysis_dir, "arm_distribution.csv"),
            analysis_payload["arm_distribution_rows"],
        )

        for problem_name, problem_info in analysis_payload["by_problem"].items():
            self._write_csv(
                os.path.join(analysis_dir, f"{problem_name}_single_method_compare.csv"),
                problem_info["single_method_rows"],
            )

    def _write_csv(self, file_path, rows):
        if not rows:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _compute_loss(self, batch, logits):
        if self.train_params["loss"] == "rank":
            return self.criterion(logits, batch["costs"], batch["feasible_mask"])
        return self.criterion(logits, batch["labels"], batch["feasible_mask"])

    def _to_device(self, batch):
        output = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                output[key] = value.to(self.device, non_blocking=True)
            else:
                output[key] = value
        return output
