import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import RankingLoss
from multitask_common import (
    CVRP_NUM_CLASSES,
    ID_TO_PROBLEM,
    JsonlLogger,
    TSP_NUM_CLASSES,
    multitask_collate_fn,
)


class MultiTaskTrainer:
    """
    联合训练器：
    - 训练时混合 TSP / CVRP batch
    - loss 按问题分别计算，再取平均
    - 验证时分别统计 TSP / CVRP，再做 macro 平均
    """

    def __init__(self, model, train_params, log_dir: str, cuda_device_num: int):
        self.model = model
        self.train_params = train_params
        self.log_dir = log_dir

        if cuda_device_num == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", cuda_device_num)
            torch.cuda.set_device(cuda_device_num)

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_params["learning_rate"],
            weight_decay=float(self.train_params["weight_decay"]),
        )
        self.rank_tsp = RankingLoss(TSP_NUM_CLASSES, TSP_NUM_CLASSES)
        self.rank_cvrp = RankingLoss(CVRP_NUM_CLASSES, CVRP_NUM_CLASSES)
        self.metrics_logger = JsonlLogger(os.path.join(log_dir, "metrics.jsonl"))

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        return checkpoint_dict

    def _move_batch(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _problem_view(self, logits: torch.Tensor, batch: dict, problem_id: int):
        problem_mask = batch["problem_ids"] == problem_id
        if not problem_mask.any():
            return None
        if problem_id == 0:
            start, end = 0, TSP_NUM_CLASSES
        else:
            start, end = TSP_NUM_CLASSES, TSP_NUM_CLASSES + CVRP_NUM_CLASSES

        return {
            "logits": logits[problem_mask][:, start:end],
            "costs": batch["global_costs"][problem_mask][:, start:end],
            "gaps": batch["global_gaps"][problem_mask][:, start:end],
            "times": batch["global_times"][problem_mask][:, start:end],
            "labels": batch["local_best"][problem_mask],
            "name": ID_TO_PROBLEM[problem_id],
        }

    def _compute_loss(self, logits: torch.Tensor, batch: dict):
        losses = {}
        for problem_id in (0, 1):
            view = self._problem_view(logits, batch, problem_id)
            if view is None:
                continue
            if self.train_params["loss"] == "rank":
                if problem_id == 0:
                    losses["tsp"] = self.rank_tsp(view["logits"], view["costs"])
                else:
                    losses["cvrp"] = self.rank_cvrp(view["logits"], view["costs"])
            else:
                losses_key = "tsp" if problem_id == 0 else "cvrp"
                losses[losses_key] = F.nll_loss(F.log_softmax(view["logits"], dim=1), view["labels"])

        if not losses:
            raise RuntimeError("Batch contains no valid problem data.")

        total_loss = sum(losses.values()) / float(len(losses))
        return total_loss, {key: float(value.detach().item()) for key, value in losses.items()}

    def train_one_epoch(self, epoch: int, train_dataset):
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_params["train_batch_size"],
            shuffle=True,
            collate_fn=multitask_collate_fn,
        )

        self.model.train()
        total_loss = 0.0
        loss_count = 0
        tsp_loss_total = 0.0
        tsp_loss_count = 0
        cvrp_loss_total = 0.0
        cvrp_loss_count = 0

        for batch in tqdm(dataloader, desc=f"train epoch {epoch}", leave=False):
            batch = self._move_batch(batch)
            logits = self.model(batch["x"], batch["scales"], batch["mask"], batch["problem_ids"])
            loss, loss_parts = self._compute_loss(logits, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            loss_count += 1
            if "tsp" in loss_parts:
                tsp_loss_total += loss_parts["tsp"]
                tsp_loss_count += 1
            if "cvrp" in loss_parts:
                cvrp_loss_total += loss_parts["cvrp"]
                cvrp_loss_count += 1

        return {
            "train_loss": total_loss / max(loss_count, 1),
            "train_tsp_loss": tsp_loss_total / max(tsp_loss_count, 1),
            "train_cvrp_loss": cvrp_loss_total / max(cvrp_loss_count, 1),
        }

    def _evaluate_one_problem(self, logits, gaps, times, labels, select_time: float):
        scores = F.softmax(logits, dim=1)
        result = {
            "acc": float((scores.argmax(dim=1) == labels).float().mean().item()),
            "single_best_gap": float(gaps.mean(dim=0).min().item()),
            "oracle_gap": float(gaps.min(dim=1)[0].mean().item()),
            "oracle_time": float(times.sum(dim=1).mean().item()),
        }

        top1_gap = None
        top1_time = None
        top2_gap = None
        top2_time = None
        max_k = min(4, scores.size(1))
        for k in range(1, max_k + 1):
            _, topk_ind = scores.topk(k, dim=1, largest=True)
            topk_gap = gaps.gather(1, topk_ind).min(dim=1)[0]
            topk_time = times.gather(1, topk_ind).sum(dim=1)
            result[f"top_{k}"] = float(topk_gap.mean().item())
            result[f"time_top_{k}"] = float(topk_time.mean().item() + select_time)
            if k == 1:
                top1_gap = topk_gap
                top1_time = topk_time
            if k == 2:
                top2_gap = topk_gap
                top2_time = topk_time

        if top1_gap is not None and top2_gap is not None:
            sort_ind = scores.max(dim=1)[0].sort(descending=True)[1].cpu().numpy()
            threshold = int(scores.size(0) * 0.8)
            reject_ind = sort_ind[threshold:]
            accept_ind = sort_ind[:threshold]
            gap_sr = torch.cat((top2_gap[reject_ind], top1_gap[accept_ind]), dim=0)
            time_sr = torch.cat((top2_time[reject_ind], top1_time[accept_ind]), dim=0)
            result["cover_80"] = float(gap_sr.mean().item())
            result["time_cover_80"] = float(time_sr.mean().item() + select_time)

        top_p_gaps = []
        top_p_times = []
        for i in range(scores.size(0)):
            for j in range(1, scores.size(1) + 1):
                top_j, ind = scores[i].topk(j, largest=True)
                if top_j.sum() >= 0.8:
                    top_p_gaps.append(gaps[i][ind].min().item())
                    top_p_times.append(times[i][ind].sum().item())
                    break
        if top_p_gaps:
            result["top_p_80"] = float(np.mean(top_p_gaps))
            result["time_top_p_80"] = float(np.mean(top_p_times) + select_time)

        return result

    @torch.no_grad()
    def evaluate(self, dataset, split_name: str):
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_params["test_batch_size"],
            shuffle=False,
            collate_fn=multitask_collate_fn,
        )

        self.model.eval()
        problem_buckets = {
            "TSP": {"logits": [], "gaps": [], "times": [], "labels": []},
            "CVRP": {"logits": [], "gaps": [], "times": [], "labels": []},
        }
        total_instances = 0
        start = time.time()

        for batch in dataloader:
            batch = self._move_batch(batch)
            logits = self.model(batch["x"], batch["scales"], batch["mask"], batch["problem_ids"])
            for problem_id in (0, 1):
                view = self._problem_view(logits, batch, problem_id)
                if view is None:
                    continue
                problem_buckets[view["name"]]["logits"].append(view["logits"].detach().cpu())
                problem_buckets[view["name"]]["gaps"].append(view["gaps"].detach().cpu())
                problem_buckets[view["name"]]["times"].append(view["times"].detach().cpu())
                problem_buckets[view["name"]]["labels"].append(view["labels"].detach().cpu())
            total_instances += int(batch["x"].size(0))

        select_time = (time.time() - start) / max(total_instances, 1)
        results = {"split": split_name, "select_time": float(select_time)}
        macro_top_1 = []

        for problem_name in ("TSP", "CVRP"):
            bucket = problem_buckets[problem_name]
            if not bucket["logits"]:
                continue
            logits = torch.cat(bucket["logits"], dim=0)
            gaps = torch.cat(bucket["gaps"], dim=0)
            times = torch.cat(bucket["times"], dim=0)
            labels = torch.cat(bucket["labels"], dim=0)
            metrics = self._evaluate_one_problem(logits, gaps, times, labels, select_time)
            for key, value in metrics.items():
                results[f"{problem_name.lower()}_{key}"] = value
            if "top_1" in metrics:
                macro_top_1.append(metrics["top_1"])

        if macro_top_1:
            results["macro_top_1"] = float(np.mean(macro_top_1))

        return results

    def run(self, train_dataset, val_dataset, load_path: Optional[str] = None):
        best_metric = None
        if load_path is not None:
            checkpoint_dict = self.load_checkpoint(load_path)
            best_metric = checkpoint_dict.get("best_metric")

        for epoch in range(self.train_params["num_epochs"]):
            train_metrics = self.train_one_epoch(epoch, train_dataset)
            val_metrics = self.evaluate(val_dataset, split_name="val")
            payload = {"epoch": epoch, **train_metrics, **val_metrics}
            self.metrics_logger.write(payload)

            metric = val_metrics.get("macro_top_1", float("inf"))
            if best_metric is None or metric < best_metric:
                best_metric = metric
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": best_metric,
                    },
                    os.path.join(self.log_dir, "checkpoint_best.pt"),
                )
