import csv
import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import augment_xy_by_8_fold, process_instance_CVRP


TSP_NUM_CLASSES = 7
CVRP_NUM_CLASSES = 5
GLOBAL_NUM_CLASSES = TSP_NUM_CLASSES + CVRP_NUM_CLASSES

PROBLEM_TO_ID = {"TSP": 0, "CVRP": 1}
ID_TO_PROBLEM = {0: "TSP", 1: "CVRP"}


def seed_everything(seed: int = 2022) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def _split_dir(problem_type: str, split: str) -> str:
    if split == "train":
        return f"{problem_type}train"
    if split == "val":
        return f"{problem_type}val"
    if split == "test":
        return f"{problem_type}test"
    if split == "lib":
        return f"{problem_type}LIB"
    raise ValueError(f"Unsupported split: {split}")


def _load_raw_labels(problem_type: str, split: str) -> Tuple[List, List]:
    split_dir = _split_dir(problem_type, split)
    dataset_path = os.path.join("datasets", split_dir, "dataset.pkl")
    label_path = os.path.join("datasets", split_dir, "raw_label.pkl")

    with open(dataset_path, "rb") as f:
        raw_dataset = pickle.load(f)
    with open(label_path, "rb") as f:
        raw_label_dict = pickle.load(f)

    labels = []
    for i in range(len(raw_dataset)):
        key = str(i)
        labels.append(
            [
                raw_label_dict[key]["ind"],
                raw_label_dict[key]["cost"],
                raw_label_dict[key]["time"],
                raw_label_dict[key]["gap"],
            ]
        )
    return raw_dataset, labels


def _to_three_channel(problem_type: str, sample) -> torch.Tensor:
    if problem_type == "CVRP":
        return process_instance_CVRP(sample)

    # TSP 原始样本是 [1, N, 2]，这里补一列 0 demand，统一成 [1, N, 3]
    coords = sample
    zeros = torch.zeros(coords.size(0), coords.size(1), 1, dtype=coords.dtype)
    return torch.cat((coords, zeros), dim=2)


def _build_global_vectors(problem_type: str, local_costs, local_times, local_gaps, local_best: int):
    global_costs = torch.full((GLOBAL_NUM_CLASSES,), float("inf"), dtype=torch.float32)
    global_times = torch.zeros(GLOBAL_NUM_CLASSES, dtype=torch.float32)
    global_gaps = torch.full((GLOBAL_NUM_CLASSES,), float("inf"), dtype=torch.float32)
    feasible_mask = torch.zeros(GLOBAL_NUM_CLASSES, dtype=torch.bool)

    local_costs = torch.tensor(local_costs, dtype=torch.float32)
    local_times = torch.tensor(local_times, dtype=torch.float32)
    local_gaps = torch.tensor(local_gaps, dtype=torch.float32)

    if problem_type == "TSP":
        offset = 0
        num_classes = TSP_NUM_CLASSES
    else:
        offset = TSP_NUM_CLASSES
        num_classes = CVRP_NUM_CLASSES

    global_costs[offset : offset + num_classes] = local_costs
    global_times[offset : offset + num_classes] = local_times
    global_gaps[offset : offset + num_classes] = local_gaps
    feasible_mask[offset : offset + num_classes] = True

    return {
        "global_costs": global_costs,
        "global_times": global_times,
        "global_gaps": global_gaps,
        "feasible_mask": feasible_mask,
        "global_best": int(offset + local_best),
        "offset": int(offset),
        "num_classes": int(num_classes),
    }


@dataclass
class MultiTaskRecord:
    x: torch.Tensor
    problem_id: int
    problem_name: str
    local_best: int
    global_best: int
    offset: int
    num_classes: int
    global_costs: torch.Tensor
    global_times: torch.Tensor
    global_gaps: torch.Tensor
    feasible_mask: torch.Tensor
    source_index: int


def _build_records(problem_type: str, split: str) -> List[MultiTaskRecord]:
    raw_dataset, labels = _load_raw_labels(problem_type, split)
    records: List[MultiTaskRecord] = []
    for idx, (sample, label) in enumerate(zip(raw_dataset, labels)):
        x = _to_three_channel(problem_type, sample)
        local_best, local_costs, local_times, local_gaps = label
        global_info = _build_global_vectors(problem_type, local_costs, local_times, local_gaps, local_best)
        records.append(
            MultiTaskRecord(
                x=x,
                problem_id=PROBLEM_TO_ID[problem_type],
                problem_name=problem_type,
                local_best=int(local_best),
                global_best=global_info["global_best"],
                offset=global_info["offset"],
                num_classes=global_info["num_classes"],
                global_costs=global_info["global_costs"],
                global_times=global_info["global_times"],
                global_gaps=global_info["global_gaps"],
                feasible_mask=global_info["feasible_mask"],
                source_index=idx,
            )
        )
    return records


class MultiTaskSelectionDataset(Dataset):
    def __init__(self, records: Iterable[MultiTaskRecord], data_aug: bool = False):
        super().__init__()
        self.records = list(records)
        if data_aug:
            self.records = self._augment_records(self.records)

    def _augment_records(self, records: List[MultiTaskRecord]) -> List[MultiTaskRecord]:
        augmented: List[MultiTaskRecord] = []
        for record in records:
            aug_coords = augment_xy_by_8_fold(record.x[:, :, :2])
            demands = record.x[:, :, 2:].repeat(8, 1, 1)
            aug_x = torch.cat((aug_coords, demands), dim=2)
            for i in range(aug_x.size(0)):
                augmented.append(
                    MultiTaskRecord(
                        x=aug_x[i : i + 1],
                        problem_id=record.problem_id,
                        problem_name=record.problem_name,
                        local_best=record.local_best,
                        global_best=record.global_best,
                        offset=record.offset,
                        num_classes=record.num_classes,
                        global_costs=record.global_costs.clone(),
                        global_times=record.global_times.clone(),
                        global_gaps=record.global_gaps.clone(),
                        feasible_mask=record.feasible_mask.clone(),
                        source_index=record.source_index,
                    )
                )
        return augmented

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MultiTaskRecord:
        return self.records[index]


def load_multitask_dataset(split: str, data_aug: bool = False) -> MultiTaskSelectionDataset:
    tsp_records = _build_records("TSP", split)
    cvrp_records = _build_records("CVRP", split)
    records = tsp_records + cvrp_records
    return MultiTaskSelectionDataset(records, data_aug=data_aug)


def load_problem_only_dataset(problem_type: str, split: str) -> MultiTaskSelectionDataset:
    return MultiTaskSelectionDataset(_build_records(problem_type, split), data_aug=False)


def multitask_collate_fn(batch: List[MultiTaskRecord]) -> dict:
    batch_x = [record.x.squeeze(0) for record in batch]
    lengths = np.array([data.shape[0] for data in batch_x])
    max_length = int(np.max(lengths))

    padded_x = [F.pad(batch_x[i], (0, 0, 0, max_length - lengths[i]))[None, :, :] for i in range(len(batch_x))]
    ninf_mask = torch.zeros(len(batch_x), max_length, dtype=torch.float32)
    for i in range(len(batch_x)):
        ninf_mask[i, lengths[i]: max_length] = float("-inf")

    return {
        "x": torch.cat(padded_x, dim=0),
        "scales": torch.tensor(lengths, dtype=torch.float32),
        "mask": ninf_mask,
        "problem_ids": torch.tensor([record.problem_id for record in batch], dtype=torch.long),
        "local_best": torch.tensor([record.local_best for record in batch], dtype=torch.long),
        "global_best": torch.tensor([record.global_best for record in batch], dtype=torch.long),
        "offsets": torch.tensor([record.offset for record in batch], dtype=torch.long),
        "num_classes": torch.tensor([record.num_classes for record in batch], dtype=torch.long),
        "global_costs": torch.stack([record.global_costs for record in batch], dim=0),
        "global_times": torch.stack([record.global_times for record in batch], dim=0),
        "global_gaps": torch.stack([record.global_gaps for record in batch], dim=0),
        "feasible_mask": torch.stack([record.feasible_mask for record in batch], dim=0),
        "problem_names": [record.problem_name for record in batch],
        "source_indices": [record.source_index for record in batch],
    }


class JsonlLogger:
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def write(self, payload: dict) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class CsvLogger:
    def __init__(self, file_path: str, fieldnames: List[str]):
        self.file_path = file_path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        need_header = not os.path.exists(file_path)
        self._file = open(file_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if need_header:
            self._writer.writeheader()
            self._file.flush()

    def write(self, payload: dict) -> None:
        row = {name: payload.get(name, "") for name in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()
