"""Cache decoded samples once and gather single-problem batches on the GPU."""

import torch

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem


_CACHE = {}


class TensorBatchLoader:
    def __init__(self, batch, batch_size, shuffle, drop_last):
        self.batch = batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.lengths = batch["n"].cpu()
        self.size = len(self.lengths)

    def __len__(self):
        return self.size // self.batch_size if self.drop_last else (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.size) if self.shuffle else torch.arange(self.size)
        stop = self.size - self.size % self.batch_size if self.drop_last else self.size
        for start in range(0, stop, self.batch_size):
            idx = order[start : min(start + self.batch_size, stop)]
            max_n = int(self.lengths[idx].max())
            device_idx = idx.to(self.batch["costs"].device)
            batch = {}
            for key, value in self.batch.items():
                if not torch.is_tensor(value) or key == "pool_ids":
                    batch[key] = value
                    continue
                if key in ("node", "node_mask"):
                    value = value[:, :max_n]
                elif key == "matrix":
                    value = value[:, :max_n, :max_n]
                batch[key] = value.index_select(0, device_idx)
            yield batch


def make_tensor_loader(problem, split, batch_size, shuffle, device, coord_augment=0):
    key = (problem, split, str(device), coord_augment)
    if key not in _CACHE:
        dataset = UnifiedProblemDataset(problem, split, coord_augment=coord_augment)
        batch = collate_single_problem([dataset[i] for i in range(len(dataset))])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        _CACHE[key] = batch
        size_mb = sum(v.numel() * v.element_size() for v in batch.values() if torch.is_tensor(v)) / 2**20
        print(f"[cache] {problem}/{split}: {len(dataset)} instances, {size_mb:.1f} MiB on {device}", flush=True)
    return TensorBatchLoader(_CACHE[key], batch_size, shuffle, drop_last=(split == "train"))
