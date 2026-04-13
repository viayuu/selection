from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from easynco_bootstrap import ensure_local_easynco

ensure_local_easynco()

from EasyNCO.data.data_utils import augment_mix, augment_reflect, augment_rotate


@dataclass(frozen=True)
class AugmentedBatch:
    coords: torch.Tensor
    augmentation_types: tuple[str, ...]


class OnlineAugmentationGenerator:
    def __init__(
        self,
        base_coords: torch.Tensor,
        *,
        augmentations: tuple[str, ...] = ('rotate', 'reflect', 'mix'),
        mix_prop: float = 0.5,
        seed: int = 2024,
    ):
        self.base_coords_cpu = base_coords.detach().cpu().to(torch.float32)
        self.mix_prop = float(mix_prop)
        self.seed = int(seed)
        self.call_index = 0
        normalized = []
        for item in augmentations:
            name = str(item).strip().lower().replace('-', '_')
            if name not in {'identity', 'rotate', 'reflect', 'mix'}:
                raise ValueError(f"Unsupported augmentation: {item}")
            normalized.append(name)
        self.augmentations = tuple(dict.fromkeys(normalized))
        self.sample_augmentations = tuple(x for x in self.augmentations if x != 'identity')

    def _sample_seed(self, deterministic_seed: int | None) -> int:
        if deterministic_seed is not None:
            return int(deterministic_seed)
        seed = self.seed + self.call_index * 100_003
        self.call_index += 1
        return seed

    def _apply_rotate(self, seed: int) -> torch.Tensor:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            return augment_rotate(problems=self.base_coords_cpu.unsqueeze(0), aug_factor=1)[0]

    def _apply_reflect(self, seed: int) -> torch.Tensor:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            return augment_reflect(problems=self.base_coords_cpu.unsqueeze(0), aug_factor=1)[0]

    def _apply_mix(self, seed: int, rng: random.Random) -> torch.Tensor:
        # 优先调用 EasyNCO 自带的 augment_mix。
        # 由于它是按 aug_factor 批量返回，这里用 aug_factor=2 生成两个候选，
        # 再随机取其中一个作为当前样本。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            mixed = augment_mix(
                problems=self.base_coords_cpu.unsqueeze(0),
                aug_factor=2,
                mix_prop=self.mix_prop,
            )
        pick = int(rng.randrange(max(int(mixed.size(0)), 1)))
        return mixed[pick]

    def _apply_one(self, augmentation_type: str, seed: int, rng: random.Random) -> torch.Tensor:
        if augmentation_type == 'identity':
            return self.base_coords_cpu.clone()
        if augmentation_type == 'rotate':
            return self._apply_rotate(seed)
        if augmentation_type == 'reflect':
            return self._apply_reflect(seed)
        if augmentation_type == 'mix':
            return self._apply_mix(seed, rng)
        raise ValueError(f"Unsupported augmentation_type={augmentation_type}")

    def sample(self, batch_size: int, *, device: str = 'cpu', deterministic_seed: int | None = None) -> AugmentedBatch:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        base_seed = self._sample_seed(deterministic_seed)
        rng = random.Random(base_seed)
        coords_list = [self.base_coords_cpu.clone()]
        labels = ['identity']

        candidate_pool = self.sample_augmentations if self.sample_augmentations else ('identity',)
        for sample_idx in range(1, batch_size):
            aug = rng.choice(candidate_pool)
            sample_seed = base_seed + sample_idx * 17 + 7
            coords_list.append(self._apply_one(aug, sample_seed, rng))
            labels.append(str(aug))

        coords = torch.stack(coords_list, dim=0).to(device=device)
        return AugmentedBatch(coords=coords, augmentation_types=tuple(labels))
