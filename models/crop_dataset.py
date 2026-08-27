"""PyTorch dataset wrapper which crops and normalizes precomputed image pairs."""

from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .inputs import prepare_pair_batch


def _stable_seed(*values: int) -> int:
    """Mix integer values without Python's process-randomized ``hash``."""

    result = 0x811C9DC5
    for value in values:
        result ^= int(value) & 0xFFFFFFFF
        result = (result * 0x01000193) & 0xFFFFFFFF
    return result


class PairCropDataset(Dataset):
    """Read, crop, normalize, and tensorize a precomputed pair in a worker.

    Crop selection is a pure function of the seed, epoch, and logical item
    index. The epoch is stored in shared memory so persistent DataLoader workers
    observe calls to :meth:`set_epoch` made by the parent process.
    """

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        *,
        crop_size: int,
        seed: int,
        crops_per_sample: int = 1,
        biased_probability: float = 0.7,
        train: bool = True,
    ) -> None:
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if crops_per_sample <= 0:
            raise ValueError("crops_per_sample must be positive")
        if not 0.0 <= biased_probability <= 1.0:
            raise ValueError("biased_probability must be in [0, 1]")
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.crop_size = int(crop_size)
        self.seed = int(seed)
        self.crops_per_sample = int(crops_per_sample)
        self.biased_probability = float(biased_probability) if train else 0.0
        self.train = bool(train)
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.indices) * self.crops_per_sample

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self._epoch.fill_(int(epoch))

    def _rng_for(self, item_index: int) -> random.Random:
        epoch = self.epoch if self.train else 0
        return random.Random(_stable_seed(self.seed, epoch, int(item_index)))

    def __getitem__(self, item_index: int):
        if item_index < 0:
            item_index += len(self)
        if not 0 <= item_index < len(self):
            raise IndexError(item_index)
        dataset_index = self.indices[item_index // self.crops_per_sample]
        original, erased, mask = self.dataset[dataset_index]
        view_a, view_b, mask_batch = prepare_pair_batch(
            [original],
            [erased],
            [mask],
            crop_size=self.crop_size,
            rng=self._rng_for(item_index),
            biased_probability=self.biased_probability,
            device=None,
        )
        return view_a[0], view_b[0], mask_batch[0]
