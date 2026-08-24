"""Batch iterator for :class:`detail_removal.dataset.DetailRemovalDataset`."""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from .dataset import DetailRemovalDataset

ImageCollection = Union[List[np.ndarray], np.ndarray]
ImageBatch = Tuple[ImageCollection, ImageCollection]


class DetailRemovalDataLoader:
    """Iterate over batches from ``DetailRemovalDataset`` without PyTorch.

    Images in the source COCO data have different dimensions, so the default
    batch format is ``(list[original], list[erased])``. Set ``stack=True``
    only after a resize/crop transform has made every image in a batch the
    same shape; then each item is a ``[batch, height, width, channels]`` array.
    """

    def __init__(
        self,
        dataset: DetailRemovalDataset,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        stack: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.stack = stack
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[ImageBatch]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)

        stop = len(indices)
        if self.drop_last:
            stop -= stop % self.batch_size
        for start in range(0, stop, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            originals, erased = zip(*(self.dataset[index] for index in batch_indices))
            yield self._collate(list(originals), list(erased))

    def _collate(
        self, originals: List[np.ndarray], erased: List[np.ndarray]
    ) -> ImageBatch:
        if not self.stack:
            return originals, erased
        shapes = {image.shape for image in originals + erased}
        if len(shapes) != 1:
            raise ValueError("Cannot stack images that do not have the same shape")
        return np.stack(originals), np.stack(erased)
