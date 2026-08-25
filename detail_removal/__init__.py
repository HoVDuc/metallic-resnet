"""Remove annotated details from jig images."""

from .coco import CocoDataset, rasterize_annotations
from .dataloader import DetailRemovalDataLoader
from .dataset import DetailRemovalDataset
from .pairs import PrecomputedPairDataLoader, PrecomputedPairDataset, generate_erased_pairs
from .transforms import (
    IndependentPhotometricAugment,
    SynchronizedPhotometricAugment,
    SynchronizedPhotometricRotate90Augment,
)

__all__ = [
    "CocoDataset",
    "DetailRemovalDataLoader",
    "DetailRemovalDataset",
    "IndependentPhotometricAugment",
    "PrecomputedPairDataLoader",
    "PrecomputedPairDataset",
    "SynchronizedPhotometricAugment",
    "SynchronizedPhotometricRotate90Augment",
    "generate_erased_pairs",
    "rasterize_annotations",
]
