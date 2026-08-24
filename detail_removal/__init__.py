"""Remove annotated details from jig images."""

from .coco import CocoDataset, rasterize_annotations
from .dataset import DetailRemovalDataset

__all__ = ["CocoDataset", "DetailRemovalDataset", "rasterize_annotations"]
