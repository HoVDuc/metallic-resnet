"""COCO loading and segmentation rasterization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CocoDataset:
    """Small indexed view over the COCO fields used by the pipeline."""

    data: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "CocoDataset":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return cls(data=data)

    @property
    def images(self) -> Sequence[Mapping[str, Any]]:
        return self.data.get("images", [])

    @property
    def annotations(self) -> Sequence[Mapping[str, Any]]:
        return self.data.get("annotations", [])

    @property
    def categories(self) -> Sequence[Mapping[str, Any]]:
        return self.data.get("categories", [])

    def category_id(self, name: str) -> int:
        for category in self.categories:
            if category.get("name") == name:
                return int(category["id"])
        raise ValueError("COCO category not found: {!r}".format(name))

    def annotations_for_image(
        self, image_id: int, category_id: int
    ) -> List[Mapping[str, Any]]:
        return [
            annotation
            for annotation in self.annotations
            if int(annotation.get("image_id", -1)) == int(image_id)
            and int(annotation.get("category_id", -1)) == int(category_id)
        ]


def _polygon_mask(
    height: int, width: int, segmentation: Sequence[Any]
) -> np.ndarray:
    instance = np.zeros((height, width), dtype=np.uint8)
    if segmentation and isinstance(segmentation[0], (int, float)):
        rings = [segmentation]
    else:
        rings = segmentation

    polygons = []
    for ring in rings:
        coordinates = np.asarray(ring, dtype=np.float64)
        if coordinates.size < 6 or coordinates.size % 2:
            LOGGER.warning("Ignoring malformed polygon ring with %d values", coordinates.size)
            continue
        polygon = np.rint(coordinates.reshape(-1, 2)).astype(np.int32)
        polygons.append(polygon)
    if polygons:
        cv2.fillPoly(instance, polygons, 255)
    return instance


def _rle_mask(height: int, width: int, segmentation: Mapping[str, Any]) -> np.ndarray:
    rle: Dict[str, Any] = dict(segmentation)
    counts = rle.get("counts")
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(rle, height, width)
    elif isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    if decoded.shape != (height, width):
        raise ValueError(
            "Decoded RLE shape {} does not match image shape {}".format(
                decoded.shape, (height, width)
            )
        )
    return (decoded.astype(bool).astype(np.uint8) * 255)


def _warn_on_bbox_mismatch(
    annotation: Mapping[str, Any], instance: np.ndarray, tolerance: float = 2.0
) -> None:
    expected = annotation.get("bbox")
    points = cv2.findNonZero(instance)
    if expected is None or points is None:
        return
    actual = cv2.boundingRect(points)
    if any(abs(float(want) - float(got)) > tolerance for want, got in zip(expected, actual)):
        LOGGER.warning(
            "Annotation %s bbox %s differs from rasterized bbox %s",
            annotation.get("id"),
            expected,
            actual,
        )


def rasterize_annotations(
    height: int,
    width: int,
    annotations: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Rasterize COCO polygon/RLE annotations into union and instance masks."""

    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive")

    union = np.zeros((height, width), dtype=np.uint8)
    instances: List[np.ndarray] = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, Mapping):
            instance = _rle_mask(height, width, segmentation)
        elif isinstance(segmentation, Sequence):
            instance = _polygon_mask(height, width, segmentation)
        else:
            LOGGER.warning("Ignoring annotation %s without segmentation", annotation.get("id"))
            continue
        _warn_on_bbox_mismatch(annotation, instance)
        instances.append(instance)
        union = cv2.bitwise_or(union, instance)
    return union, instances
