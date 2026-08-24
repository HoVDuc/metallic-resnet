"""Mask morphology, inward feathering, and component ROI extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentROI:
    label: int
    x0: int
    y0: int
    x1: int
    y1: int
    local_mask: np.ndarray

    @property
    def slices(self) -> Tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass(frozen=True)
class PreparedMask:
    mask: np.ndarray
    alpha: np.ndarray
    components: List[ComponentROI]


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _ellipse(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _inward_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return (mask > 0).astype(np.float32)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(feather_px), 0.0, 1.0).astype(np.float32)
    alpha[mask == 0] = 0.0
    return alpha


def prepare_mask(
    mask_u8: np.ndarray,
    dilate_px: int = 15,
    close_px: int = 2,
    feather_px: int = 9,
    context_margin: float = 2.0,
) -> PreparedMask:
    """Prepare a removal mask and extract expanded component ROIs.

    ``dilate_px`` and ``close_px`` are radii in pixels. ``context_margin``
    is a multiplier of each component's shorter bounding-box side.
    """

    if min(dilate_px, close_px, feather_px) < 0:
        raise ValueError("Morphology and feather sizes must be non-negative")
    if context_margin < 0:
        raise ValueError("Context margin must be non-negative")

    prepared = _binary_mask(mask_u8)
    if close_px:
        prepared = cv2.morphologyEx(prepared, cv2.MORPH_CLOSE, _ellipse(close_px))
    if dilate_px:
        prepared = cv2.dilate(prepared, _ellipse(dilate_px))

    alpha = _inward_alpha(prepared, feather_px)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        prepared, connectivity=8
    )
    image_height, image_width = prepared.shape
    components: List[ComponentROI] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        margin = int(round(context_margin * min(width, height)))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(image_width, x + width + margin)
        y1 = min(image_height, y + height + margin)
        local = np.where(labels[y0:y1, x0:x1] == label, 255, 0).astype(np.uint8)
        components.append(
            ComponentROI(
                label=label,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                local_mask=local,
            )
        )
    return PreparedMask(mask=prepared, alpha=alpha, components=components)
