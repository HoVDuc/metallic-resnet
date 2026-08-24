"""Automated seam and texture quality indicators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class QAMetrics:
    boundary_grad_ratio: float
    texture_ratio: float
    mean_shift: Tuple[float, float, float]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ellipse(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _gradient_magnitude(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def measure_component(
    original_bgr: np.ndarray, result_bgr: np.ndarray, component_mask_u8: np.ndarray
) -> QAMetrics:
    hole = component_mask_u8 > 0
    if not np.any(hole):
        raise ValueError("QA component mask is empty")
    mask = np.where(hole, 255, 0).astype(np.uint8)
    outer = cv2.dilate(mask, _ellipse(8)) > 0
    ring = outer & ~hole
    boundary = (cv2.dilate(mask, _ellipse(2)) > 0) & (
        cv2.erode(mask, _ellipse(2)) == 0
    )

    result_gradient = _gradient_magnitude(result_bgr)
    original_gradient = _gradient_magnitude(original_bgr)
    boundary_gradient = _mean(result_gradient[boundary])
    context_gradient = _mean(original_gradient[ring])
    boundary_grad_ratio = boundary_gradient / max(context_gradient, 1e-6)

    result_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result_high = result_gray - cv2.GaussianBlur(result_gray, (0, 0), 1.5)
    original_high = original_gray - cv2.GaussianBlur(original_gray, (0, 0), 1.5)
    hole_texture = float(result_high[hole].std())
    context_texture = float(original_high[ring].std()) if np.any(ring) else 0.0
    texture_ratio = hole_texture / max(context_texture, 1e-6)

    shifts: List[float] = []
    for channel in range(3):
        hole_mean = _mean(result_bgr[..., channel][hole].astype(np.float32))
        ring_mean = _mean(original_bgr[..., channel][ring].astype(np.float32))
        shifts.append(abs(hole_mean - ring_mean))
    mean_shift = (shifts[0], shifts[1], shifts[2])
    passed = bool(
        boundary_grad_ratio <= 1.5
        and texture_ratio >= 0.6
        and max(mean_shift) <= 25.0
    )
    return QAMetrics(
        boundary_grad_ratio=float(boundary_grad_ratio),
        texture_ratio=float(texture_ratio),
        mean_shift=mean_shift,
        passed=passed,
    )


def summarize_metrics(metrics: Iterable[QAMetrics]) -> Dict[str, Any]:
    components = [metric.to_dict() for metric in metrics]
    return {
        "component_count": len(components),
        "failure_count": sum(not component["passed"] for component in components),
        "components": components,
    }
