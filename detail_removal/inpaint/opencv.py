"""OpenCV xphoto FSR and core Telea adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .base import Inpainter


class MissingXPhotoError(RuntimeError):
    """Raised when an FSR method is selected without OpenCV contrib."""


def _normalize_inputs(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Inpainting requires a three-channel BGR image")
    if mask_u8.shape != image_bgr.shape[:2]:
        raise ValueError("Image and mask dimensions do not match")
    return np.where(mask_u8 > 0, 255, 0).astype(np.uint8)


@dataclass(frozen=True)
class XPhotoInpainter:
    algorithm: int
    backend: Any

    def __call__(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        hole_mask = _normalize_inputs(image_bgr, mask_u8)
        valid_mask = cv2.bitwise_not(hole_mask)
        source = cv2.bitwise_and(image_bgr, image_bgr, mask=valid_mask)
        result = np.empty_like(source)
        self.backend.inpaint(source, valid_mask, result, self.algorithm)
        return result


@dataclass(frozen=True)
class TeleaInpainter:
    radius: float = 5.0

    def __call__(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        hole_mask = _normalize_inputs(image_bgr, mask_u8)
        return cv2.inpaint(image_bgr, hole_mask, self.radius, cv2.INPAINT_TELEA)


def _xphoto_backend() -> Any:
    backend = getattr(cv2, "xphoto", None)
    required = ("inpaint", "INPAINT_FSR_BEST", "INPAINT_FSR_FAST")
    if backend is None or any(not hasattr(backend, name) for name in required):
        raise MissingXPhotoError(
            "cv2.xphoto is unavailable. Remove other OpenCV wheel variants and "
            "install opencv-contrib-python==4.10.0.84."
        )
    return backend


def create_inpainter(method: str) -> Inpainter:
    normalized = method.strip().lower()
    if normalized == "telea":
        return TeleaInpainter()
    if normalized in ("fsr-best", "fsr-fast"):
        backend = _xphoto_backend()
        algorithm = (
            backend.INPAINT_FSR_BEST
            if normalized == "fsr-best"
            else backend.INPAINT_FSR_FAST
        )
        return XPhotoInpainter(algorithm=algorithm, backend=backend)
    raise ValueError("Unknown inpainting method: {!r}".format(method))
