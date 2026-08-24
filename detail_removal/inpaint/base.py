"""Shared inpainting interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Inpainter(Protocol):
    def __call__(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Fill non-zero pixels of ``mask_u8`` in ``image_bgr``."""

        ...
