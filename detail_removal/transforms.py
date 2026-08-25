"""Photometric transforms for original/erased NumPy image pairs."""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class IndependentPhotometricAugment:
    """Apply light brightness and motion-blur augmentation to each view.

    Random parameters are drawn separately for the original and erased images.
    This prevents the model from relying on pixel-perfect photometric matches.
    """

    def __init__(
        self,
        *,
        brightness_probability: float = 0.8,
        brightness_limit: float = 0.15,
        motion_blur_probability: float = 0.3,
        motion_blur_kernels: Sequence[int] = (3, 5, 7),
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= brightness_probability <= 1.0:
            raise ValueError("brightness_probability must be in [0, 1]")
        if not 0.0 <= brightness_limit < 1.0:
            raise ValueError("brightness_limit must be in [0, 1)")
        if not 0.0 <= motion_blur_probability <= 1.0:
            raise ValueError("motion_blur_probability must be in [0, 1]")
        kernels = tuple(motion_blur_kernels)
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("motion_blur_kernels must contain positive odd sizes")

        self.brightness_probability = brightness_probability
        self.brightness_limit = brightness_limit
        self.motion_blur_probability = motion_blur_probability
        self.motion_blur_kernels = kernels
        self._rng = random.Random(seed)

    def __call__(self, original: np.ndarray, erased: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._augment(original), self._augment(erased)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        brightness_factor = None
        if self._rng.random() < self.brightness_probability:
            brightness_factor = self._rng.uniform(
                1.0 - self.brightness_limit, 1.0 + self.brightness_limit
            )
        motion_kernel = None
        if self._rng.random() < self.motion_blur_probability:
            kernel_size = self._rng.choice(self.motion_blur_kernels)
            angle = self._rng.uniform(0.0, 180.0)
            motion_kernel = _motion_blur_kernel(kernel_size, angle)
        return _apply_operations(image, brightness_factor, motion_kernel)


class SynchronizedPhotometricAugment(IndependentPhotometricAugment):
    """Apply identical brightness and motion-blur parameters to both views.

    This variant is suitable for ``PrecomputedPairDataset``: pixels that were
    equal before augmentation remain equal, so its difference mask remains a
    structural target instead of becoming an all-positive photometric mask.
    """

    def __call__(self, original: np.ndarray, erased: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        brightness_factor = None
        if self._rng.random() < self.brightness_probability:
            brightness_factor = self._rng.uniform(
                1.0 - self.brightness_limit, 1.0 + self.brightness_limit
            )
        motion_kernel = None
        if self._rng.random() < self.motion_blur_probability:
            kernel_size = self._rng.choice(self.motion_blur_kernels)
            angle = self._rng.uniform(0.0, 180.0)
            motion_kernel = _motion_blur_kernel(kernel_size, angle)
        return (
            _apply_operations(original, brightness_factor, motion_kernel),
            _apply_operations(erased, brightness_factor, motion_kernel),
        )


class SynchronizedPhotometricRotate90Augment(SynchronizedPhotometricAugment):
    """Synchronized photometric augmentation plus a random quarter turn.

    The same randomly selected turn (0, 90, 180, or 270 degrees clockwise) is
    applied to both views.  ``transform_mask`` lets the pair dataset apply the
    exact same geometric transform to a stored target mask.
    """

    def __init__(
        self,
        *,
        rotation_probability: float = 1.0,
        **kwargs: object,
    ) -> None:
        if not 0.0 <= rotation_probability <= 1.0:
            raise ValueError("rotation_probability must be in [0, 1]")
        super().__init__(**kwargs)
        self.rotation_probability = rotation_probability
        self.last_rotation_k = 0

    def __call__(self, original: np.ndarray, erased: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        augmented_original, augmented_erased = super().__call__(original, erased)
        if self._rng.random() < self.rotation_probability:
            self.last_rotation_k = self._rng.randrange(4)
        else:
            self.last_rotation_k = 0
        return (
            _rotate_quarter_turns(augmented_original, self.last_rotation_k),
            _rotate_quarter_turns(augmented_erased, self.last_rotation_k),
        )

    def transform_mask(self, mask: np.ndarray) -> np.ndarray:
        return _rotate_quarter_turns(mask, self.last_rotation_k)


def _rotate_quarter_turns(array: np.ndarray, turns_clockwise: int) -> np.ndarray:
    turns_clockwise %= 4
    if turns_clockwise == 0:
        return array.copy()
    rotation = {
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[turns_clockwise]
    return cv2.rotate(array, rotation)


def _apply_operations(
    image: np.ndarray,
    brightness_factor: Optional[float],
    motion_kernel: Optional[np.ndarray],
) -> np.ndarray:
    if image.dtype != np.uint8:
        raise TypeError("photometric augmentation requires uint8 images")
    augmented = image.copy()
    if brightness_factor is not None:
        augmented = np.clip(
            augmented.astype(np.float32) * brightness_factor, 0, 255
        ).astype(np.uint8)
    if motion_kernel is not None:
        augmented = cv2.filter2D(
            augmented,
            ddepth=-1,
            kernel=motion_kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
    return augmented


def _motion_blur_kernel(size: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = center
    angle = math.radians(angle_degrees)
    dx = int(round(radius * math.cos(angle)))
    dy = int(round(radius * math.sin(angle)))
    cv2.line(
        kernel,
        (center - dx, center - dy),
        (center + dx, center + dy),
        color=1.0,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    return kernel / kernel.sum()
