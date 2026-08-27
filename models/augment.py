"""Deterministic NumPy augmentations for paired metal-surface crops."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .inputs import sample_crop_box


@dataclass(frozen=True)
class PairAugmentConfig:
    """Configuration for label-safe pair augmentation."""

    enabled: bool = True
    rotation_probability: float = 1.0
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    scale_min: float = 0.7
    scale_max: float = 1.4
    small_rotation_probability: float = 0.3
    small_rotation_degrees: float = 10.0
    swap_probability: float = 0.5
    identity_probability: float = 0.08
    cutout_probability: float = 0.15
    copy_paste_probability: float = 0.10
    shift_probability: float = 0.3
    max_shift_pixels: int = 2
    exposure_probability: float = 0.8
    max_exposure_delta: float = 0.12
    white_balance_probability: float = 0.5
    max_white_balance_delta: float = 0.06
    illumination_probability: float = 0.5
    max_illumination_delta: float = 0.10
    specular_probability: float = 0.5
    max_specular_spots: int = 3
    noise_probability: float = 0.3
    max_noise_sigma: float = 6.0
    jpeg_probability: float = 0.3
    jpeg_quality_min: int = 65
    jpeg_quality_max: int = 95
    sync_photometric_probability: float = 0.8
    sync_brightness_delta: float = 0.20
    sync_contrast_delta: float = 0.20
    sync_gamma_probability: float = 0.3
    sync_gamma_min: float = 0.8
    sync_gamma_max: float = 1.25
    sync_motion_blur_probability: float = 0.3

    def __post_init__(self) -> None:
        probabilities = {
            name: value
            for name, value in asdict(self).items()
            if name.endswith("_probability")
        }
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities.values()):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if self.scale_min <= 0 or self.scale_max < self.scale_min:
            raise ValueError("scale range must be positive and ordered")
        if self.small_rotation_degrees < 0 or self.max_shift_pixels < 0:
            raise ValueError("rotation and shift magnitudes must be nonnegative")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("JPEG quality range must be within [1, 100]")
        if not 0 < self.sync_gamma_min <= self.sync_gamma_max:
            raise ValueError("gamma range must be positive and ordered")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _crop_and_pad(array: np.ndarray, x: int, y: int, size: int, *, mask: bool) -> np.ndarray:
    region = array[y : y + size, x : x + size]
    bottom = size - region.shape[0]
    right = size - region.shape[1]
    if bottom or right:
        border = cv2.BORDER_CONSTANT if mask else cv2.BORDER_REFLECT_101
        region = cv2.copyMakeBorder(region, 0, bottom, 0, right, border, value=0)
    return np.ascontiguousarray(region)


def _crop_resize(
    original: np.ndarray,
    erased: np.ndarray,
    mask: np.ndarray,
    *,
    crop_size: int,
    scale: float,
    rng: random.Random,
    biased_probability: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_size = max(1, int(round(crop_size * scale)))
    x, y = sample_crop_box(mask, source_size, rng, biased_probability)
    a = _crop_and_pad(original, x, y, source_size, mask=False)
    b = _crop_and_pad(erased, x, y, source_size, mask=False)
    target = _crop_and_pad(mask, x, y, source_size, mask=True)
    if source_size != crop_size:
        interpolation = cv2.INTER_AREA if source_size > crop_size else cv2.INTER_LINEAR
        a = cv2.resize(a, (crop_size, crop_size), interpolation=interpolation)
        b = cv2.resize(b, (crop_size, crop_size), interpolation=interpolation)
        target = cv2.resize(target, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    return tuple(np.ascontiguousarray(value) for value in (a, b, target))


def _synchronized_geometry(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    config: PairAugmentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < config.rotation_probability:
        turns = rng.randrange(4)
        if turns:
            a = np.rot90(a, k=-turns)
            b = np.rot90(b, k=-turns)
            mask = np.rot90(mask, k=-turns)
    if rng.random() < config.horizontal_flip_probability:
        a, b, mask = np.fliplr(a), np.fliplr(b), np.fliplr(mask)
    if rng.random() < config.vertical_flip_probability:
        a, b, mask = np.flipud(a), np.flipud(b), np.flipud(mask)
    if config.small_rotation_degrees and rng.random() < config.small_rotation_probability:
        angle = rng.uniform(-config.small_rotation_degrees, config.small_rotation_degrees)
        height, width = mask.shape
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        a = cv2.warpAffine(a, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        b = cv2.warpAffine(b, matrix, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return tuple(np.ascontiguousarray(value) for value in (a, b, mask))


def _synthetic_cutout(
    image: np.ndarray, mask: np.ndarray, rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    radius = rng.randint(max(3, min(height, width) // 64), max(4, min(height, width) // 12))
    center = (rng.randrange(width), rng.randrange(height))
    axes = (radius, max(2, int(radius * rng.uniform(0.4, 1.0))))
    removal = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(removal, center, axes, rng.uniform(0, 180), 0, 360, 255, -1)
    inpainted = cv2.inpaint(image, removal, 3, cv2.INPAINT_TELEA)
    changed = np.any(inpainted != image, axis=2)
    updated = np.logical_or(mask > 0, np.logical_and(removal > 0, changed)).astype(np.uint8)
    return inpainted, updated


def _copy_paste(
    image: np.ndarray,
    mask: np.ndarray,
    donor: np.ndarray,
    donor_mask: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    donor = cv2.resize(donor, (width, height), interpolation=cv2.INTER_AREA)
    donor_mask = cv2.resize(
        donor_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    radius = rng.randint(max(3, min(height, width) // 64), max(5, min(height, width) // 10))
    positive_pixels = np.argwhere(donor_mask > 0)
    if positive_pixels.size:
        source_y, source_x = positive_pixels[rng.randrange(len(positive_pixels))]
        source = int(source_x), int(source_y)
    else:
        source = (rng.randrange(width), rng.randrange(height))
    target = (rng.randrange(width), rng.randrange(height))
    alpha = np.zeros((height, width), dtype=np.float32)
    cv2.circle(alpha, source, radius, 1.0, -1)
    dx, dy = target[0] - source[0], target[1] - source[1]
    matrix = np.float32(((1, 0, dx), (0, 1, dy)))
    moved_donor = cv2.warpAffine(donor, matrix, (width, height),
                                 borderMode=cv2.BORDER_REFLECT_101)
    moved_alpha = cv2.warpAffine(alpha, matrix, (width, height),
                                 borderMode=cv2.BORDER_CONSTANT)
    sigma = max(1.0, radius * 0.15)
    moved_alpha = cv2.GaussianBlur(moved_alpha, (0, 0), sigma)
    result = (
        image.astype(np.float32) * (1.0 - moved_alpha[:, :, None])
        + moved_donor.astype(np.float32) * moved_alpha[:, :, None]
    )
    result = np.clip(result, 0, 255).astype(np.uint8)
    changed = np.any(result != image, axis=2)
    updated = np.logical_or(mask > 0, np.logical_and(moved_alpha > 0.05, changed)).astype(np.uint8)
    return result, updated


def _shift_one_view(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = dy = 0
    while dx == 0 and dy == 0:
        dx, dy = rng.randint(-pixels, pixels), rng.randint(-pixels, pixels)
    matrix = np.float32(((1, 0, dx), (0, 1, dy)))
    target = a if rng.random() < 0.5 else b
    shifted = cv2.warpAffine(target, matrix, (target.shape[1], target.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if target is a:
        a = shifted
    else:
        b = shifted
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
    return a, b, cv2.dilate(mask.astype(np.uint8), kernel)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, quality))
    if not success:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def _independent_photometric(
    image: np.ndarray, rng: random.Random, config: PairAugmentConfig, strength: float
) -> np.ndarray:
    if strength <= 0:
        return image
    result = image.astype(np.float32)
    if rng.random() < config.exposure_probability:
        result *= rng.uniform(
            1.0 - config.max_exposure_delta * strength,
            1.0 + config.max_exposure_delta * strength,
        )
    if rng.random() < config.white_balance_probability:
        factors = np.array([
            rng.uniform(1.0 - config.max_white_balance_delta * strength,
                        1.0 + config.max_white_balance_delta * strength)
            for _ in range(3)
        ], dtype=np.float32)
        result *= factors[None, None, :]
    if rng.random() < config.illumination_probability:
        height, width = result.shape[:2]
        yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        angle = rng.uniform(0, 2 * math.pi)
        field = xx * math.cos(angle) + yy * math.sin(angle)
        field *= rng.uniform(-config.max_illumination_delta, config.max_illumination_delta)
        result *= 1.0 + field[:, :, None] * strength
    if rng.random() < config.specular_probability:
        height, width = result.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        for _ in range(rng.randint(0, config.max_specular_spots)):
            cx, cy = rng.randrange(width), rng.randrange(height)
            radius = rng.uniform(10.0, 40.0)
            intensity = rng.uniform(10.0, 40.0) * strength
            spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius ** 2))
            result += spot[:, :, None] * intensity
    if rng.random() < config.noise_probability:
        sigma = rng.uniform(2.0, config.max_noise_sigma) * strength
        seed = rng.randrange(2 ** 32)
        noise = np.random.default_rng(seed).normal(0.0, sigma, result.shape)
        result += noise.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if rng.random() < config.jpeg_probability:
        full_quality = rng.randint(config.jpeg_quality_min, config.jpeg_quality_max)
        quality = int(round(100 - (100 - full_quality) * strength))
        result = _jpeg(result, quality)
    return np.ascontiguousarray(result)


def _synchronized_photometric(
    a: np.ndarray, b: np.ndarray, rng: random.Random, config: PairAugmentConfig
) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < config.sync_photometric_probability:
        brightness = rng.uniform(-config.sync_brightness_delta, config.sync_brightness_delta) * 255
        contrast = rng.uniform(1.0 - config.sync_contrast_delta, 1.0 + config.sync_contrast_delta)
        a = np.clip(a.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    if rng.random() < config.sync_gamma_probability:
        gamma = rng.uniform(config.sync_gamma_min, config.sync_gamma_max)
        lookup = np.clip((np.arange(256) / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
        a, b = cv2.LUT(a, lookup), cv2.LUT(b, lookup)
    if rng.random() < config.sync_motion_blur_probability:
        size = rng.choice((3, 5, 7))
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        angle = rng.uniform(0, math.pi)
        dx, dy = int(round(center * math.cos(angle))), int(round(center * math.sin(angle)))
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, 1)
        kernel /= kernel.sum()
        a = cv2.filter2D(a, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        b = cv2.filter2D(b, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def augment_pair_crop(
    original: np.ndarray,
    erased: np.ndarray,
    mask: np.ndarray,
    *,
    crop_size: int,
    biased_probability: float,
    rng: random.Random,
    config: PairAugmentConfig,
    curriculum_strength: float,
    donor_provider: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the ordered, label-safe augmentation pipeline and return uint8 arrays."""

    strength = float(np.clip(curriculum_strength, 0.0, 1.0))
    scale = rng.uniform(config.scale_min, config.scale_max) if config.enabled else 1.0
    a, b, target = _crop_resize(
        original, erased, mask, crop_size=crop_size, scale=scale, rng=rng,
        biased_probability=biased_probability,
    )
    if not config.enabled:
        return a, b, target
    a, b, target = _synchronized_geometry(a, b, target, rng, config)
    if rng.random() < config.swap_probability:
        a, b = b, a
    if rng.random() < config.identity_probability:
        source = a if rng.random() < 0.5 else b
        a, b, target = source.copy(), source.copy(), np.zeros_like(target)
    else:
        if rng.random() < config.cutout_probability:
            if rng.random() < 0.5:
                a, target = _synthetic_cutout(a, target, rng)
            else:
                b, target = _synthetic_cutout(b, target, rng)
        if donor_provider is not None and rng.random() < config.copy_paste_probability:
            donor, donor_mask = donor_provider()
            if rng.random() < 0.5:
                a, target = _copy_paste(a, target, donor, donor_mask, rng)
            else:
                b, target = _copy_paste(b, target, donor, donor_mask, rng)
    shift = int(round(config.max_shift_pixels * strength))
    if shift > 0 and rng.random() < config.shift_probability:
        a, b, target = _shift_one_view(a, b, target, rng, shift)
    a = _independent_photometric(a, rng, config, strength)
    b = _independent_photometric(b, rng, config, strength)
    a, b = _synchronized_photometric(a, b, rng, config)
    return (
        np.ascontiguousarray(a, dtype=np.uint8),
        np.ascontiguousarray(b, dtype=np.uint8),
        np.ascontiguousarray(target > 0, dtype=np.uint8),
    )
