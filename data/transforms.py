"""Deterministic, CPU-only transforms for synthetic paired-image training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class GeometricParams:
    """One shared crop/geometry sample applied identically to both views and target."""

    crop_x: float
    crop_y: float
    crop_size: int
    scale: float = 1.0
    rotation_k: int = 0
    horizontal_flip: bool = False
    vertical_flip: bool = False
    target_dilate_pixels: int = 0


def _uniform(rng: Any, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _choice_int(rng: Any, low: int, high: int) -> int:
    """Sample inclusively from both ``random.Random`` and NumPy generators."""
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    if isinstance(rng, np.random.RandomState):
        return int(rng.randint(low, high + 1))
    return int(rng.randint(low, high))


def _range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) != 2:
            raise ValueError("augmentation ranges must contain exactly two values")
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


def _sample_range(rng: Any, value: Any, default: tuple[float, float]) -> float:
    low, high = _range(value, default)
    return _uniform(rng, low, high)


def sample_shared_geometry(
    rng: Any,
    crop_size: int,
    image_shape: tuple[int, ...],
    registration_translate_pixels: int = 3,
    registration_rotate_degrees: float = 0.0,
    shared_scale: tuple[float, float] | list[float] = (0.9, 1.1),
) -> GeometricParams:
    """Sample crop, flips, quarter turns, and scale once for a pair and its target."""
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")
    if isinstance(registration_translate_pixels, bool) or registration_translate_pixels < 0:
        raise ValueError("registration_translate_pixels must be a non-negative integer")
    rotate_degrees = float(registration_rotate_degrees)
    if not math.isfinite(rotate_degrees) or rotate_degrees < 0:
        raise ValueError("registration_rotate_degrees must be finite and non-negative")
    scale_low, scale_high = _range(shared_scale, (0.9, 1.1))
    if not (0 < scale_low <= scale_high):
        raise ValueError("shared_scale must be a positive increasing range")
    radius = math.hypot(crop_size / 2.0, crop_size / 2.0)
    rotation_displacement = 2.0 * radius * math.sin(math.radians(rotate_degrees) / 2.0)
    target_tolerance = math.ceil(float(registration_translate_pixels) + rotation_displacement)
    max_x = max(0, width - crop_size)
    max_y = max(0, height - crop_size)
    return GeometricParams(
        crop_x=_uniform(rng, 0.0, float(max_x)),
        crop_y=_uniform(rng, 0.0, float(max_y)),
        crop_size=int(crop_size),
        scale=_uniform(rng, scale_low, scale_high),
        rotation_k=_choice_int(rng, 0, 3),
        horizontal_flip=bool(_choice_int(rng, 0, 1)),
        vertical_flip=bool(_choice_int(rng, 0, 1)),
        target_dilate_pixels=int(target_tolerance),
    )


def _geometry_matrix(params: GeometricParams) -> np.ndarray:
    """Return a source-to-output affine matrix centred on the sampled crop."""
    source_center_x = params.crop_x + params.crop_size / 2.0
    source_center_y = params.crop_y + params.crop_size / 2.0
    output_center = params.crop_size / 2.0
    radians = np.deg2rad(90 * (params.rotation_k % 4))
    cosine = np.cos(radians) * params.scale
    sine = np.sin(radians) * params.scale
    transform = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    if params.horizontal_flip:
        transform = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ transform
    if params.vertical_flip:
        transform = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]) @ transform
    translate_from_source = np.array(
        [[1.0, 0.0, -source_center_x], [0.0, 1.0, -source_center_y], [0.0, 0.0, 1.0]]
    )
    translate_to_output = np.array(
        [[1.0, 0.0, output_center], [0.0, 1.0, output_center], [0.0, 0.0, 1.0]]
    )
    return (translate_to_output @ transform @ translate_from_source)[:2].astype(np.float32)


def _dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    size = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel)


def apply_geometry(
    image: np.ndarray,
    mask: np.ndarray,
    params: GeometricParams,
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exactly the same sampled affine transform to an image and target mask.

    Images use reflected pixels beyond their source extent.  Targets are first dilated
    by the configured registration allowance, then use a zero border so positives never
    wrap around an edge.
    """
    if image.ndim != 3 or image.shape[:2] != mask.shape[:2]:
        raise ValueError("image must be HWC and share height/width with mask")
    matrix = _geometry_matrix(params)
    output_size = (params.crop_size, params.crop_size)
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_mask = cv2.warpAffine(
        _dilate_mask(mask, params.target_dilate_pixels),
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed_image, transformed_mask


def apply_registration_jitter(
    image: np.ndarray,
    rng: Any,
    max_translate: float = 3,
    max_angle: float = 0.5,
    *,
    translation: tuple[float, float] | None = None,
    angle: float | None = None,
) -> np.ndarray:
    """Apply a view-specific, reflected-border rotation and translation without wrapping."""
    if image.ndim != 3:
        raise ValueError("image must be uint8 HWC")
    height, width = image.shape[:2]
    tx, ty = translation or (
        _uniform(rng, -float(max_translate), float(max_translate)),
        _uniform(rng, -float(max_translate), float(max_translate)),
    )
    chosen_angle = (
        _uniform(rng, -float(max_angle), float(max_angle)) if angle is None else float(angle)
    )
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), chosen_angle, 1.0)
    matrix[:, 2] += (tx, ty)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _motion_blur(image: np.ndarray, kernel_size: int, angle_degrees: float) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return image
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2.0, kernel_size / 2.0), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    total = float(kernel.sum())
    if total <= 0:
        return image
    return cv2.filter2D(image, -1, kernel / total, borderType=cv2.BORDER_REFLECT_101)


def _noise_generator(rng: Any) -> np.random.Generator:
    if hasattr(rng, "integers"):
        return np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
    return np.random.default_rng(_choice_int(rng, 0, 2**31 - 1))


def apply_photometric(image: np.ndarray, rng: Any, config: Mapping[str, Any]) -> np.ndarray:
    """Apply view-specific photometry in the documented training order.

    The input and result remain uint8 HWC BGR, so normalization is an explicit final
    operation in the caller rather than an accidental side effect of augmentation.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HWC BGR")
    work = image.astype(np.float32)

    strength = _sample_range(rng, config.get("brightness_contrast"), (0.0, 0.0))
    contrast = _uniform(rng, 1.0 - strength, 1.0 + strength)
    brightness = _uniform(rng, -255.0 * strength, 255.0 * strength)
    work = work * contrast + brightness

    gamma = _sample_range(rng, config.get("gamma"), (1.0, 1.0))
    work = np.power(np.clip(work, 0.0, 255.0) / 255.0, gamma) * 255.0

    channel_strength = _sample_range(rng, config.get("channel_gain"), (0.0, 0.0))
    gains = np.array(
        [_uniform(rng, 1.0 - channel_strength, 1.0 + channel_strength) for _ in range(3)],
        dtype=np.float32,
    )
    work *= gains
    work = np.clip(work, 0.0, 255.0).astype(np.uint8)

    motion_kernel = int(round(_sample_range(rng, config.get("motion_blur_kernel"), (1, 1))))
    work = _motion_blur(work, motion_kernel, _uniform(rng, 0.0, 180.0))
    sigma = _sample_range(rng, config.get("defocus_sigma"), (0.0, 0.0))
    if sigma > 0:
        work = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma, sigmaY=sigma)

    noise_rng = _noise_generator(rng)
    gaussian_sigma = _sample_range(rng, config.get("gaussian_noise_sigma_255"), (0.0, 0.0))
    if gaussian_sigma > 0:
        work = work.astype(np.float32) + noise_rng.normal(0.0, gaussian_sigma, work.shape)
    poisson_scale = _sample_range(
        rng,
        config.get("poisson_noise_scale", config.get("poisson_noise")),
        (0.1, 0.1),
    )
    if poisson_scale > 0:
        work = noise_rng.poisson(np.clip(work, 0.0, 255.0) / poisson_scale) * poisson_scale

    illumination = _sample_range(rng, config.get("illumination_gradient"), (0.0, 0.0))
    if illumination > 0:
        height, width = work.shape[:2]
        horizontal = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        vertical = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        plane = np.ones((height, width), dtype=np.float32)
        plane += _uniform(rng, -illumination, illumination) * horizontal[None, :]
        plane += _uniform(rng, -illumination, illumination) * vertical[:, None]
        work = work * plane[:, :, None]

    result = np.clip(work, 0.0, 255.0).astype(np.uint8)
    quality = int(round(_sample_range(rng, config.get("jpeg_quality"), (100, 100))))
    quality = int(np.clip(quality, 1, 100))
    encoded_ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded_ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def normalize_imagenet(image: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC BGR to normalized RGB CHW float32 on CPU."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HWC BGR")
    rgb = np.ascontiguousarray(image[:, :, ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def downsample_soft_target(mask: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Area-downsample a binary mask, then map coverage ratios from 0..1 to 0.1..0.9."""
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    target_height, target_width = int(size[0]), int(size[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError("size must contain positive height and width")
    source = mask.astype(np.float32)
    if source.max(initial=0.0) > 1.0:
        source /= 255.0
    ratios = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)
    mapped = 0.1 + 0.8 * np.clip(ratios, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(mapped.astype(np.float32)))
