"""Adapter from NumPy pair-loader batches to normalized model tensors."""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.nn import functional as F

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]


def _random_float(rng: Any) -> float:
    value = rng.random()
    return float(value.item() if hasattr(value, "item") else value)


def _random_index(rng: Any, upper: int) -> int:
    if upper <= 0:
        raise ValueError("upper must be positive")
    if hasattr(rng, "integers"):
        return int(rng.integers(upper))
    return int(rng.randrange(upper))


def sample_crop_box(
    mask: np.ndarray,
    crop_size: int,
    rng: Any,
    biased_probability: float = 0.7,
) -> Tuple[int, int]:
    """Sample a top-left crop corner, optionally centred on a positive pixel."""

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if not 0.0 <= biased_probability <= 1.0:
        raise ValueError("biased_probability must be in [0, 1]")
    height, width = mask.shape
    max_x = max(0, width - crop_size)
    max_y = max(0, height - crop_size)
    positive = np.argwhere(mask > 0)
    if positive.size and _random_float(rng) < biased_probability:
        y, x = positive[_random_index(rng, len(positive))]
        crop_x = int(np.clip(int(x) - crop_size // 2, 0, max_x))
        crop_y = int(np.clip(int(y) - crop_size // 2, 0, max_y))
        return crop_x, crop_y
    return _random_index(rng, max_x + 1), _random_index(rng, max_y + 1)


def _crop_and_pad(array: np.ndarray, x: int, y: int, crop_size: int, is_mask: bool) -> np.ndarray:
    region = array[y : y + crop_size, x : x + crop_size]
    pad_bottom = crop_size - region.shape[0]
    pad_right = crop_size - region.shape[1]
    if pad_bottom or pad_right:
        border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
        region = cv2.copyMakeBorder(region, 0, pad_bottom, 0, pad_right, border, value=0)
    return np.ascontiguousarray(region)


def _normalize_bgr(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("images must be uint8 HWC BGR arrays")
    rgb = np.ascontiguousarray(image[:, :, ::-1]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return (tensor - _MEAN) / _STD


def prepare_pair_batch(
    originals: Sequence[np.ndarray],
    erased: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    crop_size: int,
    rng: Any,
    device: Optional[torch.device] = None,
    biased_probability: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop, normalize, and stack variable-size NumPy pairs for model input."""

    if not (len(originals) == len(erased) == len(masks)):
        raise ValueError("originals, erased, and masks must have equal lengths")
    if not originals:
        raise ValueError("batch must contain at least one pair")
    views_a = []
    views_b = []
    cropped_masks = []
    for original, erased_image, mask in zip(originals, erased, masks):
        if original.shape != erased_image.shape or original.shape[:2] != mask.shape:
            raise ValueError("each original, erased, mask triplet must share spatial shape")
        x, y = sample_crop_box(mask, crop_size, rng, biased_probability)
        original_crop = _crop_and_pad(original, x, y, crop_size, is_mask=False)
        erased_crop = _crop_and_pad(erased_image, x, y, crop_size, is_mask=False)
        mask_crop = _crop_and_pad(mask, x, y, crop_size, is_mask=True)
        views_a.append(_normalize_bgr(original_crop))
        views_b.append(_normalize_bgr(erased_crop))
        cropped_masks.append(torch.from_numpy((mask_crop > 0).astype(np.float32)))
    view_a = torch.stack(views_a)
    view_b = torch.stack(views_b)
    mask_batch = torch.stack(cropped_masks)
    if device is not None:
        view_a = view_a.to(device)
        view_b = view_b.to(device)
        mask_batch = mask_batch.to(device)
    return view_a, view_b, mask_batch


def build_targets(
    mask_batch: torch.Tensor,
    sizes: Mapping[str, Tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Area-downsample binary masks and map coverage to the soft 0.1..0.9 range."""

    if mask_batch.ndim != 3:
        raise ValueError("mask_batch must have shape [batch, height, width]")
    if not sizes:
        raise ValueError("sizes must contain at least one tap")
    source = mask_batch.to(dtype=torch.float32).unsqueeze(1)
    targets: dict[str, torch.Tensor] = {}
    for tap_name, size in sizes.items():
        height, width = int(size[0]), int(size[1])
        if height <= 0 or width <= 0:
            raise ValueError("tap sizes must be positive")
        ratios = F.interpolate(source, size=(height, width), mode="area").squeeze(1)
        targets[str(tap_name)] = 0.1 + 0.8 * ratios.clamp(0.0, 1.0)
    return targets
