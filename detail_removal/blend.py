"""Local color matching and mask-confined compositing."""

from __future__ import annotations

import cv2
import numpy as np


def _ring(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask_u8, kernel)
    return (dilated > 0) & (mask_u8 == 0)


def match_local_statistics(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    hole_mask_u8: np.ndarray,
    ring_px: int = 15,
) -> np.ndarray:
    """Match candidate hole pixels to the surrounding original-image ring."""

    hole = hole_mask_u8 > 0
    ring = _ring(hole_mask_u8, max(1, ring_px))
    if np.count_nonzero(hole) < 2 or np.count_nonzero(ring) < 2:
        return candidate_bgr.copy()

    adjusted = candidate_bgr.astype(np.float32)
    original = original_bgr.astype(np.float32)
    for channel in range(candidate_bgr.shape[2]):
        generated = adjusted[..., channel][hole]
        reference = original[..., channel][ring]
        generated_mean = float(generated.mean())
        generated_std = float(generated.std())
        reference_mean = float(reference.mean())
        reference_std = float(reference.std())
        if generated_std > 1e-3:
            values = (generated - generated_mean) * (reference_std / generated_std)
            values += reference_mean
        else:
            values = generated + (reference_mean - generated_mean)
        adjusted[..., channel][hole] = values
    return np.clip(np.rint(adjusted), 0, 255).astype(candidate_bgr.dtype)


def restore_local_texture(
    original_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    hole_mask_u8: np.ndarray,
    ring_px: int = 15,
) -> np.ndarray:
    """Transfer nearby high-frequency residual into a reconstructed hole.

    Residual coordinates are shifted across the component's short axis, which
    keeps the sampled texture spatially coherent instead of generating noise.
    """

    hole = hole_mask_u8 > 0
    ring = _ring(hole_mask_u8, max(1, ring_px))
    if np.count_nonzero(hole) < 2 or np.count_nonzero(ring) < 2:
        return candidate_bgr.copy()

    original = original_bgr.astype(np.float32)
    candidate = candidate_bgr.astype(np.float32)
    original_residual = original - cv2.GaussianBlur(original, (0, 0), 1.2)
    candidate_residual = candidate - cv2.GaussianBlur(candidate, (0, 0), 1.2)

    ys, xs = np.nonzero(hole)
    box_height = int(ys.max() - ys.min() + 1)
    box_width = int(xs.max() - xs.min() + 1)
    sample_y = ys.copy()
    sample_x = xs.copy()
    if box_height >= box_width:
        shift = max(1, box_width)
        sample_x = xs - shift
        invalid = sample_x < 0
        sample_x[invalid] = xs[invalid] + shift
    else:
        shift = max(1, box_height)
        sample_y = ys - shift
        invalid = sample_y < 0
        sample_y[invalid] = ys[invalid] + shift

    height, width = hole.shape
    valid = (
        (sample_y >= 0)
        & (sample_y < height)
        & (sample_x >= 0)
        & (sample_x < width)
    )
    safe_y = np.clip(sample_y, 0, height - 1)
    safe_x = np.clip(sample_x, 0, width - 1)
    valid &= ~hole[safe_y, safe_x]

    samples = np.empty((len(ys), 3), dtype=np.float32)
    samples[valid] = original_residual[safe_y[valid], safe_x[valid]]
    if not np.all(valid):
        fallback = original_residual[ring]
        indexes = np.arange(np.count_nonzero(~valid)) % len(fallback)
        samples[~valid] = fallback[indexes]

    ring_residual = original_residual[ring]
    hole_residual = candidate_residual[hole]
    for channel in range(3):
        target_std = float(ring_residual[:, channel].std())
        current_std = float(hole_residual[:, channel].std())
        needed_std = np.sqrt(max(target_std * target_std - current_std * current_std, 0.0))
        sample = samples[:, channel]
        sample_std = float(sample.std())
        if needed_std > 1e-3 and sample_std > 1e-3:
            samples[:, channel] = (sample - float(sample.mean())) * (
                needed_std / sample_std
            )
        else:
            samples[:, channel] = 0.0

    restored = candidate.copy()
    restored[ys, xs] += samples
    return np.clip(np.rint(restored), 0, 255).astype(candidate_bgr.dtype)


def alpha_composite(
    original_bgr: np.ndarray, candidate_bgr: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    if alpha.shape != original_bgr.shape[:2]:
        raise ValueError("Alpha and image dimensions do not match")
    weight = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]
    blended = original_bgr.astype(np.float32) * (1.0 - weight)
    blended += candidate_bgr.astype(np.float32) * weight
    return np.clip(np.rint(blended), 0, 255).astype(original_bgr.dtype)
