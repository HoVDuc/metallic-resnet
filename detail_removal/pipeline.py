"""ROI-based detail removal orchestration."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .blend import alpha_composite, match_local_statistics, restore_local_texture
from .inpaint.base import Inpainter
from .mask import PreparedMask


def _apply_candidates(
    image_bgr: np.ndarray,
    prepared_mask: PreparedMask,
    candidates: Sequence[np.ndarray],
) -> np.ndarray:
    if len(candidates) != len(prepared_mask.components):
        raise ValueError("Expected one inpainted candidate per component")
    result = image_bgr.copy()
    for component, candidate in zip(prepared_mask.components, candidates):
        y_slice, x_slice = component.slices
        source_roi = image_bgr[y_slice, x_slice]
        if candidate.shape != source_roi.shape:
            raise ValueError("Inpainter returned an image with the wrong shape")
        if candidate.dtype != source_roi.dtype:
            candidate = np.clip(candidate, 0, 255).astype(source_roi.dtype)
        candidate = match_local_statistics(
            source_roi, candidate, component.local_mask
        )
        candidate = restore_local_texture(
            source_roi, candidate, component.local_mask
        )
        local_alpha = prepared_mask.alpha[y_slice, x_slice].copy()
        local_alpha[component.local_mask == 0] = 0.0
        blended = alpha_composite(source_roi, candidate, local_alpha)
        result_roi = result[y_slice, x_slice]
        component_pixels = component.local_mask > 0
        result_roi[component_pixels] = blended[component_pixels]

    result[prepared_mask.mask == 0] = image_bgr[prepared_mask.mask == 0]
    return result


def remove_details(
    image_bgr: np.ndarray, prepared_mask: PreparedMask, inpainter: Inpainter
) -> np.ndarray:
    """Inpaint each prepared component while preserving pixels outside the mask."""

    if image_bgr.shape[:2] != prepared_mask.mask.shape:
        raise ValueError("Image and prepared mask dimensions do not match")
    candidates = [
        inpainter(image_bgr[component.slices[0], component.slices[1]].copy(), component.local_mask)
        for component in prepared_mask.components
    ]
    return _apply_candidates(image_bgr, prepared_mask, candidates)


def remove_details_batch(
    items: Sequence[Tuple[np.ndarray, PreparedMask]], inpainter: Inpainter
) -> List[np.ndarray]:
    """Batch variant of :func:`remove_details` for GPU-batched inpainters.

    ``items`` is a sequence of ``(image_bgr, prepared_mask)`` pairs. When
    ``inpainter`` exposes an ``inpaint_many(images, masks)`` method (see
    :class:`detail_removal.inpaint.lama.LamaInpainter`), every component ROI
    across all items is grouped into a single call so the GPU can process
    them together. Otherwise this falls back to calling
    :func:`remove_details` once per item.
    """

    if not items:
        return []
    inpaint_many = getattr(inpainter, "inpaint_many", None)
    if inpaint_many is None:
        return [remove_details(image_bgr, prepared_mask, inpainter) for image_bgr, prepared_mask in items]

    for image_bgr, prepared_mask in items:
        if image_bgr.shape[:2] != prepared_mask.mask.shape:
            raise ValueError("Image and prepared mask dimensions do not match")

    flat_rois: List[np.ndarray] = []
    flat_masks: List[np.ndarray] = []
    item_index_by_roi: List[int] = []
    for item_index, (image_bgr, prepared_mask) in enumerate(items):
        for component in prepared_mask.components:
            y_slice, x_slice = component.slices
            flat_rois.append(image_bgr[y_slice, x_slice].copy())
            flat_masks.append(component.local_mask)
            item_index_by_roi.append(item_index)

    if not flat_rois:
        return [image_bgr.copy() for image_bgr, _ in items]

    flat_candidates = inpaint_many(flat_rois, flat_masks)

    grouped_candidates: List[List[np.ndarray]] = [[] for _ in items]
    for item_index, candidate in zip(item_index_by_roi, flat_candidates):
        grouped_candidates[item_index].append(candidate)

    return [
        _apply_candidates(image_bgr, prepared_mask, candidates)
        for (image_bgr, prepared_mask), candidates in zip(items, grouped_candidates)
    ]
