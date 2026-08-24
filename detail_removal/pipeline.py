"""ROI-based detail removal orchestration."""

from __future__ import annotations

import numpy as np

from .blend import alpha_composite, match_local_statistics, restore_local_texture
from .inpaint.base import Inpainter
from .mask import PreparedMask


def remove_details(
    image_bgr: np.ndarray, prepared_mask: PreparedMask, inpainter: Inpainter
) -> np.ndarray:
    """Inpaint each prepared component while preserving pixels outside the mask."""

    if image_bgr.shape[:2] != prepared_mask.mask.shape:
        raise ValueError("Image and prepared mask dimensions do not match")
    result = image_bgr.copy()
    for component in prepared_mask.components:
        y_slice, x_slice = component.slices
        source_roi = image_bgr[y_slice, x_slice].copy()
        candidate = inpainter(source_roi, component.local_mask)
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
