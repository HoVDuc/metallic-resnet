"""GPU-batched Simple-LaMa inpainting backend.

Wraps the TorchScript model used by ``simple_lama_inpainting`` directly
instead of going through ``SimpleLama.__call__`` so that several ROIs can be
padded to a common size and run through the network in a single forward
pass, which is what makes GPU batch processing worthwhile.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import Inpainter

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]


class MissingLamaError(RuntimeError):
    """Raised when torch or simple_lama_inpainting are not installed."""


def _import_simple_lama():
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise MissingLamaError(
            "simple_lama_inpainting is not installed. Install it with "
            "`pip install simple-lama-inpainting` (requires torch)."
        ) from exc
    return SimpleLama


def _ceil_modulo(value: int, modulo: int) -> int:
    if value % modulo == 0:
        return value
    return (value // modulo + 1) * modulo


def _normalize_mask(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Inpainting requires a three-channel BGR image")
    if mask_u8.shape != image_bgr.shape[:2]:
        raise ValueError("Image and mask dimensions do not match")
    return np.where(mask_u8 > 0, 255, 0).astype(np.uint8)


def _image_to_chw01(image_bgr: np.ndarray) -> np.ndarray:
    rgb = image_bgr[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def _mask_to_chw01(hole_mask_u8: np.ndarray) -> np.ndarray:
    return (hole_mask_u8[np.newaxis, :, :] > 0).astype(np.float32)


def _pad_to(array_chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    _, height, width = array_chw.shape
    return np.pad(
        array_chw,
        ((0, 0), (0, out_h - height), (0, out_w - width)),
        mode="symmetric",
    )


class LamaInpainter:
    """Simple-LaMa backed :class:`Inpainter` with GPU batch support.

    Usable as a drop-in single-ROI :class:`Inpainter` (``inpainter(image,
    mask)``), and also exposes :meth:`inpaint_many` for callers that want to
    inpaint several ROIs in one GPU forward pass.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        modulo: int = 8,
        max_batch_size: int = 8,
    ) -> None:
        if torch is None:
            raise MissingLamaError(
                "PyTorch is not installed. Install torch with CUDA support to "
                "use the lama inpainting backend."
            )
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        simple_lama_cls = _import_simple_lama()
        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        simple_lama = simple_lama_cls(device=resolved_device)
        self._model = simple_lama.model
        self.device = resolved_device
        self.modulo = modulo
        self.max_batch_size = max_batch_size

    def __call__(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        return self.inpaint_many([image_bgr], [mask_u8])[0]

    def inpaint_many(
        self, images_bgr: Sequence[np.ndarray], masks_u8: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Inpaint several ROIs, batching the GPU forward pass.

        ROIs are padded (symmetric, bottom/right only) to the largest shape
        in each sub-batch of at most ``max_batch_size`` items before being
        stacked, then cropped back to their original size afterwards.
        """

        if len(images_bgr) != len(masks_u8):
            raise ValueError("images_bgr and masks_u8 must have the same length")
        results: List[np.ndarray] = []
        for start in range(0, len(images_bgr), self.max_batch_size):
            chunk_images = images_bgr[start : start + self.max_batch_size]
            chunk_masks = masks_u8[start : start + self.max_batch_size]
            results.extend(self._inpaint_chunk(chunk_images, chunk_masks))
        return results

    def _inpaint_chunk(
        self, images_bgr: Sequence[np.ndarray], masks_u8: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        if not images_bgr:
            return []
        prepared_images = []
        prepared_masks = []
        original_sizes = []
        max_height = 0
        max_width = 0
        for image_bgr, mask_u8 in zip(images_bgr, masks_u8):
            hole_mask = _normalize_mask(image_bgr, mask_u8)
            chw_image = _image_to_chw01(image_bgr)
            chw_mask = _mask_to_chw01(hole_mask)
            height, width = chw_image.shape[1:]
            original_sizes.append((height, width))
            max_height = max(max_height, height)
            max_width = max(max_width, width)
            prepared_images.append(chw_image)
            prepared_masks.append(chw_mask)

        out_height = _ceil_modulo(max_height, self.modulo)
        out_width = _ceil_modulo(max_width, self.modulo)
        batch_image = np.stack(
            [_pad_to(image, out_height, out_width) for image in prepared_images]
        )
        batch_mask = np.stack(
            [_pad_to(mask, out_height, out_width) for mask in prepared_masks]
        )

        image_tensor = torch.from_numpy(batch_image).to(self.device)
        mask_tensor = torch.from_numpy(batch_mask).to(self.device)
        mask_tensor = (mask_tensor > 0) * 1

        with torch.inference_mode():
            inpainted = self._model(image_tensor, mask_tensor)

        results = []
        for index, (height, width) in enumerate(original_sizes):
            candidate = inpainted[index, :, :height, :width]
            candidate = candidate.permute(1, 2, 0).detach().cpu().numpy()
            candidate = np.clip(candidate * 255, 0, 255).astype(np.uint8)
            results.append(np.ascontiguousarray(candidate[:, :, ::-1]))
        return results
