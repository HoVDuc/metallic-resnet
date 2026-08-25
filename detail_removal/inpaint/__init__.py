"""Inpainting backends."""

from __future__ import annotations

from .base import Inpainter
from .opencv import MissingXPhotoError
from .opencv import create_inpainter as _create_opencv_inpainter

_LAMA_METHODS = ("lama",)


def create_inpainter(method: str, **kwargs: object) -> Inpainter:
    """Build an :class:`Inpainter` for ``method``.

    ``"lama"`` builds a GPU-batched Simple-LaMa backend (see
    :mod:`detail_removal.inpaint.lama`); any other method is delegated to
    the OpenCV backends. Extra ``kwargs`` (e.g. ``device``,
    ``max_batch_size``) are only valid for ``"lama"``.
    """

    normalized = method.strip().lower()
    if normalized in _LAMA_METHODS:
        from .lama import LamaInpainter

        return LamaInpainter(**kwargs)
    if kwargs:
        raise TypeError("Unexpected keyword arguments for method {!r}: {}".format(
            method, sorted(kwargs)
        ))
    return _create_opencv_inpainter(normalized)


__all__ = ["Inpainter", "MissingXPhotoError", "create_inpainter"]
