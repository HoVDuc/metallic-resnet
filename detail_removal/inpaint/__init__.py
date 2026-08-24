"""Inpainting backends."""

from .opencv import MissingXPhotoError, create_inpainter

__all__ = ["MissingXPhotoError", "create_inpainter"]
