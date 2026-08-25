"""NumPy dataset that creates an original/erased image pair online."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .coco import CocoDataset, rasterize_annotations
from .inpaint import Inpainter, create_inpainter
from .mask import prepare_mask
from .pipeline import remove_details
from .selection import select_random_annotations

LOGGER = logging.getLogger(__name__)

ImagePair = Tuple[np.ndarray, np.ndarray]
PairedTransform = Callable[[np.ndarray, np.ndarray], ImagePair]


def _validate_image_pair(pair: object) -> ImagePair:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError("paired_transform must return a 2-tuple")
    original, erased = pair
    if not isinstance(original, np.ndarray) or not isinstance(erased, np.ndarray):
        raise TypeError("paired_transform outputs must be NumPy arrays")
    if original.dtype != np.uint8 or erased.dtype != np.uint8:
        raise TypeError("paired_transform outputs must have dtype uint8")
    if original.shape != erased.shape:
        raise ValueError("paired_transform outputs must have the same shape")
    return original, erased


class DetailRemovalDataset:
    """Load one COCO image and return ``(original, erased)`` as NumPy arrays.

    The erased image is generated inside :meth:`__getitem__`.  The class does
    not inherit from PyTorch so it can be used without installing torch, while
    still satisfying the ``__len__``/``__getitem__`` interface expected by a
    PyTorch ``DataLoader``.
    """

    def __init__(
        self,
        coco_path: Path,
        images_dir: Path,
        *,
        category: str = "detail",
        method: str = "fsr-best",
        inpainter: Optional[Inpainter] = None,
        seed: Optional[int] = None,
        paired_transform: Optional[PairedTransform] = None,
        dilate_px: int = 15,
        close_px: int = 2,
        feather_px: int = 9,
        context_margin: float = 2.0,
    ) -> None:
        self.coco = CocoDataset.load(Path(coco_path))
        self.images: List[Mapping[str, object]] = list(self.coco.images)
        self.images_dir = Path(images_dir)
        self.category_id = self.coco.category_id(category)
        self.inpainter = inpainter or create_inpainter(method)
        self.rng = random.Random(seed)
        self.paired_transform = paired_transform
        self.dilate_px = dilate_px
        self.close_px = close_px
        self.feather_px = feather_px
        self.context_margin = context_margin

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ImagePair:
        image_info = self.images[index]
        source_path = self.images_dir / str(image_info["file_name"])
        original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError("Could not read image: {}".format(source_path))

        height, width = original.shape[:2]
        metadata_size = (
            int(image_info.get("width", width)),
            int(image_info.get("height", height)),
        )
        if metadata_size != (width, height):
            LOGGER.warning(
                "Image %s dimensions %sx%s differ from COCO metadata %sx%s",
                source_path.name,
                width,
                height,
                metadata_size[0],
                metadata_size[1],
            )

        annotations = self.coco.annotations_for_image(
            int(image_info["id"]), self.category_id
        )
        selected = select_random_annotations(annotations, self.rng)
        mask, _ = rasterize_annotations(height, width, selected)
        prepared_mask = prepare_mask(
            mask,
            dilate_px=self.dilate_px,
            close_px=self.close_px,
            feather_px=self.feather_px,
            context_margin=self.context_margin,
        )
        erased = remove_details(original, prepared_mask, self.inpainter)

        pair = (original, erased)
        if self.paired_transform is not None:
            pair = self.paired_transform(*pair)
        return _validate_image_pair(pair)
