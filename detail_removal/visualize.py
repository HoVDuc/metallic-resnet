"""Visualize online original/erased pairs from :class:`DetailRemovalDataset`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .dataset import DetailRemovalDataset

LOGGER = logging.getLogger(__name__)
_TITLE_HEIGHT = 28


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save original, difference heatmap, and online-erased dataset samples."
    )
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=1800)
    parser.add_argument("--category", default="detail")
    parser.add_argument(
        "--method", choices=("fsr-best", "fsr-fast", "telea"), default="fsr-best"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--close", type=int, default=2)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--context-margin", type=float, default=2.0)
    return parser


def _resize_panel(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] * 3 <= max_width:
        return image
    scale = max_width / float(image.shape[1] * 3)
    size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _with_title(image: np.ndarray, title: str) -> np.ndarray:
    header = np.zeros((_TITLE_HEIGHT, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def build_comparison(
    original: np.ndarray, erased: np.ndarray, max_width: int = 1800
) -> np.ndarray:
    """Return a titled original/difference/erased visualization in BGR."""

    if original.dtype != np.uint8 or erased.dtype != np.uint8:
        raise TypeError("original and erased images must have dtype uint8")
    if original.shape != erased.shape:
        raise ValueError("original and erased images must have the same shape")
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original and erased images must be BGR images")
    if max_width <= 0:
        raise ValueError("max_width must be positive")

    difference = cv2.absdiff(original, erased)
    magnitude = difference.max(axis=2)
    heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_TURBO)
    heatmap[magnitude == 0] = 0
    panels = [_resize_panel(panel, max_width) for panel in (original, heatmap, erased)]
    titled = [
        _with_title(panel, title)
        for panel, title in zip(panels, ("Original", "Difference", "Erased"))
    ]
    return np.hstack(titled)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not write visualization: {}".format(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = DetailRemovalDataset(
        coco_path=args.coco,
        images_dir=args.images_dir,
        category=args.category,
        method=args.method,
        seed=args.seed,
        dilate_px=args.dilate,
        close_px=args.close,
        feather_px=args.feather,
        context_margin=args.context_margin,
    )
    if args.start_index >= len(dataset):
        parser.error("--start-index is outside the dataset")

    stop_index = min(len(dataset), args.start_index + args.count)
    for index in range(args.start_index, stop_index):
        original, erased = dataset[index]
        comparison = build_comparison(original, erased, args.max_width)
        stem = Path(str(dataset.images[index]["file_name"])).stem
        output_path = args.out / "{:03d}_{}.jpg".format(index, stem)
        _write_image(output_path, comparison)
        LOGGER.info("Wrote %s", output_path)
    return 0
