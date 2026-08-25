"""Inpaint every COCO annotation in a category and save visual diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from detail_removal.coco import CocoDataset, rasterize_annotations
from detail_removal.inpaint import create_inpainter
from detail_removal.mask import prepare_mask
from detail_removal.pipeline import remove_details

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inpaint all COCO annotations with the selected category and "
            "write image/mask/visualization outputs."
        )
    )
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--category",
        default="remove",
        help="COCO category to inpaint (default: remove)",
    )
    parser.add_argument(
        "--method",
        choices=("fsr-best", "fsr-fast", "telea", "lama"),
        default="fsr-best",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for --method lama (default: cuda if available, else cpu).",
    )
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--close", type=int, default=2)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--context-margin", type=float, default=2.0)
    parser.add_argument(
        "--max-width",
        type=int,
        default=1800,
        help="maximum total width of each visualization panel",
    )
    parser.add_argument("--png", action="store_true", help="save inpainted images as PNG")
    return parser


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError("Could not write image: {}".format(path))


def _resize_panel(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] * 3 <= max_width:
        return image
    scale = max_width / float(image.shape[1] * 3)
    size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _title_panel(image: np.ndarray, title: str) -> np.ndarray:
    header = np.zeros((32, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    pixels = mask > 0
    overlay[pixels] = np.rint(
        image[pixels].astype(np.float32) * 0.55
        + red[pixels].astype(np.float32) * 0.45
    ).astype(np.uint8)
    return overlay


def build_visualization(
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    max_width: int = 1800,
) -> np.ndarray:
    """Return an Original / Remove mask / Inpainted panel in BGR."""

    if original.shape != result.shape or original.shape[:2] != mask.shape:
        raise ValueError("original, mask, and result dimensions do not match")
    panels = [_resize_panel(panel, max_width) for panel in (
        original,
        _mask_overlay(original, mask),
        result,
    )]
    titled = [
        _title_panel(panel, title)
        for panel, title in zip(panels, ("Original", "Remove mask", "Inpainted"))
    ]
    return np.hstack(titled)


def process_dataset(
    *,
    coco_path: Path,
    images_dir: Path,
    out_dir: Path,
    category: str = "remove",
    method: str = "fsr-best",
    device: Optional[str] = None,
    dilate: int = 15,
    close: int = 2,
    feather: int = 9,
    context_margin: float = 2.0,
    max_width: int = 1800,
    png: bool = False,
) -> List[Mapping[str, Any]]:
    """Process every image and every annotation in ``category``."""

    if max_width <= 0:
        raise ValueError("max_width must be positive")
    dataset = CocoDataset.load(coco_path)
    category_id = dataset.category_id(category)
    inpainter = create_inpainter(method, device=device) if method == "lama" else create_inpainter(method)
    image_out = out_dir / "images"
    mask_out = out_dir / "masks"
    visual_out = out_dir / "visualizations"
    summaries: List[Mapping[str, Any]] = []

    for image_info in dataset.images:
        source_path = images_dir / str(image_info["file_name"])
        original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError("Could not read image: {}".format(source_path))
        height, width = original.shape[:2]
        annotations = dataset.annotations_for_image(int(image_info["id"]), category_id)
        union_mask, _ = rasterize_annotations(height, width, annotations)
        prepared = prepare_mask(
            union_mask,
            dilate_px=dilate,
            close_px=close,
            feather_px=feather,
            context_margin=context_margin,
        )
        result = remove_details(original, prepared, inpainter)

        source_name = Path(str(image_info["file_name"]))
        stem = source_name.stem
        image_suffix = ".png" if png else (source_name.suffix or ".jpg")
        _write_image(image_out / (stem + image_suffix), result)
        _write_image(mask_out / (stem + "_mask.png"), prepared.mask)
        visualization = build_visualization(original, prepared.mask, result, max_width)
        _write_image(visual_out / (stem + "_comparison.jpg"), visualization)

        summary: Dict[str, Any] = {
            "image_id": int(image_info["id"]),
            "source_file": str(image_info["file_name"]),
            "annotation_ids": [int(item["id"]) for item in annotations],
            "annotation_count": len(annotations),
            "component_count": len(prepared.components),
            "size": [width, height],
            "inpainted_file": str(Path("images") / (stem + image_suffix)),
            "mask_file": str(Path("masks") / (stem + "_mask.png")),
            "visualization_file": str(
                Path("visualizations") / (stem + "_comparison.jpg")
            ),
        }
        summaries.append(summary)
        LOGGER.info(
            "Processed %s: %d remove annotations -> %d components",
            source_name,
            len(annotations),
            len(prepared.components),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    return summaries


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if min(args.dilate, args.close, args.feather) < 0:
        args_parser = build_parser()
        args_parser.error("--dilate, --close, and --feather must be non-negative")
    if args.context_margin < 0:
        build_parser().error("--context-margin must be non-negative")
    process_dataset(
        coco_path=args.coco,
        images_dir=args.images_dir,
        out_dir=args.out,
        category=args.category,
        method=args.method,
        device=args.device,
        dilate=args.dilate,
        close=args.close,
        feather=args.feather,
        context_margin=args.context_margin,
        max_width=args.max_width,
        png=args.png,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
