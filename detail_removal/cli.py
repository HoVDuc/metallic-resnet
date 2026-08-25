"""Command-line entry point for COCO-driven detail removal."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import cv2
import numpy as np

from .coco import CocoDataset, rasterize_annotations
from .inpaint import create_inpainter
from .mask import ComponentROI, prepare_mask
from .pipeline import remove_details
from .qa import QAMetrics, measure_component, summarize_metrics
from .selection import select_random_annotations

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove a COCO segmentation category with OpenCV inpainting."
    )
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--category", default="detail")
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
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible random annotation selection.",
    )
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    if not cv2.imwrite(str(path), image, parameters):
        raise OSError("Could not write image: {}".format(path))


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 2] = 255
    pixels = mask > 0
    overlay[pixels] = np.rint(
        image[pixels].astype(np.float32) * 0.55
        + red[pixels].astype(np.float32) * 0.45
    ).astype(np.uint8)
    return overlay


def _write_debug(
    debug_dir: Path,
    stem: str,
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    components: Sequence[ComponentROI],
) -> None:
    overlay = _overlay_mask(original, mask)
    height, width = original.shape[:2]
    scale = min(1.0, 1800.0 / float(width * 3))
    if scale < 1.0:
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        panels = [
            cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
            for panel in (original, overlay, result)
        ]
    else:
        panels = [original, overlay, result]
    _write_image(debug_dir / (stem + "_cmp.jpg"), np.hstack(panels))

    for index, component in enumerate(components, start=1):
        y_slice, x_slice = component.slices
        crop = np.hstack(
            [original[y_slice, x_slice], overlay[y_slice, x_slice], result[y_slice, x_slice]]
        )
        _write_image(
            debug_dir / ("{}_component_{:03d}.jpg".format(stem, index)), crop
        )


def _component_metrics(
    original: np.ndarray,
    result: np.ndarray,
    components: Sequence[ComponentROI],
) -> List[QAMetrics]:
    metrics = []
    for component in components:
        full_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        y_slice, x_slice = component.slices
        full_mask[y_slice, x_slice] = component.local_mask
        metrics.append(measure_component(original, result, full_mask))
    return metrics


def _output_file_name(source_name: str, png: bool) -> str:
    source = Path(source_name)
    return source.with_suffix(".png").name if png else source.name


def _cleaned_coco(
    dataset: CocoDataset, removed_annotation_ids: Set[int], png: bool
) -> Dict[str, Any]:
    cleaned = copy.deepcopy(dict(dataset.data))
    cleaned["annotations"] = [
        annotation
        for annotation in cleaned.get("annotations", [])
        if int(annotation.get("id", -1)) not in removed_annotation_ids
    ]
    if png:
        for image in cleaned.get("images", []):
            image["file_name"] = _output_file_name(image["file_name"], True)
    return cleaned


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = CocoDataset.load(args.coco)
    category_id = dataset.category_id(args.category)
    if args.method == "lama":
        inpainter = create_inpainter(args.method, device=args.device)
    else:
        inpainter = create_inpainter(args.method)
    rng = random.Random(args.seed)

    images_output = args.out / "images"
    masks_output = args.out / "masks"
    debug_output = args.out / "debug"
    images_output.mkdir(parents=True, exist_ok=True)
    masks_output.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_output.mkdir(parents=True, exist_ok=True)

    image_summaries: List[Dict[str, Any]] = []
    removed_annotation_ids: Set[int] = set()
    for image_info in dataset.images:
        source_path = args.images_dir / image_info["file_name"]
        original = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError("Could not read image: {}".format(source_path))
        height, width = original.shape[:2]
        if (width, height) != (
            int(image_info.get("width", width)),
            int(image_info.get("height", height)),
        ):
            LOGGER.warning(
                "Image %s dimensions %sx%s differ from COCO metadata",
                source_path.name,
                width,
                height,
            )
        annotations = dataset.annotations_for_image(image_info["id"], category_id)
        selected_annotations = select_random_annotations(annotations, rng)
        selected_ids = [int(annotation["id"]) for annotation in selected_annotations]
        removed_annotation_ids.update(selected_ids)
        union_mask, _ = rasterize_annotations(height, width, selected_annotations)
        prepared = prepare_mask(
            union_mask,
            dilate_px=args.dilate,
            close_px=args.close,
            feather_px=args.feather,
            context_margin=args.context_margin,
        )
        result = remove_details(original, prepared, inpainter)

        output_name = _output_file_name(image_info["file_name"], args.png)
        stem = Path(output_name).stem
        _write_image(images_output / output_name, result)
        _write_image(masks_output / (stem + "_mask.png"), prepared.mask)
        if args.debug:
            _write_debug(
                debug_output,
                stem,
                original,
                result,
                prepared.mask,
                prepared.components,
            )

        summary = summarize_metrics(
            _component_metrics(original, result, prepared.components)
        )
        summary["file_name"] = output_name
        summary["removed_annotation_ids"] = selected_ids
        image_summaries.append(summary)
        LOGGER.info(
            "Processed %s: removed %d annotations as %d components, %d QA failures",
            image_info["file_name"],
            len(selected_ids),
            summary["component_count"],
            summary["failure_count"],
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "qa.json").open("w", encoding="utf-8") as stream:
        json.dump({"images": image_summaries}, stream, indent=2)
    with (args.out / "_annotations.coco.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            _cleaned_coco(dataset, removed_annotation_ids, args.png),
            stream,
            indent=2,
        )
    return 0
