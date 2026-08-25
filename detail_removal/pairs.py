"""Precompute and load original/erased image pairs for training."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .coco import CocoDataset, rasterize_annotations
from .inpaint import Inpainter, create_inpainter
from .mask import PreparedMask, prepare_mask
from .pipeline import remove_details, remove_details_batch
from .selection import select_random_annotations

LOGGER = logging.getLogger(__name__)
ImageCollection = Union[List[np.ndarray], np.ndarray]
PairBatch = Tuple[ImageCollection, ImageCollection, ImageCollection]
PairTask = Tuple[int, Mapping[str, Any], Sequence[Mapping[str, Any]]]
PairViewTask = Tuple[
    int,
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    Sequence[Mapping[str, Any]],
    Sequence[Mapping[str, Any]],
]
PairedTransform = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
_WORKER_CONTEXT: Optional[Mapping[str, Any]] = None
_PAIR_NAME_RE = re.compile(
    r"^(?P<key>pair_[^_]+_(?:forward|swapped)_masked)_image(?P<side>[12])(?:_|\.)"
)


class _PreparedPair:
    """Everything needed to inpaint and write one pair, before inpainting runs."""

    __slots__ = (
        "sample_id",
        "image_info",
        "selected",
        "source_width",
        "source_height",
        "original",
        "scale",
        "prepared_mask",
    )

    def __init__(
        self,
        sample_id: int,
        image_info: Mapping[str, Any],
        selected: Sequence[Mapping[str, Any]],
        source_width: int,
        source_height: int,
        original: np.ndarray,
        scale: float,
        prepared_mask: PreparedMask,
    ) -> None:
        self.sample_id = sample_id
        self.image_info = image_info
        self.selected = selected
        self.source_width = source_width
        self.source_height = source_height
        self.original = original
        self.scale = scale
        self.prepared_mask = prepared_mask


class _PreparedPairViews:
    """Prepared data for one source image pair and its symmetric-difference mask."""

    __slots__ = (
        "sample_id",
        "pair_key",
        "image_info_a",
        "image_info_b",
        "selected_a",
        "selected_b",
        "source_width",
        "source_height",
        "image_a",
        "image_b",
        "scale",
        "prepared_a",
        "prepared_b",
        "target_mask",
    )

    def __init__(
        self,
        sample_id: int,
        pair_key: str,
        image_info_a: Mapping[str, Any],
        image_info_b: Mapping[str, Any],
        selected_a: Sequence[Mapping[str, Any]],
        selected_b: Sequence[Mapping[str, Any]],
        source_width: int,
        source_height: int,
        image_a: np.ndarray,
        image_b: np.ndarray,
        scale: float,
        prepared_a: PreparedMask,
        prepared_b: PreparedMask,
        target_mask: np.ndarray,
    ) -> None:
        self.sample_id = sample_id
        self.pair_key = pair_key
        self.image_info_a = image_info_a
        self.image_info_b = image_info_b
        self.selected_a = selected_a
        self.selected_b = selected_b
        self.source_width = source_width
        self.source_height = source_height
        self.image_a = image_a
        self.image_b = image_b
        self.scale = scale
        self.prepared_a = prepared_a
        self.prepared_b = prepared_b
        self.target_mask = target_mask


def resize_to_max_size(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    """Downscale an image while preserving aspect ratio and never upscale it."""

    if max_size <= 0:
        raise ValueError("max_size must be positive")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image.copy(), 1.0
    scale = max_size / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not write image: {}".format(path))


def _ensure_empty_output_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            "Output directory is not empty: {}. Choose a new directory.".format(
                out_dir
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _build_pair_tasks(
    eligible: Sequence[Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    count: int,
    rng: random.Random,
) -> List[PairTask]:
    source_order = list(range(len(eligible)))
    rng.shuffle(source_order)
    tasks: List[PairTask] = []
    for sample_id in range(count):
        if sample_id and sample_id % len(source_order) == 0:
            rng.shuffle(source_order)
        image_info, annotations = eligible[source_order[sample_id % len(source_order)]]
        tasks.append((sample_id, image_info, select_random_annotations(annotations, rng)))
    return tasks


def _find_source_pairs(
    source: CocoDataset, category_id: int
) -> List[Tuple[str, Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]:
    """Find ``...image1``/``...image2`` source pairs by filename convention."""

    groups: dict[str, dict[int, Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]]] = {}
    for image_info in source.images:
        match = _PAIR_NAME_RE.match(str(image_info.get("file_name", "")))
        if match is None:
            continue
        key = match.group("key")
        side = int(match.group("side"))
        annotations = source.annotations_for_image(int(image_info["id"]), category_id)
        groups.setdefault(key, {})[side] = (image_info, annotations)

    pairs = []
    for key in sorted(groups):
        group = groups[key]
        if 1 not in group or 2 not in group:
            LOGGER.warning("Ignoring incomplete source pair %s", key)
            continue
        image_a, annotations_a = group[1]
        image_b, annotations_b = group[2]
        if not annotations_a or not annotations_b:
            continue
        pairs.append((key, image_a, image_b, annotations_a, annotations_b))
    return pairs


def _build_pair_view_tasks(
    eligible: Sequence[
        Tuple[str, Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ],
    count: int,
    rng: random.Random,
) -> List[PairViewTask]:
    if not eligible:
        raise ValueError("No complete image pairs with annotations were found")
    order = list(range(len(eligible)))
    rng.shuffle(order)
    tasks: List[PairViewTask] = []
    for sample_id in range(count):
        if sample_id and sample_id % len(order) == 0:
            rng.shuffle(order)
        key, image_a, image_b, annotations_a, annotations_b = eligible[
            order[sample_id % len(order)]
        ]
        tasks.append(
            (
                sample_id,
                key,
                image_a,
                image_b,
                select_random_annotations(annotations_a, rng),
                select_random_annotations(annotations_b, rng),
            )
        )
    return tasks


def _prepare_pair_task(
    task: PairTask,
    *,
    images_dir: Path,
    max_size: int,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
) -> _PreparedPair:
    sample_id, image_info, selected = task
    source_path = images_dir / str(image_info["file_name"])
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise FileNotFoundError("Could not read image: {}".format(source_path))

    source_height, source_width = source_image.shape[:2]
    source_mask, _ = rasterize_annotations(source_height, source_width, selected)
    original, scale = resize_to_max_size(source_image, max_size)
    if scale == 1.0:
        resized_mask = source_mask
    else:
        resized_mask = cv2.resize(
            source_mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    prepared_mask = prepare_mask(
        resized_mask,
        dilate_px=dilate_px,
        close_px=close_px,
        feather_px=feather_px,
        context_margin=context_margin,
    )
    return _PreparedPair(
        sample_id=sample_id,
        image_info=image_info,
        selected=selected,
        source_width=source_width,
        source_height=source_height,
        original=original,
        scale=scale,
        prepared_mask=prepared_mask,
    )


def _prepare_pair_view_task(
    task: PairViewTask,
    *,
    images_dir: Path,
    max_size: int,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
) -> _PreparedPairViews:
    (
        sample_id,
        pair_key,
        image_info_a,
        image_info_b,
        selected_a,
        selected_b,
    ) = task
    source_a = images_dir / str(image_info_a["file_name"])
    source_b = images_dir / str(image_info_b["file_name"])
    image_a_raw = cv2.imread(str(source_a), cv2.IMREAD_COLOR)
    image_b_raw = cv2.imread(str(source_b), cv2.IMREAD_COLOR)
    if image_a_raw is None:
        raise FileNotFoundError("Could not read image: {}".format(source_a))
    if image_b_raw is None:
        raise FileNotFoundError("Could not read image: {}".format(source_b))
    if image_a_raw.shape != image_b_raw.shape:
        raise ValueError(
            "Source pair {} has different image shapes: {} vs {}".format(
                pair_key, image_a_raw.shape, image_b_raw.shape
            )
        )

    source_height, source_width = image_a_raw.shape[:2]
    mask_a_raw, _ = rasterize_annotations(source_height, source_width, selected_a)
    mask_b_raw, _ = rasterize_annotations(source_height, source_width, selected_b)
    image_a, scale = resize_to_max_size(image_a_raw, max_size)
    image_b, scale_b = resize_to_max_size(image_b_raw, max_size)
    if image_a.shape != image_b.shape or scale != scale_b:
        raise ValueError("Source pair {} did not resize to matching shapes".format(pair_key))
    if scale == 1.0:
        mask_a = mask_a_raw
        mask_b = mask_b_raw
    else:
        target_size = (image_a.shape[1], image_a.shape[0])
        mask_a = cv2.resize(mask_a_raw, target_size, interpolation=cv2.INTER_NEAREST)
        mask_b = cv2.resize(mask_b_raw, target_size, interpolation=cv2.INTER_NEAREST)

    prepared_a = prepare_mask(
        mask_a,
        dilate_px=dilate_px,
        close_px=close_px,
        feather_px=feather_px,
        context_margin=context_margin,
    )
    prepared_b = prepare_mask(
        mask_b,
        dilate_px=dilate_px,
        close_px=close_px,
        feather_px=feather_px,
        context_margin=context_margin,
    )
    # A region selected on both views is considered identical. XOR preserves
    # only the non-overlapping removal regions and explicitly zeros overlap.
    target_mask = np.bitwise_xor(prepared_a.mask, prepared_b.mask)
    return _PreparedPairViews(
        sample_id=sample_id,
        pair_key=pair_key,
        image_info_a=image_info_a,
        image_info_b=image_info_b,
        selected_a=selected_a,
        selected_b=selected_b,
        source_width=source_width,
        source_height=source_height,
        image_a=image_a,
        image_b=image_b,
        scale=scale,
        prepared_a=prepared_a,
        prepared_b=prepared_b,
        target_mask=target_mask,
    )


def _finalize_pair_task(
    prepared: _PreparedPair, erased: np.ndarray, out_dir: Path
) -> Mapping[str, Any]:
    file_name = "{:06d}.png".format(prepared.sample_id)
    original_path = out_dir / "original" / file_name
    erased_path = out_dir / "erased" / file_name
    removed_mask_path = out_dir / "removed_masks" / file_name
    _write_png(original_path, prepared.original)
    _write_png(erased_path, erased)
    _write_png(removed_mask_path, prepared.prepared_mask.mask)
    return {
        "sample_id": prepared.sample_id,
        "source_image_id": int(prepared.image_info["id"]),
        "source_file_name": str(prepared.image_info["file_name"]),
        "source_size": [prepared.source_width, prepared.source_height],
        "size": [prepared.original.shape[1], prepared.original.shape[0]],
        "scale": prepared.scale,
        "selected_annotation_ids": [int(annotation["id"]) for annotation in prepared.selected],
        "original_path": str(original_path.relative_to(out_dir)),
        "erased_path": str(erased_path.relative_to(out_dir)),
        "removed_mask_path": str(removed_mask_path.relative_to(out_dir)),
    }


def _finalize_pair_view_task(
    prepared: _PreparedPairViews,
    erased_a: np.ndarray,
    erased_b: np.ndarray,
    out_dir: Path,
) -> Mapping[str, Any]:
    file_name_a = "{:06d}_a.png".format(prepared.sample_id)
    file_name_b = "{:06d}_b.png".format(prepared.sample_id)
    image_a_path = out_dir / "image_a" / file_name_a
    image_b_path = out_dir / "image_b" / file_name_b
    removed_mask_path = out_dir / "removed_masks" / "{:06d}.png".format(prepared.sample_id)
    _write_png(image_a_path, erased_a)
    _write_png(image_b_path, erased_b)
    _write_png(removed_mask_path, prepared.target_mask)
    return {
        "sample_id": prepared.sample_id,
        "pair_key": prepared.pair_key,
        "source_image_ids": [
            int(prepared.image_info_a["id"]), int(prepared.image_info_b["id"])
        ],
        "source_file_names": [
            str(prepared.image_info_a["file_name"]),
            str(prepared.image_info_b["file_name"]),
        ],
        "source_size": [prepared.source_width, prepared.source_height],
        "size": [prepared.image_a.shape[1], prepared.image_a.shape[0]],
        "scale": prepared.scale,
        "selected_annotation_ids_a": [int(item["id"]) for item in prepared.selected_a],
        "selected_annotation_ids_b": [int(item["id"]) for item in prepared.selected_b],
        "overlap_pixels": int(
            np.count_nonzero(np.bitwise_and(prepared.prepared_a.mask, prepared.prepared_b.mask))
        ),
        "image_a_path": str(image_a_path.relative_to(out_dir)),
        "image_b_path": str(image_b_path.relative_to(out_dir)),
        # Aliases keep generic pair consumers compatible while the explicit
        # names above document that both views are source images.
        "original_path": str(image_a_path.relative_to(out_dir)),
        "erased_path": str(image_b_path.relative_to(out_dir)),
        "removed_mask_path": str(removed_mask_path.relative_to(out_dir)),
    }


def _generate_pair_view_task(
    task: PairViewTask,
    *,
    images_dir: Path,
    out_dir: Path,
    max_size: int,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
    inpainter: Inpainter,
) -> Mapping[str, Any]:
    prepared = _prepare_pair_view_task(
        task,
        images_dir=images_dir,
        max_size=max_size,
        dilate_px=dilate_px,
        close_px=close_px,
        feather_px=feather_px,
        context_margin=context_margin,
    )
    erased_a = remove_details(prepared.image_a, prepared.prepared_a, inpainter)
    erased_b = remove_details(prepared.image_b, prepared.prepared_b, inpainter)
    return _finalize_pair_view_task(prepared, erased_a, erased_b, out_dir)


def _generate_pair_views_in_batches(
    tasks: Sequence[PairViewTask],
    *,
    images_dir: Path,
    out_dir: Path,
    max_size: int,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
    inpainter: Inpainter,
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    """Prepare source pairs on CPU and batch both views through the inpainter."""

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        prepared_chunk = [
            _prepare_pair_view_task(
                task,
                images_dir=images_dir,
                max_size=max_size,
                dilate_px=dilate_px,
                close_px=close_px,
                feather_px=feather_px,
                context_margin=context_margin,
            )
            for task in chunk
        ]
        items = []
        for prepared in prepared_chunk:
            items.extend(
                (
                    (prepared.image_a, prepared.prepared_a),
                    (prepared.image_b, prepared.prepared_b),
                )
            )
        erased = remove_details_batch(items, inpainter)
        for index, prepared in enumerate(prepared_chunk):
            erased_a = erased[index * 2]
            erased_b = erased[index * 2 + 1]
            yield _finalize_pair_view_task(prepared, erased_a, erased_b, out_dir)


def _generate_pair_task(
    task: PairTask,
    *,
    images_dir: Path,
    out_dir: Path,
    max_size: int,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
    inpainter: Inpainter,
) -> Mapping[str, Any]:
    prepared = _prepare_pair_task(
        task,
        images_dir=images_dir,
        max_size=max_size,
        dilate_px=dilate_px,
        close_px=close_px,
        feather_px=feather_px,
        context_margin=context_margin,
    )
    erased = remove_details(prepared.original, prepared.prepared_mask, inpainter)
    return _finalize_pair_task(prepared, erased, out_dir)


def _generate_pairs_in_batches(
    tasks: Sequence[PairTask],
    *,
    images_dir: Path,
    out_dir: Path,
    max_size: int,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
    inpainter: Inpainter,
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    """Prepare masks on CPU, then inpaint each chunk in one GPU batch call."""

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        prepared_chunk = [
            _prepare_pair_task(
                task,
                images_dir=images_dir,
                max_size=max_size,
                dilate_px=dilate_px,
                close_px=close_px,
                feather_px=feather_px,
                context_margin=context_margin,
            )
            for task in chunk
        ]
        erased_chunk = remove_details_batch(
            [(prepared.original, prepared.prepared_mask) for prepared in prepared_chunk],
            inpainter,
        )
        for prepared, erased in zip(prepared_chunk, erased_chunk):
            yield _finalize_pair_task(prepared, erased, out_dir)


def _initialize_pair_worker(
    images_dir: Path,
    out_dir: Path,
    max_size: int,
    method: str,
    dilate_px: int,
    close_px: int,
    feather_px: int,
    context_margin: float,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "images_dir": Path(images_dir),
        "out_dir": Path(out_dir),
        "max_size": max_size,
        "dilate_px": dilate_px,
        "close_px": close_px,
        "feather_px": feather_px,
        "context_margin": context_margin,
        "inpainter": create_inpainter(method),
    }


def _generate_pair_in_worker(task: PairTask) -> Mapping[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Pair worker was not initialized")
    return _generate_pair_task(task, **_WORKER_CONTEXT)


def _generate_pair_view_in_worker(task: PairViewTask) -> Mapping[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Pair worker was not initialized")
    return _generate_pair_view_task(task, **_WORKER_CONTEXT)


def generate_erased_pairs(
    *,
    coco_path: Path,
    images_dir: Path,
    out_dir: Path,
    count: int = 1000,
    max_size: int = 2048,
    category: str = "detail",
    method: str = "fsr-best",
    device: Optional[str] = None,
    inpainter: Optional[Inpainter] = None,
    seed: Optional[int] = None,
    dilate_px: int = 15,
    close_px: int = 2,
    feather_px: int = 9,
    context_margin: float = 2.0,
    workers: int = 1,
    batch_size: int = 1,
    pair_mode: str = "auto",
    progress: bool = True,
) -> None:
    """Generate lossless original/erased pairs and their removal metadata.

    When the resolved inpainter is GPU-batch-capable (currently only
    ``method="lama"`` / a :class:`~detail_removal.inpaint.lama.LamaInpainter`)
    and ``batch_size > 1``, masks are still prepared on CPU one task at a
    time but the inpainting forward pass runs once per ``batch_size`` tasks
    instead of once per task. This requires ``workers=1``.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pair_mode not in {"auto", "pair", "single"}:
        raise ValueError("pair_mode must be 'auto', 'pair', or 'single'")
    if workers > 1 and inpainter is not None:
        raise ValueError("Custom inpainter requires workers=1")
    if workers > 1 and batch_size > 1:
        raise ValueError("batch_size > 1 requires workers=1")
    if workers > 1 and method == "lama":
        raise ValueError("method='lama' requires workers=1; use --batch-size for GPU batching")
    source = CocoDataset.load(Path(coco_path))
    category_id = source.category_id(category)
    eligible = [
        (image_info, source.annotations_for_image(int(image_info["id"]), category_id))
        for image_info in source.images
    ]
    eligible = [(image_info, annotations) for image_info, annotations in eligible if annotations]
    if not eligible:
        raise ValueError("No images have annotations for category {!r}".format(category))

    source_pairs = _find_source_pairs(source, category_id)
    use_pair_mode = pair_mode == "pair" or (pair_mode == "auto" and bool(source_pairs))
    if pair_mode == "pair" and not source_pairs:
        raise ValueError("pair_mode='pair' requested, but no complete image pairs were found")
    if use_pair_mode:
        LOGGER.info("Using paired source mode with %d complete image pairs", len(source_pairs))

    out_dir = Path(out_dir)
    _ensure_empty_output_dir(out_dir)
    original_dir = out_dir / ("image_a" if use_pair_mode else "original")
    erased_dir = out_dir / ("image_b" if use_pair_mode else "erased")
    removed_mask_dir = out_dir / "removed_masks"
    original_dir.mkdir()
    erased_dir.mkdir()
    removed_mask_dir.mkdir()

    rng = random.Random(seed)
    tasks = (
        _build_pair_view_tasks(source_pairs, count, rng)
        if use_pair_mode
        else _build_pair_tasks(eligible, count, rng)
    )
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        if workers == 1:
            active_inpainter = inpainter or (
                create_inpainter(method, device=device)
                if method == "lama"
                else create_inpainter(method)
            )
            if use_pair_mode and batch_size > 1 and hasattr(active_inpainter, "inpaint_many"):
                records = _generate_pair_views_in_batches(
                    tasks,  # type: ignore[arg-type]
                    images_dir=Path(images_dir),
                    out_dir=out_dir,
                    max_size=max_size,
                    dilate_px=dilate_px,
                    close_px=close_px,
                    feather_px=feather_px,
                    context_margin=context_margin,
                    inpainter=active_inpainter,
                    batch_size=batch_size,
                )
            elif not use_pair_mode and batch_size > 1 and hasattr(active_inpainter, "inpaint_many"):
                records = _generate_pairs_in_batches(
                    tasks,  # type: ignore[arg-type]
                    images_dir=Path(images_dir),
                    out_dir=out_dir,
                    max_size=max_size,
                    dilate_px=dilate_px,
                    close_px=close_px,
                    feather_px=feather_px,
                    context_margin=context_margin,
                    inpainter=active_inpainter,
                    batch_size=batch_size,
                )
            elif use_pair_mode:
                records = (
                    _generate_pair_view_task(
                        task,  # type: ignore[arg-type]
                        images_dir=Path(images_dir),
                        out_dir=out_dir,
                        max_size=max_size,
                        dilate_px=dilate_px,
                        close_px=close_px,
                        feather_px=feather_px,
                        context_margin=context_margin,
                        inpainter=active_inpainter,
                    )
                    for task in tasks
                )
            else:
                records = (
                    _generate_pair_task(
                        task,
                        images_dir=Path(images_dir),
                        out_dir=out_dir,
                        max_size=max_size,
                        dilate_px=dilate_px,
                        close_px=close_px,
                        feather_px=feather_px,
                        context_margin=context_margin,
                        inpainter=active_inpainter,
                    )
                    for task in tasks
                )
            for record in tqdm(
                records,
                total=count,
                desc="Generating pairs",
                unit="pair",
                disable=not progress,
            ):
                manifest.write(json.dumps(record) + "\n")
        else:
            with mp.Pool(
                processes=workers,
                initializer=_initialize_pair_worker,
                initargs=(
                    Path(images_dir),
                    out_dir,
                    max_size,
                    method,
                    dilate_px,
                    close_px,
                    feather_px,
                    context_margin,
                ),
            ) as pool:
                records = pool.imap(
                    _generate_pair_view_in_worker if use_pair_mode else _generate_pair_in_worker,
                    tasks,
                )
                for record in tqdm(
                    records,
                    total=count,
                    desc="Generating pairs",
                    unit="pair",
                    disable=not progress,
                ):
                    manifest.write(json.dumps(record) + "\n")


class PrecomputedPairDataset:
    """Read precomputed pairs, optionally augment them, then derive a diff mask."""

    def __init__(
        self, root_dir: Path, *, paired_transform: Optional[PairedTransform] = None
    ) -> None:
        self.root_dir = Path(root_dir)
        manifest_path = self.root_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError("Could not find manifest: {}".format(manifest_path))
        with manifest_path.open("r", encoding="utf-8") as stream:
            self.records: List[Mapping[str, Any]] = [
                json.loads(line) for line in stream if line.strip()
            ]
        if not self.records:
            raise ValueError("Pair manifest is empty: {}".format(manifest_path))
        self.paired_transform = paired_transform

    def __len__(self) -> int:
        return len(self.records)

    def metadata(self, index: int) -> Mapping[str, Any]:
        return self.records[index]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.records[index]
        image_a_key = "image_a_path" if "image_a_path" in record else "original_path"
        image_b_key = "image_b_path" if "image_b_path" in record else "erased_path"
        original = cv2.imread(
            str(self.root_dir / str(record[image_a_key])), cv2.IMREAD_COLOR
        )
        erased = cv2.imread(
            str(self.root_dir / str(record[image_b_key])), cv2.IMREAD_COLOR
        )
        if original is None or erased is None:
            raise FileNotFoundError("Could not load precomputed pair at index {}".format(index))
        if original.shape != erased.shape:
            raise ValueError("Original and erased pair dimensions do not match")
        if self.paired_transform is not None:
            transformed = self.paired_transform(original, erased)
            original, erased = _validate_transformed_pair(transformed)
        stored_mask_path = record.get("removed_mask_path") if "image_a_path" in record else None
        if stored_mask_path is not None:
            stored_mask = cv2.imread(
                str(self.root_dir / str(stored_mask_path)), cv2.IMREAD_GRAYSCALE
            )
            if stored_mask is None or stored_mask.shape != original.shape[:2]:
                raise ValueError("Stored removed mask does not match pair dimensions")
            transform_mask = getattr(self.paired_transform, "transform_mask", None)
            if callable(transform_mask):
                stored_mask = transform_mask(stored_mask)
                if stored_mask.shape != original.shape[:2]:
                    raise ValueError("Transformed removed mask does not match pair dimensions")
            difference_mask = (stored_mask > 0).astype(np.uint8)
        else:
            difference_mask = np.any(original != erased, axis=2).astype(np.uint8)
        return original, erased, difference_mask


def _validate_transformed_pair(
    pair: object,
) -> Tuple[np.ndarray, np.ndarray]:
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


class PrecomputedPairDataLoader:
    """Batch ``(original, erased, difference_mask)`` samples without PyTorch."""

    def __init__(
        self,
        dataset: PrecomputedPairDataset,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        stack: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.stack = stack
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[PairBatch]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)
        stop = len(indices)
        if self.drop_last:
            stop -= stop % self.batch_size
        for start in range(0, stop, self.batch_size):
            samples = [
                self.dataset[index]
                for index in indices[start : start + self.batch_size]
            ]
            originals, erased, masks = zip(*samples)
            yield self._collate(list(originals), list(erased), list(masks))

    def _collate(
        self,
        originals: List[np.ndarray],
        erased: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> PairBatch:
        if not self.stack:
            return originals, erased, masks
        image_shapes = {image.shape for image in originals + erased}
        mask_shapes = {mask.shape for mask in masks}
        if len(image_shapes) != 1 or len(mask_shapes) != 1:
            raise ValueError("Cannot stack pairs that do not have the same shape")
        return np.stack(originals), np.stack(erased), np.stack(masks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute original/erased detail-removal training pairs."
    )
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=2048)
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dilate", type=int, default=8)
    parser.add_argument("--close", type=int, default=2)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--context-margin", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Tasks per GPU forward pass for batch-capable inpainters "
            "(currently --method lama). Requires --workers 1."
        ),
    )
    parser.add_argument(
        "--pair-mode",
        choices=("auto", "pair", "single"),
        default="auto",
        help=(
            "auto detects pair_*_image1/image2 sources; pair forces the new "
            "two-view format and single keeps the legacy format"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_erased_pairs(
        coco_path=args.coco,
        images_dir=args.images_dir,
        out_dir=args.out,
        count=args.count,
        max_size=args.max_size,
        category=args.category,
        method=args.method,
        device=args.device,
        seed=args.seed,
        dilate_px=args.dilate,
        close_px=args.close,
        feather_px=args.feather,
        context_margin=args.context_margin,
        workers=args.workers,
        batch_size=args.batch_size,
        pair_mode=args.pair_mode,
    )
    return 0
