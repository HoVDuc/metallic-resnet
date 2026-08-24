"""Precompute and load original/erased image pairs for training."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .coco import CocoDataset, rasterize_annotations
from .inpaint.base import Inpainter
from .inpaint.opencv import create_inpainter
from .mask import prepare_mask
from .pipeline import remove_details
from .selection import select_random_annotations

LOGGER = logging.getLogger(__name__)
ImageCollection = Union[List[np.ndarray], np.ndarray]
PairBatch = Tuple[ImageCollection, ImageCollection, ImageCollection]
PairTask = Tuple[int, Mapping[str, Any], Sequence[Mapping[str, Any]]]
PairedTransform = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
_WORKER_CONTEXT: Optional[Mapping[str, Any]] = None


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
    erased = remove_details(original, prepared_mask, inpainter)

    file_name = "{:06d}.png".format(sample_id)
    original_path = out_dir / "original" / file_name
    erased_path = out_dir / "erased" / file_name
    removed_mask_path = out_dir / "removed_masks" / file_name
    _write_png(original_path, original)
    _write_png(erased_path, erased)
    _write_png(removed_mask_path, prepared_mask.mask)
    return {
        "sample_id": sample_id,
        "source_image_id": int(image_info["id"]),
        "source_file_name": str(image_info["file_name"]),
        "source_size": [source_width, source_height],
        "size": [original.shape[1], original.shape[0]],
        "scale": scale,
        "selected_annotation_ids": [int(annotation["id"]) for annotation in selected],
        "original_path": str(original_path.relative_to(out_dir)),
        "erased_path": str(erased_path.relative_to(out_dir)),
        "removed_mask_path": str(removed_mask_path.relative_to(out_dir)),
    }


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


def generate_erased_pairs(
    *,
    coco_path: Path,
    images_dir: Path,
    out_dir: Path,
    count: int = 1000,
    max_size: int = 2048,
    category: str = "detail",
    method: str = "fsr-best",
    inpainter: Optional[Inpainter] = None,
    seed: Optional[int] = None,
    dilate_px: int = 15,
    close_px: int = 2,
    feather_px: int = 9,
    context_margin: float = 2.0,
    workers: int = 1,
    progress: bool = True,
) -> None:
    """Generate lossless original/erased pairs and their removal metadata."""

    if count <= 0:
        raise ValueError("count must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers > 1 and inpainter is not None:
        raise ValueError("Custom inpainter requires workers=1")
    source = CocoDataset.load(Path(coco_path))
    category_id = source.category_id(category)
    eligible = [
        (image_info, source.annotations_for_image(int(image_info["id"]), category_id))
        for image_info in source.images
    ]
    eligible = [(image_info, annotations) for image_info, annotations in eligible if annotations]
    if not eligible:
        raise ValueError("No images have annotations for category {!r}".format(category))

    out_dir = Path(out_dir)
    _ensure_empty_output_dir(out_dir)
    original_dir = out_dir / "original"
    erased_dir = out_dir / "erased"
    removed_mask_dir = out_dir / "removed_masks"
    original_dir.mkdir()
    erased_dir.mkdir()
    removed_mask_dir.mkdir()

    rng = random.Random(seed)
    tasks = _build_pair_tasks(eligible, count, rng)
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        if workers == 1:
            active_inpainter = inpainter or create_inpainter(method)
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
                records = pool.imap(_generate_pair_in_worker, tasks)
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
        original = cv2.imread(
            str(self.root_dir / str(record["original_path"])), cv2.IMREAD_COLOR
        )
        erased = cv2.imread(
            str(self.root_dir / str(record["erased_path"])), cv2.IMREAD_COLOR
        )
        if original is None or erased is None:
            raise FileNotFoundError("Could not load precomputed pair at index {}".format(index))
        if original.shape != erased.shape:
            raise ValueError("Original and erased pair dimensions do not match")
        if self.paired_transform is not None:
            transformed = self.paired_transform(original, erased)
            original, erased = _validate_transformed_pair(transformed)
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
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--category", default="detail")
    parser.add_argument(
        "--method", choices=("fsr-best", "fsr-fast", "telea"), default="fsr-best"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--close", type=int, default=2)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--context-margin", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
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
        seed=args.seed,
        dilate_px=args.dilate,
        close_px=args.close,
        feather_px=args.feather,
        context_margin=args.context_margin,
        workers=args.workers,
    )
    return 0
