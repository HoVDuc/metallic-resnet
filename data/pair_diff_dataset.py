"""Synthetic paired-image dataset built exclusively from precomputed erased patches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from data.transforms import (
    GeometricParams,
    apply_geometry,
    apply_photometric,
    apply_registration_jitter,
    downsample_soft_target,
    normalize_imagenet,
    sample_shared_geometry,
)


DEFAULT_AUGMENTATION = {
    "shared_scale": (0.9, 1.1),
    "brightness_contrast": 0.15,
    "gamma": (0.85, 1.2),
    "channel_gain": 0.05,
    "motion_blur_kernel": (3, 15),
    "defocus_sigma": (0.0, 1.2),
    "gaussian_noise_sigma_255": (2, 6),
    "poisson_noise_scale": (0.1, 0.1),
    "illumination_gradient": 0.08,
    "jpeg_quality": (70, 95),
}


def pair_diff_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack pair tensors while retaining per-sample metadata as dictionaries."""
    if not samples:
        raise ValueError("cannot collate an empty pair batch")
    target_names = tuple(samples[0]["targets"])
    if any(tuple(sample["targets"]) != target_names for sample in samples[1:]):
        raise ValueError("all pair samples must contain the same target taps")
    return {
        "view_a": torch.stack([sample["view_a"] for sample in samples]),
        "view_b": torch.stack([sample["view_b"] for sample in samples]),
        "targets": {
            name: torch.stack([sample["targets"][name] for sample in samples])
            for name in target_names
        },
        "meta": [sample["meta"] for sample in samples],
    }


def difference_target_sizes(crop_size: int, output_stride: int) -> dict[str, tuple[int, int]]:
    """Return the target resolution contract shared by data, model, and probes."""
    crop_size = int(crop_size)
    output_stride = int(output_stride)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if output_stride not in (4, 8):
        raise ValueError("output_stride must be 4 or 8")
    stride_four = (max(1, crop_size // 4), max(1, crop_size // 4))
    final = (
        max(1, crop_size // output_stride),
        max(1, crop_size // output_stride),
    )
    return {"f0": stride_four, "f1": stride_four, "f2": final}


def sample_erasure_sets(
    annotation_ids: Iterable[int],
    rng: np.random.Generator,
    hard_negative_probability: float = 0.25,
) -> tuple[list[int], list[int]]:
    """Select 1--3 unique annotations per view, optionally forcing overlap."""
    ids = np.asarray(sorted({int(item) for item in annotation_ids}), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("at least one annotation id is required")
    if not 0.0 <= hard_negative_probability <= 1.0:
        raise ValueError("hard_negative_probability must be in [0, 1]")

    maximum = min(3, int(ids.size))
    count_a = int(rng.integers(1, maximum + 1))
    count_b = int(rng.integers(1, maximum + 1))
    force_overlap = bool(rng.random() < hard_negative_probability)
    if force_overlap:
        shared = int(rng.choice(ids))
        remaining = ids[ids != shared]
        selected_a = [shared]
        selected_b = [shared]
        if count_a > 1:
            selected_a.extend(int(item) for item in rng.choice(remaining, count_a - 1, replace=False))
        if count_b > 1:
            selected_b.extend(int(item) for item in rng.choice(remaining, count_b - 1, replace=False))
    else:
        selected_a = [int(item) for item in rng.choice(ids, count_a, replace=False)]
        selected_b = [int(item) for item in rng.choice(ids, count_b, replace=False)]
    return sorted(selected_a), sorted(selected_b)


def sample_crop(
    rng: np.random.Generator,
    image_shape: Sequence[int],
    crop_size: int,
    positive_mask: np.ndarray | None,
    biased_probability: float = 0.7,
    *,
    positive_points: np.ndarray | None = None,
    positive_boxes: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Sample an in-bounds crop, biasing its centre to a positive pixel when possible."""
    if len(image_shape) < 2 or crop_size <= 0:
        raise ValueError("image_shape and crop_size must describe a positive image crop")
    if not 0.0 <= biased_probability <= 1.0:
        raise ValueError("biased_probability must be in [0, 1]")
    height, width = int(image_shape[0]), int(image_shape[1])
    max_x, max_y = max(0, width - crop_size), max(0, height - crop_size)
    mask_points = None if positive_mask is None else np.argwhere(positive_mask > 0)
    if positive_points is not None:
        supplied_points = np.asarray(positive_points)
        if supplied_points.ndim != 2 or supplied_points.shape[1] != 2:
            raise ValueError("positive_points must contain source-space (y, x) pairs")
        mask_points = supplied_points if mask_points is None else np.concatenate((mask_points, supplied_points))
    boxes = [] if positive_boxes is None else list(positive_boxes)
    biased = bool(
        ((mask_points is not None and len(mask_points) > 0) or boxes)
        and rng.random() < biased_probability
    )
    if biased:
        if mask_points is not None and len(mask_points) > 0:
            point_y, point_x = mask_points[int(rng.integers(0, len(mask_points)))]
        else:
            x0, y0, x1, y1 = (int(value) for value in boxes[int(rng.integers(0, len(boxes)))])
            point_x = int(rng.integers(max(0, x0), min(width, x1)))
            point_y = int(rng.integers(max(0, y0), min(height, y1)))
        x = int(np.clip(int(point_x) - crop_size // 2, 0, max_x))
        y = int(np.clip(int(point_y) - crop_size // 2, 0, max_y))
    else:
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
    return {"x": x, "y": y, "size": int(crop_size), "biased": biased}


def _read_cached(path: Path, flags: int) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError("Could not read cached pair asset: {}".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop_source(source: np.ndarray, crop: Mapping[str, Any]) -> np.ndarray:
    crop_x, crop_y, crop_size = int(crop["x"]), int(crop["y"]), int(crop["size"])
    region = source[
        crop_y : min(source.shape[0], crop_y + crop_size),
        crop_x : min(source.shape[1], crop_x + crop_size),
    ]
    pad_bottom = crop_size - region.shape[0]
    pad_right = crop_size - region.shape[1]
    if pad_bottom or pad_right:
        border_mode = cv2.BORDER_REFLECT_101 if min(region.shape[:2]) > 1 else cv2.BORDER_REFLECT
        region = cv2.copyMakeBorder(region, 0, pad_bottom, 0, pad_right, border_mode)
    return region.copy()


def _roi_intersection(
    bbox: Sequence[int], crop: Mapping[str, Any]
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    crop_x, crop_y, crop_size = int(crop["x"]), int(crop["y"]), int(crop["size"])
    intersection = (
        max(x0, crop_x),
        max(y0, crop_y),
        min(x1, crop_x + crop_size),
        min(y1, crop_y + crop_size),
    )
    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        return None
    return intersection


def compose_cached_view(
    source: np.ndarray,
    image_id: int,
    selected_ids: Iterable[int],
    annotation_entries: Mapping[str, Mapping[str, Any]],
    cache_dir: Path | str,
    rng: np.random.Generator,
    different_from: Mapping[int, int] | None = None,
    *,
    crop: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Paste selected final erased ROIs in source coordinates.

    Cache variants already contain the precompute-time feather composite that QA
    inspected.  The binary mask gates that final result so training sees exactly
    the QA-approved pixels without applying the alpha a second time.
    """
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source must be a BGR HWC image")
    cache_root = Path(cache_dir)
    active_crop: Mapping[str, Any] = (
        {"x": 0, "y": 0, "size": max(source.shape[:2])} if crop is None else crop
    )
    output = source.copy() if crop is None else _crop_source(source, crop)
    variants_used: dict[int, int] = {}
    for annotation_id in sorted({int(item) for item in selected_ids}):
        entry = annotation_entries[str(annotation_id)]
        variant_names = list(entry["variants"])
        if not variant_names:
            raise ValueError("annotation {} has no cached variants".format(annotation_id))
        choices = list(range(len(variant_names)))
        forbidden = None if different_from is None else different_from.get(annotation_id)
        if forbidden in choices and len(choices) >= 2:
            choices.remove(int(forbidden))
        variant_id = int(choices[int(rng.integers(0, len(choices)))])
        variants_used[annotation_id] = variant_id

        x0, y0, x1, y1 = (int(value) for value in entry["bbox_xyxy"])
        if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
            raise ValueError("cached bbox is outside source image for annotation {}".format(annotation_id))
        intersection = (
            (x0, y0, x1, y1)
            if crop is None
            else _roi_intersection((x0, y0, x1, y1), active_crop)
        )
        if intersection is None:
            continue
        variant = _read_cached(
            cache_root / "erased" / str(image_id) / variant_names[variant_id], cv2.IMREAD_COLOR
        )
        alpha = _read_cached(
            cache_root / "erased" / str(image_id) / "{}_alpha.png".format(annotation_id),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = _read_cached(
            cache_root / "masks" / str(image_id) / "{}.png".format(annotation_id),
            cv2.IMREAD_GRAYSCALE,
        )
        expected_shape = (y1 - y0, x1 - x0)
        if variant.shape[:2] != expected_shape or alpha.shape != expected_shape or mask.shape != expected_shape:
            raise ValueError("cached ROI dimensions disagree with bbox for annotation {}".format(annotation_id))
        ix0, iy0, ix1, iy1 = intersection
        patch_slice = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        crop_x = 0 if crop is None else int(active_crop["x"])
        crop_y = 0 if crop is None else int(active_crop["y"])
        output_slice = (
            slice(iy0 - crop_y, iy1 - crop_y),
            slice(ix0 - crop_x, ix1 - crop_x),
        )
        active_mask = mask[patch_slice] > 0
        roi = output[output_slice]
        patch = variant[patch_slice]
        roi[active_mask] = patch[active_mask]
    return output, variants_used


def _compose_target_crop(
    image_id: int,
    selected_ids: Iterable[int],
    annotation_entries: Mapping[str, Mapping[str, Any]],
    cache_dir: Path,
    crop: Mapping[str, Any],
) -> np.ndarray:
    target = np.zeros((int(crop["size"]), int(crop["size"])), dtype=np.uint8)
    for annotation_id in sorted({int(item) for item in selected_ids}):
        entry = annotation_entries[str(annotation_id)]
        x0, y0, x1, y1 = (int(value) for value in entry["bbox_xyxy"])
        intersection = _roi_intersection((x0, y0, x1, y1), crop)
        if intersection is None:
            continue
        mask = _read_cached(
            cache_dir / "masks" / str(image_id) / "{}.png".format(annotation_id),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask.shape != (y1 - y0, x1 - x0):
            raise ValueError("cached mask dimensions disagree with bbox for annotation {}".format(annotation_id))
        ix0, iy0, ix1, iy1 = intersection
        mask_slice = mask[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
        destination = target[
            iy0 - int(crop["y"]) : iy1 - int(crop["y"]),
            ix0 - int(crop["x"]) : ix1 - int(crop["x"]),
        ]
        np.maximum(destination, mask_slice, out=destination)
    return target


def _positive_source_points(
    image_id: int,
    selected_ids: Iterable[int],
    annotation_entries: Mapping[str, Mapping[str, Any]],
    cache_dir: Path,
) -> np.ndarray:
    """Return true positive mask pixels in source ``(y, x)`` coordinates."""
    point_groups: list[np.ndarray] = []
    for annotation_id in sorted({int(item) for item in selected_ids}):
        entry = annotation_entries[str(annotation_id)]
        x0, y0, x1, y1 = (int(value) for value in entry["bbox_xyxy"])
        mask = _read_cached(
            cache_dir / "masks" / str(image_id) / "{}.png".format(annotation_id),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask.shape != (y1 - y0, x1 - x0):
            raise ValueError("cached mask dimensions disagree with bbox for annotation {}".format(annotation_id))
        local_points = np.argwhere(mask > 0)
        if local_points.size:
            local_points = local_points.astype(np.int64, copy=False)
            local_points[:, 0] += y0
            local_points[:, 1] += x0
            point_groups.append(local_points)
    if not point_groups:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(point_groups, axis=0)


def split_image_ids(
    image_ids: Iterable[int],
    val_fraction: float,
    seed: int,
    linked_groups: Iterable[Iterable[int]] = ((6, 9),),
) -> tuple[list[int], list[int]]:
    """Split ids deterministically while treating linked ids as indivisible units."""
    ids = sorted({int(item) for item in image_ids})
    if len(ids) < 2:
        raise ValueError("at least two image ids are required for a split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    remaining = set(ids)
    units: list[list[int]] = []
    for linked in linked_groups:
        unit = sorted(remaining & {int(item) for item in linked})
        if unit:
            units.append(unit)
            remaining.difference_update(unit)
    units.extend([[item] for item in sorted(remaining)])
    rng = np.random.default_rng(int(seed))
    rng.shuffle(units)
    target = max(1, min(len(ids) - 1, int(round(len(ids) * val_fraction))))
    validation: list[int] = []
    for unit in units:
        if not validation or len(validation) < target:
            validation.extend(unit)
    if len(validation) == len(ids):
        moved = units[-1]
        validation = [item for item in validation if item not in moved]
    val_set = set(validation)
    return sorted(set(ids) - val_set), sorted(val_set)


class PairDiffDataset(Dataset):
    """Generate deterministic synthetic difference pairs without online inpainting."""

    def __init__(
        self,
        coco_path: Path | str,
        images_dir: Path | str,
        cache_dir: Path | str,
        *,
        image_ids: Iterable[int] | None = None,
        crop_size: int = 512,
        output_stride: int = 8,
        samples_per_epoch: int | None = None,
        base_seed: int = 0,
        hard_negative_probability: float = 0.25,
        biased_crop_probability: float = 0.7,
        registration_translate_pixels: int = 3,
        registration_rotate_degrees: float = 0.5,
        augmentation_config: Mapping[str, Any] | None = None,
        augment: bool = True,
        manifest_path: Path | str | None = None,
    ) -> None:
        self.coco_path = Path(coco_path)
        self.images_dir = Path(images_dir)
        self.cache_dir = Path(cache_dir)
        self.crop_size = int(crop_size)
        self.output_stride = int(output_stride)
        self.base_seed = int(base_seed)
        self.hard_negative_probability = float(hard_negative_probability)
        self.biased_crop_probability = float(biased_crop_probability)
        self.registration_translate_pixels = int(registration_translate_pixels)
        self.registration_rotate_degrees = float(registration_rotate_degrees)
        self.augmentation_config = dict(
            DEFAULT_AUGMENTATION if augmentation_config is None else augmentation_config
        )
        self.augment = bool(augment)
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        difference_target_sizes(self.crop_size, self.output_stride)

        coco = json.loads(self.coco_path.read_text(encoding="utf-8"))
        manifest_file = Path(manifest_path) if manifest_path is not None else self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        coco_images = {int(item["id"]): item for item in coco["images"]}
        available = sorted(int(item) for item in self.manifest.get("images", {}))
        chosen = available if image_ids is None else sorted({int(item) for item in image_ids})
        missing = [item for item in chosen if item not in coco_images or str(item) not in self.manifest["images"]]
        if missing:
            raise ValueError("image ids are missing from COCO or cache manifest: {}".format(missing))
        if not chosen:
            raise ValueError("PairDiffDataset requires at least one source image")
        for image_id in chosen:
            image_manifest = self.manifest["images"][str(image_id)]
            passing_annotations = {
                annotation_id: entry
                for annotation_id, entry in image_manifest.get("annotations", {}).items()
                if bool(entry.get("qa_passed", entry.get("qa", {}).get("passed", False)))
            }
            if not passing_annotations:
                raise ValueError(
                    "image {} has no QA-passing cached annotations".format(image_id)
                )
            for annotation_id, entry in passing_annotations.items():
                variant_paths = [
                    self.cache_dir / "erased" / str(image_id) / str(name)
                    for name in entry.get("variants", ())
                ]
                hashes = [_sha256(path) for path in variant_paths]
                if len(hashes) != len(set(hashes)):
                    raise ValueError(
                        "byte-identical cached variants for image {} annotation {}; "
                        "rebuild the cache with diverse built-in recipes".format(
                            image_id, annotation_id
                        )
                    )
            image_manifest["annotations"] = passing_annotations
        self.image_ids = chosen
        self._coco_images = coco_images
        self.samples_per_epoch = int(samples_per_epoch or len(chosen))
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")

    def __len__(self) -> int:
        return self.samples_per_epoch

    @property
    def epoch(self) -> int:
        """Current epoch read from worker-visible shared memory."""
        return int(self._epoch_state.item())

    def set_epoch(self, epoch: int) -> None:
        """Set resume-visible epoch state used by deterministic per-index sampling."""
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch_state.fill_(int(epoch))

    def _rng(self, index: int, base_seed: int | None = None) -> tuple[np.random.Generator, int]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        sequence = np.random.SeedSequence(
            [self.base_seed if base_seed is None else int(base_seed), self.epoch, int(index), worker_id]
        )
        return np.random.default_rng(sequence), worker_id

    def sample(self, index: int, *, seed: int | None = None) -> dict[str, Any]:
        """Generate one pair, optionally replacing the base seed for fixed validation."""
        if index < 0:
            raise IndexError(index)
        rng, worker_id = self._rng(index, seed)
        image_id = int(self.image_ids[int(rng.integers(0, len(self.image_ids)))])
        image_manifest = self.manifest["images"][str(image_id)]
        entries = image_manifest["annotations"]
        annotation_ids = [int(item) for item in entries]
        selected_a, selected_b = sample_erasure_sets(
            annotation_ids, rng, self.hard_negative_probability
        )
        selected_difference = sorted(set(selected_a) ^ set(selected_b))

        source_filename = str(image_manifest.get("source_filename", self._coco_images[image_id]["file_name"]))
        source = cv2.imread(str(self.images_dir / source_filename), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError("Could not read source image: {}".format(self.images_dir / source_filename))
        positive_points = _positive_source_points(
            image_id, selected_difference, entries, self.cache_dir
        )
        crop = sample_crop(
            rng,
            source.shape,
            self.crop_size,
            None,
            self.biased_crop_probability,
            positive_points=positive_points,
        )
        view_a, variants_a = compose_cached_view(
            source, image_id, selected_a, entries, self.cache_dir, rng, crop=crop
        )
        view_b, variants_b = compose_cached_view(
            source,
            image_id,
            selected_b,
            entries,
            self.cache_dir,
            rng,
            different_from=variants_a,
            crop=crop,
        )
        target = _compose_target_crop(
            image_id, selected_difference, entries, self.cache_dir, crop
        )
        if crop["biased"] and not np.any(target):
            raise RuntimeError("internal error: biased crop contains no positive mask pixels")
        if self.augment:
            geometry = sample_shared_geometry(
                rng,
                self.crop_size,
                view_a.shape,
                registration_translate_pixels=self.registration_translate_pixels,
                registration_rotate_degrees=self.registration_rotate_degrees,
                shared_scale=self.augmentation_config.get("shared_scale", (0.9, 1.1)),
            )
        else:
            geometry = GeometricParams(
                crop_x=0.0,
                crop_y=0.0,
                crop_size=self.crop_size,
                target_dilate_pixels=self.registration_translate_pixels,
            )
        view_a, transformed_target = apply_geometry(view_a, target, geometry, cv2.INTER_LINEAR)
        view_b, _ = apply_geometry(view_b, target, geometry, cv2.INTER_LINEAR)
        if self.augment:
            view_a = apply_registration_jitter(
                view_a,
                rng,
                max_translate=self.registration_translate_pixels,
                max_angle=self.registration_rotate_degrees,
            )
            view_b = apply_registration_jitter(
                view_b,
                rng,
                max_translate=self.registration_translate_pixels,
                max_angle=self.registration_rotate_degrees,
            )
            view_a = apply_photometric(view_a, rng, self.augmentation_config)
            view_b = apply_photometric(view_b, rng, self.augmentation_config)

        target_sizes = difference_target_sizes(self.crop_size, self.output_stride)
        targets = {
            name: downsample_soft_target(transformed_target, size)
            for name, size in target_sizes.items()
        }
        positive_fraction = float(np.count_nonzero(transformed_target) / transformed_target.size)
        meta = {
            "source_image_id": image_id,
            "source_filename": source_filename,
            "selected_a": selected_a,
            "selected_b": selected_b,
            "symmetric_difference": selected_difference,
            "crop": crop,
            "variant_ids": {
                "view_a": {str(key): value for key, value in variants_a.items()},
                "view_b": {str(key): value for key, value in variants_b.items()},
            },
            "positive_fraction": positive_fraction,
            "epoch": self.epoch,
            "index": int(index),
            "worker_id": worker_id,
        }
        return {
            "view_a": normalize_imagenet(view_a),
            "view_b": normalize_imagenet(view_b),
            "targets": targets,
            "meta": meta,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized_index = int(index)
        if normalized_index < 0:
            normalized_index += len(self)
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(index)
        return self.sample(normalized_index)


__all__ = [
    "PairDiffDataset",
    "compose_cached_view",
    "difference_target_sizes",
    "pair_diff_collate",
    "sample_crop",
    "sample_erasure_sets",
    "split_image_ids",
]
