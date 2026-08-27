"""Train the truncated ResNet difference model on precomputed erased pairs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from detail_removal import (
    PrecomputedPairDataLoader,
    PrecomputedPairDataset,
    SynchronizedPhotometricRotate90Augment,
)
from losses import MultiLayerDiffLoss, PosWeightScheduler, estimate_alpha_max
from models import (
    DifferenceModel,
    PairAugmentConfig,
    PairCropDataset,
    TruncatedResNet101,
    build_targets,
    prepare_pair_batch,
    tap_sizes,
)
from torchvision.models import ResNet101_Weights


LOGGER = logging.getLogger(__name__)


class IndexDataset:
    """A lightweight index view preserving the dataset's NumPy sample contract."""

    def __init__(self, dataset: PrecomputedPairDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def split_indices(size: int, validation_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    if size < 2:
        raise ValueError("at least two pairs are required for train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(round(size * validation_fraction)))
    validation_size = min(validation_size, size - 1)
    return indices[validation_size:], indices[:validation_size]


def _stats_signature(
    *, crop_size: int, seed: int, biased_probability: float, sample_limit: int,
    augmentation_key: str = "none",
) -> str:
    return "crop={}:seed={}:bias={:.6g}:limit={}:aug={}".format(
        crop_size, seed, biased_probability, sample_limit, augmentation_key
    )


def _cached_positive_fractions(
    root: Path,
    dataset: PrecomputedPairDataset,
    indices: Sequence[int],
    *,
    crop_size: int,
    seed: int,
    biased_probability: float,
    refresh: bool,
    augment_config: Optional[PairAugmentConfig] = None,
    curriculum_ramp_epochs: int = 0,
    sample_limit: int = 128,
) -> List[float]:
    """Estimate positive frequency on the actual crop distribution and cache it."""

    augmentation_json = json.dumps(
        (augment_config or PairAugmentConfig(enabled=False)).to_dict(), sort_keys=True
    )
    augmentation_key = hashlib.sha1(augmentation_json.encode("utf-8")).hexdigest()[:12]
    signature = _stats_signature(
        crop_size=crop_size,
        seed=seed,
        biased_probability=biased_probability,
        sample_limit=sample_limit,
        augmentation_key=augmentation_key,
    )
    cache_path = root / "positive_fractions.json"
    cache: dict[str, Any] = {"version": 1, "distributions": {}}
    if cache_path.is_file() and not refresh:
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if loaded.get("version") == 1 and isinstance(loaded.get("distributions"), dict):
                cache = loaded
                values = cache["distributions"].get(signature)
                if isinstance(values, dict) and values:
                    LOGGER.info("Loaded positive-fraction stats from %s", cache_path)
                    return [float(value) for value in values.values()]
        except (OSError, ValueError, TypeError):
            LOGGER.warning("Ignoring invalid positive-fraction cache: %s", cache_path)

    selected = list(indices)
    random.Random(seed).shuffle(selected)
    selected = selected[:sample_limit]
    values: dict[str, float] = {}
    if crop_size > 0:
        crops = PairCropDataset(
            dataset,
            selected,
            crop_size=crop_size,
            seed=seed,
            crops_per_sample=1,
            biased_probability=biased_probability,
            train=True,
            augment_config=augment_config,
            curriculum_ramp_epochs=curriculum_ramp_epochs,
        )
        if curriculum_ramp_epochs > 1:
            crops.set_epoch(curriculum_ramp_epochs - 1)
        for logical_index, dataset_index in enumerate(selected):
            _, _, mask = crops[logical_index]
            sample_id = str(dataset.metadata(dataset_index).get("sample_id", dataset_index))
            values[sample_id] = float(mask.mean())
    else:
        for dataset_index in selected:
            _, _, mask = dataset[dataset_index]
            sample_id = str(dataset.metadata(dataset_index).get("sample_id", dataset_index))
            values[sample_id] = float(mask.mean())
    cache["distributions"][signature] = values
    try:
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        temporary_path.replace(cache_path)
    except OSError as error:
        LOGGER.warning("Could not write positive-fraction cache %s: %s", cache_path, error)
    return list(values.values())


def _trainable_parameters(model: DifferenceModel, head_lr: float, backbone_lr: float):
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    heads = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    groups = []
    if heads:
        groups.append({"params": heads, "lr": head_lr})
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def _resolve_amp_dtype(device: torch.device, mode: str) -> Optional[torch.dtype]:
    if mode == "off" or device.type != "cuda":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 AMP is not supported by this CUDA device")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if mode != "auto":
        raise ValueError("amp must be one of: auto, fp16, bf16, off")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype] = None):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _make_grad_scaler(device: torch.device, amp_dtype: Optional[torch.dtype]):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _dtype_name(dtype: Optional[torch.dtype]) -> str:
    return "off" if dtype is None else str(dtype).replace("torch.", "", 1)


def _optimizer_step(
    *,
    scaler: Any,
    optimizer: torch.optim.Optimizer,
    trainable_params: Sequence[torch.Tensor],
    max_grad_norm: float,
) -> None:
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()


def _denormalize(view: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), device=view.device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=view.device)[:, None, None]
    rgb = (view * std + mean).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return np.ascontiguousarray((rgb[:, :, ::-1] * 255).round().astype(np.uint8))


def dump_heatmaps(
    output_dir: Path,
    view_a: torch.Tensor,
    mask_batch: torch.Tensor,
    outputs: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Save a single original / target / prediction diagnostic image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    head_name = "learned" if "learned" in outputs else next(iter(outputs))
    tap_name = "f2" if "f2" in outputs[head_name] else next(iter(outputs[head_name]))
    prediction = outputs[head_name][tap_name][0:1].unsqueeze(1)
    heatmap = F.interpolate(
        prediction, size=mask_batch.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0].detach().float().cpu().numpy()
    heatmap = cv2.applyColorMap(
        np.clip(heatmap * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    target = (mask_batch[0].detach().cpu().numpy() > 0).astype(np.uint8) * 255
    target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    panel = np.hstack((_denormalize(view_a[0]), target, heatmap))
    if not cv2.imwrite(str(output_dir / "latest.jpg"), panel):
        raise OSError("Could not write heatmap diagnostic")


def _probe_prediction(outputs: dict[str, dict[str, torch.Tensor]]) -> Tuple[str, torch.Tensor]:
    head = outputs["learned"] if "learned" in outputs else next(iter(outputs.values()))
    tap_name = "f2" if "f2" in head else next(iter(head))
    return tap_name, head[tap_name]


def _exposure_stress(view: torch.Tensor, factor: float = 1.1) -> torch.Tensor:
    mean = view.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = view.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    rgb = (view * std + mean).clamp(0, 1)
    return ((rgb * factor).clamp(0, 1) - mean) / std


def _shift_stress(view: torch.Tensor, pixels: int = 2) -> torch.Tensor:
    if pixels <= 0:
        return view
    padded = F.pad(view, (pixels, 0, pixels, 0), mode="replicate")
    return padded[:, :, : view.shape[-2], : view.shape[-1]]


def _augmentation_probes(
    loader: Any,
    *,
    model: DifferenceModel,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
    max_samples: int = 2,
) -> dict[str, float]:
    """Measure robustness to nuisance changes on one deterministic validation batch."""

    try:
        view_a, view_b, mask_batch = next(iter(loader))
    except StopIteration:
        return {}
    view_a = view_a[:max_samples].to(device, non_blocking=True)
    view_b = view_b[:max_samples].to(device, non_blocking=True)
    mask_batch = mask_batch[:max_samples].to(device, non_blocking=True)
    if channels_last:
        view_a = view_a.contiguous(memory_format=torch.channels_last)
        view_b = view_b.contiguous(memory_format=torch.channels_last)
    was_training = model.training
    model.eval()
    with torch.no_grad(), _autocast_context(device, amp_dtype):
        clean_outputs = model(view_a, view_b)
        tap_name, clean = _probe_prediction(clean_outputs)
        _, identical = _probe_prediction(model(view_a, view_a))
        _, photometric = _probe_prediction(model(view_a, _exposure_stress(view_a)))
        _, shifted = _probe_prediction(model(view_a, _shift_stress(view_a)))
        _, swapped = _probe_prediction(model(view_b, view_a))
        target = build_targets(mask_batch, {tap_name: clean.shape[-2:]})[tap_name]
        predicted_positive = clean >= 0.5
        target_positive = target > 0.5
        intersection = (predicted_positive & target_positive).sum().float()
        union = (predicted_positive | target_positive).sum().float()
        positives = target_positive.sum().float()
        probes = {
            "identical_fp_mean": float(identical.float().mean()),
            "identical_fp_max": float(identical.float().max()),
            "photometric_stress_mean": float(photometric.float().mean()),
            "photometric_stress_max": float(photometric.float().max()),
            "shift_stress_mean": float(shifted.float().mean()),
            "shift_stress_max": float(shifted.float().max()),
            "swap_consistency_mean_abs": float((clean.float() - swapped.float()).abs().mean()),
            "clean_recall_05": float(intersection / positives.clamp_min(1.0)),
            "clean_iou_05": float(intersection / union.clamp_min(1.0)),
        }
    model.train(was_training)
    return probes


def _run_epoch(
    loader: Any,
    *,
    model: DifferenceModel,
    criterion: MultiLayerDiffLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: PosWeightScheduler,
    device: torch.device,
    max_grad_norm: float,
    scaler: Any = None,
    amp_dtype: Optional[torch.dtype] = None,
    accum_steps: int = 1,
    channels_last: bool = False,
    trainable_params: Optional[Sequence[torch.Tensor]] = None,
    heatmap_dir: Optional[Path] = None,
    max_batches: Optional[int] = None,
    phase: str = "train",
    progress: bool = True,
    log_every: int = 1,
) -> Tuple[float, dict[str, float]]:
    """Run one epoch where every loader item is an already-stacked tensor batch."""

    if accum_steps <= 0:
        raise ValueError("accum_steps must be positive")
    training = optimizer is not None
    if scaler is None:
        scaler = _make_grad_scaler(device, amp_dtype)
    if training and trainable_params is None:
        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.train(training)
    total_loss = 0.0
    total_pairs = 0
    component_totals: dict[str, float] = {}
    context = torch.enable_grad() if training else torch.no_grad()
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    batches = tqdm(
        enumerate(loader),
        total=total_batches,
        desc=phase.capitalize(),
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not progress,
    )
    with context:
        pending_pairs = 0
        for batch_index, (view_a, view_b, mask_batch) in batches:
            if training and batch_index % accum_steps == 0:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            if channels_last:
                view_a = view_a.contiguous(memory_format=torch.channels_last)
                view_b = view_b.contiguous(memory_format=torch.channels_last)
            batch_size = int(view_a.shape[0])
            with _autocast_context(device, amp_dtype):
                outputs = model(view_a, view_b)
                targets = build_targets(mask_batch, tap_sizes(outputs))
                loss, components = criterion(outputs, targets, alpha=scheduler.alpha)
            if training:
                assert optimizer is not None and trainable_params is not None
                if total_batches is None:
                    group_batches = accum_steps
                else:
                    group_start = (batch_index // accum_steps) * accum_steps
                    group_batches = min(accum_steps, total_batches - group_start)
                scaler.scale(loss / group_batches).backward()
                pending_pairs += batch_size
                group_finished = (batch_index + 1) % accum_steps == 0
                epoch_finished = total_batches is not None and batch_index + 1 >= total_batches
                if group_finished or epoch_finished:
                    _optimizer_step(
                        scaler=scaler,
                        optimizer=optimizer,
                        trainable_params=trainable_params,
                        max_grad_norm=max_grad_norm,
                    )
                    scheduler.step(pending_pairs)
                    pending_pairs = 0
            if heatmap_dir is not None:
                dump_heatmaps(heatmap_dir, view_a, mask_batch, outputs)
                heatmap_dir = None
            total_loss += float(loss.detach()) * batch_size
            total_pairs += batch_size
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value) * batch_size
            if progress:
                batches.set_postfix(
                    loss="{:.4f}".format(total_loss / max(total_pairs, 1)),
                    pairs=total_pairs,
                    alpha="{:.3f}".format(scheduler.alpha),
                )
            if (batch_index + 1) % log_every == 0 or batch_index == 0:
                LOGGER.debug(
                    "%s batch=%d components=%s",
                    phase,
                    batch_index + 1,
                    {
                        name: round(value / max(total_pairs, 1), 6)
                        for name, value in component_totals.items()
                    },
                )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
        if training and pending_pairs:
            # This only occurs for an iterable without ``__len__``.
            assert optimizer is not None and trainable_params is not None
            _optimizer_step(
                scaler=scaler,
                optimizer=optimizer,
                trainable_params=trainable_params,
                max_grad_norm=max_grad_norm,
            )
            scheduler.step(pending_pairs)
    batches.close()
    if not total_pairs:
        raise ValueError("loader did not yield any batches")
    return (
        total_loss / total_pairs,
        {name: value / total_pairs for name, value in component_totals.items()},
    )


def _run_full_resolution_epoch(
    loader: PrecomputedPairDataLoader,
    *,
    model: DifferenceModel,
    criterion: MultiLayerDiffLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: PosWeightScheduler,
    device: torch.device,
    max_grad_norm: float,
    scaler: Any,
    amp_dtype: Optional[torch.dtype],
    trainable_params: Sequence[torch.Tensor],
    heatmap_dir: Optional[Path] = None,
    max_batches: Optional[int] = None,
    phase: str = "train",
    progress: bool = True,
    log_every: int = 1,
) -> Tuple[float, dict[str, float]]:
    """Compatibility path for differently-sized full-resolution NumPy images."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pairs = 0
    component_totals: dict[str, float] = {}
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    batches = tqdm(
        enumerate(loader), total=total_batches, desc=phase.capitalize(), unit="batch",
        dynamic_ncols=True, leave=False, disable=not progress,
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (originals, erased, masks) in batches:
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(originals)
            for original, erased_image, mask in zip(originals, erased, masks):
                view_a, view_b, mask_batch = prepare_pair_batch(
                    [original], [erased_image], [mask], device=device
                )
                with _autocast_context(device, amp_dtype):
                    outputs = model(view_a, view_b)
                    targets = build_targets(mask_batch, tap_sizes(outputs))
                    loss, components = criterion(outputs, targets, alpha=scheduler.alpha)
                if training:
                    scaler.scale(loss / batch_size).backward()
                if heatmap_dir is not None:
                    dump_heatmaps(heatmap_dir, view_a, mask_batch, outputs)
                    heatmap_dir = None
                total_loss += float(loss.detach())
                total_pairs += 1
                for name, value in components.items():
                    component_totals[name] = component_totals.get(name, 0.0) + float(value)
            if training:
                assert optimizer is not None
                _optimizer_step(
                    scaler=scaler,
                    optimizer=optimizer,
                    trainable_params=trainable_params,
                    max_grad_norm=max_grad_norm,
                )
                scheduler.step(batch_size)
            if (batch_index + 1) % log_every == 0 or batch_index == 0:
                LOGGER.debug("%s full-resolution batch=%d", phase, batch_index + 1)
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    batches.close()
    if not total_pairs:
        raise ValueError("loader did not yield any batches")
    return (
        total_loss / total_pairs,
        {name: value / total_pairs for name, value in component_totals.items()},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the detail-removal difference model.")
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="dataset root containing manifest.jsonl (including pairwise outputs)",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="true tensor batch size when cropping is enabled",
    )
    parser.add_argument(
        "--crop-size", type=int, default=512,
        help="square training crop size; 0 uses the legacy full-resolution path",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crops-per-sample", type=int, default=1)
    parser.add_argument("--positive-crop-probability", type=float, default=0.7)
    parser.add_argument(
        "--augmentation-ramp-fraction", type=float, default=0.3,
        help="fraction of epochs used to ramp independent photometric/shift strength",
    )
    parser.add_argument("--scale-jitter-min", type=float, default=0.7)
    parser.add_argument("--scale-jitter-max", type=float, default=1.4)
    parser.add_argument("--identity-pair-probability", type=float, default=0.08)
    parser.add_argument("--cutout-probability", type=float, default=0.15)
    parser.add_argument("--copy-paste-probability", type=float, default=0.10)
    parser.add_argument("--shift-probability", type=float, default=0.3)
    parser.add_argument("--independent-shift-pixels", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--amp", choices=("auto", "fp16", "bf16", "off"), default="auto")
    parser.add_argument("--refresh-stats", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--output-stride", type=int, choices=(4, 8), default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--disable-augment", action="store_true")
    parser.add_argument(
        "--rotate-probability",
        type=float,
        default=1.0,
        help="probability of applying a random 0/90/180/270-degree turn to train pairs",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity for the console and training log",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="log file path (default: <out>/training.log)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="write a batch log every N batches (default: 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    return parser


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(0)


def _crop_loader(
    dataset: PairCropDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "worker_init_fn": _worker_init if workers > 0 else None,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _configure_logging(out_dir: Path, level: str, log_file: Optional[Path]) -> Path:
    """Configure console and file logging and return the selected log path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = log_file or (out_dir / "training.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    return path


def _load_checkpoint(path: Path, device: torch.device):
    """Load a training checkpoint across PyTorch versions."""
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        # ``weights_only`` was added after the versions supported by this repo.
        return torch.load(str(path), map_location=device)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch <= 0 or args.crops_per_sample <= 0:
        raise ValueError("epochs, batch, and crops-per-sample must be positive")
    if args.crop_size < 0 or args.workers < 0 or args.accum_steps <= 0:
        raise ValueError("crop-size/workers must be nonnegative and accum-steps positive")
    if not 0.0 <= args.positive_crop_probability <= 1.0:
        raise ValueError("positive-crop-probability must be in [0, 1]")
    probability_args = {
        "augmentation-ramp-fraction": args.augmentation_ramp_fraction,
        "identity-pair-probability": args.identity_pair_probability,
        "cutout-probability": args.cutout_probability,
        "copy-paste-probability": args.copy_paste_probability,
        "shift-probability": args.shift_probability,
    }
    if any(not 0.0 <= value <= 1.0 for value in probability_args.values()):
        raise ValueError("augmentation probabilities/fractions must be in [0, 1]")
    if args.scale_jitter_min <= 0 or args.scale_jitter_max < args.scale_jitter_min:
        raise ValueError("scale jitter range must be positive and ordered")
    if args.independent_shift_pixels < 0:
        raise ValueError("independent-shift-pixels must be nonnegative")
    if (
        args.max_train_batches is not None
        and args.max_train_batches <= 0
    ) or (
        args.max_validation_batches is not None
        and args.max_validation_batches <= 0
    ):
        raise ValueError("maximum batch counts must be positive")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if not 0.0 <= args.rotate_probability <= 1.0:
        raise ValueError("rotate-probability must be in [0, 1]")
    training_log_path = _configure_logging(args.out, args.log_level, args.log_file)
    LOGGER.info("Starting training; log_file=%s", training_log_path)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = _resolve_amp_dtype(device, args.amp)
    scaler = _make_grad_scaler(device, amp_dtype)
    if device.type == "cuda" and args.crop_size > 0:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    pair_augment_config = PairAugmentConfig(
        enabled=not args.disable_augment,
        rotation_probability=args.rotate_probability,
        scale_min=args.scale_jitter_min,
        scale_max=args.scale_jitter_max,
        identity_probability=args.identity_pair_probability,
        cutout_probability=args.cutout_probability,
        copy_paste_probability=args.copy_paste_probability,
        shift_probability=args.shift_probability,
        max_shift_pixels=args.independent_shift_pixels,
    )
    curriculum_ramp_epochs = max(
        1, int(math.ceil(args.epochs * args.augmentation_ramp_fraction))
    )
    augmentation_config = pair_augment_config.to_dict()
    augmentation_config["curriculum_ramp_epochs"] = curriculum_ramp_epochs
    legacy_augment = None
    if args.crop_size == 0 and not args.disable_augment:
        legacy_augment = SynchronizedPhotometricRotate90Augment(
            seed=args.seed, rotation_probability=args.rotate_probability
        )
    LOGGER.info(
        "augmentation train=%s validation=disabled config=%s",
        "PairAugmentConfig" if pair_augment_config.enabled and args.crop_size > 0
        else type(legacy_augment).__name__ if legacy_augment is not None else "disabled",
        augmentation_config,
    )
    full_train_dataset = PrecomputedPairDataset(args.root, paired_transform=legacy_augment)
    raw_dataset = PrecomputedPairDataset(args.root)
    train_indices, validation_indices = split_indices(
        len(raw_dataset), args.validation_fraction, args.seed
    )
    if args.crop_size > 0:
        train_dataset = PairCropDataset(
            full_train_dataset,
            train_indices,
            crop_size=args.crop_size,
            seed=args.seed,
            crops_per_sample=args.crops_per_sample,
            biased_probability=args.positive_crop_probability,
            train=True,
            augment_config=pair_augment_config,
            curriculum_ramp_epochs=curriculum_ramp_epochs,
        )
        validation_dataset = PairCropDataset(
            raw_dataset,
            validation_indices,
            crop_size=args.crop_size,
            seed=args.seed,
            crops_per_sample=1,
            biased_probability=0.0,
            train=False,
            augment_config=PairAugmentConfig(enabled=False),
        )
        train_loader = _crop_loader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            drop_last=len(train_dataset) >= args.batch,
            workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        validation_loader = _crop_loader(
            validation_dataset,
            batch_size=args.batch,
            shuffle=False,
            drop_last=False,
            workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        probe_dataset = PairCropDataset(
            raw_dataset,
            validation_indices[: max(5, args.batch)],
            crop_size=args.crop_size,
            seed=args.seed + 10_000,
            crops_per_sample=1,
            biased_probability=1.0,
            train=True,
            augment_config=PairAugmentConfig(enabled=False),
        )
        probe_loader = _crop_loader(
            probe_dataset,
            batch_size=min(args.batch, len(probe_dataset)),
            shuffle=False,
            drop_last=False,
            workers=0,
            pin_memory=device.type == "cuda",
        )
        run_epoch = _run_epoch
    else:
        train_dataset = IndexDataset(full_train_dataset, train_indices)
        validation_dataset = IndexDataset(raw_dataset, validation_indices)
        train_loader = PrecomputedPairDataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, drop_last=True, seed=args.seed
        )
        validation_loader = PrecomputedPairDataLoader(
            validation_dataset, batch_size=args.batch, shuffle=False, drop_last=False
        )
        probe_loader = None
        run_epoch = _run_full_resolution_epoch

    weights = None if args.no_pretrained or args.weights_path else ResNet101_Weights.IMAGENET1K_V2
    backbone = TruncatedResNet101(
        weights=weights,
        weights_path=args.weights_path,
        output_stride=args.output_stride,
        trainable_stages=("layer2",),
    )
    model = DifferenceModel(backbone, learned=True, cosine=True).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = MultiLayerDiffLoss(
        tap_weights={"f0": 0.5, "f1": 1.0, "f2": 1.0},
        head_weights={"learned": 1.0, "cosine": 0.2},
    ).to(device)
    parameter_groups = _trainable_parameters(model, args.lr, args.lr / 10.0)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    trainable_params = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    positive_fractions = _cached_positive_fractions(
        args.root,
        raw_dataset,
        train_indices,
        crop_size=args.crop_size,
        seed=args.seed,
        biased_probability=args.positive_crop_probability if args.crop_size > 0 else 0.0,
        refresh=args.refresh_stats,
        augment_config=pair_augment_config if args.crop_size > 0 else None,
        curriculum_ramp_epochs=curriculum_ramp_epochs,
    )
    alpha_max = estimate_alpha_max(positive_fractions)
    hold_pairs = max(args.batch, len(train_dataset) * 2)
    ramp_pairs = max(args.batch, len(train_dataset) * 6)
    scheduler = PosWeightScheduler("linear", alpha_max, hold_pairs, ramp_pairs)
    start_epoch = 0
    if args.resume is not None:
        checkpoint = _load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        criterion.load_state_dict(checkpoint["criterion"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    LOGGER.info(
        "device=%s train_samples=%d validation_samples=%d epochs=%d batch=%d crop=%d amp=%s workers=%d",
        device,
        len(train_dataset),
        len(validation_dataset),
        args.epochs,
        args.batch,
        args.crop_size,
        _dtype_name(amp_dtype),
        args.workers if args.crop_size > 0 else 0,
    )
    heatmap_dir = args.out / "heatmaps"
    metrics_path = args.out / "metrics.jsonl"
    checkpoints_dir = args.out / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "train_sample_ids": [raw_dataset.metadata(index)["sample_id"] for index in train_indices],
        "validation_sample_ids": [raw_dataset.metadata(index)["sample_id"] for index in validation_indices],
    }
    with (args.out / "split.json").open("w", encoding="utf-8") as stream:
        json.dump(split, stream, indent=2)

    with metrics_path.open("a", encoding="utf-8") as log_stream:
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.perf_counter()
            LOGGER.info("Epoch %d/%d started", epoch + 1, args.epochs)
            if args.crop_size > 0:
                train_dataset.set_epoch(epoch)
            common_epoch_args = {
                "model": model,
                "criterion": criterion,
                "scheduler": scheduler,
                "device": device,
                "max_grad_norm": args.max_grad_norm,
                "scaler": scaler,
                "amp_dtype": amp_dtype,
                "trainable_params": trainable_params,
            }
            train_extra_args = {}
            validation_extra_args = {}
            if args.crop_size > 0:
                train_extra_args = {
                    "accum_steps": args.accum_steps,
                    "channels_last": args.channels_last,
                }
                validation_extra_args = {"channels_last": args.channels_last}
            train_loss, train_components = run_epoch(
                train_loader,
                optimizer=optimizer,
                heatmap_dir=heatmap_dir,
                max_batches=args.max_train_batches,
                phase="train",
                progress=not args.no_progress,
                log_every=args.log_every,
                **common_epoch_args,
                **train_extra_args,
            )
            validation_loss, validation_components = run_epoch(
                validation_loader,
                optimizer=None,
                max_batches=args.max_validation_batches,
                phase="validation",
                progress=not args.no_progress,
                log_every=args.log_every,
                **common_epoch_args,
                **validation_extra_args,
            )
            probes = {}
            if args.crop_size > 0:
                probes = _augmentation_probes(
                    probe_loader,
                    model=model,
                    device=device,
                    amp_dtype=amp_dtype,
                    channels_last=args.channels_last,
                )
            scheduler.on_validation(validation_loss)
            mitigated = criterion.mitigate_exploding_taps(
                validation_components,
                ratio=3.0,
                min_loss=1e-3,
                decay=0.5,
                min_weight=0.05,
            )
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "alpha": scheduler.alpha,
                "alpha_phase": scheduler.phase,
                "alpha_max": alpha_max,
                "crop_size": args.crop_size,
                "batch": args.batch,
                "accum_steps": args.accum_steps,
                "amp": args.amp,
                "amp_dtype": _dtype_name(amp_dtype),
                "pixel_subtraction_reference": 1.0,
                "train_components": train_components,
                "validation_components": validation_components,
                "mitigated_taps": mitigated,
                "augmentation_probes": probes,
                "augmentation_strength": (
                    train_dataset.curriculum_strength if args.crop_size > 0 else 0.0
                ),
                "checkpoint": str(Path("checkpoints") / "epoch_{:04d}.pt".format(epoch + 1)),
                "augmentation": augmentation_config,
            }
            log_stream.write(json.dumps(metrics) + "\n")
            log_stream.flush()
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "criterion": criterion.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "split": split,
                "args": vars(args),
            }
            epoch_checkpoint_path = checkpoints_dir / "epoch_{:04d}.pt".format(epoch + 1)
            torch.save(checkpoint, epoch_checkpoint_path)
            torch.save(checkpoint, args.out / "latest.pt")
            LOGGER.info(
                "Epoch %d/%d complete: train_loss=%.6f validation_loss=%.6f alpha=%.4f elapsed=%.2fs checkpoint=%s",
                epoch + 1,
                args.epochs,
                train_loss,
                validation_loss,
                scheduler.alpha,
                time.perf_counter() - epoch_started,
                epoch_checkpoint_path,
            )
            print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
