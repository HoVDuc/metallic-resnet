"""Run a trained difference model on two aligned, full-resolution images."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path, PosixPath
from typing import Dict, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from models import DifferenceModel, TruncatedResNet101, prepare_pair_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a cosine difference heatmap for two aligned images."
    )
    parser.add_argument("image_a", type=Path, help="path to the first image")
    parser.add_argument("image_b", type=Path, help="path to the second image")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--head",
        choices=("cosine", "learned"),
        default="cosine",
        help="difference head to use; cosine is the head available in the cosine-only checkpoint",
    )
    parser.add_argument("--tap", choices=("f0", "f1", "f2"), default="f2")
    parser.add_argument(
        "--output-stride",
        type=int,
        choices=(4, 8),
        help="override output stride when loading a raw model state dict",
    )
    parser.add_argument("--device", help="for example: cuda, cuda:0, or cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-opacity", type=float, default=0.65)
    parser.add_argument("--preview-width", type=int, default=2000)
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA autocast")
    return parser


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    return image


def _checkpoint_state(
    checkpoint_path: Path, device: torch.device
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        # PyTorch before 2.6 did not default to the restricted weights-only loader.
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    else:
        # train.py stores pathlib paths inside the otherwise weights-only-safe
        # argument metadata. Python 3.13 serializes PosixPath as
        # ``pathlib._local.PosixPath``; allow both spellings explicitly.
        with safe_globals([
            PosixPath,
            (PosixPath, "pathlib._local.PosixPath"),
        ]):
            checkpoint = torch.load(
                str(checkpoint_path), map_location=device, weights_only=True
            )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a state-dict mapping")
    if "model" in checkpoint:
        state = checkpoint["model"]
        saved_args = checkpoint.get("args", {})
    else:
        state = checkpoint
        saved_args = {}
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint['model'] must be a state-dict mapping")
    if not isinstance(saved_args, Mapping):
        saved_args = {}
    return state, saved_args


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    output_stride: Optional[int] = None,
) -> DifferenceModel:
    """Rebuild and load the cosine-only model used by the supplied checkpoint.

    The checkpoints in ``/home/anlab/Desktop/checkpoints`` contain only the
    backbone parameters.  CosineHead has no trainable parameters, so there are
    deliberately no head weights to load here.
    """

    state, saved_args = _checkpoint_state(checkpoint_path, device)
    saved_stride = int(saved_args.get("output_stride", 8))
    active_stride = output_stride if output_stride is not None else saved_stride
    has_learned_head = any(
        str(key).startswith("learned_heads.") for key in state
    )
    if has_learned_head:
        raise ValueError(
            "This inference loader is for cosine-only checkpoints, but the "
            "checkpoint also contains learned head weights"
        )
    backbone = TruncatedResNet101(
        weights=None,
        output_stride=active_stride,
        trainable_stages=(),
    )
    model = DifferenceModel(backbone, learned=False, cosine=True).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def predict_difference(
    model: DifferenceModel,
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    head: str = "cosine",
    tap: str,
    device: torch.device,
    amp: bool = True,
) -> np.ndarray:
    """Return a full-resolution float32 difference score map in ``[0, 1]``."""

    if image_a.shape != image_b.shape:
        raise ValueError(
            "Input images must have the same shape, got {} and {}".format(
                image_a.shape, image_b.shape
            )
        )
    if image_a.ndim != 3 or image_a.shape[2] != 3:
        raise ValueError("Input images must be three-channel BGR images")
    empty_mask = np.zeros(image_a.shape[:2], dtype=np.uint8)
    view_a, view_b, _ = prepare_pair_batch(
        [image_a], [image_b], [empty_mask], device=device
    )
    with torch.inference_mode(), _autocast_context(device, amp):
        outputs = model(view_a, view_b)
        if head not in outputs:
            raise ValueError("Model does not provide head {!r}".format(head))
        if tap not in outputs[head]:
            raise ValueError("Model does not provide tap {!r} for head {!r}".format(tap, head))
        prediction = outputs[head][tap][0:1].unsqueeze(1)
        prediction = F.interpolate(
            prediction,
            size=image_a.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return prediction.float().clamp(0.0, 1.0).cpu().numpy()


def infer_pair(
    image_a_path: Union[Path, str],
    image_b_path: Union[Path, str],
    checkpoint_path: Union[Path, str],
    *,
    tap: str = "f2",
    output_stride: Optional[int] = None,
    device_name: Optional[str] = None,
    threshold: float = 0.5,
    overlay_opacity: float = 0.65,
    amp: bool = True,
) -> Dict[str, np.ndarray]:
    """Run cosine-head inference for one aligned image pair.

    Returns the raw float32 score map in ``raw_score`` plus the rendered
    ``score``, ``heatmap``, ``mask``, ``overlay`` and ``preview`` arrays.  The
    two input images must have the same spatial dimensions.
    """

    image_a_path = Path(image_a_path)
    image_b_path = Path(image_b_path)
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    image_a = _read_image(image_a_path)
    image_b = _read_image(image_b_path)
    model = load_model(checkpoint_path, device, output_stride)
    raw_score = predict_difference(
        model,
        image_a,
        image_b,
        head="cosine",
        tap=tap,
        device=device,
        amp=amp,
    )
    rendered = render_outputs(
        image_a,
        image_b,
        raw_score,
        threshold=threshold,
        overlay_opacity=overlay_opacity,
    )
    return {"raw_score": raw_score, **rendered}


def render_outputs(
    image_a: np.ndarray,
    image_b: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
    overlay_opacity: float,
) -> dict[str, np.ndarray]:
    """Build grayscale, color, binary-mask, overlay, and preview images."""

    score_u8 = np.round(score * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    mask = (score >= threshold).astype(np.uint8) * 255
    weight = (score * overlay_opacity)[:, :, None]
    overlay = np.round(image_a * (1.0 - weight) + heatmap * weight).astype(np.uint8)
    return {
        "score": score_u8,
        "heatmap": heatmap,
        "mask": mask,
        "overlay": overlay,
        "preview": _build_preview(image_a, image_b, heatmap, overlay),
    }


def _build_preview(*images: np.ndarray) -> np.ndarray:
    titles = ("Image A", "Image B", "Heatmap", "Overlay")
    panels = []
    for image, title in zip(images, titles):
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
        panels.append(np.vstack((header, image)))
    return np.hstack(panels)


def _resize_preview(preview: np.ndarray, max_width: int) -> np.ndarray:
    if preview.shape[1] <= max_width:
        return preview
    scale = max_width / float(preview.shape[1])
    size = (max_width, max(1, int(round(preview.shape[0] * scale))))
    return cv2.resize(preview, size, interpolation=cv2.INTER_AREA)


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not write image: {}".format(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.overlay_opacity <= 1.0:
        parser.error("--overlay-opacity must be in [0, 1]")
    if args.preview_width <= 0:
        parser.error("--preview-width must be positive")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_a = _read_image(args.image_a)
    image_b = _read_image(args.image_b)
    model = load_model(args.checkpoint, device, args.output_stride)
    score = predict_difference(
        model,
        image_a,
        image_b,
        head=args.head,
        tap=args.tap,
        device=device,
        amp=not args.no_amp,
    )
    rendered = render_outputs(
        image_a,
        image_b,
        score,
        threshold=args.threshold,
        overlay_opacity=args.overlay_opacity,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out / "score.npy"), score)
    _write_image(args.out / "score.png", rendered["score"])
    _write_image(args.out / "heatmap.png", rendered["heatmap"])
    _write_image(args.out / "mask.png", rendered["mask"])
    _write_image(args.out / "overlay.png", rendered["overlay"])
    preview = _resize_preview(rendered["preview"], args.preview_width)
    _write_image(args.out / "preview.jpg", preview)
    print("Wrote inference outputs to {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
