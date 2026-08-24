from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .heads import CosineHead, SymConvHead


def _tap_channels(backbone: nn.Module) -> dict[str, int]:
    channels: object | None = getattr(backbone, "tap_channels", None)
    if channels is None:
        channels = getattr(backbone, "channels", None)
    if isinstance(channels, Mapping):
        return {str(name): int(value) for name, value in channels.items()}
    if isinstance(channels, Sequence):
        values = [int(value) for value in channels]
        if len(values) == 3:
            return dict(zip(("f0", "f1", "f2"), values))
    raise ValueError("a learned DifferenceModel backbone must declare tap_channels")


def tap_sizes(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, tuple[int, int]]:
    """Return spatial sizes for every tap, validated across enabled heads."""

    sizes: dict[str, tuple[int, int]] = {}
    for head_name, head_outputs in outputs.items():
        for tap_name, prediction in head_outputs.items():
            if prediction.ndim < 3:
                raise ValueError(
                    f"prediction for {head_name}/{tap_name} must have spatial dimensions"
                )
            size = (int(prediction.shape[-2]), int(prediction.shape[-1]))
            previous = sizes.setdefault(tap_name, size)
            if previous != size:
                raise ValueError(
                    f"tap {tap_name!r} has inconsistent sizes {previous} and {size}"
                )
    if not sizes:
        raise ValueError("model outputs must contain at least one tap")
    return sizes


class DifferenceModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        learned: bool = True,
        *,
        cosine: bool = True,
    ) -> None:
        super().__init__()
        if not learned and not cosine:
            raise ValueError("at least one difference head family must be enabled")
        self.backbone = backbone
        self.cosine_head = CosineHead() if cosine else None
        self.learned_heads = (
            nn.ModuleDict(
                {
                    name: SymConvHead(channels)
                    for name, channels in _tap_channels(backbone).items()
                }
            )
            if learned
            else None
        )

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        batch_size = view_a.shape[0]
        taps = self.backbone(torch.cat((view_a, view_b), dim=0))
        split_taps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, features in taps.items():
            if features.shape[0] != 2 * batch_size:
                raise ValueError(
                    f"backbone tap {name!r} returned batch {features.shape[0]}, "
                    f"expected {2 * batch_size}"
                )
            feature_a, feature_b = features.split(batch_size, dim=0)
            split_taps[name] = (feature_a, feature_b)

        outputs: dict[str, dict[str, torch.Tensor]] = {}
        if self.cosine_head is not None:
            outputs["cosine"] = {
                name: self.cosine_head(feature_a, feature_b)
                for name, (feature_a, feature_b) in split_taps.items()
            }
        if self.learned_heads is not None:
            outputs["learned"] = {
                name: self.learned_heads[name](feature_a, feature_b)
                for name, (feature_a, feature_b) in split_taps.items()
            }
        return outputs
