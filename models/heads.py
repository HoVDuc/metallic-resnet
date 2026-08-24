from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def channel_standardize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize each spatial location across its channels without affine terms."""
    mean = features.mean(dim=1, keepdim=True)
    variance = features.var(dim=1, keepdim=True, unbiased=False)
    return (features - mean) * torch.rsqrt(variance + eps)


class CosineHead(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        standardized_a = channel_standardize(feature_a, self.eps)
        standardized_b = channel_standardize(feature_b, self.eps)
        norm_a = standardized_a.norm(dim=1, keepdim=True)
        norm_b = standardized_b.norm(dim=1, keepdim=True)
        normalized_a = F.normalize(standardized_a, dim=1, eps=self.eps)
        normalized_b = F.normalize(standardized_b, dim=1, eps=self.eps)
        similarity = (normalized_a * normalized_b).sum(dim=1)
        both_zero = (norm_a <= self.eps) & (norm_b <= self.eps)
        similarity = torch.where(both_zero.squeeze(1), 1.0, similarity)
        return ((1.0 - similarity) * 0.5).clamp_(0.0, 1.0)


def _group_count(channels: int) -> int:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("positive channel count always has a divisor")


class _ConvGroupNormReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class SymConvHead(nn.Module):
    """Learned head using only permutation-invariant pair features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        hidden_1 = max(1, channels // 2)
        hidden_2 = max(1, channels // 4)
        hidden_3 = max(1, channels // 8)
        self.blocks = nn.Sequential(
            _ConvGroupNormReLU(3 * channels, hidden_1, kernel_size=1),
            _ConvGroupNormReLU(hidden_1, hidden_2, kernel_size=3),
            _ConvGroupNormReLU(hidden_2, hidden_3, kernel_size=3),
        )
        self.output = nn.Conv2d(hidden_3, 1, kernel_size=1)

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        symmetric = torch.cat(
            (
                (feature_a - feature_b).abs(),
                feature_a, feature_b,
            ),
            dim=1,
        )
        return torch.sigmoid(self.output(self.blocks(symmetric))).squeeze(1)
