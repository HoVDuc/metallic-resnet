from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import median

import torch
from torch import nn


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Mean squared error weighted by the soft positive target value."""
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or alpha_value <= 0:
        if not math.isfinite(alpha_value):
            raise ValueError(f"alpha must be finite and positive, got {alpha}")
        raise ValueError(f"alpha must be positive, got {alpha}")
    weights = 1.0 + (alpha_value - 1.0) * target
    return (weights * (prediction - target).square()).sum() / weights.sum()


def _align_single_channel_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape == target.shape:
        return prediction
    if prediction.ndim == target.ndim + 1 and prediction.shape[1] == 1:
        squeezed = prediction.squeeze(1)
        if squeezed.shape == target.shape:
            return squeezed
    return prediction


class MultiLayerDiffLoss(nn.Module):
    """Combine weighted difference-map losses across model heads and taps."""

    def __init__(
        self,
        tap_weights: Mapping[str, float],
        head_weights: Mapping[str, float],
    ) -> None:
        super().__init__()
        self.tap_weights = {str(name): float(weight) for name, weight in tap_weights.items()}
        self.head_weights = {str(name): float(weight) for name, weight in head_weights.items()}
        self._validate_weights()

    def _validate_weights(self) -> None:
        if any(not math.isfinite(weight) or weight < 0 for weight in self.tap_weights.values()):
            raise ValueError("tap weights must be finite and nonnegative")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.head_weights.values()):
            raise ValueError("head weights must be finite and nonnegative")

    def get_extra_state(self) -> dict[str, dict[str, float]]:
        """Persist mutable loss weights alongside ordinary module state."""
        return {
            "tap_weights": dict(self.tap_weights),
            "head_weights": dict(self.head_weights),
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("loss extra state must be a mapping")
        tap_weights = state.get("tap_weights")
        head_weights = state.get("head_weights")
        if not isinstance(tap_weights, Mapping) or not isinstance(head_weights, Mapping):
            raise ValueError("loss extra state must contain tap_weights and head_weights mappings")
        self.tap_weights = {str(name): float(weight) for name, weight in tap_weights.items()}
        self.head_weights = {str(name): float(weight) for name, weight in head_weights.items()}
        self._validate_weights()

    def mitigate_exploding_taps(
        self,
        components: Mapping[str, torch.Tensor | float],
        *,
        ratio: float,
        min_loss: float,
        decay: float,
        min_weight: float,
    ) -> dict[str, tuple[float, float]]:
        """Decay only the dominant tap when its loss is extreme relative to peers."""
        ratio = float(ratio)
        min_loss = float(min_loss)
        decay = float(decay)
        min_weight = float(min_weight)
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise ValueError("tap mitigation ratio must be finite and greater than one")
        if not math.isfinite(min_loss) or min_loss < 0.0:
            raise ValueError("tap mitigation min_loss must be finite and nonnegative")
        if not math.isfinite(decay) or not 0.0 < decay < 1.0:
            raise ValueError("tap mitigation decay must be finite and in (0, 1)")
        if not math.isfinite(min_weight) or min_weight < 0.0:
            raise ValueError("tap mitigation min_weight must be finite and nonnegative")

        per_tap: dict[str, list[float]] = {}
        for component_name, raw_value in components.items():
            _, separator, tap_name = str(component_name).partition("/")
            if not separator or tap_name not in self.tap_weights:
                continue
            value = float(raw_value.detach()) if isinstance(raw_value, torch.Tensor) else float(raw_value)
            if math.isfinite(value) and value >= 0.0:
                per_tap.setdefault(tap_name, []).append(value)
        losses = {
            tap_name: sum(values) / len(values)
            for tap_name, values in per_tap.items()
            if values
        }
        if len(losses) < 2:
            return {}

        candidates: list[tuple[float, str]] = []
        for tap_name, loss in losses.items():
            peers = [peer_loss for peer_name, peer_loss in losses.items() if peer_name != tap_name]
            threshold = max(min_loss, ratio * median(peers))
            if loss > threshold and self.tap_weights[tap_name] > min_weight:
                candidates.append((loss / max(threshold, 1e-12), tap_name))
        if not candidates:
            return {}

        _, tap_name = max(candidates)
        old_weight = self.tap_weights[tap_name]
        new_weight = max(min_weight, old_weight * decay)
        if new_weight >= old_weight:
            return {}
        self.tap_weights[tap_name] = new_weight
        return {tap_name: (old_weight, new_weight)}

    def forward(
        self,
        predictions: Mapping[str, Mapping[str, torch.Tensor]],
        targets: Mapping[str, torch.Tensor],
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total: torch.Tensor | None = None
        components: dict[str, torch.Tensor] = {}
        for head_name, head_weight in self.head_weights.items():
            head_predictions = predictions.get(head_name)
            if head_predictions is None:
                continue
            for tap_name, tap_weight in self.tap_weights.items():
                prediction = head_predictions.get(tap_name)
                target = targets.get(tap_name)
                if prediction is None or target is None:
                    continue
                loss = weighted_mse(
                    _align_single_channel_prediction(prediction, target), target, alpha
                )
                components[f"{head_name}/{tap_name}"] = loss.detach()
                contribution = loss * head_weight * tap_weight
                total = contribution if total is None else total + contribution
        if total is None:
            reference = next(iter(targets.values()), None)
            if reference is None:
                raise ValueError("targets must contain at least one tap")
            total = reference.new_zeros(())
        return total, components
