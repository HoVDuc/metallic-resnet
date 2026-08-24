from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch


def _require_finite_positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return number


def _require_finite_nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return number


def estimate_alpha_max(positive_fractions: Iterable[float] | torch.Tensor) -> float:
    """Estimate a positive weighting cap from crops which contain positives."""
    if isinstance(positive_fractions, torch.Tensor):
        fractions = positive_fractions.detach().reshape(-1).tolist()
    else:
        fractions = [float(value) for value in positive_fractions]
    positive = [value for value in fractions if value > 0.0]
    if not positive:
        return 1.0
    if any(value > 1.0 for value in positive):
        raise ValueError("positive fractions must be in the interval [0, 1]")
    ratios = sorted((1.0 - value) / value for value in positive)
    index = 0.9 * (len(ratios) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    percentile_90 = ratios[lower] + (ratios[upper] - ratios[lower]) * (index - lower)
    return min(10.0, percentile_90)


class PosWeightScheduler:
    """Pair-counted cosine warm-up for the soft-positive MSE weight."""

    def __init__(
        self,
        mode: str,
        alpha_max: float,
        hold_pairs: int = 32_000,
        ramp_pairs: int = 96_000,
        plateau_patience: int = 3,
        plateau_eps: float = 1e-4,
    ) -> None:
        if mode not in {"linear", "plateau"}:
            raise ValueError(f"mode must be 'linear' or 'plateau', got {mode!r}")
        alpha_max = _require_finite_positive("alpha_max", alpha_max)
        if hold_pairs < 0 or ramp_pairs <= 0:
            raise ValueError("hold_pairs must be nonnegative and ramp_pairs must be positive")
        if plateau_patience < 1:
            raise ValueError("plateau_patience must be at least one")
        plateau_eps = _require_finite_nonnegative("plateau_eps", plateau_eps)
        self.mode = mode
        self.alpha_max = alpha_max
        self.hold_pairs = int(hold_pairs)
        self.ramp_pairs = int(ramp_pairs)
        self.plateau_patience = int(plateau_patience)
        self.plateau_eps = plateau_eps
        self.total_pairs = 0
        self.ramp_start_pairs: int | None = None
        self.ema_validation_loss: float | None = None
        self.best_ema_validation_loss: float | None = None
        self.insufficient_improvements = 0

    @property
    def phase(self) -> str:
        if self.mode == "linear":
            ramp_start = self.hold_pairs
        elif self.ramp_start_pairs is None:
            return "hold"
        else:
            ramp_start = self.ramp_start_pairs
            if self.total_pairs < ramp_start + self.ramp_pairs:
                return "ramp"
            return "complete"
        if self.total_pairs <= ramp_start:
            return "hold"
        if self.total_pairs >= ramp_start + self.ramp_pairs:
            return "complete"
        return "ramp"

    @property
    def alpha(self) -> float:
        if self.mode == "linear":
            ramp_start = self.hold_pairs
        else:
            ramp_start = self.ramp_start_pairs
            if ramp_start is None:
                return 1.0
        progress = min(1.0, max(0.0, (self.total_pairs - ramp_start) / self.ramp_pairs))
        return 1.0 + (self.alpha_max - 1.0) * 0.5 * (1.0 - math.cos(math.pi * progress))

    def step(self, pairs: int) -> float:
        """Advance by this optimizer update's pair count and return the new alpha."""
        if pairs < 0:
            raise ValueError(f"pairs must be nonnegative, got {pairs}")
        self.total_pairs += int(pairs)
        return self.alpha

    def advance(self, pairs: int) -> float:
        return self.step(pairs)

    def update(self, pairs: int) -> float:
        return self.step(pairs)

    def on_validation(self, loss: float) -> float:
        """Record a validation loss and arm a plateau-mode ramp when warranted."""
        value = float(loss)
        if not math.isfinite(value):
            raise ValueError(f"validation loss must be finite, got {loss}")
        if value < 0:
            raise ValueError(f"validation loss must be nonnegative, got {loss}")
        if self.mode != "plateau" or self.ramp_start_pairs is not None:
            return self.alpha
        if self.ema_validation_loss is None:
            self.ema_validation_loss = value
            self.best_ema_validation_loss = value
            return self.alpha
        self.ema_validation_loss = 0.9 * self.ema_validation_loss + 0.1 * value
        assert self.best_ema_validation_loss is not None
        if self.best_ema_validation_loss - self.ema_validation_loss >= self.plateau_eps:
            self.best_ema_validation_loss = self.ema_validation_loss
            self.insufficient_improvements = 0
        else:
            self.insufficient_improvements += 1
            if self.insufficient_improvements >= self.plateau_patience:
                self.ramp_start_pairs = self.total_pairs
        return self.alpha

    def update_validation(self, loss: float) -> float:
        return self.on_validation(loss)

    def state_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "alpha_max": self.alpha_max,
            "hold_pairs": self.hold_pairs,
            "ramp_pairs": self.ramp_pairs,
            "plateau_patience": self.plateau_patience,
            "plateau_eps": self.plateau_eps,
            "total_pairs": self.total_pairs,
            "ramp_start_pairs": self.ramp_start_pairs,
            "ema_validation_loss": self.ema_validation_loss,
            "best_ema_validation_loss": self.best_ema_validation_loss,
            "insufficient_improvements": self.insufficient_improvements,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = set(self.state_dict())
        missing = required.difference(state)
        if missing:
            raise ValueError(f"scheduler state is missing keys: {sorted(missing)}")
        mode = str(state["mode"])
        if mode not in {"linear", "plateau"}:
            raise ValueError(f"mode must be 'linear' or 'plateau', got {mode!r}")
        alpha_max = _require_finite_positive("alpha_max", state["alpha_max"])
        hold_pairs = int(state["hold_pairs"])
        ramp_pairs = int(state["ramp_pairs"])
        plateau_patience = int(state["plateau_patience"])
        plateau_eps = _require_finite_nonnegative("plateau_eps", state["plateau_eps"])
        total_pairs = int(state["total_pairs"])
        if hold_pairs < 0 or ramp_pairs <= 0 or total_pairs < 0:
            raise ValueError("scheduler pair counts are invalid")
        if plateau_patience < 1:
            raise ValueError("plateau_patience must be at least one")
        ramp_start = state["ramp_start_pairs"]
        ramp_start_pairs = None if ramp_start is None else int(ramp_start)
        if ramp_start_pairs is not None and ramp_start_pairs < 0:
            raise ValueError("ramp_start_pairs must be nonnegative")
        ema = state["ema_validation_loss"]
        ema_validation_loss = None if ema is None else _require_finite_nonnegative(
            "ema_validation_loss", ema
        )
        best_ema = state["best_ema_validation_loss"]
        best_ema_validation_loss = (
            None
            if best_ema is None
            else _require_finite_nonnegative("best_ema_validation_loss", best_ema)
        )
        insufficient_improvements = int(state["insufficient_improvements"])
        if insufficient_improvements < 0:
            raise ValueError("insufficient_improvements must be nonnegative")
        self.mode = mode
        self.alpha_max = alpha_max
        self.hold_pairs = hold_pairs
        self.ramp_pairs = ramp_pairs
        self.plateau_patience = plateau_patience
        self.plateau_eps = plateau_eps
        self.total_pairs = total_pairs
        self.ramp_start_pairs = ramp_start_pairs
        self.ema_validation_loss = ema_validation_loss
        self.best_ema_validation_loss = best_ema_validation_loss
        self.insufficient_improvements = insufficient_improvements
