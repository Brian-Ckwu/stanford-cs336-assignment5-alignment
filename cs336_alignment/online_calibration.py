"""CPU-only, rolling bias calibration of a frozen confidence estimator."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

CONFIDENCE_EPSILON = 1e-6
BIAS_REGULARIZATION = 1e-3
BIAS_TOLERANCE = 1e-8


def _probabilities(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError("Confidences must be a finite one-dimensional sequence in [0, 1]")
    return result


def _logits(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, CONFIDENCE_EPSILON, 1 - CONFIDENCE_EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def predictive_metrics(predictions: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    """Group MSE and per-response BCE; signed bias is prediction minus outcome."""
    p, y = _probabilities(predictions), _probabilities(targets)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("Metrics require nonempty, aligned predictions and targets")
    z = _logits(p)
    return {
        "mean_prediction": float(p.mean()),
        "mse": float(np.mean((p - y) ** 2)),
        "signed_bias": float(np.mean(p - y)),
        "bce": float(np.mean(np.logaddexp(0.0, z) - y * z)),
    }


@dataclass(frozen=True)
class CalibrationObservation:
    rollout_step: int
    training_row_index: int
    raw_confidence: float
    calibrated_confidence: float
    correct_count: int
    group_size: int
    bias_used: float
    calibration_version: int

    def __post_init__(self) -> None:
        for name in ("rollout_step", "training_row_index", "correct_count", "group_size", "calibration_version"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.group_size == 0 or self.correct_count > self.group_size:
            raise ValueError("Invalid group size or correct count")
        _probabilities([self.raw_confidence, self.calibrated_confidence])
        if not math.isfinite(self.bias_used):
            raise ValueError("bias_used must be finite")


class OnlineScalarCalibrator:
    """Refit an absolute log-odds bias after each successful rollout batch.

    Warm-up counts completed batches, independently of the retained history size.
    The caller applies the published bias only to subsequent collections.
    """

    def __init__(self, *, warmup_steps: int = 5, history_steps: int = 5) -> None:
        if warmup_steps <= 0 or history_steps <= 0:
            raise ValueError("Warm-up and history lengths must be positive")
        self.warmup_steps = warmup_steps
        self.history_steps = history_steps
        self.bias = 0.0
        self.version = 0
        self.completed_steps = 0
        self.batches: list[list[CalibrationObservation]] = []

    def predict(self, raw_confidences: Sequence[float]) -> list[float]:
        p = _probabilities(raw_confidences)
        if not math.isfinite(self.bias):
            raise ValueError("Calibration bias must be finite")
        if self.bias == 0.0:
            return p.tolist()
        return _sigmoid(_logits(p) + self.bias).tolist()

    @staticmethod
    def _fit(batches: Sequence[Sequence[CalibrationObservation]]) -> float:
        observations = [item for batch in batches for item in batch]
        z = _logits(_probabilities([item.raw_confidence for item in observations]))
        y = np.asarray([item.correct_count / item.group_size for item in observations], dtype=np.float64)

        def derivative(bias: float) -> float:
            value = float(np.mean(_sigmoid(z + bias) - y) + BIAS_REGULARIZATION * bias)
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite calibration derivative")
            return value

        lower, upper = -1.0, 1.0
        for _ in range(100):
            if derivative(lower) <= 0 <= derivative(upper):
                break
            lower *= 2
            upper *= 2
        else:
            raise RuntimeError("Could not bracket calibration optimum")
        for _ in range(100):
            if upper - lower <= BIAS_TOLERANCE:
                return (lower + upper) / 2
            midpoint = (lower + upper) / 2
            if derivative(midpoint) > 0:
                upper = midpoint
            else:
                lower = midpoint
        raise RuntimeError("Calibration fit did not converge")

    def observe_and_update(self, observations: Sequence[CalibrationObservation]) -> dict[str, float]:
        batch = list(observations)
        if not batch or len({item.rollout_step for item in batch}) != 1:
            raise ValueError("Each update requires one nonempty rollout batch")
        if self.batches and batch[0].rollout_step <= self.batches[-1][0].rollout_step:
            raise ValueError("Rollout steps must increase strictly")
        if any(item.bias_used != self.bias or item.calibration_version != self.version for item in batch):
            raise ValueError("Observations must use the current calibration version and bias")
        next_batches = (self.batches + [batch])[-self.history_steps:]
        fit_performed = self.completed_steps + 1 >= self.warmup_steps
        next_bias = self._fit(next_batches) if fit_performed else self.bias
        old_bias, old_version = self.bias, self.version
        self.batches = next_batches
        self.completed_steps += 1
        self.bias = next_bias
        self.version += int(fit_performed)
        return {
            "bias_used": old_bias,
            "bias_next": self.bias,
            "bias_change": self.bias - old_bias,
            "version_used": float(old_version),
            "version_next": float(self.version),
            "fit_performed": float(fit_performed),
            "buffer_steps": float(len(self.batches)),
            "buffer_groups": float(sum(map(len, self.batches))),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": {
                "warmup_steps": self.warmup_steps,
                "history_steps": self.history_steps,
                "epsilon": CONFIDENCE_EPSILON,
                "regularization": BIAS_REGULARIZATION,
                "tolerance": BIAS_TOLERANCE,
            },
            "bias": self.bias,
            "version": self.version,
            "completed_steps": self.completed_steps,
            "batches": [[asdict(item) for item in batch] for batch in self.batches],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> OnlineScalarCalibrator:
        config = state["config"]
        if state["schema_version"] != 1 or (
            config["epsilon"], config["regularization"], config["tolerance"]
        ) != (CONFIDENCE_EPSILON, BIAS_REGULARIZATION, BIAS_TOLERANCE):
            raise ValueError("Unsupported calibration state configuration")
        result = cls(warmup_steps=config["warmup_steps"], history_steps=config["history_steps"])
        result.bias = float(state["bias"])
        result.version = state["version"]
        result.completed_steps = state["completed_steps"]
        result.batches = [[CalibrationObservation(**item) for item in batch] for batch in state["batches"]]
        if not math.isfinite(result.bias) or len(result.batches) != min(result.completed_steps, result.history_steps):
            raise ValueError("Invalid calibration state")
        if result.version != max(0, result.completed_steps - result.warmup_steps + 1):
            raise ValueError("Invalid calibration version")
        previous_step = -1
        for batch in result.batches:
            if not batch or len({item.rollout_step for item in batch}) != 1 or batch[0].rollout_step <= previous_step:
                raise ValueError("Invalid calibration history")
            previous_step = batch[0].rollout_step
        return result
