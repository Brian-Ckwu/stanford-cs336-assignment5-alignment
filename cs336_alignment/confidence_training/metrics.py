"""Metrics shared by token confidence training and confidence evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _rank(values: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(values)
    ranks = np.zeros_like(values, dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) - 1 and values[sorted_indices[end]] == values[sorted_indices[end + 1]]:
            end += 1
        ranks[sorted_indices[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    if denominator == 0:
        return 0.0
    return float((x_centered * y_centered).sum() / denominator)


def _c_star_ece(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    if len(predictions) == 0:
        return 0.0
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        upper = (
            predictions <= boundaries[index + 1]
            if index == n_bins - 1
            else predictions < boundaries[index + 1]
        )
        mask = (predictions >= boundaries[index]) & upper
        count = int(mask.sum())
        if count:
            ece += (count / len(predictions)) * abs(
                targets[mask].mean() - predictions[mask].mean()
            )
    return float(ece)


def compute_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> dict[str, float | int]:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    if not predictions:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "ece": 0.0,
            "num_examples": 0,
        }
    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    errors = prediction_array - target_array
    mse = float((errors**2).mean())
    return {
        "mae": float(np.abs(errors).mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "pearson_r": _pearson_correlation(prediction_array, target_array),
        "spearman_r": _pearson_correlation(
            _rank(prediction_array), _rank(target_array)
        ),
        "ece": _c_star_ece(prediction_array, target_array),
        "num_examples": len(predictions),
    }
