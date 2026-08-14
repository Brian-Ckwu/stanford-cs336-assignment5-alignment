#!/usr/bin/env python3
"""Fit isotonic calibration on one trace and evaluate it on saved traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.isotonic import IsotonicRegression

try:
    from .evaluation_with_cc import compute_metrics
except ImportError:
    from evaluation_with_cc import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-traces", type=Path, required=True)
    parser.add_argument("--raw-traces", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-labels", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def arrays(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    predictions = [float(record["prediction_for_metrics"]) for record in records]
    targets = [float(record["expected_accuracy"]) for record in records]
    return predictions, targets


def main() -> None:
    args = parse_args()
    if len(args.raw_traces) != len(args.dataset_labels):
        raise ValueError("--raw-traces and --dataset-labels must have equal lengths")

    calibration_records = load_trace(args.calibration_traces)
    calibration_predictions, calibration_targets = arrays(calibration_records)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrated_train = calibrator.fit_transform(
        calibration_predictions,
        calibration_targets,
    )

    result: dict[str, Any] = {
        "calibration": {
            "num_examples": len(calibration_records),
            "raw_metrics": compute_metrics(
                calibration_predictions,
                calibration_targets,
            ),
            "isotonic_metrics": compute_metrics(
                calibrated_train.tolist(),
                calibration_targets,
            ),
        },
        "isotonic_model": {
            "x_thresholds": calibrator.X_thresholds_.tolist(),
            "y_thresholds": calibrator.y_thresholds_.tolist(),
            "out_of_bounds": "clip",
            "y_min": 0.0,
            "y_max": 1.0,
        },
        "datasets": {},
    }

    for label, trace_path in zip(args.dataset_labels, args.raw_traces):
        records = load_trace(trace_path)
        predictions, targets = arrays(records)
        isotonic_predictions = calibrator.predict(predictions).tolist()
        result["datasets"][label] = {
            "num_examples": len(records),
            "raw_metrics": compute_metrics(predictions, targets),
            "isotonic_metrics": compute_metrics(isotonic_predictions, targets),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
