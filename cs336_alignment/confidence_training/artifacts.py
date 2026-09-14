"""Local-authoritative artifacts and dual validation selectors."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import TokenConfidenceConfig


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    with destination.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class RunArtifacts:
    def __init__(self, config: TokenConfidenceConfig) -> None:
        self.config = config
        self.root = config.output_path
        self.manifest_path = self.root / "run_manifest.json"
        self.config_path = self.root / "resolved_config.json"
        self.selector_path = self.root / "selectors.json"
        self.history_path = self.root / "history.jsonl"
        self.checkpoint_root = self.root / "checkpoints"
        self.resume_path = self.root / "last.pt"

    def initialize(self, provenance: dict[str, Any]) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            if not self.config.resume:
                raise FileExistsError(
                    f"Output directory is nonempty: {self.root}; use --resume only "
                    "for the identical run"
                )
            existing = json.loads(self.config_path.read_text())
            if existing["scientific_config_hash"] != self.config.scientific_hash():
                raise ValueError("Resume configuration does not match the existing run")
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("status") == "complete":
                raise ValueError("A complete run cannot be resumed")
            manifest["status"] = "running"
            manifest.pop("error", None)
            atomic_write_json(self.manifest_path, manifest)
            return

        self.checkpoint_root.mkdir(parents=True)
        selectors = {
            "mse": {"mode": "min", "value": None, "epoch": None},
            "spearman_r": {"mode": "max", "value": None, "epoch": None},
        }
        atomic_write_json(
            self.config_path,
            {
                "config": self.config.to_dict(),
                "scientific_config_hash": self.config.scientific_hash(),
                "provenance": provenance,
            },
        )
        atomic_write_json(self.selector_path, selectors)
        atomic_write_json(
            self.manifest_path,
            {
                "artifact_version": 1,
                "status": "running",
                "scientific_config_hash": self.config.scientific_hash(),
                "config": self.config.to_dict(),
                "provenance": provenance,
                "selectors": selectors,
            },
        )

    def append_history(self, row: dict[str, Any]) -> None:
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _selectors(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.selector_path.read_text())

    def improvements(self, metrics: dict[str, float | int]) -> set[str]:
        selectors = self._selectors()
        improved: set[str] = set()
        mse = float(metrics["mse"])
        spearman = float(metrics["spearman_r"])
        if selectors["mse"]["value"] is None or mse < selectors["mse"]["value"]:
            improved.add("mse")
        if (
            selectors["spearman_r"]["value"] is None
            or spearman > selectors["spearman_r"]["value"]
        ):
            improved.add("spearman_r")
        return improved

    def save_improved_checkpoint(
        self,
        *,
        epoch: int,
        optimizer_steps: int,
        metrics: dict[str, float | int],
        predictions: Sequence[dict[str, Any]],
        improved: set[str],
        model_saver: Callable[[Path], None],
    ) -> Path | None:
        if not improved:
            return None
        final_path = self.checkpoint_root / f"epoch_{epoch:06d}"
        if not final_path.exists():
            temporary = Path(
                tempfile.mkdtemp(prefix=f".epoch_{epoch:06d}.", dir=self.checkpoint_root)
            )
            try:
                model_saver(temporary / "model")
                atomic_write_json(temporary / "metrics.json", metrics)
                write_jsonl(temporary / "validation_predictions.jsonl", predictions)
                atomic_write_json(
                    temporary / "checkpoint_manifest.json",
                    {
                        "checkpoint_version": 1,
                        "epoch": epoch,
                        "optimizer_steps": optimizer_steps,
                        "selected_by": sorted(improved),
                    },
                )
                os.replace(temporary, final_path)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        selectors = self._selectors()
        for selector in improved:
            selectors[selector] = {
                "mode": "min" if selector == "mse" else "max",
                "value": float(metrics[selector]),
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "checkpoint_path": str(final_path),
                "metrics": metrics,
            }
        atomic_write_json(self.selector_path, selectors)
        referenced = {
            Path(value["checkpoint_path"]).resolve()
            for value in selectors.values()
            if value.get("checkpoint_path")
        }
        for checkpoint in self.checkpoint_root.glob("epoch_*"):
            if checkpoint.resolve() not in referenced:
                shutil.rmtree(checkpoint)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["selectors"] = selectors
        atomic_write_json(self.manifest_path, manifest)
        return final_path

    def update_runtime(self, values: dict[str, Any]) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        manifest.setdefault("runtime_summary", {}).update(values)
        atomic_write_json(self.manifest_path, manifest)

    def mark_complete(self, *, trained_through_epoch: int) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        manifest["status"] = "complete"
        manifest["trained_through_epoch"] = trained_through_epoch
        manifest["selectors"] = self._selectors()
        atomic_write_json(self.manifest_path, manifest)

    def mark_failed(self, error: str) -> None:
        if not self.manifest_path.exists():
            return
        manifest = json.loads(self.manifest_path.read_text())
        manifest["status"] = "failed"
        manifest["error"] = error
        atomic_write_json(self.manifest_path, manifest)
