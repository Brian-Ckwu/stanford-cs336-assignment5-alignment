"""Best-effort W&B logging; local artifacts remain authoritative."""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

from .config import TokenConfidenceConfig


class WandbLogger:
    def __init__(self, run: Any) -> None:
        self.run = run
        self.failed = False

    @classmethod
    def create(cls, config: TokenConfidenceConfig) -> "WandbLogger | None":
        if config.wandb_mode == "disabled":
            return None
        import wandb

        run_id = hashlib.sha256(
            f"{config.scientific_hash()}:{config.run_name}".encode()
        ).hexdigest()[:32]
        try:
            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                id=run_id,
                resume="allow",
                name=config.run_name,
                tags=[
                    "offline-confidence",
                    config.prediction_mode,
                    "lora",
                    "gsm8k",
                ],
                mode=config.wandb_mode,
                config=config.to_dict(),
            )
        except Exception as error:
            warnings.warn(f"W&B initialization failed; continuing locally: {error}")
            return None
        return cls(run)

    def log(self, values: dict[str, Any], *, step: int) -> None:
        if self.failed:
            return
        try:
            self.run.log(values, step=step)
            best_values = {
                key: value for key, value in values.items() if key.startswith("best/")
            }
            if best_values and hasattr(self.run, "summary"):
                if hasattr(self.run.summary, "update"):
                    self.run.summary.update(best_values)
                else:
                    for key, value in best_values.items():
                        self.run.summary[key] = value
        except Exception as error:
            self.failed = True
            warnings.warn(f"W&B logging failed; continuing locally: {error}")

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as error:
            warnings.warn(f"W&B finish failed: {error}")
