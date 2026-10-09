"""Append-only campaign result storage independent of optimizer checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fcntl
import pandas as pd

from .executor import EvaluationResult


class CampaignStore:
    """Persistent full-vector records reusable across active-parameter stages."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "campaign_runs.parquet"

    def append(self, result: EvaluationResult) -> None:
        manifest_path = result.run_directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "run_id": result.run_id,
            "stage": manifest["stage"],
            "campaign_fingerprint": manifest["campaign_fingerprint"],
            "status": result.status,
            "message": result.message,
            "wall_s": result.wall_s,
            "run_directory": str(result.run_directory),
        }
        row.update({f"p:{name}": value for name, value in manifest["parameters"].items()})
        row.update({f"f:{name}": value for name, value in result.features.items()})
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".campaign_runs.lock").open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame()
            if not current.empty and result.run_id in set(current["run_id"]):
                raise ValueError(f"duplicate run id: {result.run_id}")
            pd.concat([current, pd.DataFrame([row])], ignore_index=True).to_parquet(
                self.path, index=False
            )

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame()
