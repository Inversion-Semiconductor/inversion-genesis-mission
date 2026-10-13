"""Append-only campaign result storage independent of optimizer checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        row = self._row(manifest, result.as_dict())
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".campaign_runs.lock").open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current, _ = self._load_locked()
            if not current.empty and result.run_id in set(current["run_id"]):
                # Run directories are unique, and finalize writes result.json before append.
                return
            self._publish(pd.concat([current, pd.DataFrame([row])], ignore_index=True))

    def rebuild(self) -> pd.DataFrame:
        """Recreate the aggregate table from durable per-run result records."""
        rebuilt = self._rebuild_data()
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".campaign_runs.lock").open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._publish(rebuilt)
        return rebuilt

    def _row(self, manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run_id": result["run_id"],
            "stage": manifest["stage"],
            "campaign_fingerprint": manifest["campaign_fingerprint"],
            "status": result["status"],
            "message": result.get("message", ""),
            "wall_s": result.get("wall_s", 0.0),
            "run_directory": result["run_directory"],
        }
        row.update({f"p:{name}": value for name, value in manifest["parameters"].items()})
        row.update({f"f:{name}": value for name, value in result.get("features", {}).items()})
        return row

    def _publish(self, data: pd.DataFrame) -> None:
        temporary = self.root / f".{self.path.name}.{uuid4().hex}.tmp"
        try:
            data.to_parquet(temporary, index=False)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_locked(self) -> tuple[pd.DataFrame, bool]:
        """Read the aggregate table or rebuild it while the caller holds the lock."""
        if not self.path.exists():
            return pd.DataFrame(), False
        try:
            return pd.read_parquet(self.path), False
        except Exception:
            rebuilt = self._rebuild_data()
            if rebuilt.empty:
                raise
            self._publish(rebuilt)
            return rebuilt, True

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.path)
        except Exception:
            rebuilt = self._rebuild_data()
            if rebuilt.empty:
                raise
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / ".campaign_runs.lock").open("w", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                repaired, _ = self._load_locked()
                return repaired

    def _rebuild_data(self) -> pd.DataFrame:
        rows = []
        for result_path in sorted(self.root.glob("sim_*/result.json")):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(self._row(record["manifest"], record["result"]))
        return pd.DataFrame(rows)
