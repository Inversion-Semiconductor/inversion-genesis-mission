"""Append-only campaign result storage independent of optimizer checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


class CampaignStore:
    """Persistent full-vector records reusable across active-parameter stages."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "campaign_runs.parquet"

    def rebuild(
        self,
        *,
        result_filename: str = "result.json",
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Recreate an aggregate table from native or shifted per-run records.

        When selecting a shifted record, physical no-beam records fall back to
        native ``result.json`` so the feasibility classifier retains outcomes
        that have no electron distribution to shift.
        """
        rebuilt = self._rebuild_data(result_filename)
        destination = Path(output_path) if output_path is not None else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._publish(rebuilt, destination)
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
        row.update(
            {f"p:{name}": value for name, value in manifest["parameters"].items()}
        )
        row.update(
            {f"f:{name}": value for name, value in result.get("features", {}).items()}
        )
        return row

    def _publish(self, data: pd.DataFrame, path: Path | None = None) -> None:
        path = path or self.path
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            data.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> pd.DataFrame:
        """Load the aggregate cache, rebuilding native records after corruption."""
        if not self.path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.path)
        except Exception:
            rebuilt = self._rebuild_data("result.json")
            if rebuilt.empty:
                raise
            self.root.mkdir(parents=True, exist_ok=True)
            self._publish(rebuilt)
            return rebuilt

    def _rebuild_data(self, result_filename: str = "result.json") -> pd.DataFrame:
        if Path(result_filename).name != result_filename:
            raise ValueError("result_filename must be a filename")
        rows = []
        for run_directory in sorted(
            path for path in self.root.glob("sim_*") if path.is_dir()
        ):
            base_path = run_directory / "result.json"
            if not base_path.is_file():
                continue
            result_path = run_directory / result_filename
            if result_filename != "result.json" and not result_path.is_file():
                base_record = json.loads(base_path.read_text(encoding="utf-8"))
                if base_record["result"]["status"] == "ok":
                    raise FileNotFoundError(
                        f"Missing {result_filename} for successful run {run_directory.name}"
                    )
                result_path = base_path
            record = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(self._row(record["manifest"], record["result"]))
        return pd.DataFrame(rows)
