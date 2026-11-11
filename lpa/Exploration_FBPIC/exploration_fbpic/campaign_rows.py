"""Shared selection and coordinate conversion for durable campaign records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .exploration import XoptFBPICExplorer


NO_BEAM_MESSAGE = "No scalar diagnostic results found"


def is_no_beam(status: str, message: str = "") -> bool:
    """Identify completed physical no-injection outcomes, including legacy rows."""
    return status == "no_beam" or (
        status in {"error", "analysis_error"} and NO_BEAM_MESSAGE in message
    )


def compatible_rows(explorer: "XoptFBPICExplorer", data: pd.DataFrame) -> pd.DataFrame:
    """Return rows belonging to the explorer's immutable campaign fingerprint."""
    if data.empty or "campaign_fingerprint" not in data:
        return data.iloc[0:0].copy()
    fingerprint = explorer.campaign.config.fingerprint()
    return data.loc[data["campaign_fingerprint"] == fingerprint].copy()


def physical_outcome_rows(
    explorer: "XoptFBPICExplorer", data: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return compatible beam/no-beam rows with their physical outcome masks."""
    compatible = compatible_rows(explorer, data)
    parameter_columns = [f"p:{name}" for name in explorer.variable_bounds]
    missing = sorted(set(parameter_columns) - set(compatible.columns))
    if missing:
        raise ValueError(
            f"Campaign records are missing active parameter columns: {', '.join(missing)}"
        )
    statuses = compatible.get("status", pd.Series("", index=compatible.index)).astype(
        str
    )
    messages = compatible.get("message", pd.Series("", index=compatible.index)).astype(
        str
    )
    beam = statuses.eq("ok").to_numpy()
    no_beam = np.asarray(
        [
            is_no_beam(status, message)
            for status, message in zip(statuses, messages, strict=True)
        ],
        dtype=bool,
    )
    selected = beam | no_beam
    return compatible.loc[selected].copy(), beam[selected], no_beam[selected]


def model_input_matrix(explorer: "XoptFBPICExplorer", rows: pd.DataFrame) -> np.ndarray:
    """Convert active physical parameter columns in campaign rows to Xopt space."""
    parameter_names = list(explorer.variable_bounds)
    columns = [f"p:{name}" for name in parameter_names]
    missing = sorted(set(columns) - set(rows.columns))
    if missing:
        raise ValueError(
            f"Campaign records are missing active parameter columns: {', '.join(missing)}"
        )
    physical = rows.loc[:, columns].to_numpy(dtype=float)
    return np.asarray(
        [
            [
                explorer._to_model_value(parameter, value)
                for parameter, value in zip(explorer.parameters, row, strict=True)
            ]
            for row in physical
        ],
        dtype=float,
    )
