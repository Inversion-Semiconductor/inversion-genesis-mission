"""Versioned one-output GP cache artifacts for campaign post-analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .campaign_rows import is_no_beam

if TYPE_CHECKING:
    from gpytorch.module import Module

    from .exploration import XoptFBPICExplorer


CACHE_FORMAT_VERSION = 2
LEGACY_EXACT_CACHE_FORMAT_VERSION = 1
METADATA_FILENAME = "metadata.json"
TRAINING_FILENAME = "training.parquet"
MODEL_STATE_FILENAME = "model_state.pt"
FEASIBILITY_VARIABLE = "feasibility"
MAX_FEASIBILITY_TRAINING_ROWS = 512
MODEL_FAMILY = "approximate_variational_gp"
INDUCING_POINTS = 128


@dataclass(frozen=True)
class CachedSurrogate:
    """One loaded GP and the immutable data/configuration used to train it."""

    variable: str
    metadata: dict[str, Any]
    training_data: pd.DataFrame
    model: Module


@dataclass(frozen=True)
class SurrogateTrainingResult:
    """Per-variable cache directories trained or skipped by one request."""

    trained: tuple[Path, ...]
    skipped: tuple[Path, ...]


def cache_directory(cache_root: str | Path, variable: str) -> Path:
    """Return the per-output artifact directory for a scalar feature name."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", variable):
        raise ValueError("variable must contain only letters, digits, and underscores")
    return Path(cache_root) / f"var_{variable}"


def feature_variables(data: pd.DataFrame) -> tuple[str, ...]:
    """Return all campaign scalar names represented by ``f:`` columns."""
    return tuple(
        sorted(column.removeprefix("f:") for column in data if column.startswith("f:"))
    )


def train_surrogates(
    explorer: XoptFBPICExplorer,
    data_path: str | Path,
    cache_root: str | Path,
    variable: str,
    overwrite: bool = False,
) -> SurrogateTrainingResult:
    """Train and cache one GP, or a separate GP for every available scalar.

    Historical data are selected by the sole fingerprint in ``data_path`` rather
    than the current campaign fingerprint. This permits stable post-analysis of
    an archived campaign after its configuration has evolved.
    """
    data_path = Path(data_path).resolve()
    data = pd.read_parquet(data_path)
    candidates = feature_variables(data) if variable == "all" else (variable,)
    if not candidates:
        raise ValueError("campaign parquet contains no scalar feature columns")
    trained, skipped = [], []
    for name in candidates:
        destination = cache_directory(cache_root, name)
        if (
            variable == "all"
            and _is_compatible_cache(destination, data_path, name)
            and not overwrite
        ):
            skipped.append(destination)
            continue
        trained.append(train_surrogate(explorer, data_path, data, cache_root, name))
    return SurrogateTrainingResult(tuple(trained), tuple(skipped))


def train_surrogate(
    explorer: XoptFBPICExplorer,
    data_path: Path,
    data: pd.DataFrame,
    cache_root: str | Path,
    variable: str,
) -> Path:
    """Train one scalar GP and write its immutable cache artifacts."""
    training, source_fingerprint, outcome_metadata = _training_snapshot(
        explorer, data, variable
    )
    generator = _make_generator(explorer, variable)
    model = generator.train_model(data=training, update_internal=False)
    model.eval()

    import torch
    import xopt

    cache_path = cache_directory(cache_root, variable)
    cache_path.mkdir(parents=True, exist_ok=True)
    training_path = cache_path / TRAINING_FILENAME
    state_path = cache_path / MODEL_STATE_FILENAME
    training.to_parquet(training_path, index=False)
    torch.save(model.state_dict(), state_path)
    metadata = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "variable": variable,
        "model_family": MODEL_FAMILY,
        "inducing_points": INDUCING_POINTS,
        "source_data_path": str(data_path),
        "source_data_sha256": _sha256(data_path),
        "source_campaign_fingerprint": source_fingerprint,
        "training_rows": len(training),
        "input_names": list(explorer.variable_bounds),
        "input_bounds": explorer.variable_bounds,
        "training_filename": TRAINING_FILENAME,
        "training_sha256": _sha256(training_path),
        "model_state_filename": MODEL_STATE_FILENAME,
        "model_state_sha256": _sha256(state_path),
        **outcome_metadata,
        "gp_constructor": generator.gp_constructor.model_dump(mode="json"),
        "versions": {
            "xopt": getattr(xopt, "__version__", "unknown"),
            "torch": torch.__version__,
        },
    }
    (cache_path / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache_path


def load_surrogate(
    explorer: XoptFBPICExplorer, cache_root: str | Path, variable: str
) -> CachedSurrogate:
    """Recreate a cached GP without fitting its hyperparameters again."""
    cache_path = cache_directory(cache_root, variable)
    metadata_path = cache_path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing surrogate metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_format = metadata.get("cache_format_version")
    if cache_format not in {CACHE_FORMAT_VERSION, LEGACY_EXACT_CACHE_FORMAT_VERSION}:
        raise ValueError("Unsupported surrogate cache format version")
    if metadata.get("variable") != variable:
        raise ValueError("Surrogate cache variable does not match requested variable")
    legacy_exact = cache_format == LEGACY_EXACT_CACHE_FORMAT_VERSION
    if not legacy_exact and metadata.get("model_family") != MODEL_FAMILY:
        raise ValueError("Surrogate cache model family does not match this loader")
    if not legacy_exact and metadata.get("inducing_points") != INDUCING_POINTS:
        raise ValueError(
            "Surrogate cache inducing-point count does not match this loader"
        )
    if metadata.get("input_names") != list(explorer.variable_bounds):
        raise ValueError("Surrogate cache input names do not match the requested stage")
    if metadata.get("input_bounds") != explorer.variable_bounds:
        raise ValueError(
            "Surrogate cache input bounds do not match the requested stage"
        )

    training_path = cache_path / metadata["training_filename"]
    state_path = cache_path / metadata["model_state_filename"]
    if _sha256(training_path) != metadata["training_sha256"]:
        raise ValueError(
            "Surrogate cache training snapshot hash does not match metadata"
        )
    if _sha256(state_path) != metadata["model_state_sha256"]:
        raise ValueError("Surrogate cache model state hash does not match metadata")
    training = pd.read_parquet(training_path)
    generator = (
        _make_legacy_exact_generator(explorer, variable)
        if legacy_exact
        else _make_generator(explorer, variable, train_model=False)
    )
    model = generator.gp_constructor.build_model(
        input_names=list(explorer.variable_bounds),
        outcome_names=[variable],
        data=training,
        input_bounds=generator.get_model_input_bounds(training),
        **generator.tkwargs,
    )
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if legacy_exact:
        model.load_state_dict(state)
    else:
        torch.nn.Module.load_state_dict(model, state)
    model.eval()
    return CachedSurrogate(variable, metadata, training, model)


def _training_snapshot(
    explorer: XoptFBPICExplorer, data: pd.DataFrame, variable: str
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if variable == FEASIBILITY_VARIABLE:
        return _feasibility_training_snapshot(explorer, data)
    feature_column = f"f:{variable}"
    input_names = list(explorer.variable_bounds)
    columns = [
        "campaign_fingerprint",
        "status",
        *[f"p:{name}" for name in input_names],
        feature_column,
    ]
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Campaign parquet is missing columns: {', '.join(missing)}")
    fingerprints = data["campaign_fingerprint"].dropna().unique()
    if len(fingerprints) != 1:
        raise ValueError(
            "Campaign parquet must contain exactly one campaign fingerprint"
        )
    source_fingerprint = str(fingerprints[0])
    selected = data.loc[
        (data["campaign_fingerprint"] == source_fingerprint) & data["status"].eq("ok"),
        [*[f"p:{name}" for name in input_names], feature_column],
    ].copy()
    selected.columns = [*input_names, variable]
    for parameter in explorer.parameters:
        selected[parameter.name] = [
            explorer._to_model_value(parameter, float(value))
            for value in selected[parameter.name]
        ]
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) < 2:
        raise ValueError(f"Need at least two finite beam rows to train {variable!r}")
    return selected, source_fingerprint, {"outcome_kind": "scalar"}


def _feasibility_training_snapshot(
    explorer: XoptFBPICExplorer, data: pd.DataFrame
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Build a $-1/+1$ latent target from physical beam/no-beam outcomes."""
    input_names = list(explorer.variable_bounds)
    columns = [
        "campaign_fingerprint",
        "status",
        "message",
        *[f"p:{name}" for name in input_names],
    ]
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Campaign parquet is missing columns: {', '.join(missing)}")
    fingerprints = data["campaign_fingerprint"].dropna().unique()
    if len(fingerprints) != 1:
        raise ValueError(
            "Campaign parquet must contain exactly one campaign fingerprint"
        )
    source_fingerprint = str(fingerprints[0])
    selected = data.loc[
        data["campaign_fingerprint"] == source_fingerprint, columns
    ].copy()
    statuses = selected["status"].astype(str)
    messages = selected["message"].astype(str)
    beam = statuses.eq("ok")
    no_beam = np.asarray(
        [
            is_no_beam(status, message)
            for status, message in zip(statuses, messages, strict=True)
        ]
    )
    selected = selected.loc[beam | no_beam, [*[f"p:{name}" for name in input_names]]]
    selected.columns = input_names
    selected[FEASIBILITY_VARIABLE] = np.where(beam[beam | no_beam], 1.0, -1.0)
    for parameter in explorer.parameters:
        selected[parameter.name] = [
            explorer._to_model_value(parameter, float(value))
            for value in selected[parameter.name]
        ]
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) < 2 or selected[FEASIBILITY_VARIABLE].nunique() < 2:
        raise ValueError("Need both finite beam and no-beam outcomes for feasibility")
    selected = _stratified_feasibility_subset(selected)
    return (
        selected,
        source_fingerprint,
        {
            "outcome_kind": "probit_feasibility",
            "beam_rows": int(beam.sum()),
            "no_beam_rows": int(no_beam.sum()),
            "excluded_rows": int((~(beam | no_beam)).sum()),
            "sampling": "deterministic_class_balanced",
            "maximum_training_rows": MAX_FEASIBILITY_TRAINING_ROWS,
        },
    )


def _stratified_feasibility_subset(data: pd.DataFrame) -> pd.DataFrame:
    """Bound exact-GP feasibility fitting with equal deterministic class samples."""
    if len(data) <= MAX_FEASIBILITY_TRAINING_ROWS:
        return data.reset_index(drop=True)
    random_generator = np.random.default_rng(0)
    per_class = MAX_FEASIBILITY_TRAINING_ROWS // 2
    indices = []
    for label in (-1.0, 1.0):
        class_indices = data.index[data[FEASIBILITY_VARIABLE] == label].to_numpy()
        count = min(per_class, len(class_indices))
        indices.extend(
            random_generator.choice(class_indices, size=count, replace=False)
        )
    return data.loc[sorted(indices)].reset_index(drop=True)


def _is_compatible_cache(cache_path: Path, data_path: Path, variable: str) -> bool:
    """Return whether an existing cache matches this source and model family."""
    metadata_path = cache_path / METADATA_FILENAME
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("cache_format_version") == CACHE_FORMAT_VERSION
        and metadata.get("model_family") == MODEL_FAMILY
        and metadata.get("inducing_points") == INDUCING_POINTS
        and metadata.get("variable") == variable
        and metadata.get("source_data_sha256") == _sha256(data_path)
    )


def _make_generator(
    explorer: XoptFBPICExplorer, variable: str, *, train_model: bool = True
):
    """Build a bounded inducing-point GP generator for cache training or loading."""
    try:
        from xopt import VOCS
        from xopt.generators.bayesian import BayesianExplorationGenerator
        from xopt.generators.bayesian.models.approximate import (
            ApproximateModelConstructor,
        )
        from xopt.generators.bayesian.models.standard import (
            AdamNumericalOptimizerConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "install exploration_fbpic with xopt to use surrogate caches"
        ) from error
    vocs = VOCS(variables=explorer.variable_bounds, objectives={variable: "EXPLORE"})
    constructor = ApproximateModelConstructor(
        inducing_points=INDUCING_POINTS,
        train_method="adam",
        train_model=train_model,
        train_config=AdamNumericalOptimizerConfig(timeout=120.0),
    )
    return BayesianExplorationGenerator(vocs=vocs, gp_constructor=constructor)


def _make_legacy_exact_generator(explorer: XoptFBPICExplorer, variable: str):
    """Recreate the version-1 standard exact GP cache architecture without fitting."""
    try:
        from xopt import VOCS
        from xopt.generators.bayesian import BayesianExplorationGenerator
        from xopt.generators.bayesian.models.standard import StandardModelConstructor
    except ImportError as error:
        raise RuntimeError(
            "install exploration_fbpic with xopt to load surrogate caches"
        ) from error
    vocs = VOCS(variables=explorer.variable_bounds, objectives={variable: "EXPLORE"})
    return BayesianExplorationGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(train_model=False),
    )


def _sha256(path: Path) -> str:
    """Return a SHA-256 digest for a cache artifact or source parquet."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
