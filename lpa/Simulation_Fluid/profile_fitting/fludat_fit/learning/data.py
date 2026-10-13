"""Training data for ``conditions -> profile parameters`` models.

A :class:`TrainingSet` pairs lineouts, taken at sampled physical conditions,
with the local-optimisation parameters that fit them, all expressed in one
fixed :class:`~fludat_fit.parameters.ParameterSpace` for the family. Models
see normalised conditions (:attr:`TrainingSet.features`, in ``[0, 1]``) and
predict unit-cube parameters (``targets`` are the local optima in the same
cube), so the framework never depends on the family's physical units.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import NozzleDataset
from ..families import ProfileFamily, get_families
from ..fitting import FitObjective, FitResult, MultiStartLocalFit
from ..goodness_of_fit import GoodnessOfFit
from ..lineout import Lineout, LineoutConditions
from ..parameters import ParameterSpace, ParameterSpec
from ..sampling import ConditionRanges


def reference_parameter_space(
    family: ProfileFamily, lineouts: Iterable[Lineout]
) -> ParameterSpace:
    """One space that contains every lineout's own family space.

    Per-lineout bounds depend on the lineout's extent and width; a learned
    model needs a single cube, so the envelope (lowest lower, highest upper
    bound per parameter) is used.
    """
    spaces = [family.parameter_space(lineout.summary()) for lineout in lineouts]
    if not spaces:
        raise ValueError("at least one lineout is needed")
    specs = []
    for index, spec in enumerate(spaces[0].specs):
        others = [space.specs[index] for space in spaces]
        if any(other.name != spec.name for other in others):
            raise ValueError("family parameter spaces differ between lineouts")
        specs.append(
            ParameterSpec(
                spec.name,
                min(other.lower for other in others),
                max(other.upper for other in others),
                log_scale=spec.log_scale,
                unit=spec.unit,
            )
        )
    return ParameterSpace(tuple(specs))


@dataclass(frozen=True)
class TrainingSample:
    """One lineout with (optionally) its locally optimised parameters.

    Args:
        lineout: The data (already windowed for fitting).
        target: Unit-cube parameters of the local optimum, or ``None``.
        reference_goodness: Goodness of that local fit (the attainable bound).
        reference_evaluations: Objective evaluations the cold local fit needed.
    """

    lineout: Lineout
    target: np.ndarray | None = None
    reference_goodness: GoodnessOfFit | None = None
    reference_evaluations: int = 0

    @property
    def conditions(self) -> LineoutConditions:
        return self.lineout.conditions

    @classmethod
    def from_fit(cls, result: FitResult) -> TrainingSample:
        return cls(
            lineout=result.lineout,
            target=np.clip(result.space.to_unit(result.theta), 0.0, 1.0),
            reference_goodness=result.goodness,
            reference_evaluations=result.n_function_evaluations,
        )


class TrainingSet:
    """Samples of one family in one reference parameter space.

    Args:
        family: The profile family the targets parameterise.
        space: The fixed reference :class:`ParameterSpace`.
        ranges: The condition box used to normalise features.
        samples: The samples.
        name: Source description for reports.
    """

    def __init__(
        self,
        family: ProfileFamily,
        space: ParameterSpace,
        ranges: ConditionRanges,
        samples: Sequence[TrainingSample],
        *,
        name: str = "",
    ) -> None:
        if not samples:
            raise ValueError("a training set needs at least one sample")
        for sample in samples:
            if sample.target is not None and sample.target.shape != (space.dimension,):
                raise ValueError("target dimension does not match the parameter space")
        self.family = family
        self.space = space
        self.ranges = ranges
        self.samples: tuple[TrainingSample, ...] = tuple(samples)
        self.name = name
        self._objectives: dict[int, FitObjective] = {}

    # ---------------------------------------------------------------- views
    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self.samples)

    @property
    def dimension(self) -> int:
        return self.space.dimension

    @property
    def conditions(self) -> np.ndarray:
        """``(N, 3)`` physical conditions ``[x_mm, pressure_bar, angle_deg]``."""
        return np.array([sample.conditions.as_vector() for sample in self.samples])

    @property
    def features(self) -> np.ndarray:
        """``(N, 3)`` conditions normalised to ``[0, 1]`` by :attr:`ranges`."""
        return self.ranges.normalize(self.conditions)

    @property
    def has_targets(self) -> bool:
        return all(sample.target is not None for sample in self.samples)

    @property
    def targets(self) -> np.ndarray:
        """``(N, d)`` unit-cube local optima; raises without complete targets."""
        if not self.has_targets:
            raise ValueError("this training set has no local-fit targets")
        return np.array([sample.target for sample in self.samples])

    @property
    def reference_nrmse(self) -> np.ndarray:
        """Local-fit NRMSE per sample (``nan`` where no local fit exists)."""
        return np.array(
            [
                np.nan if s.reference_goodness is None else s.reference_goodness.nrmse
                for s in self.samples
            ]
        )

    def fit_objective(self, index: int) -> FitObjective:
        """The (cached) local-fit objective of sample ``index`` in the reference space."""
        objective = self._objectives.get(index)
        if objective is None:
            objective = FitObjective(
                self.samples[index].lineout, self.family, self.space
            )
            self._objectives[index] = objective
        return objective

    # --------------------------------------------------------------- splits
    def subset(
        self, indices: Sequence[int] | np.ndarray, name: str = ""
    ) -> TrainingSet:
        return TrainingSet(
            self.family,
            self.space,
            self.ranges,
            [self.samples[int(i)] for i in indices],
            name=name or self.name,
        )

    def split(
        self, test_fraction: float = 0.25, seed: int | None = 0
    ) -> tuple[TrainingSet, TrainingSet]:
        """A random ``(train, test)`` split with at least one sample in each."""
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")
        if len(self) < 2:
            raise ValueError("need at least two samples to split")
        order = np.random.default_rng(seed).permutation(len(self))
        n_test = int(np.clip(round(test_fraction * len(self)), 1, len(self) - 1))
        return self.subset(order[n_test:], "train"), self.subset(order[:n_test], "test")

    def k_folds(
        self, k: int, seed: int | None = 0
    ) -> Iterator[tuple[TrainingSet, TrainingSet]]:
        """``k`` ``(train, test)`` folds over a random permutation."""
        if k < 2 or k > len(self):
            raise ValueError(f"k must be in [2, {len(self)}]")
        order = np.random.default_rng(seed).permutation(len(self))
        for fold in np.array_split(order, k):
            train = np.setdiff1d(order, fold)
            yield self.subset(train, "train"), self.subset(fold, "test")

    # -------------------------------------------------------------- storage
    def save(self, path: str | Path) -> Path:
        """Write to an ``.npz`` (lineouts, targets, and a JSON metadata record)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        records = []
        for index, sample in enumerate(self.samples):
            arrays[f"z_{index}"] = sample.lineout.z
            arrays[f"density_{index}"] = sample.lineout.density
            if sample.lineout.weights is not None:
                arrays[f"weights_{index}"] = sample.lineout.weights
            if sample.target is not None:
                arrays[f"target_{index}"] = sample.target
            records.append(
                {
                    "conditions": sample.conditions.to_dict(),
                    "source": sample.lineout.source,
                    "reference_goodness": (
                        None
                        if sample.reference_goodness is None
                        else sample.reference_goodness.to_dict()
                    ),
                    "reference_evaluations": sample.reference_evaluations,
                }
            )
        meta = {
            "family": self.family.name,
            "name": self.name,
            "density_units": self.samples[0].lineout.density_units,
            "space": [
                {
                    "name": s.name,
                    "lower": s.lower,
                    "upper": s.upper,
                    "log_scale": s.log_scale,
                    "unit": s.unit,
                }
                for s in self.space.specs
            ],
            "ranges": self.ranges.to_dict(),
            "samples": records,
        }
        np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path, family: ProfileFamily | None = None) -> TrainingSet:
        """Read a set written by :meth:`save`; the family is resolved by name if omitted."""
        with np.load(path) as archive:
            meta = json.loads(str(archive["meta"]))
            if family is None:
                family = get_families([meta["family"]])[0]
            elif family.name != meta["family"]:
                raise ValueError(
                    f"training set is for {meta['family']!r}, not {family.name!r}"
                )
            space = ParameterSpace(
                tuple(ParameterSpec(**spec) for spec in meta["space"])
            )
            samples = []
            for index, record in enumerate(meta["samples"]):
                weights_key = f"weights_{index}"
                lineout = Lineout(
                    z=archive[f"z_{index}"],
                    density=archive[f"density_{index}"],
                    density_units=meta["density_units"],
                    conditions=LineoutConditions(**record["conditions"]),
                    source=record["source"],
                    weights=archive[weights_key] if weights_key in archive else None,
                )
                target_key = f"target_{index}"
                goodness = record["reference_goodness"]
                samples.append(
                    TrainingSample(
                        lineout=lineout,
                        target=archive[target_key] if target_key in archive else None,
                        reference_goodness=(
                            None if goodness is None else GoodnessOfFit(**goodness)
                        ),
                        reference_evaluations=int(record["reference_evaluations"]),
                    )
                )
        return cls(
            family,
            space,
            ConditionRanges.from_dict(meta["ranges"]),
            samples,
            name=meta["name"],
        )


# --------------------------------------------------------------------- build
def _local_fit(
    payload: tuple[Lineout, ProfileFamily, ParameterSpace, MultiStartLocalFit],
) -> FitResult:
    lineout, family, space, scheme = payload
    return scheme.fit(lineout, family, space=space)


def build_training_set(
    dataset: NozzleDataset,
    points: Iterable[LineoutConditions],
    family: ProfileFamily,
    *,
    ranges: ConditionRanges,
    window=None,
    scheme: MultiStartLocalFit | None = None,
    fit_targets: bool = True,
    workers: int = 1,
    progress: bool = False,
) -> TrainingSet:
    """Extract lineouts at ``points`` and (optionally) fit ``family`` to each.

    Args:
        dataset: The density cube.
        points: Conditions to sample; lineouts without density are skipped.
        family: The profile family.
        ranges: The condition box the points came from (feature normalisation).
        window: A ``cli_common.FitWindow`` (or anything with ``apply(lineout)``);
            default: no windowing.
        scheme: Local fitting scheme for the targets; default ``MultiStartLocalFit()``.
        fit_targets: If ``False`` only lineouts are stored (direct-loss training only).
        workers: Process pool size for the local fits.
        progress: Print per-sample progress to stderr.
    """
    lineouts = []
    for conditions in points:
        lineout = dataset.lineout(conditions)
        if window is not None:
            lineout = window.apply(lineout)
        if lineout.peak <= 0.0:
            if progress:
                print(f"  skip {conditions.label()}: no density", file=sys.stderr)
            continue
        lineouts.append(lineout)
    if not lineouts:
        raise ValueError("no lineout with density at the sampled points")
    space = reference_parameter_space(family, lineouts)

    if not fit_targets:
        samples = [TrainingSample(lineout) for lineout in lineouts]
    else:
        scheme = MultiStartLocalFit() if scheme is None else scheme
        payloads = [(lineout, family, space, scheme) for lineout in lineouts]
        started = time.perf_counter()
        results: list[FitResult] = []
        if workers > 1 and len(payloads) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                iterator = pool.map(_local_fit, payloads)
                for index, result in enumerate(iterator, start=1):
                    results.append(result)
                    _report(progress, index, len(payloads), result, started)
        else:
            for index, payload in enumerate(payloads, start=1):
                result = _local_fit(payload)
                results.append(result)
                _report(progress, index, len(payloads), result, started)
        samples = [TrainingSample.from_fit(result) for result in results]
    return TrainingSet(family, space, ranges, samples, name=dataset.name)


def _report(
    progress: bool, index: int, total: int, result: FitResult, started: float
) -> None:
    if progress:
        print(
            f"  [{index}/{total}] {result.conditions.label()}: NRMSE "
            f"{result.goodness.nrmse:.3g} [{time.perf_counter() - started:.0f}s]",
            file=sys.stderr,
        )


def sample_records(training_set: TrainingSet) -> list[dict[str, Any]]:
    """Flat per-sample records (conditions, features, targets by parameter name)."""
    records = []
    features = training_set.features
    for index, sample in enumerate(training_set.samples):
        record: dict[str, Any] = {**sample.conditions.to_dict()}
        record.update({f"feature_{i}": float(v) for i, v in enumerate(features[index])})
        if sample.target is not None:
            record.update(
                {
                    f"target_{name}": float(v)
                    for name, v in zip(
                        training_set.space.names, sample.target, strict=True
                    )
                }
            )
        if sample.reference_goodness is not None:
            record["reference_nrmse"] = sample.reference_goodness.nrmse
        records.append(record)
    return records
