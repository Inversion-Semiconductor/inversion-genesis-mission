"""Abstract simulation analysis datapoints with explicit completion tracking."""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import attrs
import numpy as np

from inversion_fbpic.lib.datapoint import _Datapoint
from inversion_fbpic.utils.distributions import (
    LONGITUDINAL_PROFILE_BINS,
    MOMENTS,
    OFF,
    ParticleArray,
    SPLINE,
    WeightArray,
    compute_moment_descriptor,
    crop_central_particles,
    load_openpmd_particles,
    mean_kinetic_energy_mev,
)

if TYPE_CHECKING:
    from inversion_fbpic.lib.simulation import Simulation
else:
    Simulation = Any

logger = logging.getLogger(__name__)


def _target_tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("target must be a (kind, value) tuple or serialized list.")
    if len(value) != 2:
        raise ValueError("target must contain exactly a kind and a value.")
    return tuple(value)


@attrs.define(kw_only=True, slots=False)
class _Diagnostic(_Datapoint):
    """Abstract datapoint populated by explicit or scheduled analysis.

    Concrete subclasses declare a ``SUBCLASS`` tag and implement ``_analyze``
    to return a dictionary of results. Analysis never runs during construction,
    loading, or serialization. Call ``analyze()`` to replace ``data`` and mark
    this instance complete. Serializing an incomplete instance logs a warning
    but still saves its current data.

    Completion is runtime-only: loading saved results preserves ``data`` but
    starts a new, incomplete instance. Subclasses should declare their analysis
    inputs as normal init fields so those inputs can also be saved and reloaded.

    Subclasses set ``RUN_BEFORE_SIMULATION`` to select automatic analysis at
    the end of simulation setup (True) or after simulation stepping (False,
    the default). This class-level policy is not an instance input or serialized
    parameter.

    Simulations attach their diagnostic elements automatically. The attachment
    is runtime-only, excluded from serialization, and available through
    ``attached_simulation``. A diagnostic instance may belong to only one
    simulation; use a separate instance for another simulation.
    """

    RUN_BEFORE_SIMULATION: ClassVar[bool] = False

    _analysis_complete: bool = attrs.field(
        default=False, init=False, repr=False, eq=False
    )
    _simulation: Simulation | None = attrs.field(
        default=None, init=False, repr=False, eq=False
    )

    def attach(self, simulation: Simulation) -> None:
        """Attach to a simulation, rejecting reuse by a different simulation."""
        from inversion_fbpic.lib.simulation import Simulation

        if not isinstance(simulation, Simulation):
            raise TypeError("A diagnostic must be attached to a Simulation.")
        if self._simulation is not None and self._simulation is not simulation:
            raise ValueError(
                "This diagnostic is already attached to another simulation."
            )
        self._simulation = simulation

    @property
    def attached_simulation(self) -> Simulation:
        """Return the attached simulation, or raise if no simulation is attached."""
        if self._simulation is None:
            raise ValueError("Attach this diagnostic to a Simulation before analyzing.")
        return self._simulation

    @property
    def analysis_complete(self) -> bool:
        """Whether the most recent explicit analysis completed successfully."""
        return self._analysis_complete

    @abstractmethod
    def _analyze(self) -> dict[str, Any]:
        """Perform analysis and return the desired result dictionary."""
        raise NotImplementedError

    def analyze(self) -> None:
        """Run analysis and store its results, marking completion only on success.

        Every call reruns ``_analyze()``. If it raises or returns something other
        than a dictionary, completion remains false and this wrapper does not
        replace the previous data. Mutations or external side effects performed
        by subclass code are not rolled back.

        Raises:
            TypeError: If ``_analyze()`` does not return a dictionary.
        """
        self._analysis_complete = False
        results = self._analyze()
        if not isinstance(results, dict):
            raise TypeError(
                f"{type(self).__name__}._analyze() must return a dict, "
                f"got {type(results).__name__}."
            )
        self.data = results
        self._analysis_complete = True

    def to_dict(self, *, include_nones: bool = True) -> dict[str, Any]:
        """Serialize current inputs and results, warning if analysis is incomplete."""
        if not self.analysis_complete:
            logger.warning(
                "%s analysis is incomplete; call analyze() before serializing "
                "to record completed results.",
                type(self).__name__,
            )
        return super().to_dict(include_nones=include_nones)


@attrs.define(kw_only=True, slots=False)
class _ParticleDiagnostic(_Diagnostic):
    """Abstract diagnostic for selected particles from the simulation's last dump.

    At analysis time, select the highest-numbered ``data########.h5`` file in
    the attached simulation's resolved save directory under ``hdf5``. The file
    need not exist at construction or attachment time. ``load_particles()``
    resolves ``target`` against attached density profiles and concatenates
    the selected particle arrays and macro-weights. Repeated recording names
    are loaded only once. ``select`` applies additional openPMD-viewer particle
    filtering to each recording before concatenation. When ``central_fraction``
    is not None, crop the combined distribution about its weighted mean on all
    six coordinates using ``crop_central_particles``.

    Concrete subclasses implement ``_analyze()`` and call ``load_particles()``
    to obtain the input for their calculation. This base has no concrete tag.

    Args:
        target: (tuple) Select recorded particles using ("elec_name", name
            or list of names), ("ion_name", name or list of names), or
            ("all_of_species", species). The species "e" selects all recorded
            electrons; an atomic symbol such as "He" selects recorded ions
            of that species. Names must be configured on the attached
            simulation's densities.
        select: (dict[str, list[float|None]]|None) |OPTIONAL| Additional filters
            mapping x, y, z (in meters) or ux, uy, uz (normalized momenta) to
            [minimum, maximum] ranges; None bounds are unbounded. For example,
            {"uz": [10.0, None]}. Defaults to None. None or an empty dict applies
            no further filtering.
        central_fraction: (float|None) |OPTIONAL| Central weighted fraction to
            retain on each coordinate after combining recordings. Must be
            strictly between 0 and 1. Defaults to None, which disables cropping.
    """

    target: (
        tuple[Literal["elec_name", "ion_name"], str | list[str]]
        | tuple[Literal["all_of_species"], str]
    ) = attrs.field(converter=_target_tuple)
    select: dict[str, list[float | None]] | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(dict)),
    )
    central_fraction: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(
            [attrs.validators.gt(0.0), attrs.validators.le(1.0)]
        ),
    )

    @target.validator
    def _validate_target(
        self, attribute: attrs.Attribute, value: tuple[Any, ...]
    ) -> None:
        kind, selected = value
        if kind not in ("elec_name", "ion_name", "all_of_species"):
            raise ValueError(
                "target kind must be elec_name, ion_name, or all_of_species."
            )
        if kind == "all_of_species":
            if not isinstance(selected, str) or not selected:
                raise ValueError("all_of_species requires a nonempty species string.")
        elif isinstance(selected, str):
            if not selected:
                raise ValueError("Selected recording names must be nonempty strings.")
        elif (
            not isinstance(selected, list)
            or not selected
            or not all(isinstance(name, str) and name for name in selected)
        ):
            raise ValueError(
                "Select a nonempty name or a nonempty list of recording names."
            )

    def _selected_recordings(self) -> list[str]:
        kind, selected = self.target
        densities = self.attached_simulation.densities
        if kind == "all_of_species":
            if selected == "e":
                names = [
                    density.elec_name for density in densities if density.elec_name
                ]
            else:
                names = [
                    density.ion_name
                    for density in densities
                    if density.species == selected and density.ion_name
                ]
            if not names:
                raise ValueError(f"No recorded particles match target {self.target!r}.")
        else:
            names = [selected] if isinstance(selected, str) else selected
            available = {
                getattr(density, kind)
                for density in densities
                if getattr(density, kind)
            }
            missing = [name for name in names if name not in available]
            if missing:
                raise ValueError(f"Unknown recorded {kind} target: {missing!r}.")
        return list(dict.fromkeys(names))

    def load_particles(self) -> tuple[ParticleArray, WeightArray]:
        """Load and concatenate selected particles and weights from the latest dump.

        ``select`` is passed to openPMD-viewer's ``get_particle(select=...)``
        for each recording identified by ``target``. If ``central_fraction`` is
        set, crop the concatenated particles and weights before returning them.

        Returns:
            A float64 particle array of shape (N, 6), ordered x, ux, y, uy, z,
            uz, and its corresponding macro-weights of shape (N,).

        Raises:
            ValueError: If unattached, no recordings match, no output exists,
                or fewer than two usable particles remain after cropping.
            Exception: Errors from the openPMD loader propagate unchanged.
        """
        names = self._selected_recordings()
        directory = self.attached_simulation._save_directory / "hdf5"
        diagnostic_files = [
            (int(match.group(1)), path)
            for path in directory.glob("data*.h5")
            if path.is_file()
            and (match := re.fullmatch(r"data(\d+)\.h5", path.name)) is not None
        ]
        if not diagnostic_files:
            raise ValueError(
                f"No openPMD particle diagnostic files found in {directory}."
            )
        _, last_file = max(diagnostic_files, key=lambda item: item[0])
        distributions = [
            load_openpmd_particles(last_file, name, select=self.select)
            for name in names
        ]
        particles = np.concatenate(
            [distribution[0] for distribution in distributions], axis=0
        )
        weights = np.concatenate(
            [distribution[1] for distribution in distributions], axis=0
        )
        if self.central_fraction is not None:
            particles, weights = crop_central_particles(
                particles, weights, central_fraction=self.central_fraction
            )
        return particles, weights


@attrs.define(kw_only=True, slots=False)
class MeanKineticEnergy(_ParticleDiagnostic):
    """Compute the charge-weighted mean electron kinetic energy in MeV.

    Loads selected particles from the attached simulation's latest dump using
    the inherited ``target`` and ``select`` inputs. All three normalized
    momentum components contribute, using the electron rest mass. Runs after
    the simulation and stores the result in ``data["mean_kinetic_energy_MeV"]``.
    """

    SUBCLASS: ClassVar[str] = "mean_kinetic_energy"
    RUN_BEFORE_SIMULATION: ClassVar[bool] = False

    def _analyze(self) -> dict[str, float]:
        particles, weights = self.load_particles()
        return {"mean_kinetic_energy_MeV": mean_kinetic_energy_mev(particles, weights)}


@attrs.define(kw_only=True, slots=False)
class MomentDescriptorDiagnostic(_ParticleDiagnostic):
    """Compute a moment descriptor for the selected particle recordings.

    Particles and weights are gathered by ``_ParticleDiagnostic`` from the
    attached simulation's latest dump. Their ``compute_moment_descriptor``
    result becomes ``data``. This diagnostic runs after the simulation and
    defaults to the shared 33-feature spline descriptor, including charge in pC.

    Args:
        longitudinal_mode: (int) |OPTIONAL| OFF, MOMENTS, or SPLINE. Defaults
            to SPLINE (2).
        longitudinal_bins: (int) |OPTIONAL| Number of spline profile bins,
            at least four. Defaults to 4.
        include_total_weight: (bool) |OPTIONAL| Include log_total_weight.
            Defaults to False.
        include_higher_moments: (bool) |OPTIONAL| Include coordinate skewness
            and excess kurtosis. Defaults to False.
    """

    SUBCLASS: ClassVar[str] = "moment_descriptor"
    RUN_BEFORE_SIMULATION: ClassVar[bool] = False

    OFF: ClassVar[int] = OFF
    MOMENTS: ClassVar[int] = MOMENTS
    SPLINE: ClassVar[int] = SPLINE
    LONGITUDINAL_PROFILE_BINS: ClassVar[int] = LONGITUDINAL_PROFILE_BINS

    longitudinal_mode: int = attrs.field(
        default=SPLINE, validator=attrs.validators.in_((OFF, MOMENTS, SPLINE))
    )
    longitudinal_bins: int = attrs.field(
        default=LONGITUDINAL_PROFILE_BINS,
        validator=[attrs.validators.instance_of(int), attrs.validators.ge(4)],
    )
    include_total_weight: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )
    include_higher_moments: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool)
    )

    def _analyze(self) -> dict[str, Any]:
        particles, weights = self.load_particles()
        return compute_moment_descriptor(
            particles,
            weights,
            longitudinal_mode=self.longitudinal_mode,
            longitudinal_bins=self.longitudinal_bins,
            include_total_weight=self.include_total_weight,
            include_higher_moments=self.include_higher_moments,
        )
