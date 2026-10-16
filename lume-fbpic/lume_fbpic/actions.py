"""PV-facing Action/Variable classes for lume_fbpic.

`inversion_fbpic`'s config objects (`SimulationHyperparameters`, `_LaserPulse`,
`_DensityProfile` subclasses) are frozen `attrs` classes, so writable actions here replace
the whole held object via `attrs.evolve()` rather than mutating a field in place -- unlike
`lume-cheetah`'s actions, which `setattr()` directly on Cheetah's plain mutable elements.

The `unit` of an action can be left out. `LUMEFBPICModel` then fills it in with
`default_unit()`, from `lume_fbpic.units`: for an action on a config field, the unit tagged in
brackets in that field's documentation (`[m]`, `[J]`, `[m^-3]`, ...), and for a statistic its entry
in `STAT_UNITS`. A unit given explicitly, `None` included, is kept.
"""

from __future__ import annotations

import typing

import attrs

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ParticleGroupVariable, ScalarVariable

from lume_fbpic import descriptor
from lume_fbpic.density_profiles import atom_density, nominal_density
from lume_fbpic.pwfa_config import FlatTopBunch, GaussianBunch
from lume_fbpic.lwfa_simulator import LWFASimulator
from lume_fbpic.pwfa_simulator import PWFASimulator
from lume_fbpic.simulator import BaseSimulator
from lume_fbpic.units import STAT_UNITS, config_unit

from inversion_fbpic.utils.distributions import LONGITUDINAL_PROFILE_BINS


# The prefix of the names `make_descriptor_actions()` gives by default.
DESCRIPTOR_PREFIX = "descriptor_"

# A laser's amplitude field -> the other one, derived from it (see `evolve_laser()`).
_AMPLITUDE_SIBLING = {"a0": "energy", "energy": "a0"}

# The writable fields of a PWFA bunch that `make_pwfa_actions()` makes actions for, by the class of
# the bunch: action name (after the `driver_` or `witness_` prefix) -> field.
_BUNCH_FIELDS = {
    FlatTopBunch: {"gamma": "gamma", "density": "density", "radius": "radius"},
    GaussianBunch: {
        "gamma": "gamma",
        "charge": "charge",
        "sigma_r": "sig_r",
        "sigma_z": "sig_z",
        "position": "zf",
    },
}


class _ConfigFieldAction(WritableActionMixin[BaseSimulator], ScalarVariable):
    """Writable scalar mapped onto field `field_name` of one of the simulator's config objects,
    which a subclass reaches with `_config()` and puts back with `_replace()`. The config objects
    are frozen, so a set replaces the object with an `attrs.evolve()`d copy."""

    # The class of simulator the action works on; `LUMEFBPICModel` checks it.
    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = BaseSimulator

    field_name: str

    def default_unit(self, simulator: BaseSimulator) -> str | None:
        return config_unit(self._config(simulator), self.field_name)

    def _config(self, simulator: BaseSimulator) -> typing.Any:
        """The config object this action's field is on."""
        raise NotImplementedError

    def _get(self, simulator: BaseSimulator) -> typing.Any:
        return getattr(self._config(simulator), self.field_name)

    def _replace(self, simulator: BaseSimulator, config: typing.Any) -> None:
        """Put `config` in place of the config object `_config()` returns."""
        raise NotImplementedError

    def _set(self, simulator: BaseSimulator, value: typing.Any) -> None:
        self._replace(
            simulator, attrs.evolve(self._config(simulator), **{self.field_name: value})
        )


class BunchFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of a PWFA simulator's `driver` or `witness` bunch."""

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = PWFASimulator

    bunch: typing.Literal["driver", "witness"]

    def _config(self, simulator: PWFASimulator) -> typing.Any:
        bunch = getattr(simulator, self.bunch)
        if bunch is None:
            raise ValueError(f"the simulator has no {self.bunch}")
        return bunch

    def _replace(self, simulator: PWFASimulator, config: typing.Any) -> None:
        setattr(simulator, self.bunch, config)


class DensityFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of one of the simulator's density profiles,
    addressed by index into `simulator.densities`."""

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = LWFASimulator

    density_index: int

    def _config(self, simulator: LWFASimulator) -> typing.Any:
        return simulator.densities[self.density_index]

    def _replace(self, simulator: LWFASimulator, config: typing.Any) -> None:
        densities = list(simulator.densities)
        densities[self.density_index] = config
        simulator.densities = densities


class DopantFractionAction(WritableActionMixin[LWFASimulator], ScalarVariable):
    """Writable dopant fraction of a two-gas mixture, held in two density profiles of
    `simulator.densities` (`host_index` for the background gas, `dopant_index` for the dopant).

    The fraction is `dopant atoms / (host atoms + dopant atoms)`, with each profile's atom density
    from `lume_fbpic.density_profiles.atom_density()`. Setting a new fraction rewrites BOTH
    `nominal_density` values while keeping the total atom density (host + dopant) constant. That is the scan variable `nitrogen_dopant_fraction` of the
    `initial_sample` dataset, where `n_N = n_gas * fraction` and `n_He = n_gas * (1 - fraction)`
    with `n_gas` fixed. Both profiles must have a `species`, and the fraction must lie strictly
    between 0 and 1 (a profile's `nominal_density` has to stay positive).
    """

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = LWFASimulator

    dopant_index: int
    host_index: int

    def default_unit(self, simulator: LWFASimulator) -> str | None:
        return "1"  # a fraction

    def _get(self, simulator: LWFASimulator) -> typing.Any:
        host, dopant = self._profiles(simulator)
        return atom_density(dopant) / (atom_density(host) + atom_density(dopant))

    def _profiles(self, simulator: LWFASimulator) -> tuple[typing.Any, typing.Any]:
        """The host and the dopant profiles."""
        return (
            simulator.densities[self.host_index],
            simulator.densities[self.dopant_index],
        )

    def _set(self, simulator: LWFASimulator, value: typing.Any) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Dopant fraction must be strictly between 0 and 1, got {value}."
            )
        host, dopant = self._profiles(simulator)
        total = atom_density(host) + atom_density(dopant)
        densities = list(simulator.densities)
        densities[self.host_index] = attrs.evolve(
            host, nominal_density=nominal_density(host.species, total * (1.0 - value))
        )
        densities[self.dopant_index] = attrs.evolve(
            dopant, nominal_density=nominal_density(dopant.species, total * value)
        )
        simulator.densities = densities


class FinalParticlesAction(ReadOnlyActionMixin[BaseSimulator], ParticleGroupVariable):
    """Read-only ParticleGroup sourced from `simulator.final_particles` (populated after a
    run).

    The value is `None` while there are no particles -- before a run, or in a model loaded from
    an archive written without them. `validate_value` accepts that, as LUME's own check would
    reject it and with it every `model.get()` that includes this variable (including the one
    `lume-pva` makes for all variables on each cycle).
    """

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = BaseSimulator

    def validate_value(self, value: typing.Any) -> None:
        if value is not None:
            super().validate_value(value)

    def _get(self, simulator: BaseSimulator) -> typing.Any:
        return simulator.final_particles


class HyperparameterFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of the simulator's `hyparams` config object."""

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = LWFASimulator

    def _config(self, simulator: LWFASimulator) -> typing.Any:
        return simulator.hyparams

    def _replace(self, simulator: LWFASimulator, config: typing.Any) -> None:
        simulator.hyparams = config


class LaserFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of the simulator's `laser` config object, set with
    `evolve_laser()`."""

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = LWFASimulator

    def _config(self, simulator: LWFASimulator) -> typing.Any:
        return simulator.laser

    def _replace(self, simulator: LWFASimulator, config: typing.Any) -> None:
        simulator.laser = config

    def _set(self, simulator: LWFASimulator, value: typing.Any) -> None:
        simulator.laser = evolve_laser(simulator.laser, **{self.field_name: value})


class MomentDescriptorAction(ReadOnlyActionMixin[BaseSimulator], ScalarVariable):
    """Read-only scalar: ONE feature of the moment descriptor of `simulator.final_particles`.

    The descriptor is `lume_fbpic.descriptor.compute()` -- the 33 features (3 momentum
    centroids, 21 covariance entries, 2 x `longitudinal_bins` longitudinal `uz` features,
    charge) that `build_dataset.py` writes for each simulation, of the particles its selection
    keeps: `uz >= uz_min`, then the central `central_fraction` on each of the six phase-space
    axes (either `None` skips that step). `make_descriptor_actions()` builds the whole set. The
    value is NaN before a run, or when fewer than two particles survive the selection. A
    simulator without particles gives its `recorded_outputs` value for this action's name, if it
    has one (one loaded from an archive of a reconstructed run).

    The simulator computes the descriptor once per selection (`moment_descriptor()`), so reading
    all 33 features costs one computation.
    """

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = BaseSimulator

    central_fraction: float | None = descriptor.CENTRAL_FRACTION
    feature: str
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS
    uz_min: float | None = descriptor.UZ_MIN

    def default_unit(self, simulator: BaseSimulator) -> str | None:
        """The unit of the feature (`lume_fbpic.descriptor.features()`); None for a name that
        is not a feature of the descriptor."""
        return descriptor.features(self.longitudinal_bins).get(self.feature)

    def _get(self, simulator: BaseSimulator) -> typing.Any:
        if simulator.final_particles is None:
            return simulator.recorded_outputs.get(self.name, float("nan"))
        return simulator.moment_descriptor(
            self.uz_min, self.central_fraction, self.longitudinal_bins
        ).get(self.feature, float("nan"))


class PlasmaFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of a PWFA simulator's plasma density profile."""

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = PWFASimulator

    def _config(self, simulator: PWFASimulator) -> typing.Any:
        return simulator.plasma

    def _replace(self, simulator: PWFASimulator, config: typing.Any) -> None:
        simulator.plasma = config


class StatAction(ReadOnlyActionMixin[BaseSimulator], ScalarVariable):
    """Read-only scalar sourced from `simulator.stats` (populated after a run).

    Without a `stat_name` entry, the simulator's `recorded_outputs` value for this action's name
    is used (one loaded from an archive of a reconstructed run has no statistics).
    """

    SIMULATOR: typing.ClassVar[type[BaseSimulator]] = BaseSimulator

    stat_name: str

    def default_unit(self, simulator: BaseSimulator) -> str | None:
        return STAT_UNITS.get(self.stat_name)

    def _get(self, simulator: BaseSimulator) -> typing.Any:
        if self.stat_name in simulator.stats:
            return simulator.stats[self.stat_name]
        return simulator.recorded_outputs.get(self.name, float("nan"))


class ZernikeCoefficientAction(LaserFieldAction):
    """Writable scalar mapped onto ONE entry of the laser's `zernike_coefficients` dict (for
    example `coefficient="coma_x"`), leaving the other coefficients as they are. Only
    `LasyLaserPulse` has this field; the energy/`a0` handling of `LaserFieldAction._set` carries
    over unchanged. A coefficient name the laser does not accept is rejected when the new
    config is built."""

    coefficient: str
    field_name: str = "zernike_coefficients"

    def _get(self, simulator: LWFASimulator) -> typing.Any:
        return super()._get(simulator).get(self.coefficient, 0.0)

    def _set(self, simulator: LWFASimulator, value: typing.Any) -> None:
        super()._set(simulator, {**super()._get(simulator), self.coefficient: value})


def evolve_laser(laser: typing.Any, **changes: typing.Any) -> typing.Any:
    """`attrs.evolve(laser, **changes)` for a `_LaserPulse`.

    Every `_LaserPulse` holds *both* `energy` and `a0` -- whichever wasn't given at construction
    is derived and stored too (tracked on the private `_amplitude_source` attribute) -- but its
    constructor takes only one of them. So the one that should not survive the change is dropped
    first: the other amplitude when `changes` sets `energy` or `a0`, otherwise the derived one.
    """
    changed = [name for name in _AMPLITUDE_SIBLING if name in changes]
    derived = _AMPLITUDE_SIBLING[changed[0] if changed else laser._amplitude_source]
    return attrs.evolve(laser, **{derived: None, **changes})


def make_descriptor_actions(
    *,
    central_fraction: float | None = descriptor.CENTRAL_FRACTION,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
    prefix: str = DESCRIPTOR_PREFIX,
    uz_min: float | None = descriptor.UZ_MIN,
) -> list[Action]:
    """Build one read-only `MomentDescriptorAction` per descriptor feature (33 by default).

    The names are `prefix` + the feature name (`descriptor_mean_uz`,
    `descriptor_cov_x_ux`, `descriptor_total_beam_charge_pc`, ...), in the order
    `compute_moment_descriptor` returns them. The default selection (`uz_min=30`,
    `central_fraction=0.95`) is the `build_dataset.py` default, tuned to the
    ionization-injection beams; a lower-energy bunch (the LWFA example's, for instance, has a
    mean `uz` near 2) needs a lower `uz_min`.

    Each action has the unit of its feature (`lume_fbpic.descriptor.features()`): the momenta
    `ux`, `uy`, `uz` are normalized and so dimensionless, `"1"`, the positions are in metres,
    and the charge is in pC.
    """
    return [
        MomentDescriptorAction(
            name=f"{prefix}{feature}",
            feature=feature,
            central_fraction=central_fraction,
            longitudinal_bins=longitudinal_bins,
            read_only=True,
            uz_min=uz_min,
            unit=unit,
        )
        for feature, unit in descriptor.features(longitudinal_bins).items()
    ]


def make_pwfa_actions(simulator: PWFASimulator) -> list[Action]:
    """Build the default control/output variable set for a PWFA simulator.

    Control variables (writable): the plasma density, and for the driver and, if there is
    one, the witness (named `driver_*` / `witness_*`): the Lorentz factor, and for a
    `FlatTopBunch` the density and radius, for a `GaussianBunch` the charge, the RMS radius and
    length and the position `zf`.
    Outputs (read-only): charge and mean/std energy of the `target_species`, and
    `final_particles`.
    """
    actions: list[Action] = [
        PlasmaFieldAction(name="plasma_density", field_name="nominal_density")
    ]
    actions += _bunch_actions("driver", simulator.driver)
    if simulator.witness is not None:
        actions += _bunch_actions("witness", simulator.witness)
    return actions + [
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]


def make_stat_actions() -> list[Action]:
    """One read-only `StatAction` per statistic a run records (`lume_fbpic.units.STAT_UNITS`),
    named after it: `charge_pc`, `energy_mean_mev` and `energy_std_mev`."""
    return [
        StatAction(name=name, stat_name=name, read_only=True) for name in STAT_UNITS
    ]


def _bunch_actions(bunch: str, config: typing.Any) -> list[Action]:
    """The writable actions of one PWFA bunch (`driver` or `witness`), by the kind of bunch
    (`_BUNCH_FIELDS`).

    Raises:
        ValueError: For a kind of bunch that has no entry in `_BUNCH_FIELDS`.
    """
    if type(config) not in _BUNCH_FIELDS:
        raise ValueError(f"no actions for a bunch of type {type(config).__name__}")
    return [
        BunchFieldAction(name=f"{bunch}_{name}", bunch=bunch, field_name=field)
        for name, field in _BUNCH_FIELDS[type(config)].items()
    ]
