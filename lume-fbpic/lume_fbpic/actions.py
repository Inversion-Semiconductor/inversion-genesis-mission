"""PV-facing Action/Variable classes for lume_fbpic.

`inversion_fbpic`'s config objects (`SimulationHyperparameters`, `_LaserPulse`,
`_DensityProfile`, `_ParticleBeam` and `_ExternalField` subclasses) are frozen `attrs` classes, so writable actions here replace
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
import pydantic

from lume.actions import Action, ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ParticleGroupVariable, ScalarVariable

from inversion_fbpic.lib.external_field import UniformExternalField
from inversion_fbpic.lib.particle_beam import (
    ExpressionParticleBeam,
    FileParticleBeam,
    GaussianParticleBeam,
    OpenPMDParticleBeam,
)
from lume_fbpic import descriptor, utils
from lume_fbpic.density_profiles import atom_density, nominal_density
from lume_fbpic.simulator import FBPICSimulator
from lume_fbpic.units import STAT_UNITS, config_unit

from inversion_fbpic.utils.distributions import LONGITUDINAL_PROFILE_BINS


# The prefix of the names `make_descriptor_actions()` gives by default.
DESCRIPTOR_PREFIX = "descriptor_"


# The writable fields of an `inversion_fbpic` particle beam that `make_beam_actions()` makes actions
# for, by the class of the beam: action name (after the prefix) -> field. A beam with
# `n_physical_particles` also gets a `charge` (`BeamTotalChargeAction`).
_BEAM_FIELDS = {
    ExpressionParticleBeam: {
        "gamma": "gamma0",
        "density": "n",
        "zmin": "p_zmin",
        "zmax": "p_zmax",
        "radius": "p_rmax",
    },
    FileParticleBeam: {"z_offset": "z_off"},
    GaussianParticleBeam: {
        "gamma": "gamma0",
        "gamma_spread": "sig_gamma",
        "sigma_r": "sig_r",
        "sigma_z": "sig_z",
        "emittance": "n_emit",
        "position": "zf",
    },
    OpenPMDParticleBeam: {"z_offset": "z_off"},
}


class _ConfigFieldAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable scalar mapped onto field `field_name` of one of the simulator's config objects,
    which a subclass reaches with `_config()` and puts back with `_replace()`. The config objects
    are frozen, so a set replaces the object with an `attrs.evolve()`d copy."""

    field_name: str

    def default_unit(self, simulator: FBPICSimulator) -> str | None:
        return config_unit(self._config(simulator), self.field_name)

    def _config(self, simulator: FBPICSimulator) -> typing.Any:
        """The config object this action's field is on."""
        raise NotImplementedError

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return getattr(self._config(simulator), self.field_name)

    def _replace(self, simulator: FBPICSimulator, config: typing.Any) -> None:
        """Put `config` in place of the config object `_config()` returns."""
        raise NotImplementedError

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        self._replace(
            simulator, attrs.evolve(self._config(simulator), **{self.field_name: value})
        )


class BeamFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of one of the simulator's particle beams, addressed by
    index into `simulator.beams`."""

    beam_index: int

    def _config(self, simulator: FBPICSimulator) -> typing.Any:
        return simulator.beams[self.beam_index]

    def _replace(self, simulator: FBPICSimulator, config: typing.Any) -> None:
        beams = list(simulator.beams)
        beams[self.beam_index] = config
        simulator.beams = beams


class BeamTotalChargeAction(BeamFieldAction):
    """Writable total charge [C] of a particle beam that is given as a number of physical particles
    (`GaussianParticleBeam`, `FileParticleBeam`): a positive magnitude, `n_physical_particles` times
    the magnitude of the particles' charge `q`. Setting it sets `n_physical_particles`.
    """

    field_name: str = "n_physical_particles"

    def default_unit(self, simulator: FBPICSimulator) -> str | None:
        return "C"

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        b = self._config(simulator)
        return b.n_physical_particles * abs(b.q)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        b = self._config(simulator)
        self._replace(simulator, attrs.evolve(b, n_physical_particles=value / abs(b.q)))


class DensityFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of one of the simulator's density profiles,
    addressed by index into `simulator.densities`."""

    density_index: int

    def _config(self, simulator: FBPICSimulator) -> typing.Any:
        return simulator.densities[self.density_index]

    def _replace(self, simulator: FBPICSimulator, config: typing.Any) -> None:
        densities = list(simulator.densities)
        densities[self.density_index] = config
        simulator.densities = densities


class DopantFractionAction(WritableActionMixin[FBPICSimulator], ScalarVariable):
    """Writable dopant fraction of a two-gas mixture, held in two density profiles of
    `simulator.densities` (`host_index` for the background gas, `dopant_index` for the dopant).

    The fraction is `dopant molecules / (host molecules + dopant molecules)`. Each gas's molecule
    density is its profile's atom density (`lume_fbpic.density_profiles.atom_density()`) over its
    atoms per molecule: 1 by default, so the fraction is one of atoms, and 2 for a dopant of
    diatomic molecules such as N2. Setting a new fraction rewrites BOTH `nominal_density` values
    while keeping the total molecule density (the neutral gas density) constant: with N2 in He,
    `n_N2 = n_gas * fraction` and `n_He = n_gas * (1 - fraction)`. Both profiles must have a
    `species`, and the fraction must lie strictly between 0 and 1 (a profile's `nominal_density`
    has to stay positive).
    """

    dopant_atoms_per_molecule: int = pydantic.Field(default=1, ge=1)
    dopant_index: int
    host_atoms_per_molecule: int = pydantic.Field(default=1, ge=1)
    host_index: int

    def default_unit(self, simulator: FBPICSimulator) -> str | None:
        return "1"  # a fraction

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        host, dopant = self._molecule_densities(simulator)
        return dopant / (host + dopant)

    def _molecule_densities(self, simulator: FBPICSimulator) -> tuple[float, float]:
        """The host's and the dopant's molecule densities [m^-3]."""
        host, dopant = self._profiles(simulator)
        return (
            atom_density(host) / self.host_atoms_per_molecule,
            atom_density(dopant) / self.dopant_atoms_per_molecule,
        )

    def _profiles(self, simulator: FBPICSimulator) -> tuple[typing.Any, typing.Any]:
        """The host and the dopant profiles."""
        return (
            simulator.densities[self.host_index],
            simulator.densities[self.dopant_index],
        )

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Dopant fraction must be strictly between 0 and 1, got {value}."
            )
        host, dopant = self._profiles(simulator)
        total = sum(self._molecule_densities(simulator))
        densities = list(simulator.densities)
        densities[self.host_index] = attrs.evolve(
            host,
            nominal_density=nominal_density(
                host.species, total * (1.0 - value) * self.host_atoms_per_molecule
            ),
        )
        densities[self.dopant_index] = attrs.evolve(
            dopant,
            nominal_density=nominal_density(
                dopant.species, total * value * self.dopant_atoms_per_molecule
            ),
        )
        simulator.densities = densities


class ExternalFieldAction(_ConfigFieldAction):
    """Writable strength of one component (`field_type`, `"Ex"` ... `"Bz"`) of one of the
    simulator's uniform external fields (`UniformExternalField`), addressed by index into
    `simulator.external_fields`; the other components are left as they are. The unit is V/m for
    an electric field and T for a magnetic one. An `ExpressionExternalField` has no single strength
    and is refused."""

    field_index: int
    field_type: typing.Literal["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
    field_name: str = "field_functions"

    def default_unit(self, simulator: FBPICSimulator) -> str | None:
        return "V/m" if self.field_type.startswith("E") else "T"

    def _config(self, simulator: FBPICSimulator) -> typing.Any:
        f = simulator.external_fields[self.field_index]
        if not isinstance(f, UniformExternalField):
            raise TypeError(
                f"action {self.name!r} needs a UniformExternalField, not a "
                + f"{type(f).__name__}"
            )
        return f

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return self._config(simulator).field_functions.get(self.field_type, 0.0)

    def _replace(self, simulator: FBPICSimulator, config: typing.Any) -> None:
        fields = list(simulator.external_fields)
        fields[self.field_index] = config
        simulator.external_fields = fields

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        f = self._config(simulator)
        self._replace(
            simulator,
            attrs.evolve(
                f, field_functions={**f.field_functions, self.field_type: value}
            ),
        )


class FinalParticlesAction(ReadOnlyActionMixin[FBPICSimulator], ParticleGroupVariable):
    """Read-only ParticleGroup sourced from `simulator.final_particles` (populated after a
    run).

    The value is `None` while there are no particles -- before a run, or in a model loaded from
    an archive written without them. `validate_value` accepts that, as LUME's own check would
    reject it and with it every `model.get()` that includes this variable (including the one
    `lume-pva` makes for all variables on each cycle).
    """

    def validate_value(self, value: typing.Any) -> None:
        if value is not None:
            super().validate_value(value)

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return simulator.final_particles


class HyperparameterFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of the simulator's `hyparams` config object."""

    def _config(self, simulator: FBPICSimulator) -> typing.Any:
        return simulator.hyparams

    def _replace(self, simulator: FBPICSimulator, config: typing.Any) -> None:
        simulator.hyparams = config


class LaserFieldAction(_ConfigFieldAction):
    """Writable scalar mapped onto one field of one of the simulator's lasers, addressed by
    index into `simulator.lasers` (the first by default), set with `utils.evolve_laser()`.
    """

    laser_index: int = 0

    def _config(self, simulator: FBPICSimulator) -> typing.Any:
        return simulator.lasers[self.laser_index]

    def _replace(self, simulator: FBPICSimulator, config: typing.Any) -> None:
        lasers = list(simulator.lasers)
        lasers[self.laser_index] = config
        simulator.lasers = lasers

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        self._replace(
            simulator,
            utils.evolve_laser(self._config(simulator), **{self.field_name: value}),
        )


class MomentDescriptorAction(ReadOnlyActionMixin[FBPICSimulator], ScalarVariable):
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

    central_fraction: float | None = descriptor.CENTRAL_FRACTION
    feature: str
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS
    uz_min: float | None = descriptor.UZ_MIN

    def default_unit(self, simulator: FBPICSimulator) -> str | None:
        """The unit of the feature (`lume_fbpic.descriptor.features()`); None for a name that
        is not a feature of the descriptor."""
        return descriptor.features(self.longitudinal_bins).get(self.feature)

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        if simulator.final_particles is None:
            return simulator.recorded_outputs.get(self.name, float("nan"))
        return simulator.moment_descriptor(
            self.uz_min, self.central_fraction, self.longitudinal_bins
        ).get(self.feature, float("nan"))


class StatAction(ReadOnlyActionMixin[FBPICSimulator], ScalarVariable):
    """Read-only scalar sourced from `simulator.stats` (populated after a run).

    Without a `stat_name` entry, the simulator's `recorded_outputs` value for this action's name
    is used (one loaded from an archive of a reconstructed run has no statistics).
    """

    stat_name: str

    def default_unit(self, simulator: FBPICSimulator) -> str | None:
        return STAT_UNITS.get(self.stat_name)

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
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

    def _get(self, simulator: FBPICSimulator) -> typing.Any:
        return super()._get(simulator).get(self.coefficient, 0.0)

    def _set(self, simulator: FBPICSimulator, value: typing.Any) -> None:
        super()._set(simulator, {**super()._get(simulator), self.coefficient: value})


def make_beam_actions(
    simulator: FBPICSimulator, beam_index: int = 0, prefix: str | None = None
) -> list[Action]:
    """The writable actions of one particle beam of `simulator` (`simulator.beams[beam_index]`),
    by the kind of beam (`_BEAM_FIELDS`), named `<prefix>_<name>`: for example `witness_gamma`,
    `witness_sigma_r` and `witness_charge`. `prefix` defaults to the beam's `particle_name`.

    Raises:
        ValueError: For a kind of beam that has no entry in `_BEAM_FIELDS`, or with no `prefix`
            for a beam that has no `particle_name`.
    """
    b = simulator.beams[beam_index]
    if type(b) not in _BEAM_FIELDS:
        raise ValueError(f"no actions for a beam of type {type(b).__name__}")
    p = b.particle_name if prefix is None else prefix
    if p is None:
        raise ValueError(f"beam {beam_index} has no particle_name: give a prefix")
    rv: list[Action] = [
        BeamFieldAction(name=f"{p}_{name}", beam_index=beam_index, field_name=field)
        for name, field in _BEAM_FIELDS[type(b)].items()
    ]
    if "n_physical_particles" in attrs.fields_dict(type(b)):
        rv.append(BeamTotalChargeAction(name=f"{p}_charge", beam_index=beam_index))
    return rv


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


def make_stat_actions() -> list[Action]:
    """One read-only `StatAction` per statistic a run records (`lume_fbpic.units.STAT_UNITS`),
    named after it: `charge_pc`, `energy_mean_mev` and `energy_std_mev`."""
    return [
        StatAction(name=name, stat_name=name, read_only=True) for name in STAT_UNITS
    ]
