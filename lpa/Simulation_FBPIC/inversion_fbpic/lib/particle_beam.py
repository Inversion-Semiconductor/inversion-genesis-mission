"""Serializable particle-beam configurations backed by FBPIC bunch initializers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import attrs
import numpy.typing as npt
from scipy.constants import e, m_e

from fbpic.lpa_utils.bunch import (
    add_particle_bunch,
    add_particle_bunch_file,
    add_particle_bunch_gaussian,
    add_particle_bunch_openPMD,
)
from inversion_fbpic.lib.math_expression import MathExpression
from inversion_fbpic.lib.serializable_config import SerializableConfig

if TYPE_CHECKING:
    from fbpic.lpa_utils.boosted_frame import BoostConverter
    from fbpic.main import Simulation as FBPICSimulation
    from fbpic.particles.particles import Particles


def _optional_math_expression(value: Any) -> MathExpression | None:
    if value is None:
        return None
    return MathExpression.from_any(value)


def _density_function(expression: MathExpression | None):
    """Adapt one- or two-dimensional expressions to FBPIC's ``dens_func(z, r)``."""
    if expression is None:
        return None
    if not set(expression.variables).issubset({"z", "r"}):
        raise ValueError("dens_func variables must be drawn from ('z', 'r').")

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        coordinates = {"z": z, "r": r}
        return expression(*(coordinates[name] for name in expression.variables))

    return dens_func


@attrs.define(kw_only=True, slots=False, frozen=True)
class _ParticleBeam(SerializableConfig):
    """Base class for serializable particle-beam configurations.

    Args:
        q: (float) [C] |OPTIONAL| Particle charge. Defaults to the electron charge.
        m: (float) [kg] |OPTIONAL| Particle mass. Defaults to the electron mass.
        z_injection_plane: (float|None) [m] |OPTIONAL| Optional lab-frame ballistic-injection plane. Defaults to None.
        initialize_self_field: (bool) |OPTIONAL| Whether FBPIC initializes the bunch space-charge fields. Defaults to True.
        particle_name: (str|None) |OPTIONAL| Species name for FBPIC particle diagnostics. Defaults to None.
        particle_select: (dict[str, list[float|None]]|None) |OPTIONAL| Particle-diagnostic filters. Defaults to None.
    """

    CONFIG_TYPE: ClassVar[str] = "particle_beam"
    _CONCRETE_REGISTRY: ClassVar[dict[str, type["_ParticleBeam"]]] = {}

    q: float = attrs.field(default=-e, converter=float)
    m: float = attrs.field(
        default=m_e, converter=float, validator=attrs.validators.gt(0.0)
    )
    z_injection_plane: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float)
    )
    initialize_self_field: bool = attrs.field(default=True, converter=bool)
    particle_name: str | None = attrs.field(default=None)
    particle_select: dict[str, list[float | None]] | None = attrs.field(default=None)

    def add_to_simulation(
        self, simulation: "FBPICSimulation", *, boost: "BoostConverter | None" = None
    ) -> "Particles":
        """Add this beam to an initialized FBPIC simulation."""
        raise NotImplementedError


@attrs.define(kw_only=True, slots=False, frozen=True)
class ExpressionParticleBeam(_ParticleBeam):
    """A grid-initialized particle beam via FBPIC ``add_particle_bunch``.

    Args:
        gamma0: (float) Lorentz factor of the particles.
        n: (float) [m^-3] Bunch density.
        p_zmin: (float) [m] Lower longitudinal particle bound.
        p_zmax: (float) [m] Upper longitudinal particle bound.
        p_rmin: (float) [m] |OPTIONAL| Lower radial particle bound. Defaults to 0.0.
        p_rmax: (float) [m] Upper radial particle bound.
        p_nr: (int) |OPTIONAL| Number of macroparticles per cell radially. Defaults to 2.
        p_nz: (int) |OPTIONAL| Number of macroparticles per cell longitudinally. Defaults to 2.
        p_nt: (int) |OPTIONAL| Number of macroparticles per cell azimuthally. Defaults to 4.
        dens_func: (MathExpression|None) |OPTIONAL| Relative density function in ``z``, ``r``, or both. Defaults to None.
        direction: (Literal["forward", "backward"]) |OPTIONAL| Beam propagation direction. Defaults to ``"forward"``.
        boost_positions_in_dens_func: (bool) |OPTIONAL| Whether FBPIC boosts positions passed to ``dens_func``. Defaults to False.
    """

    SUBCLASS: ClassVar[str] = "expression"

    gamma0: float = attrs.field(converter=float, validator=attrs.validators.ge(1.0))
    n: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    p_zmin: float = attrs.field(converter=float)
    p_zmax: float = attrs.field(converter=float)
    p_rmin: float = attrs.field(
        default=0.0, converter=float, validator=attrs.validators.ge(0.0)
    )
    p_rmax: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    p_nr: int = attrs.field(default=2, converter=int, validator=attrs.validators.ge(1))
    p_nz: int = attrs.field(default=2, converter=int, validator=attrs.validators.ge(1))
    p_nt: int = attrs.field(default=4, converter=int, validator=attrs.validators.ge(1))
    dens_func: MathExpression | None = attrs.field(
        default=None, converter=_optional_math_expression
    )
    direction: Literal["forward", "backward"] = attrs.field(default="forward")
    boost_positions_in_dens_func: bool = attrs.field(default=False, converter=bool)

    def __attrs_post_init__(self) -> None:
        if not self.p_zmin < self.p_zmax:
            raise ValueError("p_zmin must be less than p_zmax.")
        if not self.p_rmin < self.p_rmax:
            raise ValueError("p_rmin must be less than p_rmax.")
        if self.direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'.")
        _density_function(self.dens_func)

    def add_to_simulation(
        self, simulation: "FBPICSimulation", *, boost: "BoostConverter | None" = None
    ) -> "Particles":
        return add_particle_bunch(
            simulation,
            q=self.q,
            m=self.m,
            gamma0=self.gamma0,
            n=self.n,
            p_zmin=self.p_zmin,
            p_zmax=self.p_zmax,
            p_rmin=self.p_rmin,
            p_rmax=self.p_rmax,
            p_nr=self.p_nr,
            p_nz=self.p_nz,
            p_nt=self.p_nt,
            dens_func=_density_function(self.dens_func),
            boost=boost,
            direction=self.direction,
            z_injection_plane=self.z_injection_plane,
            initialize_self_field=self.initialize_self_field,
            boost_positions_in_dens_func=self.boost_positions_in_dens_func,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class GaussianParticleBeam(_ParticleBeam):
    """A Gaussian particle beam via FBPIC ``add_particle_bunch_gaussian``.

    Args:
        sig_r: (float) [m] Transverse RMS bunch size.
        sig_z: (float) [m] Longitudinal RMS bunch size.
        n_emit: (float) [m] Normalized bunch emittance.
        gamma0: (float) Lorentz factor of the particles.
        sig_gamma: (float) Absolute energy spread.
        n_physical_particles: (float) Number of physical particles in the bunch.
        n_macroparticles: (int) Number of macroparticles.
        tf: (float) [s] |OPTIONAL| Time at which the bunch reaches focus. Defaults to 0.0.
        zf: (float) [m] |OPTIONAL| Focus position. Defaults to 0.0.
        save_beam: (str|None) |OPTIONAL| Path stem for FBPIC's generated ``.npz`` distribution. Defaults to None.
        symmetrize: (bool) |OPTIONAL| Whether to use four-fold rotational symmetry. Defaults to False.
    """

    SUBCLASS: ClassVar[str] = "gaussian"

    sig_r: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    sig_z: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    n_emit: float = attrs.field(converter=float, validator=attrs.validators.ge(0.0))
    gamma0: float = attrs.field(converter=float, validator=attrs.validators.ge(1.0))
    sig_gamma: float = attrs.field(converter=float, validator=attrs.validators.ge(0.0))
    n_physical_particles: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    n_macroparticles: int = attrs.field(converter=int, validator=attrs.validators.ge(1))
    tf: float = attrs.field(default=0.0, converter=float)
    zf: float = attrs.field(default=0.0, converter=float)
    save_beam: str | None = attrs.field(default=None)
    symmetrize: bool = attrs.field(default=False, converter=bool)

    def add_to_simulation(
        self, simulation: "FBPICSimulation", *, boost: "BoostConverter | None" = None
    ) -> "Particles":
        return add_particle_bunch_gaussian(
            simulation,
            q=self.q,
            m=self.m,
            sig_r=self.sig_r,
            sig_z=self.sig_z,
            n_emit=self.n_emit,
            gamma0=self.gamma0,
            sig_gamma=self.sig_gamma,
            n_physical_particles=self.n_physical_particles,
            n_macroparticles=self.n_macroparticles,
            tf=self.tf,
            zf=self.zf,
            boost=boost,
            save_beam=self.save_beam,
            z_injection_plane=self.z_injection_plane,
            initialize_self_field=self.initialize_self_field,
            symmetrize=self.symmetrize,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class OpenPMDParticleBeam(_ParticleBeam):
    """An openPMD particle beam via FBPIC ``add_particle_bunch_openPMD``.

    Args:
        ts_path: (Path) Path to the openPMD time-series directory or file.
        z_off: (float) [m] |OPTIONAL| Longitudinal offset. Defaults to 0.0.
        species: (str|None) |OPTIONAL| openPMD species name. Defaults to None.
        select: (dict[str, Any]|None) |OPTIONAL| openPMD particle-selection mapping. Defaults to None.
        iteration: (int|None) |OPTIONAL| openPMD iteration. Defaults to None.
    """

    SUBCLASS: ClassVar[str] = "openpmd"

    ts_path: Path = attrs.field(
        converter=Path,
        metadata={"input_path": True, "input_path_directory": True},
    )
    z_off: float = attrs.field(default=0.0, converter=float)
    species: str | None = attrs.field(default=None)
    select: dict[str, Any] | None = attrs.field(default=None)
    iteration: int | None = attrs.field(
        default=None, converter=attrs.converters.optional(int)
    )

    def add_to_simulation(
        self, simulation: "FBPICSimulation", *, boost: "BoostConverter | None" = None
    ) -> "Particles":
        return add_particle_bunch_openPMD(
            simulation,
            q=self.q,
            m=self.m,
            ts_path=str(self.ts_path),
            z_off=self.z_off,
            species=self.species,
            select=self.select,
            iteration=self.iteration,
            boost=boost,
            z_injection_plane=self.z_injection_plane,
            initialize_self_field=self.initialize_self_field,
        )


@attrs.define(kw_only=True, slots=False, frozen=True)
class FileParticleBeam(_ParticleBeam):
    """A particle beam loaded from an FBPIC ``.npz`` file.

    Args:
        filename: (Path) Path to the FBPIC particle-distribution file.
        n_physical_particles: (float) Number of physical particles represented by the file.
        z_off: (float) [m] |OPTIONAL| Longitudinal offset. Defaults to 0.0.
        direction: (Literal["forward", "backward"]) |OPTIONAL| Beam propagation direction. Defaults to ``"forward"``.
    """

    SUBCLASS: ClassVar[str] = "file"

    filename: Path = attrs.field(converter=Path, metadata={"input_path": True})
    n_physical_particles: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    z_off: float = attrs.field(default=0.0, converter=float)
    direction: Literal["forward", "backward"] = attrs.field(default="forward")

    def __attrs_post_init__(self) -> None:
        if self.direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'.")

    def add_to_simulation(
        self, simulation: "FBPICSimulation", *, boost: "BoostConverter | None" = None
    ) -> "Particles":
        return add_particle_bunch_file(
            simulation,
            q=self.q,
            m=self.m,
            filename=str(self.filename),
            n_physical_particles=self.n_physical_particles,
            z_off=self.z_off,
            boost=boost,
            direction=self.direction,
            z_injection_plane=self.z_injection_plane,
            initialize_self_field=self.initialize_self_field,
        )
