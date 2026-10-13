"""Profile families: bounded parameterizations of ``inversion_fbpic`` density profiles.

A :class:`ProfileFamily` adapts one density-profile class to fitting. It
declares a :class:`~fludat_fit.parameters.ParameterSpace` whose bounds are
derived from the lineout being fitted, and builds the real profile object from
a parameter dictionary, so a fitted result is exactly the FBPIC configuration
that will be simulated.

Every family shares one convention: the profile's *shape* is what is
parameterized here, and its overall *amplitude* is fitted separately by the
scheme (``density = amplitude * relative_density``). Families therefore fix any
intrinsic amplitude parameter (``gauss_peak``, the first Lorentzian ``A``) to one.

``InterpolateFromH5Profile`` is deliberately absent: it reads the same density
cubes the lineouts come from. ``ExampleDensityProfile`` is absent because it is
documented as not for practical use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

import attrs
import numpy as np
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.density_profiles import (
    AsymmetricSine,
    BilateralSuperGaussian,
    GaussianPlusTriangle,
    GeneralizedGaussianPlusTriangle,
    GeneralizedLorentzianSum,
    GenericConicalTarget,
    PowerLawFlattop,
    SmoothSineFlattop,
)

from .lineout import LineoutSummary
from .parameters import ParameterSpace, ParameterSpec

DEFAULT_PROFILE_KWARGS: dict[str, Any] = {
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    # ``species=None`` with ``ionization=None`` skips the FBPIC ionization-table
    # lookup, which keeps profile construction cheap inside the optimizer loop.
    "species": None,
    "ionization": None,
}
"""Constructor arguments every ``_DensityProfile`` needs that are not fitted."""

_MIN_LENGTH_FRACTION = 1.0 / 200.0
_MAX_LENGTH_FACTOR = 2.0
_SKEW_SCALE = 5.0
"""``|skew_rate| * z_scale`` bound: density ratio ``e**5`` across the jet."""


# --------------------------------------------------------------------- scales
def _scales(summary: LineoutSummary) -> tuple[float, float, float]:
    """``(z_scale, fwhm, window_width)`` with degenerate values replaced."""
    z_scale = summary.z_scale
    if not z_scale > 0.0:
        raise ValueError("lineout has zero extent")
    fwhm = summary.fwhm if summary.fwhm > 0.0 else 0.5 * z_scale
    return z_scale, fwhm, max(summary.width, z_scale)


def _length(name: str, summary: LineoutSummary) -> ParameterSpec:
    z_scale, _, width = _scales(summary)
    return ParameterSpec(
        name,
        _MIN_LENGTH_FRACTION * z_scale,
        _MAX_LENGTH_FACTOR * width,
        log_scale=True,
        unit="m",
    )


def _position(name: str, summary: LineoutSummary) -> ParameterSpec:
    return ParameterSpec(name, summary.z_min, summary.z_max, unit="m")


def _exponent(name: str, lower: float, upper: float) -> ParameterSpec:
    return ParameterSpec(name, lower, upper, log_scale=True)


def _skew(summary: LineoutSummary) -> ParameterSpec:
    z_scale, _, _ = _scales(summary)
    return ParameterSpec(
        "skew_rate", -_SKEW_SCALE / z_scale, _SKEW_SCALE / z_scale, unit="m^-1"
    )


# --------------------------------------------------------------------- family
class ProfileFamily(ABC):
    """One parameterized density-profile class prepared for fitting.

    Subclasses set :attr:`name` and :attr:`profile_class`, and implement
    :meth:`parameter_space` and :meth:`build`. The default
    :meth:`relative_density` builds the profile and evaluates its density
    function, which guarantees the fitted curve is what FBPIC will see.

    Args:
        profile_kwargs: Overrides for :data:`DEFAULT_PROFILE_KWARGS` applied to
            every profile this family builds (``p_nz``, ``species``, ...).
    """

    name: ClassVar[str]
    profile_class: ClassVar[type[_DensityProfile]]

    def __init__(self, profile_kwargs: Mapping[str, Any] | None = None) -> None:
        self.profile_kwargs: dict[str, Any] = {
            **DEFAULT_PROFILE_KWARGS,
            **(profile_kwargs or {}),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    @abstractmethod
    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        """Bounded shape parameters for a lineout of the given scale."""

    @abstractmethod
    def build(
        self,
        parameters: Mapping[str, float],
        *,
        nominal_density: float = 1.0,
        **overrides: Any,
    ) -> _DensityProfile:
        """Construct the profile for ``parameters`` (keyed by parameter name).

        ``nominal_density`` is in m^-3 as the profile classes expect;
        ``overrides`` replace entries of :attr:`profile_kwargs`.
        """

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float] | None:
        """A heuristic starting point, or ``None`` to rely on sampled starts only."""
        return None

    def relative_density(
        self, parameters: Mapping[str, float], z: np.ndarray
    ) -> np.ndarray:
        """Evaluate the (unit-amplitude) profile on ``z`` [m]."""
        z = np.asarray(z, dtype=np.float64)
        profile = self.build(parameters)
        return np.asarray(
            profile.build_density_function()(z, np.zeros_like(z)), dtype=np.float64
        )

    def _kwargs(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        return {**self.profile_kwargs, **overrides}


def _centered(profile: _DensityProfile, center: float) -> _DensityProfile:
    """Shift a ``_FiniteSkewedProfile`` built at ``start_position=0`` to ``center``.

    ``evolve`` re-runs the constructor with the resolved ``ionization`` (``0``);
    with no species that would trigger the base class's Hydrogen fallback and an
    FBPIC table lookup, so ``ionization=None`` is restored in that case.
    """
    changes: dict[str, Any] = {"start_position": center - profile.centroid}
    if profile.species is None:
        changes["ionization"] = None
    return attrs.evolve(profile, **changes)


# ------------------------------------------------------------ simple families
class AsymmetricSineFamily(ProfileFamily):
    name = "asymmetric_sine"
    profile_class = AsymmetricSine

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        return ParameterSpace(
            (
                _position("peak_z0", summary),
                _length("upramp_length", summary),
                _length("downramp_length", summary),
            )
        )

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        return {
            "peak_z0": summary.z_peak,
            "upramp_length": fwhm,
            "downramp_length": fwhm,
        }

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        return AsymmetricSine(
            peak_z0=parameters["peak_z0"],
            upramp_length=parameters["upramp_length"],
            downramp_length=parameters["downramp_length"],
            nominal_density=nominal_density,
            **self._kwargs(overrides),
        )


class SmoothSineFlattopFamily(ProfileFamily):
    """``SmoothSineFlattop`` parameterized by the flattop centre instead of its offset."""

    name = "smooth_sine_flattop"
    profile_class = SmoothSineFlattop

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        _, _, width = _scales(summary)
        return ParameterSpace(
            (
                _position("center", summary),
                ParameterSpec(
                    "flattop_width", 0.0, _MAX_LENGTH_FACTOR * width, unit="m"
                ),
                _length("upramp_length", summary),
                _length("downramp_length", summary),
            )
        )

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        return {
            "center": summary.centroid,
            "flattop_width": 0.5 * fwhm,
            "upramp_length": 0.5 * fwhm,
            "downramp_length": 0.5 * fwhm,
        }

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        offset = (
            parameters["center"]
            - 0.5 * parameters["flattop_width"]
            - parameters["upramp_length"]
        )
        return SmoothSineFlattop(
            flattop_width=parameters["flattop_width"],
            upramp_length=parameters["upramp_length"],
            downramp_length=parameters["downramp_length"],
            offset_length=offset,
            nominal_density=nominal_density,
            **self._kwargs(overrides),
        )


class GaussianPlusTriangleFamily(ProfileFamily):
    name = "gaussian_plus_triangle"
    profile_class = GaussianPlusTriangle

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        return ParameterSpace(
            (
                _position("gauss_z0", summary),
                _length("gauss_sigma", summary),
                _position("tri_z0", summary),
                _length("tri_left_width", summary),
                _length("tri_right_width", summary),
                _exponent("tri_height", 1.0e-3, 5.0),
            )
        )

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        return {
            "gauss_z0": summary.centroid,
            "gauss_sigma": fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))),
            "tri_z0": summary.z_peak,
            "tri_left_width": 0.5 * fwhm,
            "tri_right_width": 0.5 * fwhm,
            "tri_height": 0.5,
        }

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        return GaussianPlusTriangle(
            gauss_sigma=parameters["gauss_sigma"],
            gauss_z0=parameters["gauss_z0"],
            tri_z0=parameters["tri_z0"],
            tri_left_width=parameters["tri_left_width"],
            tri_right_width=parameters["tri_right_width"],
            tri_height=parameters["tri_height"],
            nominal_density=nominal_density,
            **self._kwargs(overrides),
        )


class GeneralizedGaussianPlusTriangleFamily(ProfileFamily):
    """``GeneralizedGaussianPlusTriangle`` with ``gauss_peak`` fixed to one."""

    name = "generalized_gaussian_plus_triangle"
    profile_class = GeneralizedGaussianPlusTriangle

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        return ParameterSpace(
            (
                _position("gauss_z0", summary),
                _length("gauss_alpha", summary),
                _exponent("gauss_beta", 0.5, 12.0),
                _position("tri_z0", summary),
                _length("tri_left_width", summary),
                _length("tri_right_width", summary),
                _exponent("tri_height", 1.0e-3, 5.0),
            )
        )

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        return {
            "gauss_z0": summary.centroid,
            "gauss_alpha": fwhm / (2.0 * np.sqrt(np.log(2.0))),
            "gauss_beta": 2.0,
            "tri_z0": summary.z_peak,
            "tri_left_width": 0.5 * fwhm,
            "tri_right_width": 0.5 * fwhm,
            "tri_height": 0.5,
        }

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        return GeneralizedGaussianPlusTriangle(
            gauss_peak=1.0,
            gauss_alpha=parameters["gauss_alpha"],
            gauss_beta=parameters["gauss_beta"],
            gauss_z0=parameters["gauss_z0"],
            tri_z0=parameters["tri_z0"],
            tri_left_width=parameters["tri_left_width"],
            tri_right_width=parameters["tri_right_width"],
            tri_height=parameters["tri_height"],
            nominal_density=nominal_density,
            **self._kwargs(overrides),
        )


class BilateralSuperGaussianFamily(ProfileFamily):
    """``BilateralSuperGaussian``: independent width and shape exponent on each
    side of the center. Its peak is exactly 1.0 by construction (both sides equal
    1.0 at the center), so there is no intrinsic amplitude field to fix, unlike
    ``GeneralizedGaussianPlusTriangleFamily``'s ``gauss_peak``."""

    name = "bilateral_supergaussian"
    profile_class = BilateralSuperGaussian

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        return ParameterSpace(
            (
                _position("center", summary),
                _length("fwhm_left", summary),
                _exponent("beta_left", 0.5, 12.0),
                _length("fwhm_right", summary),
                _exponent("beta_right", 0.5, 12.0),
            )
        )

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        return {
            "center": summary.z_peak,
            "fwhm_left": fwhm,
            "beta_left": 2.0,
            "fwhm_right": fwhm,
            "beta_right": 2.0,
        }

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        return BilateralSuperGaussian(
            center=parameters["center"],
            fwhm_left=parameters["fwhm_left"],
            beta_left=parameters["beta_left"],
            fwhm_right=parameters["fwhm_right"],
            beta_right=parameters["beta_right"],
            nominal_density=nominal_density,
            **self._kwargs(overrides),
        )


class GeneralizedLorentzianSumFamily(ProfileFamily):
    """Sum of ``n_terms`` generalized Lorentzians.

    The first term has amplitude one; further terms have relative amplitudes
    ``A_i`` in ``[-1, 1]`` so that dips (for example a shock) can be modelled.
    ``nominal_density`` scales every amplitude, because the profile class takes
    its nominal density from the largest ``A``.
    """

    profile_class = GeneralizedLorentzianSum
    _CENTER_SPREAD = 0.25

    def __init__(
        self,
        n_terms: int = 1,
        profile_kwargs: Mapping[str, Any] | None = None,
        density_cutoff_ratio: float = 5.0e-3,
    ) -> None:
        super().__init__(profile_kwargs)
        if n_terms < 1:
            raise ValueError("n_terms must be at least one")
        self.n_terms = int(n_terms)
        self.density_cutoff_ratio = float(density_cutoff_ratio)
        self.name = f"generalized_lorentzian_sum[{self.n_terms}]"

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        specs: list[ParameterSpec] = []
        for i in range(self.n_terms):
            if i > 0:
                specs.append(ParameterSpec(f"A_{i}", -1.0, 1.0))
            specs.extend(
                (
                    _position(f"c_{i}", summary),
                    _length(f"w_{i}", summary),
                    _exponent(f"b_{i}", 1.0, 10.0),
                    _exponent(f"m_{i}", 0.25, 10.0),
                )
            )
        return ParameterSpace(tuple(specs))

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        z_scale, fwhm, _ = _scales(summary)
        guess: dict[str, float] = {
            "c_0": summary.centroid,
            "w_0": 0.5 * fwhm,
            "b_0": 2.0,
            "m_0": 1.0,
        }
        for i in range(1, self.n_terms):
            side = 1.0 if i % 2 else -1.0
            guess.update(
                {
                    f"A_{i}": 0.2,
                    f"c_{i}": summary.centroid
                    + side * self._CENTER_SPREAD * z_scale * ((i + 1) // 2),
                    f"w_{i}": 0.25 * fwhm,
                    f"b_{i}": 2.0,
                    f"m_{i}": 1.0,
                }
            )
        return guess

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        terms = []
        for i in range(self.n_terms):
            amplitude = 1.0 if i == 0 else parameters[f"A_{i}"]
            terms.append(
                {
                    "A": nominal_density * amplitude,
                    "c": parameters[f"c_{i}"],
                    "w": parameters[f"w_{i}"],
                    "b": parameters[f"b_{i}"],
                    "m": parameters[f"m_{i}"],
                }
            )
        return GeneralizedLorentzianSum(
            parameters=terms,
            density_cutoff_ratio=self.density_cutoff_ratio,
            **self._kwargs(overrides),
        )


class PowerLawFlattopFamily(ProfileFamily):
    """``PowerLawFlattop`` parameterized by its centre, with optional skew."""

    name = "power_law_flattop"
    profile_class = PowerLawFlattop

    def __init__(
        self, profile_kwargs: Mapping[str, Any] | None = None, fit_skew: bool = True
    ) -> None:
        super().__init__(profile_kwargs)
        self.fit_skew = bool(fit_skew)

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        specs = [
            _position("center", summary),
            _length("flattop_width", summary),
            _length("transition_length", summary),
            _exponent("transition_exponent", 2.05, 20.0),
        ]
        if self.fit_skew:
            specs.append(_skew(summary))
        return ParameterSpace(tuple(specs))

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        guess = {
            "center": summary.centroid,
            "flattop_width": 0.5 * fwhm,
            "transition_length": 0.5 * fwhm,
            "transition_exponent": 3.0,
        }
        if self.fit_skew:
            guess["skew_rate"] = 0.0
        return guess

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        profile = PowerLawFlattop(
            start_position=0.0,
            flattop_width=parameters["flattop_width"],
            transition_length=parameters["transition_length"],
            transition_exponent=parameters["transition_exponent"],
            skew_rate=parameters.get("skew_rate", 0.0),
            nominal_density=nominal_density,
            **self._kwargs(overrides),
        )
        return _centered(profile, parameters["center"])


# ---------------------------------------------------------- conical targets
MainProfileType = Literal[
    "supergaussian", "cosine_squared_flattop", "power_law_flattop", "lorentzian_flattop"
]
FringeProfileType = Literal[
    "supergaussian", "cosine_squared", "power_law", "lorentzian"
]
MAIN_PROFILE_TYPES: tuple[str, ...] = (
    "supergaussian",
    "cosine_squared_flattop",
    "power_law_flattop",
    "lorentzian_flattop",
)
FRINGE_PROFILE_TYPES: tuple[str, ...] = (
    "supergaussian",
    "cosine_squared",
    "power_law",
    "lorentzian",
)


class GenericConicalTargetFamily(ProfileFamily):
    """``GenericConicalTarget`` for one choice of main and fringe profile type.

    The main profile is parameterized by its centre. The cosine-squared flattop's
    ``ramp_length`` becomes ``ramp_fraction * fwhm`` so the class constraint
    ``ramp_length <= fwhm`` is a simple bound. Fringe parameters carry a
    ``fringe_`` prefix.

    Args:
        main_profile_type: One of :data:`MAIN_PROFILE_TYPES`.
        fringe_profile_type: One of :data:`FRINGE_PROFILE_TYPES`, or ``None``.
        fringe_side: ``"left"``, ``"right"`` or ``"both"``.
        fit_skew: Whether ``skew_rate`` is a fit parameter (else zero).
        profile_kwargs: See :class:`ProfileFamily`.
    """

    profile_class = GenericConicalTarget

    def __init__(
        self,
        main_profile_type: MainProfileType,
        fringe_profile_type: FringeProfileType | None = None,
        *,
        fringe_side: Literal["left", "right", "both"] = "both",
        fit_skew: bool = True,
        profile_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(profile_kwargs)
        if main_profile_type not in MAIN_PROFILE_TYPES:
            raise ValueError(f"unknown main_profile_type {main_profile_type!r}")
        if (
            fringe_profile_type is not None
            and fringe_profile_type not in FRINGE_PROFILE_TYPES
        ):
            raise ValueError(f"unknown fringe_profile_type {fringe_profile_type!r}")
        if fringe_side not in ("left", "right", "both"):
            raise ValueError(f"unknown fringe_side {fringe_side!r}")
        self.main_profile_type = main_profile_type
        self.fringe_profile_type = fringe_profile_type
        self.fringe_side = fringe_side
        self.fit_skew = bool(fit_skew)
        self.name = f"conical:{main_profile_type}"
        if fringe_profile_type is not None:
            self.name += f"+{fringe_profile_type}"
            if fringe_side != "both":
                self.name += f"_{fringe_side}"

    # -- parameter space ---------------------------------------------------
    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        specs = [_position("center", summary)]
        specs.extend(self._main_specs(summary))
        if self.fringe_profile_type is not None:
            specs.extend(self._fringe_specs(summary))
        if self.fit_skew:
            specs.append(_skew(summary))
        return ParameterSpace(tuple(specs))

    def _main_specs(self, summary: LineoutSummary) -> list[ParameterSpec]:
        kind = self.main_profile_type
        if kind == "supergaussian":
            return [_length("fwhm", summary), _exponent("beta", 0.5, 12.0)]
        if kind == "cosine_squared_flattop":
            return [
                _length("fwhm", summary),
                ParameterSpec("ramp_fraction", 0.01, 1.0),
            ]
        if kind == "power_law_flattop":
            return [
                _length("flattop_width", summary),
                _length("transition_length", summary),
                _exponent("transition_exponent", 2.05, 20.0),
            ]
        return [  # lorentzian_flattop
            _length("flattop_width", summary),
            _length("transition_length", summary),
            _exponent("coordinate_exponent", 2.05, 20.0),
            _exponent("profile_exponent", 0.25, 10.0),
        ]

    def _fringe_specs(self, summary: LineoutSummary) -> list[ParameterSpec]:
        specs = [
            _length("fringe_center_offset", summary),
            ParameterSpec("fringe_relative_height", 0.0, 2.0),
        ]
        kind = self.fringe_profile_type
        if kind == "supergaussian":
            specs += [
                _length("fringe_fwhm", summary),
                _exponent("fringe_beta", 0.5, 12.0),
            ]
        elif kind == "cosine_squared":
            specs += [_length("fringe_fwhm", summary)]
        elif kind == "power_law":
            specs += [
                _length("fringe_transition_length", summary),
                _exponent("fringe_transition_exponent", 2.05, 20.0),
            ]
        else:  # lorentzian
            specs += [
                _length("fringe_transition_length", summary),
                _exponent("fringe_coordinate_exponent", 2.05, 20.0),
                _exponent("fringe_profile_exponent", 0.25, 10.0),
            ]
        return specs

    # -- initial guess -----------------------------------------------------
    def initial_guess(self, summary: LineoutSummary) -> dict[str, float]:
        _, fwhm, _ = _scales(summary)
        guess: dict[str, float] = {"center": summary.centroid}
        kind = self.main_profile_type
        if kind == "supergaussian":
            guess.update(fwhm=fwhm, beta=2.0)
        elif kind == "cosine_squared_flattop":
            guess.update(fwhm=fwhm, ramp_fraction=0.5)
        elif kind == "power_law_flattop":
            guess.update(
                flattop_width=0.5 * fwhm,
                transition_length=0.5 * fwhm,
                transition_exponent=3.0,
            )
        else:
            guess.update(
                flattop_width=0.5 * fwhm,
                transition_length=0.5 * fwhm,
                coordinate_exponent=3.0,
                profile_exponent=1.0,
            )
        fringe = self.fringe_profile_type
        if fringe is not None:
            guess.update(fringe_center_offset=0.75 * fwhm, fringe_relative_height=0.2)
            if fringe == "supergaussian":
                guess.update(fringe_fwhm=0.25 * fwhm, fringe_beta=2.0)
            elif fringe == "cosine_squared":
                guess.update(fringe_fwhm=0.25 * fwhm)
            elif fringe == "power_law":
                guess.update(
                    fringe_transition_length=0.25 * fwhm, fringe_transition_exponent=3.0
                )
            else:
                guess.update(
                    fringe_transition_length=0.25 * fwhm,
                    fringe_coordinate_exponent=3.0,
                    fringe_profile_exponent=1.0,
                )
        if self.fit_skew:
            guess["skew_rate"] = 0.0
        return guess

    # -- build -------------------------------------------------------------
    def _main_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        kind = self.main_profile_type
        if kind == "supergaussian":
            return {"fwhm": parameters["fwhm"], "beta": parameters["beta"]}
        if kind == "cosine_squared_flattop":
            return {
                "fwhm": parameters["fwhm"],
                "ramp_length": parameters["ramp_fraction"] * parameters["fwhm"],
            }
        if kind == "power_law_flattop":
            return {
                "flattop_width": parameters["flattop_width"],
                "transition_length": parameters["transition_length"],
                "transition_exponent": parameters["transition_exponent"],
            }
        return {
            "flattop_width": parameters["flattop_width"],
            "transition_length": parameters["transition_length"],
            "coordinate_exponent": parameters["coordinate_exponent"],
            "profile_exponent": parameters["profile_exponent"],
        }

    def _fringe_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        kind = self.fringe_profile_type
        if kind == "supergaussian":
            return {
                "fwhm": parameters["fringe_fwhm"],
                "beta": parameters["fringe_beta"],
            }
        if kind == "cosine_squared":
            return {"fwhm": parameters["fringe_fwhm"]}
        if kind == "power_law":
            return {
                "transition_length": parameters["fringe_transition_length"],
                "transition_exponent": parameters["fringe_transition_exponent"],
            }
        return {
            "transition_length": parameters["fringe_transition_length"],
            "coordinate_exponent": parameters["fringe_coordinate_exponent"],
            "profile_exponent": parameters["fringe_profile_exponent"],
        }

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        kwargs: dict[str, Any] = {
            "start_position": 0.0,
            "main_profile_type": self.main_profile_type,
            "main_profile_parameters": self._main_parameters(parameters),
            "skew_rate": parameters.get("skew_rate", 0.0),
            "nominal_density": nominal_density,
        }
        if self.fringe_profile_type is not None:
            kwargs.update(
                fringe_profile_type=self.fringe_profile_type,
                fringe_profile_parameters=self._fringe_parameters(parameters),
                fringe_center_offset=parameters["fringe_center_offset"],
                fringe_relative_height=parameters["fringe_relative_height"],
                fringe_side=self.fringe_side,
            )
        profile = GenericConicalTarget(**kwargs, **self._kwargs(overrides))
        return _centered(profile, parameters["center"])


# ------------------------------------------------------------------ registry
def family_registry(
    profile_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, ProfileFamily]:
    """The default set of families, keyed by :attr:`ProfileFamily.name`.

    Further ``GenericConicalTargetFamily`` main/fringe combinations and larger
    Lorentzian sums can be constructed directly and passed to the fitting
    functions alongside these.
    """
    families: list[ProfileFamily] = [
        AsymmetricSineFamily(profile_kwargs),
        SmoothSineFlattopFamily(profile_kwargs),
        GaussianPlusTriangleFamily(profile_kwargs),
        GeneralizedGaussianPlusTriangleFamily(profile_kwargs),
        BilateralSuperGaussianFamily(profile_kwargs),
        GeneralizedLorentzianSumFamily(1, profile_kwargs),
        GeneralizedLorentzianSumFamily(2, profile_kwargs),
        GeneralizedLorentzianSumFamily(3, profile_kwargs),
        PowerLawFlattopFamily(profile_kwargs),
        *(
            GenericConicalTargetFamily(main, profile_kwargs=profile_kwargs)
            for main in MAIN_PROFILE_TYPES
        ),
        GenericConicalTargetFamily(
            "supergaussian", "cosine_squared", profile_kwargs=profile_kwargs
        ),
        GenericConicalTargetFamily(
            "lorentzian_flattop", "lorentzian", profile_kwargs=profile_kwargs
        ),
    ]
    return {family.name: family for family in families}


DEFAULT_FAMILY_NAMES: tuple[str, ...] = tuple(family_registry())
"""Names accepted by :func:`get_families` and the command line."""


def get_families(
    names: Iterable[str] | None = None,
    profile_kwargs: Mapping[str, Any] | None = None,
) -> list[ProfileFamily]:
    """Resolve family names against :func:`family_registry`; ``None`` gives all."""
    registry = family_registry(profile_kwargs)
    if names is None:
        return list(registry.values())
    families = []
    for name in names:
        if name not in registry:
            raise KeyError(
                f"unknown profile family {name!r}; choose from {list(registry)}"
            )
        families.append(registry[name])
    return families


class FixedParameterFamily(ProfileFamily):
    """Wrap a family with one of its parameters pinned to a fixed value.

    The named parameter is removed from the searched
    :class:`~fludat_fit.parameters.ParameterSpace` and merged back in at
    ``fixed_value`` whenever the wrapped family builds a profile or evaluates
    its relative density, so the local optimiser never sees or moves it: every
    *other* parameter is still fitted normally, free to compensate (a
    different ``skew_rate``, a different satellite-term amplitude, ...) for
    the fixed one.

    Useful when a parameter's correct value is known externally and should
    not vary per fit — for example, pinning every family's position parameter
    (``center`` on ``GenericConicalTarget``/``PowerLawFlattop``, ``c_0`` on
    ``GeneralizedLorentzianSum``) to one shared, data-driven centroid, so a
    set of otherwise-independent fits to the same lineout agree on a physical
    anchor point (e.g. the laser focus) instead of each fitting it on its own.
    This is a genuine constraint enforced during optimisation, not a
    translation applied after the fact: a rigid shift after fitting freely
    would preserve fit quality perfectly only for shapes with no other
    position-coupled parameters, but skewed or multi-term shapes can fit
    markedly worse once rigidly moved away from where they were fitted.

    Args:
        family: The family to wrap.
        parameter_name: One of ``family``'s parameter names.
        fixed_value: The value that parameter is held at.
    """

    def __init__(
        self, family: ProfileFamily, parameter_name: str, fixed_value: float
    ) -> None:
        super().__init__(family.profile_kwargs)
        self.family = family
        self.parameter_name = parameter_name
        self.fixed_value = float(fixed_value)
        self.name = f"{family.name}[{parameter_name}={fixed_value:.4g}]"
        self.profile_class = family.profile_class

    def parameter_space(self, summary: LineoutSummary) -> ParameterSpace:
        space = self.family.parameter_space(summary)
        if self.parameter_name not in space.names:
            raise KeyError(
                f"{self.family.name!r} has no parameter {self.parameter_name!r} to fix"
            )
        specs = tuple(spec for spec in space.specs if spec.name != self.parameter_name)
        return ParameterSpace(specs)

    def _with_fixed(self, parameters: Mapping[str, float]) -> dict[str, float]:
        return {**parameters, self.parameter_name: self.fixed_value}

    def relative_density(
        self, parameters: Mapping[str, float], z: np.ndarray
    ) -> np.ndarray:
        return self.family.relative_density(self._with_fixed(parameters), z)

    def build(
        self,
        parameters: Mapping[str, float],
        *,
        nominal_density: float = 1.0,
        **overrides: Any,
    ) -> _DensityProfile:
        return self.family.build(
            self._with_fixed(parameters), nominal_density=nominal_density, **overrides
        )

    def initial_guess(self, summary: LineoutSummary) -> dict[str, float] | None:
        guess = self.family.initial_guess(summary)
        if guess is None:
            return None
        return {
            key: value for key, value in guess.items() if key != self.parameter_name
        }
