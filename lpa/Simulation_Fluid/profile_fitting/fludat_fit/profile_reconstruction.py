"""Reconstruct real ``inversion_fbpic`` profile objects from a filled config's ``kwargs``.

Shared by :mod:`~fludat_fit.generate_profile_configs` (which needs the actual
profile to recompute goodness of fit after centring it) and
:mod:`~fludat_fit.plot_profile_configs` (which needs it to draw the curve).
"""

from __future__ import annotations

from typing import Any

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.density_profiles import (
    GeneralizedLorentzianSum,
    GenericConicalTarget,
)

from .families import DEFAULT_PROFILE_KWARGS, _centered


def build_conical(kwargs: dict[str, Any]) -> GenericConicalTarget:
    """``center`` substitutes for ``start_position``, exactly as the template documents."""
    kwargs = dict(kwargs)
    center = kwargs.pop("center")
    profile = GenericConicalTarget(
        start_position=0.0, **kwargs, **DEFAULT_PROFILE_KWARGS
    )
    return _centered(profile, center)


def build_lorentzian_sum(kwargs: dict[str, Any]) -> GeneralizedLorentzianSum:
    return GeneralizedLorentzianSum(**kwargs, **DEFAULT_PROFILE_KWARGS)


PROFILE_BUILDERS = {
    "GenericConicalTarget": (build_conical, "kwargs.center"),
    "GeneralizedLorentzianSum": (build_lorentzian_sum, "centroid"),
}
"""class -> (builder, where its centre lives: a ``kwargs`` key or the spec's own key)."""


def build_profile(name: str, spec: dict[str, Any]) -> _DensityProfile | None:
    """The real profile object for one filled ``profiles[name]`` entry, or ``None``
    if its class has no builder here (e.g. ``InterpolateFromH5Profile``)."""
    builder_entry = PROFILE_BUILDERS.get(spec["class"])
    if builder_entry is None:
        return None
    builder, _ = builder_entry
    return builder(spec["kwargs"])


def center_of(spec: dict[str, Any]) -> float | None:
    """The profile's centre/centroid, wherever the schema puts it for its class.

    ``center`` (conical profiles) and ``centroid`` (``GeneralizedLorentzianSum``)
    are the same concept under two names: where the profile is centred, which is
    where the laser is focused. ``None`` for classes with neither (e.g.
    ``InterpolateFromH5Profile``).
    """
    _, location = PROFILE_BUILDERS.get(spec["class"], (None, None))
    if location is None:
        return None
    if location.startswith("kwargs."):
        return spec["kwargs"].get(location.removeprefix("kwargs."))
    return spec.get(location)
