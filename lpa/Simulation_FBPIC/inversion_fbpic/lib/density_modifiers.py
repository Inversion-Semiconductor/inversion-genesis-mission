from __future__ import annotations

from typing import ClassVar, List

from pathlib import Path
import attrs

from inversion_fbpic.lib.density_core import (
    _DensityProfile,
    _DensityModifier,
    DensityCallable,
)


@attrs.define(kw_only=True, slots=False, frozen=True)
class ModifiedDensityProfile(_DensityProfile):
    """Apply density modifiers on top of another density profile.

    Particle counts, species, and other simulation parameters are inherited from
    ``base_density_profile`` and are not serialized on this config.

    Args:
        base_density_profile: (DensityProfile) The base density profile.
        modifiers: (DensityModifier|list[DensityModifier]) The modifiers to apply to the density profile.
    """

    # Keep this string stable once used in stored JSON payloads.
    SUBCLASS: ClassVar[str] = "modified_density_profile"
    _BASE_PROFILE_ATTRIBUTE_NAMES: ClassVar[tuple[str, ...]] = (
        "nominal_density",
        "p_nz",
        "p_nr",
        "p_nt",
        "species",
        "ionization",
        "p_rmax",
        "elec_name",
        "elec_select",
        "ion_name",
        "ion_select",
    )

    # Inherited from base_density_profile at init; not serialized as parameters.
    nominal_density: float = attrs.field(init=False, default=0.0)
    p_nz: int = attrs.field(init=False, default=0)
    p_nr: int = attrs.field(init=False, default=0)
    p_nt: int = attrs.field(init=False, default=0)
    species: str | None = attrs.field(init=False, default=None)
    ionization: int | None = attrs.field(init=False, default=None)
    p_rmax: float | None = attrs.field(init=False, default=None)
    elec_name: str | None = attrs.field(init=False, default=None)
    elec_select: dict[str, list[float | None]] | None = attrs.field(
        init=False, default=None
    )
    ion_name: str | None = attrs.field(init=False, default=None)
    ion_select: dict[str, list[float | None]] | None = attrs.field(
        init=False, default=None
    )

    # Attrs fields with init=True become serialized "parameters".
    base_density_profile: _DensityProfile | Path | str | dict = attrs.field()
    modifiers: (
        _DensityModifier
        | Path
        | str
        | dict
        | List[_DensityModifier | Path | str | dict]
    ) = attrs.field(default=[])

    resolved_base_density_profile: _DensityProfile = attrs.field(init=False, repr=False)
    resolved_modifiers: list[_DensityModifier] = attrs.field(
        init=False, factory=list, repr=False
    )

    def __attrs_post_init__(self) -> None:
        if self.base_density_profile is None:
            raise ValueError("base_density_profile is required.")
        if not isinstance(
            self.base_density_profile, (_DensityProfile, Path, str, dict)
        ):
            raise ValueError(
                "base_density_profile must be a DensityProfile, Path, str, or dict."
            )

        if self.modifiers is None:
            object.__setattr__(self, "modifiers", [])
        elif isinstance(self.modifiers, (_DensityModifier, Path, str, dict)):
            object.__setattr__(self, "modifiers", [self.modifiers])
        elif isinstance(self.modifiers, list):
            for modifier in self.modifiers:
                if not isinstance(modifier, (_DensityModifier, Path, str, dict)):
                    raise ValueError(
                        "Each modifier must be a DensityModifier, Path, str, or dict."
                    )
        else:
            raise ValueError(
                "modifiers must be a DensityModifier, Path, str, dict, or list of "
                "DensityModifier, Path, str, or dict."
            )

        object.__setattr__(
            self,
            "resolved_base_density_profile",
            _DensityProfile.from_any(self.base_density_profile),
        )
        object.__setattr__(
            self,
            "resolved_modifiers",
            [_DensityModifier.from_any(modifier) for modifier in self.modifiers],
        )
        for name in self._BASE_PROFILE_ATTRIBUTE_NAMES:
            object.__setattr__(
                self, name, getattr(self.resolved_base_density_profile, name)
            )
        super().__attrs_post_init__()

    def get_z_extent(self) -> tuple[float, float] | None:
        extent = list(self.resolved_base_density_profile.get_z_extent())
        for modifier in self.resolved_modifiers:
            m_extent = modifier.get_z_extent()
            if m_extent is not None:
                if m_extent[0] is not None:
                    extent[0] = min(extent[0], m_extent[0])
                if m_extent[1] is not None:
                    extent[1] = max(extent[1], m_extent[1])
        return tuple(extent)

    def get_r_extent(self) -> float | None:
        extent = self.resolved_base_density_profile.get_r_extent()
        for modifier in self.resolved_modifiers:
            m_extent = modifier.get_r_extent()
            if m_extent is not None:
                if extent is None:
                    extent = m_extent
                else:
                    extent = max(extent, m_extent)
        return extent

    def build_density_function(self) -> DensityCallable:
        df0 = self.resolved_base_density_profile.build_density_function()
        for modifier in self.resolved_modifiers:
            df0 = modifier.modify_density_function(df0)
        return df0


@attrs.define(kw_only=True, slots=False, frozen=True)
class MatchedRadialModifier(_DensityModifier):
    """
    Add radial matched profile dependence to a z-dependent density function.

    The resulting function has the form:
    density(z, r) = n_p0(z) * (1 + r^2 / (pi * r_e * n_p0(z) * w_m^4))

    where:
    - n_p0(z) is the z-dependent density profile (z_density_func)
    - r_e is the classical electron radius
    - w_m is the matched spot size parameter, calculated using plasma density

    This radial profile represents a matched beam envelope condition where the
    radial modulation depends on the local plasma density.

    Args:
        matched_density: (float) [m^-3] Nominal plasma density in m^-3 used to scale the matched spot size parameter.
            This may be unique from the base density profile's nominal density.
        radial_extent: float | None [m] |OPTIONAL| Radial extent of the density profile. If None, the radial extent is determined by the simulation grid.
    """

    SUBCLASS: ClassVar[str] = "matched_radial_modifier"

    matched_density: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    radial_extent: float | None = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.gt(0.0))
    )

    def get_z_extent(self) -> tuple[float, float] | None:
        return None  # No longitudinal extent for this modifier.

    def get_r_extent(self) -> float | None:
        return self.radial_extent

    def modify_density_function(
        self, density_function: DensityCallable
    ) -> DensityCallable:
        try:
            from inversion_fbpic.density_profiles.ofi_channels import (
                add_radial_matched_profile,
            )
        except ImportError:
            from ofi_channels import add_radial_matched_profile  # type: ignore

        return add_radial_matched_profile(density_function, self.matched_density)
