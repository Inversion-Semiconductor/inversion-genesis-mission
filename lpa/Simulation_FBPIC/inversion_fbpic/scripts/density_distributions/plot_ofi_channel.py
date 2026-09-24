"""
Generates and plots density profiles for OFI channel examples.

This script demonstrates usage of the density profile helpers in ofi_channels.py by
plotting several example profiles.

Usage:
    python plot_ofi_channel.py [case]
    python plot_ofi_channel.py --help

``case`` must be a key from ``CASE_DESCRIPTIONS`` (e.g. ``flattop``). Run with
``--help`` to print valid cases.
"""

import sys
from typing import Optional

from visualize import (
    plot_2d_density_profile,
    plot_iterative_1d_lineout,
    plot_laser_focal_plane,
)
import inversion_fbpic.density_profiles.ofi_channels as oc

CASE_DESCRIPTIONS: dict[str, str] = {
    "sgauss": "Super-Gaussian",
    "uniform": "Uniform Channel",
    "radial": "Radial Dependence",
    "guiding": "Guiding Structure",
    "ioninj": "Test Case for Ionization Injection",
    "guidetest": "Test Case for Guiding",
    "optimas": "Optimas Solution for Injection",
    "optimasdr": "Optimas Solution with Downramp",
    "longdn": "Long Downramp Injection Template",
    "flattop": "Flattop Ionization Injection",
    "marveloptimas": "Optimas Solution for Marvel HOFI Params",
    "pousadiff": "Diffraction-limited Metrology Source from DESY",
    "clarisdiff": "Diffraction-limited Metrology Source for CLARIS",
    "scapasimple": "Single gas jet for initial SCAPA experiment",
}


def _resolve_case(case: Optional[str]) -> str:
    """Map CLI input to a key in ``CASE_DESCRIPTIONS``."""
    if case is None:
        raise ValueError("No case selected.")
    stripped = case.strip()
    if stripped in CASE_DESCRIPTIONS:
        return stripped
    raise ValueError(f"Unknown case '{case}'.")


def _format_valid_cases() -> str:
    return "\n".join(f"{k}\t|\t{CASE_DESCRIPTIONS[k]}" for k in CASE_DESCRIPTIONS)


def main(case: Optional[str] = None):
    """
    Generate and plot density profiles for OFI channel examples.

    Select a profile by passing a case string (to this function or from the CLI).
    Use a key from ``CASE_DESCRIPTIONS``. If ``case`` is missing or not recognized,
    raises ``ValueError`` with the list of valid cases.
    """
    try:
        resolved = _resolve_case(case)
    except ValueError as e:
        raise ValueError(f"{e}\n\n{_format_valid_cases()}") from e

    plot_peak_density: Optional[float] = None

    print(f"Plasma profile: {CASE_DESCRIPTIONS[resolved]}")

    if resolved == "sgauss":
        density = oc.build_super_gaussian(height=1, z0=2, fwhm=0.5, power=1.0)
        plot_end = 4
        plot_peak_density = 1e18

    elif resolved == "uniform":
        density = oc.build_super_gaussian_plateau(
            height=1,
            z0_A=2,
            hwhm_A=0.5,
            power_A=1.0,
            z0_C=4,
            hwhm_C=1.0,
            power_C=1.0,
            slope_B=-0.01,
        )
        plot_end = 7
        plot_peak_density = 1e18

    elif resolved == "radial":
        density = oc.build_super_gaussian_plateau(
            height=1,
            z0_A=2,
            hwhm_A=0.5,
            power_A=1.0,
            z0_C=4,
            hwhm_C=1.0,
            power_C=1.0,
        )
        density = oc.add_radial_super_gaussian(z_density_func=density, fwhm_r=25e-6)
        plot_end = 7
        plot_peak_density = 1e18

    elif resolved == "guiding":
        nominal_density = 5e17
        density = oc.build_super_gaussian_plateau(
            height=1,
            z0_A=2,
            hwhm_A=0.5,
            power_A=1.0,
            z0_C=4,
            hwhm_C=1.0,
            power_C=1.0,
        )
        density = oc.add_radial_matched_profile(
            z_density_func=density, np0_nominal=nominal_density * 1e6
        )
        plot_end = 7
        plot_peak_density = nominal_density

    elif resolved == "ioninj":
        # I want to build two density profiles here.  First is going to be a small Gaussian-like source with
        #  mixed gas.  Second will be a long flattop of uniform gas.  The sum of these two profiles will be the
        #  longitudinal profile.  For this test I won't use the radial profiles yet.

        # First profile: Small Gaussian-like source (mixed gas)
        density1 = oc.build_super_gaussian_plateau(
            height=0.75,
            z0_A=1e-3,
            hwhm_A=0.3e-3,
            power_A=0.7,
            z0_C=1e-3,
            hwhm_C=0.2e-3,
            power_C=1.0,
        )

        # Second profile: Long flattop of uniform gas
        density2 = oc.build_super_gaussian_plateau(
            height=0.7,
            z0_A=1.65e-3,
            hwhm_A=0.4e-3,
            power_A=1.3,
            z0_C=5e-3,
            hwhm_C=0.1e-3,
            power_C=3.0,
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return density1(z, r) + density2(z, r)

        density = combined_density

        plot_end = 2e-3
        plot_peak_density = 5e17

    elif resolved == "guidetest":
        nominal_density = 3.5e17
        density1 = oc.build_super_gaussian_plateau(
            height=0.47,
            z0_A=3.5e-3,
            hwhm_A=1.0e-3,
            power_A=0.7,
            z0_C=3.5e-3,
            hwhm_C=1.5e-3,
            power_C=1.0,
        )

        # Second profile: Long flattop of uniform gas
        density2 = oc.build_super_gaussian_plateau(
            height=0.7,
            z0_A=8.125e-3,
            hwhm_A=3.0e-3,
            power_A=1.3,
            z0_C=30e-3,
            hwhm_C=0.1e-3,
            power_C=3.0,
        )

        density1 = oc.add_radial_matched_profile(
            z_density_func=density1, np0_nominal=nominal_density * 1e6
        )
        density2 = oc.add_radial_matched_profile(
            z_density_func=density2, np0_nominal=nominal_density * 1e6
        )

        n2_dope_level = 0.06

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return density1(z, r) * (1 + (7 - 1) * n2_dope_level) + density2(z, r)

        density = combined_density

        plot_end = 30e-3
        plot_peak_density = 5e17  #  10.714e17

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=plot_peak_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density1,
            label="2x $H_2$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density * (1 - n2_dope_level),
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density2,
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density1,
            label=f"2x $N_2$ Gas ({int(n2_dope_level*100)}% doped)",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=plot_peak_density * n2_dope_level,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    elif resolved == "optimas":
        nominal_density = 8e17
        density1 = oc.build_super_gaussian_plateau(
            height=0.4658182778588188,
            z0_A=3.5e-3,
            hwhm_A=1.0e-3,
            power_A=0.7,
            z0_C=3.5e-3,
            hwhm_C=1.5e-3,
            power_C=1.0,
        )

        # Second profile: Long flattop of uniform gas
        density2 = oc.build_super_gaussian_plateau(
            height=0.7,
            z0_A=8.125e-3,
            hwhm_A=3.0e-3,
            power_A=1.3,
            z0_C=30e-3,
            hwhm_C=3.0e-3,
            power_C=1.3,
        )

        density1 = oc.add_radial_matched_profile(
            z_density_func=density1, np0_nominal=nominal_density * 1e6
        )
        density2 = oc.add_radial_matched_profile(
            z_density_func=density2, np0_nominal=nominal_density * 1e6
        )

        n2_dope_level = 0.03

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return density1(z, r) * (1 + (7 - 1) * n2_dope_level) + density2(z, r)

        density = combined_density

        plot_end = 15e-3
        plot_peak_density = nominal_density / 0.7

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=plot_peak_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density1,
            label="2x $H_2$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density * (1 - n2_dope_level),
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density2,
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density1,
            label=f"2x $N_2$ Gas ({int(n2_dope_level*100)}% doped)",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=plot_peak_density * n2_dope_level,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    elif resolved == "optimasdr":
        # Varying Parameters
        laser_focal_position: float = -2.9891409128570254 * 1e-3  # Nominal 3.5 mm
        relative_injection_height: float = (
            34.91403834163108 / 100
        )  # Max 10.5% jet envelope
        total_dope_fraction: float = (
            7.5 / 100
        )  # 0.3-7.5% of bulk scale (doped_fraction cap)
        base_injection_hwhm: float = 2.4548718197864434 * 1e-3  # Nominal 1.9 mm
        jet_offset: float = 5.267394540361514 * 1e-3  # Nominal 3 mm
        bulk_flattop_end_extend: float = 9.743464055014385 * 1e-3  # Nominal about 20 mm

        # Non-varying input parameters
        flattop_correction: float = (
            0  # Removing to speedup optimas and we want downramp assistance
        )
        nominal_plasma_density: float = 0.3 * 1e18  # Maximum: 3e17 cm^-3
        downramp_hwhm: float = 3e-3
        base_jet_position: float = 11.0e-3
        jet_position: float = base_jet_position + jet_offset

        # Plasma Density Parameters
        bulk_flattop_start_position = 4.125e-3
        plateau_height: float = 0.7
        # N2 fraction within the jet envelope, derived from total N2 budget and jet strength
        dope_percentage: float = min(
            1.0, total_dope_fraction * plateau_height / relative_injection_height
        )
        peak_plasma_density: float = nominal_plasma_density / plateau_height

        bulk_flattop_end_position = jet_position + bulk_flattop_end_extend
        plot_end = bulk_flattop_end_position + 3 * downramp_hwhm

        print("Injection Jet Loc:", (jet_position) * 1e3, "mm")
        print(
            "Plasma Length:",
            (bulk_flattop_end_position - bulk_flattop_start_position) * 1e3,
            "mm",
        )

        # H2 Long flattop of uniform gas
        density_flattop = oc.build_super_gaussian_plateau(
            height=plateau_height,
            z0_A=bulk_flattop_start_position,
            hwhm_A=1.0e-3,
            power_A=1.0,
            z0_C=bulk_flattop_end_position,
            hwhm_C=downramp_hwhm,
            power_C=1.0,
        )
        density_jet_n2 = oc.build_super_gaussian_plateau(
            height=relative_injection_height * dope_percentage,
            z0_A=jet_position,
            hwhm_A=base_injection_hwhm,
            power_A=0.8,
            z0_C=jet_position,
            hwhm_C=base_injection_hwhm,
            power_C=0.8,
        )
        # H2 region within flattop
        density_jet_h2 = oc.build_super_gaussian_plateau(
            height=relative_injection_height * (1 - dope_percentage),
            z0_A=jet_position,
            hwhm_A=base_injection_hwhm,
            power_A=0.8,
            z0_C=jet_position,
            hwhm_C=base_injection_hwhm,
            power_C=0.8,
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return (
                density_flattop(z, r) + 7 * density_jet_n2(z, r) + density_jet_h2(z, r)
            )

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=peak_plasma_density,
            num=200,
        )
        plot_laser_focal_plane(laser_focal_position=laser_focal_position)
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_flattop,
            label="2x $H_2$ Gas (Bulk)",
            color="red",
            line_style="dashed",
            peak_density_value=peak_plasma_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet_h2,
            label=f"2x $H_2$ Gas (Jet, {relative_injection_height*(1-dope_percentage)/plateau_height*100:0.1f}% of bulk)",
            color="pink",
            line_style="dashed",
            fill_color="pink",
            peak_density_value=peak_plasma_density,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet_n2,
            label=f"2x $N_2$ Gas (Jet, {total_dope_fraction*100:0.2f}% of bulk)",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=peak_plasma_density,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

        density = combined_density
        plot_peak_density = peak_plasma_density

    elif resolved == "longdn":
        # Variable Parameters
        nominal_density = 1e18
        height_a = 0.9
        length_b = 531e-6  # m

        # Constant Parameters
        height_c = 0.818182
        start_a = 3.0e-3

        length_a = 1.0e-3
        start_b = start_a + length_a

        # Calculated Parameters
        height_b = (height_a + height_c) / 2
        slope_b = (height_c - height_a) / length_b

        density_a = oc.build_super_gaussian_plateau(
            height=height_a,
            z0_A=start_a,
            hwhm_A=1.1e-3,
            power_A=1.1,
            z0_C=start_b,
            hwhm_C=0,
            power_C=1.0,
        )

        density_b = oc.build_super_gaussian_plateau(
            height=height_b,
            z0_A=start_b,
            hwhm_A=0,
            power_A=1.0,
            z0_C=start_b + length_b,
            hwhm_C=0,
            power_C=1.0,
            slope_B=slope_b,
        )

        # Second profile: Long flattop of uniform gas
        density_c = oc.build_super_gaussian_plateau(
            height=height_c,
            z0_A=start_b + length_b,
            hwhm_A=0,
            power_A=1.0,
            z0_C=50e-3,
            hwhm_C=0.1e-3,
            power_C=3.0,
        )

        density_a = oc.add_radial_matched_profile(
            z_density_func=density_a, np0_nominal=nominal_density * 1e6
        )
        density_b = oc.add_radial_matched_profile(
            z_density_func=density_b, np0_nominal=nominal_density * 1e6
        )
        density_c = oc.add_radial_matched_profile(
            z_density_func=density_c, np0_nominal=nominal_density * 1e6
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return density_a(z, r) + density_b(z, r) + density_c(z, r)

        density = combined_density

        plot_end = 5e-3
        plot_peak_density = nominal_density / height_c

    elif resolved == "flattop":
        nominal_density = 6e17
        flattop_correction = (
            0.038169205678973896  # set to 0 for no correction, 1 for perfect correction
        )
        relative_injection_height = 3.000000e-02
        base_injection_hwhm = 1.000000e00 * 1e-3
        plateau_height: float = 0.7
        flattop_end_position: float = 22.590714610798642 * 1e-3  # Nominal: 18 mm

        downramp_hwhm: float = 3e-3
        base_jet_position: float = 11.0e-3
        jet_offset: float = 5e-3
        jet_position: float = base_jet_position + jet_offset

        # H2 Long flattop of uniform gas
        density_flattop = oc.build_super_gaussian_plateau(
            height=plateau_height,
            z0_A=4.125e-3,
            hwhm_A=1.0e-3,
            power_A=1.0,
            z0_C=flattop_end_position,
            hwhm_C=downramp_hwhm,
            power_C=1.0,
        )

        # N2 Region within flattop
        density_jet = oc.build_super_gaussian_plateau(
            height=relative_injection_height,
            z0_A=jet_position,
            hwhm_A=base_injection_hwhm,
            power_A=0.8,
            z0_C=jet_position,
            hwhm_C=base_injection_hwhm,
            power_C=0.8,
        )

        def subtracted_density_flattop(z, r):
            values = density_flattop(z, r) - flattop_correction * 7 * density_jet(z, r)
            return values

        subtracted_density_flattop = oc.add_radial_matched_profile(
            z_density_func=subtracted_density_flattop, np0_nominal=nominal_density * 1e6
        )
        density_jet = oc.add_radial_matched_profile(
            z_density_func=density_jet, np0_nominal=nominal_density * 1e6
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return subtracted_density_flattop(z, r) + 7 * density_jet(z, r)

        density = combined_density

        plot_end = 30e-3
        plot_peak_density = nominal_density / 0.7

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=plot_peak_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=subtracted_density_flattop,
            label="2x $H_2$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet,
            label="2x $N_2$ Gas",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=plot_peak_density,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    elif resolved == "marveloptimas":
        # Varying Parameters
        relative_injection_height: float = (
            18.411296348868333 / 100
        )  # Max 10.5% jet envelope
        total_dope_fraction: float = (
            1.4317108715686209 / 100
        )  # 0.3-7.5% of bulk scale (doped_fraction cap)
        base_injection_hwhm: float = 1.0109868925151524 * 1e-3  # Nominal 1.9 mm
        jet_offset: float = 4.50732997791046 * 1e-3  # Nominal 3 mm
        bulk_flattop_end_position: float = (
            47.151813526803224 * 1e-3
        )  # Nominal about 65 mm

        # Non-varying input parameters
        flattop_correction: float = (
            0  # Removing to speedup optimas and we want downramp assistance
        )
        nominal_density: float = 0.3 * 1e18  # Maximum: 3e17 cm^-3
        downramp_hwhm: float = 3e-3
        base_jet_position: float = 11.0e-3
        jet_position: float = base_jet_position + jet_offset
        plateau_height: float = 0.7
        # N2 fraction within the jet envelope, derived from total N2 budget and jet strength
        dope_percentage: float = min(
            1.0, total_dope_fraction * plateau_height / relative_injection_height
        )

        # H2 Long flattop of uniform gas
        density_flattop = oc.build_super_gaussian_plateau(
            height=plateau_height,
            z0_A=4.125e-3,
            hwhm_A=1.0e-3,
            power_A=1.0,
            z0_C=bulk_flattop_end_position,
            hwhm_C=downramp_hwhm,
            power_C=1.0,
        )

        # N2-doped Region within flattop
        density_jet_n2 = oc.build_super_gaussian_plateau(
            height=relative_injection_height * dope_percentage,
            z0_A=jet_position,
            hwhm_A=base_injection_hwhm,
            power_A=0.8,
            z0_C=jet_position,
            hwhm_C=base_injection_hwhm,
            power_C=0.8,
        )
        # H2 region within flattop
        density_jet_h2 = oc.build_super_gaussian_plateau(
            height=relative_injection_height * (1 - dope_percentage),
            z0_A=jet_position,
            hwhm_A=base_injection_hwhm,
            power_A=0.8,
            z0_C=jet_position,
            hwhm_C=base_injection_hwhm,
            power_C=0.8,
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return (
                density_flattop(z, r) + 7 * density_jet_n2(z, r) + density_jet_h2(z, r)
            )

        density = combined_density

        plot_end = 60e-3
        plot_peak_density = nominal_density / plateau_height

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=plot_peak_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_flattop,
            label="2x $H_2$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet_h2,
            label="2x $H_2$ Gas Jet",
            color="red",
            line_style="dotted",
            fill_color="pink",
            peak_density_value=plot_peak_density,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet_n2,
            label="2x $N_2$ Gas Jet",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=plot_peak_density,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    elif resolved == "pousadiff":
        dopant_level = 0.01
        nominal_flattop_density = 1.46e18
        low_density_tail = 4e16

        density_jet = oc.build_super_gaussian_plateau(
            height=0.8,
            z0_A=2.5e-3,
            hwhm_A=0.7e-3,
            power_A=1.0,
            z0_C=2.5e-3,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        density_bulk = oc.build_super_gaussian_plateau(
            height=1.0,
            z0_A=4.7e-3,
            hwhm_A=1.3e-3,
            power_A=1.5,
            z0_C=6.0e-3,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        density_exit = oc.build_super_gaussian_plateau(
            height=low_density_tail / nominal_flattop_density,
            z0_A=7.5e-3,
            hwhm_A=0.7e-3,
            power_A=1.0,
            z0_C=10.0e-3,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return (
                density_jet(z, r) * (1 + (7 - 1) * dopant_level)
                + density_bulk(z, r)
                + density_exit(z, r)
            )

        density = combined_density

        plot_end = 12e-3
        plot_peak_density = nominal_flattop_density

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=plot_peak_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet,
            label="2x $H_2$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density * (1 - dopant_level),
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_bulk,
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet,
            label=f"2x $N_2$ Gas ({int(dopant_level*100)}% doped)",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=plot_peak_density * dopant_level,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    elif resolved == "clarisdiff":
        dopant_level = 1.3937e-01
        nominal_flattop_density = 1.2415e18
        low_density_tail = 4e16

        end_bulk_flattop = 6.6189e-3
        jet_height = 5.1816e-01

        safety_factor = 2

        density_jet = oc.build_super_gaussian_plateau(
            height=jet_height / safety_factor,
            z0_A=2.5e-3,
            hwhm_A=0.7e-3,
            power_A=1.0,
            z0_C=2.5e-3,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        density_bulk = oc.build_super_gaussian_plateau(
            height=1.0 / safety_factor,
            z0_A=4.7e-3,
            hwhm_A=1.3e-3,
            power_A=1.5,
            z0_C=end_bulk_flattop,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        density_exit = oc.build_super_gaussian_plateau(
            height=low_density_tail / nominal_flattop_density / safety_factor,
            z0_A=end_bulk_flattop + 1.5e-3,
            hwhm_A=0.7e-3,
            power_A=1.0,
            z0_C=end_bulk_flattop + 4e-3,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return (
                density_jet(z, r) * (1 + (7 - 1) * dopant_level)
                + density_bulk(z, r)
                + density_exit(z, r)
            )

        density = combined_density

        plot_end = end_bulk_flattop + 6e-3
        plot_peak_density = nominal_flattop_density * safety_factor

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=plot_peak_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet,
            label="2x $H_2$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density * (1 - dopant_level),
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_bulk,
            color="red",
            line_style="dashed",
            peak_density_value=plot_peak_density,
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_jet,
            label=f"2x $N_2$ Gas ({int(dopant_level*100)}% doped)",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=plot_peak_density * dopant_level,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    elif resolved == "scapasimple":
        nominal_plasma_density = 1.8364512046483243 * 1e18
        nominal_gas_density = nominal_plasma_density / 2
        dopant_level = 0.03
        end_bulk_flattop: float = 8.005461064807271 * 1e-3  # Nominal: 6.0e-3

        density_bulk = oc.build_super_gaussian_plateau(
            height=1.0,
            z0_A=2.5e-3,
            hwhm_A=0.7e-3,
            power_A=1.0,
            z0_C=end_bulk_flattop,
            hwhm_C=0.7e-3,
            power_C=1.0,
        )

        # Combined density function: sum of both profiles
        def combined_density(z, r):
            return (
                density_bulk(z, r) * (1 - dopant_level)
                + 7 * density_bulk(z, r) * dopant_level
            )

        density = combined_density

        plot_end = 12e-3
        plot_peak_density = nominal_plasma_density

        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=combined_density,
            label="Combined Plasma Density",
            color="blue",
            line_style="solid",
            fill_color="lightblue",
            peak_density_value=nominal_plasma_density,
            num=200,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_bulk,
            label="$He$ Gas Density",
            color="red",
            line_style="dashed",
            peak_density_value=nominal_gas_density * (1 - dopant_level),
            num=200,
            figure=fig,
        )
        fig = plot_iterative_1d_lineout(
            z_start=-20e-6,
            z_end=plot_end,
            density_function=density_bulk,
            label=f"2x $N_2$ Gas ({dopant_level*100:.2f}%)",
            color="green",
            line_style="dotted",
            fill_color="lightgreen",
            peak_density_value=nominal_gas_density * dopant_level,
            num=200,
            units="Longitudinal Axis ($mm$)",
            figure=fig,
            show_plot=True,
        )

    else:
        # Defensive: all keys in CASE_DESCRIPTIONS should be handled above
        raise ValueError(f"Unhandled case '{resolved}'.")

    if plot_peak_density is None:
        plot_peak_density = 1

    plot_2d_density_profile(
        z_start=-20e-6,
        z_end=plot_end,
        r_width=100e-6,
        peak_density_value=plot_peak_density,
        density_function=density,
        units="mm",
        num=10000,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(_format_valid_cases())
        sys.exit(0)
    selected_case = sys.argv[1] if len(sys.argv) > 1 else None
    main(selected_case)
