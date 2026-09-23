"""Density profiles for plasma simulations using FBPIC.

This module provides functions for generating various density profiles used in
plasma simulations, including Gaussian profiles and combined Gaussian-triangular
profiles for down-ramp injection studies.

See `plot_downramp_injection.py` for examples on how to use these functions.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt


def build_gaussian_profile(
    sigma: float, center_location: float
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate a callable for a generic Gaussian profile in z.

    Args:
        sigma (float): Standard deviation, controls width of profile.
        center_location (float): Location of the Gaussian centroid.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        n = np.exp(-((z - center_location) ** 2) / (2 * sigma**2))
        return n

    return dens_func


def build_generalized_gaussian_profile(
    gauss_peak: float,
    gauss_z0: float,
    gauss_alpha: float,
    gauss_beta: float,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate a callable for a generalized normal distribution PDF.

    Args:
        gauss_peak (float): Peak value of the distribution.
        gauss_z0 (float): Location parameter (mean).
        gauss_alpha (float): Scale parameter (must be > 0).
        gauss_beta (float): Shape parameter (must be > 0).

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        exponent = -((np.abs(z - gauss_z0) / gauss_alpha) ** gauss_beta)
        return gauss_peak * np.exp(exponent)

    return dens_func


def build_gaussian_plus_triangle_z_density_function(
    sigma: float,
    center_location: float,
    z_tip: float,
    left_width: float,
    right_width: float,
    triangle_height: float = 1.0,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate a density profile combining Gaussian and triangular distributions.

    Args:
        sigma (float): Standard deviation of the Gaussian in z.
        center_location (float): Center position of the Gaussian in z.
        z_tip (float): Center (tip) position of the triangle in z.
        left_width (float): Length scale of the triangle to the left of the tip.
        right_width (float): Length scale of the triangle to the right of the tip.
        triangle_height (float, optional): Height of the triangle at the tip with respect to the Gaussian amplitude. Defaults to 1.0.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """
    gaussian_func = build_gaussian_profile(sigma=sigma, center_location=center_location)

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z)
        r = np.asarray(r)

        # Triangle (tent) component
        left_mask = (z >= z_tip - left_width) & (z < z_tip)
        right_mask = (z >= z_tip) & (z <= z_tip + right_width)
        triangle = np.zeros_like(z, dtype=float)
        # Left ramp: rises linearly to tip
        triangle[left_mask] = (
            triangle_height * (z[left_mask] - (z_tip - left_width)) / left_width
        )
        # Right ramp: falls linearly from tip
        triangle[right_mask] = (
            triangle_height * ((z_tip + right_width) - z[right_mask]) / right_width
        )

        # Sum the two components and normalize
        return (gaussian_func(z, r) + triangle) / (1 + triangle_height)

    return dens_func


def build_generalized_gaussian_with_downramp(
    gauss_peak: float,
    gauss_alpha: float,
    gauss_beta: float,
    gauss_z0: float,
    ramp_z0: float,
    ramp_left_tau: float,
    ramp_right_width: float,
    ramp_height: float,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate a density profile combining a generalized Gaussian and a triangular downramp.

    Args:
        gauss_peak (float): Peak value of the distribution.
        gauss_alpha (float): Scale parameter (must be > 0).
        gauss_beta (float): Shape parameter (must be > 0).
        gauss_z0 (float): Center position of the Gaussian in z.
        ramp_z0 (float): Center (tip) position of the triangle in z.
        ramp_left_tau (float): Length scale of the triangle to the left of the tip.
        ramp_right_width (float): Length scale of the triangle to the right of the tip.
        ramp_height (float): Height of the triangle at the tip with respect to the Gaussian amplitude.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """
    generalized_gaussian = build_generalized_gaussian_profile(
        gauss_peak=gauss_peak,
        gauss_z0=gauss_z0,
        gauss_alpha=gauss_alpha,
        gauss_beta=gauss_beta,
    )

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z)
        r = np.asarray(r)

        # Triangle (tent) component
        left_mask = z < ramp_z0
        right_mask = (z >= ramp_z0) & (z <= ramp_z0 + ramp_right_width)
        triangle = np.zeros_like(z, dtype=float)
        # Left ramp: exponential profile
        x = (z[left_mask] - (ramp_z0 - ramp_left_tau)) / ramp_left_tau
        norm = np.exp(1) - 1
        triangle[left_mask] = ramp_height * (np.exp(x) - 1) / norm
        # Right ramp: falls linearly from tip
        triangle[right_mask] = (
            ramp_height
            * ((ramp_z0 + ramp_right_width) - z[right_mask])
            / ramp_right_width
        )
        triangle = np.maximum(triangle, 0)

        # Sum the two components and normalize
        return (generalized_gaussian(z, r) + triangle) / (1 + ramp_height)

    return dens_func


def build_piecewise_linear_downramp(
    upramp_length: float,
    downramp_start_position: float,
    downramp_length: float,
    downramp_height_ratio: float,
    plateau_end_position: float,
    plateau_downramp_length: float,
) -> Callable:
    """
    Generate a callable for a piecewise linear downramp density profile in z, uniform in r.

    The profile consists of:
      - Linear upramp from 0 to 1 of ramp length upramp_length
      - Plateau at 1 from upramp_length to downramp_start_position
      - Linear downramp from 1 to downramp_height_ratio of ramp length downramp_length, starting at downramp_start_position
      - Plateau at downramp_height_ratio from downramp_start_position + downramp_length to plataeu_end_position
      - Linear downramp from downramp_height_ratio to 0 of ramp length plataeu_downramp_length, starting at plataeu_end_position
      - Elsewhere 0

    Args:
        upramp_length (float): Length of the linear upramp.
        downramp_start_position (float): Start position of the downramp.
        downramp_length (float): Length of the downramp.
        downramp_height_ratio (float): Height ratio at the end of the downramp.
        plataeu_end_position (float): End position of the plateau.
        plataeu_downramp_length (float): Length of the final downramp to zero.

    Returns:
        Callable: Function that returns the relative density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z)
        r = np.asarray(r)
        dens = np.zeros_like(z, dtype=float)
        # Linear upramp
        mask1 = (z >= 0) & (z < upramp_length)
        dens[mask1] = (z[mask1] - 0) / upramp_length
        # Plateau at 1
        mask2 = (z >= upramp_length) & (z < downramp_start_position)
        dens[mask2] = 1.0
        # Linear downramp from 1 to n0
        mask3 = (z >= downramp_start_position) & (
            z < downramp_start_position + downramp_length
        )
        dens[mask3] = (
            1.0
            + (downramp_height_ratio - 1.0)
            * (z[mask3] - downramp_start_position)
            / downramp_length
        )
        # Plateau at n0
        mask4 = (z >= downramp_start_position + downramp_length) & (
            z < plateau_end_position
        )
        dens[mask4] = downramp_height_ratio
        # Linear downramp from n0 to 0
        mask5 = (z >= plateau_end_position) & (
            z < plateau_end_position + plateau_downramp_length
        )
        dens[mask5] = downramp_height_ratio * (
            1 - (z[mask5] - plateau_end_position) / plateau_downramp_length
        )
        # Elsewhere, dens is 0
        return dens

    return dens_func


def build_piecewise_linear_modulated_downramp(
    upramp_length: float,
    downramp_start_position: float,
    downramp_length: float,
    downramp_height_ratio: float,
    plateau_end_position: float,
    plateau_downramp_length: float,
    modulation_amplitude: float,
    modulation_wavelength: float,
) -> Callable:
    """
    Generate a callable for a piecewise linear downramp density profile with sinusoidal modulation.

    This function reuses the base piecewise linear downramp and adds sinusoidal modulation
    during the downramp portion only.

    Args:
        upramp_length (float): Length of the linear upramp.
        downramp_start_position (float): Start position of the downramp.
        downramp_length (float): Length of the downramp.
        downramp_height_ratio (float): Height ratio at the end of the downramp.
        plataeu_end_position (float): End position of the plateau.
        plataeu_downramp_length (float): Length of the final downramp to zero.
        modulation_amplitude (float): Amplitude of the sinusoidal modulation (as fraction of local density).
        modulation_wavelength (float): Frequency of the sinusoidal modulation in units of 1/length.

    Returns:
        Callable: Function that returns the relative density at position (z, r).
    """

    # Get the base piecewise linear function
    base_dens_func = build_piecewise_linear_downramp(
        upramp_length=upramp_length,
        downramp_start_position=downramp_start_position,
        downramp_length=downramp_length,
        downramp_height_ratio=downramp_height_ratio,
        plateau_end_position=plateau_end_position,
        plateau_downramp_length=plateau_downramp_length,
    )

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z)
        r = np.asarray(r)

        # Get the base density profile
        base_dens = base_dens_func(z, r)

        # Add sinusoidal modulation only during the downramp portion
        modulation_mask = (z >= downramp_start_position) & (
            z < downramp_start_position + downramp_length
        )

        # Calculate modulation
        phase = (
            2
            * np.pi
            / modulation_wavelength
            * (z[modulation_mask] - downramp_start_position)
        )
        modulation = modulation_amplitude * np.sin(phase)

        # Apply modulation
        modulated_dens = base_dens.copy()
        modulated_dens[modulation_mask] += modulation

        # Ensure density is non-negative
        modulated_dens = np.maximum(modulated_dens, 0)

        return modulated_dens

    return dens_func


def build_piecewise_constant_downramp(
    downramp_position: float,
    downramp_height_ratio: float,
    plateau_end_position: float,
    plateau_cutoff_smoothness: float | None = None,
) -> Callable:
    """
    Generate a callable for a piecewise constant downramp density profile in z, uniform in r.

    The profile consists of:
        - Plateau at 1 from 0 to downramp_position
        - Plateau at downramp_height_ratio from downramp_position to plateau_end_position
        - Elsewhere 0

    If plateau_cutoff_smoothness is provided, the profile will be smoothed at the cutoff by a cosine-squared
    function within a window of [plateau_end_position - plateau_cutoff_smoothness/2.0, plateau_end_position + plateau_cutoff_smoothness/2.0]

    Args:
        downramp_position: Position of the downramp (end of the first plateau).
        downramp_height_ratio: Height ratio at the end of the downramp.
        plateau_end_position: End position of the (final) plateau.
        plateau_cutoff_smoothness: Smoothness of the cutoff at the end of the plateau. If None or 0.0, the profile is not smoothed. Defaults to None.

    Returns:
        Callable: Function that returns the relative density at position (z, r).
    """

    if (
        plateau_cutoff_smoothness is not None
        and plateau_cutoff_smoothness > 0.0
        and plateau_end_position - downramp_position <= plateau_cutoff_smoothness / 2.0
    ):
        raise ValueError(
            "build_piecewise_constant_downramp: Plateau end position from downramp_position must be greater than half of the cutoff smoothness"
        )

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z)
        r = np.asarray(r)
        dens = np.zeros_like(z, dtype=float)

        # first plateau (full density)
        dens[(z >= 0) & (z < downramp_position)] = 1.0
        # second plateau (reduced density)
        dens[(z >= downramp_position) & (z < plateau_end_position)] = (
            downramp_height_ratio
        )
        if plateau_cutoff_smoothness is not None and plateau_cutoff_smoothness > 0.0:
            # Smooth the cutoff by a cosine-squared function within a window of [plateau_end_position - plateau_cutoff_smoothness/2.0, plateau_end_position + plateau_cutoff_smoothness/2.0]
            smooth_mask = (
                z >= plateau_end_position - plateau_cutoff_smoothness / 2.0
            ) & (z <= plateau_end_position + plateau_cutoff_smoothness / 2.0)
            dens[smooth_mask] = (
                downramp_height_ratio
                * np.cos(
                    np.pi
                    / 2.0
                    * (
                        z[smooth_mask]
                        - plateau_end_position
                        + plateau_cutoff_smoothness / 2.0
                    )
                    / plateau_cutoff_smoothness
                )
                ** 2
            )
        # zero otherwise

        return dens

    return dens_func


def build_smooth_flattop_profile(
    flattop_width: float,
    upramp_length: float,
    downramp_length: float,
    offset_length: float = 0.0,
) -> Callable:
    """
    Generate a callable for a smoothed flattop density profile in z, uniform in r.

    The profile consists of:
        - Cosine-squared upramp from 0 to upramp_length
        - Cosine-squared flattop from upramp_length to (upramp_length + flattop_width)
        - Cosine-squared downramp from (upramp_length + flattop_width) to (upramp_length + flattop_width + downramp_length)
        - Elsewhere 0
    If offset_length is provided, the profile is shifted by offset_length.


    Total length of the profile is (upramp_length + flattop_width + downramp_length)
    The end of the profile is at (upramp_length + flattop_width + downramp_length + offset_length)

    Args:
        flattop_width: Width of the flattop.
        upramp_length: Length of the upramp.
        downramp_length: Length of the downramp.
        offset_length: Offset of the profile. Defaults to 0.0.
    Returns:
        Callable: Function that returns the relative density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z) - offset_length
        r = np.asarray(r)
        dens = np.zeros_like(z, dtype=float)

        # Cosine-squared upramp from 0 to upramp_length
        if upramp_length > 0.0:
            mask1 = (z >= 0) & (z < upramp_length)
            dens[mask1] = np.sin(np.pi / 2.0 * z[mask1] / upramp_length) ** 2
        # Constant from upramp_length to (upramp_length + flattop_width)
        dens[(z >= upramp_length) & (z < upramp_length + flattop_width)] = 1.0
        # Cosine-squared downramp from (upramp_length + flattop_width) to (upramp_length + flattop_width + downramp_length)
        if downramp_length > 0.0:
            mask2 = (z >= upramp_length + flattop_width) & (
                z < upramp_length + flattop_width + downramp_length
            )
            dens[mask2] = (
                np.cos(
                    np.pi
                    / 2.0
                    * (z[mask2] - upramp_length - flattop_width)
                    / downramp_length
                )
                ** 2
            )
        # zero otherwise

        return dens

    return dens_func


def build_asymmetric_cosine_blade_profile(
    upramp_length: float, downramp_length: float, offset_length: float = 0.0
) -> Callable:
    """
    Generate a callable for an asymmetric cosine-squared blade density profile in z, uniform in r.

    The profile consists of:
        - Cosine-squared upramp from 0 to upramp_length
        - Cosine-squared downramp from upramp_length to (upramp_length + downramp_length)
        - Elsewhere 0
    If offset_length is provided, the profile is shifted by offset_length.

    Total length of the profile is (upramp_length + downramp_length)
    The end of the profile is at (upramp_length + downramp_length + offset_length)

    Args:
        upramp_length: Length of the upramp.
        downramp_length: Length of the downramp.
        offset_length: Offset of the profile. Defaults to 0.0.
    Returns:
        Callable: Function that returns the relative density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        z = np.asarray(z) - offset_length
        r = np.asarray(r)
        dens = np.zeros_like(z, dtype=float)

        # Cosine-squared upramp from 0 to upramp_length
        if upramp_length > 0.0:
            mask1 = (z >= 0) & (z < upramp_length)
            dens[mask1] = np.sin(np.pi / 2.0 * z[mask1] / upramp_length) ** 2
        # Cosine-squared downramp from upramp_length to (upramp_length + downramp_length)
        if downramp_length > 0.0:
            mask2 = (z >= upramp_length) & (z < upramp_length + downramp_length)
            dens[mask2] = (
                np.cos(np.pi / 2.0 * (z[mask2] - upramp_length) / downramp_length) ** 2
            )
        # zero otherwise

        return dens

    return dens_func


def build_flattop_with_asymmetric_blade_downramp_density(
    flattop_width: float,
    flattop_upramp_length: float,
    flattop_downramp_length: float,
    blade_relative_height: float,
    blade_upramp_length: float,
    blade_downramp_length: float,
    downramp_position: float,
) -> Callable:
    """
    Generate a callable for a flattop with asymmetric blade downramp density profile in z, uniform in r.

    The profile is a combination of:
        - A cosine-smoothed flattop from 0 to (flattop_upramp_length + flattop_width + flattop_downramp_length), flat for flattop_width.
        - Asymmetric cosine-squared blade from (downramp_position - blade_upramp_length) to (downramp_position + blade_downramp_length), of maximum height blade_relative_height.

    The beginning of the profile is at the minimum of 0 and (downramp_position - blade_upramp_length)
    The end of the profile is at the maximum of (downramp_position + blade_downramp_length) and (flattop_upramp_length + flattop_width + flattop_downramp_length)

    Args:
        flattop_width: Width of the flattop.
        flattop_upramp_length: Length of the upramp of the flattop.
        flattop_downramp_length: Length of the downramp of the flattop.
        blade_relative_height: Relative height of the blade.
        blade_upramp_length: Length of the upramp of the blade.
        blade_downramp_length: Length of the downramp of the blade.
        downramp_position: Position of the downramp.
    Returns:
        Callable: Function that returns the relative density at position (z, r).
    """

    flattop = build_smooth_flattop_profile(
        flattop_width=flattop_width,
        upramp_length=flattop_upramp_length,
        downramp_length=flattop_downramp_length,
    )
    blade = build_asymmetric_cosine_blade_profile(
        upramp_length=blade_upramp_length,
        downramp_length=blade_downramp_length,
        offset_length=downramp_position - blade_upramp_length,
    )

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        return flattop(z, r) + blade_relative_height * blade(z, r)

    return dens_func
