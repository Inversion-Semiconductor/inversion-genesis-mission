"""Density profiles for plasma simulations using FBPIC.

This module is for longer plasma sources that are more idealized, characterized
by uniform sections with tanh ramps.

See `plot_downramp_injection.py` for examples on how to use these functions.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.constants import physical_constants


def build_super_gaussian(
    height: float,
    z0: float,
    fwhm: float,
    power: float = 1.0,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate a super-Gaussian ramp profile that reaches a specific height with zero slope at z0.

    The super-Gaussian has the form: height * exp(-|(z - z0)/sigma|^(2*power))
    This ensures:
    - The function reaches exactly 'height' at z0
    - The slope is exactly 0 at z0 (derivative is zero there)
    - The width is controlled by fwhm (full width at half maximum)
    - The shape is controlled by power (power=1 gives regular Gaussian, power>1 gives flatter top)

    Args:
        height (float): The height/density value at z0.
        z0 (float): The position where the function reaches its peak height with zero slope.
        fwhm (float): The full width at half maximum of the profile.
        power (float, optional): The power in the super-Gaussian. Defaults to 1.0 (regular Gaussian).
                                Higher values give flatter tops and steeper sides.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns relative density at position z and r.

        Args:
            z: Array-like of z positions.
            r: Array-like of r positions.

        Returns:
            Array-like of relative density values.
        """
        if fwhm == 0.0:
            return 0

        z = np.asarray(z)
        r = np.asarray(r)

        # Convert FWHM to sigma for super-Gaussian
        # For super-Gaussian: FWHM = 2 * sigma * (ln(2))^(1/(2*power))
        sigma = fwhm / (2 * (np.log(2)) ** (1 / (2 * power)))

        # Super-Gaussian: height * exp(-|(z - z0)/sigma|^(2*power))

        exponent = -np.abs((z - z0) / sigma) ** (2 * power)
        n = height * np.exp(exponent)

        return n

    return dens_func


def build_super_gaussian_plateau(
    height: float,
    z0_A: float,
    z0_C: float,
    hwhm_A: float,
    hwhm_C: float,
    power_A: float = 1.0,
    power_C: float = 1.0,
    slope_B: float = 0.0,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Generate a piecewise density profile with three regions:
    - Region A: Super-Gaussian upramp from -infinity to z0_A
    - Region B: Linear slope from z0_A to z0_C
    - Region C: Super-Gaussian downramp from z0_C to +infinity

    The super-Gaussian ramps ensure smooth transitions with zero slope at the boundaries.
    The heights of the super-Gaussians are automatically calculated to match the linear region.

    Args:
        height (float): The height/density value at the center of region B (z_center = (z0_A + z0_C)/2).
        z0_A (float): The end position of the upramp (start of linear region).
        z0_C (float): The start position of the downramp (end of linear region).
        hwhm_A (float): The half width at half maximum of the upramp super-Gaussian.
        hwhm_C (float): The half width at half maximum of the downramp super-Gaussian.
        power_A (float, optional): The power in the upramp super-Gaussian. Defaults to 1.0 (regular Gaussian).
                                  Higher values give flatter tops and steeper sides.
        power_C (float, optional): The power in the downramp super-Gaussian. Defaults to 1.0 (regular Gaussian).
                                  Higher values give flatter tops and steeper sides.
        slope_B (float, optional): The slope of the linear region B. Defaults to 0.0 (uniform plateau).

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """
    # Calculate the center position of region B
    z_center = (z0_A + z0_C) / 2

    # Calculate heights at the boundaries of region B
    height_A = height + slope_B * (z0_A - z_center)
    height_C = height + slope_B * (z0_C - z_center)

    # Create super-Gaussian functions for regions A and C
    super_gaussian_A = build_super_gaussian(
        height=height_A, z0=z0_A, fwhm=2 * hwhm_A, power=power_A
    )
    super_gaussian_C = build_super_gaussian(
        height=height_C, z0=z0_C, fwhm=2 * hwhm_C, power=power_C
    )

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns relative density at position z and r.

        Args:
            z: Array-like of z positions.
            r: Array-like of r positions.

        Returns:
            Array-like of relative density values.
        """
        z = np.asarray(z)
        r = np.asarray(r)

        # Initialize density array
        n = np.zeros_like(z, dtype=float)

        # Region A: Super-Gaussian upramp (z < z0_A)
        mask_A = z < z0_A
        if np.any(mask_A):
            n[mask_A] = super_gaussian_A(z[mask_A], r)

        # Region B: Linear slope (z0_A <= z <= z0_C)
        mask_B = (z >= z0_A) & (z <= z0_C)
        if np.any(mask_B):
            # Linear interpolation: height_A + slope_B * (z - z0_A)
            n[mask_B] = height_A + slope_B * (z[mask_B] - z0_A)

        # Region C: Super-Gaussian downramp (z > z0_C)
        mask_C = z > z0_C
        if np.any(mask_C):
            n[mask_C] = super_gaussian_C(z[mask_C], r)

        return n

    return dens_func


def add_radial_super_gaussian(
    z_density_func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    fwhm_r: float,
    r0: float = 0.0,
    power_r: float = 1.0,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Add radial super-Gaussian dependence to a z-dependent density function.

    The resulting function has the form:
    density(z, r) = z_density_func(z, r=0) * radial_super_gaussian(r)

    where radial_super_gaussian(r) = exp(-|(r - r0)/sigma_r|^(2*power_r))

    Args:
        z_density_func: A callable that defines the z-dependence of the density profile.
                       Should take (z, r) as arguments and return density values.
        r0 (float): The radial center position of the super-Gaussian (typically 0 for on-axis).
        fwhm_r (float): The full width at half maximum of the radial super-Gaussian.
        power_r (float, optional): The power in the radial super-Gaussian. Defaults to 1.0 (regular Gaussian).
                                  Higher values give flatter tops and steeper sides.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r) with both z and r dependence.
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns relative density at position z and r with radial super-Gaussian dependence.

        Args:
            z: Array-like of z positions.
            r: Array-like of r positions.

        Returns:
            Array-like of relative density values.
        """
        z = np.asarray(z)
        r = np.asarray(r)

        # Get the z-dependent density on-axis (r=0)
        z_density = z_density_func(z, 0)

        # Calculate radial super-Gaussian
        # Convert FWHM to sigma for super-Gaussian
        sigma_r = fwhm_r / (2 * (np.log(2)) ** (1 / (2 * power_r)))

        # Radial super-Gaussian: exp(-|(r - r0)/sigma_r|^(2*power_r))
        exponent_r = -np.abs((r - r0) / sigma_r) ** (2 * power_r)
        radial_factor = np.exp(exponent_r)

        # Combine z and r dependence
        return z_density * radial_factor

    return dens_func


def add_radial_ramp_profile(
    z_density_func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    r0: float,
    hwhm_i: float,
    hwhm_o: float,
    power_i: float = 1.0,
    power_o: float = 1.0,
    floor: float = 0.0,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Add radial ramp profile dependence to a z-dependent density function.

    The resulting function has the form:
    density(z, r) = z_density_func(z, r=0) * radial_ramp_profile(r)

    where radial_ramp_profile(r) has two regions:
    - Region A: Ramp up from 0 to 1 from r=0 to r=r0 (super-Gaussian upramp)
    - Region B: Ramp down from 1 to floor from r=r0 to r=infinity (super-Gaussian downramp)

    Args:
        z_density_func: A callable that defines the z-dependence of the density profile.
                       Should take (z, r) as arguments and return density values.
        r0 (float): The radial position where the profile reaches its peak value of 1.
        hwhm_i (float): The half width at half maximum of the inner (upramp) super-Gaussian.
        hwhm_o (float): The half width at half maximum of the outer (downramp) super-Gaussian.
        power_i (float, optional): The power in the inner super-Gaussian. Defaults to 1.0 (regular Gaussian).
                                  Higher values give flatter tops and steeper sides.
        power_o (float, optional): The power in the outer super-Gaussian. Defaults to 1.0 (regular Gaussian).
                                  Higher values give flatter tops and steeper sides.
        floor (float, optional): The minimum value the radial profile approaches at large r. Defaults to 0.0.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r) with both z and r dependence.
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns relative density at position z and r with radial ramp profile dependence.

        Args:
            z: Array-like of z positions.
            r: Array-like of r positions.

        Returns:
            Array-like of relative density values.
        """
        z = np.asarray(z)
        r = np.asarray(r)

        # Get the z-dependent density on-axis (r=0)
        z_density = z_density_func(z, 0)

        # Initialize radial factor array
        radial_factor = np.zeros_like(r, dtype=float)

        # Convert HWHM to sigma for super-Gaussian (separate for inner and outer)
        sigma_i = hwhm_i / ((np.log(2)) ** (1 / (2 * power_i)))
        sigma_o = hwhm_o / ((np.log(2)) ** (1 / (2 * power_o)))

        # Region A: Super-Gaussian upramp (|r| <= r0)
        mask_A = np.abs(r) <= r0
        if np.any(mask_A):
            # Upramp: exp(-|(|r| - r0)/sigma_i|^(2*power_i))
            # This gives 1 at |r|=r0 with zero slope, symmetric about r=0
            exponent_A = -np.abs((np.abs(r[mask_A]) - r0) / sigma_i) ** (2 * power_i)
            radial_factor[mask_A] = np.exp(exponent_A)

        # Region B: Super-Gaussian downramp (|r| > r0)
        mask_B = np.abs(r) > r0
        if np.any(mask_B):
            # Downramp: floor + (1 - floor) * exp(-|(|r| - r0)/sigma_o|^(2*power_o))
            # This gives 1 at |r|=r0 with zero slope, approaches floor at infinity
            exponent_B = -np.abs((np.abs(r[mask_B]) - r0) / sigma_o) ** (2 * power_o)
            radial_factor[mask_B] = floor + (1 - floor) * np.exp(exponent_B)

        # Combine z and r dependence
        return z_density * radial_factor

    return dens_func


def add_radial_matched_profile(
    z_density_func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    np0_nominal: float,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
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
        z_density_func: A callable that defines the z-dependence of the density profile.
                       Should take (z, r) as arguments and return density values.
        np0_nominal (float): The nominal value of the on-axis plasma density

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r) with both z and r dependence.
    """
    # Classical electron radius from scipy.constants
    # physical_constants returns (value, unit, uncertainty)
    r_e = physical_constants["classical electron radius"][0]

    w_m = 50e-6 / np.sqrt(np0_nominal / (2e17 * 1e6))

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns relative density at position z and r with radial matched profile dependence.

        Args:
            z: Array-like of z positions.
            r: Array-like of r positions.

        Returns:
            Array-like of relative density values.
        """
        z = np.asarray(z)
        r = np.asarray(r)

        # Get the z-dependent density on-axis (r=0)
        n_p0 = z_density_func(z, 0)

        # Calculate radial modulation factor
        # n_p(z,r) = n_p0(z) * (1 + r^2 / (pi * r_e * n_p0(z) * w_m^4))
        denominator = np.pi * r_e * np0_nominal * w_m**4
        radial_factor = 1 + r**2 / denominator

        # Combine z and r dependence
        return n_p0 * radial_factor

    return dens_func


def build_interpolated_lineout_profile(
    z_axis: npt.ArrayLike,
    density: npt.ArrayLike,
    normalize: bool = True,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Build a density profile from a 1D lineout, uniform in r.

    The on-axis density is linearly interpolated in z. Values outside the
    provided axis range return zero.

    Args:
        z_axis: Positions along z in meters.
        density: On-axis density values at each z position. When ``normalize``
            is True, these are scaled by their peak so the profile returns
            relative density in [0, 1].
        normalize: Whether to divide by the peak density. Defaults to True.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the relative density at position (z, r).
    """
    z_axis = np.asarray(z_axis, dtype=float)
    density = np.asarray(density, dtype=float)

    if normalize:
        peak_density = np.max(density)
        if peak_density == 0.0:
            raise ValueError("Peak density must be non-zero")
        density = density / peak_density

    if z_axis.shape != density.shape:
        raise ValueError("z_axis and density must have the same shape")

    sort_idx = np.argsort(z_axis)
    z_axis = z_axis[sort_idx]
    density = density[sort_idx]

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns relative density at position z and r."""
        z = np.asarray(z)
        r = np.asarray(r)
        return np.interp(z, z_axis, density, left=0.0, right=0.0)

    return dens_func


def scale_density_function(
    density_func: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    scale_factor: float,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
    """
    Scale a density function by a constant multiplicative factor.

    This is a simple helper function that multiplies an existing density function
    by a scalar value. Useful for adjusting density magnitudes without recreating
    the entire profile.

    Args:
        density_func: A callable that defines the density profile.
                     Should take (z, r) as arguments and return density values.
        scale_factor: The scalar multiplier to apply to the density function.

    Returns:
        Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]:
            Function that returns the scaled density at position (z, r).
    """

    def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
        """Returns scaled density at position z and r.

        Args:
            z: Array-like of z positions.
            r: Array-like of r positions.

        Returns:
            Array-like of scaled density values.
        """
        return scale_factor * density_func(z, r)

    return dens_func
