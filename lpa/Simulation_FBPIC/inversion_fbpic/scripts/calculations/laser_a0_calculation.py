"""Calculations of laser intensity parameter a0 for various laser systems.

This module provides functions to calculate the laser intensity parameter a0
and analyze laser profiles for different laser systems.

Usage:
    Set the global flags at the top to determine what laser statistics get
    printed. For example: HTU = True
    Run with any python interpreter.
"""

import numpy as np
import matplotlib.pyplot as plt

from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.lpa_utils.laser.transverse_laser_profiles import GaussianTransverseProfile
from scipy.constants import c, e, m_e, pi, epsilon_0, mu_0

# Global configuration flags
HTU: bool = False
EKSPLA: bool = False
ANTIPOV: bool = False
INVERSION: bool = True

# Physical constants
FWHM_TO_SIGMA_FACTOR = 2.35482
FWHM_TO_TAU_FACTOR = 1 / FWHM_TO_SIGMA_FACTOR * 2
WAVELENGTH_CONVERSION = 1e-6  # micrometers to meters
TIME_CONVERSION_FS = 1e15  # seconds to femtoseconds
LENGTH_CONVERSION_UM = 1e6  # meters to micrometers
INTENSITY_CONVERSION = 100**2  # W/m² to W/cm²
POWER_CONVERSION_TW = 1e-12  # W to TW
A0_ENGINEERING_FACTOR = 7.3e-19


def calculate_laser_energy_from_a0(
    a0: float,
    wavelength: float,
    w0: float,
    tau_fwhm: float,
) -> float:
    """Calculate laser energy from a0 and other laser parameters.

    This function inverts the a0 calculation to solve for laser energy.

    Args:
        a0: Normalized laser intensity parameter.
        wavelength: Laser wavelength in micrometers.
        w0: Laser focus spot size in meters.
        tau_fwhm: Temporal duration FWHM, in seconds.

    Returns:
        Laser energy in Joules.
    """
    # Calculate wavenumber
    wavenumber = 2 * pi / (wavelength * WAVELENGTH_CONVERSION)

    # From a0 to peak field
    peak_field = a0 * (m_e * c**2 * wavenumber / e)

    # From peak field to peak intensity
    peak_intensity = peak_field**2 * c * epsilon_0 / 2

    # From peak intensity to peak power
    peak_power = peak_intensity * pi * w0**2 / 2

    # From peak power to laser energy
    tau = tau_fwhm / FWHM_TO_TAU_FACTOR
    laser_energy = peak_power * tau * np.sqrt(2 * pi) / 2

    return laser_energy


def calculate_a0(
    laser_energy: float,
    wavelength: float,
    w0: float,
    tau_fwhm: float,
    disp: bool = True,
) -> float:
    """Calculate a0 and other relevant laser parameters.

    Args:
        laser_energy: Laser energy in Joules.
        wavelength: Laser wavelength in micrometers.
        w0: Laser focus spot size in meters.
        tau_fwhm: Temporal duration FWHM, in seconds.
        disp: bool = True Whether to display the results to stdout.

    Returns:
        a0_normalized: Normalized laser intensity parameter.
    """
    tau = tau_fwhm * FWHM_TO_TAU_FACTOR
    peak_power = laser_energy / tau * 2 / np.sqrt(2 * pi)  # laser_energy / tau_fwhm
    peak_intensity = 2 * peak_power / (pi * w0**2)
    peak_field = np.sqrt(peak_intensity * 2 / (c * epsilon_0))
    wavenumber = 2 * pi / (wavelength * WAVELENGTH_CONVERSION)
    a0_normalized = peak_field / (m_e * c**2 * wavenumber / e)

    if disp:
        print("Laser Parameters:")
        print(f" input tau     {tau * TIME_CONVERSION_FS:.4f} fs")
        print(f" total_power   {peak_power * POWER_CONVERSION_TW:.4f} TW")
        print(f" peak_intensity {peak_intensity / INTENSITY_CONVERSION:.4e} W/cm^2")
        print(f" peak_field    {peak_field:.4e} V/m")
        print(f" a0 = {a0_normalized}")

    a0_engineering = np.sqrt(
        A0_ENGINEERING_FACTOR * wavelength**2 * peak_intensity / INTENSITY_CONVERSION
    )
    if disp:
        print(f"engineering formula a0 = {a0_engineering}")
        print()

    return a0_normalized


def _calculate_fwhm_and_radius(
    x: np.ndarray, intensity: np.ndarray
) -> tuple[float, float]:
    """Calculate FWHM and 1/e² radius from intensity profile.

    Args:
        x: Position array in meters.
        intensity: Intensity array.

    Returns:
        Tuple of (fwhm_um, radius_1e2_um) in micrometers.
    """
    max_intensity = np.max(intensity)
    half_max = max_intensity / 2
    threshold = max_intensity / np.exp(2)

    # Calculate FWHM
    above_half_max = intensity > half_max
    if np.any(above_half_max):
        crossings = np.where(np.diff(above_half_max.astype(int)))[0]
        if len(crossings) >= 2:
            fwhm = x[crossings[-1]] - x[crossings[0]]
            fwhm_um = fwhm * LENGTH_CONVERSION_UM
        else:
            fwhm_um = np.nan
    else:
        fwhm_um = np.nan

    # Calculate 1/e² radius
    above_threshold = intensity > threshold
    if np.any(above_threshold):
        crossings = np.where(np.diff(above_threshold.astype(int)))[0]
        if len(crossings) >= 2:
            radius_1e2 = (x[crossings[-1]] - x[crossings[0]]) / 2
            radius_1e2_um = radius_1e2 * LENGTH_CONVERSION_UM
        else:
            radius_1e2_um = np.nan
    else:
        radius_1e2_um = np.nan

    return fwhm_um, radius_1e2_um


def _add_plot_annotations(
    x: np.ndarray,
    intensity: np.ndarray,
    fwhm_um: float,
    radius_1e2_um: float,
) -> None:
    """Add FWHM and 1/e² radius annotations to the plot.

    Args:
        x: Position array in meters.
        intensity: Intensity array.
        fwhm_um: FWHM in micrometers.
        radius_1e2_um: 1/e² radius in micrometers.
    """
    if not np.isnan(fwhm_um):
        half_max_normalized = 0.5
        fwhm_positions = x[
            np.where(
                np.diff(
                    (intensity / np.max(intensity) > half_max_normalized).astype(int)
                )
            )[0]
        ]
        if len(fwhm_positions) >= 2:
            plt.axvline(
                x=fwhm_positions[0] * LENGTH_CONVERSION_UM,
                color="red",
                linestyle=":",
                alpha=0.7,
                label=f"FWHM start ({fwhm_positions[0] * LENGTH_CONVERSION_UM:.1f} μm)",
            )
            plt.axvline(
                x=fwhm_positions[-1] * LENGTH_CONVERSION_UM,
                color="red",
                linestyle=":",
                alpha=0.7,
                label=f"FWHM end ({fwhm_positions[-1] * LENGTH_CONVERSION_UM:.1f} μm)",
            )
            plt.axhline(y=0.5, color="red", linestyle=":", alpha=0.5)

    if not np.isnan(radius_1e2_um):
        threshold_normalized = 1 / np.exp(2)
        radius_positions = x[
            np.where(
                np.diff(
                    (intensity / np.max(intensity) > threshold_normalized).astype(int)
                )
            )[0]
        ]
        if len(radius_positions) >= 2:
            plt.axvline(
                x=radius_positions[0] * LENGTH_CONVERSION_UM,
                color="green",
                linestyle=":",
                alpha=0.7,
                label=f"1/e² start ({radius_positions[0] * LENGTH_CONVERSION_UM:.1f} μm)",
            )
            plt.axvline(
                x=radius_positions[-1] * LENGTH_CONVERSION_UM,
                color="green",
                linestyle=":",
                alpha=0.7,
                label=f"1/e² end ({radius_positions[-1] * LENGTH_CONVERSION_UM:.1f} μm)",
            )
            plt.axhline(y=1 / np.exp(2), color="green", linestyle=":", alpha=0.5)


def plot_1d_transverse_laser_profile(
    w0: float,
    wavelength: float,
    tau_fwhm: float,
) -> None:
    """Plot the laser profile in x and calculate intensity statistics.

    Args:
        w0: Laser waist in meters.
        wavelength: Laser wavelength in micrometers.
        tau_fwhm: Laser duration FWHM in seconds.

    Returns:
        None. Results are printed to stdout and plot is displayed.
    """
    transverse_profile = GaussianTransverseProfile(
        waist=w0, lambda0=wavelength * WAVELENGTH_CONVERSION
    )
    x_range = 3 * w0
    x_points = 1000
    x = np.linspace(-x_range, x_range, x_points)

    profile = np.zeros_like(x)
    efield = np.zeros_like(x)
    intensity = np.zeros_like(x)

    tau = tau_fwhm * FWHM_TO_TAU_FACTOR
    gaussian_pulse = GaussianLaser(
        a0=1, waist=w0, tau=tau, z0=0, lambda0=wavelength * WAVELENGTH_CONVERSION
    )

    impedance = np.sqrt(mu_0 / epsilon_0)

    for i in range(len(x)):
        profile[i] = transverse_profile.evaluate(x=x[i], y=0, z=0)
        efield[i] = gaussian_pulse.E_field(x=x[i], y=0, z=0, t=0)[0]
        intensity[i] = np.square(efield[i]) / (2 * impedance)

    fwhm_um, radius_1e2_um = _calculate_fwhm_and_radius(x, intensity)

    # Print the calculated values
    print(" - Transverse Laser Statistics - ")
    print(f"Intensity FWHM: {fwhm_um:.2f} μm")
    print(f"Intensity sigma: {fwhm_um/FWHM_TO_SIGMA_FACTOR:.2f} μm")
    print(f"Intensity 1/e² radius: {radius_1e2_um:.2f} μm")
    print(f"Expected 1/e² radius from w0: {w0 * LENGTH_CONVERSION_UM:.2f} μm")
    print()

    plt.plot(
        x * LENGTH_CONVERSION_UM,
        intensity / np.max(intensity),
        label="Calculated Intensity",
    )

    _add_plot_annotations(x, intensity, fwhm_um, radius_1e2_um)

    plt.ylabel("Normalized Intensity Amplitude")
    plt.xlabel("x (μm)")
    plt.title("Transverse Laser Profile")
    plt.legend()
    plt.show()


def _calculate_standard_deviation(
    amplitude_array: np.ndarray,
    axis_array: np.ndarray,
    axis_average: float | None = None,
) -> float:
    """Calculate standard deviation weighted by amplitude.

    Args:
        amplitude_array: Array of amplitudes for weighting.
        axis_array: Array of axis values.
        axis_average: Pre-calculated average of axis values.

    Returns:
        Weighted standard deviation.
    """
    if axis_average is None:
        axis_average = np.average(axis_array, weights=amplitude_array)
    return np.sqrt(
        np.average((axis_array - axis_average) ** 2, weights=amplitude_array)
    )


def _calculate_temporal_characteristics(
    t: np.ndarray, intensity_t: np.ndarray
) -> tuple[float, float, float]:
    """Calculate temporal characteristics of the laser pulse.

    Args:
        t: Time array in seconds.
        intensity_t: Temporal intensity array.

    Returns:
        Tuple of (fwhm_fs, width_1e2_fs, sigma_fs) in femtoseconds.
    """
    max_envelope = np.max(intensity_t)
    half_max = max_envelope / 2
    threshold_1e2 = max_envelope / np.exp(2)

    # Find FWHM
    above_half_max = intensity_t >= half_max
    if np.any(above_half_max):
        crossings = np.where(np.diff(above_half_max.astype(int)))[0]
        if len(crossings) >= 2:
            fwhm_fs = (t[crossings[-1]] - t[crossings[0]]) * TIME_CONVERSION_FS
        else:
            fwhm_fs = np.nan
    else:
        fwhm_fs = np.nan

    # Find 1/e² width
    above_threshold = intensity_t > threshold_1e2
    if np.any(above_threshold):
        crossings = np.where(np.diff(above_threshold.astype(int)))[0]
        if len(crossings) >= 2:
            width_1e2_fs = (t[crossings[-1]] - t[crossings[0]]) * TIME_CONVERSION_FS / 2
        else:
            width_1e2_fs = np.nan
    else:
        width_1e2_fs = np.nan

    # Calculate sigma
    sigma_fs = (
        _calculate_standard_deviation(amplitude_array=intensity_t, axis_array=t)
        * TIME_CONVERSION_FS
    )

    return fwhm_fs, width_1e2_fs, sigma_fs


def plot_1d_longitudinal_laser_profile(
    w0: float,
    wavelength: float,
    tau_fwhm: float,
) -> None:
    """Plot the laser profile in z and calculate temporal statistics.

    Args:
        w0: Laser waist in meters.
        wavelength: Laser wavelength in micrometers.
        tau_fwhm: Laser duration FWHM in seconds.

    Returns:
        None. Results are printed to stdout and plot is displayed.
    """
    tau = tau_fwhm * FWHM_TO_TAU_FACTOR
    z0 = -40e-6
    gaussian_pulse = GaussianLaser(
        a0=1, waist=w0, tau=tau, z0=z0, lambda0=wavelength * WAVELENGTH_CONVERSION
    )

    z = np.linspace(-100e-6, 0e-6, 20000)
    t = np.linspace(-100e-6 / c, 100e-6 / c, 20000)

    envelope = np.zeros_like(z)
    values = np.zeros_like(z)
    envelope_t = np.zeros_like(t)

    my_envelope = np.zeros_like(z)
    my_values = np.zeros_like(z)
    my_envelope_t = np.zeros_like(t)

    for i in range(len(z)):
        envelope[i] = np.abs(gaussian_pulse.longitudinal_profile.evaluate(z=z[i], t=0))
        values[i] = gaussian_pulse.E_field(x=0, y=0, z=z[i], t=0)[0]
    my_envelope = np.exp(-((z - z0) ** 2) / (c**2 * tau**2))
    my_values = my_envelope * np.cos(2.0 * pi / wavelength * (z - z0))

    for i in range(len(t)):
        envelope_t[i] = np.abs(
            gaussian_pulse.longitudinal_profile.evaluate(z=z0, t=t[i])
        )
    my_envelope_t = np.exp(-((t) ** 2) / (tau**2))

    fwhm_fs, width_1e2_fs, sigma_fs = _calculate_temporal_characteristics(
        t, np.square(envelope_t)
    )
    my_fwhm_fs, my_width_1e2_fs, my_sigma_fs = _calculate_temporal_characteristics(
        t, np.square(my_envelope_t)
    )

    # Print the temporal characteristics
    print(" - Longitudinal Laser Statistics - ")
    print(f"FBPIC Intensity FWHM:\t{fwhm_fs:.2f} fs")
    print(f"FBPIC Intensity 1/e² width:\t{width_1e2_fs:.2f} fs")
    print(f"FBPIC Intensity sigma:\t{sigma_fs:.2f} fs")
    print(f"Manual Intensity FWHM:\t{my_fwhm_fs:.2f} fs")
    print(f"Manual Intensity 1/e² width:\t{my_width_1e2_fs:.2f} fs")
    print(f"Manual Intensity sigma:\t{my_sigma_fs:.2f} fs")
    print(f"Input tau:\t\t{tau * TIME_CONVERSION_FS:.2f} fs")
    print()

    plt.plot(
        z * LENGTH_CONVERSION_UM, values / np.max(values), label="FBPIC Electric Field"
    )
    plt.plot(z * LENGTH_CONVERSION_UM, my_values, label="Manual Electric Field")
    plt.plot(
        z * LENGTH_CONVERSION_UM, envelope / np.max(envelope), label="FBPIC Envelope"
    )
    plt.plot(z * LENGTH_CONVERSION_UM, my_envelope, label="Manual Envelope")
    plt.xlabel("z (μm)")
    plt.ylabel("Normalized Amplitude")
    plt.title("Longitudinal Laser Profile")
    plt.legend()
    plt.show()


def main() -> None:
    """Script entry point for calculating a0 for different laser systems.

    Returns:
        None.
    """
    if HTU:
        print("For HTU:")
        laser_energy: float = 2.5  # J
        tau_fwhm: float = 38.0e-15  # Laser duration FWHM
        w0: float = 28e-6  # Laser waist, 0.8493218 x FWHM, measured on 8/28/2025 on HTU
        wavelength: float = 0.800  # um
        calculate_a0(
            laser_energy=laser_energy,
            wavelength=wavelength,
            tau_fwhm=tau_fwhm,
            w0=w0,
        )

        plot_1d_longitudinal_laser_profile(w0, wavelength, tau_fwhm=tau_fwhm)
        plot_1d_transverse_laser_profile(w0, wavelength, tau_fwhm=tau_fwhm)

    if EKSPLA:
        print("For the UltraFlux FF1201k-F8-CEP:")
        laser_energy: float = 0.120  # J
        tau_fwhm: float = 8.0e-15  # Laser duration
        w0: float = 9.1e-6  # Laser waist
        wavelength: float = 0.900  # um
        calculate_a0(
            laser_energy=laser_energy,
            wavelength=wavelength,
            tau_fwhm=tau_fwhm,
            w0=w0,
        )

    if ANTIPOV:
        print("For the Antipov et.al. 2021 paper from DESY:")
        laser_energy: float = 2.45 * 1.3  # J
        tau_fwhm: float = 34e-15
        # w0_original: float = 21.233e-6
        w0_wide: float = 30.028e-6  # Laser waist, 0.8493218 x FWHM
        wavelength: float = 0.800  # um
        calculate_a0(
            laser_energy=laser_energy,
            wavelength=wavelength,
            tau_fwhm=tau_fwhm,
            w0=w0_wide,
        )

    if INVERSION:
        print("For our beautiful laser beam:")
        laser_energy: float = 11  # J
        tau_fwhm: float = 55.00e-15  # fs
        w0: float = 38.92e-6  # Laser waist, 0.8493218 x FWHM
        wavelength: float = 0.800  # um
        calculate_a0(
            laser_energy=laser_energy,
            wavelength=wavelength,
            tau_fwhm=tau_fwhm,
            w0=w0,
        )

        plot_1d_longitudinal_laser_profile(w0, wavelength, tau_fwhm=tau_fwhm)


if __name__ == "__main__":
    main()
