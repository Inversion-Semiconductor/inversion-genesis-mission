"""
simulation_setup_tools.py

Module containing functions used by FBPIC simulations upon simulation setup.
These are blocks of code that existed in identical form between many simulations,
and so they were moved here to reduce identical code.
"""

from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fbpic.main import Simulation

import sys
import time
import numpy as np
from pathlib import Path
from scipy.constants import c, e, m_e, epsilon_0, pi

LASER_TAU_TO_SIGMA = 1.0 / np.sqrt(2)
SIGMA_TO_FWHM_FIELD = 2.0 * np.sqrt(2.0 * np.log(2.0))
SIGMA_TO_FWHM_INTENSITY = 2.0 * np.sqrt(np.log(2.0))


def start_simulation_single(simulation_setup: Callable) -> None:
    """
    Starts a single simulation defined entirely within the `simulation_setup` function.

    Args:
        simulation_setup: Function that is used to setup a single simulation, returns a
            tuple of an FBPIC Simulation object, and a float representing the interaction time
    """
    try:
        # Set up the simulation
        print("Setting up simulation...")
        sim: Simulation
        T_interact: float

        sim, T_interact = simulation_setup()
        print("Simulation setup complete")

        # Number of iterations to perform
        N_step: int = int(T_interact / sim.dt) + 1
        print(f"Running {N_step} simulation steps...")

        # Run the simulation
        sim.step(N_step)
        print("Simulation completed successfully!")

    except Exception as exception:
        print(f"ERROR in simulation: {exception}")
        import traceback

        traceback.print_exc()
        raise


def start_simulation_1d_scan(
    simulation_setup: Callable,
    scan_cases: list[tuple],
    case_selection: int = 0,
    do_logging: bool = False,
) -> None:
    """
    Starts a set of simulations defined by the `simulation_setup` function and the
    available cases of the `scan_cases` list.

    Args:
        simulation_setup: Function that is used to setup a single simulation, returns a
            tuple of an FBPIC Simulation object, and a float representing the interaction time
        scan_cases: List where each entry is a tuple of four entries:
            1. a string that describes the case
            2. a list of values for that case
            3. a string for the save directory name for this case
            4. a string for the keyword argument of `simulation_setup` for this case
        case_selection: Index within the `scan_cases` list for which case to select.  Defaults
            to 0 to select the first (and potentially only) element
        do_logging: boolean to determine if the output should be logged to a file.

    Returns:

    """
    try:
        description: str
        cases: list[Any]
        directory: str
        setup_key: str

        description, cases, directory, setup_key = scan_cases[case_selection]

        Path(directory).mkdir(parents=True, exist_ok=True)
        if do_logging:
            sys.stdout = open(f"{directory}/output.log", "w")
            sys.stderr = open(f"{directory}/error.log", "w")

        for case in cases:
            print(f"{directory}: Case {setup_key} = {case}")

            # Set up the simulation
            print("Setting up simulation...")
            sim: Simulation
            T_interact: float
            kwargs: dict = {}

            kwargs.update({setup_key: case})

            sim, T_interact = simulation_setup(
                set_name=directory, case_name=str(case), **kwargs
            )
            print("Simulation setup complete")

            # Number of iterations to perform
            N_step = int(T_interact / sim.dt) + 1
            print(f"Running {N_step} simulation steps...")

            # Run the simulation
            start_time: float = time.time()
            sim.step(N_step, show_progress=not do_logging)
            print(f" -Finished in {(time.time() - start_time) / 60} min")
            print("Simulation completed successfully!")

    except Exception as exception:
        print(f"ERROR in simulation: {exception}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Restore stdout/stderr if they were redirected
        if do_logging:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


def calculate_laser_tau_from_fwhm_intensity(
    fwhm_intensity: float,
) -> float:
    """Calculate pulse duration, as would be passed to FBPIC, from FWHM-intensity.

    Args:
        fwhm_intensity: FWHM-intensity, in any unit.

    Returns:
        tau: Pulse duration, as would be passed to FBPIC, in the same units as the input fwhm_intensity.
    """
    return fwhm_intensity / SIGMA_TO_FWHM_INTENSITY / LASER_TAU_TO_SIGMA


def calculate_fwhm_intensity_from_laser_tau(
    laser_tau: float,
) -> float:
    """Calculate FWHM-intensity from pulse duration, as would be passed to FBPIC.

    Args:
        laser_tau: Pulse duration, as would be passed to FBPIC, in any unit.

    Returns:
        fwhm_intensity: FWHM-intensity in the same units as the input fbpic_tau.
    """
    return laser_tau * SIGMA_TO_FWHM_INTENSITY * LASER_TAU_TO_SIGMA


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
        w0: Laser focus waist in meters.
        tau_fwhm: Temporal duration FWHM-intensity, in seconds.

    Returns:
        Laser energy in Joules.
    """
    # Calculate wavenumber
    wavenumber = 2 * pi / (wavelength * 1e-6)

    # From a0 to peak field
    peak_field = a0 * (m_e * c**2 * wavenumber / e)

    # From peak field to peak intensity
    peak_intensity = peak_field**2 * c * epsilon_0 / 2

    # From peak intensity to peak power
    peak_power = peak_intensity * pi * w0**2 / 2

    # From peak power to laser energy
    tau = calculate_laser_tau_from_fwhm_intensity(tau_fwhm)
    laser_energy = peak_power * tau * np.sqrt(pi / 2.0)

    # for direct calculation (via Mathematica, now verified the above to be correct)
    # laser_energy = np.pi * np.sqrt(np.pi / 2.0) / 4.0 * epsilon_0 * peak_field ** 2 * w0 ** 2 * c * calculate_laser_tau_from_fwhm_intensity(tau_fwhm)

    return laser_energy


def calculate_laser_a0_from_energy(
    laser_energy: float,
    wavelength: float,
    w0: float,
    tau_fwhm: float,
    disp: bool = False,
) -> float:
    """Calculate a0 and optionally display other relevant laser parameters.

    Args:
        laser_energy: Laser energy in Joules.
        wavelength: Laser wavelength in micrometers.
        w0: Laser focus waist in meters.
        tau_fwhm: Temporal duration FWHM-intensity in seconds.
        disp: bool Whether to display the results to stdout. Defaults to False.

    Returns:
        a0_normalized: Normalized laser intensity parameter.
    """
    tau = calculate_laser_tau_from_fwhm_intensity(tau_fwhm)

    # From laser energy to peak power
    peak_power = laser_energy / tau / np.sqrt(pi / 2.0)

    # From peak power to peak intensity
    peak_intensity = peak_power / (pi * w0**2 / 2)

    # From peak intensity to peak field
    peak_field = np.sqrt(peak_intensity / (c * epsilon_0) * 2.0)

    # From peak field to wavenumber
    wavenumber = 2 * pi / (wavelength * 1e-6)

    # From peak field to a0
    a0 = peak_field / (m_e * c**2 * wavenumber / e)

    if disp:
        print("Laser Parameters:")
        print(f" input tau     {tau * 1e15:.4f} fs")
        print(f" total_power   {peak_power * 1e-12:.4f} TW")
        print(f" peak_intensity {peak_intensity / 100**2:.4e} W/cm^2")
        print(f" peak_field    {peak_field:.4e} V/m")
        print(f" a0 = {a0}")
        # TODO: add comment explaining what the engineering factor is
        a0_engineering = np.sqrt(7.3e-19 * wavelength**2 * peak_intensity / 100**2)
        print(f"engineering formula a0 = {a0_engineering}")
        print()

    return a0


def calculate_matched_spot_size_from_np(
    plasma_density: float,
) -> float:
    plasma_density_1e18 = plasma_density / 1e18
    b3 = 50e-6 / np.sqrt(5)
    w_0 = b3 / np.sqrt(plasma_density_1e18)
    return w_0


def calculate_acceleration_gradient(a0: float, n_p: float) -> float:
    """
    Get the acceleration gradient for a given a0 (normalized laser amplitude) and np (plasma density).

    Args:
        a0: Normalized laser intensity.
        n_p: Plasma density (m^-3).

    Returns:
        Acceleration gradient (V/m).
    """
    return 48.0e9 * np.sqrt(a0 * n_p * 1e-24)


def calculate_plasma_wavelength(n_p: float) -> float:
    """
    Get the plasma wavelength for a given plasma density.

    Args:
        n_p: Plasma density (m^-3).

    Returns:
        Plasma wavelength (m).
    """
    return 3.3e7 / np.sqrt(n_p)
