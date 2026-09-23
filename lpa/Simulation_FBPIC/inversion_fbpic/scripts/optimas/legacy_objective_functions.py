"""Legacy objective functions for electron beam analysis.

This module contains older versions of objective functions that are outdated and
should not live in the main objective function module found in `utils`. These
functions are maintained for comparison purposes and can be used with the
comparison script `compare_objective_functions.py` to evaluate different
objective function implementations.
"""

import configparser
import os
import traceback
from typing import Optional

import numpy as np
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c, e, m_e

import inversion_fbpic.utils.analysis as an
from inversion_fbpic.utils.optimas_analysis import (
    storage_cleanup_routine,
    calculate_energy_parameters,
    analyzed_params,
)

SPECIES_NAME: str = "n_elec"
DO_STORAGE_CLEANUP: bool = False
DEBUG: bool = False  # If set to True, additional plots will be generated

# Parameters for HTU Optimas Analysis
# Default minimum z [m] to be selected for analysis.
MIN_Z: Optional[float] = None
# Default minimum momentum [mc] to be selected for analysis. 20.54 = 10 MeV
MIN_UZ: float = 100


@analyzed_params("charge", "charge_weighted", "energy_mean", "energy_std")
def analyze_htu_simulation(
    simulation_directory: str,
    output_params: dict[str, float],
    file_tree: str = "lab_diags/hdf5",
    min_z: Optional[float] = MIN_Z,
    min_uz: Optional[float] = MIN_UZ,
) -> dict[str, float]:
    """Analyze the simulation output for HTU simulations.

    This function analyzes electron beams produced from HTU simulations by:
    1. Reading the last diagnostic dump in the simulation_directory
    2. Cropping out non-relativistic particles based on z-position and momentum
    3. Calculating the charge and energy spread
    4. Calculating a weighted charge based on energy proximity to 100 MeV
    5. Computing the objective function using weighted charge and energy spread

    The objective function rewards high charge from particles close to 100 MeV
    and low energy spread.

    Args:
        simulation_directory: Base path of the simulation folder where the output
            was generated.
        output_params: Dictionary where the value of the objectives and analyzed
            parameters will be stored. There is one entry per parameter, where the
            key is the name of the parameter given by the user.
        file_tree: Path from the simulation directory to the h5 files.
            Default is "lab_diags/hdf5".
        min_z: Minimum z-value to consider in meters. Default is MIN_Z.
        min_uz: Minimum normalized z-momentum to consider. Default is MIN_UZ.

    Returns:
        The `output_params` dictionary with the results from the analysis,
        including:
            - f: Objective function value
            - charge: Total charge in pC
            - charge_weighted: Energy-weighted charge in pC
            - energy_mean: Mean energy in MeV
            - energy_std: Energy spread (RMS) in MeV
    """
    print(f"ANALYSIS: Starting analysis of {simulation_directory}")
    try:
        # Open simulation diagnostics.
        diags_path: str = os.path.join(simulation_directory, file_tree)
        print(f"ANALYSIS: Looking for diagnostics at {diags_path}")

        if not os.path.exists(diags_path):
            print(f"ANALYSIS ERROR: Diagnostics path {diags_path} does not exist")
            output_params["f"] = 0
            output_params["charge"] = 0
            output_params["charge_weighted"] = 0
            output_params["energy_mean"] = 0
            output_params["energy_std"] = 0
            return output_params

        diagnostics: LpaDiagnostics = LpaDiagnostics(diags_path)

        # Get beam particles with `u_z >= 20.54 (10 MeV)`
        # Use the last available iteration instead of -1
        available_iterations = diagnostics.iterations
        last_iteration: int = available_iterations[-1]
        print(f"ANALYSIS: Using iteration {last_iteration} (last available)")

        ux, uy, uz, w = diagnostics.get_particle(
            ["ux", "uy", "uz", "w"],
            iteration=last_iteration,
            select={"uz": [min_uz, None], "z": [min_z, None]},
        )

        # Analyze distribution and fill in the output data.
        if len(w) < 2:
            raise ValueError(
                "Need at least two particles to calculate energy parameters"
            )
        else:
            # Convert charge to pC.
            charge_pc: float = w.sum() * e * 1e12

            # Calculate weighted charge based on energy proximity to 100 MeV
            # Convert uz to energy in MeV: E = (gamma - 1) * m_e * c^2 / e / 1e6
            gamma: np.ndarray = np.sqrt(
                1 + uz**2
            )  # For ultrarelativistic particles, uz >> 1, so gamma ≈ uz
            energy_mev: np.ndarray = (gamma - 1) * m_e * c**2 / e / 1e6

            # Calculate energy weighting factor (Gaussian-like, peaks at 100 MeV)
            target_energy_mev: float = 100.0
            energy_sigma: float = 10.0
            energy_weight: np.ndarray = np.exp(
                -((energy_mev - target_energy_mev) ** 2) / (2 * energy_sigma**2)
            )

            # Apply energy weighting to particle weights
            weighted_weights: np.ndarray = w * energy_weight
            weighted_charge_pc: float = weighted_weights.sum() * e * 1e12

            analysis_dict: dict[str, float] = calculate_energy_parameters(ux, uy, uz, w)

            # Objective function using weighted charge: (weighted_charge)^2 / sqrt(energy_spread)
            # This rewards high charge from particles close to 100 MeV and low energy spread
            output_params["f"] = np.square(weighted_charge_pc) / np.sqrt(
                analysis_dict["energy_std_mev"] + 1
            )
            output_params["charge"] = charge_pc
            output_params["charge_weighted"] = weighted_charge_pc
            output_params["energy_mean"] = analysis_dict["central_energy_mev"]
            output_params["energy_std"] = analysis_dict["energy_std_mev"]

    except Exception as exception:
        print(f"ANALYSIS ERROR: {exception}")
        traceback.print_exc()
        # Set default values for failed analysis
        output_params["f"] = 0
        output_params["charge"] = 0
        output_params["charge_weighted"] = 0
        output_params["energy_mean"] = 0
        output_params["energy_std"] = 0

    return output_params


@analyzed_params(
    "charge",
    "energy_mean",
    "energy_spread",
    "bunch_length",
    "emittance_x",
    "emittance_y",
)
def analyze_pax_beam(
    simulation_directory: str,
    output_params: dict[str, float],
    file_tree: str = "lab_diags/hdf5",
    do_storage_cleanup: bool = DO_STORAGE_CLEANUP,
) -> dict[str, float]:
    """Analyze chirped beam simulations with slice-normalized weighting.  Evaluates
    the objective function to find beams that have a large peak current when passed
    through a chicane's R56 transformation, as well as having a modest energy spread
    and input laser energy.

    This function analyzes electron beams from chirped beam simulations by:
    1. Reading all diagnostic dumps in the simulation_directory
    2. Filtering particles by energy and charge thresholds
    3. Finding the beam region using current profile analysis
    4. Calculating slice-normalized weights based on charge per slice energy spread
    5. Performing an R56 scan to find peak current
    6. Calculating an objective function based on peak current, R56 value,
       energy spread, and relative laser energy

    The objective function rewards high peak current, optimal R56 values,
    low energy spread, and efficient laser energy usage.

    Args:
        simulation_directory: Base path of the simulation folder where the output
            was generated.
        output_params: Dictionary where the value of the objectives and analyzed
            parameters will be stored. There is one entry per parameter, where the
            key is the name of the parameter given by the user.
        file_tree: Path from the simulation directory to the h5 files.
            Default is "lab_diags/hdf5".
        do_storage_cleanup: If True, will delete all files that aren't the one
            with the highest objective function. Defaults to DO_STORAGE_CLEANUP.

    Returns:
        The `output_params` dictionary with the results from the analysis,
        including:
            - f: Objective function value
            - charge: Total charge in pC
            - energy_mean: Mean energy in GeV
            - energy_spread: Energy spread (RMS) in GeV
            - bunch_length: Bunch length in μm
            - emittance_x: Horizontal emittance in μm⋅rad
            - emittance_y: Vertical emittance in μm⋅rad
    """
    print(f"ANALYSIS: Starting analysis of {simulation_directory}")

    # Bounds for acceptable beam energy and charge
    minimum_mean_energy = 0.8  # GeV
    minimum_beam_charge = 30.0  # pC
    min_uz_load = 400

    # Specify a range of R56 values to scan, can assume 0 to 100 um
    r56_values = np.linspace(0, 100e-6, 200)  # 0 to 100 μm in 200 steps

    # Set some default values in case the analysis fails.
    output_params["f"] = 0
    output_params["charge"] = 0
    output_params["energy_mean"] = 0
    output_params["energy_spread"] = 0
    output_params["bunch_length"] = 0
    output_params["emittance_x"] = 0
    output_params["emittance_y"] = 0

    try:
        # Open simulation diagnostics.
        diags_path: str = os.path.join(simulation_directory, file_tree)
        print(f"ANALYSIS: Looking for diagnostics at {diags_path}")

        if not os.path.exists(diags_path):
            print(f"ANALYSIS ERROR: Diagnostics path {diags_path} does not exist")
            return output_params

        diagnostics: LpaDiagnostics = LpaDiagnostics(diags_path)
        available_iterations = diagnostics.iterations

        # Loop through diags files to find the largest objective function
        previous_best_obj = 0
        best_iteration: Optional[int] = None
        for iteration in available_iterations:
            print(f"ANALYSIS: Analyzing iteration {iteration}")

            x, y, z, ux, uy, uz, w, q = diagnostics.get_particle(
                ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
                iteration=iteration,
                select={"uz": (min_uz_load, None)},
                species=SPECIES_NAME,
                plot=False,
            )

            # If beam contains at least the minimum amount of charge then continue analysis
            if not (len(w) > 0 and np.sum(w * q) * -1e12 > minimum_beam_charge):
                continue

            # Convert uz to energy in GeV for filtering
            gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
            energy_mev = (gamma - 1) * m_e * c**2 / e / 1e6
            energy_mean_gev = (
                np.average(energy_mev, weights=w) / 1000.0
            )  # Convert MeV to GeV

            # Need a minimum energy to consider
            if not energy_mean_gev > minimum_mean_energy:
                continue

            # Find z region of beam by current and crop to a 1/e^2 region
            current, axis = an.calculate_current(
                z=z, ux=ux, uy=uy, uz=uz, w=w, q=q, bins=200
            )
            threshold = float(np.max(current)) / np.exp(2)
            max_loc = axis[np.argmax(current)]
            z_region = None

            above_threshold = current > threshold
            if np.any(above_threshold):
                crossings = np.where(np.diff(above_threshold.astype(int)))[0]

                # Majority of cases will have two primary crossings
                if len(crossings) >= 2:
                    left_edge = float(axis[crossings[0]])
                    right_edge = float(axis[crossings[-1]])
                    z_region = [left_edge, right_edge]

                else:
                    z_region = [float(axis[0]), float(axis[-1])]

                # Next, enforce that the region is on either side of the global max
                if z_region[0] > max_loc:
                    z_region[0] = float(axis[0])
                if z_region[1] < max_loc:
                    z_region[1] = float(axis[-1])

                z_region = tuple(z_region)

            if z_region is None:
                continue

            # Find normalized weights through charge in slice divided by slice energy spread
            slice_energy_spread, axis_2, normalized = an.calculate_slice_energy_spread(
                diag_folder=diags_path,
                species_name=SPECIES_NAME,
                iteration=iteration,
                selection={"uz": (min_uz_load, None), "z": z_region},
                bins=200,
            )

            # Plot lineouts if in DEBUG mode
            if DEBUG:
                an.plot_slice_statistics(
                    current_axis=axis,
                    current_values=current,
                    region_axis=axis_2,
                    region_slice_energy_spread=slice_energy_spread,
                    region_normalized_weights=normalized,
                )

            # Reload the beam using the z_region to define the z cropping range
            (
                x_cropped,
                y_cropped,
                z_cropped,
                ux_cropped,
                uy_cropped,
                uz_cropped,
                w_cropped,
                q,
            ) = diagnostics.get_particle(
                ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
                iteration=iteration,
                select={"uz": (min_uz_load, None), "z": z_region},
                species=SPECIES_NAME,
                plot=False,
            )

            # Map each particle to its z slice and get the corresponding normalized value
            # Use interpolation to map particle z positions to normalized values
            # axis_2 contains the z positions (bin centers) for which we have normalized values
            # For particles outside the range of axis_2, use the nearest value
            normalized_values_per_particle = np.interp(
                z_cropped,
                axis_2,
                normalized,
                left=normalized[0] if len(normalized) > 0 else 0.0,
                right=normalized[-1] if len(normalized) > 0 else 0.0,
            )

            # Create normalized weights: w * normalized_value for each particle
            normalized_weights = w_cropped * normalized_values_per_particle

            # Using the normalized values as weights, calculate the energy spread of the beam
            energy_spread_analysis = calculate_energy_parameters(
                ux_cropped, uy_cropped, uz_cropped, normalized_weights
            )
            energy_spread_normalized = energy_spread_analysis["energy_std_mev"]
            mean_energy_normalized = float(energy_spread_analysis["central_energy_mev"])
            if DEBUG:
                print("Mean Energy:", mean_energy_normalized)
                print("Full Energy Spread:", energy_spread_normalized)
                print(
                    " % Energy Spread:  ",
                    energy_spread_normalized / mean_energy_normalized,
                )

            # Now calculate the average slice energy spread, using the normalized
            #  array for weighting the slices.
            slice_energy_spread_average = float(
                np.average(slice_energy_spread, weights=normalized)
            )
            if DEBUG:
                print("Slice Energy Spread:", slice_energy_spread_average)

            # Calculate the average uz value for R56 calculation using the normalized weights
            uz_mean = np.average(uz_cropped, weights=normalized_weights)

            # Find the peak possible current using an R56 scan
            peak_currents = np.zeros(len(r56_values))
            for i, r56 in enumerate(r56_values):
                # For each R56 setting:
                # Use normalized momentum uz (not energy) for R56 formula
                z_modified = z_cropped + r56 * ((uz_cropped - uz_mean) / uz_mean)

                try:
                    # Calculate current profile with modified z positions
                    current, z_axis = an.calculate_current(
                        z_modified,
                        ux_cropped,
                        uy_cropped,
                        uz_cropped,
                        w_cropped,
                        q,
                        bins=1000,
                    )
                    peak_currents[i] = np.max(current)

                except Exception:
                    peak_currents[i] = np.nan

            peak_possible_current = np.nanmax(peak_currents)
            if np.isnan(peak_possible_current):
                continue
            if DEBUG:
                print("Peak Possible Current:", peak_possible_current)

            # Calculate objective function for this output file, nerfing results for poor energy spread or peak current
            objective_function = (
                float(peak_possible_current) ** 0.5
                * mean_energy_normalized
                / (energy_spread_normalized * slice_energy_spread_average)
            )
            if peak_possible_current < 20e3:
                objective_function = objective_function / 1000
            if energy_spread_normalized / mean_energy_normalized > 0.05:
                objective_function = objective_function / 1000

            # Only update if we got a valid objective function that is larger than previously
            if (
                not np.isnan(peak_possible_current)
                and objective_function > previous_best_obj
            ):
                previous_best_obj = objective_function
                best_iteration = int(iteration)
                output_params["f"] = objective_function

                # Calculate beam statistics using analysis functions
                output_params["charge"] = np.sum(w_cropped * q) * -1e12
                analysis_results = an.analyze_beam(
                    x_cropped,
                    y_cropped,
                    z_cropped,
                    ux_cropped,
                    uy_cropped,
                    uz_cropped,
                    normalized_weights,
                    q,
                    bins=100,
                )
                output_params["energy_mean"] = (
                    mean_energy_normalized / 1000.0
                )  # Convert to GeV
                output_params["energy_spread"] = (
                    energy_spread_normalized / 1000.0
                )  # Convert to GeV
                output_params["bunch_length"] = (
                    analysis_results["beam_sizes"]["sigma_z_m"] * 1e6
                )  # Convert to μm
                output_params["emittance_x"] = (
                    analysis_results["emittance"]["x"] * 1e6
                )  # Convert to μm⋅rad
                output_params["emittance_y"] = (
                    analysis_results["emittance"]["y"] * 1e6
                )  # Convert to μm⋅rad

        # After evaluating all iterations, delete all .h5 files except the one
        # corresponding to the best iteration to conserve disk space.
        if do_storage_cleanup and DO_STORAGE_CLEANUP:
            storage_cleanup_routine(
                diags_path=diags_path, iteration_to_keep=best_iteration
            )
        elif do_storage_cleanup and not DO_STORAGE_CLEANUP:
            print(
                "ANALYSIS WARNING: unlock file storage cleanup within `optimas_analysis.py` to proceed..."
            )

    except Exception as exception:
        print(f"ANALYSIS ERROR: {exception}")
        traceback.print_exc()

    return output_params


def analyze_chirped_beam_simulation(
    simulation_directory: str,
    output_params: dict[str, float],
    file_tree: str = "lab_diags/hdf5",
    do_storage_cleanup: bool = DO_STORAGE_CLEANUP,
) -> dict[str, float]:
    """Analyze chirped beam simulation using legacy objective function.

    This is a legacy version of the chirped beam analysis function. It evaluates
    electron beams by:
    1. Reading all diagnostic dumps in the simulation_directory
    2. Filtering particles by energy and charge thresholds
    3. Performing an R56 scan to find peak current
    4. Calculating an objective function based on peak current, energy spread,
       and relative laser energy

    The objective function rewards high peak current, low energy spread, and
    efficient laser energy usage.

    Args:
        simulation_directory: Base path of the simulation folder where the output
            was generated.
        output_params: Dictionary where the value of the objectives and analyzed
            parameters will be stored. There is one entry per parameter, where the
            key is the name of the parameter given by the user.
        file_tree: Path from the simulation directory to the h5 files.
            Default is "lab_diags/hdf5".
        do_storage_cleanup: If True, will delete all files that aren't the one
            with the highest objective function. Defaults to DO_STORAGE_CLEANUP.

    Returns:
        The `output_params` dictionary with the results from the analysis,
        including:
            - f: Objective function value
            - charge: Total charge in pC
            - energy_mean: Mean energy in GeV
            - energy_spread: Energy spread (RMS) in GeV
            - bunch_length: Bunch length in μm
            - emittance_x: Horizontal emittance in μm⋅rad
            - emittance_y: Vertical emittance in μm⋅rad
    """
    print(f"ANALYSIS: Starting analysis of {simulation_directory}")
    minimum_energy = 0.8  # GeV, increased from 0.6
    maximum_energy = 1.5  # GeV
    power_energy_spread = 1.5

    # Specify a range of R56 values to scan, can assume 0 to 100 um
    r56_values = np.linspace(0, 100e-6, 200)  # 0 to 100 μm in 100 steps

    # Set some default values in case the analysis fails.
    output_params["f"] = 0
    output_params["charge"] = 0
    output_params["energy_mean"] = 0
    output_params["energy_spread"] = 0
    output_params["bunch_length"] = 0
    output_params["emittance_x"] = 0
    output_params["emittance_y"] = 0

    try:
        # Open simulation diagnostics.
        diags_path: str = os.path.join(simulation_directory, file_tree)
        print(f"ANALYSIS: Looking for diagnostics at {diags_path}")

        if not os.path.exists(diags_path):
            print(f"ANALYSIS ERROR: Diagnostics path {diags_path} does not exist")
            return output_params

        diagnostics: LpaDiagnostics = LpaDiagnostics(diags_path)
        available_iterations = diagnostics.iterations

        # Part III: Relative Laser Energy
        laser_energy_nominal = 3  # J
        ini_filepath = os.path.join(diags_path, "../input.ini")

        config = configparser.ConfigParser()
        config.read(ini_filepath)
        laser_energy_j = config.getfloat("Laser", "energy_j")

        # Loop through diags files
        current_objective_maximum = 0
        best_iteration: Optional[int] = None
        for iteration in available_iterations:
            print(f"ANALYSIS: Analyzing iteration {iteration}")

            x, y, z, ux, uy, uz, w, q = diagnostics.get_particle(
                ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
                iteration=iteration,
                select={"uz": [(minimum_energy * 1e3 / 0.511) / 2, None]},
                species=SPECIES_NAME,
                plot=False,
            )

            if len(w) > 0 and np.sum(w * q) * -1e12 > 30.0:
                # Convert uz to energy in GeV for filtering
                gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
                energy_mev = (gamma - 1) * m_e * c**2 / e / 1e6
                energy_mean_gev = (
                    np.average(energy_mev, weights=w) / 1000.0
                )  # Convert MeV to GeV

                # If average beam energy is > minimum and < maximum:
                if minimum_energy < energy_mean_gev < maximum_energy:
                    # Calculate the average uz value for R56 calculation
                    uz_mean = np.average(uz, weights=w)

                    # Calculate the objective function using a virtual R56 scan
                    peak_currents = []
                    for i, r56 in enumerate(r56_values):
                        # For each R56 setting:
                        # Use normalized momentum uz (not energy) for R56 formula
                        z_modified = z + r56 * ((uz - uz_mean) / uz_mean)

                        try:
                            # Calculate current profile with modified z positions
                            current, z_axis = an.calculate_current(
                                z_modified, ux, uy, uz, w, q, bins=1000
                            )
                            peak_current = np.max(current)
                            peak_currents.append(peak_current)

                        except Exception:
                            peak_currents.append(np.nan)

                    peak_currents = np.array(peak_currents)
                    iteration_objective_function = np.nanmax(peak_currents)

                    # Only update if we got a valid objective function
                    if (
                        not np.isnan(iteration_objective_function)
                        and iteration_objective_function > current_objective_maximum
                    ):
                        current_objective_maximum = iteration_objective_function
                        best_iteration = int(iteration)

                        # Calculate beam statistics using analysis functions
                        output_params["charge"] = np.sum(w * q) * -1e12
                        analysis_results = an.analyze_beam(
                            x, y, z, ux, uy, uz, w, q, bins=100
                        )
                        # Convert to GeV
                        output_params["energy_mean"] = (
                            analysis_results["energy_parameters"]["central_energy_mev"]
                            / 1000.0
                        )
                        # Convert to GeV
                        output_params["energy_spread"] = (
                            analysis_results["energy_parameters"]["energy_std_mev"]
                            / 1000.0
                        )
                        # Convert to μm
                        output_params["bunch_length"] = (
                            analysis_results["beam_sizes"]["sigma_z_m"] * 1e6
                        )
                        # Convert to μm⋅rad
                        output_params["emittance_x"] = (
                            analysis_results["emittance"]["x"] * 1e6
                        )
                        # Convert to μm⋅rad
                        output_params["emittance_y"] = (
                            analysis_results["emittance"]["y"] * 1e6
                        )

                        # Modifications
                        f_rel_laser_energy = laser_energy_j / laser_energy_nominal
                        f_energy_spread = (
                            output_params["energy_spread"]
                            / output_params["energy_mean"]
                        ) ** power_energy_spread

                        # Modify the objective function
                        output_params["f"] = (
                            float(iteration_objective_function)
                            / f_energy_spread
                            / f_rel_laser_energy
                        )

        # After evaluating all iterations, delete all .h5 files except the one
        # corresponding to the best iteration to conserve disk space.
        if do_storage_cleanup and DO_STORAGE_CLEANUP:
            storage_cleanup_routine(
                diags_path=diags_path, iteration_to_keep=best_iteration
            )
        elif do_storage_cleanup and not DO_STORAGE_CLEANUP:
            print(
                "ANALYSIS WARNING: unlock file storage cleanup within `optimas_analysis.py` to proceed..."
            )

    except Exception as exception:
        print(f"ANALYSIS ERROR: {exception}")
        traceback.print_exc()

    return output_params


def analyze_chirped_beam_normalized_by_slice(
    simulation_directory: str,
    output_params: dict[str, float],
    file_tree: str = "lab_diags/hdf5",
    do_storage_cleanup: bool = DO_STORAGE_CLEANUP,
) -> dict[str, float]:
    """Analyze chirped beam simulations with slice-normalized weighting.  Evaluates
    the objective function to find beams that have a large peak current when passed
    through a chicane's R56 transformation, as well as having a modest energy spread
    and input laser energy.

    This function analyzes electron beams from chirped beam simulations by:
    1. Reading all diagnostic dumps in the simulation_directory
    2. Filtering particles by energy and charge thresholds
    3. Finding the beam region using current profile analysis
    4. Calculating slice-normalized weights based on charge per slice energy spread
    5. Performing an R56 scan to find peak current
    6. Calculating an objective function based on peak current, R56 value,
       energy spread, and relative laser energy

    The objective function rewards high peak current, optimal R56 values,
    low energy spread, and efficient laser energy usage.

    Args:
        simulation_directory: Base path of the simulation folder where the output
            was generated.
        output_params: Dictionary where the value of the objectives and analyzed
            parameters will be stored. There is one entry per parameter, where the
            key is the name of the parameter given by the user.
        file_tree: Path from the simulation directory to the h5 files.
            Default is "lab_diags/hdf5".
        do_storage_cleanup: If True, will delete all files that aren't the one
            with the highest objective function. Defaults to DO_STORAGE_CLEANUP.

    Returns:
        The `output_params` dictionary with the results from the analysis,
        including:
            - f: Objective function value
            - charge: Total charge in pC
            - energy_mean: Mean energy in GeV
            - energy_spread: Energy spread (RMS) in GeV
            - bunch_length: Bunch length in μm
            - emittance_x: Horizontal emittance in μm⋅rad
            - emittance_y: Vertical emittance in μm⋅rad
    """
    print(f"ANALYSIS: Starting analysis of {simulation_directory}")

    # Bounds for acceptable beam energy and charge
    minimum_mean_energy = 0.7  # GeV
    maximum_mean_energy = 1.5  # GeV
    minimum_beam_charge = 30.0  # pC

    # Knobs for controlling respective contributions to overall objective function
    power_energy_spread = 1.0  # Normalized mean energy of full ebeam
    power_laser_energy = 1.0  # Relative input laser energy for simulation
    power_peak_current = 2.0  # Peak current achieved in a r56 scan
    power_r56_peak = 1.0  # R56 value where peak current is achieved
    power_overall = 2.0

    # Specify region of longitudinal momenta to crop data
    uz_selection = (
        minimum_mean_energy * 1e3 / 0.511 / 2,
        maximum_mean_energy * 1e3 / 0.511 * 2,
    )

    # Specify a range of R56 values to scan, can assume 0 to 100 um
    r56_values = np.linspace(0, 100e-6, 200)  # 0 to 100 μm in 100 steps

    # Set some default values in case the analysis fails.
    output_params["f"] = 0
    output_params["charge"] = 0
    output_params["energy_mean"] = 0
    output_params["energy_spread"] = 0
    output_params["bunch_length"] = 0
    output_params["emittance_x"] = 0
    output_params["emittance_y"] = 0

    try:
        # Open simulation diagnostics.
        diags_path: str = os.path.join(simulation_directory, file_tree)
        print(f"ANALYSIS: Looking for diagnostics at {diags_path}")

        if not os.path.exists(diags_path):
            print(f"ANALYSIS ERROR: Diagnostics path {diags_path} does not exist")
            return output_params

        diagnostics: LpaDiagnostics = LpaDiagnostics(diags_path)
        available_iterations = diagnostics.iterations

        # Read laser energy for simulation relative to nominal baseline
        laser_energy_nominal = 3  # J
        config = configparser.ConfigParser()
        config.read(os.path.join(diags_path, "../input.ini"))
        laser_energy_j = config.getfloat("Laser", "energy_j")
        f_rel_laser_energy = (
            1 / (laser_energy_j / laser_energy_nominal)
        ) ** power_laser_energy

        # Loop through diags files to find the largest objective function
        previous_best_obj = 0
        best_iteration: Optional[int] = None
        for iteration in available_iterations:
            print(f"ANALYSIS: Analyzing iteration {iteration}")

            x, y, z, ux, uy, uz, w, q = diagnostics.get_particle(
                ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
                iteration=iteration,
                select={"uz": uz_selection},
                species=SPECIES_NAME,
                plot=False,
            )

            # If beam contains at least the minimum amount of charge then continue analysis
            if not (len(w) > 0 and np.sum(w * q) * -1e12 > minimum_beam_charge):
                continue

            # Convert uz to energy in GeV for filtering
            gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
            energy_mev = (gamma - 1) * m_e * c**2 / e / 1e6
            energy_mean_gev = (
                np.average(energy_mev, weights=w) / 1000.0
            )  # Convert MeV to GeV

            # If average beam energy is > minimum and < maximum then continue analysis:
            if not minimum_mean_energy < energy_mean_gev < maximum_mean_energy:
                continue

            # Find z region of beam by current and crop to a 1/e^2 region
            current, axis = an.calculate_current(
                z=z, ux=ux, uy=uy, uz=uz, w=w, q=q, bins=200
            )
            threshold = float(np.max(current)) / np.exp(2)
            z_region = None

            above_threshold = current > threshold
            if np.any(above_threshold):
                crossings = np.where(np.diff(above_threshold.astype(int)))[0]

                # Majority of cases will have two primary crossings
                if len(crossings) >= 2:
                    left_edge = float(axis[crossings[0]])
                    right_edge = float(axis[crossings[-1]])
                    z_region = (left_edge, right_edge)

                # Some cases will have a low energy tail that goes up to the right edge
                elif len(crossings) == 1 and above_threshold[-1]:
                    left_edge = float(axis[crossings[0]])
                    right_edge = float(axis[-1])
                    z_region = (left_edge, right_edge)

            if z_region is None:
                continue

            # Find normalized weights through charge in slice divided by slice energy spread
            slice_energy_spread, axis_2, normalized = an.calculate_slice_energy_spread(
                diag_folder=diags_path,
                species_name=SPECIES_NAME,
                iteration=iteration,
                selection={"uz": uz_selection, "z": z_region},
                bins=200,
            )

            # Plot lineouts if in DEBUG mode
            if DEBUG:
                an.plot_slice_statistics(
                    current_axis=axis,
                    current_values=current,
                    region_axis=axis_2,
                    region_slice_energy_spread=slice_energy_spread,
                    region_normalized_weights=normalized,
                )

            # Reload the beam using the z_region to define the z cropping range
            (
                x_cropped,
                y_cropped,
                z_cropped,
                ux_cropped,
                uy_cropped,
                uz_cropped,
                w_cropped,
                q,
            ) = diagnostics.get_particle(
                ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
                iteration=iteration,
                select={"uz": uz_selection, "z": z_region},
                species=SPECIES_NAME,
                plot=False,
            )

            # Map each particle to its z slice and get the corresponding normalized value
            # Use interpolation to map particle z positions to normalized values
            # axis_2 contains the z positions (bin centers) for which we have normalized values
            # For particles outside the range of axis_2, use the nearest value
            normalized_values_per_particle = np.interp(
                z_cropped,
                axis_2,
                normalized,
                left=normalized[0] if len(normalized) > 0 else 0.0,
                right=normalized[-1] if len(normalized) > 0 else 0.0,
            )

            # Create normalized weights: w * normalized_value for each particle
            normalized_weights = w_cropped * normalized_values_per_particle

            # Using the normalized values as weights, calculate the energy spread of the beam
            energy_spread_analysis = calculate_energy_parameters(
                ux_cropped, uy_cropped, uz_cropped, normalized_weights
            )
            energy_spread_normalized = energy_spread_analysis["energy_std_mev"]
            mean_energy_normalized = energy_spread_analysis["central_energy_mev"]
            f_energy_spread = (1 / energy_spread_normalized) ** power_energy_spread

            # Calculate the average uz value for R56 calculation using the normalized weights
            uz_mean = np.average(uz_cropped, weights=normalized_weights)

            # Find the peak possible current using an R56 scan
            peak_currents = np.zeros(len(r56_values))
            for i, r56 in enumerate(r56_values):
                # For each R56 setting:
                # Use normalized momentum uz (not energy) for R56 formula
                z_modified = z_cropped + r56 * ((uz_cropped - uz_mean) / uz_mean)

                try:
                    # Calculate current profile with modified z positions
                    current, z_axis = an.calculate_current(
                        z_modified,
                        ux_cropped,
                        uy_cropped,
                        uz_cropped,
                        w_cropped,
                        q,
                        bins=1000,
                    )
                    peak_currents[i] = np.max(current)

                except Exception:
                    peak_currents[i] = np.nan

            peak_possible_current = np.nanmax(peak_currents)
            if np.isnan(peak_possible_current):
                continue

            # Calculate objective function for this output file
            f_i_peak = float(peak_possible_current) ** power_peak_current
            f_r56_peak = r56_values[np.nanargmax(peak_currents)] ** power_r56_peak
            objective_function = (
                f_i_peak * f_r56_peak * f_energy_spread * f_rel_laser_energy
            ) ** power_overall

            # Only update if we got a valid objective function that is larger than previously
            if (
                not np.isnan(peak_possible_current)
                and objective_function > previous_best_obj
            ):
                previous_best_obj = objective_function
                best_iteration = int(iteration)
                output_params["f"] = objective_function

                # Calculate beam statistics using analysis functions
                output_params["charge"] = np.sum(w_cropped * q) * -1e12
                analysis_results = an.analyze_beam(
                    x_cropped,
                    y_cropped,
                    z_cropped,
                    ux_cropped,
                    uy_cropped,
                    uz_cropped,
                    normalized_weights,
                    q,
                    bins=100,
                )
                output_params["energy_mean"] = (
                    mean_energy_normalized / 1000.0
                )  # Convert to GeV
                output_params["energy_spread"] = (
                    energy_spread_normalized / 1000.0
                )  # Convert to GeV
                output_params["bunch_length"] = (
                    analysis_results["beam_sizes"]["sigma_z_m"] * 1e6
                )  # Convert to μm
                output_params["emittance_x"] = (
                    analysis_results["emittance"]["x"] * 1e6
                )  # Convert to μm⋅rad
                output_params["emittance_y"] = (
                    analysis_results["emittance"]["y"] * 1e6
                )  # Convert to μm⋅rad

        # After evaluating all iterations, delete all .h5 files except the one
        # corresponding to the best iteration to conserve disk space.
        if do_storage_cleanup and DO_STORAGE_CLEANUP:
            storage_cleanup_routine(
                diags_path=diags_path, iteration_to_keep=best_iteration
            )
        elif do_storage_cleanup and not DO_STORAGE_CLEANUP:
            print(
                "ANALYSIS WARNING: unlock file storage cleanup within `optimas_analysis.py` to proceed..."
            )

    except Exception as exception:
        print(f"ANALYSIS ERROR: {exception}")
        traceback.print_exc()

    return output_params
