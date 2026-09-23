"""Analysis functions for FBPIC simulation outputs.

This module provides analysis functions that run after FBPIC simulations to
extract beam parameters, calculate objective functions, and perform optimization
metrics. It supports multiple analysis modes including HTU, HOFI, and chirped
beam analysis.

The module integrates with Optimas optimization framework and can optionally
clean up diagnostic files to conserve storage space.

Each public analysis function is decorated with ``@analyzed_params(...)`` so
that its list of output parameter names (excluding the objective ``"f"``) can
be queried programmatically via ``func.analyzed_param_names`` or via the
helper ``get_default_output_params(func)``.
"""

import os
import shutil
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from openpmd_viewer.addons import LpaDiagnostics
from scipy.optimize import minimize
from scipy.constants import c, e, m_e

from optimas.diagnostics import ExplorationDiagnostics

# Import analysis modules either through package or relative
try:
    import inversion_fbpic.utils.analysis as an
    from inversion_fbpic.utils.field_analysis import analyze_laser_evolution
    from inversion_fbpic.lib.laser import _LaserPulse
except ImportError:
    import analysis as an  # type: ignore
    from field_analysis import analyze_laser_evolution  # type: ignore
    from laser import _LaserPulse  # type: ignore

# Manually set to True if you'd like Optimas to clean up storage space
DO_STORAGE_CLEANUP: bool = False
# If True, only evaluate the final dump.  If False, evaluate every dump.
LAST_DUMP_ONLY: bool = False

# Commonly-changed variable to determine the target energy for a hofi optimization
HOFI_P_GOAL = 0.430  # GeV

# Specify the laser wavelength for laser evolution calculations
# TODO include this within input parameter file to automate
LASER_WAVELENGTH = 0.800e-6  # m

# Charge-density movie defaults (match slideshow_from_npy.py)
GENERATE_CHARGE_DENSITY_MOVIE: bool = False
CHARGE_DENSITY_MOVIE_VMAX: float = 1e6 * (1e-6 / e)
CHARGE_DENSITY_MOVIE_VMIN: float = -0.05e6 * (1e-6 / e)
CHARGE_DENSITY_MOVIE_RMAX: float = 50e-6
CHARGE_DENSITY_MOVIE_FRAMERATE: int = 10
DEBUG: bool = False  # If set to True, additional plots will be generated


def analyzed_params(*param_names: str) -> Callable:
    """Decorator that registers analyzed-parameter names on an analysis function.

    The decorated function gains an ``analyzed_param_names`` attribute that
    lists every output key **except** the objective ``"f"``.
    """

    def decorator(func: Callable) -> Callable:
        func.analyzed_param_names = list(param_names)
        return func

    return decorator


def get_default_output_params(analysis_func: Callable) -> dict[str, float]:
    """Return a zeroed output-params dict for *analysis_func*.

    Useful for building the ``analyzed_parameters`` list that Optimas
    generators require, or for pre-populating default values before
    calling the analysis function.

    The returned dict always includes ``"f": 0.0`` plus one entry for
    every name in ``analysis_func.analyzed_param_names``.

    Raises:
        AttributeError: If *analysis_func* was not decorated with
            ``@analyzed_params``.
    """
    params: dict[str, float] = {"f": 0.0}
    for name in analysis_func.analyzed_param_names:
        params[name] = 0.0
    return params


def calculate_energy_parameters(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, w: np.ndarray
) -> dict[str, float]:
    """Calculates Lorentzian gamma, central energy, and energy spread in both RMS and FWHM.

    Args:
        ux: Normalized x-momenta (dimensionless).
        uy: Normalized y-momenta (dimensionless).
        uz: Normalized z-momenta (dimensionless).
        w: Particle weights.

    Returns:
        Dictionary containing:
            - gamma_beam: Average Lorentzian gamma
            - central_energy_mev: Central energy in MeV
            - energy_std_mev: Energy RMS in MeV
            - energy_fwhm_mev: Energy FWHM in MeV
    """
    gamma: np.ndarray = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    energy_mev: np.ndarray = (gamma - 1) * m_e * c**2 / e / 1e6
    central_energy: float = float(np.average(energy_mev, weights=w))
    energy_std: float = np.sqrt(
        np.average((energy_mev - central_energy) ** 2, weights=w)
    )
    hist, bin_edges = np.histogram(energy_mev, bins=500, weights=w)
    bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_idx: int = int(np.argmax(hist))
    peak_value: float = float(hist[peak_idx])
    half_max: float = peak_value / 2
    left_idx: int = peak_idx
    right_idx: int = peak_idx
    while left_idx > 0 and hist[left_idx] > half_max:
        left_idx -= 1
    while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
        right_idx += 1
    fwhm: float = float(bin_centers[right_idx] - bin_centers[left_idx])
    return {
        "gamma_beam": float(np.average(gamma, weights=w)),
        "central_energy_mev": central_energy,
        "energy_std_mev": energy_std,
        "energy_fwhm_mev": fwhm,
    }


@analyzed_params("charge", "charge_weighted", "energy_mean", "energy_med", "energy_mad")
def analyze_hofi_simulation(
    simulation_directory: str,
    output_params: dict[str, float],
    file_tree: str = "lab_diags/hdf5",
    laser_energy_setpoint: Optional[float] = None,
    do_storage_cleanup: bool = DO_STORAGE_CLEANUP,
    species: str | list[str] | None = None,
    min_charge: float = 30.0,
) -> dict[str, float]:
    """Analyze the simulation output for HOFI simulations for lithography, and
    evaluates the objective function based on beams with the most charge, at the
    correct energy, and with the lowest energy spread.

    This function analyzes electron beams produced from HOFI simulations by:
    1. Reading all diagnostic dumps in the simulation_directory
    2. Cropping out non-Nitrogen electrons and non-relativistic particles
    3. Finding the total charge, energy distribution, and input laser energy
    4. Calculating the objective function using these parameters

    The objective function combines spectral density, momentum deviation, and
    relative laser energy to find the best iteration across all available dumps.

    Args:
        simulation_directory: Base path of the simulation folder where the output
            was generated.
        output_params: Dictionary where the value of the objectives and analyzed
            parameters will be stored. There is one entry per parameter, where the
            key is the name of the parameter given by the user.
        file_tree: Path from the simulation directory to the h5 files.
            Default is "lab_diags/hdf5".
        laser_energy_setpoint: Laser energy to use in objective function analysis.
            If None, will read the value from a LaserProfile config file in
            `simulation_directory`/cfgs if it exists.
            Defaults to None.
        do_storage_cleanup: If True, will delete all files that aren't the one
            with the highest objective function. Defaults to DO_STORAGE_CLEANUP.
        species: Name or list of names of the species to analyze. Defaults to None
            (concatenate all negatively charged particles).
        min_charge: Minimum charge in pC to consider an iteration for analysis. Defaults to 30.0 pC.

    Returns:
        The `output_params` dictionary with the results from the analysis,
        including:
            - f: Objective function value
            - charge: Total charge in pC
            - charge_weighted: Weighted charge in pC
            - energy_mean: Mean energy in GeV
            - energy_med: Median energy in GeV
            - energy_mad: Median absolute deviation of energy in GeV
    """

    if isinstance(species, str):
        species = [species]
    if isinstance(species, list) and len(species) == 0:
        raise ValueError("species must be a str, None, or non-empty list of str.")

    p_goal = HOFI_P_GOAL

    select_base = {"uz": [(p_goal * 1e3 / 0.511) / 2, None]}
    if species is None:
        select_base["charge"] = [None, np.nextafter(0.0, -np.inf)]

    print(f"ANALYSIS: Starting analysis of {simulation_directory}")

    # Set some default values in case the analysis fails.
    output_params["f"] = 0
    output_params["charge"] = 0
    output_params["charge_weighted"] = 0
    output_params["energy_mean"] = 0
    output_params["energy_med"] = 0
    output_params["energy_mad"] = 0

    try:
        # Open simulation diagnostics.
        diags_path: str = os.path.join(simulation_directory, file_tree)
        print(f"ANALYSIS: Looking for diagnostics at {diags_path}")

        if not os.path.exists(diags_path):
            print(f"ANALYSIS ERROR: Diagnostics path {diags_path} does not exist")
            return output_params

        diagnostics: LpaDiagnostics = LpaDiagnostics(diags_path)
        available_iterations = diagnostics.iterations
        last_iteration = max(available_iterations)

        current_objective_maximum = 0
        best_iteration: Optional[int] = None
        for iteration in available_iterations:
            if LAST_DUMP_ONLY and iteration != last_iteration:
                print(f"ANALYSIS: Skipping {iteration}")
                continue

            print(f"ANALYSIS: Analyzing iteration {iteration}")

            ux, uy, uz, w, q = (None,) * 5
            for spec in species if species is not None else diagnostics.avail_species:
                ux0, uy0, uz0, w0, q0 = diagnostics.get_particle(
                    ["ux", "uy", "uz", "w", "charge"],
                    iteration=iteration,
                    select=select_base,
                    species=spec,
                    plot=False,
                )
                if len(q0) == 1:
                    q0 = q0[0]
                else:
                    raise ValueError(
                        f"Species {spec} has inconsistent charge: {q0}. Expected a single value."
                    )
                if ux is None:
                    ux, uy, uz, w, q = ux0, uy0, uz0, w0, q0
                else:
                    ux = np.concatenate((ux, ux0))
                    uy = np.concatenate((uy, uy0))
                    uz = np.concatenate((uz, uz0))
                    w = np.concatenate((w, w0))
                    if not np.isclose(q, q0):
                        raise ValueError(
                            f"Species {spec} has inconsistent charge: {q0}. Expected {q}"
                        )

            if len(w) > 0:
                # Need to see at least 10 pC to make reasonable statistics
                iteration_charge = np.sum(w * q) * -1e12
                if iteration_charge > min_charge:

                    uz_gev = uz * 0.511 * 1e-3  # Convert to GeV

                    # Part I: Charge within momentum window
                    sig_p_median, sig_p_mad = weighted_mad(uz_gev, w)  # MeV

                    delta_p = 3.478 * sig_p_mad
                    window = (uz_gev > sig_p_median - delta_p) & (
                        uz_gev < sig_p_median + delta_p
                    )

                    weighted_charge = np.sum(w[window] * q) * -1e12
                    f_spectral_density = weighted_charge / delta_p

                    # Part II: Momentum Deviation
                    p_average = np.average(uz_gev, weights=w)

                    print("p_average = ", p_average)

                    term_1 = 1.48 * sig_p_mad / p_average
                    term_2 = 1 - p_average / p_goal

                    f_momentum_deviation = np.sqrt(term_1**2 + term_2**2)

                    # Part III: Relative Laser Energy
                    laser_energy_nominal = 5  # J

                    if laser_energy_setpoint is None:
                        las = None
                        for cfg in (Path(simulation_directory) / "cfgs").glob("*"):
                            try:
                                las = _LaserPulse.from_any(cfg)
                                break
                            except Exception:
                                las = None
                                continue

                        if las is None:
                            print(
                                "ANALYSIS WARNING: Could not find laser energy in input file. Using nominal value."
                            )
                            laser_energy_j = laser_energy_nominal
                        else:
                            if hasattr(las, "energy"):
                                laser_energy_j = las.energy
                            elif hasattr(las, "out_energy"):
                                laser_energy_j = las.out_energy
                            else:
                                print(
                                    "ANALYSIS WARNING: Could not find laser energy in input file. Using nominal value."
                                )
                                laser_energy_j = laser_energy_nominal

                        f_rel_laser_energy = laser_energy_j / laser_energy_nominal

                    else:
                        f_rel_laser_energy = (
                            laser_energy_setpoint / laser_energy_nominal
                        )

                    # Part IV: Emittance
                    energy_spread_rms = np.sqrt(
                        np.average((uz_gev - p_average) ** 2, weights=w)
                    )
                    uz_central = p_average * 1e3 / 0.511
                    uz_range = energy_spread_rms * 1e3 / 0.511

                    select_emittance = {
                        "uz": [(uz_central - uz_range), (uz_central + uz_range)]
                    }
                    if species is None:
                        select_emittance["charge"] = [None, np.nextafter(0.0, -np.inf)]

                    dx, dy, dz, dpx, dpy, dpz, dw, dq = (None,) * 8
                    for spec in (
                        species if species is not None else diagnostics.avail_species
                    ):
                        dx0, dy0, dz0, dpx0, dpy0, dpz0, dw0, dq0 = (
                            diagnostics.get_particle(
                                ["x", "y", "z", "ux", "uy", "uz", "w", "charge"],
                                iteration=iteration,
                                select=select_emittance,
                                species=spec,
                                plot=False,
                            )
                        )
                        if dx is None:
                            dx, dy, dz, dpx, dpy, dpz, dw, dq = (
                                dx0,
                                dy0,
                                dz0,
                                dpx0,
                                dpy0,
                                dpz0,
                                dw0,
                                dq0,
                            )
                        else:
                            dx = np.concatenate((dx, dx0))
                            dy = np.concatenate((dy, dy0))
                            dz = np.concatenate((dz, dz0))
                            dpx = np.concatenate((dpx, dpx0))
                            dpy = np.concatenate((dpy, dpy0))
                            dpz = np.concatenate((dpz, dpz0))
                            dw = np.concatenate((dw, dw0))
                            if not np.isclose(dq, dq0) and len(dq0) > 0:
                                raise ValueError(
                                    f"Species {spec} has inconsistent charge: {dq0}. Expected {dq}"
                                )

                    # Need at least 2 particles to continue
                    if len(dw) < 2:
                        continue

                    energy_params = calculate_energy_parameters(
                        ux=dpx, uy=dpy, uz=dpz, w=dw
                    )
                    emittance = an.calculate_geometric_emittance(
                        x=dx, y=dy, ux=dpx, uy=dpy, uz=dpz, w=dw
                    )
                    f_x_emit = emittance["x"] * energy_params["gamma_beam"]
                    if not np.isfinite(f_x_emit) or f_x_emit <= 0:
                        continue

                    # Finally, calculate the objective function
                    iteration_objective_function = (
                        f_spectral_density
                        / f_momentum_deviation
                        / f_rel_laser_energy
                        / f_x_emit
                    )

                    # If it is the new record, then save all output params accordingly.
                    if iteration_objective_function > current_objective_maximum:
                        current_objective_maximum = iteration_objective_function
                        best_iteration = int(iteration)
                        output_params["f"] = iteration_objective_function
                        output_params["charge"] = np.sum(w * q) * -1e12
                        output_params["charge_weighted"] = weighted_charge
                        output_params["energy_mean"] = p_average
                        output_params["energy_med"] = sig_p_median
                        output_params["energy_mad"] = sig_p_mad

        # Laser evolution and charge-density movie before deleting files
        laser_analysis(diags_path=diags_path)
        charge_density_movie_analysis(diags_path=diags_path)

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


# Helper functions for median and weighted median. Taken from
# https://optimas.readthedocs.io/en/latest/examples/bo_multitask_fbpic_waket.html


def weighted_mad(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Calculate weighted median absolute deviation.

    Args:
        x: Input data array (one dimension).
        w: Array with the weights of the same size as `x`.

    Returns:
        A tuple containing:
            - med: Weighted median of the data
            - mad: Weighted median absolute deviation
    """
    med = weighted_median(x, w)
    mad = weighted_median(np.abs(x - med), w)
    return med, mad


def weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of a 1D numpy array.

    Args:
        data: Input array (one dimension).
        weights: Array with the weights of the same size as `data`.

    Returns:
        The weighted median value.

    Raises:
        TypeError: If data or weights are not 1D arrays, or if their shapes
            don't match.
        ValueError: If quantile is not between 0 and 1 (internal check).
    """
    quantile = 0.5
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quantile > 1.0) or (quantile < 0.0):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    # assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def laser_analysis(diags_path: str) -> None:
    """
    Calls the laser evolution analysis routine of field_analysis.py

    Args:
        diags_path: Path to the .h5 data
    """
    try:
        results_file = Path(diags_path).parent / "laser_evolution_results.json"
        analyze_laser_evolution(
            series_path=diags_path,
            laser_wavelength=LASER_WAVELENGTH,
            results_file=results_file,
        )
    except Exception as exception:
        print(f"LASER ANALYSIS ERROR: {exception}")
        traceback.print_exc()


def charge_density_movie_analysis(diags_path: str) -> None:
    """Render charge-density stills from HDF5 and write an MP4.

    Uses the same display defaults as ``slideshow_from_npy.py`` (linear
    ``bwr`` color scale, vmin/vmax in cm$^{-3}$, ``rmax`` crop) but reads
    OpenPMD diagnostics directly via :func:`plot_from_hdf5_series` instead
    of intermediate ``.npy`` files.

    The movie is written next to the diagnostics folder (e.g.
    ``lab_diags/rho.mp4``). Intermediate PNGs are removed after a successful
    encode. Skipped silently when :data:`GENERATE_CHARGE_DENSITY_MOVIE` is
    False or when ``ffmpeg`` is not available.

    Args:
        diags_path: Path to the HDF5 diagnostic directory.
    """
    if not GENERATE_CHARGE_DENSITY_MOVIE:
        return

    try:
        try:
            from inversion_fbpic.utils.make_movie import make_movie
            from inversion_fbpic.utils.plotting import plot_from_hdf5_series
        except ImportError:
            from make_movie import make_movie  # type: ignore
            from plotting import plot_from_hdf5_series  # type: ignore
    # ImportError occurs if scripts aren't available locally
    # RuntimeError occurs if ffmpeg is not installed in environment
    except (ImportError, RuntimeError) as exception:
        print(f"CHARGE DENSITY MOVIE: skipped ({exception})")
        return

    diags_parent = Path(diags_path).parent
    stills_dir = diags_parent / "charge_density_stills"

    try:
        print(f"CHARGE DENSITY MOVIE: rendering stills from {diags_path}")
        stills_dir, prefix = plot_from_hdf5_series(
            series_path=diags_path,
            save_path=stills_dir,
            field_name="rho",
            component=None,
            vminmax=(CHARGE_DENSITY_MOVIE_VMIN, CHARGE_DENSITY_MOVIE_VMAX),
            scale="linear",
            cmap="bwr",
            rmax=CHARGE_DENSITY_MOVIE_RMAX,
        )
        n_stills = len(list(stills_dir.glob(f"{prefix}*.png")))
        print(
            f"CHARGE DENSITY MOVIE: wrote {n_stills} PNGs in {stills_dir}; "
            "starting ffmpeg..."
        )
        movie_file = make_movie(
            images_dir=stills_dir,
            image_prefix=prefix,
            filename=prefix,
            framerate=CHARGE_DENSITY_MOVIE_FRAMERATE,
            optimas=True,
        )
        if movie_file is None:
            return

        target_movie = diags_parent / movie_file.name
        if movie_file.resolve() != target_movie.resolve():
            shutil.move(movie_file, target_movie)
            movie_file = target_movie

        shutil.rmtree(stills_dir)
        print(f"CHARGE DENSITY MOVIE: wrote {movie_file}")
    except Exception as exception:
        print(f"CHARGE DENSITY MOVIE ERROR: {exception}")
        traceback.print_exc()


def storage_cleanup_routine(
    diags_path: str,
    iteration_to_keep: Optional[int],
) -> None:
    """Clean up diagnostic files, keeping only the specified iteration.

    Loops through a diagnostics directory and deletes all HDF5 files except
    the one corresponding to the specified iteration number. Useful for when
    you want to have simulations dump many data files in order to get statistics,
    but don't want to keep all the data after analysis is complete.

    If no best iteration was determined, all HDF5 dumps in that directory are
    removed so failed or inconclusive runs do not retain large outputs.

    Requires the `DO_STORAGE_CLEANUP` flag to be set to True at the top of this
    script. Typically, keep this flag False when running locally in order to
    preserve saved data, and then manually set it to True when running on AWS
    if needed.

    Args:
        diags_path: Filepath to the diagnostic file directory.
        iteration_to_keep: Iteration number to preserve during cleanup process.
            If None, all HDF5 files in the directory are deleted.
    """
    if DO_STORAGE_CLEANUP:
        try:
            hdf5_dir = Path(diags_path)
            target_name: Optional[str] = None
            if iteration_to_keep is not None:
                target_name = f"data{iteration_to_keep:08d}.h5"

            for h5_file in sorted(hdf5_dir.glob("*.h5")):
                if target_name is not None and h5_file.name == target_name:
                    print(
                        f"ANALYSIS: Kept {target_name} for best iteration {iteration_to_keep}"
                    )
                    continue
                try:
                    h5_file.unlink()
                    print(f"ANALYSIS: Deleted {h5_file.name}")
                except Exception as err:
                    print(f"ANALYSIS WARNING: Could not delete {h5_file.name}: {err}")
        except Exception as cleanup_exception:
            print(f"ANALYSIS WARNING: Cleanup step failed: {cleanup_exception}")


def find_optimum_predicted_result(
    exploration: str | ExplorationDiagnostics,
    objective: str,
    minimize_objective: bool,
    verbose: bool = False,
) -> tuple[tuple[dict[str, float], float], tuple[dict[str, float], float]]:
    """Find the optimal parameters predicted by the GP model built from an existing exploration.
    This searches the parameter space for the best objective function value regardless of which parameter values were evaluated during the eploration.

    Args:
        exploration: Path to the exploration directory. E.g., "./exploration". Or an ExplorationDiagnostics object.
        objective: Name of the objective function to use.
        minimize_objective: Whether to minimize the objective function.
        verbose: Whether to print verbose output, including the best and optimal parameters and objective function values. Defaults to False.
    Returns:
        A tuple containing:
            - best_evaluated: A tuple of the best evaluated parameters and the corresponding objective function value.
            - best_predicted: A tuple of the optimal predicted parameters, the corresponding objective function value, and the standard deviation of the predicted objective function value.
    """

    # get best results from exploration
    if isinstance(exploration, str):
        diags = ExplorationDiagnostics(exploration)
    else:
        diags = exploration
    f_model = diags.build_gp_model(objective, minimize=minimize_objective)

    best_params = f_model.get_best_evaluation(objective, use_model_predictions=False)[1]

    param_names = best_params.keys()
    best_param_values = list(best_params.values())

    if verbose:
        print(f"Best evaluated parameters: {best_params}")
        print(
            f"Best evaluated objective function: {f_model.evaluate_model(best_params)[0]}"
        )

    # now find the optimal parameters predicted by the GP model
    def build_params_dict(params: np.ndarray) -> dict[str, float]:
        my_params = {}
        for i, param_name in enumerate(param_names):
            my_params[param_name] = params[i]
        return my_params

    def objective_function(params: np.ndarray) -> float:
        my_params = build_params_dict(params)
        return f_model.evaluate_model(my_params)[0] * (
            1.0 if minimize_objective else -1.0
        )

    bounds = [
        (vp.lower_bound, vp.upper_bound)
        for vp in diags.varying_parameters
        if not vp.is_fixed
    ]
    result = minimize(
        objective_function, best_param_values, method="L-BFGS-B", bounds=bounds
    )

    if verbose:
        print(f"Optimal predicted parameters: {build_params_dict(result.x)}")
        print(
            f"Optimal predicted objective function: {result.fun * (1.0 if minimize_objective else -1.0)}"
        )

    return (best_params, f_model.evaluate_model(best_params)[0][0]), (
        build_params_dict(result.x),
        result.fun * (1.0 if minimize_objective else -1.0),
        f_model.evaluate_model(build_params_dict(result.x))[1][0],
    )
