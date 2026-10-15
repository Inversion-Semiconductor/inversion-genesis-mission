"""Class-based boosted-frame nitrogen ionization-injection FBPIC simulation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import itertools
import logging
from math import exp, isfinite, log
from pathlib import Path
from typing import Any, Mapping, Optional, Literal

import attrs

from inversion_fbpic.utils.laser import HighOrderLasyLaser
from inversion_fbpic.lib import (
    density_profiles as dn,
    laser as ls,
    simulation as sm,
    datapoint as dp,
    config_container as cc,
)


_MOVIE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fbpic-movies")
_LOGGER = logging.getLogger(__name__)

_SERIALIZATION_FORMAT: Literal["json", "yaml", "h5"] = "json"
_PARAMETERS_FORMAT: Literal["json", "yaml", "h5"] = "json"


class IonizationInjectionSimulation:
    """Configure and run a LASY-driven nitrogen ionization-injection case.

    ``physical_parameters`` contains flat, required laser and gas properties.
    ``hyperparameters`` holds FBPIC grid, particle-loading, diagnostic, and
    execution settings. Laser-prefixed physical parameters are passed directly
    to :class:`HighOrderLasyLaser`. Helium and nitrogen share the bilateral
    density profile specified by independent left/right FWHMs and fractional
    locations of each side's 1% point. Both super-Gaussian beta values are
    derived from the corresponding ramp geometry.
    """

    movie_future: Future[None] | None = None

    _PHYSICAL_PARAMETER_KEYS = frozenset(
        {
            "peak_neutral_density_cm3",
            "nitrogen_dopant_fraction",
            "density_fwhm_left_m",
            "density_fwhm_right_m",
            "density_left_ramp_edge_percent",
            "density_right_ramp_edge_percent",
            "density_center_location_m",
            *HighOrderLasyLaser.PHYSICAL_PARAMETER_KEYS,
        }
    )
    _OPTIONAL_PHYSICAL_PARAMETER_KEYS = (
        HighOrderLasyLaser.OPTIONAL_PHYSICAL_PARAMETER_KEYS
    )
    _HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
        "n_order": 32,
        "gamma_boost": 3.0,
        "nz": 2500,
        "nr": 300,
        "zmax": 0.0,
        "zmin": -100.0e-6,
        "rmax": 140.0e-6,
        "n_azimuthal_modes": 5,
        "p_zmin": 0.0,
        "p_rmax": 90e-6,
        "p_nz": 4,
        "p_nr": 4,
        "p_nt": 30,
        "helium_p_nz": None,
        "helium_p_nt": None,
        "r_boundary": "reflective",
        "p_zmax": None,
        "diagnostic_select": {"uz": [10.0, None]},  # which particles are recorded
        "dump_fields": False,
        "plot_movies": False,
        "number_dumps_lab": 100,
        "write_period": 50,
        "save_checkpoints": False,
        "checkpoint_period": 100,
        "use_restart": False,
        "track_electrons": False,
        "random_seed": 0,
        "lasy_file": Path(__file__).with_name("experimental_laser"),
        "lasy_t_start": 0.0,
        "lasy_antenna_position": 0.0,
        "laser_peak_delay_from_file_start_s": 200e-15,
        "laser_maximum_pulse_duration_fwhm_s": 200e-15,
        "lab_diagnostic_directory": "lab_diags",
        "results_directory": "results",
        "laser_polarization": (1, 0),
        "laser_n_azimuthal_modes": 5,
        "laser_num_points": (600, 900),
        "laser_hi_range": 8.0,
        "laser_center_and_remove_tilt": True,
        "laser_centering_angles": 72,
        "left_ramp_percent_fwhm_range_m": None,
        "left_ramp_percent_min_low_fwhm": None,
        "right_ramp_percent_fwhm_range_m": None,
        "right_ramp_percent_min_low_fwhm": None,
        "ramp_percent_min_high_fwhm": 0.55,
    }

    def __init__(
        self,
        physical_parameters: Mapping[str, Any],
        hyperparameters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.physical_parameters = dict(physical_parameters)
        self.hyperparameters = {
            **self._HYPERPARAMETER_DEFAULTS,
            **(hyperparameters or {}),
        }
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        missing_physical = self._PHYSICAL_PARAMETER_KEYS - set(self.physical_parameters)
        accepted_physical = (
            self._PHYSICAL_PARAMETER_KEYS | self._OPTIONAL_PHYSICAL_PARAMETER_KEYS
        )
        unknown_physical = set(self.physical_parameters) - accepted_physical
        unknown_hyperparameters = set(self.hyperparameters) - set(
            self._HYPERPARAMETER_DEFAULTS
        )
        if missing_physical:
            raise ValueError(f"Missing physical parameters: {sorted(missing_physical)}")
        if unknown_physical:
            raise ValueError(f"Unknown physical parameters: {sorted(unknown_physical)}")
        if unknown_hyperparameters:
            raise ValueError(
                f"Unknown hyperparameters: {sorted(unknown_hyperparameters)}"
            )
        if not 0.0 <= self.physical_parameters["nitrogen_dopant_fraction"] <= 1.0:
            raise ValueError("nitrogen_dopant_fraction must be in the interval [0, 1].")
        if (
            self.hyperparameters["plot_movies"]
            and not self.hyperparameters["dump_fields"]
        ):
            raise ValueError(
                "dump_fields must be True if plot_movies is True, since movies "
                "require field data."
            )

    @staticmethod
    def _plot_movies(simulation: sm.Simulation, save_path: Path) -> None:
        """Render density and laser profile plots as well as movies of the fields."""
        import matplotlib.pyplot as plt

        try:
            save_path.mkdir(parents=True, exist_ok=True)
            try:
                fig = simulation.plot_densities()
                try:
                    fig.savefig(save_path / "densities.png")
                finally:
                    plt.close(fig)
            except Exception:
                _LOGGER.exception("Density profile generation failed for %s", save_path)

            for i, las in enumerate(simulation.lasers):
                try:
                    fig = las.plot()
                    try:
                        fig.savefig(save_path / f"laser_{i}.png")
                    finally:
                        plt.close(fig)
                except Exception:
                    _LOGGER.exception(
                        "Laser profile %d generation failed for %s", i, save_path
                    )

            # plot movies of fields
            simulation.plot_movie(
                field_name="rho",
                vminmax=(1e15, 1e18),
                cmap="magma",
                scale="log",
                save_path=save_path,
            )
            for field, component in itertools.product(["E", "B"], ["x", "y", "z"]):
                simulation.plot_movie(
                    field_name=field,
                    component=component,
                    vminmax="even",
                    cmap="RdBu_r",
                    save_path=save_path,
                )
            simulation.plot_movie(field_name=None, component="eme", save_path=save_path)
        except Exception:
            _LOGGER.exception("Movie generation failed for %s", save_path)
            raise

    def _queue_movies(
        self, simulation: sm.Simulation, save_path: Path
    ) -> Future[None] | None:
        if not simulation.do_write_to_disk:
            return None

        import matplotlib

        matplotlib.use("Agg")
        movie_simulation = sm.Simulation.from_dict(simulation.to_dict())
        movie_simulation.working_directory = simulation.working_directory
        assert movie_simulation.hyparams is not None
        movie_simulation.hyparams = attrs.evolve(
            movie_simulation.hyparams, use_mpi=False
        )
        if not save_path.is_absolute():
            save_path = simulation.working_directory / save_path
        return _MOVIE_EXECUTOR.submit(self._plot_movies, movie_simulation, save_path)

    def _resolved_density_ramp_parameters(
        self, side: Literal["left", "right"]
    ) -> dict[str, float]:
        """Return one side's FWHM and beta from its remapped 1% point geometry."""
        fwhm_m = float(self.physical_parameters[f"density_fwhm_{side}_m"])
        center_m = float(self.physical_parameters["density_center_location_m"])
        requested_ramp_edge_percent = float(
            self.physical_parameters[f"density_{side}_ramp_edge_percent"]
        )
        if not 0.0 < requested_ramp_edge_percent <= 1.0:
            raise ValueError(f"density_{side}_ramp_edge_percent must be in (0, 1]")
        fwhm_range_m = self.hyperparameters[f"{side}_ramp_percent_fwhm_range_m"]
        percent_min_low_fwhm = self.hyperparameters[f"{side}_ramp_percent_min_low_fwhm"]
        percent_min_high_fwhm = self.hyperparameters["ramp_percent_min_high_fwhm"]
        if fwhm_range_m is None or percent_min_low_fwhm is None:
            ramp_edge_percent = requested_ramp_edge_percent
        else:
            fwhm_min_m, fwhm_max_m = fwhm_range_m
            if not 0.0 < fwhm_min_m < fwhm_max_m:
                raise ValueError(f"{side}_ramp_percent_fwhm_range_m must be increasing")
            if not 0.0 < percent_min_low_fwhm <= percent_min_high_fwhm < 1.0:
                raise ValueError(
                    "ramp percent minimum anchors must be in (0, 1) and increasing"
                )
            fwhm_progress = min(
                1.0, max(0.0, (fwhm_m - fwhm_min_m) / (fwhm_max_m - fwhm_min_m))
            )
            effective_percent_min = percent_min_low_fwhm + fwhm_progress * (
                percent_min_high_fwhm - percent_min_low_fwhm
            )
            ramp_edge_percent = effective_percent_min + (
                requested_ramp_edge_percent - percent_min_low_fwhm
            ) * (1.0 - effective_percent_min) / (1.0 - percent_min_low_fwhm)
        half_fwhm_m = fwhm_m / 2.0
        available_ramp_m = center_m - half_fwhm_m
        if available_ramp_m <= 0.0:
            raise ValueError(
                f"density_fwhm_{side}_m must be less than twice "
                "density_center_location_m to derive a positive beta"
            )
        one_percent_distance_m = half_fwhm_m + ramp_edge_percent * available_ramp_m
        width_ratio = 2.0 * one_percent_distance_m / fwhm_m
        beta = log(log(100.0) / log(2.0)) / log(width_ratio)
        if not isfinite(beta) or beta <= 0.0:
            raise ValueError(f"derived density_beta_{side} must be finite and positive")
        one_percent_location_m = (
            center_m - one_percent_distance_m
            if side == "left"
            else center_m + one_percent_distance_m
        )
        resolved = {
            f"density_fwhm_{side}_m": fwhm_m,
            f"density_beta_{side}": beta,
            "density_center_location_m": center_m,
            f"density_{side}_ramp_edge_percent_requested": requested_ramp_edge_percent,
            f"density_{side}_ramp_edge_percent": ramp_edge_percent,
            f"density_{side}_one_percent_location_m": one_percent_location_m,
            f"density_{side}_edge_fraction": exp(
                -log(2.0) * (2.0 * center_m / fwhm_m) ** beta
            ),
        }
        _LOGGER.info(
            "%s density ramp: FWHM = %.6g m, beta = %.6g, "
            "requested/effective ramp edge = %.6g/%.6g, "
            "1%% at z = %.6g m, n(0)/n_peak = %.6g",
            side.capitalize(),
            fwhm_m,
            beta,
            requested_ramp_edge_percent,
            ramp_edge_percent,
            resolved[f"density_{side}_one_percent_location_m"],
            resolved[f"density_{side}_edge_fraction"],
        )
        return resolved

    def build_simulation(self) -> sm.Simulation:
        """Build the configured simulation without setting it up or running it."""
        hyperparameters = self.hyperparameters
        physical_parameters = self.physical_parameters
        left_density_parameters = self._resolved_density_ramp_parameters("left")
        right_density_parameters = self._resolved_density_ramp_parameters("right")
        helium_p_nz = hyperparameters["helium_p_nz"]
        if helium_p_nz is None:
            helium_p_nz = hyperparameters["p_nz"]
        helium_p_nt = hyperparameters["helium_p_nt"]
        if helium_p_nt is None:
            helium_p_nt = hyperparameters["p_nt"]

        density_he = dn.BilateralSuperGaussian(
            nominal_density=physical_parameters["peak_neutral_density_cm3"]
            * 1e6
            * (1.0 - physical_parameters["nitrogen_dopant_fraction"])
            * 2.0,
            fwhm_left=left_density_parameters["density_fwhm_left_m"],
            fwhm_right=right_density_parameters["density_fwhm_right_m"],
            beta_left=left_density_parameters["density_beta_left"],
            beta_right=right_density_parameters["density_beta_right"],
            center=physical_parameters["density_center_location_m"],
            p_nz=helium_p_nz,
            p_nr=hyperparameters["p_nr"],
            p_nt=helium_p_nt,
            p_zmin=hyperparameters["p_zmin"],
            p_zmax=hyperparameters["p_zmax"],
            p_rmax=hyperparameters["p_rmax"],
            species="He",
            ionization=0,
            elec_name="electrons_he",
            elec_select=hyperparameters["diagnostic_select"],
        )

        density_n2 = dn.BilateralSuperGaussian(
            nominal_density=physical_parameters["peak_neutral_density_cm3"]
            * 1e6
            * physical_parameters["nitrogen_dopant_fraction"]
            * 14.0,
            fwhm_left=left_density_parameters["density_fwhm_left_m"],
            fwhm_right=right_density_parameters["density_fwhm_right_m"],
            beta_left=left_density_parameters["density_beta_left"],
            beta_right=right_density_parameters["density_beta_right"],
            center=physical_parameters["density_center_location_m"],
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
            p_zmax=hyperparameters["p_zmax"],
            p_rmax=hyperparameters["p_rmax"],
            species="N",
            ionization=0,
            elec_name="electrons_n2",
            elec_select=hyperparameters["diagnostic_select"],
        )

        laser = ls.LasyLaserPulse(
            energy=physical_parameters["laser_energy_J"],
            z0=0.0,  # only used for plotting
            z0_antenna=hyperparameters["lasy_antenna_position"],
            t_start=hyperparameters["lasy_t_start"],
            lasy_file=hyperparameters["lasy_file"],
            wavelength=physical_parameters["laser_wavelength_m"],
            tau_fwhm=physical_parameters["laser_pulse_duration_fwhm_s"],
            waist=physical_parameters["laser_spot_size_m"],
            focal_position=physical_parameters["laser_focal_position_m"],
            super_gaussian_order=physical_parameters["laser_super_gaussian_order"],
            zernike_coefficients={
                key.removeprefix("zernike_"): physical_parameters[key]
                for key in physical_parameters.keys()
                if key.startswith("zernike_")
            },
            spectral_bandwidth=physical_parameters.get(
                "laser_spectral_bandwidth_rad_s", None
            ),
            cep=physical_parameters.get("laser_cep_phase_rad", None),
            gdd=physical_parameters.get("laser_gdd_s2", None),
            tod=physical_parameters.get("laser_tod_s3", None),
            fod=physical_parameters.get("laser_fod_s4", None),
            gdd_relative=physical_parameters.get("laser_gdd_relative", None),
            tod_relative=physical_parameters.get("laser_tod_relative", None),
            polarization=hyperparameters["laser_polarization"],
            n_azimuthal_modes=hyperparameters["laser_n_azimuthal_modes"],
            num_points=hyperparameters["laser_num_points"],
            hi_range=hyperparameters["laser_hi_range"],
            center_and_remove_tilt=hyperparameters["laser_center_and_remove_tilt"],
            centering_angles=hyperparameters["laser_centering_angles"],
            peak_delay_from_file_start=hyperparameters[
                "laser_peak_delay_from_file_start_s"
            ],
            maximum_pulse_duration_fwhm=hyperparameters[
                "laser_maximum_pulse_duration_fwhm_s"
            ],
        )

        hyp = sm.SimulationHyperparameters(
            zmin=hyperparameters["zmin"],
            zmax=hyperparameters["zmax"],
            rmax=hyperparameters["rmax"],
            nz=hyperparameters["nz"],
            nr=hyperparameters["nr"],
            nm=hyperparameters["n_azimuthal_modes"],
            use_mpi=True,
            number_dumps=hyperparameters["number_dumps_lab"],
            gamma_boost=hyperparameters["gamma_boost"],
            field_diagnostics=(
                ["rho", "E", "B"] if hyperparameters["dump_fields"] else None
            ),
            save_directory=Path(hyperparameters["lab_diagnostic_directory"]),
            save_checkpoints=hyperparameters["save_checkpoints"],
            checkpoint_period=hyperparameters["checkpoint_period"],
            write_period=hyperparameters["write_period"],
            use_restart=hyperparameters["use_restart"],
            track_electrons=hyperparameters["track_electrons"],
            n_order_mpi=hyperparameters["n_order"],
            random_seed=hyperparameters["random_seed"],
            r_boundary=hyperparameters["r_boundary"],
        )

        parameters = cc.ConfigContainer(
            configs=[
                dp.Parameters(label="hyperparameters", data=hyperparameters),
                dp.Parameters(label="physical_parameters", data=physical_parameters),
                dp.Parameters(
                    label="resolved_density_parameters",
                    data={**left_density_parameters, **right_density_parameters},
                ),
            ]
        )

        return sm.Simulation(elements=[hyp, density_he, density_n2, laser, parameters])

    def run(self) -> None:
        """Run and save the simulation, then queue optional background movies."""
        hyperparameters = self.hyperparameters
        left_density_parameters = self._resolved_density_ramp_parameters("left")
        right_density_parameters = self._resolved_density_ramp_parameters("right")
        sim = self.build_simulation()
        parameters = cc.ConfigContainer(
            configs=[
                dp.Parameters(label="hyperparameters", data=hyperparameters),
                dp.Parameters(
                    label="physical_parameters", data=self.physical_parameters
                ),
                dp.Parameters(
                    label="resolved_density_parameters",
                    data={**left_density_parameters, **right_density_parameters},
                ),
            ]
        )
        sim.setup_simulation()
        sim.run_simulation()
        if sim.do_write_to_disk:  # MPI guard
            # contains all simulation parameters and element metadata
            sim.to_file(
                Path(hyperparameters["results_directory"])
                / f"simulation.{_SERIALIZATION_FORMAT}",
                _SERIALIZATION_FORMAT,
                overwrite=True,
            )
            # contains only the simulation parameters
            parameters.to_file(
                Path(hyperparameters["results_directory"])
                / f"parameters.{_PARAMETERS_FORMAT}",
                _PARAMETERS_FORMAT,
                group_path="/results",
                overwrite=True,
            )

        if hyperparameters["plot_movies"]:
            self.movie_future = self._queue_movies(
                sim, Path(hyperparameters["results_directory"])
            )
