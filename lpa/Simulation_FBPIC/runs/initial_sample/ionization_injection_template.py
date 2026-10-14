"""Class-based boosted-frame nitrogen ionization-injection FBPIC simulation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import itertools
import logging
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
    density profile specified by ``density_fwhm_left_m``,
    ``density_fwhm_right_m``, ``density_beta_left``, and ``density_beta_right``.
    """

    movie_future: Future[None] | None = None

    _PHYSICAL_PARAMETER_KEYS = frozenset(
        {
            "peak_neutral_density_cm3",
            "nitrogen_dopant_fraction",
            "density_fwhm_left_m",
            "density_fwhm_right_m",
            "density_beta_left",
            "density_beta_right",
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
        "r_boundary": "reflective",
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
        """Save and close profile figures, then render diagnostic movies."""
        import matplotlib.pyplot as plt

        try:
            save_path.mkdir(parents=True, exist_ok=True)
            fig = simulation.plot_densities()
            try:
                fig.savefig(save_path / "densities.png")
            finally:
                plt.close(fig)

            for i, las in enumerate(simulation.lasers):
                fig = las.plot()
                try:
                    fig.savefig(save_path / f"laser_{i}.png")
                finally:
                    plt.close(fig)

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

    def run(self) -> None:
        """Run and save the simulation, then queue optional background movies."""
        hyperparameters = self.hyperparameters
        physical_parameters = self.physical_parameters

        density_he = dn.BilateralSuperGaussian(
            nominal_density=physical_parameters["peak_neutral_density_cm3"]
            * 1e6
            * (1.0 - physical_parameters["nitrogen_dopant_fraction"])
            * 2.0,
            fwhm_left=physical_parameters["density_fwhm_left_m"],
            fwhm_right=physical_parameters["density_fwhm_right_m"],
            beta_left=physical_parameters["density_beta_left"],
            beta_right=physical_parameters["density_beta_right"],
            center=physical_parameters["density_center_location_m"],
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
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
            fwhm_left=physical_parameters["density_fwhm_left_m"],
            fwhm_right=physical_parameters["density_fwhm_right_m"],
            beta_left=physical_parameters["density_beta_left"],
            beta_right=physical_parameters["density_beta_right"],
            center=physical_parameters["density_center_location_m"],
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
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
            ]
        )

        sim = sm.Simulation(elements=[hyp, density_he, density_n2, laser, parameters])

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
