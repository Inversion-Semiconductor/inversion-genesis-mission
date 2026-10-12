"""Class-based boosted-frame nitrogen ionization-injection FBPIC simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from inversion_fbpic.utils.laser import HighOrderLasyLaser
from inversion_fbpic.lib import (
    density_profiles as dn,
    laser as ls,
    simulation as sm,
    diagnostics as dg,
    datapoint as dp,
    config_container as cc,
)


class IonizationInjectionSimulation:
    """Configure and run a LASY-driven nitrogen ionization-injection case.

    ``physical_parameters`` contains flat, required laser and gas properties.
    ``hyperparameters`` holds FBPIC grid, particle-loading, diagnostic, and
    execution settings. Laser-prefixed physical parameters are passed directly
    to :class:`HighOrderLasyLaser`.
    """

    _PHYSICAL_PARAMETER_KEYS = frozenset(
        {
            "peak_plasma_density_cm3",  # NOTE: likely to replace this with neutral density
            "nitrogen_dopant_fraction",
            "density_alpha_m",
            "density_beta",
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
        # NOTE: Add hyperparameters for...?
        # - r_boundary
        # - plotting/movies
        # - beam minimum energy/select
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
        self.lasy_laser_file = self._laser_output_path()

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

    def _laser_output_path(self) -> Path:
        requested_path = Path(self.hyperparameters["lasy_file"])
        return requested_path.parent / f"{requested_path.stem}_00000.h5"

    def run(self) -> None:
        """Create the FBPIC simulation, run it, and save the results."""
        hyperparameters = self.hyperparameters
        physical_parameters = self.physical_parameters
        neutral_density = (
            physical_parameters["peak_plasma_density_cm3"]
            * 1.0e6
            / (2.0 + (14.0 - 2.0) * physical_parameters["nitrogen_dopant_fraction"])
        )

        density_he = dn.BilateralSuperGaussian(
            nominal_density=neutral_density
            * (1.0 - physical_parameters["nitrogen_dopant_fraction"])
            * 2.0,
            fwhm_left=2.0
            * physical_parameters["density_alpha_m"]
            * np.log(2.0) ** (1.0 / physical_parameters["density_beta"]),
            fwhm_right=2.0
            * physical_parameters["density_alpha_m"]
            * np.log(2.0) ** (1.0 / physical_parameters["density_beta"]),
            beta_left=physical_parameters["density_beta"],
            beta_right=physical_parameters["density_beta"],
            center=physical_parameters["density_center_location_m"],
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
            p_rmax=hyperparameters["p_rmax"],
            species="He",
            ionization=0,
            elec_name="electrons_he",
            elec_select={"uz": [10.0, None]},  # NOTE: Parameterize?
        )

        density_n2 = dn.BilateralSuperGaussian(
            nominal_density=neutral_density
            * physical_parameters["nitrogen_dopant_fraction"]
            * 14.0,
            fwhm_left=2.0
            * physical_parameters["density_alpha_m"]
            * np.log(2.0) ** (1.0 / physical_parameters["density_beta"]),
            fwhm_right=2.0
            * physical_parameters["density_alpha_m"]
            * np.log(2.0) ** (1.0 / physical_parameters["density_beta"]),
            beta_left=physical_parameters["density_beta"],
            beta_right=physical_parameters["density_beta"],
            center=physical_parameters["density_center_location_m"],
            p_nz=hyperparameters["p_nz"],
            p_nr=hyperparameters["p_nr"],
            p_nt=hyperparameters["p_nt"],
            p_zmin=hyperparameters["p_zmin"],
            p_rmax=hyperparameters["p_rmax"],
            species="N",
            ionization=0,
            elec_name="electrons_n2",
            elec_select={"uz": [10.0, None]},  # NOTE: Parameterize?
        )

        # NOTE: add plots of the density profiles?

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

        # NOTE: add plots of the laser profile?

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
            field_diagnostics=None,  # NOTE: add field diagnostics for plotting?
            save_directory=Path(hyperparameters["lab_diagnostic_directory"]),
            save_checkpoints=hyperparameters["save_checkpoints"],
            checkpoint_period=hyperparameters["checkpoint_period"],
            write_period=hyperparameters["write_period"],
            use_restart=hyperparameters["use_restart"],
            track_electrons=hyperparameters["track_electrons"],
            n_order_mpi=hyperparameters["n_order"],
            random_seed=hyperparameters["random_seed"],
            r_boundary="reflective",  # NOTE: Parameterize?
        )

        datapoint = cc.ConfigContainer(
            configs=[
                dp.Parameters(label="hyperparameters", data=hyperparameters),
                dp.Parameters(label="physical_parameters", data=physical_parameters),
                dg.MomentDescriptorDiagnostic(selection=("all_of_species", "e")),
            ]
        )

        sim = sm.Simulation(elements=[hyp, density_he, density_n2, laser, datapoint])

        sim.setup_simulation()
        sim.run_simulation()
        if sim.do_write_to_disk:  # MPI guard
            sim.to_hdf5_file(
                Path(hyperparameters["results_directory"]) / "simulation.h5"
            )  # contains all simulation parameters, results, element metadata
            datapoint.to_hdf5_file(
                Path(hyperparameters["results_directory"]) / "datapoint.h5",
                group_path="/results",
            )  # contains just the parameters and analyzed diagnostics
            # NOTE: Easy to change filetypes here, we can use json or yaml for either of these
