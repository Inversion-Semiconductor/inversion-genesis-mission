"""lume_fbpic version of `runs/initial_sample/ionization_injection_runscript_00.py` and its
template, `ionization_injection_template.py`: a boosted-frame ionization-injection LPA in a
helium/nitrogen gas jet, driven by an unaberrated (all Zernike coefficients 0) super-Gaussian
LASY laser pulse (`LasyLaserPulse`), coarsened to run on a CPU.

It uses the runscript's physical parameters and sets the simulation up as the template does:

- Gas jet: two `BilateralSuperGaussian` profiles of the same symmetric shape (equal FWHM and
  exponent on both sides), one of He and one of N2. The dopant fraction is one of molecules: the
  He atom density is `n_gas * (1 - fraction)` and the N atom density `2 * n_gas * fraction`,
  where `n_gas` is the peak neutral density. `nominal_density` is the atom density times the
  species' ionization levels, because `_DensityProfile` divides by the level count.
- Moving window: `Simulation` sets `beta_window` from the laser's mean group velocity.
- Interaction length: the profiles' z extent, clipped to `p_zmin=0`, plus the default
  `right_buffer` (the window length).
- Laser: `z0` is 0, as in the template, where it is only used for plotting.

The constants below are a coarsened configuration for a CPU, not the runscript's production grid
(`PRODUCTION`): `nz=1000`, `nr=100`, `nm=2`, a (300, 450) / 2-mode LASY grid, `p_nt=4` and 10
lab-frame dumps. `Simulation` passes the runscript's `n_order=32` only with `use_mpi=True`
(`PRODUCTION`); this single-process run gets `n_order=-1`. It always requests CUDA, which fbpic
downgrades to CPU when it is not installed. The coarse grid has not been checked to inject
electrons with these parameters.

The template also attaches `Parameters` and a `MomentDescriptorDiagnostic` to its `Simulation`
and writes them to `results/`. Here the moment descriptor is the model's `descriptor_*`
actions.
"""

from __future__ import annotations

from pathlib import Path

from lume_fbpic.actions import (
    DensityFieldAction,
    DopantFractionAction,
    FinalParticlesAction,
    LaserFieldAction,
    ZernikeCoefficientAction,
    make_descriptor_actions,
    make_stat_actions,
)
from lume_fbpic.density_profiles import nominal_density
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.lwfa_simulator import LWFASimulator

from inversion_fbpic.lib.density_profiles import BilateralSuperGaussian
from inversion_fbpic.lib.laser import LasyLaserPulse
from inversion_fbpic.lib.simulation import SimulationHyperparameters

# --- physical parameters: PHYSICAL_PARAMETERS of ionization_injection_runscript_00.py ---
PEAK_NEUTRAL_DENSITY_CM3 = 1.5e18
NITROGEN_DOPANT_FRACTION = 0.03  # of N2 molecules
DENSITY_FWHM_M = 1.0e-3
DENSITY_BETA = 1.5
DENSITY_CENTER_LOCATION_M = 4.0e-3
LASER_WAVELENGTH_M = 800e-9
LASER_ENERGY_J = 2.5
LASER_PULSE_DURATION_FWHM_S = 30e-15
LASER_SPOT_SIZE_M = 24e-6
LASER_SUPER_GAUSSIAN_ORDER = 3.0
LASER_FOCAL_POSITION_M = 3.5e-3
# --- hyperparameters: the coarsened CPU configuration of the module docstring; the runscript's
# own are `PRODUCTION`.
NZ, NR, N_AZIMUTHAL_MODES = 1000, 100, 2
ZMIN, ZMAX, RMAX = -70.0e-6, 0.0, 200.0e-6
GAMMA_BOOST = 1.5
NUMBER_DUMPS = 10
WRITE_PERIOD = 10
P_NZ, P_NR, P_NT = 2, 2, 4
P_ZMIN, P_RMAX = 0.0, 90.0e-6
# The runscript's `diagnostic_select`: only electrons with uz >= 10 are written.
DIAGNOSTIC_SELECT = {"uz": [10.0, None]}
# --- LASY laser grid, coarsened to match (the runscript's in `PRODUCTION`) ---
LASER_N_AZIMUTHAL_MODES = 2
LASER_NUM_POINTS = (300, 450)
Z0_ANTENNA = 0.0
T_START = 0.0
# The runscript's production settings, which the constants above coarsen. POSIT: the
# HYPERPARAMETERS of runs/initial_sample/ionization_injection_runscript_00.py.
PRODUCTION = {
    "nz": 1500,
    "nr": 300,
    "nm": 5,
    "number_dumps": 50,
    "write_period": 50,
    "use_mpi": True,  # submission_script_00.sh runs 8 ranks under srun
    "p_nt": 15,
    "laser_n_azimuthal_modes": 5,
    "laser_num_points": (600, 900),
}
# The Zernike coefficients' actions are named `ZERNIKE_PREFIX` + coefficient, as the runscript's
# inputs are.
ZERNIKE_PREFIX = "zernike_"


def build_model(working_directory: str | None = None) -> LUMEFBPICModel:
    n_gas = PEAK_NEUTRAL_DENSITY_CM3 * 1.0e6  # cm^-3 -> m^-3
    hyparams = SimulationHyperparameters(
        zmin=ZMIN,
        zmax=ZMAX,
        rmax=RMAX,
        nz=NZ,
        nr=NR,
        nm=N_AZIMUTHAL_MODES,
        use_mpi=False,
        number_dumps=NUMBER_DUMPS,
        write_period=WRITE_PERIOD,
        gamma_boost=GAMMA_BOOST,
        random_seed=0,
    )
    laser = LasyLaserPulse(
        energy=LASER_ENERGY_J,
        wavelength=LASER_WAVELENGTH_M,
        tau_fwhm=LASER_PULSE_DURATION_FWHM_S,
        waist=LASER_SPOT_SIZE_M,
        focal_position=LASER_FOCAL_POSITION_M,
        super_gaussian_order=LASER_SUPER_GAUSSIAN_ORDER,
        zernike_coefficients={},  # all zero, as in the runscript
        spectral_bandwidth="auto",  # transform-limited for the duration
        cep=0.0,
        gdd_relative=0.0,
        tod_relative=0.0,
        fod=0.0,
        z0=0.0,  # only used for plotting
        z0_antenna=Z0_ANTENNA,
        t_start=T_START,
        n_azimuthal_modes=LASER_N_AZIMUTHAL_MODES,
        num_points=LASER_NUM_POINTS,
        hi_range=10.0,
        center_and_remove_tilt=True,
        centering_angles=72,
        lasy_file=Path("diags") / "lasy_laser",
    )
    # The same symmetric shape for both gases; the atom densities follow the molecular dopant
    # fraction (see the module docstring).
    shape = dict(
        fwhm_left=DENSITY_FWHM_M,
        fwhm_right=DENSITY_FWHM_M,
        beta_left=DENSITY_BETA,
        beta_right=DENSITY_BETA,
        center=DENSITY_CENTER_LOCATION_M,
        p_zmin=P_ZMIN,
        p_rmax=P_RMAX,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        ionization=0,
        elec_select=DIAGNOSTIC_SELECT,
    )
    density_he = BilateralSuperGaussian(
        nominal_density=nominal_density("He", n_gas * (1.0 - NITROGEN_DOPANT_FRACTION)),
        species="He",
        elec_name="electrons_he",
        **shape,
    )
    density_n2 = BilateralSuperGaussian(
        nominal_density=nominal_density("N", 2.0 * n_gas * NITROGEN_DOPANT_FRACTION),
        species="N",
        elec_name="electrons_n2",
        **shape,
    )
    simulator = LWFASimulator(
        hyparams,
        laser,
        [density_he, density_n2],
        target_species="electrons_n2",
        working_directory=working_directory,
    )
    actions = [
        LaserFieldAction(name="laser_energy", field_name="energy"),
        LaserFieldAction(name="laser_focal_position", field_name="focal_position"),
        LaserFieldAction(name="laser_temporal_width", field_name="tau_fwhm"),
        DensityFieldAction(
            name="nitrogen_dopant_density",
            density_index=1,
            field_name="nominal_density",
        ),
        # The runscript's `nitrogen_dopant_fraction`, of N2 molecules: rewrites BOTH gas
        # densities at constant neutral density (profile 0 = He, profile 1 = N).
        DopantFractionAction(
            name="nitrogen_dopant_fraction",
            host_index=0,
            dopant_index=1,
            dopant_atoms_per_molecule=2,
        ),
        # One action per Zernike coefficient, named like the runscript's inputs (for example
        # `zernike_astigmatism_4`, `zernike_coma_x`); phase amplitudes in radians.
        *[
            ZernikeCoefficientAction(
                name=ZERNIKE_PREFIX + coefficient, coefficient=coefficient
            )
            for coefficient in laser.zernike_coefficients
        ],
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
        # The 33 moment-descriptor scalars build_dataset.py writes (uz >= 30, central 95%).
        *make_descriptor_actions(),
    ]
    return LUMEFBPICModel(simulator, actions)


if __name__ == "__main__":
    model = build_model()
    model.set({})  # runs the simulation (an empty set changes nothing first)
    print(model.get(["charge_pc", "energy_mean_mev"]))
