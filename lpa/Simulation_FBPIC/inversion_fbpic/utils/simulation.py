"""
Laser Wakefield Acceleration (LWFA) simulation setup for a simple density profile with a downramp using FBPIC.

Outputs:
    Diagnostic HDF5 files in subdirectories for each scan case.

For detailed documentation of FBPIC structures, use:
    print(fbpic_object.__doc__)
where fbpic_object is any FBPIC object or function.
"""

# TODO: Create Laser class which handles the laser profile and its various parameters.
# TODO: Create InputDeck class (e.g., named tuple, similar to InputParameters class), update this function to use it. Should be able to handle different density profiles (e.g., using string literals and further nested dicts)
# TODO: Add ability to handle different species
# TODO: Make MPI-safe (e.g., print only on rank 0 if MPI is used)

import numpy as np
from scipy.constants import c, e, m_e, m_p, pi, epsilon_0

# Import the relevant structures from fbpic
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.main import Simulation
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.openpmd_diag import (
    BackTransformedFieldDiagnostic,
    ParticleDiagnostic,
    FieldDiagnostic,
    restart_from_checkpoint,
    set_periodic_checkpoint,
    BackTransformedParticleDiagnostic,
)
from fbpic.utils.random_seed import set_random_seed
from typing import Callable

# Try to import density profile functions from the package first, fall back to local copy if package not available
try:
    from inversion_fbpic.utils.input_params import InputParameters
    from inversion_fbpic.utils.simulation_setup_tools import (
        calculate_laser_a0_from_energy,
    )

    print("Importing relevant packages from environment")

except ImportError:
    from input_params import InputParameters  # type: ignore
    from simulation_setup_tools import calculate_laser_a0_from_energy  # type: ignore

    print("Importing relevant packages locally")

LOG_TO_FILE = True
USE_CUDA = True  # Whether to use the GPU


def setup_simulation(
    density_profile: Callable[[float, float], float],
    plasma_length: float,
    flattop_plasma_density: float = 4.0e18,
    ionization: int | None = None,
    laser_energy: float = 5.0,
    laser_focal_position: float = 2.0e-3,
    laser_focus_size: float = 20.0e-6,
    tau: float = 40.0e-15,
    wavelength: float = 0.800,
    zmax: float = 0.0e-6,
    zmin: float = -70.0e-6,
    rmax: float = 140.0e-6,
    p_rmax: float | None = None,
    z0: float = -30.0e-6,
    gamma_boost: None | float = None,
    nz: int = 2048,
    nr: int = 300,
    nm: int = 3,
    p_nz: int = 2,
    p_nr: int = 2,
    p_nt: int = 4,
    right_buffer: float | None = None,
    beam_min_uz: float = 10.0,
    number_dumps_lab: int = 100,
    save_directory: str = "diags",
    save_checkpoints: bool = False,
    checkpoint_period: int = 100,
    write_period: int = 50,
    use_restart: bool = False,
    track_electrons: bool = False,
    use_mpi: bool = False,
    params_recorder: InputParameters | None = None,
) -> tuple[Simulation, float]:
    """
    Sets up and configures the FBPIC simulation.
    The simulation is set up for a flat-top plasma density profile with a downramp using FBPIC.

    Args:
        density_profile: [Callable[[float, float], float]] The density profile function, f(z, r).
        plasma_length: [m] The total length of the plasma region.
        flattop_plasma_density: [cm^-3] The plasma density within the unperturbed flattop. Defaults to 4.0e18 cm^-3.
        ionization: [dimensionless] The ionization level of the gas. If None, only model the plasma electrons. If int, the gas starts at an ionization level of `ionization`. Defaults to None.
        laser_energy: [J] The energy of the laser. Defaults to 5.0 J.
        laser_focal_position: [m] The focal position of the laser. Defaults to 2.0e-3 m.
        laser_focus_size: [m] The target focus size of the laser. Defaults to 20.0e-6 m.
        tau: [s] The temporal width (FWHM power) of the laser. Defaults to 40.0e-15 s.
        wavelength: [um] The wavelength of the laser. Defaults to 0.800 um.
        zmax: [m] The maximum z position of the simulation box. Defaults to 0.0e-6 m.
        zmin: [m] The minimum z position of the simulation box. Defaults to -70.0e-6 m.
        rmax: [m] The maximum r position of the simulation box. Defaults to 140.0e-6 m.
        p_rmax: [m] The maximum r position of the particle distribution. If None, equal to rmax. Defaults to None.
        z0: [m] The position of the laser centroid in the simulation box. Defaults to -30.0e-6 m.
        gamma_boost: [dimensionless] The gamma boost factor. Defaults to None (no boost).
        nz: [int] The number of gridpoints along z. Defaults to 2048.
        nr: [int] The number of gridpoints along r. Defaults to 300.
        nm: [int] The number of angular modes used. Defaults to 3.
        p_nz: [int] The number of particles per cell along z. Defaults to 2.
        p_nr: [int] The number of particles per cell along r. Defaults to 2.
        p_nt: [int] The number of particles per cell along theta. Defaults to 4.
        right_buffer: [m] The right buffer of the simulation extent. If None, equal to the window size (zmax - zmin). Defaults to None.
        beam_min_uz: [float] The minimum uz value to save for the beam electrons. Defaults to 10.0.
        number_dumps_lab: [int] The number of diagnostics to save in the lab frame. Defaults to 100.
        save_directory: [str] The directory to save the diagnostics. Defaults to "diags".
        save_checkpoints: [bool] Whether to save checkpoints. Defaults to False.
        checkpoint_period: [int] The period to save checkpoints. Defaults to 100.
        write_period: [int] The period to write diagnostics to disk. Defaults to 50.
        use_restart: [bool] Whether to restart from a checkpoint. Defaults to False.
        track_electrons: [bool] Whether to track and write particle ids. Defaults to False.
        use_mpi: [bool] Whether to use MPI. Defaults to False.
        params_recorder: [InputParameters] The parameters recorder object. Defaults to None.


    Returns:
        tuple[Simulation, float]: A tuple containing a configured Simulation object ready to run and the interaction time.
    """

    if params_recorder is None:
        params_recorder = InputParameters()

    # Order of the stencil for z derivatives in the Maxwell solver.
    # Use -1 for infinite order, i.e. for exact dispersion relation in
    # all direction (adviced for single-GPU/single-CPU simulation).
    # Use a positive number (and multiple of 2) for a finite-order stencil
    # (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
    # to more overhead in MPI communications, but also to a more accurate
    # dispersion relation for electromagnetic waves. (Typically,
    # `n_order = 32` is a good trade-off.)
    # See https://arxiv.org/abs/1611.05712 for more information.
    n_order: int = 32 if use_mpi else -1

    # Boosted frame converter
    if gamma_boost is not None and gamma_boost > 1.0:
        boost = BoostConverter(gamma_boost)
    else:
        boost = None

    # The simulation timestep
    dt = (
        min(rmax / (2 * boost.gamma0 * nr) / c, (zmax - zmin) / nz / c)
        if boost is not None
        else (zmax - zmin) / nz / c
    )  # Timestep (seconds)

    # The particles
    p_zmin: float = 0.0e-6  # Position of the beginning of the plasma (meters)
    if p_rmax is None:
        p_rmax = rmax
    n_plasma: float = flattop_plasma_density * 1.0e6  # Density (electrons.meters^-3)
    n_gas = (
        n_plasma / 2
    )  # Gas density with the assumption of fully ionizing both levels of Helium

    # a0 Calculation
    a0 = calculate_laser_a0_from_energy(laser_energy, wavelength, laser_focus_size, tau)

    # The moving window (moves with the group velocity in the flattop plasma)
    v_window = c * (
        1.0
        - (wavelength * 1e-6) ** 2
        * e**2
        / (8.0 * pi**2 * c**2 * m_e * epsilon_0)
        * flattop_plasma_density
        * 1e6
    )

    # Velocity of the Galilean frame (for suppression of the NCI)
    if boost is not None:
        v_comoving = -c * np.sqrt(1.0 - 1.0 / boost.gamma0**2)
    else:
        v_comoving = 0.0

    # TODO: Make this something that is passed to this function so that it can be used for other simulations

    # The interaction length of the simulation (meters)
    if right_buffer is None:
        right_buffer = zmax - zmin
    L_interact: float = plasma_length + right_buffer
    # Interaction time, in the boosted frame (seconds)
    if boost is not None:
        T_interact = boost.interaction_time(L_interact, (zmax - zmin), v_window)
    else:
        T_interact = L_interact / v_window

    # Set the random seed
    set_random_seed(0)

    # Initialize the simulation object
    sim = Simulation(
        nz,
        zmax,
        nr,
        rmax,
        nm,
        dt,
        zmin=zmin,
        v_comoving=v_comoving if boost is not None else None,
        gamma_boost=boost.gamma0 if boost is not None else None,
        n_order=n_order,
        use_cuda=USE_CUDA,
        boundaries={"z": "open", "r": "reflective"},
        use_all_mpi_ranks=use_mpi,
    )
    # 'r': 'open' can also be used, but is more computationally expensive

    # Add the Helium atoms
    atoms_He = sim.add_new_species(
        q=e * (ionization if ionization is not None else 2),
        m=4.0 * m_p,
        n=n_gas,
        dens_func=density_profile,
        boost_positions_in_dens_func=boost is not None,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )

    # Store the created electrons in the species `elec`
    elec = sim.add_new_species(
        q=-e,
        m=m_e,
        n=(n_gas * ionization if ionization is not None else n_plasma),
        dens_func=density_profile,
        boost_positions_in_dens_func=boost is not None,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )
    if ionization is not None:
        assert ionization <= 2, "Only singly or doubly ionized He is possible"
        atoms_He.make_ionizable("He", target_species=elec, level_start=ionization)

    # Load initial fields
    # Create a Gaussian laser profile
    laser_profile = GaussianLaser(
        a0, laser_focus_size, tau, z0, zf=laser_focal_position
    )
    # Add the laser to the fields of the simulation
    add_laser_pulse(
        sim, laser_profile, gamma_boost=boost.gamma0 if boost is not None else None
    )

    if not use_restart:
        # Track electrons if required (species 0 correspond to the electrons)
        if track_electrons:
            elec.track(sim.comm)
    else:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint(sim)

    # Configure the moving window
    if boost is not None:
        (v_window_boosted,) = boost.velocity([v_window])
        sim.set_moving_window(v=v_window_boosted)
    else:
        sim.set_moving_window(v=v_window)

    # Add diagnostics
    dt_lab_diag_period = (
        (L_interact + (zmax - zmin)) / v_window / (number_dumps_lab - 1)
    )
    if boost is not None:
        sim.diags = [
            # Diagnostics in the lab frame (back-transformed)
            BackTransformedParticleDiagnostic(
                zmin,
                zmax,
                v_window,
                dt_lab_diag_period,
                number_dumps_lab,
                boost.gamma0,
                write_period,
                sim.fld,
                species={"electrons": elec},
                select={"uz": [beam_min_uz, None]},
                comm=sim.comm,
                write_dir=save_directory,
            ),
            BackTransformedFieldDiagnostic(
                zmin,
                zmax,
                v_window,
                dt_lab_diag_period,
                number_dumps_lab,
                boost.gamma0,
                fieldtypes=["rho", "E", "B"],
                period=write_period,
                fldobject=sim.fld,
                comm=sim.comm,
                write_dir=save_directory,
            ),
        ]
    else:
        sim.diags = [
            # Diagnostics in the lab frame (back-transformed)
            ParticleDiagnostic(
                dt_period=dt_lab_diag_period,
                species={"electrons": elec},
                select={"uz": [beam_min_uz, None]},
                comm=sim.comm,
                write_dir=save_directory,
            ),
            FieldDiagnostic(
                dt_period=dt_lab_diag_period,
                fldobject=sim.fld,
                comm=sim.comm,
                write_dir=save_directory,
            ),
        ]

    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint(sim, checkpoint_period)

    # Save input parameters to a config .ini file:

    # Add Laser parameters
    params_recorder.add(
        "Laser",
        {
            "energy_J": laser_energy,
            "temporal_width_s": tau,
            "focus_size_m": laser_focus_size,
            "focal_position_m": laser_focal_position,
            "a0": a0,
        },
    )

    # Add Simulation parameters
    params_recorder.add(
        "Simulation",
        {
            "gamma_boost": gamma_boost,
            "nz": nz,
            "dz": (zmax - zmin) / nz,
            "nr": nr,
            "dr": rmax / nr,
        },
    )

    # Save to INI file
    params_recorder.save_to_ini(save_directory)

    return sim, T_interact
