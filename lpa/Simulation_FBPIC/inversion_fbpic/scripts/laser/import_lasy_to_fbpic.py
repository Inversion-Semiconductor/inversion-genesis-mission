"""
This is an example to build a realistic laser from FROG data, import it into FBPIC, and run sims!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import time
from scipy.constants import mu_0, epsilon_0, c

from inversion_fbpic.utils.laser import HTULasyLaser

from openpmd_viewer.addons import LpaDiagnostics

from lasy.profiles import Profile
from lasy.profiles.transverse import GaussianTransverseProfile
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.laser import Laser

from fbpic.main import Simulation
from fbpic.utils.random_seed import set_random_seed
from fbpic.openpmd_diag import FieldDiagnostic, set_periodic_checkpoint
from fbpic.lpa_utils.laser import FromLasyFileLaser, add_laser_pulse

SKIP_SIMULATION: bool = True
LOAD_ITERATION: int = 10

LASER_SPOT_SIZE: float = 28e-6  # m
LASER_ENERGY: float = 2.5  # J
LASER_WAVELENGTH: float = 0.800e-6  # m
BACK_PROPAGATION_DISTANCE: float = 2.25e-3  # m

#### Load Data from File ####

# Parameters to point to folder of saved data, and the particular scan number and date
SUPER_PATH: Path = Path("/Users/cedoss/Desktop/data/HTU_FROG_Data/")
SCAN_NUMBER: int = 9
YEAR: int = 2023
MONTH: int = 7
DAY: int = 27
INDEX: int = 20

path_parent = SUPER_PATH / f"{MONTH:02d}-{DAY:02d}-{YEAR:04d}-Scan{SCAN_NUMBER:02d}"
path_file = path_parent / f"Scan{SCAN_NUMBER:03d}_U_FROG_Grenouille_{INDEX:03d}.txt"
htu_laser = HTULasyLaser(path_file)
if do_plot_after_load := True:
    htu_laser.plot_temporal_profile(title=f"{MONTH:02d}-{DAY:02d}-{YEAR:04d}: {path_file.name}")

#### Use temporal laser profile to crease a full laser profile ####

longitudinal_profile = htu_laser.get_laser_profile()
transverse_profile = GaussianTransverseProfile(
    w0=LASER_SPOT_SIZE,
    wavelength=LASER_WAVELENGTH,
    z_foc=0e-3,  # Initialize at waist, then back-propagate later
)

combined_profile: Profile = CombinedLongitudinalTransverseProfile(
    wavelength=LASER_WAVELENGTH,
    pol=(1, 0),  # Linearly polarized in the x direction
    laser_energy=LASER_ENERGY,   # J
    long_profile=longitudinal_profile,
    trans_profile=transverse_profile,
)

pulse_duration = htu_laser.calculate_fwhm_intensity() * 1.5 * 1e-15  # s
spot_size = LASER_SPOT_SIZE * 1.5  # m

dimensions = 'rt'  # Use cylindrical geometry
lo = (0.0, -3.5 * pulse_duration)  # Lower bounds of the simulation box
hi = (5 * spot_size, 3.5 * pulse_duration)  # Upper bounds of the simulation box
num_points = (500, 2000)  # Number of points in each dimension

if do_plot_mid_test := False:
    x = np.linspace(-hi[0], hi[0], num_points[0])
    t = np.linspace(lo[1], hi[1], num_points[1])

    plt.plot(x*1e6, np.abs(combined_profile.evaluate(x=x, y=0, t=0) ** 2), c='b')
    plt.xlabel("x (um)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Transverse Profile")
    plt.show()

    plt.plot(t*1e15, np.abs(combined_profile.evaluate(x=0, y=0, t=t) ** 2), c='r')
    plt.xlabel("t (fs)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Longitudinal Profile")
    plt.show()

laser = Laser(dimensions, lo, hi, num_points, combined_profile)
laser.normalize(2.5)  # Joules
if do_plot_before_save := False:
    laser.show()
    plt.show()

laser.propagate(-BACK_PROPAGATION_DISTANCE)  # Propagate backwards by 1 mm
laser.write_to_file('lasy_laser', 'h5')

# Note that, in the lasy file, tmin is now a large, negative number.
# (The peak of laser intensity still occurs 2.5*pulse_duration after tmin.)
# FBPIC ignores tmin when reading the file, and sets the start of
# the lasy time axis to zero instead. So, by default, the peak of
# the laser intensity would occur at `t= 2.5*pulse_duration` in FBPIC.
laser_profile = FromLasyFileLaser('diags/lasy_laser_00000.h5', t_start=0.1 * pulse_duration)

#### Test plot to see if it worked! ####
if do_plot_after_read := False:
    x = np.linspace(-hi[0], hi[0], num_points[0])
    t = np.linspace(lo[1], hi[1], num_points[1])

    plt.plot(x*1e6, laser_profile.E_field(x=x, y=0, z=0, t=0)[0], c='b')
    plt.xlabel("x (um)")
    plt.ylabel("Electric Field in x (V/m)")
    plt.title("Transverse Profile")
    plt.show()

    plt.plot(t*3e8*1e6, laser_profile.E_field(x=0, y=0, z=0, t=t)[0], c='r')
    plt.xlabel("z (um)")
    plt.ylabel("Electric Field in x (V/m)")
    plt.title("Longitudinal Profile")
    plt.show()

if do_plot_before_sim := False:
    impedance = np.sqrt(mu_0 / epsilon_0)
    x = np.linspace(lo[0], hi[0], num_points[0])
    t = np.linspace(lo[1], hi[1], num_points[1])
    efield = np.zeros((len(x), len(t)))
    intensity = np.zeros((len(x), len(t)))

    for i in range(len(x)):
        for j in range(len(t)):
            efield[i, j] = laser_profile.E_field(x=x[i], y=0, z=0, t=t[j])[0]
            intensity[i, j] = np.square(efield[i, j]) / (2 * impedance)

    plt.imshow(intensity)
    plt.show()

if not SKIP_SIMULATION:
    #### Setup basic laser propagation in FBPIC ####
    # Delete any existing "diags" folder:
    diag_dir = Path("diags")
    if diag_dir.exists():
        shutil.rmtree(diag_dir)

    # The simulation box
    Nz: int = 1000  # Number of gridpoints along z
    zmax: float = 0.0e-6  # Right end of the simulation box (meters)
    zmin: float = -150.0e-6  # Left end of the simulation box (meters)
    Nr: int = 100  # Number of gridpoints along r
    rmax: float = 150.0e-6  # Length of the box along r (meters)
    Nm: int = 2  # Number of modes used

    dt: float = (zmax - zmin) / Nz / c  # Timestep (seconds)
    n_order = -1
    use_cuda = False
    v_window: float = c  # Speed of the window

    number_dumps: int = 20  # Used to calculate the diagnostic period
    # The interaction length of the simulation (meters)
    L_interact: float = 10e-3
    # Interaction time (seconds) (to calculate number of PIC iterations)
    T_interact: float = L_interact / v_window

    save_checkpoints: bool = False  # Whether to write checkpoint files
    checkpoint_period: int = 100  # Period for writing the checkpoints
    use_restart: bool = False  # Whether to restart from a previous checkpoint
    track_electrons: bool = False  # Whether to track and write particle ids

    # Set the random seed
    set_random_seed(0)

    sim = Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt,
        zmin=zmin,
        n_order=n_order,
        use_cuda=use_cuda,
        boundaries={"z": "open", "r": "reflective"},
    )


    add_laser_pulse(sim, laser_profile, method='antenna', z0_antenna=0)
    # Here, modify the `t_start`, which will result in the peak of intensity being
    # emitted at `(2.5+0.5)*pulse_duration` instead of `2.5*pulse_duration`.
    # Since the `z0_antenna` was set to `0` here, this is as if the centroid
    # of the laser would have been initialized at `z = -3*c*pulse_duration` at `t=0`
    # (with then ``direct`` method).

    sim.set_moving_window(v=v_window)

    # Add diagnostics
    save_folder = "diags"
    diag_period = int((T_interact / dt) / number_dumps)
    sim.diags = [
        FieldDiagnostic(
            diag_period,
            sim.fld,
            comm=sim.comm,
            write_dir=save_folder,
        ),
    ]

    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint(sim, checkpoint_period)

    n_step = int(T_interact / sim.dt) + 1
    print(f" -Number of steps: {n_step}")

    # Run the simulation
    start_time = time.time()
    sim.step(n_step, show_progress=True)
    print(f" -Finished in {(time.time()-start_time)/60} min")
    print("")

#### Quality Assurance ####

# From the first dump after the initial, load in the Ex field and plot vs z

diag_dir = Path("diags/hdf5")
if not diag_dir.exists():
    raise FileNotFoundError(f"Diagnostics folder not found: {diag_dir}")

ts = LpaDiagnostics(str(diag_dir), check_all_files=True)
iterations = sorted(ts.iterations)
if LOAD_ITERATION > 1 and len(iterations) < LOAD_ITERATION + 1:
    raise RuntimeError(f"Need at least {LOAD_ITERATION} dumps.")

ex_field = ts.get_field(field="E", coord="x", iteration=iterations[LOAD_ITERATION], plot=True)
plt.show()

z_min = ex_field[1].imshow_extent[0]
z_max = ex_field[1].imshow_extent[1]

ex_field_arr = np.array(ex_field[0])
z_axis = np.linspace(z_min, z_max, np.shape(ex_field_arr)[1])

middle_index = int(np.shape(ex_field_arr)[0]/2)
ex_field_center = np.array(ex_field_arr[middle_index])

plt.figure()
plt.plot(z_axis, ex_field_center)
plt.xlabel("z (µm)")
plt.ylabel("Ex (V/m)")
plt.title("Ex field vs z (first dump)")
plt.tight_layout()
plt.show()
