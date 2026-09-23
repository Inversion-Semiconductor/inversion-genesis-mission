"""Generate a laser profile to model the HTU laser using the lasy package.

Code is taken heavily from the example on the official ReadTheDocs page:
https://lasydoc.readthedocs.io/en/latest/tutorials/gaussian_laser.html
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import get_full_field

# Laser parameters
WAVELENGTH: float = 800e-9  # Laser wavelength in meters
POLARIZATION: Tuple[int, int] = (1, 0)  # Linearly polarized in the x direction
ENERGY: float = 2.5  # Energy of the laser pulse in joules
SPOT_SIZE: float = 28e-6  # Waist of the laser pulse in meters
PULSE_DURATION: float = 38e-15  # Pulse duration of the laser in seconds
T_PEAK: float = 0.0  # Location of the peak of the laser pulse in time

# Simulation parameters
POSITION_IN_SIMULATION_WINDOW: float = 30e-6
DISTANCE_FROM_START_TO_FOCUS: float = 2.25e-3

# File output parameters
FILE_PREFIX: str = "test_output"  # The file name will start with this prefix
FILE_FORMAT: str = "h5"  # Format to be used for the output file


def create_laser_profile() -> GaussianProfile:
    """Create a Gaussian laser profile with the specified parameters."""
    return GaussianProfile(
        WAVELENGTH, POLARIZATION, ENERGY, SPOT_SIZE, PULSE_DURATION, T_PEAK
    )


def create_laser_simulation() -> Laser:
    """Create and configure the laser simulation."""
    laser_profile: GaussianProfile = create_laser_profile()
    
    dimensions: str = "rt"  # Use cylindrical geometry
    lo: Tuple[float, float] = (0, -2.5 * PULSE_DURATION)  # Lower bounds
    hi: Tuple[float, float] = (5 * SPOT_SIZE, 2.5 * PULSE_DURATION)  # Upper bounds
    num_points: Tuple[int, int] = (300, 500)  # Number of points in each dimension
    
    return Laser(dimensions, lo, hi, num_points, laser_profile)


def propagate_laser(laser: Laser) -> None:
    """Propagate the laser pulse upstream of the focal plane."""
    total_distance: float = DISTANCE_FROM_START_TO_FOCUS + POSITION_IN_SIMULATION_WINDOW
    laser.propagate(distance=-total_distance, show_progress=False)


def visualize_laser_field(laser: Laser) -> None:
    """Visualize the laser field distribution."""
    E_rt, extent = get_full_field(laser)
    extent[2:] *= 1e6
    extent[:2] *= 1e15
    tmin, tmax, rmin, rmax = extent
    vmax: float = np.abs(E_rt).max()
    
    plt.imshow(
        E_rt,
        origin="lower",
        aspect="auto",
        vmax=vmax,
        vmin=-vmax,
        extent=[tmin, tmax, rmin, rmax],
        cmap="bwr",
    )
    plt.xlabel("t (fs)")
    plt.ylabel("r (µm)")
    plt.show()


def main() -> None:
    """Main function to generate and visualize the laser profile."""
    # Create laser simulation
    laser: Laser = create_laser_simulation()
    
    # Show laser profile at focus (commented out)
    # laser.show()
    # plt.title("Laser profile at focus")
    # plt.show()
    
    # Propagate laser upstream
    propagate_laser(laser)
    
    # Show laser profile at simulation start
    laser.show()
    plt.title("Laser profile at simulation start")
    plt.show()
    
    # Save laser profile to file
    laser.write_to_file(FILE_PREFIX, FILE_FORMAT)
    
    # Visualize field
    visualize_laser_field(laser)


if __name__ == "__main__":
    main()
