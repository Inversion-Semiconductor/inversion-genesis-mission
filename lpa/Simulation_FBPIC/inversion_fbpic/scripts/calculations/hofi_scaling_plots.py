"""
Calculates the optimal plasma parameters for HOFI plasma acceleration.

Uses the target ebeam energy, laser energy budget, and target intensity at focus to
calculate out the matched spot size, depletion length, depletion length, and
the laser pulse duration.

Note: the calculation is very strict and so it gives answers that have no room for error.
For instance, it will calculate out the required density so that the depletion length is
exactly equal to the required length for the target energy.
"""

from scipy.constants import c, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Input parameters
E_BEAM = 1.27  # GeV, Target electron beam energy
WAVELENGTH = 800e-9  # m, laser wavelength

E_LASER = np.linspace(2, 9, 200)
TAU_FWHM_FS = np.linspace(30, 70, 200)  # fs, pulse duration in FWHM intensity

MIN_SPOT_SIZE = 24e-6

POINT_1 = (6.9, 50)
POINT_1_LABEL = "Optimas Start"

POINT_2 = (3.153019108589136, 39.76856018121651)
POINT_2_LABEL = "Optimas End"

# Constants
pa = 8.7 * 1e9  # GW, constant
b1 = 96  # GeV/m, constant
b2 = 33e-6 / (2*pi)  # m, constant
b3 = 50e-6 / np.sqrt(5)  # m, constant
buffer = 0.99

# Laser Wavenumber
k0 = 2 * pi / WAVELENGTH

# Create 2D meshgrid
E_LASER_mesh, TAU_FWHM_FS_mesh = np.meshgrid(E_LASER, TAU_FWHM_FS)

# Initialize results arrays
results = np.zeros_like(E_LASER_mesh)  # Will store plasma density
a0_values = np.zeros_like(E_LASER_mesh)  # Will store a0 for contour lines

# Loop over all combinations
for i in range(len(TAU_FWHM_FS)):
    tau_fwhm = TAU_FWHM_FS[i] * 1e-15
    tau = tau_fwhm/1.17741
    for j in range(len(E_LASER)):
        e_laser = E_LASER[j]

        # Solve a_0
        a_0 = 8/(pa*np.sqrt(2*pi)) * (b1**2 * b2**4)/(b3**2) * e_laser/(E_BEAM**2) * buffer**2 * c**2 * tau * k0**2

        # Solve for density to reach desired electron beam energy
        n_p = (1 / E_BEAM * buffer * np.sqrt(a_0)/2 * b1 * b2 ** 2 * c * tau * k0 ** 2) ** 2

        # Calculations for parameters that only depend on tau
        w_0 = b3 / np.sqrt(n_p)

        lambda_p = 33e-6 / np.sqrt(n_p)

        p_0 = e_laser/tau * 2/np.sqrt(2*pi)
        a_0 = (16/pa * p_0 / (k0**2 * w_0**2))**(1/2)
        l_depl = c * tau * k0**2 * b2**2 / n_p
        l_deph = 2/3 * k0**2 * b2 **2 * w_0 / n_p
        l_acc = l_depl * buffer

        # Check conditions
        dephasing_condition = l_acc < l_deph
        spot_size_condition = w_0 > MIN_SPOT_SIZE

        # Set values based on conditions
        if not dephasing_condition:
            results[i, j] = -1.0  # Red - dephasing condition fails
            a0_values[i, j] = -1.0
        elif not spot_size_condition:
            results[i, j] = -2.0  # Blue - spot size too small
            a0_values[i, j] = -2.0
        else:
            results[i, j] = n_p  # m^-3, plasma density
            a0_values[i, j] = a_0  # dimensionless

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Find valid region values for colorbar limits
valid_values = results[results > 0]
if len(valid_values) > 0:
    vmin_valid = valid_values.min()
    vmax_valid = valid_values.max()
else:
    vmin_valid = 0
    vmax_valid = 100

# Create custom colormap
# Use viridis for valid values, and special colors for violations
cmap = plt.cm.viridis
colors_list = ['red', 'blue', 'orange'] + [cmap(i) for i in range(cmap.N)]
custom_cmap = ListedColormap(colors_list)

# Normalize the data
norm_data = results.copy()
# Map special values to indices: -1.0 -> 0 (red), -2.0 -> 1 (blue)
# Valid values map to indices >= 2
violation_mask = results < 0
valid_mask = results > 0

# Normalize valid values to range [2, 256]
if len(valid_values) > 0:
    norm_data[valid_mask] = 2 + (results[valid_mask] - vmin_valid) / (vmax_valid - vmin_valid) * (cmap.N - 1)

# Map violations
norm_data[results == -1.0] = 0  # Red
norm_data[results == -2.0] = 2  # Blue


# Create contour plot
levels = np.linspace(0, 3+cmap.N, 256)
contour = ax.contourf(E_LASER_mesh, TAU_FWHM_FS_mesh, norm_data, 
                      levels=levels, cmap=custom_cmap, vmin=0, vmax=2+cmap.N)

# Add contour lines for a_0 values (only in valid region)
if len(valid_values) > 0:
    # Create a masked array for contour lines (hide invalid regions)
    a0_masked = np.ma.masked_where(a0_values <= 0, a0_values)
    # Define contour levels for a_0
    a0_contour_levels = np.linspace(1.0, 3.0, 11)
    contour_lines = ax.contour(E_LASER_mesh, TAU_FWHM_FS_mesh, a0_masked,
                               levels=a0_contour_levels, colors='white', 
                               linewidths=0.8, alpha=0.6)
    # Add labels to contour lines
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

# Add colorbar for valid region
cbar = plt.colorbar(contour, ax=ax, label='Plasma Density $n_p$ (x$10^{18} $ cm$^{-3}$)')
# Adjust colorbar ticks to show actual values
if len(valid_values) > 0:
    cbar_ticks = np.linspace(2, 2+cmap.N, 5)
    cbar_labels = np.linspace(vmin_valid, vmax_valid, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{val:.2f}' for val in cbar_labels])

# Add the two points
ax.plot(POINT_1[0], POINT_1[1], 'ko', markersize=8, label=POINT_1_LABEL)
ax.plot(POINT_2[0], POINT_2[1], 'wo', markersize=8, label=POINT_2_LABEL)

# Add dashed black arrow from POINT_1 to POINT_2
ax.annotate('', xy=POINT_2, xytext=POINT_1,
            arrowprops=dict(arrowstyle='->', color='black', linestyle='--', linewidth=2))

# Add legend for violation regions and points
red_patch = mpatches.Patch(color='red', label='Dephasing Length too Small')
blue_patch = mpatches.Patch(color='blue', label=f'Spot Size too small (< {MIN_SPOT_SIZE*1e6:0.1f} um)')
point1_patch = mpatches.Patch(color='black', label=POINT_1_LABEL)
point2_patch = mpatches.Patch(color='white', label=POINT_2_LABEL)
ax.legend(handles=[red_patch, blue_patch, point1_patch, point2_patch], loc='upper right')

ax.set_xlabel('Laser Energy (J)', fontsize=12)
ax.set_ylabel('Pulse Duration FWHM Intensity (fs)', fontsize=12)
ax.set_title(f'HOFI Plasma Density with $a_0$ Contours (E_beam={E_BEAM} GeV)', fontsize=14)

plt.tight_layout()
plt.savefig('hofi_parameter_space.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved as 'hofi_parameter_space.png'")
print(f"Valid parameter space: {np.sum(valid_mask)/results.size*100:.1f}% of explored region")

