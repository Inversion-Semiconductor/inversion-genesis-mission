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

# Input parameters
E_LASER = 1.0  # J, energy budget of laser
WAVELENGTH = 800e-9  # m, laser wavelength
BUFFER = 1.3
PLASMA_DENSITY = 2.0  # e18 cm-3
TAU_FWHM = 30.77e-15  # s

SPOT_SIZE_FACTOR = 1

# Constants
pa = 8.7 * 1e9  # W, constant
b1 = 96  # GeV/m, constant
b2 = 33e-6 / (2*pi)  # m, constant
b3 = 50e-6 / np.sqrt(5)  # m, constant

# Laser Wavenumber
k0 = 2 * pi / WAVELENGTH

# Calculations for all other parameters
tau = TAU_FWHM / 1.17741
w_0 = b3/np.sqrt(PLASMA_DENSITY) * SPOT_SIZE_FACTOR
p_0 = E_LASER/tau * 2/np.sqrt(2*pi)
a_0 = (16/pa * p_0 / (k0**2 * w_0**2))**(1/2)
l_depl = c * tau * k0**2 * b2**2 / PLASMA_DENSITY
l_deph = 2/3 * k0**2 * b2 **2 * w_0 / PLASMA_DENSITY
l_diff = w_0**2 * k0
l_acc = l_depl * BUFFER
e_0 = b1*np.sqrt(PLASMA_DENSITY)
acceleration_gradient = e_0 * np.sqrt(a_0) / 2
final_energy = acceleration_gradient * l_acc

k_p = np.sqrt(PLASMA_DENSITY /  b2**2)
l_p = 5.31e5/np.sqrt(PLASMA_DENSITY*1e18)*1e-2
i_a = 17 * 1e3  # kA, Alfven current constant
q_b = i_a * a_0**(3/2) * b2 / (2* c * PLASMA_DENSITY**(1/2))

print(f"Plasma Density:      {PLASMA_DENSITY*10:.2f} x10^17 cm^-3")
print(f"Plasma Wavelength:   {2*pi/k_p*1e6:.2f} um")
print(f"Plasma Wavelength:   {l_p*1e6:.2f} um")
print(f"Tau                  {tau*1e15:.2f} fs")
print(f"Tau FWHM:            {TAU_FWHM*1e15:.2f} fs")
print(F"Laser Power:         {p_0*1e-12:.2f} TW")
print(f"Laser waist size:     {w_0*1e6:.2f} um")
print(f"Laser a0:            {a_0:.2f}")
print(f"Plasma Length:       {l_acc*1e2:.2f} cm")
print(f"Electron Beam Energy: {final_energy:.2f} GeV")
print(f"Depletion length:    {l_depl*1e2:.2f} cm")
print(f"Dephasing length:    {l_deph*1e2:.2f} cm")
print(f"Diffraction length:  {l_diff*1e2:.2f} cm")
print(f"Optimal Charge:      {q_b*1e12:.2f} pC")
print()

print("Resonant Condition: c*tau < lambda_p")
print(f" {c*tau:.3e} < {33e-6/np.sqrt(PLASMA_DENSITY):.3e}  -{c*tau < 33e-6/np.sqrt(PLASMA_DENSITY)}-")
print("Dephasing Condition: l_acc < l_deph")
print(f" {l_acc:.3e} < {l_deph:.3e}  -{l_acc < l_deph}-")
print()

k_p = PLASMA_DENSITY**0.5 / b2
dr = 0.15/k_p
print(f"Recommended Maximum dr: {dr*1e6} um")
z_foc = 2.0e-3 # m
zr = pi * np.square(w_0) / WAVELENGTH
w_start = w_0*np.sqrt(1 + (z_foc/zr)**2)
minimum_r_range = 2.65*w_start * 1.7
print(f"Recommended Minimum R_range: {minimum_r_range*1e6} um")
minimum_r_cells = minimum_r_range/dr
print(f"Corresponding R Grid Size: {round(minimum_r_cells)}")
