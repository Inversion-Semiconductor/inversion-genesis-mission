"""
Simple demonstration script to model the free space propagation of an electron beam
given its emittance, spot size, and energy.  Assumes a Gaussian beam characterized by
Twiss parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

# Beam parameters at focus (alpha = 0, gamma = 1/beta)
EMIT_X_NORM = 5e-6  # m-rad
GAMMA_LORENTZ = 2e2
SIGMA_INITIAL = 40e-6 # m, rms

beta_i = GAMMA_LORENTZ * np.square(SIGMA_INITIAL) / EMIT_X_NORM

x=np.linspace(-0.1, 0.1, 100)
beta_x = beta_i + np.square(x)/beta_i
sig_x = np.sqrt(beta_x * EMIT_X_NORM / GAMMA_LORENTZ)

plt.plot(x, sig_x*1e6, label='RMS Spot Size')
plt.title(f"Gamma = {GAMMA_LORENTZ}, Emit = {EMIT_X_NORM}")
plt.xlabel("Z (m)")
plt.ylabel("Sigma (um)")
plt.legend()
plt.show()
