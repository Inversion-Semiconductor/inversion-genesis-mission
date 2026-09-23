"""Plot LWFA scaling limits for regenerating a target-power talk figure."""

from scipy.constants import c, m_e, pi, e, epsilon_0
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from inversion_fbpic.utils.simulation_setup_tools import (
    calculate_laser_energy_from_a0,
    calculate_fwhm_intensity_from_laser_tau,
)

a0 = 2.2
wavelength = 800e-9  # m
k0 = 2 * pi / wavelength
den_arr = np.linspace(7e16 * 100**3, 1.5e18 * 100**3, 10)
kp = np.sqrt(den_arr * e**2 / (m_e * epsilon_0 * c**2))

w0 = 2 * np.sqrt(a0) / kp
tau = 2 / (c * kp)
tau_fwhm = calculate_fwhm_intensity_from_laser_tau(
    laser_tau=tau,
)

energy_diffraction = (
    m_e * c**2 * np.sqrt(a0) * np.square(kp * w0) * (k0 / kp) / e * 1e-9
)
energy_depletion = (
    m_e * c**2 * (np.sqrt(a0) / 2) * (kp * c * tau) * np.square(k0 / kp) / e * 1e-9
)

energy_arr = calculate_laser_energy_from_a0(
    a0=a0,
    wavelength=wavelength * 1e6,
    w0=w0,
    tau_fwhm=tau_fwhm,
)
power_arr = energy_arr / tau * 2 / np.sqrt(2 * pi)

power_tw = power_arr * 1e-12
target_power_tw = 200
target_energy_low_gev = 0.5
target_energy_high_gev = 1.25
energy_depletion_scaled = energy_depletion * 0.5


def interpolate_log_x_from_y(x_values, y_values, y_target):
    sort_indices = np.argsort(y_values)
    return np.exp(
        np.interp(
            np.log(y_target),
            np.log(y_values[sort_indices]),
            np.log(x_values[sort_indices]),
        )
    )


def interpolate_log_y_from_x(x_values, y_values, x_target):
    sort_indices = np.argsort(x_values)
    return np.exp(
        np.interp(
            np.log(x_target),
            np.log(x_values[sort_indices]),
            np.log(y_values[sort_indices]),
        )
    )


fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(power_tw, energy_depletion_scaled, label="Depletion-Limited")
ax.loglog(power_tw, energy_diffraction, label="Diffraction-Limited")

x_min, x_max = ax.get_xlim()

depletion_fill_energy = np.geomspace(
    target_energy_low_gev,
    target_energy_high_gev,
    200,
)
depletion_fill_power = interpolate_log_x_from_y(
    power_tw,
    energy_depletion_scaled,
    depletion_fill_energy,
)
depletion_fill_mask = depletion_fill_power <= target_power_tw
ax.fill_betweenx(
    depletion_fill_energy[depletion_fill_mask],
    depletion_fill_power[depletion_fill_mask],
    target_power_tw,
    color="lightblue",
    alpha=0.35,
    label="Depletion target region",
)

diffraction_energy_at_target_power = interpolate_log_y_from_x(
    power_tw,
    energy_diffraction,
    target_power_tw,
)
if diffraction_energy_at_target_power > target_energy_low_gev:
    diffraction_fill_energy = np.geomspace(
        target_energy_low_gev,
        diffraction_energy_at_target_power,
        200,
    )
    diffraction_fill_power = interpolate_log_x_from_y(
        power_tw,
        energy_diffraction,
        diffraction_fill_energy,
    )
    ax.fill_betweenx(
        diffraction_fill_energy,
        diffraction_fill_power,
        target_power_tw,
        color="lightcoral",
        alpha=0.35,
        label="Diffraction target region",
    )

ax.axhline(
    target_energy_low_gev,
    color="black",
    linestyle="--",
    linewidth=1,
)
ax.axhline(
    target_energy_high_gev,
    color="black",
    linestyle="--",
    linewidth=1,
)
ax.axvline(
    target_power_tw,
    color="tab:red",
    linestyle="--",
    linewidth=1,
    label="200 TW",
)

ax.set_xlim(x_min, x_max)
ax.set_xlabel("Laser power")
ax.set_ylabel("Electron energy")
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g} TW"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g} GeV"))
ax.tick_params(axis="y", which="both", right=True, labelright=True)
ax.legend()

plt.show()
