"""Physical momentum diffusion in the Twin-S plasma versus the option-C noise levels (NRL formulary rates)."""
import numpy as np
ne_cm3 = 8e18; wp = 1.60e14; mc2_eV = 511e3
print("Te[eV] lnL  nu_ei/wp  u_th   sigma_phys[ (u)/sqrt(1/wp) ]  u_n over 53/wp  over 14.6/wp")
for Te in (5, 20, 50):
    lnL = 24 - np.log(np.sqrt(ne_cm3) / Te)
    nu = 2.91e-6 * ne_cm3 * lnL * Te ** -1.5          # s^-1, electron collision rate
    uth = np.sqrt(Te / mc2_eV)
    sig = np.sqrt(2 * (nu / wp) * uth ** 2)            # momentum diffusion ~ nu u_th^2 per component
    print("%5g %5.1f %9.2e %7.4f %12.2e %14.4f %12.4f" % (Te, lnL, nu / wp, uth, sig, sig * np.sqrt(53.2), sig * np.sqrt(14.6)))
print("option C: u_n=0.03 over 14.6 -> sigma=%.2e; wake sigma=0.01 -> u_n(70)=%.3f" % (0.03 / np.sqrt(14.6), 0.01 * np.sqrt(70)))
print("wake electrons in the wake have |u| ~ 1: collision rate falls ~ (u_th/u)^3 =", (np.sqrt(20 / mc2_eV) / 1.0) ** 3)
