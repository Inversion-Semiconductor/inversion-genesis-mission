"""Twin-S laser and bunch timeline and P7 growth rates (inputs: twin_s.json, plan 12.10)."""
import numpy as np
c = 299792458.0
dz = 40e-6 / 800                     # m, = c dt (dt_cells = 1)
z0, zc, F, zlo, zhi = -10e-6, 38e-6, 28e-6, 10e-6, 66e-6
S = lambda z: np.exp(-np.log(2) * 4 * (z - zc) ** 2 / F**2) * ((z >= zlo) & (z <= zhi))
wp0 = 5.64e4 * np.sqrt(8e24 * 1e-6)  # rad/s
print("omega_p0 = %.3e /s, omega_p0*dt = %.4f" % (wp0, wp0 * dz / c))
for name, z in (("laser at z_lo", zlo), ("laser at peak", zc), ("laser at z_hi", zhi)):
    print(f"{name:24s} step {(z - z0) / dz:7.0f}")
print("bunch 12 um behind laser leaves plasma at step %.0f" % ((zhi + 12e-6 - z0) / dz))
for n in (1200, 1500, 1800):
    zl = z0 + n * dz
    zb = zl - 12e-6
    print(f"step {n}: laser z={zl*1e6:5.1f} um S={S(zl):.2f}; bunch z={zb*1e6:5.1f} um S={S(zb):.2f}, omega_p,loc={np.sqrt(S(zb)):.2f} omega_p0")
steps = np.array([1200, 1500, 1800, 2000]); ux = np.array([2.3e6, 1.5e8, 8.7e10, 1.7e11])
for a, b, ua, ub in zip(steps[:-1], steps[1:], ux[:-1], ux[1:]):
    r = np.log(ub / ua) / (b - a)
    print(f"steps {a}-{b}: rate {r:.4f}/step = {r / (wp0 * dz / c):.2f} omega_p0")
