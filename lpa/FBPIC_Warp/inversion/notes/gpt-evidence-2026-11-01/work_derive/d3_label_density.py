"""D3: is the label-space density change dn = -d zeta/du bounded on the
standard-map sheet (K = 2, P0 = 0.3)? zeta = xi_t/|F| is the tangential
tangent pulled back to labels. Quantiles of |dn| on a 2^22 lattice, for steps
where max|F| du < 0.2 (resolved). Also the spread of the label-IBP Monte Carlo
estimator mean(dn psi) + mean(xi_n d_n psi) with N = 2e4 random labels, using
exact per-particle dn (interpolated from the fine lattice).
"""
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("d2", "d2_sheet_norms.py")
d2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2)
NL = d2.NL
u = (np.arange(NL) + 0.5) / NL
rng = np.random.default_rng(3)
print(" n  |dn| q50       q90       q99       rms       | lag(fine)  | MC N=2e4: LAG mean+-sd      IBP mean+-sd")
for NF in (2, 4, 6, 8, 9):
    x, p, dx, dp, fx, fp, _ = d2.run(2.0, NF, u, track=False)
    Fn = np.hypot(fx, fp)
    tx, tp = fx / Fn, fp / Fn
    xit, xin = dx * tx + dp * tp, dx * tp - dp * tx
    zeta = xit / Fn
    dn = -np.gradient(zeta, u[1] - u[0])
    gx, gp = -np.sin(x), 0.5 * np.cos(p)
    psi = np.cos(x) + 0.5 * np.sin(p)
    lag_t = gx * dx + gp * dp
    ibp_t = dn * psi + xin * (gx * tp - gp * tx)
    q = np.quantile(np.abs(dn), [0.5, 0.9, 0.99])
    L, I = [], []
    for _ in range(200):
        idx = rng.integers(0, NL, 20000)
        L.append(lag_t[idx].mean())
        I.append(ibp_t[idx].mean())
    print(f"{NF:2d} {q[0]:.2e} {q[1]:.2e} {q[2]:.2e} {np.sqrt(np.mean(dn**2)):.2e} | {lag_t.mean():+.4f} | "
          f"{np.mean(L):+.4f}+-{np.std(L):.4f}   {np.mean(I):+.4f}+-{np.std(I):.4f}")
