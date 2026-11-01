"""Moment (Gaussian) score projection of a momentum kick, 1D-1V, one bin.

Lagrangian: each marker gets du = xi (uniform kick). Weight form:
dW_p = W_p * xi * (u_p - ubar) / C  (C = sample variance of u in the bin).
Compare the change of sum(W), sum(W u), sum(W v(u)) with v = u / sqrt(1 + u^2),
and sum(W (u - ubar)^2), against the exact (Lagrangian) linear response.
"""
import numpy as np

rng = np.random.default_rng(2)
N, xi = 400_000, 1.0
v = lambda u: u / np.sqrt(1 + u * u)
dv = lambda u: (1 + u * u) ** -1.5
cases = {
    "gaussian u0=2 s=0.5": rng.normal(2, 0.5, N),
    "bimodal +-2, s=0.3": np.where(rng.random(N) < 0.5, -2, 2) + rng.normal(0, 0.3, N),
    "skewed (lognormal)": rng.lognormal(0.5, 0.6, N),
    "near-cold u0=4 s=1e-3": rng.normal(4, 1e-3, N),
}
print(f"{'case':24s} {'dSumW':>9s} {'dJ_u rel':>9s} {'dJ_v rel':>9s} {'d var rel':>9s} {'rms dW/W':>9s}")
for name, u in cases.items():
    W = np.ones(N) / N
    ub = np.sum(W * u)
    C = np.sum(W * (u - ub) ** 2)
    dW = W * xi * (u - ub) / C
    ex_Ju, ex_Jv = xi * np.sum(W), xi * np.sum(W * dv(u))
    ex_var = 0.0  # a uniform kick does not change the variance
    print(
        f"{name:24s} {np.sum(dW):+9.1e} {np.sum(dW*u)/ex_Ju:9.4f} {np.sum(dW*v(u))/ex_Jv:9.4f}"
        f" {np.sum(dW*(u-ub)**2):+9.2e} {np.sqrt(np.mean((dW/W)**2)):9.2e}"
    )
