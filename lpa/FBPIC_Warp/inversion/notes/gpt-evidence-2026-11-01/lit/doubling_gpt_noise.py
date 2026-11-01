"""Doubling map x -> 2x mod 1: residual of GPT projection with a Fourier basis.

Tangent xi_i = 2^n for every particle (exact Lagrangian tangent).
Projection: min_c sum_i w |xi_i - sum_k c_k psi_k'(x_i)|^2, psi_k = cos/sin(2 pi k x), k<=K.
Prints rms(P xi) / 2^n and compares with N^{-1/2}. Exact continuum answer: P xi = 0.
"""
import numpy as np

rng = np.random.default_rng(0)
n = 20
xi0 = 2.0**n
for N in (10**3, 10**4, 10**5):
    for K in (4, 16):
        for sampling in ("random", "lattice+chaos"):
            if sampling == "random":
                x = rng.random(N)
            else:
                # quiet start, then 30 doubling steps in float64 with a tiny jitter (mimics mixing)
                x = (np.arange(N) + 0.5) / N
                x = (x + 1e-12 * rng.standard_normal(N)) % 1.0
                for _ in range(30):
                    x = (2 * x) % 1.0
            k = np.arange(1, K + 1)
            G = np.concatenate([-2*np.pi*k*np.sin(2*np.pi*np.outer(x, k)),
                                 2*np.pi*k*np.cos(2*np.pi*np.outer(x, k))], axis=1)  # grad psi at x_i
            w = np.full(N, 1.0 / N)
            xi = np.full(N, xi0)
            A = G * np.sqrt(w)[:, None]
            c, *_ = np.linalg.lstsq(A, xi * np.sqrt(w), rcond=None)
            Pxi = G @ c
            r = np.sqrt(np.sum(w * Pxi**2)) / xi0
            print(f"N={N:6d} K={K:2d} {sampling:14s} rms(P xi)/2^n = {r:.3e}   N^-1/2 = {N**-0.5:.3e}")
