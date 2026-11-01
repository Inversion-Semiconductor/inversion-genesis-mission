# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check functional J_chk and its analytic state gradient (plan Section 6.2).

    J_chk(s) = sum_s sum_i w_i T(z_i - z_win) (a_s . xi_i + (b_s . xi_i)^2)
             + sum_m |P (.) E_m|^2

xi is (x, y, z - z_win, ux, uy, uz) divided by the fixed scales
`CHECK_SCALES`, which are outside `config_sha256` and are recorded with every
result. a_s and b_s are fixed random 6-vectors per species (seed 0, species
in sorted name order). P is a fixed positive random spectral mask (seed 0),
band-limited to mode 0 and |k_z| < k0 / 3, applied to the spectral Ep, Em,
and Ez. T is a quintic smoothstep in zeta = z - z_win: 0 for
zeta <= 10 dz, 1 for zeta >= 20 dz. z_win is `comm._zmin_global_domain`.

The functional does not depend on particle order and needs no target. Each
term (the field term and each species) is returned separately, with its
gradient in the layout of `driver.contract`.
"""

from __future__ import annotations

import cupy
import numpy as np
from scipy.constants import e

CHECK_SCALES = {"x": 5e-6, "y": 5e-6, "zeta": 5e-6, "ux": 1.0, "uy": 1.0, "uz": 10.0}
XI_NAMES = ("x", "y", "zeta", "ux", "uy", "uz")
FIELD_TERM_NAMES = ("Ep", "Em", "Ez")
TAPER_START, TAPER_END = 10.0, 20.0  # in cells


def taper(zeta, dz):
    """T(zeta) and dT/dzeta: quintic smoothstep from 10 dz to 20 dz."""
    width = (TAPER_END - TAPER_START) * dz
    t = cupy.clip((zeta - TAPER_START * dz) / width, 0.0, 1.0)
    value = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
    slope = 30.0 * t**2 * (1.0 - t) ** 2 / width
    return value, slope


def particle_term(arrays, z_win, dz, a, b, scales=CHECK_SCALES):
    """Value and gradient of sum_i w_i T(zeta_i) (a . xi_i + (b . xi_i)^2).

    Args:
        arrays: x, y, z, ux, uy, uz, w (CuPy, length N).
        z_win: Left edge of the physical domain [m].
        dz: Cell length [m].
        a, b: Coefficient 6-vectors in the order of `XI_NAMES`.

    Returns:
        (J, gradient) with the gradient by component name (x, y, z, ux, uy,
        uz, w), each of length N.
    """
    n = int(arrays["x"].shape[0])
    if n == 0:
        zero = cupy.zeros(0)
        return 0.0, {name: zero for name in ("x", "y", "z", "ux", "uy", "uz", "w")}
    zeta = arrays["z"] - z_win
    raw = {
        "x": arrays["x"],
        "y": arrays["y"],
        "zeta": zeta,
        "ux": arrays["ux"],
        "uy": arrays["uy"],
        "uz": arrays["uz"],
    }
    xi = [raw[name] / scales[name] for name in XI_NAMES]
    linear = sum(float(a[j]) * xi[j] for j in range(6))
    quadratic = sum(float(b[j]) * xi[j] for j in range(6))
    f = linear + quadratic**2
    T, dT = taper(zeta, dz)
    w = arrays["w"]
    J = float(cupy.sum(w * T * f))
    wT = w * T
    grad = {}
    for j, name in enumerate(XI_NAMES):
        g = wT * (float(a[j]) + 2.0 * quadratic * float(b[j])) / scales[name]
        grad["z" if name == "zeta" else name] = g
    grad["z"] = grad["z"] + w * dT * f
    grad["w"] = T * f
    return J, grad


def field_term(spect_mode0, mask):
    """Value and paired-real gradient of sum |P E|^2 over Ep, Em, Ez of mode 0."""
    J = 0.0
    grad = {}
    p2 = mask * mask
    for name in FIELD_TERM_NAMES:
        value = spect_mode0[name]
        J += float(cupy.sum(p2 * (value.real**2 + value.imag**2)))
        grad[name] = 2.0 * p2 * value
    return J, grad


class CheckFunctional:
    """J_chk for one simulation layout; `__call__(state)` returns the terms.

    Args:
        state: `StepState` (fixes the species names and the spectral grid).
        lambda0: Laser wavelength [m]; the mask keeps |k_z| < k0 / 3.
        seed: Seed of a_s, b_s, and P.
        field_weight: Multiplier of the field term (1 in the plan).
    """

    def __init__(self, state, lambda0: float, seed: int = 0, field_weight=1.0):
        rng = np.random.default_rng(seed)
        self.coefficients = {}
        for name in sorted(state.species):
            self.coefficients[name] = (rng.normal(size=6), rng.normal(size=6))
        spect0 = state.sim.fld.spect[0]
        kz = np.asarray(spect0.kz)
        k0 = 2 * np.pi / lambda0
        mask = rng.uniform(0.0, 1.0, size=kz.shape) * (np.abs(kz) < k0 / 3)
        self.mask = cupy.asarray(mask * np.sqrt(field_weight))
        self.lambda0 = lambda0
        self.seed = seed

    def record(self) -> dict:
        """The fixed choices to store with every result."""
        return {
            "scales": CHECK_SCALES,
            "taper_cells": (TAPER_START, TAPER_END),
            "seed": self.seed,
            "lambda0": self.lambda0,
            "mask_band": "mode 0, |kz| < k0/3",
            "coefficients": {
                n: (a.tolist(), b.tolist()) for n, (a, b) in self.coefficients.items()
            },
        }

    def __call__(self, state) -> dict:
        """Terms {"field": (J, g), species name: (J, g)} in the `contract` layout."""
        sim = state.sim
        z_win = float(sim.comm._zmin_global_domain)
        dz = float(sim.comm.dz)
        Jf, gf = field_term(
            {name: getattr(sim.fld.spect[0], name) for name in FIELD_TERM_NAMES},
            self.mask,
        )
        terms = {"field": (Jf, {"fields": [gf]})}
        for name, sp in state.species.items():
            n = int(sp.Ntot)
            arrays = {
                c: getattr(sp, c)[:n] for c in ("x", "y", "z", "ux", "uy", "uz", "w")
            }
            a, b = self.coefficients[name]
            J, g = particle_term(arrays, z_win, dz, a, b)
            terms[name] = (J, {"particles": {name: g}})
        return terms


# Residual objective (plan Section 2.4, Phase P7)
# -----------------------------------------------

PHASE_SPACE = ("x", "ux", "y", "uy", "z", "uz")  # xi_4 = z - z_win
LOSS_MEANS = (0, 1, 4, 5)
N_RESIDUALS = 11


def _smoothstep5(t):
    """S5(t) and dS5/dt on [0, 1], clamped outside."""
    t = cupy.clip(t, 0.0, 1.0)
    value = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    slope = 30.0 * t * t * (1.0 - t) ** 2
    return value, slope


def observables(base, Q_f_over_e, floor_sigma, floor_mean_uz):
    """The 11 observables v and dv/dbase from the 13 weighted sums (host).

    base = (S0, A_0..A_5, B_0..B_5) with S0 = sum w sigma, A_k = sum w sigma
    xi_k, B_k = sum w sigma xi_k^2. With D = S0 + Q_f / e, m_k = A_k / D,
    M_k = B_k / D, and s_k = sqrt(max(M_k - m_k^2, 0)):

        v = (log(e D), m_0, m_1, m_4, m_5, log(s_k + f_k) for k < 5,
             log((s_5 + f_5) / (m_5 + f_mu))).

    log(e D) = log(Q + Q_f). Where s_k = 0 (an empty selection), ds_k is set
    to 0: every base tangent is then 0 as well.

    Returns:
        (v (11,), dv/dbase (11, 13)).
    """
    base = np.asarray(base, float)
    S0, A, B = base[0], base[1:7], base[7:13]
    D = S0 + Q_f_over_e
    m, M = A / D, B / D
    dm = np.zeros((6, 13))
    dM = np.zeros((6, 13))
    for k in range(6):
        dm[k, 0], dm[k, 1 + k] = -m[k] / D, 1.0 / D
        dM[k, 0], dM[k, 7 + k] = -M[k] / D, 1.0 / D
    var = M - m * m
    s = np.sqrt(np.maximum(var, 0.0))
    ds = np.zeros((6, 13))
    for k in range(6):
        if s[k] > 0:
            ds[k] = (dM[k] - 2.0 * m[k] * dm[k]) / (2.0 * s[k])
    f = np.asarray(floor_sigma, float)
    v = np.empty(N_RESIDUALS)
    dv = np.zeros((N_RESIDUALS, 13))
    v[0] = np.log(D) + np.log(e)
    dv[0, 0] = 1.0 / D
    for j, k in enumerate(LOSS_MEANS):
        v[1 + j], dv[1 + j] = m[k], dm[k]
    for k in range(5):
        v[5 + k] = np.log(s[k] + f[k])
        dv[5 + k] = ds[k] / (s[k] + f[k])
    v[10] = np.log((s[5] + f[5]) / (m[5] + floor_mean_uz))
    dv[10] = ds[5] / (s[5] + f[5]) - dm[5] / (m[5] + floor_mean_uz)
    return v, dv


class ResidualValue:
    """Whitened residuals r_w at a state and their contraction with tangents.

    Attributes:
        residuals: r_w, shape (11,), host.
        base: The 13 weighted sums, host.
        dr_dbase: L^-1 dv/dbase, shape (11, 13), host.
    """

    def __init__(self, residuals, base, dr_dbase, arrays, sigma_slope, species):
        self.residuals = residuals
        self.base = base
        self.dr_dbase = dr_dbase
        self._arrays = arrays
        self._sigma_slope = sigma_slope
        self._species = species

    def base_gradients(self) -> list[dict]:
        """d base_b / d(particle arrays), in the `driver.contract` layout.

        One dict per base sum; used by the L1 tests. Memory: 13 x 7 arrays.
        """
        a = self._arrays
        sigma, w = a["sigma"], a["w"]
        ws = w * sigma
        dws_duz = w * self._sigma_slope
        zero = cupy.zeros_like(w)
        out = []

        def entry(g_w, g_uz, g_xi=None, k=None):
            grad = {c: zero for c in ("x", "y", "z", "ux", "uy", "uz")}
            grad = dict(grad)
            grad["w"] = g_w
            grad["uz"] = g_uz
            if g_xi is not None:
                name = PHASE_SPACE[k]
                grad[name] = grad[name] + g_xi
            return {"particles": {self._species: grad}}

        out.append(entry(sigma, dws_duz))
        for k in range(6):
            xi = a["xi"][k]
            out.append(entry(sigma * xi, dws_duz * xi, ws, k))
        for k in range(6):
            xi = a["xi"][k]
            out.append(entry(sigma * xi * xi, dws_duz * xi * xi, 2.0 * ws * xi, k))
        return out

    def contract(self, tangents) -> np.ndarray:
        """dr_w/dtheta for each tangent: shape (11, K)."""
        a = self._arrays
        t = tangents.particles[self._species]
        sigma, w, xi = a["sigma"], a["w"], a["xi"]
        ws = w * sigma
        K = tangents.K
        base_dot = np.zeros((13, K))
        for k in range(K):
            dws = sigma * t["w"][k] + w * self._sigma_slope * t["uz"][k]
            base_dot[0, k] = float(cupy.sum(dws))
            for j, name in enumerate(PHASE_SPACE):
                dxi = t[name][k]
                base_dot[1 + j, k] = float(cupy.sum(dws * xi[j] + ws * dxi))
                base_dot[7 + j, k] = float(
                    cupy.sum(dws * xi[j] * xi[j] + 2.0 * ws * xi[j] * dxi)
                )
        return self.dr_dbase @ base_dot


class ResidualObjective:
    """The 11 whitened residuals of plan Section 2.4 with alpha_D = 0 (D29).

    r_w = L^-1 (v(s) - v_y) on the selected `elec_from_N` beam, with the
    quintic smoothstep selection on u_z (Section 2.3). The target enters
    through its charge, its raw moments, and its charge floor Q_f.

    Args:
        species: Name of the observed species.
        u_c, delta_u: Selection center and width.
        target_Q: Target charge Q_y [C]; target_moments: its 12 raw moments
            (6 first, 6 second, in the order of `PHASE_SPACE`).
        Q_f: Charge floor [C].
        L: Lower Cholesky factor of Sigma (11 x 11).
        floor_sigma: The 6 spread floors f_k; floor_mean_uz: f_mu.
    """

    def __init__(
        self,
        species,
        u_c,
        delta_u,
        target_Q,
        target_moments,
        Q_f,
        L,
        floor_sigma,
        floor_mean_uz,
    ):
        self.species = species
        self.u_c, self.delta_u = float(u_c), float(delta_u)
        self.Q_f_over_e = float(Q_f) / e
        self.L = np.asarray(L, float)
        self.floor_sigma = np.asarray(floor_sigma, float)
        self.floor_mean_uz = float(floor_mean_uz)
        moments = np.asarray(target_moments, float)
        D_y = float(target_Q) / e + self.Q_f_over_e
        base_y = np.concatenate([[float(target_Q) / e], moments[:6] * D_y])
        base_y = np.concatenate([base_y, moments[6:] * D_y])
        self.v_target, _ = observables(
            base_y, self.Q_f_over_e, self.floor_sigma, self.floor_mean_uz
        )

    def evaluate(self, arrays, z_win) -> ResidualValue:
        """Residuals of particle arrays x, y, z, ux, uy, uz, w (CuPy)."""
        from scipy.linalg import solve_triangular

        width = 4.0 * self.delta_u
        sigma, slope = _smoothstep5((arrays["uz"] - self.u_c + 0.5 * width) / width)
        sigma_slope = slope / width
        xi = [arrays[c] for c in PHASE_SPACE]
        xi[4] = arrays["z"] - z_win
        ws = arrays["w"] * sigma
        base = np.empty(13)
        base[0] = float(cupy.sum(ws))
        for k in range(6):
            base[1 + k] = float(cupy.sum(ws * xi[k]))
            base[7 + k] = float(cupy.sum(ws * xi[k] * xi[k]))
        v, dv = observables(base, self.Q_f_over_e, self.floor_sigma, self.floor_mean_uz)
        r = solve_triangular(self.L, v - self.v_target, lower=True)
        dr = solve_triangular(self.L, dv, lower=True)
        kept = {"sigma": sigma, "w": arrays["w"], "xi": xi}
        return ResidualValue(r, base, dr, kept, sigma_slope, self.species)

    def __call__(self, state) -> ResidualValue:
        sp = state.species[self.species]
        n = int(sp.Ntot)
        arrays = {c: getattr(sp, c)[:n] for c in ("x", "y", "z", "ux", "uy", "uz", "w")}
        return self.evaluate(arrays, float(state.sim.comm._zmin_global_domain))
