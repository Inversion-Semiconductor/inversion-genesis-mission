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
