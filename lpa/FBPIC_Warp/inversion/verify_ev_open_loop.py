"""Phase P3 step 2: open-loop parity of EV ionization with production.

Freeze the gathered fields and the momenta of a set of N5+ ions for 200 steps.
Each ion has its own rest-frame field, chosen so that p5 and p6 each cover
about 1e-3 to 0.3 over the set. Run the EV kernel once, and the production
ionizer with 32 Philox seeds. Pass if sum_i w_i P_l (l = 5, 6, 7) and the
cumulative birth weight each satisfy |EV - mean| <= 3 SE.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python verify_ev_open_loop.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cupy
import numpy as np
from fbpic.autodiff import ionization as ev
from fbpic.particles import Particles
from scipy.constants import c, e, m_e, m_p
from scipy.optimize import brentq
from twin_config import load_config

N_STEPS = 200
N_SEEDS = 32
N_PER_FIELD = 40  # ions per prescribed field value (Npz)
N_FIELDS = 60


def _x(E, coeffs, k):
    a, q, cexp = (
        coeffs[n][k] for n in ("adk_prefactor", "adk_power", "adk_exp_prefactor")
    )
    return a * E**q * np.exp(cexp / E)


def field_values(dt):
    """Geometric set of fields from p5 = 1e-3 to p6 = 0.3."""
    coeffs = ev.adk_coefficients("N", dt)

    def p(E, k):
        return -np.expm1(-_x(E, coeffs, k))

    e_lo = brentq(lambda E: p(E, 0) - 1e-3, 1e10, 1e14)
    e_hi = brentq(lambda E: p(E, 1) - 0.3, 1e10, 1e15)
    fields = np.geomspace(e_lo, e_hi, N_FIELDS)
    return fields, coeffs, p(fields, 0), p(fields, 1)


def make_ions(dt, fields, rng_seed):
    """Production ions with one prescribed field per group, at rest."""
    count = N_PER_FIELD * N_FIELDS
    ions = Particles(
        e,
        14 * m_p,
        1e24,
        count,
        0,
        10e-6,
        1,
        0,
        2e-6,
        1,
        dt,
        continuous_injection=False,
        use_cuda=True,
        grid_shape=(16, 4),
    )
    electrons = Particles(
        -e,
        m_e,
        1e24,
        0,
        0,
        10e-6,
        1,
        0,
        2e-6,
        1,
        dt,
        continuous_injection=False,
        use_cuda=True,
        grid_shape=(16, 4),
    )
    ions.track(SimpleNamespace(rank=0, size=1))
    electrons.track(SimpleNamespace(rank=0, size=1))
    ions.make_ionizable("N", electrons, level_start=5, rng_seed=rng_seed, rng_stream=1)
    ions.send_particles_to_gpu()
    electrons.send_particles_to_gpu()
    n = int(ions.Ntot)
    assert n == count
    per_ion = cupy.asarray(np.repeat(fields, N_PER_FIELD))
    for name in ("ux", "uy", "uz", "Ey", "Ez", "Bx", "By", "Bz"):
        getattr(ions, name)[:n] = 0.0
    ions.Ex[:n] = per_ion
    ions.inv_gamma[:n] = 1.0
    return ions, electrons


def stochastic(dt, fields, seed):
    ions, electrons = make_ions(dt, fields, seed)
    for it in range(N_STEPS):
        ions.ionizer.handle_ionization(ions, iteration=it)
    n = int(ions.Ntot)
    level = cupy.asnumpy(ions.ionizer.ionization_level[:n]).astype(int)
    w = cupy.asnumpy(ions.w[:n])
    births = float(cupy.asnumpy(electrons.w[: int(electrons.Ntot)]).sum())
    return [float(w[level == lev].sum()) for lev in (5, 6, 7)] + [births]


def expected(dt, fields, epsilon):
    ions, _ = make_ions(dt, fields, 1)
    n = int(ions.Ntot)
    state = SimpleNamespace(
        species={"nitrogen": ions},
        ev_species="nitrogen",
        ev=ev.EVArrays(cupy.ones(n), cupy.zeros(n), cupy.zeros(n), cupy.zeros(n)),
    )
    coeffs = ev.adk_coefficients("N", dt)
    a, q, cexp = (
        coeffs["adk_prefactor"],
        coeffs["adk_power"],
        coeffs["adk_exp_prefactor"],
    )
    emitted = 0.0
    max_pending_ratio = 0.0
    no_mask = cupy.zeros(1, np.int32)
    for _ in range(N_STEPS):
        outs = [cupy.empty(n) for _ in range(5)]
        emit = cupy.empty(n, np.int32)
        evs = state.ev
        ev._launch(
            ev.ev_ionize,
            n,
            [
                ions.ux[:n],
                ions.uy[:n],
                ions.uz[:n],
                ions.Ex[:n],
                ions.Ey[:n],
                ions.Ez[:n],
                ions.Bx[:n],
                ions.By[:n],
                ions.Bz[:n],
                ions.w[:n],
                evs.pop5,
                evs.pop6,
                evs.pop7,
                evs.w_pending,
                float(a[0]),
                float(q[0]),
                float(cexp[0]),
                float(a[1]),
                float(q[1]),
                float(cexp[1]),
                epsilon,
                0,
                no_mask,
                *outs[:4],
                outs[4],
                emit,
            ],
        )
        state.ev = ev.EVArrays(*outs[:4])
        emitted += float(outs[4].sum())
    w = ions.w[:n]
    evs = state.ev
    max_pending_ratio = float((evs.w_pending / w).max())
    total = cupy.asnumpy(evs.pop5 + evs.pop6 + evs.pop7)
    return {
        "sums": [float((w * p).sum()) for p in (evs.pop5, evs.pop6, evs.pop7)]
        + [emitted + float(evs.w_pending.sum())],
        "emitted": emitted,
        "max_pending_over_w": max_pending_ratio,
        "max_population_sum_error": float(np.abs(total - 1).max()),
    }


def main():
    config = load_config("twin_s")
    grid = config["grid"]
    dt = grid["dt_cells"] * (grid["zmax"] - grid["zmin"]) / grid["Nz"] / c
    fields, _, p5, p6 = field_values(dt)
    runs = np.array([stochastic(dt, fields, seed) for seed in range(1, N_SEEDS + 1)])
    mean = runs.mean(axis=0)
    se = runs.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
    result = expected(dt, fields, config["ev"]["epsilon"])
    names = ["sum w P5", "sum w P6", "sum w P7", "birth weight"]
    z = (np.array(result["sums"]) - mean) / se
    report = {
        "dt": dt,
        "fields_V_per_m": [float(fields[0]), float(fields[-1])],
        "p5_range": [float(p5.min()), float(p5.max())],
        "p6_range": [float(p6.min()), float(p6.max())],
        "n_ions": N_PER_FIELD * N_FIELDS,
        "rows": {
            n: {"ev": float(v), "mean": float(m), "se": float(s), "z": float(zz)}
            for n, v, m, s, zz in zip(names, result["sums"], mean, se, z)
        },
        "ev_emitted": result["emitted"],
        "max_pending_over_w": result["max_pending_over_w"],
        "max_population_sum_error": result["max_population_sum_error"],
        "pass": bool(np.all(np.abs(z) <= 3.0)),
    }
    print(json.dumps(report, indent=2))
    assert report["pass"], "EV and stochastic marginals differ by more than 3 SE"
    assert result["max_pending_over_w"] <= config["ev"]["epsilon"] * (1 + 1e-12)
    print("PASS")


if __name__ == "__main__":
    main()
