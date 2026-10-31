# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L2: parity of the functional stepper with production `Simulation.step`.

Inversion plan Section 6.1. Twin-XS and Twin-S with pre-ionized nitrogen:
`begin` plus 300 `step` calls against production `sim.step(300)` in one call,
from the same initial state. Fields within max(2e-11, 10 d_AA) relative in the
infinity norm, where d_AA is the difference of two production runs. Particle
counts and IDs equal; particle arrays matched by ID within the same bound.

d_AA is the largest difference over the pairs of 3 production runs. One pair
underestimates the run-to-run spread: on Twin-XS, one electron bifurcates
between production runs, and two production runs can differ by 8e-4 in the
infinity norm of uy after 300 steps (plan Section 12, Phase P2).
The 1000-step Twin-S case covers the laser-plasma interaction, which starts
after about 400 steps.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

INVERSION = Path(__file__).resolve().parents[1] / "inversion"
sys.path.insert(0, str(INVERSION))

N_STEPS = 300
SPECT_NAMES = ("Ep", "Em", "Ez", "Bp", "Bm", "Bz", "rho_prev")
INTERP_NAMES = ("Er", "Et", "Ez", "Br", "Bt", "Bz")
PARTICLE_NAMES = ("x", "y", "z", "ux", "uy", "uz", "w", "inv_gamma")


def _snapshot(twin):
    sim = twin.sim
    fields = {}
    for m in range(len(sim.fld.interp)):
        for name in SPECT_NAMES:
            fields[f"spect{m}.{name}"] = cupy.asnumpy(getattr(sim.fld.spect[m], name))
        for name in INTERP_NAMES:
            fields[f"interp{m}.{name}"] = cupy.asnumpy(getattr(sim.fld.interp[m], name))
    particles = {}
    for key, sp in twin.species.items():
        n = int(sp.Ntot)
        ids = cupy.asnumpy(sp.tracker.id[:n])
        order = np.argsort(ids)
        particles[key] = {"id": ids[order]}
        for name in PARTICLE_NAMES:
            particles[key][name] = cupy.asnumpy(getattr(sp, name)[:n])[order]
    return {
        "fields": fields,
        "particles": particles,
        "iteration": int(sim.iteration),
        "zmin": float(sim.comm._zmin_global_domain),
    }


def _relative(a, b):
    if b.size == 0:
        return 0.0
    scale = float(np.abs(b).max())
    if scale == 0.0:
        return float(np.abs(a).max())
    return float(np.abs(a - b).max()) / scale


def _differences(a, b):
    out = {k: _relative(a["fields"][k], b["fields"][k]) for k in a["fields"]}
    for key, pa in a["particles"].items():
        pb = b["particles"][key]
        np.testing.assert_array_equal(pa["id"], pb["id"], err_msg=key)
        for name in PARTICLE_NAMES:
            out[f"{key}.{name}"] = _relative(pa[name], pb[name])
    return out


def _production(size, n_steps):
    from twin_config import build_twin, load_config

    config = load_config(size)
    twin = build_twin(config["theta"]["nominal"], config, ionization="none")
    twin.sim.step(n_steps, show_progress=False)
    return _snapshot(twin)


def _stepper(size, n_steps):
    from fbpic.autodiff.stepper import begin, step
    from twin_config import build_twin, load_config

    config = load_config(size)
    twin = build_twin(config["theta"]["nominal"], config, ionization="none")
    state = begin(twin.sim, twin.species)
    for _ in range(n_steps):
        step(state)
    cupy.cuda.runtime.deviceSynchronize()
    return _snapshot(twin)


@pytest.mark.parametrize(
    "size, n_steps", [("twin_xs", N_STEPS), ("twin_s", N_STEPS), ("twin_s", 1000)]
)
def test_stepper_matches_production(size, n_steps):
    prods = [_production(size, n_steps) for _ in range(3)]
    mine = _stepper(size, n_steps)
    assert mine["iteration"] == prods[0]["iteration"] == n_steps
    assert mine["zmin"] == prods[0]["zmin"]
    pairs = [_differences(a, b) for i, a in enumerate(prods) for b in prods[i + 1 :]]
    d_aa = {key: max(p[key] for p in pairs) for key in pairs[0]}
    d_sa = _differences(mine, prods[0])
    worst = []
    for key, value in d_sa.items():
        bound = max(2e-11, 10 * d_aa[key])
        worst.append((value / bound, key, value, d_aa[key]))
    worst.sort(reverse=True)
    print(f"\n{size}, {n_steps} steps: largest ratios (diff / bound, diff, d_AA):")
    for ratio, key, value, aa in worst[:6]:
        print(f"  {key:24s} {ratio:9.3e} {value:9.3e} {aa:9.3e}")
    failures = [w for w in worst if w[0] > 1.0]
    assert not failures, failures[:5]
