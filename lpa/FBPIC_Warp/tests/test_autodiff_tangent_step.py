# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L2t: the primal result does not depend on the tangents (plan Section 6.1).

(a) Bitwise, in one run: a hook snapshots every primal array before each
    tangent-only section of `step` (`tangent_*:before`) and requires bitwise
    equality after it (`tangent_*:after`). Tangent operations run as
    separate launches after each primal operation, so this check is
    deterministic and sees a stash or swap leak of any size.
(b) Smoke test: `step` with and without tangents, 300 steps, primal arrays
    within max(2e-11, 10 d_AA), with d_AA from 3 production runs as in L2.
    Counts and IDs equal.

Twin-XS and Twin-S, E_L tangent (K = 1). (a) runs with `ionization="none"`
and with the EV model (Phase P6; the snapshot then includes the EV arrays
of the ions). (b) runs with `ionization="none"`.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

TESTS = Path(__file__).resolve().parent
INVERSION = TESTS.parent / "inversion"
sys.path.insert(0, str(INVERSION))
N_STEPS = 300
SPECIES_ARRAYS = (
    "x",
    "y",
    "z",
    "ux",
    "uy",
    "uz",
    "w",
    "inv_gamma",
    "Ex",
    "Ey",
    "Ez",
    "Bx",
    "By",
    "Bz",
    "prefix_sum",
)


def _l2_module():
    spec = importlib.util.spec_from_file_location(
        "l2_stepper", TESTS / "test_autodiff_stepper.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primal_snapshot(state):
    from fbpic.autodiff.field_tangents import INTERP_NAMES, SPECT_NAMES

    sim = state.sim
    arrays = {}
    for m, grid in enumerate(sim.fld.spect):
        for name in SPECT_NAMES:
            arrays[f"spect{m}.{name}"] = getattr(grid, name).copy()
    for m, grid in enumerate(sim.fld.interp):
        for name in INTERP_NAMES:
            arrays[f"interp{m}.{name}"] = getattr(grid, name).copy()
    scalars = {
        "time": sim.time,
        "iteration": sim.iteration,
        "zmin": sim.comm._zmin_global_domain,
        "interp_zmin": sim.fld.interp[0].zmin,
    }
    for name, sp in state.species.items():
        n = int(sp.Ntot)
        for attr in SPECIES_ARRAYS:
            value = getattr(sp, attr)
            arrays[f"{name}.{attr}"] = (
                value.copy() if attr == "prefix_sum" else value[:n].copy()
            )
        arrays[f"{name}.id"] = sp.tracker.id[:n].copy()
        scalars[f"{name}.Ntot"] = n
        scalars[f"{name}.sorted"] = sp.sorted
        scalars[f"{name}.prefix_sum_shift"] = sp.prefix_sum_shift
    if state.ev is not None:
        for attr, value in zip(
            ("pop5", "pop6", "pop7", "w_pending"), state.ev.arrays()
        ):
            arrays[f"ev.{attr}"] = value.copy()
    return arrays, scalars


class BitwiseGuard:
    """Hook that checks bitwise primal equality around each tangent section."""

    def __init__(self):
        self.before = None
        self.sections = {}

    def __call__(self, state, name):
        if not name.startswith("tangent_"):
            return
        section, when = name.split(":")
        if when == "before":
            self.before = (section, _primal_snapshot(state))
            return
        label, (arrays, scalars) = self.before
        assert label == section, (label, section)
        after_arrays, after_scalars = _primal_snapshot(state)
        assert after_scalars == scalars, (section, scalars, after_scalars)
        for key, value in arrays.items():
            other = after_arrays[key]
            assert value.shape == other.shape and bool(
                cupy.array_equal(value, other)
            ), (section, key)
        self.sections[section] = self.sections.get(section, 0) + 1
        self.before = None


def _begin_kwargs(config, ionization):
    if ionization == "none":
        return {}
    return {
        "ionization": "ev",
        "ev_species": "nitrogen",
        "ev_target": "elec_from_N",
        "ev_options": {"epsilon": config["ev"]["epsilon"], "K_e": config["ev"]["K_e"]},
    }


@pytest.mark.parametrize("ionization", ["none", "ev"])
@pytest.mark.parametrize("size", ["twin_xs", "twin_s"])
def test_l2t_bitwise_in_run(size, ionization):
    from builders import twin_builder

    from fbpic.autodiff.driver import resolve_seeds, seed_tangents
    from fbpic.autodiff.stepper import begin, step

    build = twin_builder(size, ionization=ionization)
    theta = np.array(build.config["theta"]["nominal"])
    sim, species, profile = build(theta)
    seeds = resolve_seeds(sim, profile, theta, ("E_L",))
    state = begin(sim, species, **_begin_kwargs(build.config, ionization))
    tangents = seed_tangents(state, seeds)
    guard = BitwiseGuard()
    state.hooks = {"*": guard}
    for _ in range(N_STEPS):
        step(state, tangents)
    print(f"\n{size} {ionization}: bitwise-checked sections {guard.sections}")
    assert guard.sections["tangent_gather"] == N_STEPS
    if ionization == "ev":
        assert guard.sections["tangent_ev"] == N_STEPS
    assert guard.sections["tangent_realign"] == N_STEPS // sim.comm.exchange_period


def _stepper_snapshot(size, with_tangents):
    from builders import twin_builder

    from fbpic.autodiff.driver import resolve_seeds, seed_tangents
    from fbpic.autodiff.stepper import begin, step

    l2 = _l2_module()
    build = twin_builder(size)
    theta = np.array(build.config["theta"]["nominal"])
    sim, species, profile = build(theta)
    seeds = resolve_seeds(sim, profile, theta, ("E_L",))
    state = begin(sim, species)
    tangents = seed_tangents(state, seeds) if with_tangents else None
    for _ in range(N_STEPS):
        step(state, tangents)
    cupy.cuda.runtime.deviceSynchronize()
    twin = type("TwinView", (), {"sim": sim, "species": species})
    return l2._snapshot(twin)


@pytest.mark.parametrize("size", ["twin_xs", "twin_s"])
def test_l2t_smoke(size):
    l2 = _l2_module()
    prods = [l2._production(size, N_STEPS) for _ in range(3)]
    plain = _stepper_snapshot(size, False)
    tangent = _stepper_snapshot(size, True)
    assert tangent["iteration"] == plain["iteration"] == N_STEPS
    assert tangent["zmin"] == plain["zmin"]
    pairs = [l2._differences(a, b) for i, a in enumerate(prods) for b in prods[i + 1 :]]
    d_aa = {key: max(p[key] for p in pairs) for key in pairs[0]}
    d_tp = l2._differences(tangent, plain)
    worst = sorted(
        ((value / max(2e-11, 10 * d_aa[key]), key, value, d_aa[key])
         for key, value in d_tp.items()),
        reverse=True,
    )  # fmt: skip
    print(f"\n{size}: largest ratios (diff / bound, diff, d_AA):")
    for ratio, key, value, aa in worst[:6]:
        print(f"  {key:24s} {ratio:9.3e} {value:9.3e} {aa:9.3e}")
    failures = [w for w in worst if w[0] > 1.0]
    assert not failures, failures[:5]
