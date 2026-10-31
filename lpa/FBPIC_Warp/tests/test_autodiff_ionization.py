# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L1: tangents of the expected-value (EV) ionization (Phase P6 step 1).

Inversion plan Sections 5.3 and 6.1. Each tangent kernel passes:

1. A dot product with the Warp-generated VJP of the primal kernel
   (`ev_ionize`, `push_ion_momenta_ev`) for three random draws, e <= 1e-12,
   with the emission flags fixed (`use_mask = 1`).
2. `check_jvp` (vector form, pair rule) against central differences of the
   primal kernel.

The ions at E = 0 are tested separately: the VJP is NaN there (adjoint of
sqrt at 0). The emission copy is tested as an exact index gather.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from scipy.constants import c, m_p

cupy = pytest.importorskip("cupy")

from fbpic.autodiff import ionization as ev  # noqa: E402
from fbpic.autodiff import tangent as tg  # noqa: E402
from fbpic.autodiff.particles import push_ion_momenta_ev  # noqa: E402
from fbpic.autodiff.state import TANGENT_PARTICLE_NAMES  # noqa: E402
from fbpic.autodiff.verification import check_jvp, dot_product_error  # noqa: E402
from fbpic.utils.warp import get_runtime  # noqa: E402

DEVICE = os.environ.get("FBPIC_AUTODIFF_DEVICE", "cuda:0")
DOT_TOLERANCE = 1e-12
DT = 1.6678204759907605e-16  # Twin-S time step
MOMENTA = ("ux", "uy", "uz")
GATHERED = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
POPULATIONS = ("pop5", "pop6", "pop7", "w_pending")
EV_OUTPUTS = ("pop5_out", "pop6_out", "pop7_out", "w_pending_out", "birth_w")
EV_INPUTS = MOMENTA + GATHERED + POPULATIONS
ION_MASS = 14 * m_p


@pytest.fixture(scope="module", autouse=True)
def runtime():
    wp.init()
    wp.set_device(DEVICE)
    get_runtime().use()


def _sync():
    get_runtime().synchronize()
    wp.synchronize_device(DEVICE)


def _wp(value, requires_grad=False):
    value = np.asarray(value)
    dtype = wp.int32 if value.dtype == np.int32 else wp.float64
    return wp.array(value, dtype=dtype, device=DEVICE, requires_grad=requires_grad)


def _coefficients():
    a = ev.adk_coefficients("N", DT)
    return [
        float(v)
        for level in (0, 1)
        for v in (
            a["adk_prefactor"][level],
            a["adk_power"][level],
            a["adk_exp_prefactor"][level],
        )
    ]


def _ev_fixture(zero_field=False, seed=5):
    """Ions with rest-frame fields from the underflow band to saturation.

    |E| spans 1e11 V/m (x = 0 by underflow below about 1.19e11, E > 0) to 1.8e13 V/m (x5 about
    30, P5 nearly emptied in one step). Half of the ions emit.
    """
    rng = np.random.default_rng(seed)
    n = 40
    magnitude = np.exp(rng.uniform(np.log(4e12), np.log(1.8e13), n))
    magnitude[:3] = (1e11, 1.1e11, 2e12)
    direction = rng.normal(size=(3, n))
    direction /= np.linalg.norm(direction, axis=0)
    arrays = {}
    for k, name in enumerate(("Ex", "Ey", "Ez")):
        arrays[name] = magnitude * direction[k]
    for name in ("Bx", "By", "Bz"):
        arrays[name] = rng.normal(size=n) * 0.3 * magnitude / c
    for name in MOMENTA:
        arrays[name] = rng.normal(scale=0.01, size=n)
    pops = rng.dirichlet((1.0, 1.0, 1.0), size=n).T
    arrays["pop5"], arrays["pop6"], arrays["pop7"] = pops
    arrays["w"] = 1e6 * (1.0 + rng.uniform(size=n))
    arrays["w_pending"] = arrays["w"] * rng.uniform(0, 1e-4, size=n)
    arrays["emit_mask"] = (rng.uniform(size=n) < 0.5).astype(np.int32)
    if zero_field:
        for name in GATHERED:
            arrays[name][:6] = 0.0
    return arrays


EV_SCALES = {
    **{name: 0.01 for name in MOMENTA},
    **{name: 1e13 for name in ("Ex", "Ey", "Ez")},
    **{name: 1e13 / c for name in ("Bx", "By", "Bz")},
    "pop5": 1.0,
    "pop6": 1.0,
    "pop7": 1.0,
    "w_pending": 100.0,
}
EV_OUT_SCALES = {
    "pop5_out": 1.0,
    "pop6_out": 1.0,
    "pop7_out": 1.0,
    "w_pending_out": 1e6,
    "birth_w": 1e6,
}


def _launch_ev_ionize(values):
    """Primal `ev_ionize` with the emission flags fixed; returns the outputs."""
    count = values["ux"].shape[0]
    outputs = {
        name: wp.zeros(count, dtype=wp.float64, device=DEVICE, requires_grad=True)
        for name in EV_OUTPUTS
    }
    emit = wp.zeros(count, dtype=wp.int32, device=DEVICE)
    wp.launch(
        ev.ev_ionize,
        dim=count,
        inputs=[values[name] for name in MOMENTA + GATHERED]
        + [values["w"]]
        + [values[name] for name in POPULATIONS]
        + _coefficients()
        + [1e-4, 1, values["emit_mask"]],
        outputs=[outputs[name] for name in EV_OUTPUTS[:4]] + [outputs["birth_w"], emit],
        device=DEVICE,
    )
    return outputs


def _ev_tangent(arrays, directions_list):
    """Hand tangent `ev_tangent` with K = len(directions_list)."""
    K = len(directions_list)
    n = arrays["ux"].size

    def stacked(name):
        return cupy.asarray(
            np.stack([d.get(name, np.zeros(n)) for d in directions_list])
        )

    outs = [cupy.empty((K, n)) for _ in EV_OUTPUTS]
    ev._launch(
        ev.ev_tangent,
        n,
        [cupy.asarray(arrays[name]) for name in MOMENTA + GATHERED]
        + [cupy.asarray(arrays[name]) for name in ("w", "pop5", "pop6")]
        + _coefficients()
        + [cupy.asarray(arrays["emit_mask"])]
        + [stacked(name) for name in MOMENTA + GATHERED + POPULATIONS]
        + outs,
    )
    _sync()
    host = [cupy.asnumpy(o) for o in outs]
    return [{name: host[i][k] for i, name in enumerate(EV_OUTPUTS)} for k in range(K)]


def _vjp(arrays, names, launch, cotangent):
    values = {n: _wp(v, requires_grad=n in names) for n, v in arrays.items()}
    tape = wp.Tape()
    with tape:
        outputs = launch(values)
    tape.backward(
        grads={
            out: wp.array(cotangent[n], dtype=wp.float64, device=DEVICE)
            for n, out in outputs.items()
        }
    )
    wp.synchronize_device(DEVICE)
    return {n: values[n].grad.numpy() for n in names}


def _dot_products(arrays, names, scales, out_scales, tangent_fn, launch, seed):
    rng = np.random.default_rng(seed)
    directions = [
        {n: scales[n] * rng.normal(size=arrays[n].shape) for n in names}
        for _ in range(3)
    ]
    tangents = tangent_fn(directions)
    errors = []
    for direction, tangent in zip(directions, tangents):
        cotangent = {
            n: rng.normal(size=t.shape) / out_scales[n] for n, t in tangent.items()
        }
        adjoint = _vjp(arrays, names, launch, cotangent)
        errors.append(
            dot_product_error(
                tangent, cotangent, direction, adjoint, scales, out_scales
            )
        )
    return errors


def test_ev_fixture_coverage():
    """The fixture spans the underflow band, the ADK range, and saturation."""
    arrays = _ev_fixture()
    a = ev.adk_coefficients("N", DT)
    gamma = np.sqrt(1 + sum(arrays[n] ** 2 for n in MOMENTA))
    u_dot_E = sum(arrays[u] * arrays[f] for u, f in zip(MOMENTA, ("Ex", "Ey", "Ez")))
    cB = [c * arrays[n] for n in ("Bx", "By", "Bz")]
    ux, uy, uz = (arrays[n] for n in MOMENTA)
    Ex, Ey, Ez = (arrays[n] for n in ("Ex", "Ey", "Ez"))
    E = np.sqrt(
        -(u_dot_E**2)
        + (gamma * Ex + uy * cB[2] - uz * cB[1]) ** 2
        + (gamma * Ey + uz * cB[0] - ux * cB[2]) ** 2
        + (gamma * Ez + ux * cB[1] - uy * cB[0]) ** 2
    )
    x5 = (
        a["adk_prefactor"][0]
        * E ** a["adk_power"][0]
        * np.exp(a["adk_exp_prefactor"][0] / E)
        / gamma
    )
    print(f"x5 range {x5.min():.1e} to {x5.max():.1e}; zeros {np.sum(x5 == 0)}")
    assert np.sum(x5 == 0) >= 2
    assert x5.max() > 10
    assert np.sum((x5 > 1e-3) & (x5 < 1)) >= 5


@pytest.mark.parametrize("group", ["momenta", "fields", "populations", "all"])
def test_ev_tangent_dot_product(group):
    arrays = _ev_fixture()
    names = {
        "momenta": MOMENTA,
        "fields": GATHERED,
        "populations": POPULATIONS,
        "all": EV_INPUTS,
    }[group]
    errors = _dot_products(
        arrays,
        names,
        EV_SCALES,
        EV_OUT_SCALES,
        lambda ds: _ev_tangent(arrays, ds),
        _launch_ev_ionize,
        seed=17,
    )
    print(f"ev {group}: dot-product errors {errors}")
    assert max(errors) <= DOT_TOLERANCE, errors


def test_ev_tangent_check_jvp():
    arrays = _ev_fixture()
    report = check_jvp(
        _launch_ev_ionize,
        lambda host, d: _ev_tangent(host, [d])[0],
        arrays,
        EV_INPUTS,
        device=DEVICE,
        scales=EV_SCALES,
        seed=23,
    )
    print(f"ev: best errors {report['best_error']}")


def test_ev_tangent_zero_field():
    """E = 0 ions: finite tangents, dp = 0, so dP' = dP and dW' = dW (or dW
    goes to the birth weight when the ion emits)."""
    arrays = _ev_fixture(zero_field=True)
    rng = np.random.default_rng(3)
    direction = {n: EV_SCALES[n] * rng.normal(size=arrays[n].shape) for n in EV_INPUTS}
    out = _ev_tangent(arrays, [direction])[0]
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    zero = slice(0, 6)
    emit = arrays["emit_mask"][zero] != 0
    np.testing.assert_array_equal(out["pop5_out"][zero], direction["pop5"][zero])
    np.testing.assert_array_equal(out["pop6_out"][zero], direction["pop6"][zero])
    np.testing.assert_array_equal(out["pop7_out"][zero], direction["pop7"][zero])
    pending = direction["w_pending"][zero]
    np.testing.assert_array_equal(
        out["w_pending_out"][zero], np.where(emit, 0, pending)
    )
    np.testing.assert_array_equal(out["birth_w"][zero], np.where(emit, pending, 0))


# EV ion push
# -----------

ION_MOMENTA = ("ux", "uy", "uz", "inv_gamma")


def _ion_push_fixture():
    rng = np.random.default_rng(41)
    n = 9
    arrays = {name: rng.normal(scale=0.01, size=n) for name in MOMENTA}
    arrays["inv_gamma"] = 1 / np.sqrt(1 + sum(arrays[m] ** 2 for m in MOMENTA))
    for name in GATHERED:
        scale = 1e12 if name.startswith("E") else 3e3
        arrays[name] = rng.normal(size=n) * scale
    arrays["w"] = 1e6 * (1 + rng.uniform(size=n))
    arrays["w_tl"] = arrays["w"] * rng.uniform(5.0, 7.0, size=n)
    return arrays


ION_SCALES = {
    **{name: 1.0 for name in ION_MOMENTA},
    **{g: 1e12 if g.startswith("E") else 3e3 for g in GATHERED},
    "w_tl": 1e6,
}
ION_INPUTS = ION_MOMENTA + GATHERED + ("w_tl",)


def _launch_ion_push(values):
    count = values["ux"].shape[0]
    outputs = {
        name: wp.zeros(count, dtype=wp.float64, device=DEVICE, requires_grad=True)
        for name in ION_MOMENTA
    }
    wp.launch(
        push_ion_momenta_ev,
        dim=count,
        inputs=[values[name] for name in ION_MOMENTA + GATHERED]
        + [values["w"], values["w_tl"], ION_MASS, DT],
        outputs=list(outputs.values()),
        device=DEVICE,
    )
    return outputs


def _ion_push_tangent(arrays, directions_list):
    K = len(directions_list)
    n = arrays["ux"].size

    def stacked(name):
        return cupy.asarray(
            np.stack([d.get(name, np.zeros(n)) for d in directions_list])
        )

    tangents = [stacked(name) for name in ION_MOMENTA]
    tg.vay_ion_tangent(
        [cupy.asarray(arrays[name]) for name in ION_MOMENTA],
        [cupy.asarray(arrays[name]) for name in GATHERED],
        cupy.asarray(arrays["w"]),
        cupy.asarray(arrays["w_tl"]),
        ION_MASS,
        DT,
        tangents,
        [stacked(name) for name in GATHERED],
        stacked("w_tl"),
    )
    _sync()
    host = [cupy.asnumpy(t) for t in tangents]
    return [{name: host[i][k] for i, name in enumerate(ION_MOMENTA)} for k in range(K)]


@pytest.mark.parametrize("group", ["momenta", "fields", "charge", "all"])
def test_ion_push_tangent_dot_product(group):
    arrays = _ion_push_fixture()
    names = {
        "momenta": ION_MOMENTA,
        "fields": GATHERED,
        "charge": ("w_tl",),
        "all": ION_INPUTS,
    }[group]
    errors = _dot_products(
        arrays,
        names,
        ION_SCALES,
        {name: 1.0 for name in ION_MOMENTA},
        lambda ds: _ion_push_tangent(arrays, ds),
        _launch_ion_push,
        seed=43,
    )
    print(f"ion push {group}: dot-product errors {errors}")
    assert max(errors) <= DOT_TOLERANCE, errors


def test_ion_push_tangent_check_jvp():
    arrays = _ion_push_fixture()
    report = check_jvp(
        _launch_ion_push,
        lambda host, d: _ion_push_tangent(host, [d])[0],
        arrays,
        ION_INPUTS,
        device=DEVICE,
        scales=ION_SCALES,
        seed=47,
    )
    print(f"ion push: best errors {report['best_error']}")


# Emission copy
# -------------


def test_emission_copy_is_index_gather():
    """New electron tangents equal the parent tangents, in slot order, and
    the weight tangent equals dW' before the reset."""
    arrays = _ev_fixture()
    n = arrays["ux"].size
    K = 2
    rng = np.random.default_rng(11)
    ion = SimpleNamespace(
        Ntot=n,
        **{k: cupy.asarray(arrays[k]) for k in MOMENTA + GATHERED + ("w",)},
    )
    ev_old = SimpleNamespace(
        pop5=cupy.asarray(arrays["pop5"]), pop6=cupy.asarray(arrays["pop6"])
    )
    emit = cupy.asarray(arrays["emit_mask"])
    t_ion = {
        comp: cupy.asarray(rng.normal(size=(K, n)))
        for comp in TANGENT_PARTICLE_NAMES + ev.EV_TANGENT_NAMES
    }
    n_elec = 5
    t_elec = {
        comp: cupy.asarray(rng.normal(size=(K, n_elec)))
        for comp in TANGENT_PARTICLE_NAMES
    }
    before_ion = {k: v.copy() for k, v in t_ion.items()}
    before_elec = {k: v.copy() for k, v in t_elec.items()}
    a = ev.adk_coefficients("N", DT)
    state = SimpleNamespace(
        species={"nitrogen": ion},
        ev_species="nitrogen",
        ev_target="elec_from_N",
        ev_params=dict(a),
        ev_inputs={"ev": ev_old, "emit": emit},
    )
    tangents = SimpleNamespace(
        K=K, particles={"nitrogen": t_ion, "elec_from_N": t_elec}
    )
    gathered = [cupy.asarray(rng.normal(size=(K, n)) * EV_SCALES[g]) for g in GATHERED]
    ev.ionize_tangent(state, tangents, gathered)
    _sync()
    parents = np.flatnonzero(arrays["emit_mask"])
    reference = _ev_tangent(
        arrays,
        [
            {
                **{m: cupy.asnumpy(before_ion[m][k]) for m in MOMENTA},
                **{g: cupy.asnumpy(gathered[i][k]) for i, g in enumerate(GATHERED)},
                **{p: cupy.asnumpy(before_ion[p][k]) for p in POPULATIONS},
            }
            for k in range(K)
        ],
    )
    for comp in TANGENT_PARTICLE_NAMES:
        new = cupy.asnumpy(t_elec[comp])
        assert new.shape == (K, n_elec + parents.size)
        np.testing.assert_array_equal(new[:, :n_elec], cupy.asnumpy(before_elec[comp]))
        if comp == "w":
            expected = np.stack([reference[k]["birth_w"][parents] for k in range(K)])
        else:
            expected = cupy.asnumpy(before_ion[comp])[:, parents]
        np.testing.assert_array_equal(new[:, n_elec:], expected)
    for name, out in zip(ev.EV_TANGENT_NAMES, EV_OUTPUTS[:4]):
        expected = np.stack([reference[k][out] for k in range(K)])
        np.testing.assert_array_equal(cupy.asnumpy(t_ion[name]), expected)
