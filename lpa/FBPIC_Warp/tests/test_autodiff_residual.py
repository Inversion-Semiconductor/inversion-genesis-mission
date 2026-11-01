# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L1: the residual objective of plan Section 2.4 (Phase P7 step 1).

`ResidualObjective` (CuPy, with the tangent contraction) is checked against
the independent host implementation of Phase P3 (`inversion/observable.py`:
`beam_statistics`, `whitened_residuals`):

1. The 11 whitened residuals agree to rounding.
2. `check_jvp` (vector form, pair rule): the contraction of a random
   particle tangent agrees with central FD of the host residuals, on a
   normal beam, an empty selection (every sigma = 0), and a nearly empty
   selection (one particle with sigma = 1e-6).
3. The explicit base gradients agree with the contraction.

The whitening factor L is a random lower-triangular matrix, so that the
off-diagonal path is tested; the twin uses a diagonal L.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e

cupy = pytest.importorskip("cupy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inversion"))

from observable import beam_statistics, whitened_residuals  # noqa: E402

from fbpic.autodiff.driver import contract  # noqa: E402
from fbpic.autodiff.loss import ResidualObjective  # noqa: E402
from fbpic.autodiff.verification import check_jvp  # noqa: E402

DEVICE = os.environ.get("FBPIC_AUTODIFF_DEVICE", "cuda:0")
COMPONENTS = ("x", "y", "z", "ux", "uy", "uz", "w")
U_C, DELTA_U = 40.0, 4.0
Z_WIN = 1.0e-4
SCALES = {
    "x": 1e-6,
    "y": 1e-6,
    "z": 1e-6,
    "ux": 1.0,
    "uy": 1.0,
    "uz": 10.0,
    "w": 1e5,
}


def _beam(rng, n, uz_low=20.0, uz_high=80.0):
    return {
        "x": rng.normal(scale=2e-6, size=n),
        "y": rng.normal(scale=2e-6, size=n),
        "z": Z_WIN + 1e-5 + rng.normal(scale=1e-6, size=n),
        "ux": rng.normal(scale=2.0, size=n),
        "uy": rng.normal(scale=2.0, size=n),
        "uz": rng.uniform(uz_low, uz_high, size=n),
        "w": 1e5 * (1.0 + rng.uniform(size=n)),
    }


def _target_and_whitening(seed=1):
    rng = np.random.default_rng(seed)
    beam = _beam(rng, 3000)
    stats0 = beam_statistics(beam, Z_WIN, U_C, DELTA_U, 0.0)
    Q_f = 1e-3 * stats0["Q"]
    stats = beam_statistics(beam, Z_WIN, U_C, DELTA_U, Q_f)
    target = {"Q": stats["Q"], "Q_f": Q_f, "moments": stats["moments"]}
    m = stats["moments"]
    sigma = np.sqrt(np.maximum(m[6:] - m[:6] ** 2, 0.0))
    L = np.tril(rng.normal(scale=0.1, size=(11, 11)), -1)
    L += np.diag(rng.uniform(0.5, 2.0, size=11))
    whitening = {
        "L": L,
        "floor_sigma": 1e-3 * sigma,
        "floor_mean_uz": 1e-3 * abs(m[5]),
    }
    return target, whitening


def _objective(target, whitening):
    return ResidualObjective(
        "beam",
        U_C,
        DELTA_U,
        target["Q"],
        target["moments"],
        target["Q_f"],
        whitening["L"],
        whitening["floor_sigma"],
        whitening["floor_mean_uz"],
    )


def _reference(arrays, target, whitening):
    stats = beam_statistics(arrays, Z_WIN, U_C, DELTA_U, target["Q_f"])
    return whitened_residuals(stats, target, whitening)


def _contract(objective, host, directions):
    value = objective.evaluate({c: cupy.asarray(host[c]) for c in COMPONENTS}, Z_WIN)
    tangents = SimpleNamespace(
        K=1,
        particles={
            "beam": {
                c: cupy.asarray(np.asarray(directions.get(c, 0 * host[c]))[None, :])
                for c in COMPONENTS
            }
        },
    )
    return value, tangents


def _check(arrays, target, whitening, seed):
    objective = _objective(target, whitening)

    def forward(values):
        host = {c: values[c].numpy() for c in COMPONENTS}
        return {"r": _reference(host, target, whitening)}

    def tangent(host, directions):
        value, tangents = _contract(objective, host, directions)
        return {"r": value.contract(tangents)[:, 0]}

    return check_jvp(
        forward,
        tangent,
        arrays,
        COMPONENTS,
        device=DEVICE,
        scales=SCALES,
        seed=seed,
    )


def test_residual_values():
    target, whitening = _target_and_whitening()
    arrays = _beam(np.random.default_rng(2), 2000)
    value = _objective(target, whitening).evaluate(
        {c: cupy.asarray(arrays[c]) for c in COMPONENTS}, Z_WIN
    )
    reference = _reference(arrays, target, whitening)
    error = np.max(np.abs(value.residuals - reference)) / np.max(np.abs(reference))
    print(f"residuals {np.round(value.residuals, 3)}; relative difference {error:.1e}")
    assert error <= 1e-11


def test_residual_at_target_is_zero():
    target, whitening = _target_and_whitening()
    beam = _beam(np.random.default_rng(1), 3000)
    value = _objective(target, whitening).evaluate(
        {c: cupy.asarray(beam[c]) for c in COMPONENTS}, Z_WIN
    )
    assert np.max(np.abs(value.residuals)) <= 1e-9


def test_residual_jvp():
    target, whitening = _target_and_whitening()
    arrays = _beam(np.random.default_rng(3), 2000)
    report = _check(arrays, target, whitening, seed=5)
    print(f"residual JVP: best errors {report['best_error']}")


def test_residual_empty_selection():
    target, whitening = _target_and_whitening()
    arrays = _beam(np.random.default_rng(4), 50, uz_low=1.0, uz_high=U_C - 3 * DELTA_U)
    objective = _objective(target, whitening)
    value, tangents = _contract(
        objective,
        arrays,
        {c: np.random.default_rng(6).normal(size=50) * SCALES[c] for c in COMPONENTS},
    )
    assert np.isfinite(value.residuals).all()
    np.testing.assert_allclose(
        value.residuals, _reference(arrays, target, whitening), rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(value.contract(tangents), 0.0)
    report = _check(arrays, target, whitening, seed=7)
    print(f"empty selection: best errors {report['best_error']}")


def test_residual_nearly_empty_selection():
    target, whitening = _target_and_whitening()
    arrays = _beam(np.random.default_rng(8), 50, uz_low=1.0, uz_high=U_C - 3 * DELTA_U)
    # S5(t) = 1e-6 at t = 0.0464...; uz = u_c - 2 delta_u + 4 delta_u t
    from scipy.optimize import brentq

    t = brentq(lambda t: t**3 * (t * (6 * t - 15) + 10) - 1e-6, 0.0, 0.5)
    arrays["uz"][0] = U_C - 2 * DELTA_U + 4 * DELTA_U * t
    value = _objective(target, whitening).evaluate(
        {c: cupy.asarray(arrays[c]) for c in COMPONENTS}, Z_WIN
    )
    assert np.isfinite(value.residuals).all()
    report = _check(arrays, target, whitening, seed=9)
    print(f"nearly empty selection: best errors {report['best_error']}")


def test_base_gradients_match_contraction():
    target, whitening = _target_and_whitening()
    arrays = _beam(np.random.default_rng(10), 500)
    rng = np.random.default_rng(11)
    directions = {c: rng.normal(size=500) * SCALES[c] for c in COMPONENTS}
    value, tangents = _contract(_objective(target, whitening), arrays, directions)
    base_dot = np.array(
        [contract(g, tangents, None)[0] for g in value.base_gradients()]
    )
    expected = value.dr_dbase @ base_dot
    got = value.contract(tangents)[:, 0]
    error = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    print(f"base gradients against contraction: {error:.1e}")
    assert error <= 1e-12
    Q = e * value.base[0]
    assert Q > 0
