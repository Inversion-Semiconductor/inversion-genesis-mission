# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L1 of the check functional J_chk (plan Section 6.2, Phase P5 step 5).

The analytic gradient of each term is compared with central differences
along 3 random directions, with the epsilon list and the pair rule of
`check_vjp` (tolerance 1e-6; 1e-10 for the quadratic field term, whose
central difference is exact up to rounding).
"""

from itertools import pairwise

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from fbpic.autodiff.loss import field_term, particle_term  # noqa: E402

COMPONENTS = ("x", "y", "z", "ux", "uy", "uz", "w")
DZ = 2e-7
Z_WIN = 1e-6
EPSILONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def _particles(rng, n=200):
    # zeta spans the taper (10 dz to 20 dz) and both flat parts.
    return {
        "x": cupy.asarray(rng.normal(scale=3e-6, size=n)),
        "y": cupy.asarray(rng.normal(scale=3e-6, size=n)),
        "z": cupy.asarray(Z_WIN + rng.uniform(5 * DZ, 30 * DZ, size=n)),
        "ux": cupy.asarray(rng.normal(scale=2.0, size=n)),
        "uy": cupy.asarray(rng.normal(scale=0.5, size=n)),
        "uz": cupy.asarray(rng.normal(loc=5.0, scale=5.0, size=n)),
        "w": cupy.asarray(rng.uniform(1e5, 1e6, size=n)),
    }


def _pair_rule(analytical, evaluate, tolerance):
    errors = []
    for eps in EPSILONS:
        numerical = (evaluate(eps) - evaluate(-eps)) / (2 * eps)
        scale = max(abs(analytical), abs(numerical), 1e-300)
        errors.append(abs(analytical - numerical) / scale)
    assert any(a <= tolerance and b <= tolerance for a, b in pairwise(errors)), errors
    return min(errors)


@pytest.mark.parametrize("block", ["positions", "momenta", "weights", "all"])
def test_particle_term_gradient(block):
    rng = np.random.default_rng(5)
    arrays = _particles(rng)
    a, b = rng.normal(size=6), rng.normal(size=6)
    _, grad = particle_term(arrays, Z_WIN, DZ, a, b)
    names = {
        "positions": ("x", "y", "z"),
        "momenta": ("ux", "uy", "uz"),
        "weights": ("w",),
        "all": COMPONENTS,
    }[block]
    scales = {"x": 3e-6, "y": 3e-6, "z": 3e-6, "ux": 2, "uy": 0.5, "uz": 5, "w": 1e6}
    best = []
    for _ in range(3):
        d = {
            n: cupy.asarray(rng.normal(size=arrays[n].shape)) * scales[n] / 10
            for n in names
        }
        analytical = sum(float(cupy.dot(grad[n], d[n])) for n in names)

        def evaluate(eps, d=d):
            moved = {n: v + eps * d[n] if n in d else v for n, v in arrays.items()}
            return particle_term(moved, Z_WIN, DZ, a, b)[0]

        best.append(_pair_rule(analytical, evaluate, 1e-6))
    print(f"particle term {block}: best errors {best}")


def test_field_term_gradient():
    rng = np.random.default_rng(6)
    shape = (16, 8)
    fields = {
        n: cupy.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
        for n in ("Ep", "Em", "Ez")
    }
    mask = cupy.asarray(rng.uniform(size=shape) * (rng.uniform(size=shape) > 0.3))
    _, grad = field_term(fields, mask)
    best = []
    for _ in range(3):
        d = {
            n: cupy.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
            for n in fields
        }
        analytical = sum(float(cupy.real(cupy.vdot(grad[n], d[n]))) for n in fields)

        def evaluate(eps, d=d):
            return field_term({n: v + eps * d[n] for n, v in fields.items()}, mask)[0]

        best.append(_pair_rule(analytical, evaluate, 1e-10))
    print(f"field term: best errors {best}")


def test_empty_species():
    empty = {n: cupy.zeros(0) for n in COMPONENTS}
    J, grad = particle_term(empty, Z_WIN, DZ, np.ones(6), np.ones(6))
    assert J == 0.0 and all(g.size == 0 for g in grad.values())
