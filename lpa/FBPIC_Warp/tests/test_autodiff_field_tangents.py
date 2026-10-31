# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L1: linearity of the field operators that `step` applies to tangents.

Inversion plan Sections 4.6 and 6.1, Phase P4 step 5. Each production field
operation runs through `FieldTangents.apply` (the path the stepper uses) on
three tangent slots holding x, y, and x + alpha y. The check is

    |A(x + alpha y) - A x - alpha A y|_inf <= 1e-12 (|A x|_inf + |alpha| |A y|_inf)

per array and mode, for 3 random draws, at the Twin-S shape and at
2912 x 1080, modes m = 0 and 1. The vector transforms use the DHT orders
m - 1 and m + 1, the scalar transforms order m.

The k = 0 round-trip probe of plan Appendix A (row 1) runs Twin-S for 500
steps and compares the spectral E and B with interp2spect(spect2interp(.)).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from fbpic.autodiff.field_tangents import (  # noqa: E402
    INTERP_NAMES,
    SPECT_NAMES,
    FieldTangents,
)

INVERSION = Path(__file__).resolve().parents[1] / "inversion"
sys.path.insert(0, str(INVERSION))
TOLERANCE = 1e-12
SHAPES = {"twin_s": None, "2912x1080": (2912, 1080)}


def _fields_only(shape):
    from scipy.constants import c

    from fbpic.main import Simulation
    from fbpic.utils.cuda import send_data_to_gpu
    from fbpic.utils.warp import get_runtime
    from twin_config import load_config

    grid = dict(load_config("twin_s")["grid"])
    if shape is not None:
        dz = (grid["zmax"] - grid["zmin"]) / grid["Nz"]
        dr = grid["rmax"] / grid["Nr"]
        grid["Nz"], grid["Nr"] = shape
        grid["zmax"] = grid["zmin"] + dz * shape[0]
        grid["rmax"] = dr * shape[1]
    dz = (grid["zmax"] - grid["zmin"]) / grid["Nz"]
    sim = Simulation(
        grid["Nz"],
        grid["zmax"],
        grid["Nr"],
        grid["rmax"],
        grid["Nm"],
        grid["dt_cells"] * dz / c,
        zmin=grid["zmin"],
        n_order=grid["n_order"],
        n_guard=grid["n_guard"],
        exchange_period=grid["exchange_period"],
        use_cuda=True,
        boundaries={"z": "open", "r": "reflective"},
        particle_shape="linear",
        filter_currents=True,
        current_correction="curl-free",
        hankel_precision=grid["hankel_precision"],
        use_cuda_graphs=False,
        verbose_level=0,
    )
    sim.set_moving_window(v=c)
    get_runtime().use()
    send_data_to_gpu(sim)
    return sim


@pytest.fixture(scope="module", params=list(SHAPES))
def sim(request):
    return request.param, _fields_only(SHAPES[request.param])


def _operations(sim):
    fld, comm = sim.fld, sim.comm

    def shift():
        for grid in fld.spect:
            comm.moving_win.shift_spect_grid(grid, 3)

    ops = {
        "spect2interp_E": lambda: fld.spect2interp("E"),
        "spect2interp_B": lambda: fld.spect2interp("B"),
        "interp2spect_E": lambda: fld.interp2spect("E"),
        "interp2spect_B": lambda: fld.interp2spect("B"),
        "interp2spect_J": lambda: fld.interp2spect("J"),
        "interp2spect_rho_prev": lambda: fld.interp2spect("rho_prev"),
        "interp2spect_rho_next": lambda: fld.interp2spect("rho_next"),
        "spect2partial_interp_E": lambda: fld.spect2partial_interp("E"),
        "partial_interp2spect_B": lambda: fld.partial_interp2spect("B"),
        "damp_EB_open_boundary": lambda: comm.damp_EB_open_boundary(fld.interp),
        "divide_by_volume_rho": lambda: fld.divide_by_volume("rho"),
        "divide_by_volume_J": lambda: fld.divide_by_volume("J"),
        "filter_J": lambda: fld.filter_spect("J"),
        "filter_rho_next": lambda: fld.filter_spect("rho_next"),
        "filter_rho_prev": lambda: fld.filter_spect("rho_prev"),
        "correct_currents": lambda: fld.correct_currents(check_exchanges=False),
        "push_eb_and_rho": lambda: fld.push(False, check_exchanges=False),
        "shift_spect_grid": shift,
        "exchange_and_damp_EB": sim.exchange_and_damp_EB,
    }
    return ops


def _fill(tangents, rng, alpha):
    for arrays in (*tangents.spect, *tangents.interp):
        for value in arrays.values():
            shape = value.shape[1:]
            for k in range(2):
                value[k] = cupy.asarray(
                    rng.normal(size=shape) + 1j * rng.normal(size=shape)
                )
            value[2] = value[0] + alpha * value[1]


def _worst(tangents, alpha):
    worst = {}
    for kind, modes, names in (
        ("spect", tangents.spect, SPECT_NAMES),
        ("interp", tangents.interp, INTERP_NAMES),
    ):
        for m, arrays in enumerate(modes):
            for name in names:
                value = arrays[name]
                ax = float(cupy.abs(value[0]).max())
                ay = float(cupy.abs(value[1]).max())
                residual = float(cupy.abs(value[2] - value[0] - alpha * value[1]).max())
                scale = ax + abs(alpha) * ay
                worst[f"{kind}{m}.{name}"] = residual / scale if scale else residual
    return worst


def test_field_operator_linearity(sim):
    label, simulation = sim
    rng = np.random.default_rng(7)
    tangents = FieldTangents(simulation.fld, 3)
    primal = {
        f"{m}.{name}": getattr(grid, name).copy()
        for m, grid in enumerate(simulation.fld.spect)
        for name in SPECT_NAMES
    }
    results = {}
    for op_name, operation in _operations(simulation).items():
        errors = []
        for _ in range(3):
            alpha = float(rng.uniform(-2.0, 2.0))
            _fill(tangents, rng, alpha)
            tangents.apply(simulation.fld, operation)
            worst = _worst(tangents, alpha)
            key = max(worst, key=worst.get)
            errors.append((worst[key], key))
        results[op_name] = max(errors)
        print(
            f"{label} {op_name}: worst {results[op_name][0]:.2e} ({results[op_name][1]})"
        )
    for key, value in primal.items():
        m, name = key.split(".")
        assert cupy.array_equal(getattr(simulation.fld.spect[int(m)], name), value), key
    failures = {k: v for k, v in results.items() if v[0] > TOLERANCE}
    assert not failures, failures


def test_k0_round_trip_probe():
    """Appendix A row 1: interp2spect(spect2interp(s)) against s after 500 steps."""
    from fbpic.autodiff.stepper import begin, run
    from twin_config import build_twin

    twin = build_twin((0.27, 9e-05), "twin_s", ionization="none")
    state = begin(twin.sim, twin.species)
    run(state, 500)
    fld = twin.sim.fld
    probe = FieldTangents(fld, 1)
    for m, grid in enumerate(fld.spect):
        for name in ("Ep", "Em", "Ez", "Bp", "Bm", "Bz"):
            probe.spect[m][name][0] = getattr(grid, name)

    def round_trip():
        fld.spect2interp("E")
        fld.spect2interp("B")
        fld.interp2spect("E")
        fld.interp2spect("B")

    probe.apply(fld, round_trip)
    report = {}
    for m, grid in enumerate(fld.spect):
        for name in ("Ep", "Em", "Ez", "Bp", "Bm", "Bz"):
            s = cupy.asnumpy(getattr(grid, name))
            rt = cupy.asnumpy(probe.spect[m][name][0])
            scale = float(np.abs(s).max())
            report[f"{m}.{name}"] = {
                "round_trip": float(np.abs(rt - s).max()) / scale,
                "kz0_row_max": float(np.abs(s[0]).max()) / scale,
                "kr0_column_max": float(np.abs(s[:, 0]).max()) / scale,
                "kr0_column_round_trip": float(np.abs(rt[:, 0] - s[:, 0]).max())
                / scale,
            }
    print(json.dumps(report, indent=1))
    out = INVERSION / "results" / "p4_k0_round_trip.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
