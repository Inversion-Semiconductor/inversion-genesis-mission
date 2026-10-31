# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""L1: tangent (JVP) kernels of gather, Vay push, position push, deposition.

Inversion plan Sections 4.7 and 6.1. Each kernel passes two checks:

1. A dot product with the existing reverse-mode clone at Nm = 5:
   e = |<J u, v> - <u, J^T v>| / (|J u||v| + |u||J^T v|) <= 1e-12 for three
   random draws, computed in one launch with K = 3 tangents.
2. `check_jvp` (vector form, pair rule) against central differences of the
   production forward kernel at Nm = 2 and Nm = 5.

Particles at r = 0 are tested separately: the clone VJP is NaN there.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from scipy.constants import e, m_e, m_p

cupy = pytest.importorskip("cupy")

from fbpic.autodiff import tangent as tg  # noqa: E402
from fbpic.autodiff.particles import (  # noqa: E402
    get_gather_kernel,
    push_momenta,
    push_positions,
)
from fbpic.autodiff.verification import check_jvp, dot_product_error  # noqa: E402
from fbpic.utils.warp import get_runtime  # noqa: E402

DEVICE = os.environ.get("FBPIC_AUTODIFF_DEVICE", "cuda:0")
DOT_TOLERANCE = 1e-12
FIELDS = ("Er", "Et", "Ez", "Br", "Bt", "Bz")
GATHERED = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
MOMENTA = ("ux", "uy", "uz", "inv_gamma")
POSITIONS = ("x", "y", "z")
DT = 1e-15


@pytest.fixture(scope="module", autouse=True)
def runtime():
    wp.init()
    wp.set_device(DEVICE)
    get_runtime().use()


def _sync():
    get_runtime().synchronize()
    wp.synchronize_device(DEVICE)


def _cp(value):
    return cupy.asarray(value)


def _wp(value, requires_grad=False):
    value = np.asarray(value)
    dtype = wp.float64
    if np.iscomplexobj(value):
        value = np.stack((value.real, value.imag), axis=-1)
        dtype = wp.vec2d
    elif value.dtype == np.int32:
        dtype = wp.int32
    return wp.array(value, dtype=dtype, device=DEVICE, requires_grad=requires_grad)


def _pairs(value):
    value = np.asarray(value)
    if np.iscomplexobj(value):
        return np.stack((value.real, value.imag), axis=-1)
    return value


def _random_like(rng, value, scale):
    direction = rng.normal(size=value.shape)
    if np.iscomplexobj(value):
        direction = direction + 1j * rng.normal(size=value.shape)
    return scale * direction / np.sqrt(direction.size)


# Gather
# ------

GATHER_DZ, GATHER_DR = 2e-6, 3e-6
GATHER_NZ, GATHER_NR = 8, 6
GATHER_ZMIN, GATHER_RMAX = -5e-6, 18e-6


def _gather_fixture(modes=5, axis=False):
    rng = np.random.default_rng(211)
    # The last two radii are below 0.5 dr (guard-cell fold); the last but
    # two lies beyond rmax_gather.
    radius = GATHER_DR * np.array([1.17, 2.28, 3.36, 4.41, 2.64, 6.3, 0.21, 0.43, 0.08])
    theta = np.array([0.22, 1.04, -1.73, 2.45, -0.53, 0.7, 2.9, -2.2, 1.3])
    z = GATHER_ZMIN + GATHER_DZ * np.array(
        [1.17, 2.28, 3.36, 4.41, 5.22, 3.3, 2.61, 6.12, 1.4]
    )
    if axis:
        radius[0] = 0.0
    arrays = {"x": radius * np.cos(theta), "y": radius * np.sin(theta), "z": z}
    for mode in range(modes):
        for field in FIELDS:
            scale = 1e9 if field.startswith("E") else 3.0
            arrays[f"{field}_{mode}"] = scale * (
                rng.normal(size=(GATHER_NZ, GATHER_NR))
                + 1j * rng.normal(size=(GATHER_NZ, GATHER_NR))
            )
    return arrays


def _gather_scales(arrays):
    scales = {name: 3e-6 for name in POSITIONS}
    scales.update(
        {n: (1e9 if n.startswith("E") else 3.0) for n in arrays if n not in POSITIONS}
    )
    return scales


def _gather_grids(arrays, modes, library):
    grids = []
    for mode in range(modes):
        grid = SimpleNamespace(
            invdz=1 / GATHER_DZ,
            zmin=GATHER_ZMIN,
            Nz=GATHER_NZ,
            invdr=1 / GATHER_DR,
            rmin=0.0,
            Nr=GATHER_NR,
        )
        for field in FIELDS:
            setattr(grid, field, library(arrays[f"{field}_{mode}"]))
        grids.append(grid)
    return grids


def _gather_tangent(arrays, directions_list, modes):
    """Tangent outputs (one dict per direction) from one K-tangent launch."""
    K = len(directions_list)
    n = arrays["x"].size

    def stacked(name, shape):
        return _cp(
            np.stack(
                [
                    d.get(name, np.zeros(shape, dtype=arrays[name].dtype))
                    for d in directions_list
                ]
            )
        )

    positions = [_cp(arrays[name]) for name in POSITIONS]
    dpositions = [stacked(name, (n,)) for name in POSITIONS]
    grids = _gather_grids(arrays, modes, _cp)
    field_tangents = [
        {f: stacked(f"{f}_{m}", (GATHER_NZ, GATHER_NR)) for f in FIELDS}
        for m in range(modes)
    ]
    outputs = [cupy.zeros((K, n)) for _ in GATHERED]
    tg.gather_tangent(
        *positions, dpositions, grids, field_tangents, GATHER_RMAX, outputs
    )
    _sync()
    host = [cupy.asnumpy(o) for o in outputs]
    return [{g: host[i][k] for i, g in enumerate(GATHERED)} for k in range(K)]


def _launch_gather(values, modes, kernel):
    from fbpic.particles.gathering.cuda_methods_fused import GroupFields, ModeFields

    count = values["x"].shape[0]
    outputs = {
        name: wp.zeros(count, dtype=wp.float64, device=DEVICE, requires_grad=True)
        for name in GATHERED
    }
    fields = GroupFields()
    for slot in range(5):
        grids = ModeFields()
        for field in FIELDS:
            setattr(grids, field, values[f"{field}_{min(slot, modes - 1)}"])
        setattr(fields, f"mode{slot}", grids)
    wp.launch(
        kernel,
        dim=count,
        inputs=[values[n] for n in POSITIONS]
        + [
            GATHER_RMAX,
            1 / GATHER_DZ,
            GATHER_ZMIN,
            GATHER_NZ,
            1 / GATHER_DR,
            0.0,
            GATHER_NR,
            fields,
        ],
        outputs=list(outputs.values()),
        device=DEVICE,
    )
    return outputs


def _vjp(arrays, names, launch, cotangent):
    values = {n: _wp(v, requires_grad=n in names) for n, v in arrays.items()}
    tape = wp.Tape()
    with tape:
        outputs = launch(values)
    tape.backward(
        grads={
            out: wp.array(_pairs(cotangent[n]), dtype=out.dtype, device=DEVICE)
            for n, out in outputs.items()
        }
    )
    wp.synchronize_device(DEVICE)
    return {n: values[n].grad.numpy() for n in names}


def _dot_products(arrays, names, scales, out_scales, tangent_fn, launch, seed):
    rng = np.random.default_rng(seed)
    directions = [
        {n: _random_like(rng, arrays[n], scales[n]) for n in names} for _ in range(3)
    ]
    tangents = tangent_fn(directions)
    errors = []
    for direction, tangent in zip(directions, tangents):
        cotangent = {
            n: _random_like(rng, np.asarray(t), 1.0 / out_scales[n])
            for n, t in tangent.items()
        }
        adjoint = _vjp(arrays, names, launch, cotangent)
        errors.append(
            dot_product_error(
                tangent,
                cotangent,
                {n: _pairs(direction[n]) for n in names},
                adjoint,
                scales,
                out_scales,
            )
        )
    return errors


GATHER_OUT_SCALES = {g: (1e9 if g.startswith("E") else 3.0) for g in GATHERED}


@pytest.mark.parametrize("group", ["positions", "electric", "magnetic", "all"])
def test_gather_tangent_dot_product(group):
    arrays = _gather_fixture()
    if group == "positions":
        names = POSITIONS
    elif group == "all":
        names = tuple(arrays)
    else:
        prefix = "E" if group == "electric" else "B"
        names = tuple(n for n in arrays if n.startswith(prefix))
    errors = _dot_products(
        arrays,
        names,
        _gather_scales(arrays),
        GATHER_OUT_SCALES,
        lambda ds: _gather_tangent(arrays, ds, 5),
        lambda v: _launch_gather(v, 5, get_gather_kernel()),
        seed=17,
    )
    print(f"gather {group}: dot-product errors {errors}")
    assert max(errors) <= DOT_TOLERANCE, errors


@pytest.mark.parametrize("modes", [2, 5])
@pytest.mark.parametrize("group", ["positions", "fields"])
def test_gather_tangent_check_jvp(modes, group):
    from fbpic.particles.gathering.cuda_methods_fused import make_kernel

    arrays = _gather_fixture(modes)
    names = POSITIONS if group == "positions" else tuple(arrays)[3:]
    report = check_jvp(
        lambda v: _launch_gather(v, modes, make_kernel(0, modes)),
        lambda host, d: _gather_tangent(host, [d], modes)[0],
        arrays,
        names,
        device=DEVICE,
        scales=_gather_scales(arrays),
        seed=19,
        linear=group == "fields",
    )
    print(f"gather Nm={modes} {group}: best errors {report['best_error']}")


def test_gather_tangent_on_axis():
    """At r = 0: finite tangents, production branch, d r = d phi = 0."""
    arrays = _gather_fixture(axis=True)
    rng = np.random.default_rng(23)
    scales = _gather_scales(arrays)
    direction = {n: _random_like(rng, v, scales[n]) for n, v in arrays.items()}
    no_transverse = dict(direction)
    for name in ("x", "y"):
        no_transverse[name] = direction[name].copy()
        no_transverse[name][0] = 0.0
    full, frozen = _gather_tangent(arrays, [direction, no_transverse], 5)
    for name in GATHERED:
        assert np.isfinite(full[name]).all()
        assert full[name][0] == frozen[name][0], name


# Momentum and position push
# --------------------------


def _momenta_fixture(ions=False):
    rng = np.random.default_rng(91)
    momentum = rng.normal(scale=3.0 if not ions else 0.01, size=(3, 7))
    arrays = {name: value.copy() for name, value in zip(MOMENTA[:3], momentum)}
    arrays["inv_gamma"] = 1 / np.sqrt(1 + np.sum(momentum**2, axis=0))
    electric_scale, magnetic_scale = (1e11, 300.0) if ions else (1e12, 3e3)
    for name in GATHERED:
        scale = electric_scale if name.startswith("E") else magnetic_scale
        arrays[name] = rng.normal(size=7) * scale
    return arrays


def _momenta_scales(ions):
    scales = {name: 1.0 for name in MOMENTA}
    scales.update(
        {g: (1e11 if ions else 1e12) if g.startswith("E") else (300.0 if ions else 3e3)
         for g in GATHERED}
    )  # fmt: skip
    return scales


def _species(ions):
    return (2 * e, 4 * m_p) if ions else (-e, m_e)


def _vay_tangent(arrays, directions_list, ions):
    K = len(directions_list)
    n = arrays["ux"].size
    q, m = _species(ions)

    def stacked(name):
        return _cp(np.stack([d.get(name, np.zeros(n)) for d in directions_list]))

    tangents = [stacked(name) for name in MOMENTA]
    tg.vay_tangent(
        [_cp(arrays[name]) for name in MOMENTA],
        [_cp(arrays[name]) for name in GATHERED],
        q,
        m,
        DT,
        tangents,
        [stacked(name) for name in GATHERED],
    )
    _sync()
    host = [cupy.asnumpy(t) for t in tangents]
    return [{name: host[i][k] for i, name in enumerate(MOMENTA)} for k in range(K)]


def _launch_vay_clone(values, ions):
    q, m = _species(ions)
    count = values["ux"].shape[0]
    outputs = {
        name: wp.zeros(count, dtype=wp.float64, device=DEVICE, requires_grad=True)
        for name in MOMENTA
    }
    wp.launch(
        push_momenta,
        dim=count,
        inputs=[values[name] for name in MOMENTA + GATHERED] + [q, m, DT],
        outputs=list(outputs.values()),
        device=DEVICE,
    )
    return outputs


def _launch_vay_production(values, ions):
    from fbpic.particles.push.cuda_methods import push_p_gpu

    q, m = _species(ions)
    count = values["ux"].shape[0]
    wp.launch(
        push_p_gpu.kernel,
        dim=count,
        inputs=[values[name] for name in MOMENTA + GATHERED] + [q, m, count, DT],
        device=DEVICE,
    )
    return {name: values[name] for name in MOMENTA}


@pytest.mark.parametrize("ions", [False, True])
@pytest.mark.parametrize("group", ["momenta", "fields", "all"])
def test_vay_tangent_dot_product(ions, group):
    arrays = _momenta_fixture(ions)
    names = {"momenta": MOMENTA, "fields": GATHERED, "all": MOMENTA + GATHERED}[group]
    scales = _momenta_scales(ions)
    errors = _dot_products(
        arrays,
        names,
        scales,
        {name: 1.0 for name in MOMENTA},
        lambda ds: _vay_tangent(arrays, ds, ions),
        lambda v: _launch_vay_clone(v, ions),
        seed=29,
    )
    print(f"vay ions={ions} {group}: dot-product errors {errors}")
    assert max(errors) <= DOT_TOLERANCE, errors


@pytest.mark.parametrize("ions", [False, True])
def test_vay_tangent_check_jvp(ions):
    arrays = _momenta_fixture(ions)
    report = check_jvp(
        lambda v: _launch_vay_production(v, ions),
        lambda host, d: _vay_tangent(host, [d], ions)[0],
        arrays,
        MOMENTA + GATHERED,
        device=DEVICE,
        scales=_momenta_scales(ions),
        seed=31,
    )
    print(f"vay ions={ions}: best errors {report['best_error']}")


PUSH = (1.0, -0.7, 0.2)


def _position_fixture():
    arrays = _momenta_fixture()
    rng = np.random.default_rng(102)
    for name in POSITIONS:
        arrays[name] = rng.normal(size=7) * 1e-6
    return arrays


def _position_scales():
    scales = {name: 1e-6 for name in POSITIONS}
    scales.update({name: 1.0 for name in MOMENTA})
    return scales


def _position_tangent(arrays, directions_list):
    K = len(directions_list)
    n = arrays["x"].size

    def stacked(name):
        return _cp(np.stack([d.get(name, np.zeros(n)) for d in directions_list]))

    positions = [stacked(name) for name in POSITIONS]
    tg.position_tangent(
        [_cp(arrays[name]) for name in MOMENTA],
        [stacked(name) for name in MOMENTA],
        DT,
        positions,
        PUSH,
    )
    _sync()
    host = [cupy.asnumpy(t) for t in positions]
    return [{name: host[i][k] for i, name in enumerate(POSITIONS)} for k in range(K)]


def _launch_positions_clone(values):
    count = values["x"].shape[0]
    outputs = {
        name: wp.zeros(count, dtype=wp.float64, device=DEVICE, requires_grad=True)
        for name in POSITIONS
    }
    wp.launch(
        push_positions,
        dim=count,
        inputs=[values[name] for name in POSITIONS + MOMENTA] + [DT, *PUSH],
        outputs=list(outputs.values()),
        device=DEVICE,
    )
    return outputs


def _launch_positions_production(values):
    from fbpic.particles.push.cuda_methods import push_x_gpu

    wp.launch(
        push_x_gpu.kernel,
        dim=values["x"].shape[0],
        inputs=[values[name] for name in POSITIONS + MOMENTA] + [DT, *PUSH],
        device=DEVICE,
    )
    return {name: values[name] for name in POSITIONS}


def test_position_tangent_dot_product():
    arrays = _position_fixture()
    names = POSITIONS + MOMENTA
    errors = _dot_products(
        arrays,
        names,
        _position_scales(),
        {name: 1e-6 for name in POSITIONS},
        lambda ds: _position_tangent(arrays, ds),
        _launch_positions_clone,
        seed=37,
    )
    print(f"position: dot-product errors {errors}")
    assert max(errors) <= DOT_TOLERANCE, errors


def test_position_tangent_check_jvp():
    report = check_jvp(
        _launch_positions_production,
        lambda host, d: _position_tangent(host, [d])[0],
        _position_fixture(),
        POSITIONS + MOMENTA,
        device=DEVICE,
        scales=_position_scales(),
        seed=41,
    )
    print(f"position: best errors {report['best_error']}")


# Deposition
# ----------

DEP_NZ, DEP_NR = 8, 6
DEP_PARTICLES = ("x", "y", "z", "w", "ux", "uy", "uz", "inv_gamma")


def _deposit_fixture(current, modes=5, axis=False):
    rng = np.random.default_rng(83)
    upper_z = np.array([2, 2, 2, 3, 3, 5, 0, 0, 1, 1, 4, 6, 6, 6, 6, 7, 7, 3])
    upper_r = np.array([1, 1, 1, 2, 2, 4, 0, 0, 1, 1, 4, 5, 5, 5, 5, 6, 6, 0])
    cell = upper_z * (DEP_NR + 1) + upper_r
    order = np.argsort(cell, kind="stable")
    cell, upper_z, upper_r = cell[order], upper_z[order], upper_r[order]
    count = len(cell)
    # radius - 0.5 lies in (upper_r - 1, upper_r]; the upper_r = 0 particles
    # sit below 0.5 dr and use the guard-cell fold.
    radii = np.where(upper_r == 0, 0.3, upper_r + 0.25)
    if axis:
        radii[upper_r == 0] = 0.0
    angles = rng.uniform(-2, 2, count)
    momenta = rng.normal(size=(3, count))
    arrays = {
        "x": radii * np.cos(angles),
        "y": radii * np.sin(angles),
        "z": upper_z.astype(float) + 0.25,
        "w": rng.uniform(0.5, 2, count),
        "ux": momenta[0],
        "uy": momenta[1],
        "uz": momenta[2],
        "inv_gamma": 1 / np.sqrt(1 + np.sum(momenta**2, axis=0)),
        "prefix": np.cumsum(np.bincount(cell, minlength=DEP_NZ * (DEP_NR + 1))).astype(
            np.int32
        ),
    }
    for mode in range(modes):
        arrays[f"beta{mode}"] = rng.uniform(0, 0.05, DEP_NR + 1)
    scale = 1e8 if current else 1.0
    for component in range((3 if current else 1) * modes):
        arrays[f"grid{component}"] = scale * (
            rng.normal(size=(DEP_NZ, DEP_NR)) + 1j * rng.normal(size=(DEP_NZ, DEP_NR))
        )
    return arrays


def _grid_names(current, modes):
    return [f"grid{i}" for i in range((3 if current else 1) * modes)]


def _deposit_tangent(arrays, directions_list, current, modes):
    K = len(directions_list)
    n = arrays["x"].size

    def stacked(name, shape):
        return _cp(
            np.stack(
                [
                    d.get(name, np.zeros(shape, dtype=arrays[name].dtype))
                    for d in directions_list
                ]
            )
        )

    grids = [
        SimpleNamespace(
            invdz=1.0,
            zmin=0.0,
            Nz=DEP_NZ,
            invdr=1.0,
            rmin=0.0,
            Nr=DEP_NR,
            d_ruyten_linear_coef=_cp(arrays[f"beta{m}"]),
        )
        for m in range(modes)
    ]
    outputs = [stacked(name, (DEP_NZ, DEP_NR)) for name in _grid_names(current, modes)]
    tg.deposit_tangent(
        [_cp(arrays[name]) for name in DEP_PARTICLES],
        [stacked(name, (n,)) for name in DEP_PARTICLES],
        1.0,
        grids,
        _cp(arrays["prefix"]),
        outputs,
        current,
    )
    _sync()
    host = [cupy.asnumpy(o) for o in outputs]
    names = _grid_names(current, modes)
    return [{names[i]: host[i][k] for i in range(len(names))} for k in range(K)]


def _launch_deposit_clone(values, current):
    from fbpic.autodiff.deposition import deposit_five_modes

    outputs = deposit_five_modes(
        *(values[name] for name in DEP_PARTICLES),
        [values[name] for name in _grid_names(current, 5)],
        [values[f"beta{i}"] for i in range(5)],
        values["prefix"],
        q=1.0,
        invdz=1.0,
        zmin=0.0,
        nz=DEP_NZ,
        invdr=1.0,
        rmin=0.0,
        nr=DEP_NR,
        current=current,
    )
    return {f"grid{i}": array for i, array in enumerate(outputs)}


def _launch_deposit_production(values, current, modes):
    from fbpic.autodiff.deposition import _structs
    from fbpic.particles.deposition.cuda_methods_fused import make_kernel

    grids = [wp.clone(values[name]) for name in _grid_names(current, modes)]
    arguments = [
        *(values[name] for name in ("x", "y", "z", "w")),
        1.0,
        *(values[name] for name in ("ux", "uy", "uz", "inv_gamma")),
        1.0,
        0.0,
        DEP_NZ,
        1.0,
        0.0,
        DEP_NR,
    ]
    group = 2 if current else 5
    for start in range(0, modes, group):
        count = min(group, modes - start)
        selected = [
            grids[field * modes + mode]
            for field in range(3 if current else 1)
            for mode in range(start, start + count)
        ]
        buffers, betas = _structs(
            selected, [values[f"beta{i}"] for i in range(start, start + count)]
        )
        wp.launch(
            make_kernel(current, start, count),
            dim=values["prefix"].shape,
            inputs=arguments + [buffers, values["prefix"], betas],
            device=DEVICE,
        )
    return {name: grid for name, grid in zip(_grid_names(current, modes), grids)}


def _deposit_scales(arrays):
    scales = {
        name: float(np.max(np.abs(arrays[name])))
        for name in arrays
        if name != "prefix" and not name.startswith("beta")
    }
    scales.update({name: 0.2 for name in POSITIONS})
    return scales


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("group", ["positions", "weights", "momenta", "grids", "all"])
def test_deposit_tangent_dot_product(current, group):
    if group == "momenta" and not current:
        pytest.skip("Charge does not depend on the momenta.")
    arrays = _deposit_fixture(current)
    grids = tuple(_grid_names(current, 5))
    names = {
        "positions": POSITIONS,
        "weights": ("w",),
        "momenta": MOMENTA,
        "grids": grids,
        "all": DEP_PARTICLES + grids,
    }[group]
    scales = _deposit_scales(arrays)
    out_scales = {name: scales[name] for name in grids}
    errors = _dot_products(
        arrays,
        names,
        scales,
        out_scales,
        lambda ds: _deposit_tangent(arrays, ds, current, 5),
        lambda v: _launch_deposit_clone(v, current),
        seed=43,
    )
    print(f"deposit current={current} {group}: dot-product errors {errors}")
    assert max(errors) <= DOT_TOLERANCE, errors


@pytest.mark.parametrize("modes", [2, 5])
@pytest.mark.parametrize("current", [False, True])
def test_deposit_tangent_check_jvp(modes, current):
    arrays = _deposit_fixture(current, modes)
    names = POSITIONS + ("w",) + (MOMENTA if current else ())
    report = check_jvp(
        lambda v: _launch_deposit_production(v, current, modes),
        lambda host, d: _deposit_tangent(host, [d], current, modes)[0],
        arrays,
        names,
        device=DEVICE,
        scales=_deposit_scales(arrays),
        seed=47,
    )
    print(f"deposit Nm={modes} current={current}: best errors {report['best_error']}")


@pytest.mark.parametrize("current", [False, True])
def test_deposit_tangent_on_axis(current):
    """At r = 0: finite tangents, production branch, d r = d phi = 0."""
    arrays = _deposit_fixture(current, axis=True)
    on_axis = (arrays["x"] == 0) & (arrays["y"] == 0)
    assert on_axis.sum() == 3
    rng = np.random.default_rng(53)
    scales = _deposit_scales(arrays)
    names = DEP_PARTICLES
    direction = {n: _random_like(rng, arrays[n], scales[n]) for n in names}
    frozen = {n: v.copy() for n, v in direction.items()}
    frozen["x"][on_axis] = 0.0
    frozen["y"][on_axis] = 0.0
    full, reference = _deposit_tangent(arrays, [direction, frozen], current, 5)
    for name, value in full.items():
        assert np.isfinite(value).all()
        scale = float(np.max(np.abs(value)))
        np.testing.assert_allclose(value, reference[name], rtol=0, atol=1e-14 * scale)
