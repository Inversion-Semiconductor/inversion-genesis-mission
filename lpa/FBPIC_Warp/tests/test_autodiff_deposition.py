"""Check fused-forward deposition gradients with fixed sorted-cell metadata."""

import os

import numpy as np
import pytest
import warp as wp

from fbpic.autodiff.deposition import _structs, deposit_five_modes, make_particle_kernel
from fbpic.autodiff.verification import check_vjp
from fbpic.particles.deposition.cuda_methods_fused import make_kernel

DEVICE = os.environ.get("FBPIC_AUTODIFF_DEVICE", "cuda:0")
NZ, NR = 8, 6
PARTICLE_NAMES = ("x", "y", "z", "w", "ux", "uy", "uz", "inv_gamma")


def fixture(current, empty=False):
    rng = np.random.default_rng(83)
    upper_z = np.array([2, 2, 2, 3, 3, 5, 0, 0, 1, 1, 4, 6, 6, 6, 6, 7, 7])
    upper_r = np.array([1, 1, 1, 2, 2, 4, 0, 0, 1, 1, 4, 5, 5, 5, 5, 6, 6])
    cell = upper_z * (NR + 1) + upper_r
    order = np.argsort(cell, kind="stable")
    if empty:
        order = np.empty(0, dtype=int)
    cell, upper_z, upper_r = cell[order], upper_z[order], upper_r[order]
    count = len(cell)
    radii = upper_r + 0.25
    angles = rng.uniform(-2, 2, count)
    moments = rng.normal(size=(3, count))
    arrays = {
        "x": radii * np.cos(angles),
        "y": radii * np.sin(angles),
        "z": upper_z.astype(float) + 0.25,
        "w": rng.uniform(0.5, 2, count),
        "ux": moments[0],
        "uy": moments[1],
        "uz": moments[2],
        "inv_gamma": 1 / np.sqrt(1 + np.sum(moments**2, axis=0)),
        "prefix": np.cumsum(np.bincount(cell, minlength=NZ * (NR + 1))).astype(
            np.int32
        ),
    }
    for mode in range(5):
        arrays[f"beta{mode}"] = rng.uniform(0, 0.05, NR + 1)
    scale = 1e8 if current else 1
    for component in range(15 if current else 5):
        arrays[f"grid{component}"] = scale * (
            rng.normal(size=(NZ, NR)) + 1j * rng.normal(size=(NZ, NR))
        )
    return arrays


def to_warp(value, requires_grad=False):
    dtype = wp.float64
    if np.iscomplexobj(value):
        value = np.stack((value.real, value.imag), axis=-1)
        dtype = wp.vec2d
    elif value.dtype == np.int32:
        dtype = wp.int32
    return wp.array(value, dtype=dtype, device=DEVICE, requires_grad=requires_grad)


def forward(values, current):
    grids = [values[f"grid{i}"] for i in range(15 if current else 5)]
    outputs = deposit_five_modes(
        *(values[name] for name in PARTICLE_NAMES),
        grids,
        [values[f"beta{i}"] for i in range(5)],
        values["prefix"],
        q=1.0,
        invdz=1.0,
        zmin=0.0,
        nz=NZ,
        invdr=1.0,
        rmin=0.0,
        nr=NR,
        current=current,
    )
    return {f"grid{i}": array for i, array in enumerate(outputs)}


def original(values, current, particle_wise=False):
    from fbpic.autodiff.deposition import particle_cells_from_prefix

    grids = [wp.clone(values[f"grid{i}"]) for i in range(15 if current else 5)]
    prefix = values["prefix"]
    metadata = prefix
    if particle_wise:
        metadata = wp.empty(values["x"].shape, dtype=wp.int32, device=DEVICE)
        wp.launch(
            particle_cells_from_prefix, dim=prefix.shape, inputs=[prefix, metadata]
        )
    arguments = [
        *(values[name] for name in ("x", "y", "z", "w")),
        1.0,
        *(values[name] for name in ("ux", "uy", "uz", "inv_gamma")),
        1.0,
        0.0,
        NZ,
        1.0,
        0.0,
        NR,
    ]
    for start in range(0, 5, 2 if current else 5):
        count = min(2 if current else 5, 5 - start)
        selected = [
            grids[field * 5 + mode]
            for field in range(3 if current else 1)
            for mode in range(start, start + count)
        ]
        buffers, betas = _structs(
            selected, [values[f"beta{i}"] for i in range(start, start + count)]
        )
        kernel = (
            make_particle_kernel(current, start, count)
            if particle_wise
            else make_kernel(current, start, count)
        )
        wp.launch(
            kernel,
            dim=values["x"].shape if particle_wise else prefix.shape,
            inputs=arguments + [buffers, metadata, betas],
            device=DEVICE,
        )
    return grids


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_fused_forward_and_particle_expression_parity(current, empty):
    values = {name: to_warp(value) for name, value in fixture(current, empty).items()}
    result = list(forward(values, current).values())
    expected = original(values, current)
    particle = original(values, current, particle_wise=True)
    wp.synchronize_device(DEVICE)
    for value, reference, per_particle in zip(result, expected, particle):
        data = reference.numpy()
        scale = float(np.max(np.abs(data), initial=0)) or 1
        np.testing.assert_allclose(value.numpy(), data, rtol=2e-12, atol=2e-12 * scale)
        np.testing.assert_allclose(
            per_particle.numpy(), data, rtol=2e-12, atol=2e-12 * scale
        )


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("inputs", ["positions", "weights", "initial_grids"])
def test_deposition_vjp(current, inputs):
    names = {
        "positions": ("x", "y", "z"),
        "weights": ("w",),
        "initial_grids": tuple(f"grid{i}" for i in range(15 if current else 5)),
    }[inputs]
    check_vjp(
        lambda values: forward(values, current),
        fixture(current),
        names,
        device=DEVICE,
        linear=inputs != "positions",
        seed=31,
    )


@pytest.mark.parametrize("names", [("ux", "uy", "uz"), ("inv_gamma",)])
def test_current_momentum_vjp(names):
    check_vjp(
        lambda values: forward(values, True),
        fixture(True),
        names,
        device=DEVICE,
        linear=True,
        seed=91,
    )


@pytest.mark.parametrize("current", [False, True])
def test_empty_deposition_grid_identity(current):
    check_vjp(
        lambda values: forward(values, current),
        fixture(current, empty=True),
        tuple(f"grid{i}" for i in range(15 if current else 5)),
        device=DEVICE,
        linear=True,
    )


def test_ruyten_coefficients_are_fixed_inputs():
    values = {name: to_warp(value) for name, value in fixture(False).items()}
    values["beta0"].requires_grad = True
    with pytest.raises(ValueError, match="Ruyten coefficient arrays must remain fixed"):
        forward(values, False)


@pytest.mark.parametrize("current", [False, True])
def test_deposition_and_filter_compose(current):
    from fbpic.autodiff.fields import filter_scalar

    host = fixture(current)
    host["filter_z"] = np.linspace(0.2, 0.9, NZ)
    host["filter_r"] = np.linspace(0.1, 0.8, NR)

    def filtered(values):
        deposited = forward(values, current)
        outputs = {}
        for name, grid in deposited.items():
            result = wp.empty_like(grid, requires_grad=True)
            wp.launch(
                filter_scalar,
                dim=(NZ, NR),
                inputs=[grid, NZ, NR, values["filter_z"], values["filter_r"], result],
                device=DEVICE,
            )
            outputs[name] = result
        return outputs

    check_vjp(filtered, host, ("x", "y", "z", "w"), device=DEVICE, seed=61)
