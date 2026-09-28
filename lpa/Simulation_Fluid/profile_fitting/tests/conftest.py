"""Shared fixtures: a headless Matplotlib backend, synthetic lineouts, a tiny cube."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from fludat_fit.families import GenericConicalTargetFamily  # noqa: E402
from fludat_fit.lineout import Lineout, LineoutConditions  # noqa: E402
from fludat_proc.density_cube import DensityCube, write_density_cube  # noqa: E402

CONDITIONS = LineoutConditions(x_mm=1.0, pressure_bar=12.5)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def make_lineout(
    relative: np.ndarray,
    z: np.ndarray,
    *,
    amplitude: float = 2.0e19,
    noise: float = 0.0,
    units: str = "cm^-3",
    seed: int = 0,
) -> Lineout:
    density = amplitude * relative
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        density = density + noise * density.max() * rng.standard_normal(z.size)
    return Lineout(
        z=z,
        density=np.clip(density, 0.0, None),
        density_units=units,
        conditions=CONDITIONS,
        source="synthetic",
    )


@pytest.fixture
def z_grid() -> np.ndarray:
    return np.linspace(-6.0e-3, 6.0e-3, 601)


@pytest.fixture
def supergaussian_truth() -> tuple[GenericConicalTargetFamily, dict[str, float]]:
    family = GenericConicalTargetFamily("supergaussian")
    return family, {"center": 0.4e-3, "fwhm": 3.0e-3, "beta": 3.0, "skew_rate": 200.0}


@pytest.fixture
def supergaussian_lineout(z_grid, supergaussian_truth) -> Lineout:
    family, params = supergaussian_truth
    return make_lineout(family.relative_density(params, z_grid), z_grid, noise=0.005)


def write_gaussian_cube(
    path: Path, *, sigma: float = 1.0e-3, nozzle: str = "test"
) -> Path:
    """A cube whose z profile is a Gaussian jet scaled by (1 + x) * pressure / 10."""
    z = np.linspace(-4.0e-3, 4.0e-3, 161)
    x = np.array([0.0, 1.0, 2.0])
    pressure = np.array([5.0, 10.0])
    jet = np.exp(-(z**2) / (2 * sigma**2))
    density = (
        1.0e19
        * jet[:, None, None]
        * (1 + x)[None, :, None]
        * (pressure / 10.0)[None, None, :]
    )
    cube = DensityCube(
        nozzle=nozzle,
        z=z,
        x=x,
        pressure=pressure,
        density=density,
        density_units="cm^-3",
    )
    return write_density_cube(path, cube, density_scale=1.0)


@pytest.fixture
def small_cube_path(tmp_path: Path) -> Path:
    return write_gaussian_cube(tmp_path / "test.h5")
