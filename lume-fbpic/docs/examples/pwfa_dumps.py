"""The openPMD dumps of a `PWFASimulator` run, read for the plots of the PWFA examples
(`pwfa_gaussian.py`, `pwfa_clara_febe.py`)."""

from __future__ import annotations

import pathlib

import h5py
import numpy
from scipy.constants import c, m_e

# The bunches a PWFA run writes (the plasma only on request, and not read here).
BUNCHES = ("driver", "witness")


def distance(path: pathlib.Path) -> float:
    """How far the window has moved at the dump at `path` [m]: c times its time."""
    with h5py.File(path, "r") as f:
        return c * float(_step(f).attrs["time"])


def dumps(diagnostics_directory: pathlib.Path) -> list[pathlib.Path]:
    """The dumps of a run in its `diagnostics_directory()`, in order.

    Raises:
        FileNotFoundError: If there are none.
    """
    paths = sorted(pathlib.Path(diagnostics_directory).glob("data*.h5"))
    if not paths:
        raise FileNotFoundError(f"no dumps under {diagnostics_directory}")
    return paths


def read_dump(path: pathlib.Path) -> dict:
    """The dump at `path`, mode 0 of the fields:

    - `distance` [m], as `distance()` gives it;
    - `E_z` and `F_r = E_r - c B_theta` [GV/m] and `rho` [C/m^3] on the grid, shape (r, z);
    - `zeta` [m], the z of each grid column less `z_driver`, the driver's charge-weighted mean z,
      and `r` [m], the radius of each grid row;
    - `particles`: for each of the `BUNCHES` in the dump, the positions `x`, `y`, `z` [m], the
      momenta `ux`, `uy`, `uz` normalized to `m c`, and the `weight` (electrons per
      macroparticle).
    """
    with h5py.File(path, "r") as f:
        step = _step(f)
        mesh = step["fields/E"]
        dr, dz = mesh.attrs["gridSpacing"]
        z0 = mesh.attrs["gridGlobalOffset"][1]
        e_z, e_r, b_t, rho = (
            step[name][0]
            for name in ("fields/E/z", "fields/E/r", "fields/B/t", "fields/rho")
        )
        particles = {
            name: _bunch(step[f"particles/{name}"])
            for name in BUNCHES
            if name in step["particles"]
        }
        time = float(step.attrs["time"])
    z_driver = numpy.average(
        particles["driver"]["z"], weights=particles["driver"]["weight"]
    )
    n_r, n_z = e_z.shape
    return {
        "distance": c * time,
        "E_z": e_z * 1.0e-9,
        "F_r": (e_r - c * b_t) * 1.0e-9,
        "rho": rho,
        "zeta": z0 + dz * numpy.arange(n_z) - z_driver,
        "r": dr * (numpy.arange(n_r) + 0.5),
        "particles": particles,
        "z_driver": z_driver,
    }


def _bunch(group: h5py.Group) -> dict[str, numpy.ndarray]:
    return {
        **{axis: group[f"position/{axis}"][:] for axis in "xyz"},
        **{f"u{axis}": group[f"momentum/{axis}"][:] / (m_e * c) for axis in "xyz"},
        "weight": group["weighting"][:],
    }


def _step(f: h5py.File) -> h5py.Group:
    """The one iteration a fbpic dump file holds."""
    (step,) = f["data"].values()
    return step
