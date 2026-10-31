# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Field tangents that reuse the production linear field operators.

Inversion plan Sections 4.4 and 4.6. Every field operation of the step is
linear in the field arrays, with coefficients that do not depend on the
parameters. The tangent of an operation is therefore the same operation
applied to the tangent arrays. `FieldTangents` holds K copies of every field
array of every mode, and `apply` runs a production method once per tangent
with the tangent arrays swapped into the production grid objects.

The swap replaces attribute references only; no primal data is copied or
written. `apply` checks that the production method did not rebind any
swapped attribute, and it restores the primal references even if the
method raises. L2t (plan Section 6.1) checks bitwise that the primal arrays
do not change.
"""

from __future__ import annotations

from contextlib import contextmanager

import cupy

SPECT_NAMES = (
    "Ep",
    "Em",
    "Ez",
    "Bp",
    "Bm",
    "Bz",
    "Jp",
    "Jm",
    "Jz",
    "rho_prev",
    "rho_next",
)
INTERP_NAMES = ("Er", "Et", "Ez", "Br", "Bt", "Bz", "Jr", "Jt", "Jz", "rho")
# The state part of the field tangents (plan Section 4.2). The other arrays
# are rewritten within each step before they are read.
STATE_SPECT_NAMES = ("Ep", "Em", "Ez", "Bp", "Bm", "Bz", "rho_prev")
STATE_INTERP_NAMES = ("Er", "Et", "Ez", "Br", "Bt", "Bz")


class FieldTangents:
    """K tangents of every spectral and interpolation field array.

    Attributes:
        K: Number of tangents.
        spect: One dict per mode from a spectral array name to a complex128
            CuPy array of shape (K, Nz, Nr).
        interp: The same for the interpolation arrays.
    """

    def __init__(self, fld, K: int):
        self.K = int(K)
        self.spect = [
            {
                name: cupy.zeros((self.K, *getattr(grid, name).shape), complex)
                for name in SPECT_NAMES
            }
            for grid in fld.spect
        ]
        self.interp = [
            {
                name: cupy.zeros((self.K, *getattr(grid, name).shape), complex)
                for name in INTERP_NAMES
            }
            for grid in fld.interp
        ]

    def gather_view(self):
        """Per-mode E and B interpolation tangents for `gather_tangent`."""
        return [
            {name: mode[name] for name in STATE_INTERP_NAMES} for mode in self.interp
        ]

    def deposit_view(self, fieldtype: str):
        """Field-major, mode-minor interpolation tangents for `deposit_tangent`."""
        names = ("Jr", "Jt", "Jz") if fieldtype == "J" else ("rho",)
        return [mode[name] for name in names for mode in self.interp]

    @contextmanager
    def swapped(self, fld, k: int):
        """Swap tangent k into the production grid objects of `fld`."""
        saved = []
        try:
            for grids, tangents, names in (
                (fld.spect, self.spect, SPECT_NAMES),
                (fld.interp, self.interp, INTERP_NAMES),
            ):
                for grid, mode in zip(grids, tangents):
                    for name in names:
                        view = mode[name][k]
                        saved.append((grid, name, getattr(grid, name), view))
                        setattr(grid, name, view)
            yield
        finally:
            rebound = []
            for grid, name, primal, view in reversed(saved):
                if getattr(grid, name) is not view:
                    rebound.append(name)
                setattr(grid, name, primal)
            if rebound:
                raise RuntimeError(
                    f"A field operation rebound swapped arrays: {sorted(set(rebound))}"
                )

    def apply(self, fld, operation):
        """Run operation() once per tangent with that tangent swapped in."""
        for k in range(self.K):
            with self.swapped(fld, k):
                operation()

    def zero(self):
        for arrays in (*self.spect, *self.interp):
            for value in arrays.values():
                value[...] = 0
