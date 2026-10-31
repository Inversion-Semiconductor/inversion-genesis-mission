# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Laser initial fields and their parameter tangents (inversion plan Section 4.8).

With the direct method, the laser sets the six initial interpolation fields
once, through the linear map L of `add_laser_direct`. `laser_initial_fields`
evaluates L into new arrays without changing the simulation. The E_L tangent
is F0 / (2 E_L), because every field scales as sqrt(E_L). The z_f tangent
(`GaussianLaserDzf`) belongs to Stage B.
"""

from __future__ import annotations

import numpy as np

FIELD_NAMES = ("Er", "Et", "Ez", "Br", "Bt", "Bz")


def laser_initial_fields(sim, profile, boost=None) -> list[dict]:
    """Evaluate the `add_laser_direct` map into new zero arrays.

    Steps 1 to 4 of plan Section 4.8: sample and decompose the profile,
    compute Ez and B on a global host `Fields`, and place the result into
    the rows of the local array that `add_laser_direct` writes. The guard
    rows stay zero. `sim` is not changed. Call it before the first step: the
    grid positions and `sim.time` enter the profile.

    Args:
        sim: Production `Simulation` (one rank).
        profile: Laser profile with `E_field` and `propag_direction`.
        boost: Boost converter, or None in the lab frame.

    Returns:
        One dict per mode from Er, Et, Ez, Br, Bt, Bz to complex128 host
        arrays with the shape of `sim.fld.interp[m]` arrays.
    """
    from fbpic.fields import Fields
    from fbpic.lpa_utils.laser.direct_injection import (
        calculate_laser_fields,
        get_laser_Er_Et,
    )

    comm, fld = sim.comm, sim.fld
    if comm.size != 1:
        raise ValueError("laser_initial_fields supports one rank only (D1).")
    laser_Er, laser_Et = get_laser_Er_Et(sim, profile, boost)
    global_Nz, _ = comm.get_Nz_and_iz(local=False, with_damp=True, with_guard=False)
    global_zmin, global_zmax = comm.get_zmin_zmax(
        local=False, with_damp=True, with_guard=False
    )
    global_fld = Fields(
        global_Nz,
        global_zmax,
        fld.Nr,
        fld.rmax,
        fld.Nm,
        fld.dt,
        zmin=global_zmin,
        n_order=fld.n_order,
        use_cuda=False,
    )
    for m in range(fld.Nm):
        global_fld.interp[m].Er = comm.gather_grid_array(
            np.ascontiguousarray(laser_Er[:, :, m]), with_damp=True
        )
        global_fld.interp[m].Et = comm.gather_grid_array(
            np.ascontiguousarray(laser_Et[:, :, m]), with_damp=True
        )
    calculate_laser_fields(global_fld, profile.propag_direction)
    Nz_local, iz_start_local_domain = comm.get_Nz_and_iz(
        local=True, with_damp=True, with_guard=False, rank=comm.rank
    )
    _, iz_start_local_array = comm.get_Nz_and_iz(
        local=True, with_damp=True, with_guard=True, rank=comm.rank
    )
    iz_in_array = iz_start_local_domain - iz_start_local_array
    shape = tuple(fld.interp[0].Er.shape)
    out = []
    for m in range(fld.Nm):
        mode = {}
        for name in FIELD_NAMES:
            array = np.zeros(shape, dtype=np.complex128)
            local = comm.scatter_grid_array(
                getattr(global_fld.interp[m], name), with_damp=True
            )
            array[iz_in_array : iz_in_array + Nz_local, :] += local
            mode[name] = array
        out.append(mode)
    return out


def energy_seed(initial_fields: list[dict], energy: float) -> list[dict]:
    """E_L tangent of the initial interp fields: F0 / (2 E_L), per J."""
    return [
        {name: value / (2.0 * energy) for name, value in mode.items()}
        for mode in initial_fields
    ]


def amplitude_seed(initial_fields: list[dict], a0: float) -> list[dict]:
    """a0 tangent of the initial interp fields: F0 / a0 (L4w, plan Section 6.5)."""
    return [
        {name: value / a0 for name, value in mode.items()} for mode in initial_fields
    ]
