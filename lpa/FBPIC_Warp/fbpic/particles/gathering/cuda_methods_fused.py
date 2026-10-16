# Copyright 2016, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Fuse linear field gathering across groups of up to five azimuthal modes.

One thread handles one particle and computes its cylindrical geometry and
linear weights once per group. The first group writes the six particle fields;
later groups add to them. Each mode is converted to Cartesian components before
it is summed, as in the per-mode kernels.
"""

from functools import cache

import warp as wp

from fbpic.utils.warp import as_warp_array, cmul, launch_kernel

from .cuda_inline_functions import add_linear_gather_for_mode

wp.set_module_options({"enable_backward": False})

# Modes per launch. Warp has no arrays of arrays, so a group of mode grids is
# passed as a struct with one fixed field per mode. Five modes (30 grid arrays)
# keep the kernel parameter block well below the 4 KB CUDA limit.
MAX_GROUP_MODES = 5
FIELD_NAMES = ("Er", "Et", "Ez", "Br", "Bt", "Bz")


@wp.struct
class ModeFields:
    Er: wp.array2d(dtype=wp.vec2d)
    Et: wp.array2d(dtype=wp.vec2d)
    Ez: wp.array2d(dtype=wp.vec2d)
    Br: wp.array2d(dtype=wp.vec2d)
    Bt: wp.array2d(dtype=wp.vec2d)
    Bz: wp.array2d(dtype=wp.vec2d)


@wp.struct
class GroupFields:
    mode0: ModeFields
    mode1: ModeFields
    mode2: ModeFields
    mode3: ModeFields
    mode4: ModeFields


@wp.func
def select_mode(fields: GroupFields, local_mode: int):
    # local_mode is a compile-time constant at each call site (unrolled loop).
    selected = fields.mode0
    if local_mode == 1:
        selected = fields.mode1
    elif local_mode == 2:
        selected = fields.mode2
    elif local_mode == 3:
        selected = fields.mode3
    elif local_mode == 4:
        selected = fields.mode4
    return selected


def group_fields(grids) -> GroupFields:
    """Borrow the complex128 E and B arrays of up to five mode grids.

    Unused struct slots repeat the last grid; the kernel never reads them.
    """
    fields = GroupFields()
    for slot in range(MAX_GROUP_MODES):
        grid = grids[min(slot, len(grids) - 1)]
        mode = ModeFields()
        for name in FIELD_NAMES:
            setattr(mode, name, as_warp_array(getattr(grid, name), wp.vec2d))
        setattr(fields, f"mode{slot}", mode)
    return fields


@cache
def make_kernel(first_mode: int, mode_count: int):
    """Return a forward-only linear gather kernel for one group of modes.

    Args:
        first_mode: Global index of the first mode in the group.
        mode_count: Number of adjacent modes in the group, from 1 to 5.

    Returns:
        A kernel that writes (first group) or adds (later groups) Ex, Ey, Ez,
        Bx, By, and Bz for each particle.
    """
    if first_mode < 0 or not 1 <= mode_count <= MAX_GROUP_MODES:
        raise ValueError("A mode group needs a non-negative start and 1-5 modes.")
    accumulate = first_mode > 0

    @wp.kernel(enable_backward=False, module="unique")
    def gather(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        rmax_gather: wp.float64,
        invdz: wp.float64,
        zmin: wp.float64,
        Nz: wp.int32,
        invdr: wp.float64,
        rmin: wp.float64,
        Nr: wp.int32,
        fields: GroupFields,
        Ex: wp.array(dtype=wp.float64),
        Ey: wp.array(dtype=wp.float64),
        Ez: wp.array(dtype=wp.float64),
        Bx: wp.array(dtype=wp.float64),
        By: wp.array(dtype=wp.float64),
        Bz: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        if i < x.shape[0]:
            # Cylindrical conversion
            xj = x[i]
            yj = y[i]
            zj = z[i]
            rj = wp.sqrt(xj * xj + yj * yj)
            if rj != wp.float64(0.0):
                invr = wp.float64(1.0) / rj
                cos = xj * invr
                sin = yj * invr
            else:
                cos = wp.float64(1.0)
                sin = wp.float64(0.0)
            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - wp.float64(0.5)
            z_cell = invdz * (zj - zmin) - wp.float64(0.5)

            ex = wp.float64(0.0)
            ey = wp.float64(0.0)
            ez = wp.float64(0.0)
            bx = wp.float64(0.0)
            by = wp.float64(0.0)
            bz = wp.float64(0.0)

            # Only perform gathering for particles that are below rmax_gather
            if rj < rmax_gather:
                ir_lower = wp.int32(wp.floor(r_cell))
                ir_upper = ir_lower + 1
                iz_lower = wp.int32(wp.floor(z_cell))
                iz_upper = iz_lower + 1
                # Linear weights
                Sr_lower = wp.float64(ir_upper) - r_cell
                Sr_upper = r_cell - wp.float64(ir_lower)
                Sz_lower = wp.float64(iz_upper) - z_cell
                Sz_upper = z_cell - wp.float64(iz_lower)
                Sr_guard = wp.float64(0.0)
                # Guard cells in lower r
                if ir_lower < 0:
                    Sr_guard = Sr_lower
                    Sr_lower = wp.float64(0.0)
                    ir_lower = wp.int32(0)
                # Absorbing in upper r
                if ir_lower > Nr - 1:
                    ir_lower = Nr - 1
                if ir_upper > Nr - 1:
                    ir_upper = Nr - 1
                # Periodic boundaries in z
                if iz_lower < 0:
                    iz_lower += Nz
                if iz_upper < 0:
                    iz_upper += Nz
                if iz_lower > Nz - 1:
                    iz_lower -= Nz
                if iz_upper > Nz - 1:
                    iz_upper -= Nz
                S_ll = Sz_lower * Sr_lower
                S_lu = Sz_lower * Sr_upper
                S_ul = Sz_upper * Sr_lower
                S_uu = Sz_upper * Sr_upper
                S_lg = Sz_lower * Sr_guard
                S_ug = Sz_upper * Sr_guard

                # exp(-i m theta), advanced to the first mode of the group
                phase_step = wp.vec2d(cos, -sin)
                phase = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
                for _ in range(wp.static(first_mode)):
                    phase = cmul(phase, phase_step)
                # Compile-time bounds unroll the modes of the group.
                for local_mode in range(wp.static(mode_count)):
                    mode = select_mode(fields, local_mode)
                    m = wp.static(first_mode) + local_mode
                    Fr, Ft, Fz = add_linear_gather_for_mode(
                        m,
                        wp.float64(0.0),
                        wp.float64(0.0),
                        wp.float64(0.0),
                        phase,
                        mode.Er,
                        mode.Et,
                        mode.Ez,
                        iz_lower,
                        iz_upper,
                        ir_lower,
                        ir_upper,
                        S_ll,
                        S_lu,
                        S_lg,
                        S_ul,
                        S_uu,
                        S_ug,
                    )
                    ex += cos * Fr - sin * Ft
                    ey += sin * Fr + cos * Ft
                    ez += Fz
                    Fr, Ft, Fz = add_linear_gather_for_mode(
                        m,
                        wp.float64(0.0),
                        wp.float64(0.0),
                        wp.float64(0.0),
                        phase,
                        mode.Br,
                        mode.Bt,
                        mode.Bz,
                        iz_lower,
                        iz_upper,
                        ir_lower,
                        ir_upper,
                        S_ll,
                        S_lu,
                        S_lg,
                        S_ul,
                        S_uu,
                        S_ug,
                    )
                    bx += cos * Fr - sin * Ft
                    by += sin * Fr + cos * Ft
                    bz += Fz
                    phase = cmul(phase, phase_step)

            if wp.static(accumulate):
                Ex[i] += ex
                Ey[i] += ey
                Ez[i] += ez
                Bx[i] += bx
                By[i] += by
                Bz[i] += bz
            else:
                Ex[i] = ex
                Ey[i] = ey
                Ez[i] = ez
                Bx[i] = bx
                By[i] = by
                Bz[i] = bz

    gather.key = f"fused_linear_gather_m{first_mode}_n{mode_count}"
    return gather


def gather_fused(particles, grids, rmax_gather: float) -> None:
    """Gather E and B at the particles from all linear-shape mode grids.

    Args:
        particles: GPU species with FP64 positions and field arrays.
        grids: Interpolation grids for every mode, including guard cells.
        rmax_gather: Particles at this radius or beyond receive zero fields.

    Launches one kernel per group of five modes. No host transfer occurs.
    """
    if particles.Ntot == 0:
        return
    grid0 = grids[0]
    positions = [
        as_warp_array(array, wp.float64)
        for array in (particles.x, particles.y, particles.z)
    ]
    scalars = [
        float(rmax_gather),
        float(grid0.invdz),
        float(grid0.zmin),
        int(grid0.Nz),
        float(grid0.invdr),
        float(grid0.rmin),
        int(grid0.Nr),
    ]
    outputs = [
        as_warp_array(getattr(particles, name), wp.float64)
        for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
    ]
    for first_mode in range(0, len(grids), MAX_GROUP_MODES):
        group = grids[first_mode : first_mode + MAX_GROUP_MODES]
        launch_kernel(
            make_kernel(first_mode, len(group)),
            (particles.Ntot,),
            positions + scalars + [group_fields(group)] + outputs,
            particles.gather_tpb,
        )
