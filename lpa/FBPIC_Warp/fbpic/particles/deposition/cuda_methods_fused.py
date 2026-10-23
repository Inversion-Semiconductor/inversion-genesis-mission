# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Fuse sorted, linear FP64 charge and current deposition across modes.

One thread traverses one sorted cell in reverse particle order. Charge fuses
groups of up to five modes. Current uses two-mode groups to limit register
pressure. Arrays remain caller-owned. No device-to-host transfer occurs in
dispatch.

An opt-in variant (``make_lane_kernel``) splits each cell over several adjacent
lanes, so that the particle loads of one warp use whole sectors.
"""

from functools import cache

import warp as wp
from scipy.constants import c

from fbpic.particles.deposition.cuda_particle_shapes import Sr_linear, Sz_linear
from fbpic.utils.warp import as_warp_array, cmul, launch_kernel

LIGHT_SPEED = wp.constant(wp.float64(c))

# Warp has no arrays of arrays, so the output grids and Ruyten coefficients of
# one mode group are passed as structs with a fixed field per slot: up to five
# modes, and up to 15 output components (Jr, Jt, Jz for five modes). Unused
# slots repeat the last array; the kernel never reads them.
CHARGE_GROUP_MODES = 5
CURRENT_GROUP_MODES = 2


@wp.struct
class OutputBuffers:
    component0: wp.array2d(dtype=wp.vec2d)
    component1: wp.array2d(dtype=wp.vec2d)
    component2: wp.array2d(dtype=wp.vec2d)
    component3: wp.array2d(dtype=wp.vec2d)
    component4: wp.array2d(dtype=wp.vec2d)
    component5: wp.array2d(dtype=wp.vec2d)
    component6: wp.array2d(dtype=wp.vec2d)
    component7: wp.array2d(dtype=wp.vec2d)
    component8: wp.array2d(dtype=wp.vec2d)
    component9: wp.array2d(dtype=wp.vec2d)
    component10: wp.array2d(dtype=wp.vec2d)
    component11: wp.array2d(dtype=wp.vec2d)
    component12: wp.array2d(dtype=wp.vec2d)
    component13: wp.array2d(dtype=wp.vec2d)
    component14: wp.array2d(dtype=wp.vec2d)


@wp.struct
class ModeBetas:
    mode0: wp.array(dtype=wp.float64)
    mode1: wp.array(dtype=wp.float64)
    mode2: wp.array(dtype=wp.float64)
    mode3: wp.array(dtype=wp.float64)
    mode4: wp.array(dtype=wp.float64)


def output_buffers(arrays):
    """Borrow complex128 output arrays in field-major, mode-minor order."""
    buffers = OutputBuffers()
    for index in range(15):
        setattr(
            buffers,
            f"component{index}",
            as_warp_array(arrays[min(index, len(arrays) - 1)], wp.vec2d),
        )
    return buffers


def mode_betas(arrays):
    """Borrow each selected mode's float64 Ruyten coefficients."""
    betas = ModeBetas()
    for index in range(5):
        setattr(
            betas,
            f"mode{index}",
            as_warp_array(arrays[min(index, len(arrays) - 1)], wp.float64),
        )
    return betas


@wp.func
def beta_value(betas: ModeBetas, mode: int, radial_bin: int):
    value = wp.float64(0.0)
    if mode == 0:
        value = betas.mode0[radial_bin]
    elif mode == 1:
        value = betas.mode1[radial_bin]
    elif mode == 2:
        value = betas.mode2[radial_bin]
    elif mode == 3:
        value = betas.mode3[radial_bin]
    else:
        value = betas.mode4[radial_bin]
    return value


@wp.func
def atomic_output(
    outputs: OutputBuffers, component: int, iz: int, ir: int, value: wp.vec2d
):
    if component == 0:
        wp.atomic_add(outputs.component0, iz, ir, value)
    elif component == 1:
        wp.atomic_add(outputs.component1, iz, ir, value)
    elif component == 2:
        wp.atomic_add(outputs.component2, iz, ir, value)
    elif component == 3:
        wp.atomic_add(outputs.component3, iz, ir, value)
    elif component == 4:
        wp.atomic_add(outputs.component4, iz, ir, value)
    elif component == 5:
        wp.atomic_add(outputs.component5, iz, ir, value)
    elif component == 6:
        wp.atomic_add(outputs.component6, iz, ir, value)
    elif component == 7:
        wp.atomic_add(outputs.component7, iz, ir, value)
    elif component == 8:
        wp.atomic_add(outputs.component8, iz, ir, value)
    elif component == 9:
        wp.atomic_add(outputs.component9, iz, ir, value)
    elif component == 10:
        wp.atomic_add(outputs.component10, iz, ir, value)
    elif component == 11:
        wp.atomic_add(outputs.component11, iz, ir, value)
    elif component == 12:
        wp.atomic_add(outputs.component12, iz, ir, value)
    elif component == 13:
        wp.atomic_add(outputs.component13, iz, ir, value)
    else:
        wp.atomic_add(outputs.component14, iz, ir, value)


@cache
def make_kernel(current: bool, first_mode: int, mode_count: int):
    """Return a forward-only kernel with compile-time mode groups.

    Args:
        current: Select Jr, Jt, and Jz instead of rho.
        first_mode: First global mode index in this group.
        mode_count: Number of adjacent modes, between one and five.

    Returns:
        A kernel that adds cell sums into complex128 output arrays.
    """
    if first_mode < 0 or not 1 <= mode_count <= CHARGE_GROUP_MODES:
        raise ValueError("A mode group needs a non-negative start and 1-5 modes.")
    field_count = 3 if current else 1
    component_count = mode_count * field_count
    cell_values_type = wp.types.matrix(shape=(4 * component_count, 2), dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def deposit(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        w: wp.array(dtype=wp.float64),
        q: wp.float64,
        ux: wp.array(dtype=wp.float64),
        uy: wp.array(dtype=wp.float64),
        uz: wp.array(dtype=wp.float64),
        inv_gamma: wp.array(dtype=wp.float64),
        invdz: wp.float64,
        zmin: wp.float64,
        nz: int,
        invdr: wp.float64,
        rmin: wp.float64,
        nr: int,
        outputs: OutputBuffers,
        prefix: wp.array(dtype=wp.int32),
        betas: ModeBetas,
    ):
        cell = wp.tid()
        if cell < prefix.shape[0]:
            end = prefix[cell]
            start = 0
            if cell > 0:
                start = prefix[cell - 1]
            values = cell_values_type()
            for offset in range(end - start):
                particle = end - 1 - offset
                particle_x = x[particle]
                particle_y = y[particle]
                radius = wp.sqrt(particle_x * particle_x + particle_y * particle_y)
                cosine = wp.float64(1.0)
                sine = wp.float64(0.0)
                if radius != wp.float64(0.0):
                    inverse_radius = wp.float64(1.0) / radius
                    cosine = particle_x * inverse_radius
                    sine = particle_y * inverse_radius
                radial_position = invdr * (radius - rmin) - wp.float64(0.5)
                axial_position = invdz * (z[particle] - zmin) - wp.float64(0.5)
                beta_index = min(int(wp.ceil(radial_position)), nr)
                weight = q * w[particle]
                axial0 = Sz_linear(axial_position, 0)
                axial1 = Sz_linear(axial_position, 1)
                velocity_r = wp.float64(0.0)
                velocity_t = wp.float64(0.0)
                velocity_z = wp.float64(0.0)
                current_weight = wp.float64(0.0)
                if wp.static(current):
                    velocity_r = cosine * ux[particle] + sine * uy[particle]
                    velocity_t = cosine * uy[particle] - sine * ux[particle]
                    velocity_z = uz[particle]
                    current_weight = weight * LIGHT_SPEED * inv_gamma[particle]
                phase_step = wp.vec2d(cosine, sine)
                phase = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
                for _ in range(wp.static(first_mode)):
                    phase = cmul(phase, phase_step)
                # These compile-time bounds unroll mode and component selection.
                for local_mode in range(wp.static(mode_count)):
                    selected_mode = wp.static(first_mode) + local_mode
                    coefficient = beta_value(betas, local_mode, beta_index)
                    parity = wp.float64(1 - 2 * (selected_mode % 2))
                    for field in range(wp.static(field_count)):
                        flip = parity
                        scale = weight * phase
                        if wp.static(current):
                            velocity = velocity_z
                            if field == 0:
                                velocity = velocity_r
                            elif field == 1:
                                velocity = velocity_t
                            if field < 2:
                                flip = -flip
                            scale = current_weight * velocity * phase
                        radial0 = Sr_linear(radial_position, 0, flip, coefficient)
                        radial1 = Sr_linear(radial_position, 1, flip, coefficient)
                        component = field * wp.static(mode_count) + local_mode
                        values[4 * component] += radial0 * axial0 * scale
                        values[4 * component + 1] += radial1 * axial0 * scale
                        values[4 * component + 2] += radial0 * axial1 * scale
                        values[4 * component + 3] += radial1 * axial1 * scale
                    phase = cmul(phase, phase_step)
            if end > start:
                upper_z = cell // (nr + 1)
                upper_r = cell % (nr + 1)
                for component in range(wp.static(component_count)):
                    for dz in range(2):
                        physical_z = upper_z - 1 + dz
                        if physical_z < 0:
                            physical_z += nz
                        for dr in range(2):
                            physical_r = min(upper_r - 1 + dr, nr - 1)
                            if physical_r < 0:
                                physical_r = -(1 + physical_r)
                            atomic_output(
                                outputs,
                                component,
                                physical_z,
                                physical_r,
                                values[4 * component + 2 * dz + dr],
                            )

    deposit.key = (
        f"fused_linear_{'J' if current else 'rho'}_m{first_mode}_n{mode_count}"
    )
    return deposit


# Lanes per cell for make_lane_kernel. The kernel is correct for any count, but
# only divisors of the 32-thread warp keep each group of lanes in one warp.
LANE_COUNTS = (2, 4, 8, 16, 32)


@cache
def make_lane_kernel(current: bool, first_mode: int, mode_count: int, lanes: int):
    """Return a fused kernel in which several adjacent lanes share one cell.

    The body matches ``make_kernel``. Only the work split differs: lane ``g`` of
    a cell takes the particles at offsets ``g``, ``g + lanes``, and so on. The
    lanes of one cell therefore read adjacent particles, so one warp load uses
    whole sectors instead of one double per sector. Each lane adds its own
    partial sums with atomics, so a cell gets up to ``lanes`` times the atomics
    of ``make_kernel``.

    Args:
        current: Select Jr, Jt, and Jz instead of rho.
        first_mode: First global mode index in this group.
        mode_count: Number of adjacent modes, between one and five.
        lanes: Lanes per cell; one of ``LANE_COUNTS``.

    Returns:
        A kernel for ``cells * lanes`` threads that adds cell sums into
        complex128 output arrays.

    Raises:
        ValueError: The mode group or the lane count is not supported.
    """
    if first_mode < 0 or not 1 <= mode_count <= CHARGE_GROUP_MODES:
        raise ValueError("A mode group needs a non-negative start and 1-5 modes.")
    if lanes not in LANE_COUNTS:
        raise ValueError(f"lanes must be one of {LANE_COUNTS}, got {lanes!r}")
    field_count = 3 if current else 1
    component_count = mode_count * field_count
    cell_values_type = wp.types.matrix(shape=(4 * component_count, 2), dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def deposit(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        w: wp.array(dtype=wp.float64),
        q: wp.float64,
        ux: wp.array(dtype=wp.float64),
        uy: wp.array(dtype=wp.float64),
        uz: wp.array(dtype=wp.float64),
        inv_gamma: wp.array(dtype=wp.float64),
        invdz: wp.float64,
        zmin: wp.float64,
        nz: int,
        invdr: wp.float64,
        rmin: wp.float64,
        nr: int,
        outputs: OutputBuffers,
        prefix: wp.array(dtype=wp.int32),
        betas: ModeBetas,
    ):
        thread = wp.tid()
        # Thread indices are non-negative, so floor and truncation agree here.
        cell = thread // wp.static(lanes)
        lane = thread - cell * wp.static(lanes)
        if cell < prefix.shape[0]:
            end = prefix[cell]
            start = 0
            if cell > 0:
                start = prefix[cell - 1]
            values = cell_values_type()
            for offset in range(lane, end - start, wp.static(lanes)):
                particle = end - 1 - offset
                particle_x = x[particle]
                particle_y = y[particle]
                radius = wp.sqrt(particle_x * particle_x + particle_y * particle_y)
                cosine = wp.float64(1.0)
                sine = wp.float64(0.0)
                if radius != wp.float64(0.0):
                    inverse_radius = wp.float64(1.0) / radius
                    cosine = particle_x * inverse_radius
                    sine = particle_y * inverse_radius
                radial_position = invdr * (radius - rmin) - wp.float64(0.5)
                axial_position = invdz * (z[particle] - zmin) - wp.float64(0.5)
                beta_index = min(int(wp.ceil(radial_position)), nr)
                weight = q * w[particle]
                axial0 = Sz_linear(axial_position, 0)
                axial1 = Sz_linear(axial_position, 1)
                velocity_r = wp.float64(0.0)
                velocity_t = wp.float64(0.0)
                velocity_z = wp.float64(0.0)
                current_weight = wp.float64(0.0)
                if wp.static(current):
                    velocity_r = cosine * ux[particle] + sine * uy[particle]
                    velocity_t = cosine * uy[particle] - sine * ux[particle]
                    velocity_z = uz[particle]
                    current_weight = weight * LIGHT_SPEED * inv_gamma[particle]
                phase_step = wp.vec2d(cosine, sine)
                phase = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
                for _ in range(wp.static(first_mode)):
                    phase = cmul(phase, phase_step)
                for local_mode in range(wp.static(mode_count)):
                    selected_mode = wp.static(first_mode) + local_mode
                    coefficient = beta_value(betas, local_mode, beta_index)
                    parity = wp.float64(1 - 2 * (selected_mode % 2))
                    for field in range(wp.static(field_count)):
                        flip = parity
                        scale = weight * phase
                        if wp.static(current):
                            velocity = velocity_z
                            if field == 0:
                                velocity = velocity_r
                            elif field == 1:
                                velocity = velocity_t
                            if field < 2:
                                flip = -flip
                            scale = current_weight * velocity * phase
                        radial0 = Sr_linear(radial_position, 0, flip, coefficient)
                        radial1 = Sr_linear(radial_position, 1, flip, coefficient)
                        component = field * wp.static(mode_count) + local_mode
                        values[4 * component] += radial0 * axial0 * scale
                        values[4 * component + 1] += radial1 * axial0 * scale
                        values[4 * component + 2] += radial0 * axial1 * scale
                        values[4 * component + 3] += radial1 * axial1 * scale
                    phase = cmul(phase, phase_step)
            # A lane without particles has nothing to add.
            if lane < end - start:
                upper_z = cell // (nr + 1)
                upper_r = cell % (nr + 1)
                for component in range(wp.static(component_count)):
                    for dz in range(2):
                        physical_z = upper_z - 1 + dz
                        if physical_z < 0:
                            physical_z += nz
                        for dr in range(2):
                            physical_r = min(upper_r - 1 + dr, nr - 1)
                            if physical_r < 0:
                                physical_r = -(1 + physical_r)
                            atomic_output(
                                outputs,
                                component,
                                physical_z,
                                physical_r,
                                values[4 * component + 2 * dz + dr],
                            )

    deposit.key = (
        f"fused_linear_{'J' if current else 'rho'}_m{first_mode}_n{mode_count}"
        f"_lanes{lanes}"
    )
    return deposit


def deposit_fused(particles, grids, fieldtype: str, weight) -> None:
    """Add sorted particles to caller-owned complex128 mode grids.

    Args:
        particles: GPU species with sorted FP64 particle arrays and prefix sums.
        grids: Interpolation grids for every mode, including guard cells.
        fieldtype: Select ``'rho'`` or ``'J'``.
        weight: Effective FP64 weights after sorting, including ionization levels.

    Charge launches one kernel per group of five modes. Current launches one
    kernel per group of two modes, for example (0, 1), (2, 3), and (4) for five
    modes. This function borrows GPU arrays and performs no host transfers.

    Two optional species attributes select experimental launch settings. They
    are absent on species built without ``Particles.__init__``, which then use
    the defaults.

    - ``deposit_lanes``: lanes per cell for each field type, for example
      ``{"rho": 1, "J": 4}``. One lane uses ``make_kernel``; more lanes use
      ``make_lane_kernel``.
    - ``deposit_tpb_single_mode``: threads per block for a group of one mode,
      for example the last current group (4) for five modes. None uses
      ``deposit_tpb``.
    """
    grid0 = grids[0]
    arguments = [
        as_warp_array(array, wp.float64)
        for array in (particles.x, particles.y, particles.z, weight)
    ]
    arguments.append(float(particles.q))
    arguments.extend(
        as_warp_array(array, wp.float64)
        for array in (particles.ux, particles.uy, particles.uz, particles.inv_gamma)
    )
    arguments.extend(
        (
            float(grid0.invdz),
            float(grid0.zmin),
            int(grid0.Nz),
            float(grid0.invdr),
            float(grid0.rmin),
            int(grid0.Nr),
        )
    )
    prefix = as_warp_array(particles.prefix_sum, wp.int32)
    current = fieldtype == "J"
    group_size = CURRENT_GROUP_MODES if current else CHARGE_GROUP_MODES
    fields = ("Jr", "Jt", "Jz") if current else ("rho",)
    lanes = getattr(particles, "deposit_lanes", {}).get(fieldtype, 1)
    single_mode_tpb = getattr(particles, "deposit_tpb_single_mode", None)
    for first_mode in range(0, len(grids), group_size):
        selected_grids = grids[first_mode : first_mode + group_size]
        mode_count = len(selected_grids)
        outputs = [getattr(grid, field) for field in fields for grid in selected_grids]
        betas = [grid.d_ruyten_linear_coef for grid in selected_grids]
        if lanes == 1:
            kernel = make_kernel(current, first_mode, mode_count)
        else:
            kernel = make_lane_kernel(current, first_mode, mode_count, lanes)
        tpb = particles.deposit_tpb
        if mode_count == 1 and single_mode_tpb is not None:
            tpb = single_mode_tpb
        launch_kernel(
            kernel,
            (particles.prefix_sum.shape[0] * lanes,),
            arguments + [output_buffers(outputs), prefix, mode_betas(betas)],
            tpb,
        )
