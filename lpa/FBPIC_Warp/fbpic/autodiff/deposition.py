"""Retain fused FP64 deposition forward execution with a particle-wise VJP.

The cell permutation, prefix sums, grid coefficients, and ionization decisions
are fixed. Explicit output versions preserve initial grid values for backward.
"""

from collections.abc import Sequence
from functools import cache

import warp as wp

from fbpic.particles.deposition.cuda_methods_fused import (
    LIGHT_SPEED,
    ModeBetas,
    OutputBuffers,
    atomic_output,
    beta_value,
    make_kernel,
)
from fbpic.particles.deposition.cuda_particle_shapes import Sr_linear, Sz_linear
from fbpic.utils.warp import cmul


@wp.kernel(enable_backward=False, module="unique")
def particle_cells_from_prefix(
    prefix: wp.array(dtype=wp.int32), cells: wp.array(dtype=wp.int32)
):
    cell = wp.tid()
    if cell < prefix.shape[0]:
        start = wp.int32(0)
        if cell > 0:
            start = prefix[cell - 1]
        for particle in range(start, prefix[cell]):
            cells[particle] = cell


@cache
def make_particle_kernel(current: bool, first_mode: int, mode_count: int):
    """Return a backward-enabled per-particle deposition operator.

    The production forward operator accumulates a dynamic cell-local matrix.
    This equivalent particle expression has only static mode and stencil loops.
    """
    if first_mode < 0 or mode_count < 1 or first_mode + mode_count > 5:
        raise ValueError("Mode groups must lie within modes zero through four.")
    field_count = 3 if current else 1

    @wp.kernel(enable_backward=True, module="unique")
    def particle_deposit(
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
        cells: wp.array(dtype=wp.int32),
        betas: ModeBetas,
    ):
        particle = wp.tid()
        if particle < x.shape[0]:
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
            cell = cells[particle]
            upper_z = cell // (nr + 1)
            upper_r = cell % (nr + 1)
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
                    for dz in range(2):
                        axial = axial0
                        if dz == 1:
                            axial = axial1
                        physical_z = upper_z - 1 + dz
                        if physical_z < 0:
                            physical_z += nz
                        for dr in range(2):
                            radial = radial0
                            if dr == 1:
                                radial = radial1
                            physical_r = min(upper_r - 1 + dr, nr - 1)
                            if physical_r < 0:
                                physical_r = -(1 + physical_r)
                            atomic_output(
                                outputs,
                                component,
                                physical_z,
                                physical_r,
                                radial * axial * scale,
                            )
                phase = cmul(phase, phase_step)

    return particle_deposit


def _structs(grids, betas):
    outputs, coefficients = OutputBuffers(), ModeBetas()
    for index in range(15):
        setattr(outputs, f"component{index}", grids[min(index, len(grids) - 1)])
    for index in range(5):
        setattr(coefficients, f"mode{index}", betas[min(index, len(betas) - 1)])
    return outputs, coefficients


def deposit_five_modes(
    x: wp.array,
    y: wp.array,
    z: wp.array,
    w: wp.array,
    ux: wp.array,
    uy: wp.array,
    uz: wp.array,
    inv_gamma: wp.array,
    initial_grids: Sequence[wp.array],
    betas: Sequence[wp.array],
    prefix: wp.array,
    *,
    q: float,
    invdz: float,
    zmin: float,
    nz: int,
    invdr: float,
    rmin: float,
    nr: int,
    current: bool = False,
    block_dim: int = 32,
) -> list[wp.array]:
    """Return fused grid versions and register their fixed-stencil VJP.

    Args:
        x: Sorted FP64 particle x positions in meters.
        y: Sorted FP64 particle y positions in meters.
        z: Sorted FP64 particle z positions in meters.
        w: FP64 effective particle weights, including fixed ion levels.
        ux: FP64 particle momentum x divided by mass times light speed.
        uy: FP64 particle momentum y divided by mass times light speed.
        uz: FP64 particle momentum z divided by mass times light speed.
        inv_gamma: FP64 inverse relativistic gamma factors.
        initial_grids: Five charge grids or fifteen field-major current grids.
            Complex128 storage uses vec2d. Inputs remain unchanged.
        betas: Five fixed FP64 Ruyten coefficient arrays.
        prefix: Fixed int32 sorted-cell inclusive prefix sums. Values must be
            nondecreasing, end at the particle count, and match the sorted cells.
            The caller validates these values without adding hot-path transfers.
        q: Fixed particle charge in coulombs.
        invdz: Fixed inverse axial cell width in inverse meters.
        zmin: Fixed axial grid origin in meters.
        nz: Fixed axial grid size, including guards.
        invdr: Fixed inverse radial cell width in inverse meters.
        rmin: Fixed radial grid origin in meters.
        nr: Fixed radial grid size.
        current: Deposit currents instead of charge.
        block_dim: Production fused-forward thread block size.

    Returns:
        New Warp-owned grid arrays with gradients enabled. The active Warp tape
        receives a custom backward callback. No host transfer occurs.

    Raises:
        ValueError: The fixed five-mode input layouts are inconsistent.
    """
    count = 15 if current else 5
    if len(initial_grids) != count or len(betas) != 5:
        raise ValueError(
            "Five modes require five charge grids or fifteen current grids."
        )
    device = x.device
    particles = (x, y, z, w, ux, uy, uz, inv_gamma)
    if any(array.shape != x.shape for array in particles):
        raise ValueError("Sorted particle arrays must have identical shapes.")
    if prefix.shape != (nz * (nr + 1),):
        raise ValueError("Prefix sums must cover all axial and radial source cells.")
    for array in (*particles, *betas, prefix, *initial_grids):
        if array.device != device:
            raise ValueError("All deposition arrays must share one device.")
    if any(beta.requires_grad for beta in betas):
        raise ValueError(
            "Ruyten coefficient arrays must remain fixed without gradients."
        )
    for grid in initial_grids:
        if grid.dtype != wp.vec2d or grid.shape != (nz, nr):
            raise ValueError("Each mode grid must use vec2d and shape (nz, nr).")
    outputs = [wp.clone(grid, requires_grad=True) for grid in initial_grids]
    cells = wp.empty(x.shape, dtype=wp.int32, device=device)
    wp.launch(
        particle_cells_from_prefix,
        dim=prefix.shape,
        inputs=[prefix, cells],
        device=device,
        record_tape=False,
    )
    # Warp 1.18 exposes the active tape through its runtime, as wp.copy does.
    tape = wp._src.context.runtime.tape
    arguments = [x, y, z, w, q, ux, uy, uz, inv_gamma, invdz, zmin, nz, invdr, rmin, nr]
    group_size = 2 if current else 5
    for first_mode in range(0, 5, group_size):
        mode_count = min(group_size, 5 - first_mode)
        selected = [
            outputs[field * 5 + mode]
            for field in range(3 if current else 1)
            for mode in range(first_mode, first_mode + mode_count)
        ]
        buffers, coefficients = _structs(
            selected, betas[first_mode : first_mode + mode_count]
        )
        wp.launch(
            make_kernel(current, first_mode, mode_count),
            dim=prefix.shape,
            inputs=arguments + [buffers, prefix, coefficients],
            block_dim=block_dim,
            device=device,
            record_tape=False,
        )
        if tape is not None:
            backward_kernel = make_particle_kernel(current, first_mode, mode_count)
            backward_inputs = arguments + [buffers, cells, coefficients]

            def backward(kernel=backward_kernel, inputs=backward_inputs):
                wp.launch(
                    kernel,
                    dim=x.shape,
                    inputs=inputs,
                    adj_inputs=[tape.get_adjoint(value) for value in inputs],
                    adjoint=True,
                    device=device,
                    record_tape=False,
                )

            tape.record_func(
                backward,
                [array for array in (*particles, *selected) if array.requires_grad],
            )
    return outputs
