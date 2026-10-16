"""Out-of-place FP64 particle kernels for eager reverse differentiation.

Floating inputs and outputs must use distinct allocations. Grid geometry,
timestep, push coefficients, species properties, and ion levels stay fixed.
Production kernels and their forward-only module options remain unchanged.
"""

from functools import cache

import warp as wp
from scipy.constants import c, e

from fbpic.particles.push.cuda_inline_functions import push_p_vay

LIGHT_SPEED = wp.constant(wp.float64(c))
ELEMENTARY_CHARGE = wp.constant(wp.float64(e))


@wp.kernel(enable_backward=True, module="unique")
def push_positions(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    dt: wp.float64,
    x_push: wp.float64,
    y_push: wp.float64,
    z_push: wp.float64,
    x_out: wp.array(dtype=wp.float64),
    y_out: wp.array(dtype=wp.float64),
    z_out: wp.array(dtype=wp.float64),
):
    """Write advanced positions in meters without overwriting particle inputs."""
    particle = wp.tid()
    if particle < x.shape[0]:
        cdt = LIGHT_SPEED * dt
        inverse_gamma = inv_gamma[particle]
        x_out[particle] = x[particle] + cdt * x_push * inverse_gamma * ux[particle]
        y_out[particle] = y[particle] + cdt * y_push * inverse_gamma * uy[particle]
        z_out[particle] = z[particle] + cdt * z_push * inverse_gamma * uz[particle]


@wp.kernel(enable_backward=True, module="unique")
def push_momenta(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    electric_x: wp.array(dtype=wp.float64),
    electric_y: wp.array(dtype=wp.float64),
    electric_z: wp.array(dtype=wp.float64),
    magnetic_x: wp.array(dtype=wp.float64),
    magnetic_y: wp.array(dtype=wp.float64),
    magnetic_z: wp.array(dtype=wp.float64),
    charge: wp.float64,
    mass: wp.float64,
    dt: wp.float64,
    ux_out: wp.array(dtype=wp.float64),
    uy_out: wp.array(dtype=wp.float64),
    uz_out: wp.array(dtype=wp.float64),
    inv_gamma_out: wp.array(dtype=wp.float64),
):
    """Write Vay momenta and inverse gamma using the production scalar helper."""
    particle = wp.tid()
    if particle < ux.shape[0]:
        electric_coefficient = charge * dt / (mass * LIGHT_SPEED)
        magnetic_coefficient = wp.float64(0.5) * charge * dt / mass
        ux_new, uy_new, uz_new, inverse_gamma_new = push_p_vay(
            ux[particle],
            uy[particle],
            uz[particle],
            inv_gamma[particle],
            electric_x[particle],
            electric_y[particle],
            electric_z[particle],
            magnetic_x[particle],
            magnetic_y[particle],
            magnetic_z[particle],
            electric_coefficient,
            magnetic_coefficient,
        )
        ux_out[particle] = ux_new
        uy_out[particle] = uy_new
        uz_out[particle] = uz_new
        inv_gamma_out[particle] = inverse_gamma_new


@wp.kernel(enable_backward=True, module="unique")
def push_ion_momenta(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    electric_x: wp.array(dtype=wp.float64),
    electric_y: wp.array(dtype=wp.float64),
    electric_z: wp.array(dtype=wp.float64),
    magnetic_x: wp.array(dtype=wp.float64),
    magnetic_y: wp.array(dtype=wp.float64),
    magnetic_z: wp.array(dtype=wp.float64),
    mass: wp.float64,
    dt: wp.float64,
    ionization_level: wp.array(dtype=wp.uint64),
    ux_out: wp.array(dtype=wp.float64),
    uy_out: wp.array(dtype=wp.float64),
    uz_out: wp.array(dtype=wp.float64),
    inv_gamma_out: wp.array(dtype=wp.float64),
):
    """Write the ion push with fixed levels; neutral particles retain their state."""
    particle = wp.tid()
    if particle < ux.shape[0]:
        ux_new = ux[particle]
        uy_new = uy[particle]
        uz_new = uz[particle]
        inverse_gamma_new = inv_gamma[particle]
        if ionization_level[particle] != wp.uint64(0):
            charge = wp.float64(ionization_level[particle]) * ELEMENTARY_CHARGE
            electric_coefficient = charge * dt / (mass * LIGHT_SPEED)
            magnetic_coefficient = wp.float64(0.5) * charge * dt / mass
            ux_new, uy_new, uz_new, inverse_gamma_new = push_p_vay(
                ux[particle],
                uy[particle],
                uz[particle],
                inv_gamma[particle],
                electric_x[particle],
                electric_y[particle],
                electric_z[particle],
                magnetic_x[particle],
                magnetic_y[particle],
                magnetic_z[particle],
                electric_coefficient,
                magnetic_coefficient,
            )
        ux_out[particle] = ux_new
        uy_out[particle] = uy_new
        uz_out[particle] = uz_new
        inv_gamma_out[particle] = inverse_gamma_new


@cache
def get_gather_kernel() -> wp.Kernel:
    """Return an eager differentiable clone of the five-mode linear gather.

    The kernel signature and FP64 storage match the production gather.
    Particle positions and complex field grids have continuous derivatives.
    Integer cell choices and domain masks stay fixed within each smooth region.
    The exact axis and cell or domain boundaries need separate derivative rules.
    """
    from fbpic.particles.gathering.cuda_methods_fused import (
        gather_field_gpu_linear_five_modes,
    )

    return wp.kernel(
        gather_field_gpu_linear_five_modes.kernel.func,
        enable_backward=True,
        module="unique",
    )
