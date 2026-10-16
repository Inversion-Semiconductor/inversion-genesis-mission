# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import warp_kernel

from scipy.constants import c, e
from .cuda_inline_functions import push_p_vay

wp.set_module_options({"enable_backward": False})


c = wp.constant(wp.float64(c))
e = wp.constant(wp.float64(e))


@warp_kernel
def push_x_gpu(
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
):
    """
    Advance the particles' positions over `dt` using the momenta ux, uy, uz,
    multiplied by the scalar coefficients x_push, y_push, z_push.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    dt : float (seconds)
        The timestep by which the position is advanced

    x_push, y_push, z_push: float, dimensionless
        Multiplying coefficient for the momenta in x, y and z
        e.g. if x_push=1., the particles are pushed forward in x
             if x_push=-1., the particles are pushed backward in x
    """
    # Timestep multiplied by c
    cdt = c * dt

    i = wp.tid()
    if i < x.shape[0]:
        # Particle push
        inv_g = inv_gamma[i]
        x[i] += cdt * x_push * inv_g * ux[i]
        y[i] += cdt * y_push * inv_g * uy[i]
        z[i] += cdt * z_push * inv_g * uz[i]


@warp_kernel
def push_p_gpu(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    q: wp.float64,
    m: wp.float64,
    Ntot: wp.int32,
    dt: wp.float64,
):
    """
    Advance the particles' momenta, using Warp on the GPU

    Parameters
    ----------
    ux, uy, uz : 1darray of floats
        The velocity of the particles
        (is modified by this function)

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles

    q : float
        The charge of the particle species

    m : float
        The mass of the particle species

    Ntot : int
        The total number of particles

    dt : float
        The time by which the momenta is advanced
    """
    # Set a few constants
    econst = q * dt / (m * c)
    bconst = wp.float64(0.5) * q * dt / m

    # Cuda 1D grid
    ip = wp.tid()

    # Loop over the particles
    if ip < Ntot:
        ux_new, uy_new, uz_new, inv_gamma_new = push_p_vay(
            ux[ip],
            uy[ip],
            uz[ip],
            inv_gamma[ip],
            Ex[ip],
            Ey[ip],
            Ez[ip],
            Bx[ip],
            By[ip],
            Bz[ip],
            econst,
            bconst,
        )
        ux[ip] = ux_new
        uy[ip] = uy_new
        uz[ip] = uz_new
        inv_gamma[ip] = inv_gamma_new


@warp_kernel
def push_p_after_plane_gpu(
    z: wp.array(dtype=wp.float64),
    z_plane: wp.float64,
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    q: wp.float64,
    m: wp.float64,
    Ntot: wp.int32,
    dt: wp.float64,
):
    """
    Advance the particles' momenta, using Warp on the GPU.
    Only the particles that are located beyond the plane z=z_plane
    have their momentum modified ; the others particles move ballistically.

    Parameters
    ----------
    z: 1darray of floats
        The position of the particles in the z direction

    z_plane: float
        Position beyond which the particles should be

    For the other parameters, see the docstring of push_p_gpu
    """
    # Set a few constants
    econst = q * dt / (m * c)
    bconst = wp.float64(0.5) * q * dt / m

    # Cuda 1D grid
    ip = wp.tid()

    # Loop over the particles
    if ip < Ntot and z[ip] > z_plane:
        ux_new, uy_new, uz_new, inv_gamma_new = push_p_vay(
            ux[ip],
            uy[ip],
            uz[ip],
            inv_gamma[ip],
            Ex[ip],
            Ey[ip],
            Ez[ip],
            Bx[ip],
            By[ip],
            Bz[ip],
            econst,
            bconst,
        )
        ux[ip] = ux_new
        uy[ip] = uy_new
        uz[ip] = uz_new
        inv_gamma[ip] = inv_gamma_new


@warp_kernel
def push_p_ioniz_gpu(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    m: wp.float64,
    Ntot: wp.int32,
    dt: wp.float64,
    ionization_level: wp.array(dtype=wp.uint64),
):
    """
    Advance the particles' momenta, using Warp on the GPU
    This take into account that the particles are ionizable, and thus
    that their charge is determined by `ionization_level`

    Parameters
    ----------
    ionization_level : 1darray of ints
        The number of electrons that each ion is missing
        (compared to a neutral atom)

    For the other parameters, see the docstring of push_p_gpu
    """
    # Cuda 1D grid
    ip = wp.tid()

    # Loop over the particles
    if ip < Ntot:
        if ionization_level[ip] != 0:
            # Set a few constants
            econst = wp.float64(ionization_level[ip]) * e * dt / (m * c)
            bconst = wp.float64(0.5) * wp.float64(ionization_level[ip]) * e * dt / m
            # Use the Vay pusher
            ux_new, uy_new, uz_new, inv_gamma_new = push_p_vay(
                ux[ip],
                uy[ip],
                uz[ip],
                inv_gamma[ip],
                Ex[ip],
                Ey[ip],
                Ez[ip],
                Bx[ip],
                By[ip],
                Bz[ip],
                econst,
                bconst,
            )
            ux[ip] = ux_new
            uy[ip] = uy_new
            uz[ip] = uz_new
            inv_gamma[ip] = inv_gamma_new
