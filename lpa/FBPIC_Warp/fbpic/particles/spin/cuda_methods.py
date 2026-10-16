# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is for the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It carries out the spin-push related methods for all particles
on GPU using CUDA.
"""
from scipy.constants import e
import warp as wp
from fbpic.utils.warp import warp_kernel

# Import inline function
from .warp_inline_functions import push_s_BMT, copy_ionized_electron_spin_batch

e = wp.constant(wp.float64(e))


@warp_kernel
def push_s_gpu(
    sx: wp.array(dtype=wp.float64),
    sy: wp.array(dtype=wp.float64),
    sz: wp.array(dtype=wp.float64),
    ux_prev: wp.array(dtype=wp.float64),
    uy_prev: wp.array(dtype=wp.float64),
    uz_prev: wp.array(dtype=wp.float64),
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    q: wp.float64,
    m: wp.float64,
    Ntot: int,
    dt: wp.float64,
    anom: wp.float64,
):
    """
    Advance particle's spin vector by dt as it precesses around fields.

    For a given timestep n, we expect the variables :
    s_prev, u_prev                      to be defined at time n-1/2
    E, B                                to be defined at time n
    s, u                                to be defined at time n+1/2
    """
    # Define constant pre-factor of Omega (a B-field like vector in
    # the B.M.T. equation.
    tauconst = dt * q / (wp.float64(2.0) * m)

    # Cuda 1D grid (position of current thread)
    ip = wp.tid()

    # Loop over the particles
    if ip < Ntot:
        sx_new, sy_new, sz_new = push_s_BMT(
            sx[ip],
            sy[ip],
            sz[ip],
            ux_prev[ip],
            uy_prev[ip],
            uz_prev[ip],
            ux[ip],
            uy[ip],
            uz[ip],
            Ex[ip],
            Ey[ip],
            Ez[ip],
            Bx[ip],
            By[ip],
            Bz[ip],
            tauconst,
            anom,
        )
        sx[ip] = sx_new
        sy[ip] = sy_new
        sz[ip] = sz_new


@warp_kernel
def push_s_ioniz_gpu(
    sx: wp.array(dtype=wp.float64),
    sy: wp.array(dtype=wp.float64),
    sz: wp.array(dtype=wp.float64),
    ux_prev: wp.array(dtype=wp.float64),
    uy_prev: wp.array(dtype=wp.float64),
    uz_prev: wp.array(dtype=wp.float64),
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    m: wp.float64,
    Ntot: int,
    dt: wp.float64,
    anom: wp.float64,
    ionization_level: wp.array(dtype=wp.uint64),
):
    """
    Advance particle's spin vector by dt as it precesses around fields.
    This takes into account that the particles are ionizable, and thus
    that their charge is determined by `ionization_level`

    Parameters
    ----------
    ionization_level : 1darray of ints
        The number of electrons that each ion is missing
        (compared to a neutral atom)

    For the other parameters, see the docstring of push_p_gpu
    """
    # Cuda 1D grid (position of current thread)
    ip = wp.tid()

    # Loop over the particles
    if ip < Ntot:
        if ionization_level[ip] != wp.uint64(0):
            # Set a few constants
            tauconst = wp.float64(ionization_level[ip]) * dt * e / (wp.float64(2.0) * m)
            # push spin
            sx_new, sy_new, sz_new = push_s_BMT(
                sx[ip],
                sy[ip],
                sz[ip],
                ux_prev[ip],
                uy_prev[ip],
                uz_prev[ip],
                ux[ip],
                uy[ip],
                uz[ip],
                Ex[ip],
                Ey[ip],
                Ez[ip],
                Bx[ip],
                By[ip],
                Bz[ip],
                tauconst,
                anom,
            )
            sx[ip] = sx_new
            sy[ip] = sy_new
            sz[ip] = sz_new


@warp_kernel
def copy_ionized_electron_spin_cuda(
    N_batch: int,
    batch_size: int,
    elec_old_Ntot: int,
    ion_Ntot: int,
    store_electrons_per_level: bool,
    cumulative_n_ionized: wp.array2d(dtype=wp.int64),
    i_level: int,
    ionized_from: wp.array(dtype=wp.int16),
    elec_sx: wp.array(dtype=wp.float64),
    elec_sy: wp.array(dtype=wp.float64),
    elec_sz: wp.array(dtype=wp.float64),
    ion_sx: wp.array(dtype=wp.float64),
    ion_sy: wp.array(dtype=wp.float64),
    ion_sz: wp.array(dtype=wp.float64),
    rand_sx: wp.array(dtype=wp.float64),
    rand_sy: wp.array(dtype=wp.float64),
    rand_sz: wp.array(dtype=wp.float64),
):
    """
    Generate spins for newly generated electrons.
    For the first ionized electron, the spin is copied
    from the parent ion's spin vector. For all subsequent
    ionized electrons, the spin vector is randomly oriented,
    ie sampled from sphere point picking.
    """
    # And now create the spins
    i_batch = wp.tid()
    if i_batch < N_batch:
        copy_ionized_electron_spin_batch(
            i_batch,
            batch_size,
            elec_old_Ntot,
            ion_Ntot,
            cumulative_n_ionized,
            i_level,
            ionized_from,
            store_electrons_per_level,
            elec_sx,
            elec_sy,
            elec_sz,
            ion_sx,
            ion_sy,
            ion_sz,
            rand_sx,
            rand_sy,
            rand_sz,
        )
