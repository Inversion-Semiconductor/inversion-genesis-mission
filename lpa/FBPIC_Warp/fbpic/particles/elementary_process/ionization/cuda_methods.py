# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in particle ionization.

Apart from synthactic details, this file is very close to numba_methods.py
"""
import warp as wp
from fbpic.utils.warp import warp_kernel
from scipy.constants import c

# Import inline functions
from .warp_inline_functions import (
    get_ionization_probability,
    get_E_amplitude,
    copy_ionized_electrons_batch,
)

c = wp.constant(wp.float64(c))


@warp_kernel
def ionize_ions_cuda(
    N_batch: int,
    batch_size: int,
    Ntot: int,
    level_start: int,
    level_max: int,
    n_levels: int,
    n_ionized: wp.array2d(dtype=wp.int64),
    ionized_from: wp.array(dtype=wp.int16),
    ionization_level: wp.array(dtype=wp.uint64),
    random_draw: wp.array(dtype=wp.float32),
    adk_prefactor: wp.array(dtype=wp.float64),
    adk_power: wp.array(dtype=wp.float64),
    adk_exp_prefactor: wp.array(dtype=wp.float64),
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    w_times_level: wp.array(dtype=wp.float64),
):
    """
    For each ion macroparticle, decide whether it is going to
    be further ionized during this timestep, based on the ADK rate.

    Increment the elements in `ionization_level` accordingly, and update
    `w_times_level` of the ions to take into account the change in level
    of the corresponding macroparticle.

    For the purpose of counting and creating the corresponding electrons,
    `ionized_from` (one element per macroparticle) is set to -1 at the position
    of the unionized ions, and to the level (before ionization) otherwise
    `n_ionized` (one element per batch, and per ionizable level that needs
    to be distinguished) counts the total number of ionized particles
    in the current batch.
    """
    # Loop over batches of particles
    i_batch = wp.tid()
    if i_batch < N_batch:

        # Set the count of ionized particles in the batch to 0
        for i_level in range(n_levels):
            n_ionized[i_level, i_batch] = wp.int64(0)

        # Loop through the batch
        N_max = min((i_batch + 1) * batch_size, Ntot)
        for ip in range(i_batch * batch_size, N_max):

            # Skip the ionization routine, if the maximal ionization level
            # has already been reached for this macroparticle
            level = int(ionization_level[ip])
            if level >= level_max:
                ionized_from[ip] = wp.int16(-1)
                continue

            # Calculate the amplitude of the electric field,
            # in the frame of the electrons (device inline function)
            E, gamma = get_E_amplitude(
                ux[ip],
                uy[ip],
                uz[ip],
                Ex[ip],
                Ey[ip],
                Ez[ip],
                c * Bx[ip],
                c * By[ip],
                c * Bz[ip],
            )
            # Get ADK rate (device inline function)
            p = get_ionization_probability(
                E,
                gamma,
                adk_prefactor[level],
                adk_power[level],
                adk_exp_prefactor[level],
            )
            # Ionize particles
            if wp.float64(random_draw[ip]) < p:
                # Set the corresponding flag and update particle count
                ionized_from[ip] = wp.int16(level - level_start)
                if n_levels == 1:
                    # No need to distinguish ionization levels
                    n_ionized[0, i_batch] += wp.int64(1)
                else:
                    # Distinguish count for each ionizable level
                    n_ionized[level - level_start, i_batch] += wp.int64(1)
                # Update the ionization level and the corresponding weight
                ionization_level[ip] += wp.uint64(1)
                w_times_level[ip] = w[ip] * wp.float64(ionization_level[ip])
            else:
                ionized_from[ip] = wp.int16(-1)


@warp_kernel
def copy_ionized_electrons_cuda(
    N_batch: int,
    batch_size: int,
    elec_old_Ntot: int,
    ion_Ntot: int,
    cumulative_n_ionized: wp.array2d(dtype=wp.int64),
    ionized_from: wp.array(dtype=wp.int16),
    i_level: int,
    store_electrons_per_level: bool,
    elec_x: wp.array(dtype=wp.float64),
    elec_y: wp.array(dtype=wp.float64),
    elec_z: wp.array(dtype=wp.float64),
    elec_inv_gamma: wp.array(dtype=wp.float64),
    elec_ux: wp.array(dtype=wp.float64),
    elec_uy: wp.array(dtype=wp.float64),
    elec_uz: wp.array(dtype=wp.float64),
    elec_w: wp.array(dtype=wp.float64),
    elec_Ex: wp.array(dtype=wp.float64),
    elec_Ey: wp.array(dtype=wp.float64),
    elec_Ez: wp.array(dtype=wp.float64),
    elec_Bx: wp.array(dtype=wp.float64),
    elec_By: wp.array(dtype=wp.float64),
    elec_Bz: wp.array(dtype=wp.float64),
    ion_x: wp.array(dtype=wp.float64),
    ion_y: wp.array(dtype=wp.float64),
    ion_z: wp.array(dtype=wp.float64),
    ion_inv_gamma: wp.array(dtype=wp.float64),
    ion_ux: wp.array(dtype=wp.float64),
    ion_uy: wp.array(dtype=wp.float64),
    ion_uz: wp.array(dtype=wp.float64),
    ion_w: wp.array(dtype=wp.float64),
    ion_Ex: wp.array(dtype=wp.float64),
    ion_Ey: wp.array(dtype=wp.float64),
    ion_Ez: wp.array(dtype=wp.float64),
    ion_Bx: wp.array(dtype=wp.float64),
    ion_By: wp.array(dtype=wp.float64),
    ion_Bz: wp.array(dtype=wp.float64),
):
    """
    Create the new electrons by copying the properties (position, momentum,
    etc) of the ions that they originate from.
    """
    # Select the current batch
    i_batch = wp.tid()
    if i_batch < N_batch:
        copy_ionized_electrons_batch(
            i_batch,
            batch_size,
            elec_old_Ntot,
            ion_Ntot,
            cumulative_n_ionized,
            ionized_from,
            i_level,
            store_electrons_per_level,
            elec_x,
            elec_y,
            elec_z,
            elec_inv_gamma,
            elec_ux,
            elec_uy,
            elec_uz,
            elec_w,
            elec_Ex,
            elec_Ey,
            elec_Ez,
            elec_Bx,
            elec_By,
            elec_Bz,
            ion_x,
            ion_y,
            ion_z,
            ion_inv_gamma,
            ion_ux,
            ion_uy,
            ion_uz,
            ion_w,
            ion_Ex,
            ion_Ey,
            ion_Ez,
            ion_Bx,
            ion_By,
            ion_Bz,
        )
