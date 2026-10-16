# Copyright 2023, FBPIC contributors
# Author: Michael J. Quin
# Scientific supervision: Matteo Tamburini
# Code optimization: Kristjan Poder
# License: 3-Clause-BSD-LBNL
"""
This file is for use with the Fourier-Bessel Particle-In-Cell code (FB-PIC).

It defines a function to push the 'classical' spin vector, according to the
Bargmann-Michel-Telegdi equation and other spin-push related functions.
"""
import warp as wp
from scipy.constants import c

c = wp.constant(wp.float64(c))


@wp.func
def push_s_BMT(
    sx_i: wp.float64,
    sy_i: wp.float64,
    sz_i: wp.float64,
    ux_i: wp.float64,
    uy_i: wp.float64,
    uz_i: wp.float64,
    ux_f: wp.float64,
    uy_f: wp.float64,
    uz_f: wp.float64,
    Ex: wp.float64,
    Ey: wp.float64,
    Ez: wp.float64,
    Bx: wp.float64,
    By: wp.float64,
    Bz: wp.float64,
    tauconst: wp.float64,
    anom: wp.float64,
):
    """
    Push spin of a single particle using the 'Tamburini' pusher, as
    detailed in https://arxiv.org/abs/2303.16966
    """
    # Momentum at a half-timestep forward using Boris assumption
    ux = (ux_i + ux_f) / wp.float64(2.0)
    uy = (uy_i + uy_f) / wp.float64(2.0)
    uz = (uz_i + uz_f) / wp.float64(2.0)

    # Assume 1./gamma at midpoint can be calculated from (average)
    # momentum at midpoint
    inv_gamma = wp.float64(1.0) / wp.sqrt(
        wp.float64(1.0)
        + ux ** wp.float64(2)
        + uy ** wp.float64(2)
        + uz ** wp.float64(2)
    )
    # Define parameter: 1./(1+gamma)
    inv_1pgamma = inv_gamma / (wp.float64(1.0) + inv_gamma)
    # Scalar product of momentum and B field
    uB = ux * Bx + uy * By + uz * Bz

    # Define E/B field dependent parameters: tau and upsilon (ups).
    # Tau vector is analogous to the 'tau' (or 't') vector used in
    # the Boris-Vay pusher.
    taux = tauconst * (
        (anom + inv_gamma) * Bx
        - (anom + inv_1pgamma) * (uy * Ez - uz * Ey) * inv_gamma / c
        - anom * inv_gamma * inv_1pgamma * uB * ux
    )
    tauy = tauconst * (
        (anom + inv_gamma) * By
        - (anom + inv_1pgamma) * (uz * Ex - ux * Ez) * inv_gamma / c
        - anom * inv_gamma * inv_1pgamma * uB * uy
    )
    tauz = tauconst * (
        (anom + inv_gamma) * Bz
        - (anom + inv_1pgamma) * (ux * Ey - uy * Ex) * inv_gamma / c
        - anom * inv_gamma * inv_1pgamma * uB * uz
    )

    tau2 = taux ** wp.float64(2) + tauy ** wp.float64(2) + tauz ** wp.float64(2)

    # ---------- Tamburini Method ---------------------------------------------
    ups = wp.float64(1.0) / (wp.float64(1.0) + tau2)
    stau = sx_i * taux + sy_i * tauy + sz_i * tauz

    # New spin components
    sx_f = ups * (
        sx_i
        + wp.float64(2.0) * (sy_i * tauz - sz_i * tauy)
        + stau * taux
        + (sz_i * taux - sx_i * tauz) * tauz
        - (sx_i * tauy - sy_i * taux) * tauy
    )
    sy_f = ups * (
        sy_i
        + wp.float64(2.0) * (sz_i * taux - sx_i * tauz)
        + stau * tauy
        + (sx_i * tauy - sy_i * taux) * taux
        - (sy_i * tauz - sz_i * tauy) * tauz
    )
    sz_f = ups * (
        sz_i
        + wp.float64(2.0) * (sx_i * tauy - sy_i * taux)
        + stau * tauz
        + (sy_i * tauz - sz_i * tauy) * tauy
        - (sz_i * taux - sx_i * tauz) * taux
    )
    # -------------------------------------------------------------------------

    return sx_f, sy_f, sz_f


@wp.func
def copy_ionized_electron_spin_batch(
    i_batch: int,
    batch_size: int,
    elec_old_Ntot: int,
    ion_Ntot: int,
    cumulative_n_ionized: wp.array2d(dtype=wp.int64),
    i_level: int,
    ionized_from: wp.array(dtype=wp.int16),
    store_electrons_per_level: bool,
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
    elec_index = wp.int64(elec_old_Ntot) + cumulative_n_ionized[i_level, i_batch]
    # Loop through the ions in this batch
    N_max = min((i_batch + 1) * batch_size, ion_Ntot)
    for ion_index in range(i_batch * batch_size, N_max):

        # Determine if a new electron should be created from this ion
        create_electron = False
        # If electrons are not distinguished by level, take all ionized ions
        if (not store_electrons_per_level) and int(ionized_from[ion_index]) >= 0:
            create_electron = True
        # If electrons are to be distinguished by level, take only those
        # of the corresponding i_level
        if store_electrons_per_level and int(ionized_from[ion_index]) == i_level:
            create_electron = True

        if create_electron:
            if int(ionized_from[ion_index]) == 0:
                # Copy the ion data to the current electron_index
                elec_sx[elec_index] = ion_sx[ion_index]
                elec_sy[elec_index] = ion_sy[ion_index]
                elec_sz[elec_index] = ion_sz[ion_index]
            else:
                # Or use one of the random spins
                rand_index = elec_index - wp.int64(elec_old_Ntot)
                elec_sx[elec_index] = rand_sx[rand_index]
                elec_sy[elec_index] = rand_sy[rand_index]
                elec_sz[elec_index] = rand_sz[rand_index]

            # Update the electron_index
            elec_index += wp.int64(1)
