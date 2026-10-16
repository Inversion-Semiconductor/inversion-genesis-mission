# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines cuda methods that are used in Compton scattering (on GPU).
"""
import warp as wp
from fbpic.utils.warp import warp_kernel
from ..warp_random import random_uniform

# Import the inline functions
from .warp_inline_functions import (
    lorentz_transform,
    get_scattering_probability,
    get_photon_density_gaussian,
    INV_MC,
)


@warp_kernel
def get_photon_density_gaussian_cuda(
    photon_n: wp.array(dtype=wp.float64),
    elec_Ntot: int,
    elec_x: wp.array(dtype=wp.float64),
    elec_y: wp.array(dtype=wp.float64),
    elec_z: wp.array(dtype=wp.float64),
    ct: wp.float64,
    photon_n_lab_max: wp.float64,
    inv_laser_waist2: wp.float64,
    inv_laser_ctau2: wp.float64,
    laser_initial_z0: wp.float64,
    gamma_boost: wp.float64,
    beta_boost: wp.float64,
):
    """
    Fill the array `photon_n` with the values of the photon density
    (in the simulation frame) in the scattering laser pulse, at
    the position of the electron macroparticles.

    Parameters
    ----------
    elec_x, elec_y, elec_z: 1d arrays of floats
        The position of the electrons (in the frame of the simulation)
    ct: float
        Current time in the simulation frame (multiplied by c)
    photon_n_lab_max: float
        Peak photon density (in the lab frame)
        (i.e. at the peak of the Gaussian pulse)
    inv_laser_waist2, inv_laser_ctau2, laser_initial_z0: floats
        Properties of the Gaussian laser pulse (in the lab frame)
    gamma_boost, beta_boost: floats
        Properties of the Lorentz boost between the lab and simulation frame.
    """
    # Loop over electrons
    i_elec = wp.tid()
    if i_elec < elec_Ntot:

        photon_n[i_elec] = get_photon_density_gaussian(
            elec_x[i_elec],
            elec_y[i_elec],
            elec_z[i_elec],
            ct,
            photon_n_lab_max,
            inv_laser_waist2,
            inv_laser_ctau2,
            laser_initial_z0,
            gamma_boost,
            beta_boost,
        )


@warp_kernel
def determine_scatterings_cuda(
    N_batch: int,
    batch_size: int,
    elec_Ntot: int,
    nscatter_per_elec: wp.array(dtype=wp.int64),
    nscatter_per_batch: wp.array(dtype=wp.int64),
    random_states: wp.array(dtype=wp.uint32),
    dt: wp.float64,
    elec_ux: wp.array(dtype=wp.float64),
    elec_uy: wp.array(dtype=wp.float64),
    elec_uz: wp.array(dtype=wp.float64),
    elec_inv_gamma: wp.array(dtype=wp.float64),
    ratio_w_electron_photon: wp.float64,
    photon_n: wp.array(dtype=wp.float64),
    photon_p: wp.float64,
    photon_beta_x: wp.float64,
    photon_beta_y: wp.float64,
    photon_beta_z: wp.float64,
):
    """
    For each electron macroparticle, decide how many photon macroparticles
    it will emit during `dt`, using the integrated Klein-Nishina formula.

    Electrons are processed in batches of size `batch_size`, with a parallel
    loop over batches. The batching allows quicker calculation of the
    total number of photons to be created.
    """
    # Loop over batches of particles
    i_batch = wp.tid()
    if i_batch < N_batch:

        # Set the count of scattered particles in the batch to 0
        nscatter_per_batch[i_batch] = wp.int64(0)

        # Loop through the batch
        # (Note: a while loop is used here, because numba 0.34 does
        # not support nested prange and range loops)
        N_max = min((i_batch + 1) * batch_size, elec_Ntot)
        ip = i_batch * batch_size
        while ip < N_max:

            # Set the count of scattered photons for this electron to 0
            nscatter_per_elec[ip] = wp.int64(0)

            # For each electron, calculate the probability of scattering
            p = get_scattering_probability(
                dt,
                elec_ux[ip],
                elec_uy[ip],
                elec_uz[ip],
                elec_inv_gamma[ip],
                photon_n[ip],
                photon_p,
                photon_beta_x,
                photon_beta_y,
                photon_beta_z,
            )

            # Determine the number of photons produced by this electron
            r = random_uniform(random_states, i_batch)
            nscatter = int(p * ratio_w_electron_photon + r)
            # Note: if p is 0, the above formula will return nscatter=0
            # since r is in [0, 1). Similarly, if p is very small,
            # nscatter will be 1 with probabiliy p * ratio_w_electron_photon,
            # and 0 otherwise.
            nscatter_per_elec[ip] = wp.int64(nscatter)
            nscatter_per_batch[i_batch] += wp.int64(nscatter)

            # Increment ip
            ip = ip + 1


@warp_kernel
def scatter_photons_electrons_cuda(
    N_batch: int,
    batch_size: int,
    photon_old_Ntot: int,
    elec_Ntot: int,
    cumul_nscatter_per_batch: wp.array(dtype=wp.int64),
    nscatter_per_elec: wp.array(dtype=wp.int64),
    random_states: wp.array(dtype=wp.uint32),
    photon_p: wp.float64,
    photon_px: wp.float64,
    photon_py: wp.float64,
    photon_pz: wp.float64,
    photon_x: wp.array(dtype=wp.float64),
    photon_y: wp.array(dtype=wp.float64),
    photon_z: wp.array(dtype=wp.float64),
    photon_inv_gamma: wp.array(dtype=wp.float64),
    photon_ux: wp.array(dtype=wp.float64),
    photon_uy: wp.array(dtype=wp.float64),
    photon_uz: wp.array(dtype=wp.float64),
    photon_w: wp.array(dtype=wp.float64),
    elec_x: wp.array(dtype=wp.float64),
    elec_y: wp.array(dtype=wp.float64),
    elec_z: wp.array(dtype=wp.float64),
    elec_inv_gamma: wp.array(dtype=wp.float64),
    elec_ux: wp.array(dtype=wp.float64),
    elec_uy: wp.array(dtype=wp.float64),
    elec_uz: wp.array(dtype=wp.float64),
    elec_w: wp.array(dtype=wp.float64),
    inv_ratio_w_elec_photon: wp.float64,
):
    """
    Given the number of photons that are emitted by each electron
    macroparticle, determine the properties (momentum, energy) of
    each scattered photon and fill the arrays `photon_*` accordingly.

    Also, apply a recoil on the electrons.
    """
    # Loop over batches of particles
    i_batch = wp.tid()
    if i_batch < N_batch:

        # Photon index: this is incremented each time
        # a scattered photon is identified
        i_photon = wp.int64(photon_old_Ntot) + cumul_nscatter_per_batch[i_batch]

        # Loop through the electrons in this batch
        N_max = min((i_batch + 1) * batch_size, elec_Ntot)
        for i_elec in range(i_batch * batch_size, N_max):

            # Prepare calculation of scattered photons from this electron
            if nscatter_per_elec[i_elec] > 0:

                # Prepare Lorentz transformation to the electron rest frame
                elec_gamma = wp.float64(1.0) / elec_inv_gamma[i_elec]
                elec_u = wp.sqrt(
                    elec_ux[i_elec] ** wp.float64(2)
                    + elec_uy[i_elec] ** wp.float64(2)
                    + elec_uz[i_elec] ** wp.float64(2)
                )
                elec_beta = elec_u * elec_inv_gamma[i_elec]
                if elec_u != 0:
                    elec_inv_u = wp.float64(1.0) / elec_u
                    elec_nx = elec_inv_u * elec_ux[i_elec]
                    elec_ny = elec_inv_u * elec_uy[i_elec]
                    elec_nz = elec_inv_u * elec_uz[i_elec]
                else:
                    # Avoid division by 0; provide arbitrary direction
                    # for the Lorentz transform (since beta=0 anyway)
                    elec_nx = wp.float64(0.0)
                    elec_ny = wp.float64(0.0)
                    elec_nz = wp.float64(1.0)

                # Transform momentum of photon to the electron rest frame
                photon_rest_p, photon_rest_px, photon_rest_py, photon_rest_pz = (
                    lorentz_transform(
                        photon_p,
                        photon_px,
                        photon_py,
                        photon_pz,
                        elec_gamma,
                        elec_beta,
                        elec_nx,
                        elec_ny,
                        elec_nz,
                    )
                )
                # Find cos and sin of the spherical angle that represent
                # the direction of the incoming photon in the rest frame
                cos_theta = photon_rest_pz / photon_rest_p
                if cos_theta ** wp.float64(2) < 1:
                    sin_theta = wp.sqrt(wp.float64(1) - cos_theta ** wp.float64(2))
                    inv_photon_rest_pxy = wp.float64(1.0) / (sin_theta * photon_rest_p)
                    cos_phi = photon_rest_px * inv_photon_rest_pxy
                    sin_phi = photon_rest_py * inv_photon_rest_pxy
                else:
                    sin_theta = wp.float64(0.0)
                    # Avoid division by 0; provide arbitrary direction
                    # for the phi angle (since theta is 0 or pi anyway)
                    cos_phi = wp.float64(1.0)
                    sin_phi = wp.float64(0.0)

            # Loop through the number of scatterings for this electron
            for i_scat in range(int(nscatter_per_elec[i_elec])):

                # Draw scattering angle in the rest frame, from the
                # Klein-Nishina cross-section (See Ozmutl, E. N.
                # "Sampling of Angular Distribution in Compton Scattering"
                # Appl. Radiat. Isot. 43, 6, pp. 713-715 (1992))
                k = photon_rest_p * INV_MC
                c0 = (
                    wp.float64(2.0)
                    * (
                        wp.float64(2.0) * k ** wp.float64(2)
                        + wp.float64(2.0) * k
                        + wp.float64(1.0)
                    )
                    / (wp.float64(2.0) * k + wp.float64(1.0)) ** wp.float64(3)
                )
                b = (wp.float64(2.0) + c0) / (wp.float64(2.0) - c0)
                a = wp.float64(2.0) * (b - wp.float64(1.0))
                # Use rejection method to draw x
                reject = bool(True)
                while reject:
                    # - Draw x with an approximate probability distribution
                    r1 = random_uniform(random_states, i_batch)
                    x = b - (b + wp.float64(1.0)) * (wp.float64(0.5) * c0) ** r1
                    # - Calculate approximate probability distribution h
                    h = a / (b - x)
                    # - Calculate expected (exact) probability distribution f
                    factor = wp.float64(1) + k * (wp.float64(1) - x)
                    f = (
                        (wp.float64(1) + x ** wp.float64(2)) * factor
                        + k ** wp.float64(2) * (wp.float64(1) - x) ** wp.float64(2)
                    ) / factor ** wp.float64(3)
                    # - Keep x according to rejection rule
                    r2 = random_uniform(random_states, i_batch)
                    if r2 < f / h:
                        reject = False

                # Get scattered momentum in the rest frame
                new_photon_rest_p = photon_rest_p / (
                    wp.float64(1) + k * (wp.float64(1) - x)
                )
                # - First in a system of axes aligned with the incoming photon
                cos_theta_s = x
                sin_theta_s = wp.sqrt(wp.float64(1) - x ** wp.float64(2))
                r3 = random_uniform(random_states, i_batch)
                phi_s = wp.float64(6.283185307179586) * r3
                cos_phi_s = wp.cos(phi_s)
                sin_phi_s = wp.sin(phi_s)
                new_photon_rest_pX = new_photon_rest_p * sin_theta_s * cos_phi_s
                new_photon_rest_pY = new_photon_rest_p * sin_theta_s * sin_phi_s
                new_photon_rest_pZ = new_photon_rest_p * cos_theta_s
                # - Then rotate it to the original system of axes
                new_photon_rest_px = (
                    sin_theta * cos_phi * new_photon_rest_pZ
                    + cos_theta * cos_phi * new_photon_rest_pX
                    - sin_phi * new_photon_rest_pY
                )
                new_photon_rest_py = (
                    sin_theta * sin_phi * new_photon_rest_pZ
                    + cos_theta * sin_phi * new_photon_rest_pX
                    + cos_phi * new_photon_rest_pY
                )
                new_photon_rest_pz = (
                    cos_theta * new_photon_rest_pZ - sin_theta * new_photon_rest_pX
                )

                # Transform momentum of photon back to the simulation frame
                # (i.e. Lorentz transform with opposite direction)
                new_photon_p, new_photon_px, new_photon_py, new_photon_pz = (
                    lorentz_transform(
                        new_photon_rest_p,
                        new_photon_rest_px,
                        new_photon_rest_py,
                        new_photon_rest_pz,
                        elec_gamma,
                        elec_beta,
                        -elec_nx,
                        -elec_ny,
                        -elec_nz,
                    )
                )

                # Create the new photon by copying the electron position
                photon_x[i_photon] = elec_x[i_elec]
                photon_y[i_photon] = elec_y[i_elec]
                photon_z[i_photon] = elec_z[i_elec]
                photon_w[i_photon] = elec_w[i_elec] * inv_ratio_w_elec_photon
                # The photon's ux, uy, uz corresponds to the actual px, py, pz
                photon_ux[i_photon] = new_photon_px
                photon_uy[i_photon] = new_photon_py
                photon_uz[i_photon] = new_photon_pz
                # The photon's inv_gamma corresponds to 1./p (consistent
                # with the code for the particle pusher and for the
                # openPMD back-transformed diagnostics)
                photon_inv_gamma[i_photon] = wp.float64(1.0) / new_photon_p

                # Update the photon index
                i_photon += wp.int64(1)

            # Add recoil to electrons
            # Note: In order to reproduce the right distribution of electron
            # momentum, the electrons should recoil with the momentum
            # of *one single* photon, with a probability p (calculated by
            # get_scattering_probability). Here we reuse the momentum of
            # the last photon generated above. This requires that at least one
            # photon be created for this electron, which occurs with a
            # probability p*ratio_w_elec_photon. Thus, given that at least one
            # photon has been created, we should add recoil to the corresponding
            # electron only with a probability inv_ratio_w_elec_photon.
            if nscatter_per_elec[i_elec] > 0:
                r = random_uniform(random_states, i_batch)
                if r < inv_ratio_w_elec_photon:
                    elec_ux[i_elec] += INV_MC * (photon_px - new_photon_px)
                    elec_uy[i_elec] += INV_MC * (photon_py - new_photon_py)
                    elec_uz[i_elec] += INV_MC * (photon_pz - new_photon_pz)
