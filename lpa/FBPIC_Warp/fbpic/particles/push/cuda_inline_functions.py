# Copyright 2018, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def push_p_vay(
    ux_i: wp.float64,
    uy_i: wp.float64,
    uz_i: wp.float64,
    inv_gamma_i: wp.float64,
    Ex: wp.float64,
    Ey: wp.float64,
    Ez: wp.float64,
    Bx: wp.float64,
    By: wp.float64,
    Bz: wp.float64,
    econst: wp.float64,
    bconst: wp.float64,
):
    """
    Push at single macroparticle, using the Vay pusher
    """
    # Get the magnetic rotation vector
    taux = bconst * Bx
    tauy = bconst * By
    tauz = bconst * Bz
    tau2 = taux * taux + tauy * tauy + tauz * tauz

    # Get the momenta at the half timestep
    uxp = ux_i + econst * Ex + inv_gamma_i * (uy_i * tauz - uz_i * tauy)
    uyp = uy_i + econst * Ey + inv_gamma_i * (uz_i * taux - ux_i * tauz)
    uzp = uz_i + econst * Ez + inv_gamma_i * (ux_i * tauy - uy_i * taux)
    sigma = wp.float64(1) + uxp * uxp + uyp * uyp + uzp * uzp - tau2
    utau = uxp * taux + uyp * tauy + uzp * tauz

    # Get the new 1./gamma
    inv_gamma_f = wp.sqrt(
        wp.float64(2.0)
        / (sigma + wp.sqrt(sigma * sigma + wp.float64(4) * (tau2 + utau * utau)))
    )

    # Reuse the tau and utau arrays to save memory
    tx = inv_gamma_f * taux
    ty = inv_gamma_f * tauy
    tz = inv_gamma_f * tauz
    ut = inv_gamma_f * utau
    s = wp.float64(1.0) / (wp.float64(1) + tau2 * inv_gamma_f * inv_gamma_f)

    # Get the new u
    ux_f = s * (uxp + tx * ut + uyp * tz - uzp * ty)
    uy_f = s * (uyp + ty * ut + uzp * tx - uxp * tz)
    uz_f = s * (uzp + tz * ut + uxp * ty - uyp * tx)

    return (ux_f, uy_f, uz_f, inv_gamma_f)
