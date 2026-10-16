# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import cmul

wp.set_module_options({"enable_backward": False})


@wp.func
def add_linear_gather_for_mode(
    m: wp.int32,
    Fr: wp.float64,
    Ft: wp.float64,
    Fz: wp.float64,
    exptheta_m: wp.vec2d,
    Fr_grid: wp.array2d(dtype=wp.vec2d),
    Ft_grid: wp.array2d(dtype=wp.vec2d),
    Fz_grid: wp.array2d(dtype=wp.vec2d),
    iz_lower: wp.int32,
    iz_upper: wp.int32,
    ir_lower: wp.int32,
    ir_upper: wp.int32,
    S_ll: wp.float64,
    S_lu: wp.float64,
    S_lg: wp.float64,
    S_ul: wp.float64,
    S_uu: wp.float64,
    S_ug: wp.float64,
):
    """
    Add the contribution of the gathered field from azimuthal mode `m` to the
    fields felt by one macroparticle (`Fr`, `Ft`, `Fz`), using linear weights.

    Parameters:
    -----------
    m: int
        The index of the azimuthal mode that is added.

    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (before the contribution of mode `m` has been added)

    exptheta_m: complex
        The complex azimuthal factor $e^{-i m \theta}$ where $\theta$ is
        the azimuthal position of the macroparticle considered.

    Fr_grid, Ft_grid, Fz_grid: 2darrays of complexs
        The fields on the interpolation grid for mode `m`

    iz_lower, iz_upper, ir_lower, ir_upper: ints
        Lower and upper index in z and r from which the macroparticle
        considered should gather the fields (in the arrays F*_grid)

    S_ll, S_lu, S_lg, S_ul, S_uu, S_ug: floats
        The weights with which the fields are gathered, for the macroparticle
        considered. `S_lg` and `S_ug` are used for fields gathered from below
        the axis.

    Returns:
    --------
    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (after the contribution of mode `m` has been added)
    """
    # Create temporary variables
    # for the "per mode" gathering
    Fr_m = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    Ft_m = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    Fz_m = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    # Lower cell in z, Lower cell in r
    Fr_m += S_ll * Fr_grid[iz_lower, ir_lower]
    Ft_m += S_ll * Ft_grid[iz_lower, ir_lower]
    Fz_m += S_ll * Fz_grid[iz_lower, ir_lower]
    # Lower cell in z, Upper cell in r
    Fr_m += S_lu * Fr_grid[iz_lower, ir_upper]
    Ft_m += S_lu * Ft_grid[iz_lower, ir_upper]
    Fz_m += S_lu * Fz_grid[iz_lower, ir_upper]
    # Upper cell in z, Lower cell in r
    Fr_m += S_ul * Fr_grid[iz_upper, ir_lower]
    Ft_m += S_ul * Ft_grid[iz_upper, ir_lower]
    Fz_m += S_ul * Fz_grid[iz_upper, ir_lower]
    # Upper cell in z, Upper cell in r
    Fr_m += S_uu * Fr_grid[iz_upper, ir_upper]
    Ft_m += S_uu * Ft_grid[iz_upper, ir_upper]
    Fz_m += S_uu * Fz_grid[iz_upper, ir_upper]
    # Add the fields from the guard cells
    if ir_lower == ir_upper == 0:
        flip_factor = wp.float64(1 - 2 * (m % 2))
        # Lower cell in z
        Fr_m += -flip_factor * S_lg * Fr_grid[iz_lower, 0]
        Ft_m += -flip_factor * S_lg * Ft_grid[iz_lower, 0]
        Fz_m += flip_factor * S_lg * Fz_grid[iz_lower, 0]
        # Upper cell in z
        Fr_m += -flip_factor * S_ug * Fr_grid[iz_upper, 0]
        Ft_m += -flip_factor * S_ug * Ft_grid[iz_upper, 0]
        Fz_m += flip_factor * S_ug * Fz_grid[iz_upper, 0]
    # Add the contribution from mode m to Fr, Ft, Fz
    # (Take into account factor 2 in the definition of azimuthal modes)
    if m == 0:
        factor = wp.float64(1.0)
    else:
        factor = wp.float64(2.0)
    Fr += factor * cmul(Fr_m, exptheta_m)[0]
    Ft += factor * cmul(Ft_m, exptheta_m)[0]
    Fz += factor * cmul(Fz_m, exptheta_m)[0]

    return (Fr, Ft, Fz)


@wp.func
def add_cubic_gather_for_mode(
    m: wp.int32,
    Fr: wp.float64,
    Ft: wp.float64,
    Fz: wp.float64,
    exptheta_m: wp.vec2d,
    Fr_grid: wp.array2d(dtype=wp.vec2d),
    Ft_grid: wp.array2d(dtype=wp.vec2d),
    Fz_grid: wp.array2d(dtype=wp.vec2d),
    ir_lowest: wp.int32,
    iz_lowest: wp.int32,
    Sr_arr: wp.vec4d,
    Sz_arr: wp.vec4d,
    Nr: wp.int32,
    Nz: wp.int32,
):
    """
    Add the contribution of the gathered field from azimuthal mode `m` to the
    fields felt by one macroparticle (`Fr`, `Ft`, `Fz`), using cubic weights.

    Parameters:
    -----------
    m: int
        The index of the azimuthal mode that is added.

    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (before the contribution of mode `m` has been added)

    exptheta_m: complex
        The complex azimuthal factor $e^{-i m \theta}$ where $\theta$ is
        the azimuthal position of the macroparticle considered.

    Fr_grid, Ft_grid, Fz_grid: 2darrays of complexs
        The fields on the interpolation grid for mode `m`

    ir_lowest, iz_lowest: ints
        The lowest indices in r and z from which the macroparticle
        considered should gather the fields (in the arrays F*_grid)
        These indices can in fact be negative and out-of-bound
        (e.g. for particles close to the axis) but get corrected within
        this function.

    Sr_arr, Sz_arr: 1darrays containing 4 floats
        The weights in r and z with which the macroparticle
        considered should gather the fields (in the arrays F*_grid)

    Nr, Nz: ints
        Dimensions of the field arrays.

    Returns:
    --------
    Fr, Ft, Fz: floats
        The fields felt by one macroparticle, which represent either E or B
        (after the contribution of mode `m` has been added)
    """
    # Create temporary variables
    # for the "per mode" gathering
    Fr_m = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    Ft_m = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    Fz_m = wp.vec2d(wp.float64(0.0), wp.float64(0.0))

    # Loop over the 4x4 cells from which to gather fields
    for index_r in range(4):

        # Radial index
        ir = ir_lowest + index_r
        # Calculate shape factor for the longitudinal
        # and transverse components of the field
        Sr_long = Sr_arr[index_r]
        Sr_perp = Sr_long
        if ir < 0:
            Sr_long *= wp.float64(1 - 2 * (m % 2))
            Sr_perp *= wp.float64(-(1 - 2 * (m % 2)))
        # Adjust radial index to avoid out of bound
        if ir < 0:
            ir = abs(ir) - 1
        elif ir > Nr - 1:
            ir = Nr - 1

        for index_z in range(4):

            # Longitudinal index
            iz = iz_lowest + index_z
            # Get shape factor
            Sz = Sz_arr[index_z]
            # Adjust longitudinal index to avoid out of bound
            if iz < 0:
                iz += Nz
            elif iz > Nz - 1:
                iz -= Nz

            # Get the fields
            Fr_m += Sz * Sr_perp * Fr_grid[iz, ir]
            Ft_m += Sz * Sr_perp * Ft_grid[iz, ir]
            Fz_m += Sz * Sr_long * Fz_grid[iz, ir]

    # Add the contribution from mode m to Fr, Ft, Fz
    # (Take into account factor 2 in the definition of azimuthal modes)
    if m == 0:
        factor = wp.float64(1.0)
    else:
        factor = wp.float64(2.0)
    Fr += factor * cmul(Fr_m, exptheta_m)[0]
    Ft += factor * cmul(Ft_m, exptheta_m)[0]
    Fz += factor * cmul(Fz_m, exptheta_m)[0]

    return (Fr, Ft, Fz)
