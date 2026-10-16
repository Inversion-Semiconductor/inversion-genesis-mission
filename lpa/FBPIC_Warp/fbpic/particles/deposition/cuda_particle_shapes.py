# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import square, cube

wp.set_module_options({"enable_backward": False})


@wp.func
def Sz_linear(
    cell_position: wp.float64,
    index: wp.int32,
):
    s = wp.ceil(cell_position) - cell_position
    if index == 1:
        s = wp.float64(1.0) - s
    return s


# transversal (r)
@wp.func
def Sr_linear(
    cell_position: wp.float64,
    index: wp.int32,
    flip: wp.float64,
    beta_n: wp.float64,
):
    ir = wp.int32(wp.ceil(cell_position)) - 1
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - wp.float64(ir)
    s = wp.float64(1.0) - u + beta_n * (wp.float64(1.0) - u) * u
    if index == 1:
        s = wp.float64(1.0) - s
    # Check if the cell to which the particle deposits is below the axis
    if index + ir < 0:
        # In this case, flip the sign of the particle contribution
        s *= flip
    return s


# Cubic shapes


# longitudinal (z)
@wp.func
def Sz_cubic(
    cell_position: wp.float64,
    index: wp.int32,
):
    iz = wp.int32(wp.ceil(cell_position)) - 2
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - wp.float64(iz) - wp.float64(1)
    s = wp.float64(0.0)
    if index == 0:
        s = wp.float64(1.0) / wp.float64(6.0) * cube(wp.float64(1.0) - u)
    elif index == 1:
        s = (
            wp.float64(1.0)
            / wp.float64(6.0)
            * (
                wp.float64(3.0) * cube(u)
                - wp.float64(6.0) * square(u)
                + wp.float64(4.0)
            )
        )
    elif index == 2:
        s = (
            wp.float64(1.0)
            / wp.float64(6.0)
            * (
                wp.float64(3.0) * cube(wp.float64(1.0) - u)
                - wp.float64(6.0) * square(wp.float64(1.0) - u)
                + wp.float64(4.0)
            )
        )
    elif index == 3:
        s = wp.float64(1.0) / wp.float64(6.0) * cube(u)
    return s


# transversal (r)
@wp.func
def Sr_cubic(
    cell_position: wp.float64,
    index: wp.int32,
    flip: wp.float64,
    beta_n: wp.float64,
):
    ir = wp.int32(wp.ceil(cell_position)) - 2
    # u: position of the particle with respect to its left neighbor gridpoint
    # (u is between 0 and 1)
    u = cell_position - wp.float64(ir) - wp.float64(1)
    s = wp.float64(0.0)
    if index == 0:
        s = wp.float64(1.0) / wp.float64(6.0) * cube(wp.float64(1.0) - u)
    elif index == 1:
        s = (
            wp.float64(1.0)
            / wp.float64(6.0)
            * (
                wp.float64(3.0) * cube(u)
                - wp.float64(6.0) * square(u)
                + wp.float64(4.0)
            )
        )
        s += beta_n * (wp.float64(1.0) - u) * u  # Add Ruyten correction
    elif index == 2:
        s = (
            wp.float64(1.0)
            / wp.float64(6.0)
            * (
                wp.float64(3.0) * cube(wp.float64(1.0) - u)
                - wp.float64(6.0) * square(wp.float64(1.0) - u)
                + wp.float64(4.0)
            )
        )
        s -= beta_n * (wp.float64(1.0) - u) * u  # Add Ruyten correction
    elif index == 3:
        s = wp.float64(1.0) / wp.float64(6.0) * cube(u)
    # Check if the cell to which the particle deposits is below the axis
    if index + ir < 0:
        # In this case, flip the sign of the particle contribution
        s *= flip
    return s
