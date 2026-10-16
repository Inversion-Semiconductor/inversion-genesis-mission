# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file tests the particle pusher and the external fields by
by studying the motion of particles in plane laser wave.

Usage :
from the top-level directory of FBPIC run
$ python tests/test_external_fields.py

"""
import numpy as np
from scipy.constants import e, m_e, c
from fbpic.main import Simulation
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.external_fields import ExternalField
import math
import pytest
from types import SimpleNamespace
from fbpic.utils.cuda import cuda_installed

# Parameters
# ----------
use_cuda = True

show = True  # Whether to show the results to the user, or to
# automatically determine if they are correct.

# Dimensions of the box
# (low_resolution since the external field are not resolved on the grid)
Nz = 5
Nr = 10
Nm = 2
zmin = 0.0e-6
zmax = 0.8e-6
rmax = 2.0e-6

# Properties of the laser
a0 = 1.0
lambda0 = 0.8e-6
k0 = 2 * np.pi / lambda0

# Particles (one per cell, only initialized near the axis)
p_rmax = rmax / Nr
p_nt = 1
p_nr = 1
p_nz = 1
n = 1.0


def run_external_laser_field_simulation(show, gamma_boost=None):
    """
    Runs a simulation with a set of particles whose motion corresponds
    to that of a particle that is initially at rest (in the lab frame)
    before being reached by a plane wave (propagating to the right)

    In the lab frame, the motion is given by
    ux = a0 sin ( k0(z-ct) )
    uz = ux^2 / 2    (from the conservation of gamma - uz)

    In the boosted frame, the motion is given by
    ux = a0 sin ( k0 gamma0 (1-beta0) (z-ct) )
    uz = - gamma0 beta0 + gamma0 (1-beta0) ux^2 / 2
    """
    # Time parameters
    dt = lambda0 / c / 200  # 200 points per laser period
    N_step = 400  # Two laser periods

    # Initialize BoostConverter object
    if gamma_boost is None:
        boost = BoostConverter(gamma0=1.0)
    else:
        boost = BoostConverter(gamma_boost)
    # Reduce time resolution, for the case of a boosted simulation
    if gamma_boost is not None:
        dt = dt * (1.0 + boost.beta0) / boost.gamma0

    # Initialize the simulation
    sim = Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt,
        initialize_ions=False,
        zmin=zmin,
        use_cuda=use_cuda,
        boundaries={"z": "periodic", "r": "reflective"},
        gamma_boost=gamma_boost,
    )
    # Add electrons
    sim.ptcl = []
    sim.add_new_species(-e, m_e, n=n, p_rmax=p_rmax, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt)

    # Add the external fields
    sim.external_fields = [
        ExternalField(
            laser_func, "Ex", a0 * m_e * c**2 * k0 / e, lambda0, gamma_boost=gamma_boost
        ),
        ExternalField(
            laser_func, "By", a0 * m_e * c * k0 / e, lambda0, gamma_boost=gamma_boost
        ),
    ]

    # Prepare the arrays for the time history of the pusher
    Nptcl = sim.ptcl[0].Ntot
    x = np.zeros((N_step, Nptcl))
    y = np.zeros((N_step, Nptcl))
    z = np.zeros((N_step, Nptcl))
    ux = np.zeros((N_step, Nptcl))
    uy = np.zeros((N_step, Nptcl))
    uz = np.zeros((N_step, Nptcl))

    # Prepare the particles with proper transverse and longitudinal momentum,
    # at t=0 in the simulation frame
    k0p = k0 * boost.gamma0 * (1.0 - boost.beta0)
    sim.ptcl[0].ux = a0 * np.sin(k0p * sim.ptcl[0].z)
    sim.ptcl[0].uz[:] = (
        -boost.gamma0 * boost.beta0
        + boost.gamma0 * (1 - boost.beta0) * 0.5 * sim.ptcl[0].ux ** 2
    )

    # Push the particles over N_step and record the corresponding history
    for i in range(N_step):
        # Record the history
        x[i, :] = sim.ptcl[0].x[:]
        y[i, :] = sim.ptcl[0].y[:]
        z[i, :] = sim.ptcl[0].z[:]
        ux[i, :] = sim.ptcl[0].ux[:]
        uy[i, :] = sim.ptcl[0].uy[:]
        uz[i, :] = sim.ptcl[0].uz[:]
        # Take a simulation step
        sim.step(1)

    # Compute the analytical solution
    t = sim.dt * np.arange(N_step)
    # Conservation of ux
    ux_analytical = np.zeros((N_step, Nptcl))
    uz_analytical = np.zeros((N_step, Nptcl))
    for i in range(N_step):
        ux_analytical[i, :] = a0 * np.sin(k0p * (z[i, :] - c * t[i]))
        uz_analytical[i, :] = (
            -boost.gamma0 * boost.beta0
            + boost.gamma0 * (1 - boost.beta0) * 0.5 * ux_analytical[i, :] ** 2
        )

    # Show the results
    if show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        plt.subplot(211)
        plt.plot(t, ux_analytical, "--")
        plt.plot(t, ux, "o")
        plt.xlabel("t")
        plt.ylabel("ux")

        plt.subplot(212)
        plt.plot(t, uz_analytical, "--")
        plt.plot(t, uz, "o")
        plt.xlabel("t")
        plt.ylabel("uz")

        plt.show()
    else:
        assert np.allclose(ux, ux_analytical, atol=5.0e-2)
        assert np.allclose(uz, uz_analytical, atol=5.0e-2)


def laser_func(F, x, y, z, t, amplitude, length_scale):
    """
    Function to be called at each timestep on the particles
    """
    return F + amplitude * math.cos(2 * np.pi * (z - c * t) / length_scale)


def test_external_fields_lab(show=False):
    "Function that is run by py.test"
    run_external_laser_field_simulation(show, None)


def test_external_fields_boost(show=False):
    "Function that is run by py.test"
    run_external_laser_field_simulation(show, gamma_boost=10)


# These callbacks exercise scalar syntax accepted by the baseline API.
_callback_coefficients = (0.125, 2.5, -3.25)
_callback_table = np.arange(12, dtype=np.float64).reshape(4, 3) / 7
_callback_large_integers = np.array([2**60, 2**60 + 1, 2**60 + 2], dtype=np.int64)
_callback_large_tuple = (2**60, 2**60 + 1, 2**60 + 2)
_callback_float32_table = np.array([2**24, 1.0e-8, -0.25], dtype=np.float32)
_callback_int32_table = np.array([-(2**31), 2**31 - 1], dtype=np.int32)
_callback_uint32_table = np.array([0, 2**32 - 1], dtype=np.uint32)


def _callback_expm1(F, x, y, z, t, amplitude, length_scale):
    return math.expm1(x)


def _callback_log1p(F, x, y, z, t, amplitude, length_scale):
    return math.log1p(x)


def _callback_hypot(F, x, y, z, t, amplitude, length_scale):
    return math.hypot(x, y)


def _callback_tuple(F, x, y, z, t, amplitude, length_scale):
    return _callback_coefficients[int(x)]


def _callback_array(F, x, y, z, t, amplitude, length_scale):
    return _callback_table[int(x), int(y)]


def _callback_integer_array(F, x, y, z, t, amplitude, length_scale):
    difference = _callback_large_integers[int(x)] - _callback_large_integers[0]
    return difference


def _callback_integer_tuple(F, x, y, z, t, amplitude, length_scale):
    return _callback_large_tuple[int(x)] - _callback_large_tuple[0]


def _callback_max(F, x, y, z, t, amplitude, length_scale):
    return max(x, 0)


def _callback_float_accumulator(F, x, y, z, t, amplitude, length_scale):
    total = 0.0
    total += 1
    return total


def _callback_promoted_accumulator(F, x, y, z, t, amplitude, length_scale):
    total = 0
    total += x
    return total


def _callback_integer_accumulator(F, x, y, z, t, amplitude, length_scale):
    total = _callback_large_integers[0]
    total += _callback_large_integers[int(x)] - _callback_large_integers[0]
    return total - _callback_large_integers[0]


def _callback_float_modulo(F, x, y, z, t, amplitude, length_scale):
    return x % y


def _callback_integer_modulo(F, x, y, z, t, amplitude, length_scale):
    return int(x) % int(y)


def _callback_float_floor(F, x, y, z, t, amplitude, length_scale):
    return x // y


def _callback_integer_floor(F, x, y, z, t, amplitude, length_scale):
    return int(x) // int(y)


def _callback_negative_literal_power(F, x, y, z, t, amplitude, length_scale):
    return 2**-2


def _callback_piecewise(F, x, y, z, t, amplitude, length_scale):
    return amplitude if x > 0 else 0


def _callback_float32_promotion(F, x, y, z, t, amplitude, length_scale):
    return _callback_float32_table[int(x)] + 1 - _callback_float32_table[int(x)]


def _callback_int32_promotion(F, x, y, z, t, amplitude, length_scale):
    return _callback_int32_table[int(x)] - _callback_int32_table[0]


def _callback_uint32_promotion(F, x, y, z, t, amplitude, length_scale):
    return _callback_uint32_table[int(x)] + _callback_uint32_table[int(x)]


# Keep a lambda: this case checks that the callback compiler accepts one.
_callback_lambda = lambda F, x, y, z, t, amplitude, length_scale: math.expm1(x)  # noqa: E731  # fmt: skip


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "callback,x,y",
    [
        (_callback_expm1, [-1.0e-20, 0.0, 1.0e-20, 1.0e-10, 0.3], [0.0] * 5),
        (_callback_log1p, [-1.0e-20, 0.0, 1.0e-20, 1.0e-10, 0.3], [0.0] * 5),
        (
            _callback_hypot,
            [1.0e300, 1.0e-300, 1.0e308, 3.0e-250],
            [1.0e300, 1.0e-300, 1.0e308, 4.0e-250],
        ),
        (_callback_tuple, [-3, -2, -1, 0, 1, 2], [0.0] * 6),
        (_callback_array, [-4, -1, 0, 2, 3], [-3, -1, 0, 1, 2]),
        (_callback_integer_array, [0, 1, 2], [0.0] * 3),
        (_callback_integer_tuple, [0, 1, 2], [0.0] * 3),
        (_callback_lambda, [-1.0e-20, 0.0, 1.0e-20, 1.0e-10], [0.0] * 4),
        (_callback_max, [-0.5, 0.0, 0.5], [0.0] * 3),
        (_callback_float_accumulator, [0.0, 1.0], [0.0] * 2),
        (_callback_promoted_accumulator, [-0.5, 0.0, 0.5], [0.0] * 3),
        (_callback_integer_accumulator, [0, 1, 2], [0.0] * 3),
        (_callback_float_modulo, [-5.5, -0.5, 5.5, 5.5], [2.0, 1.0, -2.0, 2.0]),
        (_callback_integer_modulo, [-3.0, 3.0, -3.0, 3.0], [2.0, -2.0, -2.0, 2.0]),
        (_callback_float_floor, [1.0, -1.0, -5.5, 5.5], [0.1, 0.1, 2.0, -2.0]),
        (_callback_integer_floor, [-3.0, 3.0, -3.0, 3.0], [2.0, -2.0, -2.0, 2.0]),
        (_callback_negative_literal_power, [0.0], [0.0]),
        (_callback_piecewise, [-0.5, 0.0, 0.5], [0.0] * 3),
        (_callback_float32_promotion, [0, 1, 2], [0.0] * 3),
        (_callback_int32_promotion, [0, 1], [0.0] * 2),
        (_callback_uint32_promotion, [0, 1], [0.0] * 2),
    ],
    ids=[
        "expm1",
        "log1p",
        "hypot",
        "tuple",
        "array",
        "integer-array",
        "integer-tuple",
        "lambda",
        "max",
        "float-accumulator",
        "promoted-accumulator",
        "integer-accumulator",
        "float-modulo",
        "integer-modulo",
        "float-floor",
        "integer-floor",
        "literal-power",
        "piecewise",
        "float32-promotion",
        "int32-promotion",
        "uint32-promotion",
    ],
)
def test_external_field_callback_gpu(callback, x, y):
    """Compare scalar callbacks without a full particle simulation."""
    import cupy
    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    field = ExternalField(callback, "Ex", 1.0, 1.0)
    # The unchanged Numba CPU path defines scalar promotion in the baseline.
    expected = field.cpu_func(np.zeros_like(x), x, y, np.zeros_like(x), 0.0, 1.0, 1.0)
    assert field.gpu_func is None
    particles = SimpleNamespace(
        Ntot=len(x),
        x=cupy.asarray(x),
        y=cupy.asarray(y),
        z=cupy.zeros(len(x)),
        Ex=cupy.zeros(len(x)),
    )
    launches_before = runtime.launch_count
    field.apply_expression([particles], 0.0)
    assert runtime.launch_count > launches_before
    np.testing.assert_allclose(particles.Ex.get(), expected, rtol=8.0e-14, atol=0.0)


def test_external_field_cpu_keeps_gpu_compilation_lazy():
    """Apply a CPU callback without preparing a GPU callback."""

    def callback(F, x, y, z, t, amplitude, length_scale):
        return F + np.interp(x, np.array([0.0, 1.0]), np.array([2.0, 4.0]))

    x = np.array([0.0, 0.25, 0.75, 1.0])
    particles = SimpleNamespace(
        Ntot=len(x), x=x, y=np.zeros_like(x), z=np.zeros_like(x), Ex=np.zeros_like(x)
    )
    field = ExternalField(callback, "Ex", 1.0, 1.0)
    assert field.gpu_func is None
    field.apply_expression([particles], 0.0)
    assert field.gpu_func is None
    np.testing.assert_allclose(particles.Ex, 2.0 + 2.0 * x)


if __name__ == "__main__":

    test_external_fields_lab(show)
    test_external_fields_boost(show)
