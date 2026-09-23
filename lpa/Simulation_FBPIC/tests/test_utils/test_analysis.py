"""
Unit test that generates a fake beam with 100 particles and checks the basic analysis routine

TODO currently just tests for functionality.  Would be great to expand this to also check for if it is correct!
"""

import numpy as np
from inversion_fbpic.utils import analysis


def generate_fake_beam(N=100, seed=42):
    np.random.seed(seed)
    x = np.random.normal(0, 1e-6, N)
    y = np.random.normal(0, 1e-6, N)
    z = np.random.normal(0, 1e-4, N)
    ux = np.random.normal(0, 1, N)
    uy = np.random.normal(0, 1, N)
    uz = np.random.normal(10, 2, N)  # Forward boost
    w = np.ones(N)
    q = np.full(N, -1.602e-19)  # electron charge
    return x, y, z, ux, uy, uz, w, q


def test_analysis_on_fake_beam():
    x, y, z, ux, uy, uz, w, q = generate_fake_beam()

    # Test calculate_current
    current, axis = analysis.calculate_current(z, ux, uy, uz, w, q, bins=10)
    assert current.shape == (10,)
    assert axis.shape == (10,)
    assert np.all(current >= 0)

    # Test calculate_emittance
    emit = analysis.calculate_geometric_emittance(x, y, ux, uy, uz, w)
    assert "x" in emit and "y" in emit
    assert emit["x"] > 0 and emit["y"] > 0

    # Test calculate_twiss_parameters
    twiss = analysis.calculate_twiss_parameters(x, y, ux, uy, uz, w)
    for plane in ["x", "y"]:
        for param in ["alpha", "beta", "gamma"]:
            assert param in twiss[plane]

    # Test calculate_energy_parameters
    energy = analysis.calculate_energy_parameters(ux, uy, uz, w)
    assert "central_energy_mev" in energy
    assert energy["central_energy_mev"] > 0
    assert "energy_fwhm_mev" in energy

    # Test calculate_beam_size
    sizes = analysis.calculate_beam_size(x, y, z, w)
    for key in ["sigma_x_m", "sigma_y_m", "sigma_z_m"]:
        assert key in sizes
        assert sizes[key] > 0

    # Test analyze_beam (comprehensive)
    results = analysis.analyze_beam(x, y, z, ux, uy, uz, w, q, bins=10)
    assert "current" in results
    assert "twiss_parameters" in results
    assert "emittance" in results
    assert "energy_parameters" in results
    assert "beam_sizes" in results
    assert "total_charge_c" in results
    # Check total charge
    assert np.isclose(results["total_charge_c"], np.sum(w * q))
