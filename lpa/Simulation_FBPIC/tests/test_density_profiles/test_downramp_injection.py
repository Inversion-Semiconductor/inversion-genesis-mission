"""
Unit tests for the density profile functions in downramp_injection.py.

This module contains tests for:
- build_gaussian_profile: Tests for the basic Gaussian density profile function
- build_gaussian_plus_triangle_z_density_function: Tests for the combined Gaussian and triangular density profile

The tests verify:
- Basic properties (symmetry, normalization)
- Edge cases
- Array input handling
- Proper combination of Gaussian and triangular components
"""

import numpy as np
from inversion_fbpic.density_profiles.downramp_injection import (
    build_gaussian_profile,
    build_gaussian_plus_triangle_z_density_function,
    build_piecewise_linear_downramp,
)
import pytest


def test_gaussian_profile_basic() -> None:
    """Test basic properties of the Gaussian profile function."""
    sigma = 1.0
    center = 0.0
    density_func = build_gaussian_profile(sigma=sigma, center_location=center)

    # Test at center
    assert np.isclose(density_func(0.0, 0.0), 1.0)

    # Test at one sigma away
    assert np.isclose(density_func(sigma, 0.0), np.exp(-0.5))

    # Test symmetry
    z1 = 1.0
    z2 = -1.0
    assert np.isclose(density_func(z1, 0.0), density_func(z2, 0.0))

    # Test array inputs
    z = np.array([-1.0, 0.0, 1.0])
    r = np.array([0.0, 0.0, 0.0])
    result = density_func(z, r)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_gaussian_profile_shifted() -> None:
    """Test Gaussian profile with non-zero center location."""
    sigma = 1.0
    center = 2.0
    density_func = build_gaussian_profile(sigma=sigma, center_location=center)

    # Test at center
    assert np.isclose(density_func(center, 0.0), 1.0)

    # Test at one sigma away
    assert np.isclose(density_func(center + sigma, 0.0), np.exp(-0.5))


def test_gaussian_plus_triangle_basic() -> None:
    """Test basic properties of the combined Gaussian and triangle profile."""
    sigma = 1.0
    center = 0.0
    z_tip = 1.0
    left_width = 1.0
    right_width = 1.0
    triangle_height = 1.0

    density_func = build_gaussian_plus_triangle_z_density_function(
        sigma=sigma,
        center_location=center,
        z_tip=z_tip,
        left_width=left_width,
        right_width=right_width,
        triangle_height=triangle_height,
    )

    # Calculate expected value at triangle tip
    gaussian_at_tip = np.exp(-((z_tip - center) ** 2) / (2 * sigma**2))
    expected_tip_value = (gaussian_at_tip + triangle_height) / (1 + triangle_height)

    # Test at triangle tip
    assert np.isclose(density_func(z_tip, 0.0), expected_tip_value)

    # Test at triangle edges
    gaussian_at_left = np.exp(-((z_tip - left_width - center) ** 2) / (2 * sigma**2))
    gaussian_at_right = np.exp(-((z_tip + right_width - center) ** 2) / (2 * sigma**2))

    assert np.isclose(
        density_func(z_tip - left_width, 0.0), gaussian_at_left / (1 + triangle_height)
    )
    assert np.isclose(
        density_func(z_tip + right_width, 0.0),
        gaussian_at_right / (1 + triangle_height),
    )

    # Test array inputs
    z = np.array([z_tip - left_width, z_tip, z_tip + right_width])
    r = np.array([0.0, 0.0, 0.0])
    result = density_func(z, r)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_gaussian_plus_triangle_normalization() -> None:
    """Test that the combined profile is properly normalized."""
    sigma = 1.0
    center = 1.0  # Align Gaussian center with triangle tip
    z_tip = 1.0
    left_width = 1.0
    right_width = 1.0
    triangle_height = 2.0

    density_func = build_gaussian_plus_triangle_z_density_function(
        sigma=sigma,
        center_location=center,
        z_tip=z_tip,
        left_width=left_width,
        right_width=right_width,
        triangle_height=triangle_height,
    )

    # Calculate expected value at triangle tip
    gaussian_at_tip = np.exp(
        -((z_tip - center) ** 2) / (2 * sigma**2)
    )  # This will be 1.0 since z_tip = center
    expected_value = (gaussian_at_tip + triangle_height) / (1 + triangle_height)

    # Test at triangle tip
    assert np.isclose(density_func(z_tip, 0.0), expected_value)


def test_gaussian_plus_triangle_edge_cases() -> None:
    """Test edge cases of the combined profile."""
    sigma = 1.0
    center = 0.0
    z_tip = 1.0
    left_width = 1.0
    right_width = 1.0
    triangle_height = 1.0

    density_func = build_gaussian_plus_triangle_z_density_function(
        sigma=sigma,
        center_location=center,
        z_tip=z_tip,
        left_width=left_width,
        right_width=right_width,
        triangle_height=triangle_height,
    )

    # Test far from the profile
    assert np.isclose(density_func(100.0, 0.0), 0.0, atol=1e-10)

    # Test at the exact boundaries
    gaussian_at_left = np.exp(-((z_tip - left_width - center) ** 2) / (2 * sigma**2))
    gaussian_at_right = np.exp(-((z_tip + right_width - center) ** 2) / (2 * sigma**2))

    assert np.isclose(
        density_func(z_tip - left_width, 0.0), gaussian_at_left / (1 + triangle_height)
    )
    assert np.isclose(
        density_func(z_tip + right_width, 0.0),
        gaussian_at_right / (1 + triangle_height),
    )


def test_piecewise_linear_downramp_basic():
    """Test the piecewise linear downramp profile at key points."""
    upramp_length = 2.0
    downramp_start_position = 4.0
    downramp_length = 2.0
    downramp_height_ratio = 0.5
    plateau_end_position = 8.0
    plateau_downramp_length = 2.0

    density_func = build_piecewise_linear_downramp(
        upramp_length=upramp_length,
        downramp_start_position=downramp_start_position,
        downramp_length=downramp_length,
        downramp_height_ratio=downramp_height_ratio,
        plateau_end_position=plateau_end_position,
        plateau_downramp_length=plateau_downramp_length,
    )

    # Before upramp
    assert density_func(-1.0, 0.0) == 0.0
    # Start of upramp
    assert density_func(0.0, 0.0) == 0.0
    # End of upramp
    assert density_func(upramp_length, 0.0) == 1.0
    # On first plateau
    assert density_func((upramp_length + downramp_start_position) / 2, 0.0) == 1.0
    # Start of first downramp
    assert density_func(downramp_start_position, 0.0) == 1.0
    # End of first downramp
    assert density_func(
        downramp_start_position + downramp_length, 0.0
    ) == pytest.approx(downramp_height_ratio)
    # On second plateau
    assert density_func(
        (downramp_start_position + downramp_length + plateau_end_position) / 2, 0.0
    ) == pytest.approx(downramp_height_ratio)
    # Start of final downramp
    assert density_func(plateau_end_position, 0.0) == pytest.approx(
        downramp_height_ratio
    )
    # End of final downramp
    assert density_func(
        plateau_end_position + plateau_downramp_length, 0.0
    ) == pytest.approx(0.0)
    # After profile
    assert (
        density_func(plateau_end_position + plateau_downramp_length + 1.0, 0.0) == 0.0
    )
