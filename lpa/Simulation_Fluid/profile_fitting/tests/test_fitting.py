from __future__ import annotations

import json

import numpy as np
import pytest

from fludat_fit.families import (
    GeneralizedLorentzianSumFamily,
    GenericConicalTargetFamily,
    get_families,
)
from fludat_fit.fitting import (
    MAX_DEVIATION_EXPONENT,
    FitObjective,
    FitResult,
    FittingScheme,
    MultiStartLocalFit,
    compare_families,
    density_units_to_m3,
    fit_family,
    fit_lineouts,
    optimal_amplitude,
)
from fludat_fit.goodness_of_fit import GoodnessOfFit, rank_results
from fludat_fit.lineout import LineoutConditions
from fludat_fit.parameters import ParameterSpace, ParameterSpec
from fludat_fit.plotting import plot_fit_comparison

from .conftest import make_lineout


def test_goodness_of_fit_of_a_perfect_and_a_shifted_model():
    z = np.linspace(0.0, 1.0, 101)
    data = np.sin(np.pi * z) ** 2
    perfect = GoodnessOfFit.compute(z, data, data, n_parameters=2)
    assert perfect.sse == 0.0 and perfect.nrmse == 0.0 and perfect.r_squared == 1.0
    assert np.isfinite(perfect.bic)

    shifted = GoodnessOfFit.compute(z, data, data + 0.1, n_parameters=2)
    assert shifted.rmse == pytest.approx(0.1)
    assert shifted.nrmse == pytest.approx(0.1)
    assert shifted.max_abs_error == pytest.approx(0.1)
    assert shifted.integrated_relative_error == pytest.approx(0.1 / 0.5, rel=1e-2)
    assert shifted.bic > perfect.bic and shifted.aic > perfect.aic
    assert shifted.n_points == 101 and shifted.n_parameters == 2
    assert set(shifted.to_dict()) >= {"sse", "bic", "nrmse"}


def test_ranking_orders_by_metric_and_pushes_nan_last():
    class R:
        def __init__(self, **values):
            base = dict(
                n_points=1,
                n_parameters=1,
                sse=0,
                rmse=0,
                nrmse=0,
                r_squared=1,
                max_abs_error=0,
                integrated_relative_error=0,
                aic=0,
                bic=0,
            )
            self.goodness = GoodnessOfFit(**{**base, **values})

    a, b, c = R(bic=2.0, r_squared=0.5), R(bic=1.0, r_squared=0.9), R(bic=float("nan"))
    assert rank_results([a, b, c], by="bic") == [b, a, c]
    assert rank_results([a, b], by="r_squared") == [b, a]
    with pytest.raises(KeyError):
        rank_results([a], by="vibes")


def test_optimal_amplitude_is_non_negative_least_squares():
    f = np.array([1.0, 2.0, 3.0])
    w = np.ones(3)
    assert optimal_amplitude(f, 2.5 * f, w) == pytest.approx(2.5)
    assert optimal_amplitude(f, -f, w) == 0.0
    assert optimal_amplitude(np.zeros(3), f, w) == 0.0


def test_density_units_conversion():
    assert density_units_to_m3("cm^-3") == 1.0e6
    assert density_units_to_m3(" m^-3 ") == 1.0
    assert density_units_to_m3("kg/m^3") is None


def test_fit_recovers_known_supergaussian(supergaussian_lineout, supergaussian_truth):
    family, truth = supergaussian_truth
    scheme = MultiStartLocalFit(n_starts=3, seed=0)
    result = fit_family(supergaussian_lineout.trimmed(), family, scheme)

    assert isinstance(scheme, FittingScheme)
    assert result.success
    assert result.family_name == "conical:supergaussian"
    assert result.goodness.nrmse < 0.01
    assert result.amplitude == pytest.approx(2.0e19, rel=0.02)
    assert result.parameters["center"] == pytest.approx(truth["center"], abs=3e-5)
    assert result.parameters["fwhm"] == pytest.approx(truth["fwhm"], rel=0.02)
    assert result.parameters["beta"] == pytest.approx(truth["beta"], rel=0.05)
    assert result.parameters["skew_rate"] == pytest.approx(truth["skew_rate"], abs=15.0)
    assert result.n_starts == 4  # sampled starts plus the heuristic guess
    assert result.n_function_evaluations > 0

    # The exported profile reproduces the fitted curve in m^-3.
    profile = result.build_profile(species="H", p_nz=2)
    assert profile.species == "H" and profile.p_nz == 2
    assert profile.nominal_density == pytest.approx(result.amplitude * 1.0e6)
    z = result.lineout.z
    np.testing.assert_allclose(
        profile.build_density_function()(z, 0 * z) * profile.nominal_density,
        result.model_density() * 1.0e6,
        rtol=1e-9,
    )

    payload = result.to_dict()
    assert payload["profile_config"]["subclass"] == "generic_conical_target"
    assert payload["profile_config_nominal_density_is_placeholder"] is False
    json.dumps(payload)


def test_unknown_units_need_an_explicit_nominal_density(z_grid):
    family = get_families(["asymmetric_sine"])[0]
    lineout = make_lineout(
        family.relative_density(
            {"peak_z0": 0.0, "upramp_length": 2e-3, "downramp_length": 1e-3}, z_grid
        ),
        z_grid,
        units="kg/m^3",
        amplitude=0.02,
    )
    result = fit_family(lineout, family, MultiStartLocalFit(n_starts=1))
    assert result.nominal_density_m3() is None
    with pytest.raises(ValueError, match="pass nominal_density"):
        result.build_profile()
    profile = result.build_profile(nominal_density=4.0e24)
    assert profile.nominal_density == 4.0e24
    assert result.to_dict()["profile_config_nominal_density_is_placeholder"] is True


def test_fixed_amplitude_and_other_optimizers(
    supergaussian_lineout, supergaussian_truth
):
    family, _ = supergaussian_truth
    lineout = supergaussian_lineout.trimmed()
    fixed = MultiStartLocalFit(n_starts=1, fit_amplitude=False).fit(lineout, family)
    assert fixed.amplitude == lineout.peak
    assert fixed.goodness.n_parameters == fixed.space.dimension

    powell = MultiStartLocalFit(n_starts=1, method="Powell").fit(lineout, family)
    assert powell.scheme == "multistart_local[Powell]"
    assert powell.goodness.nrmse < 0.05


def test_scheme_needs_a_start_and_accepts_warm_starts(
    supergaussian_lineout, supergaussian_truth
):
    family, truth = supergaussian_truth
    lineout = supergaussian_lineout.trimmed()
    with pytest.raises(ValueError):
        MultiStartLocalFit(n_starts=-1)
    cold = MultiStartLocalFit(n_starts=0, include_initial_guess=False)
    with pytest.raises(ValueError, match="no starting points"):
        cold.fit(lineout, family)
    # A warm start alone is enough, and a fixed reference space is honoured.
    space = family.parameter_space(lineout.summary())
    wide = ParameterSpace(
        tuple(
            ParameterSpec(
                spec.name,
                spec.lower * 0.5 if spec.log_scale else spec.lower - 1e-3,
                spec.upper * 2.0 if spec.log_scale else spec.upper + 1e-3,
                log_scale=spec.log_scale,
                unit=spec.unit,
            )
            for spec in space.specs
        )
    )
    warm = cold.fit(lineout, family, space=wide, warm_start=truth)
    assert warm.space is wide and warm.n_starts == 1
    assert warm.goodness.nrmse < 0.01
    by_vector = cold.fit(
        lineout, family, space=wide, warm_start=wide.to_unit(wide.from_dict(truth))
    )
    assert by_vector.goodness.nrmse == pytest.approx(warm.goodness.nrmse, rel=1e-6)


def test_fit_objective_is_the_optimizer_loss(
    supergaussian_lineout, supergaussian_truth
):
    family, truth = supergaussian_truth
    lineout = supergaussian_lineout.trimmed()
    objective = FitObjective(lineout, family)
    u_truth = objective.space.to_unit(objective.space.from_dict(truth))
    loss = objective(u_truth)
    assert 0.0 < loss < 1e-3  # noise only
    assert objective(np.full(objective.dimension, 0.5)) > loss
    assert objective.evaluations == 2
    result = objective.result(u_truth, scheme="oracle")
    assert result.scheme == "oracle" and result.goodness.nrmse < 0.01
    assert result.n_function_evaluations == 2  # result() does not count as a call
    # The loss is the SSE term (normalised by the sum of squared data) plus the
    # smooth worst-point term (default weight 1.0): a weighted power mean of
    # |residual| / peak, standing in for the exact max (see FitObjective).
    sse_term = result.goodness.sse / float(np.sum(lineout.density**2))
    relative_residual = np.abs(result.model_density() - lineout.density) / lineout.peak
    power_mean = np.mean(relative_residual**MAX_DEVIATION_EXPONENT) ** (
        1.0 / MAX_DEVIATION_EXPONENT
    )
    assert loss == pytest.approx(sse_term + power_mean**2, rel=1e-9)


def test_max_deviation_weight_zero_recovers_the_pure_sse_objective(
    supergaussian_lineout, supergaussian_truth
):
    family, truth = supergaussian_truth
    lineout = supergaussian_lineout.trimmed()
    objective = FitObjective(lineout, family, max_deviation_weight=0.0)
    u = objective.space.to_unit(objective.space.from_dict(truth))
    loss = objective(u)
    sse = objective.result(u, scheme="oracle").goodness.sse
    assert loss == pytest.approx(sse / float(np.sum(lineout.density**2)), rel=1e-9)
    with pytest.raises(ValueError, match="non-negative"):
        FitObjective(lineout, family, max_deviation_weight=-1.0)


class _FakeProfile:
    """Just enough of ``_DensityProfile`` for :meth:`FitObjective.evaluate`."""

    def __init__(self, density_fn, z_extent):
        self._density_fn = density_fn
        self._z_extent = z_extent

    def build_density_function(self):
        return self._density_fn

    def get_z_extent(self):
        return self._z_extent


class _ConstantRelativeFamily:
    """A stand-in family whose relative density is a fixed constant everywhere.

    ``build()`` reports the lineout's own extent, so the extent term (default
    ``allowed_extent_ratio=2.0``) stays exactly zero and does not interfere
    with tests isolating the other two terms.
    """

    name = "constant_relative"

    def __init__(self, value: float, z_extent=None) -> None:
        self.value = value
        self.z_extent = z_extent

    def relative_density(self, parameters, z):
        return np.full_like(np.asarray(z, dtype=np.float64), self.value)

    def build(self, parameters, *, nominal_density=1.0, **overrides):
        z_extent = self.z_extent
        return _FakeProfile(lambda z, r: self.relative_density(parameters, z), z_extent)


def test_max_deviation_weight_one_is_calibrated_by_the_flat_constant_case():
    """The docstring's calibration: flat data, constant offset -> equal terms."""
    peak = 3.7
    offset = 0.42
    z = np.linspace(0.0, 1.0, 25)
    lineout = make_lineout(np.ones_like(z), z, amplitude=peak, units="m^-3")
    space = ParameterSpace((ParameterSpec("dummy", 0.0, 1.0),))
    # fit_amplitude=False fixes the amplitude to the data's peak, so the model is
    # exactly peak * (1 + offset / peak) = peak + offset: a constant residual.
    family = _ConstantRelativeFamily(1.0 + offset / peak, z_extent=(z[0], z[-1]))

    sse_only = FitObjective(
        lineout, family, space, fit_amplitude=False, max_deviation_weight=0.0
    )
    combined = FitObjective(lineout, family, space, fit_amplitude=False)
    u = np.array([0.5])

    expected_term = (offset / peak) ** 2
    assert sse_only(u) == pytest.approx(expected_term)
    assert combined(u) == pytest.approx(2.0 * expected_term)  # both terms equal


def test_max_deviation_term_penalises_an_isolated_outlier_more_than_sse_alone():
    """A single badly-missed point barely moves the SSE term but dominates ours."""
    n = 500
    z = np.linspace(0.0, 1.0, n)
    peak = 1.0
    lineout = make_lineout(np.ones_like(z), z, amplitude=peak, units="m^-3")
    space = ParameterSpace((ParameterSpec("dummy", 0.0, 1.0),))

    class _AlmostFlatWithOneOutlier:
        name = "outlier"

        def relative_density(self, parameters, z):
            relative = np.ones_like(np.asarray(z, dtype=np.float64))
            relative[0] = (
                3.0  # with amplitude fixed to peak, this point misses by 2 * peak
            )
            return relative

        def build(self, parameters, *, nominal_density=1.0, **overrides):
            return _FakeProfile(
                lambda z, r: self.relative_density(parameters, z), (z[0], z[-1])
            )

    family = _AlmostFlatWithOneOutlier()
    u = np.array([0.5])
    sse_only = FitObjective(
        lineout, family, space, fit_amplitude=False, max_deviation_weight=0.0
    )(u)
    combined = FitObjective(lineout, family, space, fit_amplitude=False)(u)
    # One point out of n misses by 2 * peak: sse_term = (2 * peak) ** 2 / n, barely
    # visible. The worst-point term is a power mean over all n points (only one of
    # which is nonzero), so it is diluted relative to the true max but still
    # dominates: (2**p / n) ** (2 / p), roughly two orders of magnitude above sse_only.
    expected_max_term = (2.0**MAX_DEVIATION_EXPONENT / n) ** (
        2.0 / MAX_DEVIATION_EXPONENT
    )
    assert sse_only == pytest.approx(4.0 / n)
    assert combined == pytest.approx(sse_only + expected_max_term)
    assert combined > 50 * sse_only  # still overwhelmingly outlier-driven


def test_extent_term_is_a_hinge_on_the_window_width_ratio():
    """A perfect in-window fit whose reported extent is (not) too wide."""
    z = np.linspace(0.0, 1.0, 11)
    window_width = z[-1] - z[0]
    lineout = make_lineout(np.ones_like(z), z, amplitude=1.0, units="m^-3")
    space = ParameterSpace((ParameterSpec("dummy", 0.0, 1.0),))

    class _PerfectFitWithExtent:
        name = "wide"
        extent = (0.0, 0.0)

        def relative_density(self, parameters, z):
            return np.ones_like(np.asarray(z, dtype=np.float64))

        def build(self, parameters, *, nominal_density=1.0, **overrides):
            return _FakeProfile(
                lambda z, r: self.relative_density(parameters, z), self.extent
            )

    family = _PerfectFitWithExtent()
    u = np.array([0.5])
    common = dict(fit_amplitude=False, max_deviation_weight=0.0)

    # Exactly at the default allowed ratio (2x the window): no penalty yet.
    family.extent = (0.0, 2.0 * window_width)
    at_threshold = FitObjective(lineout, family, space, **common)
    assert at_threshold(u) == pytest.approx(0.0, abs=1e-12)

    # Comfortably past it: a quadratic hinge on the excess ratio.
    family.extent = (0.0, 5.0 * window_width)
    over_threshold = FitObjective(lineout, family, space, **common)
    assert over_threshold(u) == pytest.approx(9.0)  # excess = 5 - 2 = 3, 3**2 = 9
    assert over_threshold.allowed_extent_ratio == pytest.approx(2.0)

    # extent_weight=0.0 disables the term regardless of how wide the extent is.
    disabled = FitObjective(lineout, family, space, **common, extent_weight=0.0)
    assert disabled(u) == pytest.approx(0.0)

    # A custom allowed_extent_ratio shifts where the hinge kicks in.
    lenient = FitObjective(lineout, family, space, **common, allowed_extent_ratio=10.0)
    assert lenient(u) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="extent_weight must be non-negative"):
        FitObjective(lineout, family, space, extent_weight=-1.0)
    with pytest.raises(ValueError, match="allowed_extent_ratio must be positive"):
        FitObjective(lineout, family, space, allowed_extent_ratio=0.0)


def test_extent_term_catches_a_realistic_heavy_tailed_lorentzian_term():
    """The motivating case: a term whose shape parameter (m) blows up get_z_extent()."""
    family = GeneralizedLorentzianSumFamily(n_terms=1)
    z = np.linspace(-5.0e-3, 5.0e-3, 401)
    lineout = make_lineout(np.exp(-((z / 1.0e-3) ** 2) / 2.0), z, units="m^-3")
    space = family.parameter_space(lineout.summary())

    # m pinned just above its lower bound: a heavy, slowly-decaying power-law tail.
    extreme = {
        "c_0": 0.0,
        "w_0": 5.0e-4,
        "b_0": 2.0,
        "m_0": space["m_0"].lower + 1.0e-3,
    }
    theta = space.clip(space.from_dict(extreme))
    u = space.to_unit(theta)

    profile = family.build(space.as_dict(theta))
    z_min, z_max = profile.get_z_extent()
    window_width = lineout.z[-1] - lineout.z[0]
    extent_ratio = (z_max - z_min) / window_width
    assert extent_ratio > 50.0  # confirms this really is the pathological case

    with_extent = FitObjective(
        lineout, family, space
    )  # defaults: weight 1.0, ratio 2.0
    without_extent = FitObjective(lineout, family, space, extent_weight=0.0)
    loss_with = with_extent(u)
    loss_without = without_extent(u)
    expected_excess = extent_ratio - with_extent.allowed_extent_ratio
    assert loss_with == pytest.approx(loss_without + expected_excess**2, rel=1e-6)
    assert loss_with > loss_without + 1000.0  # the extent term dominates the total


def test_multi_start_local_fit_exposes_and_threads_the_extent_parameters(
    supergaussian_lineout, supergaussian_truth
):
    family, _truth = supergaussian_truth
    lineout = supergaussian_lineout.trimmed()
    default = MultiStartLocalFit(n_starts=1)
    assert default.extent_weight == pytest.approx(1.0)
    assert default.allowed_extent_ratio == pytest.approx(2.0)

    disabled = MultiStartLocalFit(n_starts=1, extent_weight=0.0)
    result = disabled.fit(lineout, family)
    assert result.goodness.nrmse < 0.05  # still fits fine with the term switched off


def test_compare_families_ranks_the_generating_family_first(supergaussian_lineout):
    lineout = supergaussian_lineout.trimmed()
    names = [
        "asymmetric_sine",
        "conical:supergaussian",
        "generalized_lorentzian_sum[1]",
    ]
    comparison = compare_families(lineout, names, MultiStartLocalFit(n_starts=3))

    assert len(comparison) == 3
    assert comparison.best.family_name == "conical:supergaussian"
    bics = [r.goodness.bic for r in comparison]
    assert bics == sorted(bics)
    assert comparison.rank_by == "bic"
    table = comparison.table()
    assert "conical:supergaussian" in table.splitlines()[2]
    payload = comparison.to_dict(include_profiles=False)
    assert [r["family"] for r in payload["results"]] == [
        r.family_name for r in comparison
    ]
    assert "profile_config" not in payload["results"][0]

    by_nrmse = compare_families(
        lineout,
        [GenericConicalTargetFamily("supergaussian")],
        MultiStartLocalFit(n_starts=1),
        rank_by="nrmse",
    )
    assert by_nrmse.rank_by == "nrmse" and len(by_nrmse) == 1
    with pytest.raises(TypeError):
        compare_families(
            lineout, ["asymmetric_sine", GenericConicalTargetFamily("supergaussian")]
        )


def test_fit_lineouts_pairs_conditions_with_parameter_vectors(
    z_grid, supergaussian_truth
):
    family, truth = supergaussian_truth
    lineouts = []
    for x_mm, fwhm in ((0.0, 2.0e-3), (1.0, 3.0e-3)):
        lineout = make_lineout(
            family.relative_density({**truth, "fwhm": fwhm}, z_grid), z_grid
        )
        lineouts.append(
            lineout.__class__(
                z=lineout.z,
                density=lineout.density,
                density_units=lineout.density_units,
                conditions=LineoutConditions(x_mm=x_mm, pressure_bar=10.0),
            )
        )
    results = fit_lineouts(lineouts, family, MultiStartLocalFit(n_starts=2))
    assert [r.conditions.x_mm for r in results] == [0.0, 1.0]
    thetas = np.array([r.theta for r in results])
    assert thetas.shape == (2, results[0].space.dimension)
    index = results[0].space.index("fwhm")
    assert thetas[:, index] == pytest.approx([2.0e-3, 3.0e-3], rel=0.03)


def test_custom_scheme_satisfies_the_protocol(
    supergaussian_lineout, supergaussian_truth
):
    """A stand-in for a learned model: returns fixed parameters with a fitted amplitude."""
    family, truth = supergaussian_truth

    class Oracle:
        name = "oracle"

        def fit(self, lineout, family):
            space = family.parameter_space(lineout.summary())
            theta = space.from_dict(truth)
            relative = family.relative_density(truth, lineout.z)
            amplitude = optimal_amplitude(
                relative, lineout.density, lineout.effective_weights()
            )
            goodness = GoodnessOfFit.compute(
                lineout.z,
                lineout.density,
                amplitude * relative,
                n_parameters=space.dimension + 1,
            )
            return FitResult(
                family=family,
                lineout=lineout,
                space=space,
                theta=theta,
                amplitude=amplitude,
                goodness=goodness,
                scheme=self.name,
            )

    assert isinstance(Oracle(), FittingScheme)
    comparison = compare_families(supergaussian_lineout, [family], Oracle())
    assert comparison.best.scheme == "oracle"
    assert comparison.best.goodness.nrmse < 0.01


def test_plot_fit_comparison(supergaussian_lineout):
    comparison = compare_families(
        supergaussian_lineout.trimmed(),
        ["asymmetric_sine", "conical:supergaussian"],
        MultiStartLocalFit(n_starts=1),
    )
    figure = plot_fit_comparison(comparison.lineout, comparison.results, top=5)
    ax_fit, ax_res = figure.axes
    assert len(ax_fit.lines) == 3  # data + two models
    assert len(ax_res.lines) == 3  # two residuals + zero line
    assert ax_fit.get_legend() is not None
