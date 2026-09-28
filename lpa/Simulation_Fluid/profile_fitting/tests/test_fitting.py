from __future__ import annotations

import json

import numpy as np
import pytest

from fludat_fit.families import GenericConicalTargetFamily, get_families
from fludat_fit.fitting import (
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
    profile = result.build_profile(species="H", ionization=0, p_nz=2)
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
    # The loss is the SSE normalised by the sum of squared data.
    sse = result.goodness.sse
    assert loss == pytest.approx(sse / float(np.sum(lineout.density**2)), rel=1e-9)


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
