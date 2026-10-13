from __future__ import annotations

import csv
import json
import math

import pytest

from fludat_fit.cli_common import FitWindow
from fludat_fit.dataset import MAX_ANGLE_DEG, NozzleDataset
from fludat_fit.families import get_families
from fludat_fit.fit_statistics import (
    ConditionRanges,
    FitStatistics,
    ParameterStats,
    fit_points,
    main,
)
from fludat_fit.fitting import MultiStartLocalFit
from fludat_fit.lineout import LineoutConditions

FAMILIES = ["conical:supergaussian", "asymmetric_sine", "generalized_lorentzian_sum[1]"]


def test_ranges_from_dataset_and_validation(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    ranges = ConditionRanges.from_dataset(dataset)
    assert ranges.x_mm == (0.0, 2.0)
    assert ranges.pressure_bar == (5.0, 10.0)
    assert ranges.angle_deg == (0.0, 0.0)  # axial lineouts unless asked otherwise
    assert ranges.varying_axes == ["x_mm", "pressure_bar"]

    tilted = ConditionRanges.from_dataset(
        dataset, x_range=(1.0, 1.0), angle_range=(20.0, -20.0)
    )
    assert tilted.varying_axes == ["pressure_bar", "angle_deg"]
    assert tilted.angle_deg == (-20.0, 20.0)
    with pytest.warns(UserWarning, match="clamped"):
        clamped = ConditionRanges.from_dataset(dataset, pressure_range=(1.0, 10.0))
    assert clamped.pressure_bar == (5.0, 10.0)
    with pytest.warns(UserWarning, match="clamped"):
        steep = ConditionRanges.from_dataset(dataset, angle_range=(0.0, 120.0))
    assert steep.angle_deg == (0.0, MAX_ANGLE_DEG)
    with pytest.raises(ValueError, match="does not overlap"):
        ConditionRanges.from_dataset(dataset, pressure_range=(20.0, 30.0))


def test_sampling_methods():
    ranges = ConditionRanges((0.0, 2.0), (5.0, 5.0), (-10.0, 10.0))
    lhs = ranges.sample(8, "lhs", seed=1)
    assert len(lhs) == 8
    assert all(
        0.0 <= c.x_mm <= 2.0 and c.pressure_bar == 5.0 and -10.0 <= c.angle_deg <= 10.0
        for c in lhs
    )
    assert lhs == ranges.sample(8, "lhs", seed=1)
    assert lhs != ranges.sample(8, "lhs", seed=2)
    assert len(ranges.sample(5, "random", seed=0)) == 5

    grid = ranges.sample(5, "grid")
    assert len(grid) == 9  # ceil(sqrt(5)) = 3 per varying axis
    assert {c.x_mm for c in grid} == {0.0, 1.0, 2.0}
    assert {c.angle_deg for c in grid} == {-10.0, 0.0, 10.0}

    fixed = ConditionRanges((1.0, 1.0), (5.0, 5.0))
    assert fixed.sample(10) == [LineoutConditions(1.0, 5.0, 0.0)]
    with pytest.raises(ValueError):
        ranges.sample(0)
    with pytest.raises(ValueError):
        ranges.sample(3, "sobol")  # type: ignore[arg-type]


@pytest.fixture
def statistics(small_cube_path) -> FitStatistics:
    dataset = NozzleDataset.load(small_cube_path)
    # Axial lineouts keep the truth a pure Gaussian in the path coordinate.
    ranges = ConditionRanges.from_dataset(dataset, x_range=(0.5, 1.5))
    points = ranges.sample(4, "lhs", seed=0)
    comparisons = fit_points(
        dataset,
        points,
        get_families(FAMILIES),
        MultiStartLocalFit(n_starts=1),
        FitWindow(max_points=120),
    )
    return FitStatistics(
        comparisons, ranges=ranges, sampling="lhs", dataset_name=dataset.name
    )


def test_statistics_summary(statistics):
    assert statistics.n_points == 4
    assert set(statistics.family_names) == set(FAMILIES)
    summary = statistics.summary()
    assert [s.family for s in summary][0] == "conical:supergaussian"
    best = summary[0]
    assert best.n_points == 4 and best.best_fraction == 1.0 and best.mean_rank == 1.0
    assert best.success_rate == 1.0
    assert best.metrics["nrmse"].max < 1e-3
    assert best.ranking_key.median == pytest.approx(best.metrics["bic"].median)
    worst = summary[-1]
    assert worst.mean_rank > 1.0
    assert worst.metrics["nrmse"].median > best.metrics["nrmse"].median
    assert statistics.best_family_counts() == {"conical:supergaussian": 4}

    table = statistics.table()
    assert table.splitlines()[2].split()[1] == "conical:supergaussian"
    payload = statistics.to_dict()
    assert payload["n_points"] == 4 and payload["ranges"]["x_mm"] == [0.5, 1.5]
    assert payload["ranges"]["angle_deg"] == [0.0, 0.0]
    first = payload["points"][0]["results"][0]
    assert len(payload["points"]) == 4 and "profile_config" not in first
    with_profiles = statistics.to_dict(include_profiles=True)
    assert "profile_config" in with_profiles["points"][0]["results"][0]
    json.dumps(payload)


def test_statistics_rows_and_plot(statistics, tmp_path):
    rows = statistics.rows()
    assert len(rows) == 4 * len(FAMILIES)
    first = rows[0]
    assert first["rank"] == 1 and first["family"] == "conical:supergaussian"
    expected = {
        "x_mm",
        "pressure_bar",
        "angle_deg",
        "nrmse",
        "bic",
        "amplitude",
        "fwhm",
    }
    assert expected <= set(first)
    path = statistics.write_csv(tmp_path / "out" / "stats.csv")
    with path.open() as handle:
        read = list(csv.DictReader(handle))
    assert len(read) == len(rows)
    assert "peak_z0" in read[0] and read[0]["peak_z0"] == ""  # other family's parameter
    assert float(read[0]["fwhm"]) == pytest.approx(first["fwhm"])

    figure = statistics.plot(condition_axis="x_mm", top=2)
    assert len(figure.axes) == 3
    assert len(figure.axes[2].collections) == 2  # two scatter series

    # Labels carry each family's free-parameter count (shape params + amplitude,
    # see FitObjective.n_parameters), e.g. "conical:supergaussian (N)" -- but
    # results_of() lookups inside plot() must still use the plain family name.
    summaries = {s.family: s.n_parameters for s in statistics.summary()}
    y_labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
    assert y_labels == [f"{family} ({n})" for family, n in summaries.items()]
    legend_labels = [
        text.get_text() for text in figure.axes[2].get_legend().get_texts()
    ]
    assert legend_labels == y_labels[:2]


def test_parameter_stats_of_computes_a_confidence_window_around_the_median():
    stats = ParameterStats.of([0.0, 1.0, 2.0, 3.0, 4.0], confidence=0.8)
    assert stats.n == 5
    assert stats.median == pytest.approx(2.0)
    assert stats.confidence == pytest.approx(0.8)
    # 10th/90th percentile of [0..4]
    assert stats.ci[0] == pytest.approx(0.4) and stats.ci[1] == pytest.approx(3.6)
    assert stats.to_dict() == {
        "n": 5,
        "median": pytest.approx(2.0),
        "confidence": pytest.approx(0.8),
        "ci": [pytest.approx(0.4), pytest.approx(3.6)],
    }


def test_parameter_stats_of_rejects_an_invalid_confidence_level():
    with pytest.raises(ValueError, match="confidence"):
        ParameterStats.of([1.0, 2.0], confidence=1.0)


def test_parameter_stats_of_handles_all_nan_input():
    stats = ParameterStats.of([float("nan"), float("nan")])
    assert stats.n == 0
    assert math.isnan(stats.median)
    assert math.isnan(stats.ci[0]) and math.isnan(stats.ci[1])


def test_parameter_distribution_covers_every_family_and_its_own_parameters(
    statistics,
):
    distribution = statistics.parameter_distribution(confidence=0.95)

    assert set(distribution) == set(statistics.family_names)
    for family in statistics.family_names:
        entry = distribution[family]
        results = [result for _, result in statistics.results_of(family)]
        assert entry["n_points"] == len(results)
        assert set(entry["parameters"]) == {"amplitude", *results[0].parameters}
        for name, stats in entry["parameters"].items():
            assert stats["n"] == len(results)
            assert stats["confidence"] == pytest.approx(0.95)
            assert stats["ci"][0] <= stats["median"] <= stats["ci"][1]


def test_cli_writes_param_stats_json(small_cube_path, tmp_path):
    output = tmp_path / "params.json"
    code = main(
        [
            str(small_cube_path),
            "--families",
            *FAMILIES,
            "--samples",
            "4",
            "--starts",
            "2",
            "--param-stats",
            str(output),
            "--confidence",
            "0.9",
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text())
    assert set(payload) == set(FAMILIES)
    for entry in payload.values():
        for stats in entry["parameters"].values():
            assert stats["confidence"] == pytest.approx(0.9)


def test_fit_points_skips_empty_lineouts_and_runs_in_parallel(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    points = [LineoutConditions(1.0, 10.0), LineoutConditions(1.0, 5.0, 15.0)]
    comparisons = fit_points(
        dataset,
        points,
        get_families(["asymmetric_sine"]),
        MultiStartLocalFit(n_starts=1),
        FitWindow(max_points=80),
        workers=2,
    )
    assert [c.lineout.conditions.pressure_bar for c in comparisons] == [10.0, 5.0]
    assert comparisons[1].lineout.conditions.angle_deg == 15.0
    # A lineout with no density is skipped.
    zero = dataset.lineout(LineoutConditions(1.0, 10.0))
    dataset.lineout = lambda conditions, t_values=None: zero.__class__(  # type: ignore[method-assign]
        z=zero.z,
        density=0.0 * zero.density,
        density_units=zero.density_units,
        conditions=conditions,
    )
    empty = fit_points(
        dataset,
        [LineoutConditions(1.0, 10.0)],
        get_families(["asymmetric_sine"]),
        MultiStartLocalFit(n_starts=1),
    )
    assert empty == []
    with pytest.raises(ValueError, match="no comparisons"):
        FitStatistics([])


def test_cli_end_to_end(small_cube_path, tmp_path, capsys):
    json_path = tmp_path / "stats.json"
    csv_path = tmp_path / "stats.csv"
    plot_path = tmp_path / "stats.png"
    code = main(
        [
            str(small_cube_path),
            "--x-range",
            "0",
            "2",
            "--pressure-range",
            "10",
            "10",
            "--angle-range",
            "-15",
            "15",
            "--samples",
            "4",
            "--sampling",
            "grid",
            "--families",
            *FAMILIES,
            "--starts",
            "1",
            "--max-points",
            "100",
            "--quiet",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
            "--plot",
            str(plot_path),
            "--plot-axis",
            "angle_deg",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "best family counts:" in out
    payload = json.loads(json_path.read_text())
    assert payload["sampling"] == "grid" and payload["n_points"] == 4
    assert payload["ranges"]["angle_deg"] == [-15.0, 15.0]
    assert {p["conditions"]["pressure_bar"] for p in payload["points"]} == {10.0}
    angles = sorted({p["conditions"]["angle_deg"] for p in payload["points"]})
    assert angles == [-15.0, 15.0]
    assert sorted({p["conditions"]["x_mm"] for p in payload["points"]}) == [0.0, 2.0]
    assert csv_path.is_file() and plot_path.is_file()


def test_cli_list_and_errors(capsys):
    assert main(["--list-families"]) == 0
    assert "asymmetric_sine" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--samples", "2"])
