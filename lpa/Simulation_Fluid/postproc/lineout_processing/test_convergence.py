from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import convergence
from ansys_lineouts import parse_lineout_file
from convergence import (
    calculate_field_convergence,
    calculate_convergence,
    deduplicate_and_sort_profile,
    load_resolution_cgns_fields,
    load_resolution_lineouts,
    parse_grid_size_filename,
    plot_convergence,
    relative_integrated_difference,
    summarize_convergence,
)


def _write_lineout(path: Path, sections: dict[str, list[tuple[float, float]]]) -> None:
    lines = ['(title "Density")', '(labels "Position" "Density")', ""]
    for label, points in sections.items():
        lines.append(f'((xy/key/label "{label}")')
        lines.extend(f"{z} {density}" for z, density in points)
        lines.append(")")
        lines.append("")
    path.write_text("\n".join(lines))


def _write_cgns_field(path: Path, density_scale: float) -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
        ]
    )
    with h5py.File(path, "w") as file:
        zone = file.create_group("Base/Zone")
        coordinates = zone.create_group("GridCoordinates")
        coordinates.create_group("CoordinateX").create_dataset(
            " data", data=points[:, 0]
        )
        coordinates.create_group("CoordinateY").create_dataset(
            " data", data=points[:, 1]
        )
        solution = zone.create_group("FlowSolution.N:1")
        solution.create_group("Density").create_dataset(
            " data", data=density_scale * (points[:, 0] + 2 * points[:, 1] + 3)
        )


class AnsysLineoutTests(unittest.TestCase):
    def test_parser_returns_labeled_float64_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0_1.txt"
            _write_lineout(path, {"x-0-0": [(0.0, 1.0), (1.0, 2.0)]})

            sections = parse_lineout_file(path)

        np.testing.assert_array_equal(
            sections["x-0-0"], np.array([[0.0, 1.0], [1.0, 2.0]])
        )
        self.assertEqual(sections["x-0-0"].dtype, np.float64)

    def test_parser_rejects_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.txt"
            _write_lineout(
                path,
                {"x-0-0": [(0.0, 1.0)], "x-0-0 ": [(1.0, 2.0)]},
            )
            path.write_text(
                path.read_text().replace('"x-0-0 "', '"x-0-0"')
            )

            with self.assertRaisesRegex(ValueError, "Duplicate lineout section"):
                parse_lineout_file(path)


class ConvergenceCalculationTests(unittest.TestCase):
    def test_parse_grid_size_filename(self) -> None:
        self.assertEqual(parse_grid_size_filename(Path("0_075.txt")), 0.075)
        with self.assertRaisesRegex(ValueError, "maximum grid size"):
            parse_grid_size_filename(Path("coarse.txt"))

    def test_deduplicate_and_sort_profile_keeps_last_value(self) -> None:
        profile = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 3.0]])

        result = deduplicate_and_sort_profile(profile)

        np.testing.assert_array_equal(result, np.array([[0.0, 0.0], [1.0, 3.0]]))

    def test_relative_integrated_difference_interpolates_shared_grid(self) -> None:
        coarse = np.array([[0.0, 0.0], [1.0, 2.0]])
        finer = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

        error, z_min, z_max = relative_integrated_difference(coarse, finer)

        self.assertAlmostEqual(error, 1.0)
        self.assertEqual((z_min, z_max), (0.0, 1.0))

    def test_load_and_calculate_default_to_finest_resolution(self) -> None:
        sections = {"x-0-0": [(0.0, 0.0), (1.0, 1.0)]}
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_lineout(input_dir / "0_1.txt", {"x-0-0": [(0.0, 0.0), (1.0, 3.0)]})
            _write_lineout(input_dir / "0_05.txt", {"x-0-0": [(0.0, 0.0), (1.0, 2.0)]})
            _write_lineout(input_dir / "0_02.txt", sections)
            (input_dir / "convergence_metrics.csv").write_text("generated output")

            metrics = calculate_convergence(load_resolution_lineouts(input_dir))

        self.assertEqual(
            [(metric.coarse_grid_size_mm, metric.finer_grid_size_mm) for metric in metrics],
            [(0.1, 0.02), (0.05, 0.02)],
        )
        self.assertEqual([metric.relative_error for metric in metrics], [2.0, 1.0])

    def test_calculate_can_use_adjacent_resolutions(self) -> None:
        sections = {"x-0-0": [(0.0, 0.0), (1.0, 1.0)]}
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_lineout(input_dir / "0_1.txt", {"x-0-0": [(0.0, 0.0), (1.0, 3.0)]})
            _write_lineout(input_dir / "0_05.txt", {"x-0-0": [(0.0, 0.0), (1.0, 2.0)]})
            _write_lineout(input_dir / "0_02.txt", sections)

            metrics = calculate_convergence(
                load_resolution_lineouts(input_dir), reference_mode="adjacent"
            )

        self.assertEqual(
            [(metric.coarse_grid_size_mm, metric.finer_grid_size_mm) for metric in metrics],
            [(0.1, 0.05), (0.05, 0.02)],
        )
        self.assertEqual([metric.relative_error for metric in metrics], [0.5, 1.0])

    def test_load_rejects_inconsistent_lineout_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_lineout(input_dir / "0_1.txt", {"x-0-0": [(0.0, 1.0), (1.0, 1.0)]})
            _write_lineout(input_dir / "0_05.txt", {"x-0-5": [(0.0, 1.0), (1.0, 1.0)]})

            with self.assertRaisesRegex(ValueError, "do not match"):
                load_resolution_lineouts(input_dir)

    def test_cgns_fields_use_resolution_names_and_support_x_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_cgns_field(input_dir / "0_2.cgns", density_scale=1.0)
            _write_cgns_field(input_dir / "0_1.cgns", density_scale=1.1)

            metrics = calculate_field_convergence(
                load_resolution_cgns_fields(input_dir),
                x_min=0.0,
                x_points=20,
                z_points=20,
            )

        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.lineout_label, "field")
        self.assertEqual(metric.x_min_m, 0.0)
        self.assertEqual(metric.x_max_m, 1.0)
        self.assertAlmostEqual(metric.relative_error, 1 / 11)

    def test_cgns_convergence_triangulates_each_resolution_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_cgns_field(input_dir / "0_3.cgns", density_scale=1.0)
            _write_cgns_field(input_dir / "0_2.cgns", density_scale=1.1)
            _write_cgns_field(input_dir / "0_1.cgns", density_scale=1.2)

            with patch("convergence.Delaunay", wraps=convergence.Delaunay) as delaunay:
                metrics = calculate_field_convergence(
                    load_resolution_cgns_fields(input_dir),
                    x_min=0.0,
                    x_points=20,
                    z_points=20,
                )

        self.assertEqual(len(metrics), 2)
        self.assertEqual(delaunay.call_count, 3)

    def test_cgns_convergence_supports_cubic_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_cgns_field(input_dir / "0_2.cgns", density_scale=1.0)
            _write_cgns_field(input_dir / "0_1.cgns", density_scale=1.1)

            metrics = calculate_field_convergence(
                load_resolution_cgns_fields(input_dir),
                x_min=0.0,
                x_points=20,
                z_points=20,
                interpolation="cubic",
            )

        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics[0].relative_error, 0.0)

    def test_cgns_summary_uses_pointwise_error_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_cgns_field(input_dir / "0_2.cgns", density_scale=1.0)
            _write_cgns_field(input_dir / "0_1.cgns", density_scale=1.0)
            fields = load_resolution_cgns_fields(input_dir)
            coarse = fields[1]
            fields[1] = convergence.ResolutionField(
                grid_size_mm=coarse.grid_size_mm,
                path=coarse.path,
                x=coarse.x,
                z=coarse.z,
                density=coarse.density * (1.0 + 0.1 * coarse.x),
            )

            metrics = calculate_field_convergence(
                fields,
                x_min=0.0,
                x_points=20,
                z_points=20,
            )
            summaries = summarize_convergence(metrics)

        self.assertEqual(len(summaries), 1)
        self.assertGreater(summaries[0].standard_deviation, 0.0)
        self.assertNotAlmostEqual(
            summaries[0].mean_relative_error,
            metrics[0].relative_error,
        )

    def test_peak_normalized_error_averages_before_normalizing(self) -> None:
        coarse_density = np.array([[1.0, 3.0], [6.0, 8.0]])
        finer_density = np.array([[2.0, 5.0], [10.0, 14.0]])

        distribution = convergence._peak_normalized_absolute_error_distribution(
            coarse_density, finer_density
        )

        self.assertEqual(distribution.reference_peak_density, 14.0)
        self.assertEqual(distribution.sample_count, 4)
        self.assertAlmostEqual(distribution.mean_absolute_error, 3.25)
        self.assertAlmostEqual(
            distribution.standard_deviation_absolute_error, np.sqrt(3.6875)
        )
        self.assertAlmostEqual(distribution.mean_relative_error, 3.25 / 14.0)
        self.assertAlmostEqual(
            distribution.standard_deviation_relative_error, np.sqrt(3.6875) / 14.0
        )

    def test_cgns_summary_plot_can_select_field_error_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            _write_cgns_field(input_dir / "0_2.cgns", density_scale=1.0)
            _write_cgns_field(input_dir / "0_1.cgns", density_scale=1.0)
            fields = load_resolution_cgns_fields(input_dir)
            coarse = fields[1]
            fields[1] = convergence.ResolutionField(
                grid_size_mm=coarse.grid_size_mm,
                path=coarse.path,
                x=coarse.x,
                z=coarse.z,
                density=coarse.density * (1.0 + 0.1 * coarse.x),
            )
            metrics = calculate_field_convergence(
                fields, x_min=0.0, x_points=20, z_points=20
            )
            summaries = summarize_convergence(metrics)
            figures = {
                field_error: plot_convergence(
                    metrics,
                    summaries,
                    view="summary",
                    field_error=field_error,
                )
                for field_error in ("local-relative", "peak-normalized", "both")
            }

        for field_error in ("local-relative", "peak-normalized", "both"):
            self.assertEqual(len(figures[field_error].axes), 1)
            self.assertEqual(figures[field_error].axes[0].get_title(), "Mean field errors")
        self.assertEqual(len(figures["local-relative"].axes[0].collections), 1)
        self.assertEqual(len(figures["local-relative"].axes[0].lines), 1)
        self.assertEqual(len(figures["peak-normalized"].axes[0].collections), 1)
        self.assertEqual(len(figures["peak-normalized"].axes[0].lines), 1)
        self.assertEqual(len(figures["both"].axes[0].collections), 2)
        self.assertEqual(len(figures["both"].axes[0].lines), 2)

    def test_summary_plot_uses_meaningful_log_y_limits(self) -> None:
        summaries = [
            convergence.ConvergenceSummary(
                coarse_grid_size_mm=0.1,
                finer_grid_size_mm=0.01,
                mean_relative_error=0.02,
                standard_deviation=0.03,
                minimum_relative_error=0.0,
                maximum_relative_error=0.05,
            )
        ]

        figure = plot_convergence([], summaries, view="summary")

        np.testing.assert_allclose(figure.axes[0].get_ylim(), (0.01, 0.1))


if __name__ == "__main__":
    unittest.main()