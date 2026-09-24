from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from cgns_to_density_hdf5 import CgnsDataError
from shock_evaluator import (
    CgnsScalarFieldData,
    InteractiveShockEvaluator,
    ShockEvaluation,
    density_gradient_magnitude,
    evaluate_oblique_shock,
    initial_log_limits,
    load_cgns_scalar_fields,
    main,
    normalize_lineout_values,
    parse_log_limits,
    perpendicular_segment_shock_direction,
    sample_segment,
    select_upstream_endpoint,
    third_point_shock_direction,
    validate_gamma,
    validate_shock_angle_mode,
)


def _write_cgns_fields(path: Path, *, include_gradients: bool = True) -> None:
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
    z = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    with h5py.File(path, "w") as cgns_file:
        zone = cgns_file.create_group("Base/Zone")
        coordinates = zone.create_group("GridCoordinates")
        coordinates.create_group("CoordinateX").create_dataset(" data", data=x)
        coordinates.create_group("CoordinateY").create_dataset(" data", data=z)
        solution = zone.create_group("FlowSolution.N:1")
        solution.create_group("Density").create_dataset(" data", data=x + 2.0 * z + 1.0)
        solution.create_group("Temperature").create_dataset(" data", data=300.0 + x - z)
        solution.create_group("Mach").create_dataset(" data", data=np.full_like(x, 2.0))
        solution.create_group("Axial_Velocity").create_dataset(" data", data=100.0 - 50.0 * z)
        solution.create_group("Radial_Velocity").create_dataset(" data", data=50.0 * z)
        solution.create_group("Pressure").create_dataset(" data", data=1.0 + z)
        solution.create_group("CoordinateX").create_dataset(" data", data=x)
        solution.create_group("InvalidLength").create_dataset(" data", data=x[:3])
        if include_gradients:
            solution.create_group("dp-dX").create_dataset(" data", data=np.ones_like(x))
            solution.create_group("dp-dY").create_dataset(" data", data=np.full_like(x, 2.0))


class ShockEvaluatorTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_load_discovers_point_aligned_fields_and_ignores_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)

            data = load_cgns_scalar_fields(path)

        self.assertEqual(
            set(data.fields),
            {
                "Density",
                "Temperature",
                "Mach",
                "Axial_Velocity",
                "Radial_Velocity",
                "Pressure",
                "dp-dX",
                "dp-dY",
            },
        )
        np.testing.assert_array_equal(data.x, np.array([0.0, 1.0, 0.0, 1.0, 0.5]))
        np.testing.assert_array_equal(data.z, np.array([0.0, 0.0, 1.0, 1.0, 0.5]))

    def test_load_clips_all_fields_to_requested_x_and_z_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)

            data = load_cgns_scalar_fields(
                path,
                x_bounds=(0.5, 1.0),
                z_bounds=(0.0, 1.0),
            )

        np.testing.assert_array_equal(data.x, np.array([1.0, 1.0, 0.5]))
        np.testing.assert_array_equal(data.z, np.array([0.0, 1.0, 0.5]))
        self.assertEqual(data.fields["Density"].size, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)
            with self.assertRaisesRegex(CgnsDataError, "fewer than three"):
                load_cgns_scalar_fields(path, x_bounds=(0.75, 1.0))

    def test_gradient_magnitude_requires_both_components(self) -> None:
        magnitude = density_gradient_magnitude(
            {"dp-dX": np.array([3.0]), "dp-dY": np.array([4.0])}
        )
        np.testing.assert_array_equal(magnitude, np.array([5.0]))

        with self.assertRaisesRegex(CgnsDataError, "dp-dY"):
            density_gradient_magnitude({"dp-dX": np.array([3.0])})

    def test_log_limits_require_positive_increasing_values(self) -> None:
        lower, upper = initial_log_limits(np.array([0.0, 1.0, 10.0, np.nan]))
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, lower)
        self.assertEqual(parse_log_limits("1e-3", "2.5"), (1e-3, 2.5))

        for limits in (("0", "2"), ("3", "2"), ("not a number", "2")):
            with self.assertRaises(ValueError):
                parse_log_limits(*limits)
        with self.assertRaisesRegex(CgnsDataError, "no positive"):
            initial_log_limits(np.array([0.0, np.nan]))

    def test_segment_samples_follow_distance_in_millimetres(self) -> None:
        points, distance_mm = sample_segment((0.0, 0.0), (0.003, 0.004), count=3)

        np.testing.assert_allclose(
            points,
            np.array([[0.0, 0.0], [0.0015, 0.002], [0.003, 0.004]]),
        )
        np.testing.assert_allclose(distance_mm, np.array([0.0, 2.5, 5.0]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            sample_segment((0.0, 0.0), (0.0, 0.0), count=3)

    def test_normalize_lineout_values_preserves_gaps_and_zero_baseline(self) -> None:
        np.testing.assert_allclose(
            normalize_lineout_values(np.array([0.0, 10.0, np.nan, 20.0])),
            np.array([0.0, 0.5, np.nan, 1.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            normalize_lineout_values(np.array([-4.0, 0.0, 2.0, np.nan])),
            np.array([-1.0, 0.0, 0.5, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            normalize_lineout_values(np.array([0.0, 0.0, np.nan])),
            np.array([0.0, 0.0, np.nan]),
            equal_nan=True,
        )

    def test_controller_updates_checked_lineouts_from_selected_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)
            viewer = InteractiveShockEvaluator(
                load_cgns_scalar_fields(path),
                line_samples=5,
                show_color_limit_boxes=True,
            )

        viewer.segment = ((0.0, 0.0), (1.0, 1.0))
        viewer._update_lineout()
        self.assertEqual([line.get_label() for line in viewer.lineout_axes.lines], ["Density"])
        np.testing.assert_allclose(
            viewer.lineout_axes.lines[0].get_ydata(),
            np.linspace(1.0, 4.0, 5),
        )

        viewer.field_checkboxes.set_active(list(viewer.data.fields).index("Temperature"))
        self.assertEqual(len(viewer.lineout_axes.lines), 2)
        self.assertEqual(
            viewer.lineout_axes.get_ylabel(),
            "independently magnitude-normalized field value",
        )
        self.assertNotEqual(viewer.lineout_axes.get_ylim(), (-1.05, 1.05))
        for line in viewer.lineout_axes.lines:
            finite_values = line.get_ydata()[np.isfinite(line.get_ydata())]
            self.assertGreaterEqual(finite_values.min(), 0.0)
            self.assertLessEqual(finite_values.max(), 1.0)
        assert viewer.lower_limit_box is not None
        assert viewer.upper_limit_box is not None
        viewer.lower_limit_box.set_val("1")
        viewer.upper_limit_box.set_val("10")
        self.assertEqual(viewer.limits, (1.0, 10.0))

    def test_color_limit_boxes_are_hidden_unless_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)
            data = load_cgns_scalar_fields(path)
            default_viewer = InteractiveShockEvaluator(data)
            interactive_viewer = InteractiveShockEvaluator(
                data,
                show_color_limit_boxes=True,
            )

        self.assertIsNone(default_viewer.lower_limit_box)
        self.assertIsNone(default_viewer.upper_limit_box)
        self.assertIsNotNone(interactive_viewer.lower_limit_box)
        self.assertIsNotNone(interactive_viewer.upper_limit_box)

    def test_controller_uses_one_figure_with_enlarged_shock_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)
            viewer = InteractiveShockEvaluator(load_cgns_scalar_fields(path))

        self.assertIs(viewer.figure, viewer.heatmap_figure)
        self.assertIs(viewer.figure, viewer.lineout_figure)
        self.assertIs(viewer.heatmap_axes.figure, viewer.lineout_axes.figure)
        self.assertEqual(viewer.shock_report_text.get_fontsize(), 11.0)

    def test_oblique_shock_evaluation_calculates_angle_and_ratios(self) -> None:
        result = evaluate_oblique_shock(
            gamma=1.4,
            mach_upstream=2.0,
            upstream_velocity=np.array([100.0, 0.0]),
            downstream_velocity=np.array([50.0, 50.0]),
            upstream_pressure=1.0,
            downstream_pressure=2.0,
            upstream_density=1.0,
            downstream_density=1.5,
            upstream_temperature=1.0,
            downstream_temperature=1.25,
        )

        self.assertAlmostEqual(np.rad2deg(result.beta_rad), 45.0)
        self.assertEqual(result.pressure_ratio_simulated, 2.0)
        self.assertEqual(result.density_ratio_simulated, 1.5)
        self.assertEqual(result.temperature_ratio_simulated, 1.25)
        self.assertAlmostEqual(result.pressure_ratio_predicted, 2.166666666666666)
        self.assertAlmostEqual(result.density_ratio_predicted, 1.714285714285714)
        self.assertAlmostEqual(result.temperature_ratio_predicted, 1.2638888888888888)
        self.assertEqual(validate_gamma(1.67), 1.67)
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            validate_gamma(1.0)
        self.assertEqual(validate_shock_angle_mode("auto"), "auto")
        with self.assertRaisesRegex(ValueError, "auto, perp, manual"):
            validate_shock_angle_mode("invalid")

    def test_shock_report_uses_aligned_fixed_width_columns(self) -> None:
        evaluation = ShockEvaluation(
            gamma=1.532,
            mach_upstream=3.3076,
            beta_rad=np.deg2rad(88.083),
            velocity_change=np.array([1.0, 2.0]),
            pressure_ratio_simulated=0.022930661,
            density_ratio_simulated=0.084918933,
            temperature_ratio_simulated=0.270043084,
            pressure_ratio_predicted=0.36077325,
            density_ratio_predicted=0.5306587,
            temperature_ratio_predicted=0.6798593,
        )

        report_rows = InteractiveShockEvaluator._format_shock_report(evaluation).splitlines()
        data_rows = report_rows[3:]
        for row in data_rows:
            observed_column = row[19:37]
            predicted_column = row[37:55]
            percent_error_column = row[55:73]
            self.assertEqual(len(observed_column), 18)
            self.assertEqual(len(predicted_column), 18)
            self.assertEqual(len(percent_error_column), 18)
            self.assertIn("e", observed_column)
            self.assertIn("e", predicted_column)
            self.assertIn("e", percent_error_column)
        self.assertAlmostEqual(
            float(data_rows[0][55:73]),
            abs(0.022930661 - 0.36077325) / 0.36077325 * 100.0,
            places=4,
        )

    def test_perpendicular_mode_selects_higher_mach_endpoint_as_upstream(self) -> None:
        start_is_upstream, endpoint = select_upstream_endpoint(1.0, 3.0)
        self.assertFalse(start_is_upstream)
        self.assertEqual(endpoint, "drag end")
        beta_rad, direction = perpendicular_segment_shock_direction(
            (0.0, 0.0),
            (0.0, 1.0),
            np.array([100.0, 0.0]),
        )
        self.assertAlmostEqual(np.rad2deg(beta_rad), 90.0)
        np.testing.assert_allclose(direction, np.array([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "equal Mach"):
            select_upstream_endpoint(2.0, 2.0)

    def test_third_point_shock_direction_uses_midpoint_ray(self) -> None:
        beta_rad, direction = third_point_shock_direction(
            (0.0, 0.0),
            (0.0, 2.0),
            (1.0, 1.0),
            np.array([100.0, 0.0]),
        )

        self.assertAlmostEqual(np.rad2deg(beta_rad), 90.0)
        np.testing.assert_allclose(direction, np.array([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "differ from the segment midpoint"):
            third_point_shock_direction(
                (0.0, 0.0),
                (0.0, 2.0),
                (0.0, 1.0),
                np.array([100.0, 0.0]),
            )

    def test_controller_third_point_mode_waits_then_evaluates_direction(self) -> None:
        x = np.array([0.0, 1.0, 0.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0])
        fields = {
            "Density": 1.0 + x,
            "dp-dX": np.ones(4),
            "dp-dY": np.ones(4),
            "Mach": np.full(4, 2.0),
            "Axial_Velocity": np.full(4, 100.0),
            "Radial_Velocity": np.zeros(4),
            "Pressure": 1.0 + x,
            "Temperature": 1.0 + x,
        }
        viewer = InteractiveShockEvaluator(
            CgnsScalarFieldData(x=x, z=z, fields=fields),
            shock_angle="manual",
        )
        viewer.segment = ((0.0, 0.0), (0.0, 1.0))
        viewer._third_point = None

        viewer._update_shock_evaluation()

        self.assertIsNone(viewer.shock_evaluation)
        self.assertIn("third point", viewer.shock_report_text.get_text())
        viewer._third_point = (1.0, 0.5)
        viewer._update_shock_evaluation()

        self.assertIsNotNone(viewer.shock_evaluation)
        assert viewer.shock_evaluation is not None
        self.assertEqual(viewer.shock_evaluation.angle_source, "third point")
        self.assertAlmostEqual(np.rad2deg(viewer.shock_evaluation.beta_rad), 90.0)
        np.testing.assert_allclose(viewer.shock_direction_artist.get_xdata(), [0.0, 1.0])
        np.testing.assert_allclose(viewer.shock_direction_point_artist.get_ydata(), [0.5])

    def test_manual_third_point_click_refreshes_lineout_window(self) -> None:
        x = np.array([0.0, 1.0, 0.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0])
        fields = {
            "Density": 1.0 + x,
            "dp-dX": np.ones(4),
            "dp-dY": np.ones(4),
            "Mach": np.full(4, 2.0),
            "Axial_Velocity": np.full(4, 100.0),
            "Radial_Velocity": np.zeros(4),
            "Pressure": 1.0 + x,
            "Temperature": 1.0 + x,
        }
        viewer = InteractiveShockEvaluator(
            CgnsScalarFieldData(x=x, z=z, fields=fields),
            shock_angle="manual",
        )
        viewer.segment = ((0.0, 0.0), (0.0, 1.0))
        viewer.heatmap_figure.canvas.draw()
        third_point_pixel = viewer.heatmap_axes.transData.transform((1.0, 0.5))
        event = MouseEvent(
            "button_press_event",
            viewer.heatmap_figure.canvas,
            *third_point_pixel,
            button=1,
        )

        with patch.object(viewer, "_update_lineout") as update_lineout:
            viewer._on_press(event)

        update_lineout.assert_called_once_with()
        self.assertIsNotNone(viewer.shock_evaluation)

    def test_controller_perpendicular_mode_keeps_drag_endpoint_ratios(self) -> None:
        x = np.array([0.0, 1.0, 0.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0])
        fields = {
            "Density": 1.0 + x,
            "dp-dX": np.ones(4),
            "dp-dY": np.ones(4),
            "Mach": 1.0 + 2.0 * x,
            "Axial_Velocity": np.full(4, 100.0),
            "Radial_Velocity": np.zeros(4),
            "Pressure": 1.0 + x,
            "Temperature": 1.0 + x,
        }
        viewer = InteractiveShockEvaluator(
            CgnsScalarFieldData(x=x, z=z, fields=fields),
            shock_angle="perp",
        )
        viewer.segment = ((0.0, 0.0), (0.0, 1.0))

        viewer._update_shock_evaluation()

        self.assertIsNotNone(viewer.shock_evaluation)
        assert viewer.shock_evaluation is not None
        self.assertEqual(viewer.shock_evaluation.upstream_endpoint, "drag end")
        self.assertEqual(viewer.shock_evaluation.mach_upstream, 3.0)
        self.assertAlmostEqual(np.rad2deg(viewer.shock_evaluation.beta_rad), 90.0)
        self.assertEqual(viewer.shock_evaluation.pressure_ratio_simulated, 2.0)
        self.assertEqual(viewer.shock_evaluation.angle_source, "segment normal")

    def test_controller_reports_shock_state_after_segment_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)
            viewer = InteractiveShockEvaluator(
                load_cgns_scalar_fields(path),
                line_samples=5,
                gamma=1.67,
            )

        viewer.segment = ((0.0, 0.0), (1.0, 1.0))
        viewer._update_shock_evaluation()

        self.assertIsNotNone(viewer.shock_evaluation)
        assert viewer.shock_evaluation is not None
        self.assertEqual(viewer.shock_evaluation.gamma, 1.67)
        self.assertAlmostEqual(np.rad2deg(viewer.shock_evaluation.beta_rad), 45.0)
        self.assertEqual(len(viewer.shock_direction_artist.get_xdata()), 2)
        self.assertIn("drag-end/start", viewer.shock_report_text.get_text())

    def test_main_retains_drag_callbacks_through_show(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.cgns"
            _write_cgns_fields(path)

            def exercise_drag() -> None:
                gc.collect()
                self.assertEqual(len(plt.get_fignums()), 1)
                figure = plt.figure(plt.get_fignums()[0])
                heatmap_axes = figure.axes[0]
                canvas = figure.canvas
                canvas.draw()
                start = heatmap_axes.transData.transform((0.0, 0.0))
                end = heatmap_axes.transData.transform((1.0, 1.0))
                for event_name, position in (
                    ("button_press_event", start),
                    ("motion_notify_event", end),
                    ("button_release_event", end),
                ):
                    canvas.callbacks.process(
                        event_name,
                        MouseEvent(event_name, canvas, *position, button=1),
                    )
                self.assertEqual(len(figure.axes[1].lines), 1)

            with (
                patch.object(sys, "argv", ["shock_evaluator.py", str(path)]),
                patch("shock_evaluator.plt.show", side_effect=exercise_drag),
            ):
                main()


if __name__ == "__main__":
    unittest.main()