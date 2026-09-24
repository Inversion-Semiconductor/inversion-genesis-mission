#!/usr/bin/env python3
"""Interactively inspect density-gradient shocks in an unstructured CGNS field.

Example:

	python shock_evaluator.py data/generic_field_raw/5_bar_htu_all_fields.cgns
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
from matplotlib.widgets import CheckButtons, TextBox
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from cgns_to_density_hdf5 import CgnsDataError, _read_cgns_data

DEFAULT_ZONE_PATH = "Base/Zone"
DEFAULT_FLOW_SOLUTION = "FlowSolution.N:1"
DEFAULT_X_COORDINATE = "CoordinateX"
DEFAULT_Z_COORDINATE = "CoordinateY"
DEFAULT_DENSITY_X_GRADIENT = "dp-dX"
DEFAULT_DENSITY_Y_GRADIENT = "dp-dY"
COORDINATE_FIELD_NAMES = {"CoordinateX", "CoordinateY", "CoordinateZ"}
SHOCK_ANGLE_MODES = ("auto", "perp", "manual")

SHOCK_STATE_FIELD_NAMES = (
	"Mach",
	"Axial_Velocity",
	"Radial_Velocity",
	"Pressure",
	"Density",
	"Temperature",
)
# Oblique Shock Equations

def oblique_p2p1(gamma: float, m1: float, beta: float) -> float:
	"""
	Returns the ratio of the downstream to the upstream (static) pressure over an oblique shock given...
	
	Parameters
	----------
	gamma : float
        Heat capacity ratio.
	m1: float
        The upstream mach number.
	beta: float
        The angle of the shock relative to the upstream flow axis.
		
	Returns
	---------
    float: The ratio of the downstream pressure to the upstream pressure.
	"""
	return 1.0 + 2.0 * gamma / (gamma + 1.0) * (m1 ** 2 * np.sin(beta) ** 2 - 1.0)

def oblique_rho2rho1(gamma: float, m1: float, beta: float) -> float:
	"""
    Returns the ratio of the downstream to the upstream density over an oblique shock given...
    
    Parameters
    ----------
    gamma : float
        Heat capacity ratio.
    m1: float
        The upstream mach number.
    beta: float
        The angle of the shock relative to the upstream flow axis.
        
    Returns
    ---------
    float: The ratio of the downstream density to the upstream density.
    """
	if gamma <= 1.0:
		raise ValueError("Heat capacity ratio gamma must be greater than 1.")
	return (gamma + 1.0) * (m1 ** 2 * np.sin(beta) ** 2) / ( 
		(gamma - 1.0) * m1 ** 2 * np.sin(beta) ** 2 + 2.0
     )

def oblique_T2T1(gamma: float, m1: float, beta: float) -> float:
    """
	Returns the ratio of the downstream to the upstream temperature over an oblique shock given...
    
    Parameters
    ----------
    gamma : float
        Heat capacity ratio.
    m1: float
        The upstream mach number.
    beta: float
        The angle of the shock relative to the upstream flow axis.
        
    Returns
    ---------
    float: The ratio of the downstream temperature to the upstream temperature.
    """
    return oblique_p2p1(gamma, m1, beta) / oblique_rho2rho1(gamma, m1, beta)


#/Oblique Shock Equations


@dataclass(frozen=True)
class CgnsScalarFieldData:
	"""Point coordinates and scalar solution fields loaded from one CGNS zone."""

	x: np.ndarray
	z: np.ndarray
	fields: dict[str, np.ndarray]


def _validate_point_array(
	values: np.ndarray,
	*,
	name: str,
	point_count: int | None = None,
) -> np.ndarray:
	"""Validate one finite, one-dimensional point array."""
	if values.ndim != 1:
		raise CgnsDataError(f"{name} must be one-dimensional")
	if point_count is not None and values.size != point_count:
		raise CgnsDataError(
			f"{name} has {values.size} values; expected {point_count} point values"
		)
	if not np.all(np.isfinite(values)):
		raise CgnsDataError(f"{name} must contain only finite values")
	return values


@dataclass(frozen=True)
class ShockEvaluation:
	"""Observed and oblique-shock-predicted state ratios for one segment."""

	gamma: float
	mach_upstream: float
	beta_rad: float
	velocity_change: np.ndarray
	pressure_ratio_simulated: float
	density_ratio_simulated: float
	temperature_ratio_simulated: float
	pressure_ratio_predicted: float
	density_ratio_predicted: float
	temperature_ratio_predicted: float
	angle_source: str = "velocity change"
	upstream_endpoint: str = "drag start"


def validate_gamma(gamma: float) -> float:
	"""Validate the heat-capacity ratio used by the oblique-shock equations."""
	if not np.isfinite(gamma) or gamma <= 1.0:
		raise ValueError("gamma must be finite and greater than 1")
	return float(gamma)


def validate_shock_angle_mode(mode: str) -> str:
	"""Validate the method used to determine the oblique-shock angle."""
	if mode not in SHOCK_ANGLE_MODES:
		raise ValueError(
			f"shock_angle must be one of {', '.join(SHOCK_ANGLE_MODES)}"
		)
	return mode


def select_upstream_endpoint(
	start_mach: float,
	end_mach: float,
) -> tuple[bool, str]:
	"""Return whether the drag start is upstream, based on the larger endpoint Mach."""
	if not np.isfinite((start_mach, end_mach)).all() or min(start_mach, end_mach) <= 0.0:
		raise ValueError("Endpoint Mach numbers must be finite and positive")
	if np.isclose(start_mach, end_mach):
		raise ValueError("Cannot infer upstream endpoint from equal Mach numbers")
	if start_mach > end_mach:
		return True, "drag start"
	return False, "drag end"


def perpendicular_segment_shock_direction(
	segment_start: tuple[float, float],
	segment_end: tuple[float, float],
	upstream_velocity: np.ndarray,
) -> tuple[float, np.ndarray]:
	"""Return beta and a physical `(axial, radial)` normal to the selected segment."""
	start = np.asarray(segment_start, dtype=np.float64)
	end = np.asarray(segment_end, dtype=np.float64)
	upstream_velocity = np.asarray(upstream_velocity, dtype=np.float64)
	if start.shape != (2,) or end.shape != (2,):
		raise ValueError("Segment endpoints must each contain z and x coordinates")
	if upstream_velocity.shape != (2,) or not np.isfinite(upstream_velocity).all():
		raise ValueError("Upstream velocity must contain two finite components")
	segment_direction = end - start
	if not np.isfinite(segment_direction).all() or np.allclose(segment_direction, 0.0):
		raise ValueError("Segment endpoints must be distinct and finite")
	upstream_speed = float(np.linalg.norm(upstream_velocity))
	if upstream_speed == 0.0:
		raise ValueError("Upstream velocity must be nonzero")

	# Heatmap coordinates are `(z, x)`; physical velocity order is `(axial=x, radial=z)`.
	physical_segment_direction = np.array((segment_direction[1], segment_direction[0]))
	shock_direction = np.array(
		(-physical_segment_direction[1], physical_segment_direction[0])
	)
	shock_direction /= np.linalg.norm(shock_direction)
	if np.dot(shock_direction, upstream_velocity) < 0.0:
		shock_direction *= -1.0
	beta_rad = float(
		np.arccos(
			np.clip(
				float(np.dot(shock_direction, upstream_velocity)) / upstream_speed,
				-1.0,
				1.0,
			)
		)
	)
	return beta_rad, shock_direction


def third_point_shock_direction(
	segment_start: tuple[float, float],
	segment_end: tuple[float, float],
	direction_point: tuple[float, float],
	upstream_velocity: np.ndarray,
) -> tuple[float, np.ndarray]:
	"""Return beta and a physical direction from the segment midpoint to a third point."""
	start = np.asarray(segment_start, dtype=np.float64)
	end = np.asarray(segment_end, dtype=np.float64)
	direction_point = np.asarray(direction_point, dtype=np.float64)
	upstream_velocity = np.asarray(upstream_velocity, dtype=np.float64)
	if any(point.shape != (2,) for point in (start, end, direction_point)):
		raise ValueError("Segment endpoints and direction point must each contain z and x")
	if not np.isfinite(np.concatenate((start, end, direction_point))).all():
		raise ValueError("Segment endpoints and direction point must be finite")
	if upstream_velocity.shape != (2,) or not np.isfinite(upstream_velocity).all():
		raise ValueError("Upstream velocity must contain two finite components")
	midpoint = 0.5 * (start + end)
	plot_direction = direction_point - midpoint
	if np.allclose(plot_direction, 0.0):
		raise ValueError("Third point must differ from the segment midpoint")
	upstream_speed = float(np.linalg.norm(upstream_velocity))
	if upstream_speed == 0.0:
		raise ValueError("Upstream velocity must be nonzero")

	# Heatmap coordinates are `(z, x)`; physical velocity order is `(axial=x, radial=z)`.
	shock_direction = np.array((plot_direction[1], plot_direction[0]))
	shock_direction /= np.linalg.norm(shock_direction)
	beta_rad = float(
		np.arccos(
			np.clip(
				float(np.dot(shock_direction, upstream_velocity)) / upstream_speed,
				-1.0,
				1.0,
			)
		)
	)
	return beta_rad, shock_direction


def evaluate_oblique_shock(
	*,
	gamma: float,
	mach_upstream: float,
	upstream_velocity: np.ndarray,
	downstream_velocity: np.ndarray,
	upstream_pressure: float,
	downstream_pressure: float,
	upstream_density: float,
	downstream_density: float,
	upstream_temperature: float,
	downstream_temperature: float,
	beta_rad: float | None = None,
	angle_source: str = "velocity change",
	upstream_endpoint: str = "drag start",
) -> ShockEvaluation:
	"""Compare endpoint ratios with oblique-shock predictions.

	Velocity components are ordered `(axial, radial)`, matching the CGNS fields.
	"""
	gamma = validate_gamma(gamma)
	if not np.isfinite(mach_upstream) or mach_upstream <= 0.0:
		raise ValueError("Upstream Mach number must be finite and positive")
	upstream_velocity = np.asarray(upstream_velocity, dtype=np.float64)
	downstream_velocity = np.asarray(downstream_velocity, dtype=np.float64)
	if upstream_velocity.shape != (2,) or downstream_velocity.shape != (2,):
		raise ValueError("Upstream and downstream velocities must have two components")
	if not np.isfinite(np.concatenate((upstream_velocity, downstream_velocity))).all():
		raise ValueError("Upstream and downstream velocities must be finite")
	velocity_change = upstream_velocity - downstream_velocity
	upstream_speed = float(np.linalg.norm(upstream_velocity))
	change_speed = float(np.linalg.norm(velocity_change))
	if upstream_speed == 0.0:
		raise ValueError("Upstream velocity must be nonzero")
	if beta_rad is None and change_speed == 0.0:
		raise ValueError("Upstream and downstream velocities must differ")
	if beta_rad is None:
		beta_rad = float(
			np.arccos(
				np.clip(
					float(np.dot(velocity_change, upstream_velocity))
					/ (upstream_speed * change_speed),
					-1.0,
					1.0,
				)
			)
		)
	elif not np.isfinite(beta_rad) or not 0.0 <= beta_rad <= np.pi:
		raise ValueError("Shock angle beta must be finite and between 0 and pi radians")
	state_values = np.asarray(
		(
			upstream_pressure,
			downstream_pressure,
			upstream_density,
			downstream_density,
			upstream_temperature,
			downstream_temperature,
		),
		dtype=np.float64,
	)
	if not np.all(np.isfinite(state_values)):
		raise ValueError("Shock-state values must be finite")
	if np.any(state_values[[0, 2, 4]] == 0.0):
		raise ValueError("Upstream pressure, density, and temperature must be nonzero")
	return ShockEvaluation(
		gamma=gamma,
		mach_upstream=float(mach_upstream),
		beta_rad=beta_rad,
		velocity_change=velocity_change,
		pressure_ratio_simulated=float(downstream_pressure / upstream_pressure),
		density_ratio_simulated=float(downstream_density / upstream_density),
		temperature_ratio_simulated=float(downstream_temperature / upstream_temperature),
		pressure_ratio_predicted=oblique_p2p1(gamma, mach_upstream, beta_rad),
		density_ratio_predicted=oblique_rho2rho1(gamma, mach_upstream, beta_rad),
		temperature_ratio_predicted=oblique_T2T1(gamma, mach_upstream, beta_rad),
		angle_source=angle_source,
		upstream_endpoint=upstream_endpoint,
	)


def _validate_bounds(
	bounds: tuple[float, float] | None,
	*,
	axis_name: str,
) -> tuple[float, float] | None:
	"""Validate optional increasing source-coordinate bounds in metres."""
	if bounds is None:
		return None
	lower, upper = bounds
	if not np.isfinite((lower, upper)).all() or lower >= upper:
		raise ValueError(f"{axis_name} bounds must be finite and increasing")
	return float(lower), float(upper)


def load_cgns_scalar_fields(
	cgns_path: str | Path,
	*,
	zone_path: str = DEFAULT_ZONE_PATH,
	flow_solution: str = DEFAULT_FLOW_SOLUTION,
	x_coordinate: str = DEFAULT_X_COORDINATE,
	z_coordinate: str = DEFAULT_Z_COORDINATE,
	x_bounds: tuple[float, float] | None = None,
	z_bounds: tuple[float, float] | None = None,
) -> CgnsScalarFieldData:
	"""Load valid point-aligned scalar fields, optionally clipped in source metres."""
	cgns_path = Path(cgns_path).resolve()
	if not cgns_path.is_file():
		raise FileNotFoundError(cgns_path)
	x_bounds = _validate_bounds(x_bounds, axis_name="x")
	z_bounds = _validate_bounds(z_bounds, axis_name="z")

	try:
		with h5py.File(cgns_path, "r") as cgns_file:
			zone = cgns_file[zone_path]
			coordinates = zone["GridCoordinates"]
			x = _validate_point_array(
				_read_cgns_data(
					coordinates[x_coordinate],
					f"{cgns_path} coordinate {x_coordinate!r}",
				),
				name=f"{cgns_path} coordinate {x_coordinate!r}",
			)
			z = _validate_point_array(
				_read_cgns_data(
					coordinates[z_coordinate],
					f"{cgns_path} coordinate {z_coordinate!r}",
				),
				name=f"{cgns_path} coordinate {z_coordinate!r}",
				point_count=x.size,
			)
			if x.size < 3:
				raise CgnsDataError("CGNS field must contain at least three points")

			solution = zone[flow_solution]
			fields: dict[str, np.ndarray] = {}
			for field_name, node in solution.items():
				if field_name in COORDINATE_FIELD_NAMES or not isinstance(node, h5py.Group):
					continue
				try:
					values = _read_cgns_data(
						node,
						f"{cgns_path} field {flow_solution}/{field_name}",
					)
					fields[field_name] = _validate_point_array(
						values,
						name=f"{cgns_path} field {flow_solution}/{field_name}",
						point_count=x.size,
					)
				except CgnsDataError:
					continue
	except KeyError as exc:
		raise CgnsDataError(
			f"{cgns_path} is missing required CGNS node {exc.args[0]!r}"
		) from exc
	except OSError as exc:
		raise CgnsDataError(f"Could not read CGNS file {cgns_path}: {exc}") from exc

	if not fields:
		raise CgnsDataError(f"{cgns_path} has no valid point-aligned scalar fields")
	mask = np.ones(x.size, dtype=bool)
	if x_bounds is not None:
		mask &= (x >= x_bounds[0]) & (x <= x_bounds[1])
	if z_bounds is not None:
		mask &= (z >= z_bounds[0]) & (z <= z_bounds[1])
	if np.count_nonzero(mask) < 3:
		raise CgnsDataError("Requested x/z bounds contain fewer than three CGNS points")
	if not np.all(mask):
		x = x[mask]
		z = z[mask]
		fields = {field_name: values[mask] for field_name, values in fields.items()}
	return CgnsScalarFieldData(x=x, z=z, fields=fields)


def density_gradient_magnitude(
	fields: Mapping[str, np.ndarray],
	*,
	x_gradient_name: str = DEFAULT_DENSITY_X_GRADIENT,
	y_gradient_name: str = DEFAULT_DENSITY_Y_GRADIENT,
) -> np.ndarray:
	"""Return the density-gradient magnitude from the two required components."""
	try:
		x_gradient = fields[x_gradient_name]
		y_gradient = fields[y_gradient_name]
	except KeyError as exc:
		raise CgnsDataError(
			"Required density-gradient field is missing: "
			f"{exc.args[0]!r}; expected {x_gradient_name!r} and {y_gradient_name!r}"
		) from exc
	if x_gradient.shape != y_gradient.shape:
		raise CgnsDataError("Density-gradient component arrays must have matching shapes")
	return np.hypot(x_gradient, y_gradient)


def initial_log_limits(values: np.ndarray) -> tuple[float, float]:
	"""Choose positive, finite logarithmic limits robust to isolated outliers."""
	positive_values = values[np.isfinite(values) & (values > 0.0)]
	if positive_values.size == 0:
		raise CgnsDataError("Density-gradient magnitude has no positive finite values")
	lower, upper = np.quantile(positive_values, (0.01, 0.99))
	if lower == upper:
		lower = float(positive_values.min())
		upper = float(positive_values.max())
	if lower == upper:
		lower *= 0.9
		upper *= 1.1
	return float(lower), float(upper)


def parse_log_limits(lower_text: str, upper_text: str) -> tuple[float, float]:
	"""Parse and validate explicitly entered positive heatmap limits."""
	try:
		lower = float(lower_text)
		upper = float(upper_text)
	except ValueError as exc:
		raise ValueError("Limits must be numeric values") from exc
	if not np.isfinite((lower, upper)).all() or lower <= 0.0 or lower >= upper:
		raise ValueError("Limits must be finite, positive, and increasing")
	return lower, upper


def sample_segment(
	start: tuple[float, float],
	end: tuple[float, float],
	*,
	count: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return `(z, x)` source-coordinate samples and start-relative distance in mm."""
	if count < 2:
		raise ValueError("line sample count must be at least 2")
	start_point = np.asarray(start, dtype=np.float64)
	end_point = np.asarray(end, dtype=np.float64)
	if start_point.shape != (2,) or end_point.shape != (2,):
		raise ValueError("Line endpoints must each contain z and x coordinates")
	if not np.isfinite(np.concatenate((start_point, end_point))).all():
		raise ValueError("Line endpoints must be finite")
	if np.array_equal(start_point, end_point):
		raise ValueError("Line endpoints must be distinct")
	fractions = np.linspace(0.0, 1.0, count)
	points = start_point + fractions[:, np.newaxis] * (end_point - start_point)
	distance_mm = fractions * np.linalg.norm(end_point - start_point) * 1000.0
	return points, distance_mm


def normalize_lineout_values(values: np.ndarray) -> np.ndarray:
	"""Scale one lineout to unit magnitude without shifting its zero baseline."""
	values = np.asarray(values, dtype=np.float64)
	finite_values = values[np.isfinite(values)]
	if finite_values.size == 0:
		return values.copy()
	magnitude = float(np.max(np.abs(finite_values)))
	if magnitude == 0.0:
		return values.copy()
	return values / magnitude


class InteractiveShockEvaluator:
	"""Coordinate the heatmap selector and live scalar-field lineout window."""

	def __init__(
		self,
		data: CgnsScalarFieldData,
		*,
		line_samples: int = 400,
		gamma: float = 1.4,
		shock_angle: str = "auto",
		show_color_limit_boxes: bool = False,
		title: str | None = None,
	) -> None:
		if line_samples < 2:
			raise ValueError("line_samples must be at least 2")
		self.data = data
		self.line_samples = line_samples
		self.gamma = validate_gamma(gamma)
		self.shock_angle = validate_shock_angle_mode(shock_angle)
		self.show_color_limit_boxes = show_color_limit_boxes
		self.gradient_magnitude = density_gradient_magnitude(data.fields)
		self.limits = initial_log_limits(self.gradient_magnitude)
		self.title = title or "Density-gradient magnitude"
		self.triangulation = Triangulation(data.z, data.x)
		self._delaunay = Delaunay(np.column_stack((data.z, data.x)))
		self._samplers = {
			name: LinearNDInterpolator(self._delaunay, values, fill_value=np.nan)
			for name, values in data.fields.items()
		}
		self.segment: tuple[tuple[float, float], tuple[float, float]] | None = None
		self._drag_start: tuple[float, float] | None = None
		self._third_point: tuple[float, float] | None = None
		self.shock_evaluation: ShockEvaluation | None = None

		self.figure, self.heatmap_axes, self.lineout_axes = self._create_figure()
		# Compatibility aliases keep update callbacks focused on their owning axes.
		self.heatmap_figure = self.figure
		self.lineout_figure = self.figure

	def _create_figure(self) -> tuple[Figure, Axes, Axes]:
		"""Create the unified heatmap, lineout, control, and report workspace."""
		figure = plt.figure(figsize=(17, 10))
		heatmap_axes = figure.add_axes((0.06, 0.48, 0.43, 0.43))
		lineout_axes = figure.add_axes((0.56, 0.48, 0.38, 0.43))
		checkbox_axes = figure.add_axes((0.76, 0.08, 0.18, 0.30))
		norm = colors.LogNorm(*self.limits)
		self.heatmap = heatmap_axes.tripcolor(
			self.triangulation,
			self.gradient_magnitude,
			shading="gouraud",
			cmap="magma",
			norm=norm,
		)
		self.colorbar = figure.colorbar(
			self.heatmap,
			ax=heatmap_axes,
			label=r"$|\nabla \rho|$ [source units]",
		)
		heatmap_axes.set_xlabel("z [mm]")
		heatmap_axes.set_ylabel("x [mm]")
		heatmap_axes.set_title(self.title)
		heatmap_axes.set_aspect("equal")
		heatmap_axes.xaxis.set_major_formatter(lambda value, _: f"{value * 1000:g}")
		heatmap_axes.yaxis.set_major_formatter(lambda value, _: f"{value * 1000:g}")

		self.lower_limit_box: TextBox | None = None
		self.upper_limit_box: TextBox | None = None
		if self.show_color_limit_boxes:
			lower_axes = figure.add_axes((0.07, 0.39, 0.18, 0.04))
			upper_axes = figure.add_axes((0.31, 0.39, 0.18, 0.04))
			self.lower_limit_box = TextBox(
				lower_axes,
				"lower limit",
				initial=f"{self.limits[0]:.6g}",
			)
			self.upper_limit_box = TextBox(
				upper_axes,
				"upper limit",
				initial=f"{self.limits[1]:.6g}",
			)
			self.lower_limit_box.on_submit(self._on_limits_submitted)
			self.upper_limit_box.on_submit(self._on_limits_submitted)
		self.status_text = figure.text(0.06, 0.44, "Drag across the map to sample a lineout.")
		self.segment_artist = Line2D([], [], color="cyan", linewidth=1.5)
		self.shock_direction_artist = Line2D(
			[],
			[],
			color="lime",
			linestyle="--",
			linewidth=1.5,
		)
		self.shock_direction_point_artist = Line2D(
			[],
			[],
			color="lime",
			marker="o",
			linestyle="None",
		)
		heatmap_axes.add_line(self.segment_artist)
		heatmap_axes.add_line(self.shock_direction_artist)
		heatmap_axes.add_line(self.shock_direction_point_artist)
		figure.canvas.mpl_connect("button_press_event", self._on_press)
		figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
		figure.canvas.mpl_connect("button_release_event", self._on_release)

		field_names = list(self.data.fields)
		selected = [name == "Density" for name in field_names]
		self.field_checkboxes = CheckButtons(checkbox_axes, field_names, selected)
		self.field_checkboxes.on_clicked(self._on_field_selection_changed)
		checkbox_axes.set_title("Lineout fields", fontsize=10)
		for label in self.field_checkboxes.labels:
			label.set_fontsize(8)
		self.lineout_axes = lineout_axes
		self._style_lineout_axes()
		self.lineout_empty_text = lineout_axes.text(
			0.5,
			0.5,
			"Drag a segment on the heatmap to display checked fields.",
			ha="center",
			va="center",
			transform=lineout_axes.transAxes,
		)
		self.shock_report_text = figure.text(
			0.06,
			0.06,
			"Shock ratios will appear after selecting a segment.",
			family="monospace",
			fontsize=11,
			va="bottom",
		)
		return figure, heatmap_axes, lineout_axes

	def _style_lineout_axes(self) -> None:
		self.lineout_axes.set_xlabel("distance from segment start [mm]")
		self.lineout_axes.set_ylabel("field value [source units]")
		self.lineout_axes.grid()

	def _on_limits_submitted(self, _: str) -> None:
		assert self.lower_limit_box is not None
		assert self.upper_limit_box is not None
		try:
			self.limits = parse_log_limits(
				self.lower_limit_box.text,
				self.upper_limit_box.text,
			)
		except ValueError as exc:
			self.status_text.set_text(str(exc))
			self.heatmap_figure.canvas.draw_idle()
			return
		self.heatmap.set_norm(colors.LogNorm(*self.limits))
		self.colorbar.update_normal(self.heatmap)
		self.status_text.set_text("Applied logarithmic limits.")
		self.heatmap_figure.canvas.draw_idle()

	def _event_point(self, event: MouseEvent) -> tuple[float, float] | None:
		if event.inaxes is not self.heatmap_axes or event.xdata is None or event.ydata is None:
			return None
		return float(event.xdata), float(event.ydata)

	def _on_press(self, event: MouseEvent) -> None:
		if event.button != 1:
			return
		if (
			self.shock_angle == "manual"
			and self.segment is not None
			and self._third_point is None
		):
			third_point = self._event_point(event)
			if third_point is None:
				return
			self._third_point = third_point
			self._update_shock_evaluation()
			self._update_lineout()
			self.heatmap_figure.canvas.draw_idle()
			return
		self._drag_start = self._event_point(event)

	def _on_motion(self, event: MouseEvent) -> None:
		if self._drag_start is None:
			return
		end = self._event_point(event)
		if end is None:
			return
		self.segment_artist.set_data((self._drag_start[0], end[0]), (self._drag_start[1], end[1]))
		self.heatmap_figure.canvas.draw_idle()

	def _on_release(self, event: MouseEvent) -> None:
		if event.button != 1 or self._drag_start is None:
			return
		end = self._event_point(event)
		start = self._drag_start
		self._drag_start = None
		if end is None or np.array_equal(start, end):
			self.status_text.set_text("Select two distinct points inside the heatmap.")
			self.heatmap_figure.canvas.draw_idle()
			return
		self.segment = (start, end)
		self._third_point = None
		self.segment_artist.set_data((start[0], end[0]), (start[1], end[1]))
		if self.shock_angle == "manual":
			self.shock_evaluation = None
			self.shock_direction_artist.set_data([], [])
			self.shock_direction_point_artist.set_data([], [])
			self.shock_report_text.set_text(
				"Click a third point on the heatmap to select the shock direction."
			)
			self.status_text.set_text("Segment selected; click a third point for shock direction.")
		else:
			self._update_shock_evaluation()
		self._update_lineout()
		self.heatmap_figure.canvas.draw_idle()

	def _on_field_selection_changed(self, _: str) -> None:
		if self.segment is not None:
			self._update_lineout()

	def _selected_field_names(self) -> list[str]:
		return [
			name
			for name, selected in zip(
				self.data.fields,
				self.field_checkboxes.get_status(),
				strict=True,
			)
			if selected
		]

	def _sample_field_at_point(self, field_name: str, point: tuple[float, float]) -> float:
		"""Evaluate one scalar CGNS field at one `(z, x)` source-coordinate point."""
		value = float(np.asarray(self._samplers[field_name](np.asarray([point]))).item())
		if not np.isfinite(value):
			raise ValueError(f"{field_name} is outside the interpolation domain")
		return value

	def _update_shock_evaluation(self) -> None:
		"""Calculate and display endpoint shock ratios for the selected segment."""
		assert self.segment is not None
		missing_fields = [
			field_name
			for field_name in SHOCK_STATE_FIELD_NAMES
			if field_name not in self._samplers
		]
		if missing_fields:
			self.shock_evaluation = None
			self.shock_direction_artist.set_data([], [])
			self.shock_direction_point_artist.set_data([], [])
			self.shock_report_text.set_text(
				"Shock evaluation unavailable: missing " + ", ".join(missing_fields)
			)
			self.status_text.set_text("Lineout updated; shock-state fields are unavailable.")
			return

		start, end = self.segment
		try:
			start_values = {
				field_name: self._sample_field_at_point(field_name, start)
				for field_name in SHOCK_STATE_FIELD_NAMES
			}
			end_values = {
				field_name: self._sample_field_at_point(field_name, end)
				for field_name in SHOCK_STATE_FIELD_NAMES
			}
			if self.shock_angle == "perp":
				start_is_upstream, upstream_endpoint = select_upstream_endpoint(
					start_values["Mach"],
					end_values["Mach"],
				)
				if start_is_upstream:
					upstream_values, downstream_values = start_values, end_values
				else:
					upstream_values, downstream_values = end_values, start_values
				upstream_velocity = np.array(
					(upstream_values["Axial_Velocity"], upstream_values["Radial_Velocity"])
				)
				beta_rad, shock_direction = perpendicular_segment_shock_direction(
					start,
					end,
					upstream_velocity,
				)
				angle_source = "segment normal"
			elif self.shock_angle == "manual":
				if self._third_point is None:
					raise ValueError("Select a third point for the shock direction")
				upstream_values, downstream_values = start_values, end_values
				upstream_endpoint = "drag start"
				upstream_velocity = np.array(
					(upstream_values["Axial_Velocity"], upstream_values["Radial_Velocity"])
				)
				beta_rad, shock_direction = third_point_shock_direction(
					start,
					end,
					self._third_point,
					upstream_velocity,
				)
				angle_source = "third point"
			else:
				upstream_values, downstream_values = start_values, end_values
				upstream_endpoint = "drag start"
				upstream_velocity = np.array(
					(upstream_values["Axial_Velocity"], upstream_values["Radial_Velocity"])
				)
				beta_rad = None
				shock_direction = upstream_velocity - np.array(
					(downstream_values["Axial_Velocity"], downstream_values["Radial_Velocity"])
				)
				angle_source = "velocity change"
			self.shock_evaluation = evaluate_oblique_shock(
				gamma=self.gamma,
				mach_upstream=upstream_values["Mach"],
				upstream_velocity=upstream_velocity,
				downstream_velocity=np.array(
					(downstream_values["Axial_Velocity"], downstream_values["Radial_Velocity"])
				),
				upstream_pressure=upstream_values["Pressure"],
				downstream_pressure=downstream_values["Pressure"],
				upstream_density=upstream_values["Density"],
				downstream_density=downstream_values["Density"],
				upstream_temperature=upstream_values["Temperature"],
				downstream_temperature=downstream_values["Temperature"],
				beta_rad=beta_rad,
				angle_source=angle_source,
				upstream_endpoint=upstream_endpoint,
			)
			self.shock_evaluation = replace(
				self.shock_evaluation,
				pressure_ratio_simulated=end_values["Pressure"] / start_values["Pressure"],
				density_ratio_simulated=end_values["Density"] / start_values["Density"],
				temperature_ratio_simulated=(
					end_values["Temperature"] / start_values["Temperature"]
				),
			)
		except ValueError as exc:
			self.shock_evaluation = None
			self.shock_direction_artist.set_data([], [])
			self.shock_direction_point_artist.set_data([], [])
			self.shock_report_text.set_text(f"Shock evaluation unavailable: {exc}")
			self.status_text.set_text("Lineout updated; select endpoints inside the field.")
			return

		self._draw_shock_direction(
			start,
			end,
			shock_direction,
			direction_point=self._third_point if self.shock_angle == "manual" else None,
		)
		self.shock_report_text.set_text(self._format_shock_report(self.shock_evaluation))
		self.status_text.set_text(
			f"Lineout updated; beta = {np.rad2deg(self.shock_evaluation.beta_rad):.2f} degrees."
		)

	def _draw_shock_direction(
		self,
		start: tuple[float, float],
		end: tuple[float, float],
		physical_direction: np.ndarray,
		*,
		direction_point: tuple[float, float] | None = None,
	) -> None:
		"""Draw the supplied physical `(axial, radial)` direction from the midpoint."""
		midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
		segment_length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
		plot_direction = np.array((physical_direction[1], physical_direction[0]))
		plot_direction /= np.linalg.norm(plot_direction)
		line_end = (
			np.asarray(direction_point)
			if direction_point is not None
			else midpoint + segment_length * plot_direction
		)
		self.shock_direction_artist.set_data(
			(midpoint[0], line_end[0]),
			(midpoint[1], line_end[1]),
		)
		if direction_point is None:
			self.shock_direction_point_artist.set_data([], [])
		else:
			self.shock_direction_point_artist.set_data((line_end[0],), (line_end[1],))

	@staticmethod
	def _format_shock_report(evaluation: ShockEvaluation) -> str:
		"""Format observed and predicted shock ratios for the live result panel."""
		label_width = 19
		value_width = 18
		def row(label: str, observed: float, predicted: float) -> str:
			if predicted == 0.0:
				percent_error = 0.0 if observed == 0.0 else float("nan")
			else:
				percent_error = abs(observed - predicted) / abs(predicted) * 100.0
			return (
				f"{label:<{label_width}}{observed:>{value_width}.2f}"
				f"{predicted:>{value_width}.2f}{percent_error:>{value_width}.2f}"
			)

		return "\n".join(
			(
				f"gamma = {evaluation.gamma:.3f}  M1 = {evaluation.mach_upstream:.2f}  "
				f"beta = {np.rad2deg(evaluation.beta_rad):.1f} deg",
				f"upstream = {evaluation.upstream_endpoint}  angle = {evaluation.angle_source}",
				f"{'ratio':<{label_width}}{'drag-end/start':>{value_width}}"
				f"{'predicted':>{value_width}}{'percent error [%]':>{value_width}}",
				row(
					"P(end)/P(start)",
					evaluation.pressure_ratio_simulated,
					evaluation.pressure_ratio_predicted,
				),
				row(
					"rho(end)/rho(start)",
					evaluation.density_ratio_simulated,
					evaluation.density_ratio_predicted,
				),
				row(
					"T(end)/T(start)",
					evaluation.temperature_ratio_simulated,
					evaluation.temperature_ratio_predicted,
				),
			)
		)

	def _update_lineout(self) -> None:
		if self.segment is None:
			return
		points, distance_mm = sample_segment(*self.segment, count=self.line_samples)
		self.lineout_axes.clear()
		self._style_lineout_axes()
		selected_names = self._selected_field_names()
		if not selected_names:
			self.lineout_axes.text(
				0.5,
				0.5,
				"Select one or more fields to plot.",
				ha="center",
				va="center",
				transform=self.lineout_axes.transAxes,
			)
		else:
			lineout_values = {
				field_name: np.asarray(self._samplers[field_name](points))
				for field_name in selected_names
			}
			normalize = len(selected_names) > 1
			if normalize:
				self.lineout_axes.set_ylabel("independently magnitude-normalized field value")
				self.lineout_axes.set_title("Selected fields independently scaled to unit magnitude")
			for field_name in selected_names:
				values = lineout_values[field_name]
				if normalize:
					values = normalize_lineout_values(values)
				self.lineout_axes.plot(distance_mm, values, label=field_name)
			self.lineout_axes.legend()
		self.lineout_figure.canvas.draw_idle()

	def show(self) -> None:
		"""Run the GUI event loop while retaining this controller's callbacks."""
		plt.show()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Interactively evaluate density-gradient shocks in one CGNS field."
	)
	parser.add_argument("cgns_path", type=Path, help="Input unstructured CGNS field")
	parser.add_argument(
		"--line-samples",
		type=int,
		default=400,
		help="Number of samples per selected segment (default: 400)",
	)
	parser.add_argument(
		"--gamma",
		type=float,
		default=1.4,
		help="Heat-capacity ratio for oblique-shock predictions (default: 1.4)",
	)
	parser.add_argument(
		"--shock-angle",
		choices=SHOCK_ANGLE_MODES,
		default="auto",
		help=(
			"Shock-angle method: auto uses endpoint velocities, perp uses the "
			"segment normal, manual uses a third click from the segment midpoint "
			"(default: auto)"
		),
	)
	parser.add_argument(
		"--show-color-limit-boxes",
		action="store_true",
		help="Show live lower and upper heatmap color-limit text boxes",
	)
	parser.add_argument(
		"--flow-solution",
		default=DEFAULT_FLOW_SOLUTION,
		help=f"CGNS flow-solution group (default: {DEFAULT_FLOW_SOLUTION})",
	)
	parser.add_argument(
		"--x-coordinate",
		default=DEFAULT_X_COORDINATE,
		help=f"CGNS transverse coordinate name (default: {DEFAULT_X_COORDINATE})",
	)
	parser.add_argument(
		"--z-coordinate",
		default=DEFAULT_Z_COORDINATE,
		help=f"CGNS axial coordinate name (default: {DEFAULT_Z_COORDINATE})",
	)
	parser.add_argument(
		"--x-bounds",
		nargs=2,
		type=float,
		metavar=("MIN_M", "MAX_M"),
		default=None,
		help="Optional transverse x bounds in source-coordinate metres",
	)
	parser.add_argument(
		"--z-bounds",
		nargs=2,
		type=float,
		metavar=("MIN_M", "MAX_M"),
		default=None,
		help="Optional axial z bounds in source-coordinate metres",
	)
	args = parser.parse_args()

	try:
		data = load_cgns_scalar_fields(
			args.cgns_path,
			flow_solution=args.flow_solution,
			x_coordinate=args.x_coordinate,
			z_coordinate=args.z_coordinate,
			x_bounds=tuple(args.x_bounds) if args.x_bounds is not None else None,
			z_bounds=tuple(args.z_bounds) if args.z_bounds is not None else None,
		)
		evaluator = InteractiveShockEvaluator(
			data,
			line_samples=args.line_samples,
			gamma=args.gamma,
			shock_angle=args.shock_angle,
			show_color_limit_boxes=args.show_color_limit_boxes,
			title=args.cgns_path.name,
		)
	except (CgnsDataError, ValueError) as exc:
		parser.error(str(exc))
	evaluator.show()


if __name__ == "__main__":
	main()
