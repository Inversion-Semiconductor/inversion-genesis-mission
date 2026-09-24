#!/usr/bin/env python3
"""Measure density convergence across ANSYS lineout or CGNS grid resolutions.

By default, each resolution is compared with the finest available resolution.
Adjacent-grid comparisons remain available as an alternative.
The error is the integrated absolute density difference, normalized by the
reference density magnitude. CGNS comparisons also report the mean and standard
deviation of absolute density error, normalized by peak reference density.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from ansys_lineouts import parse_lineout_file
from cgns_to_density_hdf5 import load_cgns_density_points

GRID_SIZE_STEM = re.compile(r"\d+(?:_\d+)?$")


@dataclass(frozen=True)
class ResolutionLineouts:
	"""Lineouts from one simulation resolution."""

	grid_size_mm: float
	path: Path
	sections: dict[str, np.ndarray]


@dataclass(frozen=True)
class ResolutionField:
	"""Pointwise CGNS density field from one simulation resolution."""

	grid_size_mm: float
	path: Path
	x: np.ndarray
	z: np.ndarray
	density: np.ndarray


@dataclass(frozen=True)
class RelativeErrorDistribution:
	"""Summary statistics for pointwise relative errors on a comparison grid."""

	sample_count: int
	mean: float
	standard_deviation: float
	minimum: float
	maximum: float


@dataclass(frozen=True)
class PeakNormalizedAbsoluteErrorDistribution:
	"""Absolute density-error statistics normalized by peak reference density."""

	sample_count: int
	reference_peak_density: float
	mean_absolute_error: float
	standard_deviation_absolute_error: float
	minimum_absolute_error: float
	maximum_absolute_error: float

	@property
	def mean_relative_error(self) -> float:
		"""Return the mean absolute error relative to the reference peak."""
		return self.mean_absolute_error / self.reference_peak_density

	@property
	def standard_deviation_relative_error(self) -> float:
		"""Return the absolute-error standard deviation relative to the peak."""
		return self.standard_deviation_absolute_error / self.reference_peak_density

	@property
	def minimum_relative_error(self) -> float:
		"""Return the minimum absolute error relative to the reference peak."""
		return self.minimum_absolute_error / self.reference_peak_density

	@property
	def maximum_relative_error(self) -> float:
		"""Return the maximum absolute error relative to the reference peak."""
		return self.maximum_absolute_error / self.reference_peak_density


@dataclass(frozen=True)
class ConvergenceMetric:
	"""Relative integrated error for one lineout and one grid pair."""

	coarse_grid_size_mm: float
	finer_grid_size_mm: float
	lineout_label: str
	relative_error: float
	z_min_m: float
	z_max_m: float
	x_min_m: float | None = None
	x_max_m: float | None = None
	pointwise_error_distribution: RelativeErrorDistribution | None = None
	peak_normalized_absolute_error_distribution: (
		PeakNormalizedAbsoluteErrorDistribution | None
	) = None


@dataclass(frozen=True)
class ConvergenceSummary:
	"""Aggregate relative errors for one adjacent-resolution pair."""

	coarse_grid_size_mm: float
	finer_grid_size_mm: float
	mean_relative_error: float
	standard_deviation: float
	minimum_relative_error: float
	maximum_relative_error: float
	mean_peak_normalized_absolute_error: float | None = None
	standard_deviation_peak_normalized_absolute_error: float | None = None
	minimum_peak_normalized_absolute_error: float | None = None
	maximum_peak_normalized_absolute_error: float | None = None


def parse_grid_size_filename(path: Path) -> float:
	"""Return a grid size in mm from a filename such as ``0_075.txt``."""
	if GRID_SIZE_STEM.fullmatch(path.stem) is None:
		raise ValueError(
			f"Could not parse a maximum grid size from {path.name!r}; "
			"expected a name such as '0_075.txt'"
		)
	return float(path.stem.replace("_", "."))


def deduplicate_and_sort_profile(profile: np.ndarray) -> np.ndarray:
	"""Sort a profile by z and keep the last density for repeated positions."""
	if profile.ndim != 2 or profile.shape[1] != 2:
		raise ValueError("A density profile must be a two-column (z, density) array")
	if profile.shape[0] < 2:
		raise ValueError("A density profile must contain at least two samples")

	sorted_profile = profile[np.argsort(profile[:, 0])]
	z = sorted_profile[:, 0]
	_, reverse_indices = np.unique(z[::-1], return_index=True)
	indices = np.sort(sorted_profile.shape[0] - 1 - reverse_indices)
	unique_profile = sorted_profile[indices]
	if unique_profile.shape[0] < 2:
		raise ValueError("A density profile must contain two distinct z positions")
	return unique_profile


def load_resolution_lineouts(input_dir: Path) -> list[ResolutionLineouts]:
	"""Load and validate resolution-tagged ANSYS density lineout files."""
	input_dir = input_dir.resolve()
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	paths = sorted(
		path
		for path in input_dir.iterdir()
		if path.is_file()
		and not path.name.startswith(".")
		and GRID_SIZE_STEM.fullmatch(path.stem) is not None
	)
	if len(paths) < 2:
		raise ValueError(
			f"Expected at least two resolution-tagged lineout files in {input_dir}"
		)

	resolutions = [
		ResolutionLineouts(
			grid_size_mm=parse_grid_size_filename(path),
			path=path,
			sections={
				label: deduplicate_and_sort_profile(profile)
				for label, profile in parse_lineout_file(path).items()
			},
		)
		for path in paths
	]
	resolutions.sort(key=lambda resolution: resolution.grid_size_mm)

	grid_sizes = [resolution.grid_size_mm for resolution in resolutions]
	if len(set(grid_sizes)) != len(grid_sizes):
		raise ValueError(f"Duplicate maximum grid size file in {input_dir}")

	expected_labels = set(resolutions[0].sections)
	for resolution in resolutions[1:]:
		current_labels = set(resolution.sections)
		if current_labels != expected_labels:
			missing = sorted(expected_labels - current_labels)
			extra = sorted(current_labels - expected_labels)
			raise ValueError(
				f"Lineout labels in {resolution.path.name} do not match "
				f"{resolutions[0].path.name}; missing={missing}, extra={extra}"
			)

	return resolutions


def load_resolution_cgns_fields(input_dir: Path) -> list[ResolutionField]:
	"""Load full 2D density fields from resolution-tagged ``.cgns`` files."""
	input_dir = input_dir.resolve()
	if not input_dir.is_dir():
		raise NotADirectoryError(input_dir)

	paths = sorted(
		path
		for path in input_dir.iterdir()
		if path.is_file()
		and path.suffix.lower() == ".cgns"
		and GRID_SIZE_STEM.fullmatch(path.stem) is not None
	)
	if len(paths) < 2:
		raise ValueError(
			"Expected at least two resolution-tagged .cgns files in "
			f"{input_dir}; expected names such as '0_1.cgns'"
		)

	fields: list[ResolutionField] = []
	for path in paths:
		x, z, density = load_cgns_density_points(path)
		fields.append(
			ResolutionField(
				grid_size_mm=parse_grid_size_filename(path),
				path=path,
				x=x,
				z=z,
				density=density,
			)
		)
	fields.sort(key=lambda field: field.grid_size_mm)

	grid_sizes = [field.grid_size_mm for field in fields]
	if len(set(grid_sizes)) != len(grid_sizes):
		raise ValueError(f"Duplicate maximum grid size file in {input_dir}")
	return fields


def _clip_field(
	field: ResolutionField,
	*,
	x_min: float | None,
	x_max: float | None,
	z_min: float | None,
	z_max: float | None,
) -> ResolutionField:
	"""Return field samples inside the requested rectangular clipping bounds."""
	mask = np.ones(field.x.size, dtype=bool)
	if x_min is not None:
		mask &= field.x >= x_min
	if x_max is not None:
		mask &= field.x <= x_max
	if z_min is not None:
		mask &= field.z >= z_min
	if z_max is not None:
		mask &= field.z <= z_max
	if np.count_nonzero(mask) < 3:
		raise ValueError(
			f"Clipping bounds leave fewer than three CGNS samples in {field.path}"
		)
	return ResolutionField(
		grid_size_mm=field.grid_size_mm,
		path=field.path,
		x=field.x[mask],
		z=field.z[mask],
		density=field.density[mask],
	)


def _shared_field_bounds(
	fields: list[ResolutionField],
) -> tuple[float, float, float, float]:
	"""Return common `(x_min, x_max, z_min, z_max)` field bounds."""
	common_x_min = max(float(field.x.min()) for field in fields)
	common_x_max = min(float(field.x.max()) for field in fields)
	common_z_min = max(float(field.z.min()) for field in fields)
	common_z_max = min(float(field.z.max()) for field in fields)
	if common_x_max <= common_x_min or common_z_max <= common_z_min:
		raise ValueError("CGNS fields do not have a nonzero shared 2D domain")
	return common_x_min, common_x_max, common_z_min, common_z_max


def _regrid_cgns_fields(
	fields: list[ResolutionField],
	*,
	x_min: float | None,
	x_max: float | None,
	z_min: float | None,
	z_max: float | None,
	x_points: int,
	z_points: int,
	interpolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
	"""Clip and interpolate every CGNS field once on one shared regular grid."""
	if x_points < 2 or z_points < 2:
		raise ValueError("x_points and z_points must both be at least 2")
	if interpolation not in {"linear", "cubic"}:
		raise ValueError(f"Unknown CGNS interpolation method {interpolation!r}")
	clipped_fields = [
		_clip_field(
			field, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max
		)
		for field in fields
	]
	common_x_min, common_x_max, common_z_min, common_z_max = _shared_field_bounds(
		clipped_fields
	)
	x_grid = np.linspace(common_x_min, common_x_max, x_points)
	z_grid = np.linspace(common_z_min, common_z_max, z_points)
	z_mesh, x_mesh = np.meshgrid(z_grid, x_grid, indexing="ij")

	grids: dict[float, np.ndarray] = {}
	for field in clipped_fields:
		triangulation = Delaunay(np.column_stack((field.z, field.x)))
		interpolator_class = (
			LinearNDInterpolator
			if interpolation == "linear"
			else CloughTocher2DInterpolator
		)
		interpolator = interpolator_class(
			triangulation,
			field.density,
			fill_value=0.0,
		)
		grids[field.grid_size_mm] = np.asarray(interpolator(z_mesh, x_mesh))

	return z_grid, x_grid, grids


def _relative_integrated_gridded_field_difference(
	coarse_density: np.ndarray,
	finer_density: np.ndarray,
	*,
	x_grid: np.ndarray,
	z_grid: np.ndarray,
) -> float:
	"""Return area-integrated relative difference between regular density grids."""
	difference_integral = np.trapezoid(
		np.trapezoid(np.abs(coarse_density - finer_density), x_grid, axis=1),
		z_grid,
	)
	reference_integral = np.trapezoid(
		np.trapezoid(np.abs(finer_density), x_grid, axis=1),
		z_grid,
	)
	if reference_integral == 0:
		raise ValueError("Finer CGNS field has zero integrated magnitude")
	return float(difference_integral / reference_integral)


def _pointwise_relative_error_distribution(
	coarse_density: np.ndarray,
	finer_density: np.ndarray,
) -> RelativeErrorDistribution:
	"""Summarize local relative errors where the reference density is nonzero."""
	reference_magnitude = np.abs(finer_density)
	nonzero_reference = reference_magnitude > 0.0
	if not np.any(nonzero_reference):
		raise ValueError("Finer CGNS field has no nonzero density samples")

	pointwise_errors = (
		np.abs(coarse_density[nonzero_reference] - finer_density[nonzero_reference])
		/ reference_magnitude[nonzero_reference]
	)
	return RelativeErrorDistribution(
		sample_count=pointwise_errors.size,
		mean=float(np.mean(pointwise_errors)),
		standard_deviation=float(np.std(pointwise_errors)),
		minimum=float(np.min(pointwise_errors)),
		maximum=float(np.max(pointwise_errors)),
	)


def _peak_normalized_absolute_error_distribution(
	coarse_density: np.ndarray,
	finer_density: np.ndarray,
) -> PeakNormalizedAbsoluteErrorDistribution:
	"""Summarize absolute density error relative to peak reference density."""
	reference_peak_density = float(np.max(np.abs(finer_density)))
	if reference_peak_density == 0.0:
		raise ValueError("Finer CGNS field has zero peak density")

	absolute_errors = np.abs(coarse_density - finer_density)
	return PeakNormalizedAbsoluteErrorDistribution(
		sample_count=absolute_errors.size,
		reference_peak_density=reference_peak_density,
		mean_absolute_error=float(np.mean(absolute_errors)),
		standard_deviation_absolute_error=float(np.std(absolute_errors)),
		minimum_absolute_error=float(np.min(absolute_errors)),
		maximum_absolute_error=float(np.max(absolute_errors)),
	)


def relative_integrated_field_difference(
	coarse_field: ResolutionField,
	finer_field: ResolutionField,
	*,
	x_min: float | None = None,
	x_max: float | None = None,
	z_min: float | None = None,
	z_max: float | None = None,
	x_points: int = 200,
	z_points: int = 200,
	interpolation: str = "linear",
) -> tuple[float, float, float, float, float]:
	"""Return full-field relative error and the common `(z, x)` bounds.

	Both unstructured fields are linearly interpolated to a common regular grid.
	Samples outside a source field's convex hull are assigned zero density, matching
	the CGNS-to-HDF5 converter's default outside-domain policy.
	"""
	z_grid, x_grid, grids = _regrid_cgns_fields(
		[coarse_field, finer_field],
		x_min=x_min,
		x_max=x_max,
		z_min=z_min,
		z_max=z_max,
		x_points=x_points,
		z_points=z_points,
		interpolation=interpolation,
	)
	relative_error = _relative_integrated_gridded_field_difference(
		grids[coarse_field.grid_size_mm],
		grids[finer_field.grid_size_mm],
		x_grid=x_grid,
		z_grid=z_grid,
	)

	return (
		relative_error,
		float(z_grid[0]),
		float(z_grid[-1]),
		float(x_grid[0]),
		float(x_grid[-1]),
	)


def relative_integrated_difference(
	coarse_profile: np.ndarray,
	finer_profile: np.ndarray,
) -> tuple[float, float, float]:
	"""Return relative error and shared z limits for two density profiles."""
	coarse_profile = deduplicate_and_sort_profile(coarse_profile)
	finer_profile = deduplicate_and_sort_profile(finer_profile)
	coarse_z, coarse_density = coarse_profile.T
	finer_z, finer_density = finer_profile.T

	z_min = max(float(coarse_z[0]), float(finer_z[0]))
	z_max = min(float(coarse_z[-1]), float(finer_z[-1]))
	if z_max <= z_min:
		raise ValueError("Density profiles do not have a nonzero shared z range")

	z_values = np.unique(
		np.concatenate(
			(
				coarse_z[(coarse_z >= z_min) & (coarse_z <= z_max)],
				finer_z[(finer_z >= z_min) & (finer_z <= z_max)],
			)
		)
	)
	if z_values.size < 2:
		raise ValueError("Density profiles do not share enough z samples to integrate")

	coarse_values = np.interp(z_values, coarse_z, coarse_density)
	finer_values = np.interp(z_values, finer_z, finer_density)
	difference_integral = np.trapezoid(np.abs(coarse_values - finer_values), z_values)
	reference_integral = np.trapezoid(np.abs(finer_values), z_values)
	if reference_integral == 0:
		raise ValueError("Finer density profile has zero integrated magnitude")

	return float(difference_integral / reference_integral), z_min, z_max


def calculate_convergence(
	resolutions: list[ResolutionLineouts],
	*,
	reference_mode: str = "finest",
) -> list[ConvergenceMetric]:
	"""Calculate relative profile errors using the requested reference grid.

	``finest`` compares every coarser grid with the finest resolution. ``adjacent``
	compares each grid with the immediately finer available grid.
	"""
	if len(resolutions) < 2:
		raise ValueError("At least two resolutions are required for convergence")
	if reference_mode not in {"finest", "adjacent"}:
		raise ValueError(f"Unknown convergence reference mode {reference_mode!r}")

	metrics: list[ConvergenceMetric] = []
	descending = sorted(resolutions, key=lambda resolution: resolution.grid_size_mm, reverse=True)
	finest = descending[-1]
	grid_pairs = (
		((coarse, finest) for coarse in descending[:-1])
		if reference_mode == "finest"
		else zip(descending, descending[1:])
	)
	for coarse, finer in grid_pairs:
		for label in sorted(coarse.sections):
			relative_error, z_min, z_max = relative_integrated_difference(
				coarse.sections[label], finer.sections[label]
			)
			metrics.append(
				ConvergenceMetric(
					coarse_grid_size_mm=coarse.grid_size_mm,
					finer_grid_size_mm=finer.grid_size_mm,
					lineout_label=label,
					relative_error=relative_error,
					z_min_m=z_min,
					z_max_m=z_max,
				)
			)
	return metrics


def calculate_field_convergence(
	fields: list[ResolutionField],
	*,
	reference_mode: str = "finest",
	x_min: float | None = None,
	x_max: float | None = None,
	z_min: float | None = None,
	z_max: float | None = None,
	x_points: int = 200,
	z_points: int = 200,
	interpolation: str = "linear",
) -> list[ConvergenceMetric]:
	"""Calculate full-2D CGNS field convergence at each grid resolution."""
	if len(fields) < 2:
		raise ValueError("At least two CGNS fields are required for convergence")
	if reference_mode not in {"finest", "adjacent"}:
		raise ValueError(f"Unknown convergence reference mode {reference_mode!r}")

	z_grid, x_grid, grids = _regrid_cgns_fields(
		fields,
		x_min=x_min,
		x_max=x_max,
		z_min=z_min,
		z_max=z_max,
		x_points=x_points,
		z_points=z_points,
		interpolation=interpolation,
	)
	descending = sorted(fields, key=lambda field: field.grid_size_mm, reverse=True)
	finest = descending[-1]
	field_pairs = (
		((coarse, finest) for coarse in descending[:-1])
		if reference_mode == "finest"
		else zip(descending, descending[1:])
	)
	metrics: list[ConvergenceMetric] = []
	for coarse, finer in field_pairs:
		coarse_density = grids[coarse.grid_size_mm]
		finer_density = grids[finer.grid_size_mm]
		relative_error = _relative_integrated_gridded_field_difference(
			coarse_density,
			finer_density,
			x_grid=x_grid,
			z_grid=z_grid,
		)
		metrics.append(
			ConvergenceMetric(
				coarse_grid_size_mm=coarse.grid_size_mm,
				finer_grid_size_mm=finer.grid_size_mm,
				lineout_label="field",
				relative_error=relative_error,
				z_min_m=float(z_grid[0]),
				z_max_m=float(z_grid[-1]),
				x_min_m=float(x_grid[0]),
				x_max_m=float(x_grid[-1]),
				pointwise_error_distribution=_pointwise_relative_error_distribution(
					coarse_density,
					finer_density,
				),
				peak_normalized_absolute_error_distribution=(
					_peak_normalized_absolute_error_distribution(
						coarse_density,
						finer_density,
					)
				),
			)
		)
	return metrics


def summarize_convergence(metrics: list[ConvergenceMetric]) -> list[ConvergenceSummary]:
	"""Summarize per-lineout convergence metrics for each grid pair."""
	grouped: dict[tuple[float, float], list[float]] = {}
	for metric in metrics:
		grouped.setdefault(
			(metric.coarse_grid_size_mm, metric.finer_grid_size_mm), []
		).append(metric.relative_error)

	summaries: list[ConvergenceSummary] = []
	for (coarse_size, finer_size), errors in sorted(grouped.items()):
		group_metrics = [
			metric
			for metric in metrics
			if (metric.coarse_grid_size_mm, metric.finer_grid_size_mm)
			== (coarse_size, finer_size)
		]
		distributions = [
			metric.pointwise_error_distribution for metric in group_metrics
		]
		peak_distributions = [
			metric.peak_normalized_absolute_error_distribution
			for metric in group_metrics
		]
		if all(distribution is not None for distribution in distributions):
			field_distributions = [
				distribution for distribution in distributions if distribution is not None
			]
			sample_count = sum(
				distribution.sample_count for distribution in field_distributions
			)
			mean = sum(
				distribution.sample_count * distribution.mean
				for distribution in field_distributions
			) / sample_count
			second_moment = sum(
				distribution.sample_count
				* (distribution.standard_deviation**2 + distribution.mean**2)
				for distribution in field_distributions
			) / sample_count
			standard_deviation = float(np.sqrt(max(second_moment - mean**2, 0.0)))
			minimum = min(distribution.minimum for distribution in field_distributions)
			maximum = max(distribution.maximum for distribution in field_distributions)
		else:
			mean = float(np.mean(errors))
			standard_deviation = float(np.std(errors))
			minimum = float(np.min(errors))
			maximum = float(np.max(errors))

		if all(distribution is not None for distribution in peak_distributions):
			field_peak_distributions = [
				distribution
				for distribution in peak_distributions
				if distribution is not None
			]
			sample_count = sum(
				distribution.sample_count for distribution in field_peak_distributions
			)
			peak_normalized_mean = sum(
				distribution.sample_count * distribution.mean_relative_error
				for distribution in field_peak_distributions
			) / sample_count
			peak_normalized_second_moment = sum(
				distribution.sample_count
				* (
					distribution.standard_deviation_relative_error**2
					+ distribution.mean_relative_error**2
				)
				for distribution in field_peak_distributions
			) / sample_count
			peak_normalized_standard_deviation = float(
				np.sqrt(
					max(
						peak_normalized_second_moment - peak_normalized_mean**2,
						0.0,
					)
				)
			)
			peak_normalized_minimum = min(
				distribution.minimum_relative_error
				for distribution in field_peak_distributions
			)
			peak_normalized_maximum = max(
				distribution.maximum_relative_error
				for distribution in field_peak_distributions
			)
		else:
			peak_normalized_mean = None
			peak_normalized_standard_deviation = None
			peak_normalized_minimum = None
			peak_normalized_maximum = None

		summaries.append(
			ConvergenceSummary(
				coarse_grid_size_mm=coarse_size,
				finer_grid_size_mm=finer_size,
				mean_relative_error=float(mean),
				standard_deviation=standard_deviation,
				minimum_relative_error=float(minimum),
				maximum_relative_error=float(maximum),
				mean_peak_normalized_absolute_error=peak_normalized_mean,
				standard_deviation_peak_normalized_absolute_error=(
					peak_normalized_standard_deviation
				),
				minimum_peak_normalized_absolute_error=peak_normalized_minimum,
				maximum_peak_normalized_absolute_error=peak_normalized_maximum,
			)
		)
	return summaries


def write_convergence_csv(
	output_path: Path,
	metrics: list[ConvergenceMetric],
	summaries: list[ConvergenceSummary],
) -> Path:
	"""Write per-lineout metrics with their aggregate grid-pair statistics."""
	output_path = output_path.resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	summary_by_grid = {
		(summary.coarse_grid_size_mm, summary.finer_grid_size_mm): summary
		for summary in summaries
	}
	fieldnames = (
		"coarse_grid_size_mm",
		"finer_grid_size_mm",
		"lineout_label",
		"relative_integrated_absolute_difference",
		"z_min_m",
		"z_max_m",
		"x_min_m",
		"x_max_m",
		"mean_relative_error",
		"standard_deviation",
		"minimum_relative_error",
		"maximum_relative_error",
		"mean_peak_normalized_absolute_error",
		"standard_deviation_peak_normalized_absolute_error",
		"minimum_peak_normalized_absolute_error",
		"maximum_peak_normalized_absolute_error",
	)
	with output_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		for metric in metrics:
			summary = summary_by_grid[
				(metric.coarse_grid_size_mm, metric.finer_grid_size_mm)
			]
			writer.writerow(
				{
					"coarse_grid_size_mm": metric.coarse_grid_size_mm,
					"finer_grid_size_mm": metric.finer_grid_size_mm,
					"lineout_label": metric.lineout_label,
					"relative_integrated_absolute_difference": metric.relative_error,
					"z_min_m": metric.z_min_m,
					"z_max_m": metric.z_max_m,
					"x_min_m": metric.x_min_m,
					"x_max_m": metric.x_max_m,
					"mean_relative_error": summary.mean_relative_error,
					"standard_deviation": summary.standard_deviation,
					"minimum_relative_error": summary.minimum_relative_error,
					"maximum_relative_error": summary.maximum_relative_error,
					"mean_peak_normalized_absolute_error": (
						summary.mean_peak_normalized_absolute_error
					),
					"standard_deviation_peak_normalized_absolute_error": (
						summary.standard_deviation_peak_normalized_absolute_error
					),
					"minimum_peak_normalized_absolute_error": (
						summary.minimum_peak_normalized_absolute_error
					),
					"maximum_peak_normalized_absolute_error": (
						summary.maximum_peak_normalized_absolute_error
					),
				}
			)
	return output_path


def _positive_plot_values(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeroes so they remain visible on logarithmic axes."""
	positive = values[values > 0]
	if positive.size == 0:
		return np.full_like(values, np.finfo(float).tiny)
	return np.maximum(values, positive.min() / 10)


def _style_convergence_axis(axis: plt.Axes) -> None:
	axis.set_xscale("log")
	axis.set_yscale("log")
	axis.set_xlabel("maximum grid size [mm]")
	axis.set_ylabel("relative density error")
	axis.grid(which="both")


def _set_log_y_limits(axis: plt.Axes, values: np.ndarray) -> None:
	"""Set whole-decade log limits from the positive plotted data values."""
	positive_values = values[np.isfinite(values) & (values > 0.0)]
	if positive_values.size == 0:
		return
	y_min = 10.0 ** np.floor(np.log10(positive_values.min()))
	y_max = 10.0 ** np.ceil(np.log10(positive_values.max()))
	if y_min == y_max:
		y_min /= 10.0
		y_max *= 10.0
	axis.set_ylim(y_min, y_max)


def plot_convergence(
	metrics: list[ConvergenceMetric],
	summaries: list[ConvergenceSummary],
	*,
	view: str = "both",
	field_error: str = "both",
	title: str | None = None,
) -> plt.Figure:
	"""Plot convergence curves and selected CGNS field-error distributions."""
	if view not in {"summary", "lineouts", "both"}:
		raise ValueError(f"Unknown convergence plot view {view!r}")
	if field_error not in {"local-relative", "peak-normalized", "both"}:
		raise ValueError(f"Unknown CGNS field-error mode {field_error!r}")

	if view == "both":
		figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
		summary_axis, lineout_axis = axes
	else:
		figure, axis = plt.subplots(figsize=(8, 5))
		summary_axis = axis if view == "summary" else None
		lineout_axis = axis if view == "lineouts" else None

	if summary_axis is not None:
		ordered_summaries = sorted(summaries, key=lambda summary: summary.coarse_grid_size_mm)
		grid_sizes = np.array([summary.coarse_grid_size_mm for summary in ordered_summaries])
		means = np.array([summary.mean_relative_error for summary in ordered_summaries])
		standard_deviations = np.array(
			[summary.standard_deviation for summary in ordered_summaries]
		)
		is_field_summary = bool(ordered_summaries) and all(
			summary.mean_peak_normalized_absolute_error is not None
			for summary in ordered_summaries
		)
		show_local_relative = (
			not is_field_summary or field_error in {"local-relative", "both"}
		)
		show_peak_normalized = (
			is_field_summary and field_error in {"peak-normalized", "both"}
		)
		_style_convergence_axis(summary_axis)
		plot_values: list[np.ndarray] = []
		if show_local_relative:
			plot_means = _positive_plot_values(means)
			lower = _positive_plot_values(means - standard_deviations)
			upper = _positive_plot_values(means + standard_deviations)
			summary_axis.plot(
				grid_sizes,
				plot_means,
				marker="o",
				label=("mean local relative error" if is_field_summary else "mean"),
			)
			summary_axis.fill_between(
				grid_sizes,
				lower,
				upper,
				alpha=0.25,
				label="mean +/- 1 std",
			)
			plot_values.extend((plot_means, upper))
		if show_peak_normalized:
			peak_means = np.array(
				[
					summary.mean_peak_normalized_absolute_error
					for summary in ordered_summaries
				]
			)
			peak_standard_deviations = np.array(
				[
					summary.standard_deviation_peak_normalized_absolute_error
					for summary in ordered_summaries
				]
			)
			peak_plot_means = _positive_plot_values(peak_means)
			peak_lower = _positive_plot_values(peak_means - peak_standard_deviations)
			peak_upper = _positive_plot_values(peak_means + peak_standard_deviations)
			summary_axis.plot(
				grid_sizes,
				peak_plot_means,
				marker="s",
				label="mean absolute error / peak reference density",
			)
			summary_axis.fill_between(
				grid_sizes,
				peak_lower,
				peak_upper,
				alpha=0.25,
				label="peak-normalized mean +/- 1 std",
			)
			plot_values.extend((peak_plot_means, peak_upper))
		_set_log_y_limits(summary_axis, np.concatenate(plot_values))
		summary_axis.set_title(
			"Mean field errors" if is_field_summary else "Mean convergence across lineouts"
		)
		summary_axis.legend()

	if lineout_axis is not None:
		metrics_by_label: dict[str, list[ConvergenceMetric]] = {}
		for metric in metrics:
			metrics_by_label.setdefault(metric.lineout_label, []).append(metric)
		for label, label_metrics in sorted(metrics_by_label.items()):
			ordered_metrics = sorted(
				label_metrics, key=lambda metric: metric.coarse_grid_size_mm
			)
			grid_sizes = np.array(
				[metric.coarse_grid_size_mm for metric in ordered_metrics]
			)
			errors = _positive_plot_values(
				np.array([metric.relative_error for metric in ordered_metrics])
			)
			lineout_axis.plot(grid_sizes, errors, marker="o", label=label)
		_style_convergence_axis(lineout_axis)
		_set_log_y_limits(
			lineout_axis,
			np.array([metric.relative_error for metric in metrics]),
		)
		lineout_axis.set_title("Convergence by lineout")
		lineout_axis.legend(ncol=2, fontsize="small")

	if title:
		figure.suptitle(title)
	figure.tight_layout()
	return figure


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compare ANSYS density lineouts at each grid size with a "
			"reference resolution."
		)
	)
	parser.add_argument(
		"input_dir",
		type=Path,
		help="Directory of resolution-tagged ANSYS lineout or CGNS files, e.g. 0_1.txt",
	)
	parser.add_argument(
		"--input-format",
		choices=("auto", "lineout", "cgns"),
		default="auto",
		help="Input format (default: auto-detect from resolution-tagged files)",
	)
	parser.add_argument(
		"--view",
		choices=("summary", "lineouts", "both"),
		default="both",
		help="Plot mean-plus-spread, individual lineouts, or both (default: both)",
	)
	parser.add_argument(
		"--reference",
		choices=("finest", "adjacent"),
		default="finest",
		help=(
			"Reference comparison: finest treats the highest resolution as "
			"ground truth; adjacent uses the immediately finer grid (default: finest)"
		),
	)
	parser.add_argument(
		"--field-error",
		choices=("local-relative", "peak-normalized", "both"),
		default="both",
		help=(
			"CGNS summary metric: local-relative divides each cell's error by "
			"its reference density; peak-normalized divides the absolute-error "
			"mean and standard deviation by peak reference density; both is default"
		),
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Save the plot to this path instead of showing it interactively",
	)
	parser.add_argument(
		"--metrics-output",
		type=Path,
		default=None,
		help="CSV output path (default: alongside the plot or inside input_dir)",
	)
	parser.add_argument("--x-min", type=float, default=None, help="CGNS clip lower x bound [m]")
	parser.add_argument("--x-max", type=float, default=None, help="CGNS clip upper x bound [m]")
	parser.add_argument("--z-min", type=float, default=None, help="CGNS clip lower z bound [m]")
	parser.add_argument("--z-max", type=float, default=None, help="CGNS clip upper z bound [m]")
	parser.add_argument(
		"--x-points",
		type=int,
		default=200,
		help="CGNS common-grid points along x (default: 200)",
	)
	parser.add_argument(
		"--z-points",
		type=int,
		default=200,
		help="CGNS common-grid points along z (default: 200)",
	)
	parser.add_argument(
		"--interpolation",
		choices=("linear", "cubic"),
		default="linear",
		help=(
			"CGNS unstructured interpolation method: linear (default) or "
			"cubic Clough-Tocher"
		),
	)
	parser.add_argument("-t", "--title", default=None, help="Optional plot title")
	args = parser.parse_args()

	input_format = args.input_format
	if input_format == "auto":
		cgns_paths = list(args.input_dir.glob("*.cgns"))
		input_format = "cgns" if cgns_paths else "lineout"
	if input_format == "cgns":
		fields = load_resolution_cgns_fields(args.input_dir)
		metrics = calculate_field_convergence(
			fields,
			reference_mode=args.reference,
			x_min=args.x_min,
			x_max=args.x_max,
			z_min=args.z_min,
			z_max=args.z_max,
			x_points=args.x_points,
			z_points=args.z_points,
			interpolation=args.interpolation,
		)
	else:
		resolutions = load_resolution_lineouts(args.input_dir)
		metrics = calculate_convergence(resolutions, reference_mode=args.reference)
	summaries = summarize_convergence(metrics)
	plot_view = "summary" if input_format == "cgns" else args.view
	figure = plot_convergence(
		metrics,
		summaries,
		view=plot_view,
		field_error=args.field_error,
		title=args.title,
	)

	metrics_output = args.metrics_output
	if metrics_output is None:
		metrics_output = (
			args.output.with_suffix(".csv")
			if args.output is not None
			else args.input_dir / "convergence_metrics.csv"
		)
	metrics_path = write_convergence_csv(metrics_output, metrics, summaries)
	print(f"Wrote {metrics_path.resolve()}")

	if args.output is not None:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(args.output, dpi=150, bbox_inches="tight")
		print(f"Wrote {args.output.resolve()}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
