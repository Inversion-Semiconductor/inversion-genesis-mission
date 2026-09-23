from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal
import warnings

import attrs
from inversion_fbpic.lib.density_core import _DensityProfile, DensityCallable
import numpy as np
from numpy import typing as npt
import h5py
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class _DensityGrid:
    """Validated density values and their coordinate axes."""

    dimension_labels: tuple[str, ...]
    coordinates: tuple[npt.NDArray[np.floating], ...]
    raw_density: npt.NDArray[np.floating]


@dataclass(frozen=True)
class _LineoutConfiguration:
    """Validated parameterized path and fixed coordinates for a density lineout."""

    path_axes: dict[str, tuple[float, float]]
    interpolation_points: dict[str, float]
    scalar_axis: str | None


@attrs.define(kw_only=True, slots=False, frozen=True)
class InterpolateFromH5Profile(_DensityProfile):
    """Longitudinal density profile interpolated from an HDF5 density dataset.

    File structure:
        The HDF5 file must contain a root-level N-dimensional density dataset named
        by ``density_name``, with finite values. Its ``DIMENSION_LABELS`` attribute
        must contain one unique label per dimension in dataset-shape order. For each
        label, the file must contain a root-level one-dimensional coordinate dataset
        with at least two values, the same length as its corresponding density
        dimension, and finite, strictly increasing values. Density values must be
        finite and non-negative.

        ``lineout_axis`` can select one coordinate varied to create the longitudinal
        lineout, or define a parameterized linear path through multiple coordinates.
        In the path form, each coordinate follows
        ``coordinate(t) = origin + coefficient * t``.
        ``interpolation_points`` must provide exactly one in-bounds position for every
        labeled coordinate not included in the lineout. The selected lineout is
        normalized to its positive peak, which becomes ``nominal_density``; values
        outside the lineout coordinate range evaluate to zero.

        For example, ``my_profile.h5`` contains a root-level ``density`` dataset
        with ``DIMENSION_LABELS = ["z_m", "x_mm", "pressure_bar"]`` and coordinate
        datasets named ``z_m``, ``x_mm``, and ``pressure_bar``. A lineout can use
        ``lineout_axis="z_m"`` with
        ``interpolation_points={"x_mm": 2.0, "pressure_bar": 1.0}``, or an oblique
        ``x_mm``-``z_m`` line with
        ``lineout_axis={"z_m": {"origin": 0.0, "coefficient": 1 / sqrt(2)},
        "x_mm": {"origin": 2.0, "coefficient": 1.0e-3 / sqrt(2)}}`` and
        ``interpolation_points={"pressure_bar": 1.0}``. The user is responsible for
        choosing coefficients with compatible units.

    Args:
        filename: (str | Path) Path to the HDF5 file. Config-file serialization
            records it relative to the config file, including ``..`` segments,
            so a config and its referenced data can be relocated together.
        density_name: (str) Name of the multidimensional density dataset.
        lineout_axis: (str | dict[str, dict[str, float]]) Coordinate dataset varied to form the longitudinal lineout, or parameterized linear path definitions. In the dictionary form, every key is a coordinate dataset and its value must contain finite ``origin`` and ``coefficient`` values defining ``coordinate(t) = origin + coefficient * t``. At least one coefficient must be nonzero; zero coefficients keep their coordinate fixed at its origin. The returned density function uses the sampled ``t`` coordinate after applying ``centering_mode`` and ``longitudinal_offset``.
        density_scale: (float) |OPTIONAL| Scalar multiple to apply to the density. Defaults to 1.0.
        interpolation_points: (dict[str, float]) |OPTIONAL| Positions for every other coordinate axis. Defaults to {}.
        neutral: (bool) |OPTIONAL| Whether the data is of the neutral atomic density or of the ionized electron density. Defaults to True.
        centroid_axis: (str | None) |OPTIONAL| Coordinate dataset used to calculate the density-weighted centroid for ``centering_mode="centroid"``. If provided, this axis is varied while other parameterized-linear-path axes are fixed at their origins and remaining axes are fixed by ``interpolation_points``. It must be included in ``lineout_axis`` with a nonzero coefficient so its centroid can be converted to the lineout parameter. Defaults to None, which uses the selected lineout.
        longitudinal_offset: (float) [m] |OPTIONAL| Offset applied to the lineout coordinates. Defaults to 0.0.
        density_cutoff_ratio: (float) |OPTIONAL| Relative-to-peak cutoff used to determine the longitudinal extent. Defaults to 1e-3.
        centering_mode: (Literal[str]) |OPTIONAL| Density centering method.
            If `'centroid'`, the density-weighted centroid will be located at `z=longitudinal_offset`.
            If `'left'`, the cutoff-derived left extent will be located at `z=longitudinal_offset`.
            If `'right'`, the cutoff-derived right extent will be located at `z=longitudinal_offset`.
            Defaults to `'left'`.
        nominal_density: (float) [m^-3] |NOT REFERENCED| The peak density for this profile. This quantity is determined by the dataset used.
    """

    SUBCLASS: ClassVar[str] = "interp_from_h5"

    # The selected HDF5 lineout supplies the physical density scale.
    nominal_density: float = attrs.field(init=False, default=0.0)
    filename: str | Path = attrs.field(converter=Path, metadata={"input_path": True})
    density_name: str = attrs.field(converter=str)
    neutral: bool = attrs.field(converter=bool, default=True)
    lineout_axis: str | dict[str, dict[str, float]] = attrs.field()
    centroid_axis: str | None = attrs.field(
        default=None, converter=attrs.converters.optional(str)
    )
    density_scale: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0), default=1.0
    )
    interpolation_points: dict[str, float] = attrs.field(factory=dict)
    longitudinal_offset: float = attrs.field(default=0.0, converter=float)
    density_cutoff_ratio: float = attrs.field(
        default=1e-3,
        converter=float,
        validator=[attrs.validators.ge(0.0), attrs.validators.lt(1.0)],
    )
    centering_mode: Literal["centroid", "left", "right"] = attrs.field(
        default="left",
        validator=attrs.validators.in_(["centroid", "left", "right"]),
    )

    lineout: RegularGridInterpolator = attrs.field(init=False, repr=False, eq=False)
    centroid: float = attrs.field(init=False)
    z_extent: tuple[float, float] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

        density_grid = self._read_density_grid()
        lineout_configuration = self._validate_lineout_configuration(
            density_grid.dimension_labels
        )
        centroid_axis = self._validate_centroid_axis(
            density_grid.dimension_labels, lineout_configuration.path_axes
        )

        # Sample the multidimensional density at fixed transverse conditions.
        longitudinal_coordinates, lineout_values = self._extract_lineout(
            density_grid, lineout_configuration
        )
        lineout_values *= self.density_scale
        normalized_lineout, nominal_density = self._normalize_lineout(lineout_values)
        if self.neutral and self.species is not None:
            nominal_density *= self._get_num_ionization_levels(self.species)
        object.__setattr__(self, "nominal_density", nominal_density)

        # Derive lineout support before shifting into simulation coordinates.
        centroid, z_extent = self._derive_metadata(
            longitudinal_coordinates, normalized_lineout
        )
        if centroid_axis is not None:
            centroid = self._slice_centroid(
                density_grid,
                lineout_configuration,
                centroid_axis,
            )
        anchor = {
            "centroid": centroid,
            "left": z_extent[0],
            "right": z_extent[1],
        }[self.centering_mode]
        coordinate_shift = self.longitudinal_offset - anchor
        shifted_coordinates = longitudinal_coordinates + coordinate_shift

        # The final one-dimensional interpolator represents a radially uniform density profile.
        object.__setattr__(
            self,
            "lineout",
            RegularGridInterpolator(
                (shifted_coordinates,),
                normalized_lineout,
                bounds_error=False,
                fill_value=0.0,
            ),
        )
        object.__setattr__(self, "centroid", centroid + coordinate_shift)
        object.__setattr__(
            self,
            "z_extent",
            tuple(bound + coordinate_shift for bound in z_extent),
        )

    def _read_density_grid(
        self,
    ) -> _DensityGrid:
        """Load and validate the labeled density grid from the source HDF5 file.

        Returns
        -------
        _DensityGrid
            Validated dimension labels, strictly increasing coordinate arrays in
            density-dimension order, and finite non-negative density values.

        Raises
        ------
        ValueError
            If the density dataset or coordinate datasets are missing or malformed,
            or if density or coordinate values fail validation.
        """
        with h5py.File(self.filename, "r") as h5_file:
            if self.density_name not in h5_file:
                raise ValueError(
                    f"Density dataset {self.density_name!r} was not found in {self.filename}."
                )

            density_dataset = h5_file[self.density_name]
            if (
                not isinstance(density_dataset, h5py.Dataset)
                or density_dataset.ndim == 0
            ):
                raise ValueError(
                    f"Density dataset {self.density_name!r} must be at least one-dimensional."
                )
            if "DIMENSION_LABELS" not in density_dataset.attrs:
                raise ValueError(
                    f"Density dataset {self.density_name!r} must define DIMENSION_LABELS."
                )

            dimension_labels = tuple(
                label.decode() if isinstance(label, bytes) else str(label)
                for label in density_dataset.attrs["DIMENSION_LABELS"]
            )
            raw_density = np.asarray(density_dataset, dtype=float)

            if len(dimension_labels) != raw_density.ndim:
                raise ValueError(
                    "DIMENSION_LABELS must provide one label for each density dimension."
                )
            if len(set(dimension_labels)) != len(dimension_labels):
                raise ValueError("DIMENSION_LABELS must not contain duplicate labels.")
            if not np.all(np.isfinite(raw_density)):
                raise ValueError("Density dataset must contain only finite values.")
            if np.any(raw_density < 0.0):
                raise ValueError("Density dataset must not contain negative values.")

            # Each label must identify a root-level coordinate dataset for its array dimension.
            coordinates: list[npt.NDArray[np.floating]] = []
            for axis_index, label in enumerate(dimension_labels):
                if label not in h5_file:
                    raise ValueError(
                        f"Coordinate dataset {label!r} was not found in {self.filename}."
                    )
                coordinate = np.asarray(h5_file[label], dtype=float)
                if (
                    coordinate.ndim != 1
                    or coordinate.size != raw_density.shape[axis_index]
                ):
                    raise ValueError(
                        f"Coordinate dataset {label!r} must be one-dimensional with "
                        f"length {raw_density.shape[axis_index]}."
                    )
                if coordinate.size < 2:
                    raise ValueError(
                        f"Coordinate dataset {label!r} must contain at least two values."
                    )
                if not np.all(np.isfinite(coordinate)) or np.any(
                    np.diff(coordinate) <= 0.0
                ):
                    raise ValueError(
                        f"Coordinate dataset {label!r} must be finite and strictly increasing."
                    )
                coordinates.append(coordinate)

        return _DensityGrid(
            dimension_labels=dimension_labels,
            coordinates=tuple(coordinates),
            raw_density=raw_density,
        )

    def _validate_lineout_configuration(
        self, dimension_labels: tuple[str, ...]
    ) -> _LineoutConfiguration:
        """Validate and normalize the density lineout configuration.

        Parameters
        ----------
        dimension_labels : tuple[str, ...]
            Coordinate labels from the density dataset, ordered by array dimension.

        Returns
        -------
        _LineoutConfiguration
            Parameterized linear path definitions, finite fixed interpolation
            coordinates, and the scalar coordinate label when ``lineout_axis`` is a
            string.

        Raises
        ------
        ValueError
            If lineout labels or parameterized linear path definitions are invalid,
            no path coefficient varies, or interpolation points do not exactly cover
            non-path axes.
        """
        if isinstance(self.lineout_axis, str):
            if self.lineout_axis not in dimension_labels:
                raise ValueError(
                    f"lineout_axis {self.lineout_axis!r} is not in DIMENSION_LABELS."
                )
            path_axes = {self.lineout_axis: (0.0, 1.0)}
            scalar_axis = self.lineout_axis
        elif isinstance(self.lineout_axis, dict) and self.lineout_axis:
            path_axes = {}
            scalar_axis = None
            has_nonzero_coefficient = False
            for label, definition in self.lineout_axis.items():
                if label not in dimension_labels:
                    raise ValueError(
                        f"lineout_axis label {label!r} is not in DIMENSION_LABELS."
                    )
                if not isinstance(definition, dict) or set(definition) != {
                    "origin",
                    "coefficient",
                }:
                    raise ValueError(
                        f"lineout_axis definition for {label!r} must contain exactly "
                        "'origin' and 'coefficient'."
                    )
                try:
                    origin = float(definition["origin"])
                    coefficient = float(definition["coefficient"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"lineout_axis definition for {label!r} must use numeric "
                        "'origin' and 'coefficient' values."
                    ) from error
                if not np.isfinite(origin) or not np.isfinite(coefficient):
                    raise ValueError(
                        f"lineout_axis definition for {label!r} must use finite values."
                    )
                if coefficient != 0.0:
                    has_nonzero_coefficient = True
                path_axes[label] = (origin, coefficient)
            if not has_nonzero_coefficient:
                raise ValueError(
                    "lineout_axis must contain at least one nonzero coefficient."
                )
        else:
            raise ValueError(
                "lineout_axis must be a coordinate label or a non-empty dictionary "
                "of parameterized linear path definitions."
            )

        expected_points = set(dimension_labels) - set(path_axes)
        provided_points = set(self.interpolation_points)
        if provided_points != expected_points:
            missing = sorted(expected_points - provided_points)
            unexpected = sorted(provided_points - expected_points)
            details = []
            if missing:
                details.append(f"missing interpolation points for {missing}")
            if unexpected:
                details.append(f"unexpected interpolation points for {unexpected}")
            raise ValueError("; ".join(details))

        interpolation_points = {}
        for label, value in self.interpolation_points.items():
            try:
                interpolation_point = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"interpolation point for {label!r} must be finite."
                ) from error
            if not np.isfinite(interpolation_point):
                raise ValueError(f"interpolation point for {label!r} must be finite.")
            interpolation_points[label] = interpolation_point
        return _LineoutConfiguration(
            path_axes=path_axes,
            interpolation_points=interpolation_points,
            scalar_axis=scalar_axis,
        )

    def _validate_centroid_axis(
        self,
        dimension_labels: tuple[str, ...],
        path_axes: dict[str, tuple[float, float]],
    ) -> str | None:
        """Validate the optional axis used for fixed-slice centroid centering.

        Parameters
        ----------
        dimension_labels : tuple[str, ...]
            Coordinate labels from the density dataset, ordered by array dimension.
        path_axes : dict[str, tuple[float, float]]
            Validated parameterized linear path definitions mapping labels to
            ``(origin, coefficient)`` pairs.

        Returns
        -------
        str or None
            The validated centroid-axis label, or ``None`` when the centroid of the
            selected lineout should be used.

        Raises
        ------
        ValueError
            If the requested centroid axis is not a path axis or has a zero path
            coefficient.
        """
        if self.centroid_axis is None:
            return None
        if self.centroid_axis not in dimension_labels:
            raise ValueError(
                f"centroid_axis {self.centroid_axis!r} is not in DIMENSION_LABELS."
            )
        if self.centroid_axis not in path_axes:
            raise ValueError(
                "centroid_axis must be included in lineout_axis so it can be "
                "converted to the lineout parameter."
            )
        if path_axes[self.centroid_axis][1] == 0.0:
            raise ValueError(
                "centroid_axis must have a nonzero lineout_axis coefficient so it "
                "can be converted to the lineout parameter."
            )
        return self.centroid_axis

    @staticmethod
    def _lineout_coordinates(
        coordinate_by_label: dict[str, npt.NDArray[np.floating]],
        path_axes: dict[str, tuple[float, float]],
        scalar_axis: str | None,
    ) -> npt.NDArray[np.floating]:
        """Return source coordinates for scalar lineouts or path parameters.

        Parameters
        ----------
        coordinate_by_label : dict[str, numpy.ndarray]
            Strictly increasing one-dimensional coordinate arrays keyed by dataset
            label.
        path_axes : dict[str, tuple[float, float]]
            Parameterized linear path definitions mapping each path label to an
            ``(origin, coefficient)`` pair.
        scalar_axis : str or None
            Label of a scalar lineout axis. ``None`` selects parameterized linear
            path sampling.

        Returns
        -------
        numpy.ndarray
            The scalar axis values unchanged, or evenly spaced path parameters over
            the intersection of every varying path coordinate's bounds.

        Raises
        ------
        ValueError
            If the parameterized linear path does not intersect the density grid.
        """
        if scalar_axis is not None:
            # Preserve the source grid exactly; scalar lineouts may be nonuniform.
            return coordinate_by_label[scalar_axis]

        path_bounds = []
        for label, (origin, coefficient) in path_axes.items():
            if coefficient == 0.0:
                continue
            coordinate = coordinate_by_label[label]
            bounds = (
                (coordinate[0] - origin) / coefficient,
                (coordinate[-1] - origin) / coefficient,
            )
            path_bounds.append((float(min(bounds)), float(max(bounds))))

        t_min = max(lower_bound for lower_bound, _ in path_bounds)
        t_max = min(upper_bound for _, upper_bound in path_bounds)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min > t_max:
            raise ValueError("lineout_axis path does not intersect the density grid.")

        # A parameterized linear path has no preferred source axis, so sample at the
        # finest resolution supplied by any coordinate participating in the path.
        num_points = sum(
            coordinate_by_label[label].size
            for label, (_, coefficient) in path_axes.items()
            if coefficient != 0.0
        )
        return np.linspace(t_min, t_max, num_points)

    @staticmethod
    def _slice_centroid(
        density_grid: _DensityGrid,
        lineout_configuration: _LineoutConfiguration,
        centroid_axis: str,
    ) -> float:
        """Return a fixed-slice centroid expressed in the lineout parameter.

        Parameters
        ----------
        density_grid : _DensityGrid
            Validated coordinate labels, coordinate arrays, and density values.
        lineout_configuration : _LineoutConfiguration
            Validated parameterized linear path definitions and fixed interpolation
            coordinates.
        centroid_axis : str
            Varying path label used for the density-weighted centroid slice.

        Returns
        -------
        float
            Density-weighted centroid converted from ``centroid_axis`` coordinates
            to the parameterized lineout coordinate.

        Raises
        ------
        ValueError
            If the fixed slice is out of bounds or has no positive total density.
        """
        dimension_labels = density_grid.dimension_labels
        coordinates = density_grid.coordinates
        raw_density = density_grid.raw_density
        path_axes = lineout_configuration.path_axes
        interpolation_points = lineout_configuration.interpolation_points
        centroid_axis_index = dimension_labels.index(centroid_axis)
        centroid_coordinates = coordinates[centroid_axis_index]
        query_points = np.empty((centroid_coordinates.size, raw_density.ndim))
        for axis_index, label in enumerate(dimension_labels):
            if label == centroid_axis:
                query_points[:, axis_index] = centroid_coordinates
            elif label in path_axes:
                # A fixed-slice centroid uses the supplied path origin, not the
                # position reached by any particular lineout sample.
                query_points[:, axis_index] = path_axes[label][0]
            else:
                query_points[:, axis_index] = interpolation_points[label]

        try:
            density_values = RegularGridInterpolator(
                coordinates, raw_density, bounds_error=True
            )(query_points)
        except ValueError as error:
            raise ValueError(
                "centroid_axis slice, lineout_axis origins, and "
                "interpolation_points must lie within their coordinate dataset "
                "bounds."
            ) from error

        if float(np.sum(density_values)) <= 0.0:
            raise ValueError("centroid_axis slice must have a positive total density.")
        centroid_coordinate = float(
            np.average(centroid_coordinates, weights=density_values)
        )
        origin, coefficient = path_axes[centroid_axis]
        # Centering shifts the density function's t coordinate, not an HDF5 axis.
        return (centroid_coordinate - origin) / coefficient

    def _extract_lineout(
        self,
        density_grid: _DensityGrid,
        lineout_configuration: _LineoutConfiguration,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Sample the density grid along a scalar axis or parameterized linear path.

        Parameters
        ----------
        density_grid : _DensityGrid
            Validated coordinate labels, coordinate arrays, and density values.
        lineout_configuration : _LineoutConfiguration
            Validated parameterized linear path definitions, fixed interpolation
            coordinates, and optional scalar lineout axis.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Sampled longitudinal coordinates or path parameters and the corresponding
            unnormalized density values.

        Raises
        ------
        ValueError
            If the requested path or fixed interpolation coordinates are outside the
            density-grid bounds.
        """
        dimension_labels = density_grid.dimension_labels
        coordinates = density_grid.coordinates
        raw_density = density_grid.raw_density
        path_axes = lineout_configuration.path_axes
        interpolation_points = lineout_configuration.interpolation_points
        scalar_axis = lineout_configuration.scalar_axis
        full_interpolator = RegularGridInterpolator(
            coordinates, raw_density, bounds_error=True
        )
        coordinate_by_label = dict(zip(dimension_labels, coordinates, strict=True))
        longitudinal_coordinates = self._lineout_coordinates(
            coordinate_by_label, path_axes, scalar_axis
        )

        # Vary the path coordinates together while holding all remaining axes fixed.
        query_points = np.empty((longitudinal_coordinates.size, raw_density.ndim))
        for axis_index, label in enumerate(dimension_labels):
            if label in path_axes:
                origin, coefficient = path_axes[label]
                if coefficient == 0.0:
                    query_points[:, axis_index] = origin
                else:
                    values = origin + coefficient * longitudinal_coordinates
                    # Bound clipping only absorbs endpoint roundoff after intersecting
                    # the path with every participating coordinate grid.
                    query_points[:, axis_index] = np.clip(
                        values, coordinates[axis_index][0], coordinates[axis_index][-1]
                    )
            else:
                query_points[:, axis_index] = interpolation_points[label]

        try:
            lineout_values = full_interpolator(query_points)
        except ValueError as error:
            raise ValueError(
                "lineout_axis path and interpolation_points must lie within their "
                "coordinate dataset bounds."
            ) from error

        return longitudinal_coordinates, lineout_values

    @staticmethod
    def _normalize_lineout(
        lineout_values: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], float]:
        """Normalize a lineout by its positive finite peak density.

        Parameters
        ----------
        lineout_values : numpy.ndarray
            Sampled, unnormalized density values.

        Returns
        -------
        tuple[numpy.ndarray, float]
            Lineout values normalized to a peak of one and the original peak density.

        Raises
        ------
        ValueError
            If the lineout peak is non-finite or not positive.
        """
        peak_density = float(np.max(lineout_values))
        if not np.isfinite(peak_density) or peak_density <= 0.0:
            raise ValueError(
                "The selected density lineout must have a positive finite peak."
            )
        return lineout_values / peak_density, peak_density

    def _derive_metadata(
        self,
        longitudinal_coordinates: npt.NDArray[np.floating],
        normalized_lineout: npt.NDArray[np.floating],
    ) -> tuple[float, tuple[float, float]]:
        """Calculate the density-weighted centroid and cutoff-bounded extent.

        Parameters
        ----------
        longitudinal_coordinates : numpy.ndarray
            Sampled source coordinates or parameterized linear path coordinates for
            the lineout.
        normalized_lineout : numpy.ndarray
            Lineout density values normalized to a positive peak of one.

        Returns
        -------
        tuple[float, tuple[float, float]]
            Density-weighted centroid and the first and last sampled coordinates at
            or above ``density_cutoff_ratio``.

        Raises
        ------
        ValueError
            If no lineout sample reaches ``density_cutoff_ratio``.
        """
        # The centroid preserves the physical center of mass of the selected lineout.
        centroid = float(
            np.average(longitudinal_coordinates, weights=normalized_lineout)
        )

        # Use sampled cutoff locations, preserving the input grid's spatial resolution.
        in_support = normalized_lineout >= self.density_cutoff_ratio
        if not np.any(in_support):
            raise ValueError(
                "The selected density lineout does not reach density_cutoff_ratio."
            )
        if (in_support[0] or in_support[-1]) and self.density_cutoff_ratio > 0.0:
            warnings.warn(
                "The selected density lineout reaches a data boundary "
                "while still at or above density_cutoff_ratio; its extent is "
                "truncated by the available data.",
                UserWarning,
                stacklevel=4,
            )
        support_coordinates = longitudinal_coordinates[in_support]

        return centroid, (float(support_coordinates[0]), float(support_coordinates[-1]))

    def get_z_extent(self) -> tuple[float, float]:
        return self.z_extent

    def get_r_extent(self) -> float | None:
        return None

    def build_density_function(self) -> DensityCallable:
        def dens_func(z: npt.ArrayLike, r: npt.ArrayLike) -> npt.ArrayLike:
            z_values, _ = np.broadcast_arrays(np.asarray(z), np.asarray(r))
            density = np.asarray(self.lineout(z_values[..., np.newaxis])).reshape(
                z_values.shape
            )
            return density.item() if density.ndim == 0 else density

        return dens_func
