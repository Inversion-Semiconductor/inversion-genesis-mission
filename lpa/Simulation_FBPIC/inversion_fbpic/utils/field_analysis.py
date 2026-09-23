"""Bandpass-based analysis of FBPIC laser fields.

This module is the computational backend for the
``analyze-laser-evolution`` script. It exposes two public functions:

  * :func:`analyze_iteration` -- for a single iteration in an OpenPMD time
    series, build a cosine-squared bandpass centred on the laser
    wavenumber, integrate the in-band EM energy density (assuming
    cylindrical symmetry), and use the analytic-signal envelope of the
    in-band electric field to compute the on-axis intensity and the
    engineering ``a0`` at the location of the laser peak.

  * :func:`analyze_laser_evolution` -- apply :func:`analyze_iteration` to
    every iteration in a time series, optionally write the per-iteration
    summary records to JSON, and return the dict that was (or would have
    been) written to JSON. An optional ``on_iteration`` callback is
    invoked with the full per-iteration record -- including the field
    arrays needed for plotting -- so that plotting code can live entirely
    outside of this module.

The :func:`cosine_squared_band` helper is also exported as the canonical
bandpass kernel; it is reused by :mod:`inversion_fbpic.utils.plotting`.

Both top-level functions are intentionally plotting-free so that they
can be imported from other analyses without pulling in matplotlib.

Typical usage example:

    from inversion_fbpic.utils.field_analysis import analyze_laser_evolution

    results = analyze_laser_evolution(
        "diags/hdf5",
        laser_wavelength=0.8e-6,
        results_file="laser_evolution_results.json",
        verbose=True,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Union, Optional

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import epsilon_0, mu_0, pi


A0_ENGINEERING_FACTOR: float = 7.3e-19
"""Prefactor in ``a0 = sqrt(C * lambda[um]^2 * I[W/cm^2])`` (engineering).
Equivalent to standard 0.85**2 x 10^-18"""


def _open_series(series_path: Union[Path, str]) -> OpenPMDTimeSeries:
    """Opens an OpenPMD time series, falling back to ``<path>/hdf5`` if needed.

    Args:
        series_path: Either the HDF5 directory itself or its parent (an
            ``hdf5`` subdirectory will be tried automatically).

    Returns:
        An opened :class:`openpmd_viewer.OpenPMDTimeSeries` for the
        located directory.

    Raises:
        FileNotFoundError: If neither ``series_path`` nor
            ``series_path / "hdf5"`` contains a valid time series.
    """
    series_path = Path(series_path)
    try:
        return OpenPMDTimeSeries(str(series_path))
    except (FileNotFoundError, OSError):
        fallback = series_path / "hdf5"
        try:
            return OpenPMDTimeSeries(str(fallback))
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(
                f"No HDF5 files found in `{series_path}` or `{fallback}`"
            ) from exc


def cosine_squared_band(k: np.ndarray, k0: float, kw: float) -> np.ndarray:
    """Cosine-squared bandpass window in wavenumber space.

    Evaluates to 1 at ``k == k0``, smoothly falls to 0 at
    ``k == k0 +/- kw`` (and stays at 0 beyond). This is the canonical
    bandpass kernel used by both the laser-band analysis in this module
    and the multi-band split in
    :func:`inversion_fbpic.utils.plotting.em_bandpass_filter`.

    Args:
        k: Wavenumber grid (any shape) at which to evaluate the window.
        k0: Centre wavenumber of the band.
        kw: Half-width of the band; the window is zero where
            ``|k - k0| > kw``.

    Returns:
        Window values in ``[0, 1]`` with the same shape as ``k``.
    """
    if kw <= 0:
        raise ValueError(f"Half-width parameter {kw} must be positive.")
    win = np.cos(pi / 2.0 / kw * (k - k0)) ** 2
    win[np.abs(k - k0) > kw] = 0
    return win


def _build_laser_bandpass(
    shape: tuple[int, int],
    bbox: tuple[float, float, float, float],
    laser_k: float,
    band_half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a cosine-squared bandpass window centred on the laser wavenumber.

    The window is evaluated on the 2D radial-wavenumber grid
    ``sqrt(kz^2 + kr^2)`` so it captures forward-propagating laser modes
    independently of small angular offsets. By construction
    ``win(k_radial = laser_k) = 1``.

    Args:
        shape: ``(n_r, n_z)`` shape of the RZ field arrays.
        bbox: ``(zmin, zmax, rmin, rmax)`` of the data grid in metres.
        laser_k: Centre wavenumber of the bandpass in rad/m, typically
            ``2*pi / laser_wavelength``.
        band_half_width: Half-width of the bandpass in rad/m.

    Returns:
        A ``(win, kz)`` tuple containing the bandpass window and the
        signed longitudinal-wavenumber grid (rad/m). The latter is
        consumed by :func:`_analytic_signal_in_band`.
    """
    nr, nz = shape
    dz = (bbox[1] - bbox[0]) / nz
    dr = (bbox[3] - bbox[2]) / nr
    kz_1d = 2.0 * pi * np.fft.fftfreq(nz, d=dz)
    kr_1d = 2.0 * pi * np.fft.fftfreq(nr, d=dr)
    kz, kr = np.meshgrid(kz_1d, kr_1d)
    k_radial = np.sqrt(kz**2 + kr**2)
    win = cosine_squared_band(k_radial, laser_k, band_half_width)
    return win, kz


def _bandpassed_energy_density(
    fields: tuple[np.ndarray, ...], win: np.ndarray
) -> np.ndarray:
    """Computes the EM energy density of the bandpassed field components.

    Each component is multiplied by ``win`` in Fourier space, transformed
    back to real space, and squared (``.real ** 2``, since the window
    inherits the Hermitian symmetry of the real input field). The result
    is the standard ``0.5*eps0*|E|^2 + 0.5/mu0*|B|^2`` energy density of
    the in-band part of the field.

    Args:
        fields: Tuple ``(Ex, Ey, Ez, Bx, By, Bz)`` of real field
            components on the RZ grid.
        win: 2D Fourier bandpass window matching the field shape.

    Returns:
        Energy density of the bandpassed field, in J/m^3, with the same
        shape as the input components.
    """
    ex_sqr, ey_sqr, ez_sqr, bx_sqr, by_sqr, bz_sqr = [
        np.fft.ifft2(np.fft.fft2(f) * win).real ** 2 for f in fields
    ]
    return 0.5 * epsilon_0 * (ex_sqr + ey_sqr + ez_sqr) + 0.5 / mu_0 * (
        bx_sqr + by_sqr + bz_sqr
    )


def _analytic_signal_in_band(
    field: np.ndarray, win: np.ndarray, kz: np.ndarray
) -> np.ndarray:
    """Returns the forward-propagating analytic signal of a real field.

    Keeps only positive ``kz`` components inside the supplied bandpass
    window and multiplies by 2 to compensate for the discarded
    negative-frequency half. ``|.|`` of the result is the slowly varying
    laser-field envelope, free of the fast ``2*omega`` oscillation.

    Args:
        field: Real 2D field array on the RZ grid.
        win: 2D Fourier bandpass window matching the field shape.
        kz: Signed longitudinal-wavenumber grid (rad/m) matching the
            field shape.

    Returns:
        Complex-valued analytic signal with the same shape as ``field``.
    """
    analytic_win = np.where(kz > 0, win, 0.0)
    return 2.0 * np.fft.ifft2(np.fft.fft2(field) * analytic_win)


def _cylindrical_energy(
    energy_density: np.ndarray,
    r_axis: np.ndarray,
    dz: float,
    dr: float,
    rmax_window: Optional[float],
) -> float:
    """Integrates an RZ energy density assuming cylindrical symmetry.

    The data spans both signs of ``r`` (an x-z slice through the moving
    window); rows with ``r < 0`` are dropped to avoid double-counting and
    the standard ``2*pi*r dr dz`` volume element is applied.

    Args:
        energy_density: ``(n_r, n_z)`` array of energy density in J/m^3.
        r_axis: 1D radial coordinate array (m) of length ``n_r``.
        dz: Longitudinal grid spacing in metres.
        dr: Radial grid spacing in metres.
        rmax_window: If not ``None``, restrict the integration to rows
            with ``0 <= r <= rmax_window`` (m).

    Returns:
        Total energy in Joules. Zero if no rows survive the mask.
    """
    mask = r_axis >= 0
    if rmax_window is not None:
        mask = mask & (r_axis <= rmax_window)
    r_pos = r_axis[mask]
    if r_pos.size == 0:
        return 0.0
    block = energy_density[mask, :]
    return float(2.0 * pi * np.sum(block * r_pos[:, None]) * dr * dz)


def _intensity_to_a0(intensity_w_m2: float, wavelength_m: float) -> float:
    """Converts intensity to the engineering normalised vector potential ``a0``.

    Uses ``a0 = sqrt(7.3e-19 * lambda[um]^2 * I[W/cm^2])``.

    Args:
        intensity_w_m2: Peak intensity in W/m^2.
        wavelength_m: Laser central wavelength in metres.

    Returns:
        Dimensionless normalised vector potential ``a0``.
    """
    wavelength_um = wavelength_m * 1e6
    intensity_w_cm2 = intensity_w_m2 * 1e-4
    return float(np.sqrt(A0_ENGINEERING_FACTOR * wavelength_um**2 * intensity_w_cm2))


def analyze_laser_iteration(
    series: OpenPMDTimeSeries,
    iteration: int,
    *,
    laser_wavelength: float,
    band_half_width_frac: float,
    rmax_window: Optional[float],
    on_axis_rmax: float,
) -> dict[str, Any]:
    """Computes energy, on-axis ``a0`` and peak position for a single iteration.

    Args:
        series: An opened :class:`openpmd_viewer.OpenPMDTimeSeries`
            exposing the EM fields.
        iteration: Iteration number to analyse.
        laser_wavelength: Central laser wavelength in metres.
        band_half_width_frac: Half-width of the cosine-squared bandpass,
            expressed as a fraction of the laser wavenumber.
        rmax_window: Radial limit (m) for the cylindrical-symmetry
            energy integration. ``None`` uses the full radial extent.
        on_axis_rmax: Half-width (m) of the near-axis region used to find
            the longitudinal peak of the laser envelope.

    Returns:
        Dict with two top-level keys:

        * ``"summary"`` -- scalar, JSON-serialisable results.
          Keys: ``iteration`` (int), ``time_s`` (float or ``None``),
          ``z_peak_m``, ``energy_j``, ``intensity_w_cm2`` and ``a0``
          (floats).
        * ``"fields"`` -- arrays and indices for plotting.
          Keys: ``laser_energy_density``, ``envelope_energy_density``,
          ``z_axis``, ``r_axis`` (numpy arrays), plus ``z_peak_idx`` and
          ``r_peak_idx`` (ints).
    """
    ex, info = series.get_field("E", "x", iteration=iteration)
    ey, _ = series.get_field("E", "y", iteration=iteration)
    ez, _ = series.get_field("E", "z", iteration=iteration)
    bx, _ = series.get_field("B", "x", iteration=iteration)
    by, _ = series.get_field("B", "y", iteration=iteration)
    bz, _ = series.get_field("B", "z", iteration=iteration)

    bbox = (info.zmin, info.zmax, info.rmin, info.rmax)
    nr, nz = ex.shape

    laser_k = 2.0 * pi / laser_wavelength
    band_half_width = laser_k * band_half_width_frac
    win, kz = _build_laser_bandpass(ex.shape, bbox, laser_k, band_half_width)

    laser_energy_density = _bandpassed_energy_density((ex, ey, ez, bx, by, bz), win)

    z_axis = np.linspace(info.zmin, info.zmax, nz)
    r_axis = np.linspace(info.rmin, info.rmax, nr)
    dz = float((info.zmax - info.zmin) / nz)
    dr = float((info.rmax - info.rmin) / nr)

    total_energy_j = _cylindrical_energy(
        laser_energy_density, r_axis, dz, dr, rmax_window
    )

    ex_a = _analytic_signal_in_band(ex, win, kz)
    ey_a = _analytic_signal_in_band(ey, win, kz)
    ez_a = _analytic_signal_in_band(ez, win, kz)
    e_envelope_sqr = np.abs(ex_a) ** 2 + np.abs(ey_a) ** 2 + np.abs(ez_a) ** 2

    bx_a = _analytic_signal_in_band(bx, win, kz)
    by_a = _analytic_signal_in_band(by, win, kz)
    bz_a = _analytic_signal_in_band(bz, win, kz)
    envelope_energy_density = 0.5 * epsilon_0 * e_envelope_sqr + 0.5 / mu_0 * (
        np.abs(bx_a) ** 2 + np.abs(by_a) ** 2 + np.abs(bz_a) ** 2
    )

    on_axis_mask = np.abs(r_axis) <= on_axis_rmax
    if not np.any(on_axis_mask):
        on_axis_mask = np.abs(r_axis) == np.min(np.abs(r_axis))
    on_axis_envelope = np.mean(envelope_energy_density[on_axis_mask, :], axis=0)
    z_peak_idx = int(np.argmax(on_axis_envelope))
    z_peak = float(z_axis[z_peak_idx])

    r_peak_idx = int(np.argmin(np.abs(r_axis)))
    impedance = float(np.sqrt(mu_0 / epsilon_0))
    e_sqr_peak = float(e_envelope_sqr[r_peak_idx, z_peak_idx])
    intensity_w_m2 = e_sqr_peak / (2.0 * impedance)
    a0 = _intensity_to_a0(intensity_w_m2, laser_wavelength)

    try:
        time = float(info.time)
    except AttributeError:
        time = None

    summary: dict[str, Any] = {
        "iteration": int(iteration),
        "time_s": time,
        "z_peak_m": z_peak,
        "energy_j": float(total_energy_j),
        "intensity_w_cm2": float(intensity_w_m2 * 1e-4),
        "a0": float(a0),
    }
    fields: dict[str, Any] = {
        "laser_energy_density": laser_energy_density,
        "envelope_energy_density": envelope_energy_density,
        "z_axis": z_axis,
        "r_axis": r_axis,
        "z_peak_idx": z_peak_idx,
        "r_peak_idx": r_peak_idx,
    }
    return {"summary": summary, "fields": fields}


def analyze_laser_evolution(
    series_path: Union[Path, str],
    *,
    laser_wavelength: float,
    band_half_width_frac: float = 0.5,
    rmax_window: Optional[float] = None,
    on_axis_rmax: float = 5e-6,
    energy_threshold: float = 0.0,
    start_iteration: Optional[int] = None,
    end_iteration: Optional[int] = None,
    results_file: Union[Path, str, None] = None,
    on_iteration: Optional[Callable[[dict[str, Any]], None]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Analyses every iteration in an OpenPMD time series.

    Args:
        series_path: Path to the OpenPMD time series (or its parent
            directory; an ``hdf5/`` subdirectory is tried automatically).
        laser_wavelength: Central laser wavelength in metres.
        band_half_width_frac: Half-width of the laser-centred bandpass as
            a fraction of the laser wavenumber.
        rmax_window: Radial limit (m) for the energy integration.
            ``None`` uses the full radial extent.
        on_axis_rmax: Half-width (m) of the near-axis region used to find
            the longitudinal peak of the laser envelope.
        energy_threshold: Iterations whose computed energy falls below
            this value (J) are dropped from the output. They typically
            correspond to the laser not yet being in the box.
        start_iteration: Smallest iteration number to include
            (inclusive). ``None`` means no lower bound.
        end_iteration: Largest iteration number to include (inclusive).
            ``None`` means no upper bound.
        results_file: If given, the returned dict is also written to
            this path as JSON.
        on_iteration: Optional callback invoked once per kept iteration
            with the full :func:`analyze_iteration` record (including
            the field arrays). The summary entry is already appended to
            the returned dict by the time the callback fires.
        verbose: If ``True``, print per-iteration progress to stdout.

    Returns:
        A dict of the form::

            {
                "metadata": { ...analysis parameters... },
                "iterations": [ ...per-iteration summary dicts... ],
            }

        which is exactly what gets written to ``results_file`` when one
        is provided.
    """
    series = _open_series(series_path)
    iterations: list[int] = list(series.iterations)
    if start_iteration is not None:
        iterations = [it for it in iterations if it >= start_iteration]
    if end_iteration is not None:
        iterations = [it for it in iterations if it <= end_iteration]

    laser_k = 2.0 * pi / laser_wavelength
    metadata: dict[str, Any] = {
        "series_path": str(series_path),
        "laser_wavelength_m": float(laser_wavelength),
        "band_half_width_frac": float(band_half_width_frac),
        "band_k0_rad_per_m": float(laser_k),
        "band_half_width_rad_per_m": float(laser_k * band_half_width_frac),
        "rmax_window_m": None if rmax_window is None else float(rmax_window),
        "on_axis_rmax_m": float(on_axis_rmax),
        "energy_threshold_j": float(energy_threshold),
    }

    if verbose:
        print(f"Processing {len(iterations)} iterations from {series_path}")
        print(
            f"Laser wavelength = {laser_wavelength * 1e9:.1f} nm | "
            f"k_laser = {laser_k:.3e} rad/m | "
            f"band = [{(1 - band_half_width_frac) * laser_k:.3e}, "
            f"{(1 + band_half_width_frac) * laser_k:.3e}] rad/m"
        )
        print("-" * 72)

    summaries: list[dict[str, Any]] = []
    for it in iterations:
        try:
            record = analyze_laser_iteration(
                series,
                int(it),
                laser_wavelength=laser_wavelength,
                band_half_width_frac=band_half_width_frac,
                rmax_window=rmax_window,
                on_axis_rmax=on_axis_rmax,
            )
        except Exception as exc:
            if verbose:
                print(f"Iteration {it:6d}: error - {exc}")
            continue

        summary = record["summary"]
        if summary["energy_j"] < energy_threshold:
            if verbose:
                print(
                    f"Iteration {it:6d}: E = {summary['energy_j']*1e3:8.4f} mJ "
                    f"below threshold ({energy_threshold*1e3:.4f} mJ), skipping."
                )
            continue

        summaries.append(summary)
        if verbose:
            print(
                f"Iteration {it:6d}: "
                f"z_peak = {summary['z_peak_m'] * 1e3:8.3f} mm | "
                f"E = {summary['energy_j'] * 1e3:7.2f} mJ | "
                f"a0 = {summary['a0']:6.3f}"
            )

        if on_iteration is not None:
            on_iteration(record)

    if verbose:
        print("-" * 72)
        print(f"Kept {len(summaries)} / {len(iterations)} iterations after filtering.")

    output: dict[str, Any] = {"metadata": metadata, "iterations": summaries}

    if results_file is not None and summaries:
        results_path = Path(results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"Wrote per-iteration results to: {results_path}")

    return output
