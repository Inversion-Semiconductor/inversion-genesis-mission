"""LASY laser pulse implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import attrs
import numpy as np
from scipy.constants import c

from inversion_fbpic.lib.laser import _gaussian_r_extent, _LaserPulse

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from fbpic.lpa_utils.laser.laser_profiles import LaserProfile


def _zernike_names() -> tuple[str, ...]:
    """Canonical Zernike coefficient names accepted by ``HighOrderLasyLaser``."""
    # Lazy import: ``inversion_fbpic.utils.laser`` pulls in lasy.
    from inversion_fbpic.utils.laser import ZERNIKE_OSA_INDICES

    return tuple(ZERNIKE_OSA_INDICES)


def _default_zernike_coefficients() -> dict[str, float]:
    return {name: 0.0 for name in _zernike_names()}


def _normalize_zernike_coefficients(value: Any) -> dict[str, float]:
    """Validate Zernike names and return a complete, canonically ordered dict."""
    if value is None:
        return _default_zernike_coefficients()
    if not isinstance(value, Mapping):
        raise ValueError("zernike_coefficients must be a mapping of name -> amplitude.")
    names = _zernike_names()
    unknown = set(value) - set(names)
    if unknown:
        raise ValueError(
            f"Unknown zernike_coefficients keys: {sorted(unknown)}. "
            f"Allowed: {list(names)}"
        )
    return {name: float(value.get(name, 0.0)) for name in names}


def _as_pair(value: Any, name: str) -> tuple[Any, Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence of two numbers.")
    seq = list(value)
    if len(seq) != 2:
        raise ValueError(f"{name} must have exactly two entries, got {len(seq)}.")
    return seq[0], seq[1]


def _to_float_pair(value: Any) -> tuple[float, float]:
    """Convert a two-element sequence to a tuple of floats (real Jones vector)."""
    first, second = _as_pair(value, "polarization")
    return (float(first), float(second))


def _to_int_pair(value: Any) -> tuple[int, int]:
    """Convert a two-element sequence to a tuple of ints, each at least 2."""
    first, second = _as_pair(value, "num_points")
    pair = (int(first), int(second))
    if min(pair) < 2:
        raise ValueError("num_points entries must be at least 2.")
    return pair


def _format_jones(pol: tuple[float, float]) -> str:
    """Return a human-readable label for a real Jones vector."""
    return f"Jones ({pol[0]:g}, {pol[1]:g})"


def _resolve_comm(comm: Any | None) -> tuple[int, Any | None]:
    """Return ``(rank, mpi_comm)`` for the communicator forms ``prepare`` accepts.

    ``comm`` may be ``None`` (serial, rank 0), FBPIC's ``BoundaryCommunicator``
    (exposes ``rank`` and ``mpi_comm``, the latter ``None`` without MPI), or an
    mpi4py communicator such as ``MPI.COMM_WORLD`` (exposes ``Get_rank``).
    """
    if comm is None:
        return 0, None
    if hasattr(comm, "mpi_comm"):
        return int(comm.rank), comm.mpi_comm
    if hasattr(comm, "Get_rank"):
        return int(comm.Get_rank()), comm
    raise TypeError(
        "comm must be None, FBPIC's sim.comm, or an mpi4py communicator; "
        f"got {type(comm).__name__}."
    )


@attrs.define(kw_only=True, slots=False, frozen=True)
class LasyLaserPulse(_LaserPulse):
    """
    Super-Gaussian laser pulse with Zernike aberrations, built with LASY.

    Wraps ``inversion_fbpic.utils.laser.HighOrderLasyLaser``: the pulse is
    constructed at focus, back-propagated to the simulation start plane,
    optionally re-centred, normalized to ``energy``, and written to a LASY HDF5
    file that FBPIC reads through ``FromLasyFileLaser``. The build is expensive
    and happens once, in ``prepare()``, on MPI rank 0 only; other ranks wait at
    a barrier and receive the file path.

    Only ``energy`` may be provided. ``a0`` is measured numerically from the
    field at focus during ``prepare()`` and is reported as ``out_a0``
    (``null`` in YAML written before the build).

    LASY pulses can only be emitted through an antenna, so ``method`` is fixed
    to ``"antenna"``, ``v_antenna`` to ``0.0``, and ``z0_antenna`` is required.
    FBPIC resets the LASY time axis to zero, so the peak intensity leaves the
    antenna at ``t = t_start + 3 * tau_fwhm``. ``z0`` is informational only
    (nominal centroid at ``t = 0``, used for plotting extents); for a
    consistent picture set ``z0 = z0_antenna - c * (t_start + 3 * tau_fwhm)``.

    Args:
        wavelength: (float) [m] Central wavelength of the laser pulse in meters.
        tau_fwhm: (float) [s] Full-width at half-maximum intensity duration of the laser pulse in seconds.
        waist: (float) [m] Super-Gaussian spot size (1/e^2 radius for order 2) at focus in meters.
        focal_position: (float) [m] Focal position of the laser pulse in meters, relative to the simulation start plane.
        super_gaussian_order: (float) Super-Gaussian order of the transverse profile. 2.0 is Gaussian.
        zernike_coefficients: (dict[str, float]) [wavelengths] |OPTIONAL| Zernike phase amplitudes at focus keyed by name (astigmatism_2, astigmatism_4, coma_y, coma_x, trefoil_y, trefoil_x, spherical_3, astigmatism_6, coma_5_y, coma_5_x, secondary_trefoil_y, secondary_trefoil_x). Missing names default to 0.0; unknown names are rejected.
        polarization: (tuple[float, float]) |OPTIONAL| Real Jones vector (Ex, Ey) passed to LASY. Defaults to (1, 0), linear along x.
        n_azimuthal_modes: (int) |OPTIONAL| Number of azimuthal modes in the LASY r-t grid. Defaults to 5.
        num_points: (tuple[int, int]) |OPTIONAL| LASY grid points (radial, temporal). Defaults to (600, 900).
        hi_range: (float) [waists] |OPTIONAL| Radial extent of the LASY grid in units of `waist`. Defaults to 8.0.
        center_and_remove_tilt: (bool) |OPTIONAL| Re-centre the fluence and remove the mean transverse phase gradient at the start plane. Defaults to True.
        centering_angles: (int) |OPTIONAL| Number of polar angles used for centering; must be at least 2 * n_azimuthal_modes - 1. Defaults to 72.
        lasy_file: (Path|str) |OPTIONAL| Output prefix for the LASY HDF5 file; the written file is `<parent>/<stem>_00000.h5`. If not absolute, this is relative to the `working_directory` passed to `Simulation.setup_simulation()` (or to `prepare(relative_to=...)`), falling back to the current working directory. Defaults to `diags/lasy_laser`.
        t_start: (float) [s] |OPTIONAL| Delay before the antenna starts emitting the LASY file, as in FBPIC's `FromLasyFileLaser`. Defaults to 0.0.
    """

    SUBCLASS: ClassVar[str] = "lasy"

    wavelength: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    tau_fwhm: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    waist: float = attrs.field(converter=float, validator=attrs.validators.gt(0.0))
    focal_position: float = attrs.field(converter=float)
    super_gaussian_order: float = attrs.field(
        converter=float, validator=attrs.validators.gt(0.0)
    )
    zernike_coefficients: dict[str, float] = attrs.field(
        factory=_default_zernike_coefficients,
        converter=_normalize_zernike_coefficients,
        hash=False,
    )
    polarization: tuple[float, float] = attrs.field(
        default=(1.0, 0.0), converter=_to_float_pair
    )
    n_azimuthal_modes: int = attrs.field(
        default=5, converter=int, validator=attrs.validators.ge(1)
    )
    num_points: tuple[int, int] = attrs.field(
        default=(600, 900), converter=_to_int_pair
    )
    hi_range: float = attrs.field(
        default=8.0, converter=float, validator=attrs.validators.gt(0.0)
    )
    center_and_remove_tilt: bool = attrs.field(default=True, converter=bool)
    centering_angles: int = attrs.field(
        default=72, converter=int, validator=attrs.validators.ge(1)
    )
    lasy_file: Path = attrs.field(default=Path("diags/lasy_laser"), converter=Path)
    t_start: float = attrs.field(default=0.0, converter=float)

    # Filled by prepare(). The HighOrderLasyLaser exists only on the rank that
    # built it; the written file path is known on every rank.
    _high_order_laser: Any = attrs.field(init=False, default=None, repr=False, eq=False)
    lasy_file_path: Path | None = attrs.field(
        init=False, default=None, repr=False, eq=False
    )

    def __attrs_post_init__(self) -> None:
        # Deliberately does not call the base implementation: a0 is derived
        # numerically at build time, not analytically at construction.
        if self.a0 is not None:
            raise ValueError(
                "LasyLaserPulse derives a0 numerically from energy during prepare(); "
                "do not pass a0."
            )
        if self.energy is None or self.energy <= 0:
            raise ValueError("energy must be provided and > 0.")
        object.__setattr__(self, "_amplitude_source", "energy")

        if self.method not in (None, "antenna"):
            raise ValueError(
                "LasyLaserPulse can only be emitted with method='antenna'."
            )
        object.__setattr__(self, "method", "antenna")
        if self.v_antenna is not None and self.v_antenna != 0.0:
            raise ValueError(
                "LasyLaserPulse requires a stationary antenna (v_antenna=0)."
            )
        object.__setattr__(self, "v_antenna", 0.0)
        if self.z0_antenna is None:
            raise ValueError(
                "z0_antenna is required: LASY pulses are emitted by an antenna."
            )

        minimum_angles = 2 * self.n_azimuthal_modes - 1
        if self.centering_angles < minimum_angles:
            raise ValueError(
                "centering_angles must be at least "
                f"{minimum_angles} for the configured azimuthal modes."
            )
        if np.hypot(*self.polarization) == 0.0:
            raise ValueError("polarization must be a non-zero Jones vector.")

    # ------------------------------------------------------------------
    # Mapping onto HighOrderLasyLaser
    # ------------------------------------------------------------------

    @property
    def physical_parameters(self) -> dict[str, Any]:
        """``HighOrderLasyLaser`` physical parameters built from this config."""
        parameters: dict[str, Any] = {
            "laser_wavelength_m": self.wavelength,
            "laser_energy_J": self.energy,
            "laser_pulse_duration_fwhm_s": self.tau_fwhm,
            "laser_spot_size_m": self.waist,
            "laser_super_gaussian_order": self.super_gaussian_order,
            "laser_focal_position_m": self.focal_position,
        }
        parameters.update(
            {
                f"zernike_{name}": amplitude
                for name, amplitude in self.zernike_coefficients.items()
            }
        )
        return parameters

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """``HighOrderLasyLaser`` hyperparameters built from this config."""
        return {
            "polarization": self.polarization,
            "n_azimuthal_modes": self.n_azimuthal_modes,
            "num_points": self.num_points,
            "hi_range": self.hi_range,
            "center_and_remove_tilt": self.center_and_remove_tilt,
            "centering_angles": self.centering_angles,
        }

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare()`` has produced the LASY file."""
        return self.lasy_file_path is not None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def resolve_lasy_file(self, relative_to: Path | str | None = None) -> Path:
        """Absolute output prefix for the LASY file.

        A relative ``lasy_file`` is anchored at *relative_to* when given, otherwise
        at the current working directory.
        """
        if self.lasy_file.is_absolute():
            return self.lasy_file
        base = Path(relative_to) if relative_to is not None else Path.cwd()
        return (base / self.lasy_file).resolve()

    def prepare(
        self, comm: Any | None = None, *, relative_to: Path | str | None = None
    ) -> None:
        """
        Build the LASY pulse, write its HDF5 file, and measure a0 at focus.

        Runs the expensive build on rank 0 only. With MPI, the other ranks wait
        at a barrier and then receive the written path and a0 by broadcast.
        Calling this again after a successful build is a no-op.

        Args:
            comm: (BoundaryCommunicator|mpi4py.MPI.Comm|None) Communicator for the
                rank-0 build and barrier. Accepts FBPIC's ``sim.comm``, an mpi4py
                communicator such as ``MPI.COMM_WORLD``, or ``None`` when running
                without MPI (the calling process builds the file itself).
            relative_to: (Path|str|None) Directory a relative ``lasy_file`` is written
                under. ``Simulation`` passes its ``working_directory``; ``None`` means
                the current working directory.
        """
        if self.is_prepared:
            return

        rank, mpi_comm = _resolve_comm(comm)

        payload: tuple[str, float] | None = None
        if rank == 0:
            from inversion_fbpic.utils.laser import HighOrderLasyLaser

            high_order_laser = HighOrderLasyLaser(
                self.physical_parameters, self.hyperparameters
            )
            written_path = high_order_laser.save(self.resolve_lasy_file(relative_to))
            focus_a0 = high_order_laser.compute_focus_a0()
            object.__setattr__(self, "_high_order_laser", high_order_laser)
            payload = (str(written_path.resolve()), float(focus_a0))

        if mpi_comm is not None:
            mpi_comm.barrier()
            payload = mpi_comm.bcast(payload, root=0)
        if payload is None:
            raise RuntimeError("Rank 0 did not produce the LASY laser file.")

        path_str, focus_a0 = payload
        lasy_file_path = Path(path_str)
        if not lasy_file_path.is_file():
            raise FileNotFoundError(
                f"LASY laser file was not created: {lasy_file_path}"
            )
        object.__setattr__(self, "lasy_file_path", lasy_file_path)
        object.__setattr__(self, "a0", focus_a0)
        object.__setattr__(self, "out_a0", focus_a0)

    def to_dict(self, *, include_nones: bool = True) -> dict[str, Any]:
        payload = super().to_dict(include_nones=include_nones)
        # ``lasy_file`` is an output prefix anchored at the simulation working
        # directory, not an input file, so it is written verbatim rather than
        # relative to the config file being saved.
        payload["parameters"]["lasy_file"] = self.lasy_file.as_posix()
        return payload

    def resolve_laser_energy(self) -> float:
        return float(self.energy)

    def resolve_laser_a0(self) -> float:
        if self.out_a0 is None:
            raise RuntimeError(
                "a0 is measured from the LASY field during prepare(); call prepare() first."
            )
        return float(self.out_a0)

    def build_laser_profile(self) -> LaserProfile | list[LaserProfile]:
        from fbpic.lpa_utils.laser.laser_profiles import FromLasyFileLaser

        if not self.is_prepared:
            self.prepare(None)
        return FromLasyFileLaser(str(self.lasy_file_path), t_start=self.t_start)

    # ------------------------------------------------------------------
    # Extents and plotting
    # ------------------------------------------------------------------

    def get_r_extent(
        self, simulation_extent: tuple[float, float], num_sigma: float = 3.0
    ) -> float:
        return _gaussian_r_extent(
            self.waist,
            self.wavelength,
            self.focal_position,
            simulation_extent,
            num_sigma,
        )

    def plot(
        self,
        *,
        mode: Literal["lineout", "lineout_and_2d"] = "lineout_and_2d",
        ax: "plt.Axes | None" = None,
        num: int = 600,
        output_path: Path | str | None = None,
        show: bool = False,
        label: str | None = None,
    ) -> "plt.Figure":
        """Plot the start-plane LASY envelope and (optionally) a face-on |E| map.

        Requires the LASY ``Laser`` object, so this only works on the rank that
        ran ``prepare()`` (it runs ``prepare()`` itself if needed).

        Args:
            mode: (Literal["lineout", "lineout_and_2d"]) Panel layout.
                ``"lineout"`` shows only the on-axis longitudinal envelope.
                ``"lineout_and_2d"`` (default) adds a face-on field-amplitude
                map at the start plane with the polarization direction marked.
            ax: (matplotlib.axes.Axes|None) If provided, the longitudinal envelope
                is also drawn on this external axes (for combined overlay figures).
            num: (int) Number of points for the resampled longitudinal lineout.
            output_path: (Path|str|None) If given, the figure is saved here.
            show: (bool) Whether to call ``plt.show()``.
            label: (str|None) Label for the external *ax* lineout. Defaults to
                ``SUBCLASS (polarization)``.

        Returns:
            The created matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        from scipy.constants import e as q_e, m_e as m_electron

        from inversion_fbpic.utils.laser import polar_fields, transverse_fluence

        if not self.is_prepared:
            self.prepare(None)
        if self._high_order_laser is None:
            raise RuntimeError(
                "plot() needs the LASY Laser object, which only exists on the rank "
                "that ran prepare()."
            )
        laser = self._high_order_laser.laser
        _, time = laser.grid.axes
        e_to_a0 = q_e / (m_electron * c * laser.profile.omega0)
        pol_label = _format_jones(self.polarization)

        # On-axis envelope vs. time, mapped to z about the nominal centroid z0.
        on_axis = np.abs(polar_fields(laser, np.array([0.0]))[0, 0, :]) * e_to_a0
        t_peak = time[int(np.argmax(on_axis))]
        z_of_t = self.z0 - c * (time - t_peak)
        order = np.argsort(z_of_t)
        z_arr = np.linspace(z_of_t.min(), z_of_t.max(), num)
        envelope = np.interp(z_arr, z_of_t[order], on_axis[order])

        if ax is not None:
            default_label = f"{self.SUBCLASS} ({pol_label})"
            ax.plot(
                z_arr * 1e3,
                envelope,
                lw=1.5,
                label=label if label is not None else default_label,
            )

        if mode == "lineout":
            fig, ax_z = plt.subplots(1, 1, figsize=(8, 4.5))
        else:
            fig, (ax_z, ax_xy) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax_z.plot(z_arr * 1e6, envelope, color="C0", lw=1.5)
        ax_z.set_xlabel("z (um)")
        ax_z.set_ylabel("On-axis envelope amplitude (a\u2080)")
        ax_z.set_title(f"Longitudinal envelope at start plane\n{pol_label}")
        ax_z.grid(True, alpha=0.3)

        if mode == "lineout_and_2d":
            radius, angles, fluence, peak_field = transverse_fluence(laser, 361)
            amplitude = np.abs(peak_field)
            # Explicit polar cell edges: the Cartesian mesh is not monotonic, so
            # pcolormesh cannot infer them from cell centres.
            d_theta = angles[1] - angles[0]
            theta_edges = np.append(angles - d_theta / 2.0, angles[-1] + d_theta / 2.0)
            r_edges = np.concatenate(
                ([0.0], 0.5 * (radius[1:] + radius[:-1]), [radius[-1]])
            )
            theta_grid, r_grid = np.meshgrid(theta_edges, r_edges, indexing="ij")
            x_um = r_grid * np.cos(theta_grid) * 1e6
            y_um = r_grid * np.sin(theta_grid) * 1e6
            im = ax_xy.pcolormesh(x_um, y_um, amplitude, cmap="inferno", shading="flat")
            ax_xy.set_aspect("equal")

            # Zoom to where the azimuthally averaged fluence is above 1e-3 of peak.
            radial_fluence = fluence.mean(axis=0)
            above = np.flatnonzero(radial_fluence > 1e-3 * radial_fluence.max())
            r_view = radius[int(above[-1])] * 1e6 if above.size else radius[-1] * 1e6
            ax_xy.set_xlim(-r_view, r_view)
            ax_xy.set_ylim(-r_view, r_view)

            ax_xy.set_xlabel(r"x ($\mu$m)")
            ax_xy.set_ylabel(r"y ($\mu$m)")
            ax_xy.set_title(f"Face-on |E| at start plane\n{pol_label}")
            cbar = fig.colorbar(im, ax=ax_xy, fraction=0.046, pad=0.04)
            cbar.set_label("|E| (V/m)")

            px, py = self.polarization
            norm = float(np.hypot(px, py))
            arrow = 0.35 * r_view
            ax_xy.annotate(
                "",
                xy=(px / norm * arrow, py / norm * arrow),
                xytext=(-px / norm * arrow, -py / norm * arrow),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
            )
            if self.out_a0 is not None:
                ax_xy.text(
                    0.02,
                    0.98,
                    f"a\u2080 at focus = {self.out_a0:.3g}",
                    transform=ax_xy.transAxes,
                    color="white",
                    va="top",
                    fontsize=9,
                )

        fig.suptitle(f"{self.SUBCLASS}  \u2014  {pol_label}", fontsize=11)
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
