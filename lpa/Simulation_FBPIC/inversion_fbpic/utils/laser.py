"""
laser.py

Module containing useful utilities for modeling laser properties.  Contains the following
- Class for loading HTU longitudinal laser profile from FROG data through LASY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from lasy.profiles.longitudinal.longitudinal_profile_from_data import (
    LongitudinalProfileFromData,
)


class HTULasyLaser:
    """Utility wrapper around LASY laser profiles for HTU data.

    The class reads data exported from a FROG (frequency-resolved optical gating)
    measurement and prepares it for consumption by
    `LongitudinalProfileFromData`. In addition, several helpers are provided to
    visualize and analyze the temporal laser profile.

    Attributes:
        data_file: Absolute path to the FROG export containing the six-column
            data (wavelength [nm], spectral amplitude, spectral phase, time
            [fs], temporal amplitude, temporal phase).
        domain: Representation domain to expose via LASY. Must be either
            `temporal` or `spectral`.
        lo: Minimum time (seconds) covered by the data after conversion.
        hi: Maximum time (seconds) covered by the data after conversion.
        laser_profile: LASY longitudinal profile instance constructed from the
            data file.
    """

    def __init__(self, data_file: Path, domain: str = "temporal") -> None:
        """Initialize the HTU LASY laser profile.

        Args:
            data_file: Path to the tab-separated FROG export file.
            domain: Representation domain used by LASY, either `temporal` or
                `spectral`.  Note: 'spectral' has issues with plotting

        Raises:
            ValueError: If `domain` is not one of the supported options.
        """
        self.data_file: Path = data_file
        self.domain = domain
        self.lo: float = 0.0
        self.hi: float = 0.0

        data = self.load_frog_as_lasy_data()
        self.laser_profile: LongitudinalProfileFromData = LongitudinalProfileFromData(
            data, self.lo, self.hi
        )

    def get_laser_profile(self) -> LongitudinalProfileFromData:
        """Return the LASY longitudinal profile instance."""
        return self.laser_profile

    def load_frog_as_lasy_data(self) -> Mapping[str, Union[np.ndarray, float, bool]]:
        """Load the FROG export and format it for LASY consumption.

        Returns:
            Mapping with the keys expected by `LongitudinalProfileFromData`.

        Raises:
            ValueError: If fewer than two temporal samples are present when
            `domain` is `spectral` or if the domain selection is invalid.
        """
        raw = np.loadtxt(self.data_file, skiprows=1)

        wavelength_nm = raw[:, 0]
        spectral_amp = raw[:, 1]
        spectral_phase = raw[:, 2]
        time_fs = raw[:, 3]
        temporal_amp = raw[:, 4]
        temporal_phase = raw[:, 5]

        if self.domain == "spectral":
            spectral_mask = wavelength_nm != 0
            wavelength_nm = wavelength_nm[spectral_mask]
            spectral_amp = spectral_amp[spectral_mask]
            spectral_phase = spectral_phase[spectral_mask]
            time_fs = time_fs[spectral_mask]

        wavelength_m = wavelength_nm * 1e-9
        time_s = time_fs * 1e-15

        # Use weighted average for central wavelength in temporal dict
        central_wavelength = np.average(
            wavelength_m, weights=np.maximum(spectral_amp, 1e-30)
        )

        if self.domain == "spectral":
            # Use temporal spacing to set requested dt for the FFT
            if len(time_s) <= 1:
                msg = "Need at least two time samples to infer dt for spectral data."
                raise ValueError(msg)

            dt = float(np.abs(np.mean(np.diff(time_s))))

            data: dict[str, Union[np.ndarray, float, bool]] = {
                "datatype": "spectral",
                "axis_is_wavelength": True,
                "axis": wavelength_m,
                "intensity": spectral_amp,
                "phase": spectral_phase,
                "dt": dt,
            }

        elif self.domain == "temporal":
            data = {
                "datatype": "temporal",
                "axis": time_s,
                "intensity": temporal_amp,
                "phase": temporal_phase,
                "wavelength": central_wavelength,
            }
        else:
            msg = "domain must be 'spectral' or 'temporal'"
            raise ValueError(msg)

        self.lo = float(time_s.min())
        self.hi = float(time_s.max())

        return data

    def get_time_axis(self, num: int = 2000) -> np.ndarray:
        """Return a uniformly spaced temporal axis covering the raw data range.

        Args:
            num: Number of points used to represent the time axis.

        Returns:
            Array containing the time steps in seconds.
        """
        return np.linspace(self.lo, self.hi, num)

    def get_envelope(self, num: int = 2000) -> np.ndarray:
        """Evaluate the complex laser envelope on a sampled grid.

        Args:
            num: Number of points used to evaluate the envelope.

        Returns:
            Complex-valued array representing the laser envelope.
        """
        time_axis = self.get_time_axis(num=num)
        return self.laser_profile.evaluate(time_axis)

    def get_intensity_profile(self, num: int = 2000) -> np.ndarray:
        """Compute the intensity profile from the laser envelope.

        Args:
            num: Number of points used to evaluate the profile.

        Returns:
            Real-valued array containing the intensity at each sampled point.
        """
        envelope = self.get_envelope(num=num)
        return np.abs(envelope) ** 2

    def get_phase_profile(self, num: int = 2000) -> np.ndarray:
        """Compute the phase profile from the laser envelope.

        Args:
            num: Number of points used to evaluate the profile.

        Returns:
            Real-valued array containing the unwrapped phase in radians.
        """
        envelope = self.get_envelope(num=num)
        return np.unwrap(np.angle(envelope))

    def calculate_fwhm_intensity(self, num: int = 2000) -> float:
        """Estimate the full width at half maximum (FWHM) of the intensity.

        Args:
            num: Number of points used to evaluate the profile.

        Returns:
            Estimated FWHM value in femtoseconds.
        """
        t_fs = self.get_time_axis(num=num) * 1e15
        intensity = self.get_intensity_profile(num=num)

        hist, bin_edges = np.histogram(t_fs, bins=num, weights=intensity)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_idx = int(np.argmax(hist))
        peak_value = float(hist[peak_idx])
        half_max = peak_value / 2
        left_idx = peak_idx
        right_idx = peak_idx
        while left_idx > 0 and hist[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
            right_idx += 1
        fwhm = bin_centers[right_idx] - bin_centers[left_idx]

        return float(fwhm)

    def plot_temporal_profile(
        self,
        title: Optional[str] = None,
        num: int = 2000,
        show: bool = True,
    ) -> plt.Figure:
        """Plot the temporal intensity and phase profiles.

        Args:
            title: Optional title for the intensity subplot.
            num: Number of points used to evaluate the profiles.
            show: Whether to call `plt.show()` before returning.

        Returns:
            Matplotlib figure instance containing the generated subplots.
        """
        t_fs = self.get_time_axis(num=num) * 1e15
        intensity = self.get_intensity_profile(num=num)
        phase = self.get_phase_profile(num=num)

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t_fs, intensity, label="Imported via LASY")
        ax[0].set_ylabel("Normalized Intensity")
        if title is not None:
            ax[0].set_title(title)
        ax[0].legend()

        ax[1].plot(t_fs, phase)
        ax[1].set_ylabel("Phase [rad]")
        ax[1].set_xlabel("Time [fs]")

        plt.tight_layout()
        if show:
            plt.show()

        return fig
