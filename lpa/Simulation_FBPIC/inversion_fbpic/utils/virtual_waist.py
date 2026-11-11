"""Virtual-waist beam transport and campaign-scalar recalculation helpers."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from inversion_fbpic.utils import analysis
from inversion_fbpic.utils.distributions import (
    compute_moment_descriptor,
    crop_central_particles,
    load_openpmd_particles,
    mean_kinetic_energy_mev,
)


def select_central_charge_slice(
    uz: np.ndarray, weights: np.ndarray, charge_fraction: float = 0.05
) -> tuple[np.ndarray, float, float]:
    """Select the symmetric mean-``uz`` interval containing a charge fraction."""
    if not 0 < charge_fraction <= 1:
        raise ValueError("charge_fraction must be in (0, 1]")
    mean_uz = np.average(uz, weights=weights)
    distance = np.abs(uz - mean_uz)
    order = np.argsort(distance)
    cumulative_fraction = np.cumsum(weights[order]) / weights.sum()
    half_width = distance[order[np.searchsorted(cumulative_fraction, charge_fraction)]]
    return distance <= half_width, float(mean_uz), float(half_width)


def find_final_particle_dump(simulation_directory: str | Path) -> Path:
    """Return the numerically final ``data########.h5`` particle diagnostic."""
    simulation_directory = Path(simulation_directory)
    dumps = []
    for path in (simulation_directory / "lab_diags" / "hdf5").glob("data*.h5"):
        match = re.fullmatch(r"data(\d+)\.h5", path.name)
        if match is not None:
            dumps.append((int(match.group(1)), path))
    if not dumps:
        raise FileNotFoundError(
            f"No openPMD data########.h5 diagnostics in {simulation_directory}"
        )
    return max(dumps, key=lambda item: item[0])[1]


def load_selected_species(
    h5_path: str | Path, species: tuple[str, ...], uz_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load, combine, and apply the campaign's minimum-``uz`` selection."""
    if not species:
        raise ValueError("at least one electron species is required")
    particle_sets, weight_sets = [], []
    for species_name in species:
        particles, weights = load_openpmd_particles(h5_path, species_name)
        selected = particles[:, 5] >= uz_min
        particle_sets.append(particles[selected])
        weight_sets.append(weights[selected])
    particles = np.concatenate(particle_sets)
    weights = np.concatenate(weight_sets)
    if len(particles) < 2:
        raise ValueError(f"fewer than two particles remain after uz >= {uz_min:g}")
    return particles, weights


def central_slice_waist(
    particles: np.ndarray, weights: np.ndarray, charge_fraction: float = 0.05
) -> dict[str, float]:
    """Calculate virtual x/y waists and their average from the central slice."""
    x, ux, y, uy, _, uz = particles.T
    selected, mean_uz, half_width = select_central_charge_slice(
        uz, weights, charge_fraction
    )
    twiss = analysis.calculate_twiss_parameters(
        x[selected],
        y[selected],
        ux[selected],
        uy[selected],
        uz[selected],
        weights[selected],
    )
    waist_x_m = twiss["x"]["alpha"] / twiss["x"]["gamma"]
    waist_y_m = twiss["y"]["alpha"] / twiss["y"]["gamma"]
    return {
        "mean_uz": mean_uz,
        "half_width_uz": half_width,
        "charge_fraction": float(weights[selected].sum() / weights.sum()),
        "waist_x_m": float(waist_x_m),
        "waist_y_m": float(waist_y_m),
        "waist_average_m": float((waist_x_m + waist_y_m) / 2.0),
    }


def drift_to_waist_relative_plane(
    particles: np.ndarray, waist_average_m: float, target_z_m: float
) -> tuple[np.ndarray, float]:
    """Drift transverse coordinates to a distance downstream of virtual waist."""
    if np.any(particles[:, 5] == 0):
        raise ValueError("Cannot drift particles with uz = 0")
    drift_distance_m = target_z_m - waist_average_m
    drifted = particles.copy()
    drifted[:, 0] += drift_distance_m * drifted[:, 1] / drifted[:, 5]
    drifted[:, 2] += drift_distance_m * drifted[:, 3] / drifted[:, 5]
    return drifted, float(drift_distance_m)


def calculate_campaign_features(
    particles: np.ndarray, weights: np.ndarray, central_fraction: float
) -> dict[str, float]:
    """Apply the campaign crop and calculate its canonical 34 scalar features."""
    cropped_particles, cropped_weights = crop_central_particles(
        particles, weights, central_fraction=central_fraction
    )
    features = compute_moment_descriptor(cropped_particles, cropped_weights)
    features["mean_kinetic_energy_MeV"] = mean_kinetic_energy_mev(
        cropped_particles, cropped_weights
    )
    return features


def calculate_shifted_features_for_run(
    simulation_directory: str | Path,
    *,
    species: tuple[str, ...],
    target_z_m: float = 5.0e-3,
    uz_min: float = 20.0,
    central_fraction: float = 0.95,
    charge_fraction: float = 0.05,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Calculate shifted campaign features without modifying any result record."""
    simulation_directory = Path(simulation_directory)
    particles, weights = load_selected_species(
        find_final_particle_dump(simulation_directory), species, uz_min
    )
    waist = central_slice_waist(particles, weights, charge_fraction)
    drifted, drift_distance_m = drift_to_waist_relative_plane(
        particles, waist["waist_average_m"], target_z_m
    )
    features = calculate_campaign_features(drifted, weights, central_fraction)
    metadata = {
        "method": "central_charge_mean_uz_slice",
        "target_z_relative_to_virtual_waist_m": target_z_m,
        "drift_distance_from_final_diagnostic_m": drift_distance_m,
        "uz_min": uz_min,
        "central_fraction": central_fraction,
        "requested_charge_fraction": charge_fraction,
        "species": list(species),
        **waist,
    }
    return features, metadata


def write_shifted_result_record(
    base_record: dict[str, Any],
    shifted_features: dict[str, float],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a result record retaining base metadata with shifted features."""
    shifted = deepcopy(base_record)
    shifted["result"]["features"] = shifted_features
    shifted["shifted_beam_postprocess"] = metadata
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(shifted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def shift_run_result(
    simulation_directory: str | Path,
    *,
    species: tuple[str, ...],
    target_z_m: float = 5.0e-3,
    uz_min: float = 20.0,
    central_fraction: float = 0.95,
    charge_fraction: float = 0.05,
    output_filename: str = "results_shifted.json",
) -> tuple[dict[str, float], dict[str, Any], Path]:
    """Recalculate one run's features at a virtual-waist-relative drift plane.

    The original ``result.json`` is read only.  A separate shifted result record
    preserves its manifest and status while replacing only scalar features.
    """
    simulation_directory = Path(simulation_directory)
    base_result_path = simulation_directory / "result.json"
    if not base_result_path.is_file():
        raise FileNotFoundError(f"Missing {base_result_path}")
    payload = json.loads(base_result_path.read_text(encoding="utf-8"))
    if payload["result"]["status"] != "ok":
        raise ValueError("only successful result.json records can be shifted")

    shifted_features, metadata = calculate_shifted_features_for_run(
        simulation_directory,
        species=species,
        target_z_m=target_z_m,
        uz_min=uz_min,
        central_fraction=central_fraction,
        charge_fraction=charge_fraction,
    )
    output_path = simulation_directory / output_filename
    write_shifted_result_record(payload, shifted_features, metadata, output_path)
    return shifted_features, metadata, output_path
