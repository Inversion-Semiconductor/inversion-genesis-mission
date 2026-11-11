"""Plot energy-slice virtual vacuum waists from an openPMD particle dump.

The diagnostic computes charge-weighted transverse Twiss parameters for each
uniform-``uz`` slice.  In a drift, the slice waist is at ``alpha / gamma``
relative to the diagnostic plane, using the usual Twiss sign convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

from inversion_fbpic.utils import analysis
from inversion_fbpic.utils.distributions import load_openpmd_particles


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot virtual vacuum waist location and charge by energy slice."
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Final openPMD data########.h5 particle diagnostic.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        required=True,
        help="Electron species to combine.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--phase-space-output",
        type=Path,
        help="Output PNG path for the central-slice transverse phase space.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of uniform-uz slices (default: 60).",
    )
    parser.add_argument(
        "--uz-min",
        type=float,
        default=20.0,
        help="Minimum normalized longitudinal momentum (default: 20).",
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=100,
        help="Minimum macro-particles per slice (default: 100).",
    )
    return parser.parse_args()


def load_species(
    h5_path: Path, species: list[str], uz_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load and combine selected openPMD electron species above ``uz_min``."""
    particle_sets: list[np.ndarray] = []
    weight_sets: list[np.ndarray] = []
    for species_name in species:
        particles, weights = load_openpmd_particles(h5_path, species_name)
        mask = particles[:, 5] >= uz_min
        particle_sets.append(particles[mask])
        weight_sets.append(weights[mask])

    particles = np.concatenate(particle_sets)
    weights = np.concatenate(weight_sets)
    if len(particles) == 0:
        raise ValueError(f"No particles remain after uz >= {uz_min:g}")
    return particles, weights


def select_central_charge_slice(
    uz: np.ndarray, weights: np.ndarray, charge_fraction: float = 0.05
) -> tuple[np.ndarray, float, float]:
    """Select the symmetric mean-uz interval containing ``charge_fraction``."""
    if not 0 < charge_fraction <= 1:
        raise ValueError("charge_fraction must be in (0, 1]")
    mean_uz = np.average(uz, weights=weights)
    distance = np.abs(uz - mean_uz)
    order = np.argsort(distance)
    cumulative_fraction = np.cumsum(weights[order]) / weights.sum()
    half_width = distance[order[np.searchsorted(cumulative_fraction, charge_fraction)]]
    return distance <= half_width, mean_uz, half_width


def calculate_slice_waists(
    particles: np.ndarray,
    weights: np.ndarray,
    bins: int,
    min_particles: int,
) -> dict[str, np.ndarray]:
    """Return charge-weighted energy, charge, and drift-waist data per uz slice."""
    if bins < 1:
        raise ValueError("bins must be positive")
    if min_particles < 3:
        raise ValueError("min_particles must be at least 3")

    x, ux, y, uy, _, uz = particles.T
    gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
    energy_mev = (gamma - 1.0) * 0.51099895
    edges = np.linspace(uz.min(), uz.max(), bins + 1)

    slice_energy: list[float] = []
    slice_charge: list[float] = []
    waist_x: list[float] = []
    waist_y: list[float] = []
    for index in range(bins):
        upper_op = (
            uz <= edges[index + 1] if index == bins - 1 else uz < edges[index + 1]
        )
        mask = (uz >= edges[index]) & upper_op
        if np.count_nonzero(mask) < min_particles:
            continue

        twiss = analysis.calculate_twiss_parameters(
            x[mask], y[mask], ux[mask], uy[mask], uz[mask], weights[mask]
        )
        if not all(
            np.isfinite(twiss[plane][parameter])
            for plane in ("x", "y")
            for parameter in ("alpha", "gamma")
        ):
            continue
        if twiss["x"]["gamma"] <= 0 or twiss["y"]["gamma"] <= 0:
            continue

        slice_energy.append(np.average(energy_mev[mask], weights=weights[mask]))
        slice_charge.append(np.sum(weights[mask]) * e * 1.0e12)
        waist_x.append(twiss["x"]["alpha"] / twiss["x"]["gamma"] * 1.0e3)
        waist_y.append(twiss["y"]["alpha"] / twiss["y"]["gamma"] * 1.0e3)

    if not slice_energy:
        raise ValueError(
            "No energy slices contained enough particles for Twiss analysis"
        )
    return {
        "energy_mev": np.asarray(slice_energy),
        "charge_pc": np.asarray(slice_charge),
        "waist_x_mm": np.asarray(waist_x),
        "waist_y_mm": np.asarray(waist_y),
        "mean_energy_mev": np.average(energy_mev, weights=weights),
    }


def plot_slice_waists(slice_data: dict[str, np.ndarray], output: Path) -> None:
    """Write the energy-slice virtual waist and charge plot."""
    fig, waist_axis = plt.subplots(figsize=(10, 6))
    waist_axis.axhline(0.0, color="0.5", linewidth=1.0)
    x_waist_line = waist_axis.plot(
        slice_data["energy_mev"],
        slice_data["waist_x_mm"],
        "o-",
        color="tab:blue",
        label="x waist",
    )[0]
    y_waist_line = waist_axis.plot(
        slice_data["energy_mev"],
        slice_data["waist_y_mm"],
        "s-",
        color="tab:orange",
        label="y waist",
    )[0]
    mean_energy_line = waist_axis.axvline(
        slice_data["mean_energy_mev"],
        color="black",
        linestyle="--",
        label="charge-weighted mean energy",
    )
    waist_axis.set_xlabel("Kinetic energy (MeV)")
    waist_axis.set_ylabel("Virtual waist relative to diagnostic plane (mm)")
    waist_axis.grid(True, alpha=0.3)

    charge_axis = waist_axis.twinx()
    charge_bars = charge_axis.bar(
        slice_data["energy_mev"],
        slice_data["charge_pc"],
        width=np.diff(
            np.sort(slice_data["energy_mev"]), prepend=slice_data["energy_mev"][0]
        ).mean()
        * 0.8,
        alpha=0.25,
        color="tab:green",
        label="slice charge",
    )
    charge_axis.set_ylabel("Slice charge (pC)")

    waist_axis.legend(
        [x_waist_line, y_waist_line, mean_energy_line, charge_bars],
        ["x waist", "y waist", "charge-weighted mean energy", "slice charge"],
        loc="best",
    )
    fig.suptitle("Energy-slice virtual vacuum waists")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _scatter_sample(
    mask: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a random subset of a boolean mask for legible scatter plots."""
    indices = np.flatnonzero(mask)
    if len(indices) <= maximum:
        return indices
    return rng.choice(indices, size=maximum, replace=False)


def plot_central_slice_phase_space(
    particles: np.ndarray,
    weights: np.ndarray,
    output: Path,
) -> dict[str, float]:
    """Plot transverse phase space for the central 5%-charge mean-uz slice."""
    x, ux, y, uy, _, uz = particles.T
    selected, mean_uz, half_width = select_central_charge_slice(uz, weights)
    lower_energy = uz < mean_uz - half_width
    higher_energy = uz > mean_uz + half_width
    rng = np.random.default_rng(728)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    phase_spaces = (
        (axes[0], x * 1.0e6, ux / uz, "x", "$x' = u_x / u_z$"),
        (axes[1], y * 1.0e6, uy / uz, "y", "$y' = u_y / u_z$"),
    )
    for axis, position, momentum, position_label, momentum_label in phase_spaces:
        axis.scatter(
            position[_scatter_sample(lower_energy, 50000, rng)],
            momentum[_scatter_sample(lower_energy, 50000, rng)],
            s=2,
            color="lightcoral",
            alpha=0.12,
            linewidths=0,
            label="lower energy",
        )
        axis.scatter(
            position[_scatter_sample(higher_energy, 50000, rng)],
            momentum[_scatter_sample(higher_energy, 50000, rng)],
            s=2,
            color="lightskyblue",
            alpha=0.12,
            linewidths=0,
            label="higher energy",
        )
        axis.scatter(
            position[selected],
            momentum[selected],
            s=3,
            color="tab:green",
            alpha=0.8,
            linewidths=0,
            label="central 5% charge slice",
        )
        axis.set_xlabel(f"{position_label} (um)")
        axis.set_ylabel(momentum_label)
        axis.grid(True, alpha=0.25)

    axes[0].set_title("$x'$ vs x")
    axes[1].set_title("$y'$ vs y")
    axes[1].legend(loc="best", markerscale=3)
    selected_charge_fraction = weights[selected].sum() / weights.sum()
    fig.suptitle(
        "Mean-$u_z$ central slice: "
        f"$u_z = {mean_uz:.2f} \\pm {half_width:.2f}$ "
        f"({selected_charge_fraction:.1%} of charge)"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "mean_uz": float(mean_uz),
        "half_width": float(half_width),
        "charge_fraction": float(selected_charge_fraction),
    }


def main() -> None:
    """Load, analyze, and plot one final particle diagnostic."""
    args = parse_args()
    particles, weights = load_species(args.h5_path, args.species, args.uz_min)
    slice_data = calculate_slice_waists(
        particles, weights, args.bins, args.min_particles
    )
    plot_slice_waists(slice_data, args.output)
    phase_space_output = args.phase_space_output or args.output.with_name(
        f"{args.output.stem}_phase_space.png"
    )
    central_slice = plot_central_slice_phase_space(
        particles, weights, phase_space_output
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {phase_space_output}")
    print(f"Charge-weighted mean energy: {slice_data['mean_energy_mev']:.3f} MeV")
    print(
        "Central slice: "
        f"uz = {central_slice['mean_uz']:.3f} +/- {central_slice['half_width']:.3f}; "
        f"charge fraction = {central_slice['charge_fraction']:.3%}"
    )


if __name__ == "__main__":
    main()
