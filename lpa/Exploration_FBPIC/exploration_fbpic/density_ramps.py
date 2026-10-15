"""Plot campaign density-ramp extremes without running FBPIC."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import Campaign


_LEFT_RAMP_PARAMETERS = ("density_fwhm_left_m", "density_left_ramp_edge_percent")


@dataclass(frozen=True)
class DensityRampCase:
    """One lower/upper left-ramp configuration and its constructed density."""

    fwhm_left_m: float
    requested_ramp_edge_percent: float
    effective_ramp_edge_percent: float
    one_percent_location_m: float
    beta_left: float
    density: Any


@dataclass(frozen=True)
class DensityRampParameterPoint:
    """One campaign coordinate mapped into FWHM and derived-beta space."""

    fwhm_left_m: float
    requested_ramp_edge_percent: float
    effective_ramp_edge_percent: float
    beta_left: float


def _load_template_class(campaign: Campaign) -> type[Any]:
    """Import the campaign template class directly from its configured path."""
    template_path = campaign.config.template_script
    specification = importlib.util.spec_from_file_location(
        f"density_ramp_template_{campaign.config.fingerprint()}", template_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not import campaign template {template_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    try:
        return getattr(module, campaign.config.template_class)
    except AttributeError as error:
        raise AttributeError(
            f"template {template_path} does not define "
            f"{campaign.config.template_class}"
        ) from error


def build_density_ramp_cases(
    campaign: Campaign, simulation_half_length: float
) -> tuple[DensityRampCase, ...]:
    """Build the four left-ramp extremes with all other inputs nominal.

    ``simulation_half_length`` is the physical distance from the left plotting
    boundary at zero to the density peak, in metres.
    """
    if simulation_half_length <= 0.0:
        raise ValueError("simulation_half_length must be positive")

    try:
        fwhm = campaign.config.parameters[_LEFT_RAMP_PARAMETERS[0]]
        ramp_edge_percent = campaign.config.parameters[_LEFT_RAMP_PARAMETERS[1]]
    except KeyError as error:
        raise ValueError(
            "campaign must define density_fwhm_left_m and "
            "density_left_ramp_edge_percent"
        ) from error
    if not fwhm.variable or not ramp_edge_percent.variable:
        raise ValueError("left-ramp parameters must each have lower and upper bounds")

    template_class = _load_template_class(campaign)
    nominal_parameters = {
        name: parameter.nominal
        for name, parameter in campaign.config.parameters.items()
    }
    cases = []
    for fwhm_left_m in (fwhm.lower, fwhm.upper):
        for ramp_edge_percent_value in (
            ramp_edge_percent.lower,
            ramp_edge_percent.upper,
        ):
            parameters = {
                **nominal_parameters,
                "density_center_location_m": simulation_half_length,
                "density_fwhm_left_m": fwhm_left_m,
                "density_left_ramp_edge_percent": ramp_edge_percent_value,
            }
            simulation = template_class(
                parameters, campaign.config.template_hyperparameters()
            ).build_simulation()
            try:
                density = next(
                    density
                    for density in simulation.densities
                    if hasattr(density, "fwhm_left") and hasattr(density, "beta_left")
                )
            except StopIteration as error:
                raise ValueError(
                    "template simulation has no density profile with left-ramp parameters"
                ) from error
            cases.append(
                DensityRampCase(
                    fwhm_left_m=float(fwhm_left_m),
                    requested_ramp_edge_percent=float(ramp_edge_percent_value),
                    effective_ramp_edge_percent=float(
                        (
                            density.fwhm_left
                            / 2.0
                            * (
                                (np.log(100.0) / np.log(2.0))
                                ** (1.0 / density.beta_left)
                                - 1.0
                            )
                        )
                        / (density.center - density.fwhm_left / 2.0)
                    ),
                    one_percent_location_m=float(
                        density.center
                        - density.fwhm_left
                        / 2.0
                        * (np.log(100.0) / np.log(2.0)) ** (1.0 / density.beta_left)
                    ),
                    beta_left=float(density.beta_left),
                    density=density,
                )
            )
    return tuple(cases)


def build_density_ramp_parameter_space(
    campaign: Campaign,
    simulation_half_length: float,
    points_per_edge: int = 5,
) -> tuple[DensityRampParameterPoint, ...]:
    """Sample the perimeter of the left-ramp campaign region in derived space."""
    if points_per_edge < 2:
        raise ValueError("points_per_edge must be at least 2")
    if simulation_half_length <= 0.0:
        raise ValueError("simulation_half_length must be positive")

    try:
        fwhm = campaign.config.parameters[_LEFT_RAMP_PARAMETERS[0]]
        ramp_edge_percent = campaign.config.parameters[_LEFT_RAMP_PARAMETERS[1]]
    except KeyError as error:
        raise ValueError(
            "campaign must define density_fwhm_left_m and "
            "density_left_ramp_edge_percent"
        ) from error
    if not fwhm.variable or not ramp_edge_percent.variable:
        raise ValueError("left-ramp parameters must each have lower and upper bounds")

    template_class = _load_template_class(campaign)
    nominal_parameters = {
        name: parameter.nominal
        for name, parameter in campaign.config.parameters.items()
    }
    fwhm_values = np.linspace(fwhm.lower, fwhm.upper, points_per_edge)
    ramp_values = (
        np.geomspace(ramp_edge_percent.lower, ramp_edge_percent.upper, points_per_edge)
        if ramp_edge_percent.log_scale
        else np.linspace(
            ramp_edge_percent.lower, ramp_edge_percent.upper, points_per_edge
        )
    )
    boundary = [
        *((fwhm_value, ramp_values[0]) for fwhm_value in fwhm_values),
        *((fwhm_values[-1], ramp_value) for ramp_value in ramp_values[1:]),
        *((fwhm_value, ramp_values[-1]) for fwhm_value in fwhm_values[-2::-1]),
        *((fwhm_values[0], ramp_value) for ramp_value in ramp_values[-2:0:-1]),
    ]
    points = []
    for fwhm_left_m, ramp_edge_percent_value in boundary:
        parameters = {
            **nominal_parameters,
            "density_center_location_m": simulation_half_length,
            "density_fwhm_left_m": fwhm_left_m,
            "density_left_ramp_edge_percent": ramp_edge_percent_value,
        }
        simulation = template_class(
            parameters, campaign.config.template_hyperparameters()
        ).build_simulation()
        density = next(
            density
            for density in simulation.densities
            if hasattr(density, "fwhm_left") and hasattr(density, "beta_left")
        )
        points.append(
            DensityRampParameterPoint(
                fwhm_left_m=float(fwhm_left_m),
                requested_ramp_edge_percent=float(ramp_edge_percent_value),
                effective_ramp_edge_percent=float(
                    (
                        density.fwhm_left
                        / 2.0
                        * (
                            (np.log(100.0) / np.log(2.0)) ** (1.0 / density.beta_left)
                            - 1.0
                        )
                    )
                    / (density.center - density.fwhm_left / 2.0)
                ),
                beta_left=float(density.beta_left),
            )
        )
    return tuple(points)


def plot_density_ramp_parameter_space(
    campaign: Campaign,
    simulation_half_length: float,
    output: str | Path,
    points_per_edge: int = 5,
    direct_beta_bounds: tuple[float, float] = (1.0, 4.0),
) -> Path:
    """Save the derived left-ramp boundary beside direct FWHM/beta bounds."""
    import matplotlib.pyplot as plt

    points = build_density_ramp_parameter_space(
        campaign, simulation_half_length, points_per_edge
    )
    fwhm_mm = np.array([point.fwhm_left_m for point in points]) * 1.0e3
    beta = np.array([point.beta_left for point in points])
    figure, axis = plt.subplots(figsize=(8, 5.5))
    boundary_fwhm_mm = np.append(fwhm_mm, fwhm_mm[0])
    boundary_beta = np.append(beta, beta[0])
    axis.fill(
        boundary_fwhm_mm,
        boundary_beta,
        color="tab:blue",
        alpha=0.12,
        label="Derived remapped region",
    )
    axis.plot(boundary_fwhm_mm, boundary_beta, color="tab:blue", linewidth=1.5)
    axis.scatter(fwhm_mm, beta, color="tab:blue", zorder=3, label="5 points per edge")

    beta_low, beta_high = direct_beta_bounds
    rectangle_x = [
        float(np.min(fwhm_mm)),
        float(np.max(fwhm_mm)),
        float(np.max(fwhm_mm)),
        float(np.min(fwhm_mm)),
        float(np.min(fwhm_mm)),
    ]
    rectangle_y = [beta_low, beta_low, beta_high, beta_high, beta_low]
    axis.plot(
        rectangle_x,
        rectangle_y,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label=r"Direct $\beta \in [1, 4]$",
    )
    axis.set_xlabel("left FWHM (mm)")
    axis.set_ylabel("derived left beta")
    axis.set_title("Left-ramp parameter-space comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_density_ramps(
    campaign: Campaign,
    simulation_half_length: float,
    output: str | Path,
    samples: int = 1000,
) -> Path:
    """Save a 2x2 plot of the normalized density across left-ramp extremes."""
    if samples < 2:
        raise ValueError("samples must be at least 2")

    import matplotlib.pyplot as plt

    cases = build_density_ramp_cases(campaign, simulation_half_length)
    z = np.linspace(0.0, simulation_half_length, samples)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 6.5))
    for axis, case in zip(axes.flat, cases, strict=True):
        normalized_density = case.density.build_density_function()(z, 0.0)
        axis.plot(z * 1.0e3, normalized_density, color="tab:blue", linewidth=2)
        axis.set_title(
            f"FWHM {case.fwhm_left_m * 1.0e3:g} mm | "
            f"edge {case.requested_ramp_edge_percent:.0%} -> "
            f"{case.effective_ramp_edge_percent:.0%}\n"
            f"z1% = {case.one_percent_location_m * 1.0e3:.2f} mm | "
            f"beta = {case.beta_left:.2f}",
            fontsize=11,
        )
        axis.set_xlim(0.0, simulation_half_length * 1.0e3)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.3)

    for axis in axes[-1, :]:
        axis.set_xlabel("z (mm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized neutral density")
    fig.suptitle("Left density-ramp extrema", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
