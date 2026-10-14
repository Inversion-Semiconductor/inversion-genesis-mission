"""Plot campaign density-ramp extremes without running FBPIC."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import Campaign


_LEFT_RAMP_PARAMETERS = ("density_fwhm_left_m", "density_beta_left")


@dataclass(frozen=True)
class DensityRampCase:
    """One lower/upper left-ramp configuration and its constructed density."""

    fwhm_left_m: float
    beta_left: float
    density: Any


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
        beta = campaign.config.parameters[_LEFT_RAMP_PARAMETERS[1]]
    except KeyError as error:
        raise ValueError(
            "campaign must define density_fwhm_left_m and density_beta_left"
        ) from error
    if not fwhm.variable or not beta.variable:
        raise ValueError("left-ramp parameters must each have lower and upper bounds")

    template_class = _load_template_class(campaign)
    nominal_parameters = {
        name: parameter.nominal
        for name, parameter in campaign.config.parameters.items()
    }
    cases = []
    for fwhm_left_m in (fwhm.lower, fwhm.upper):
        for beta_left in (beta.lower, beta.upper):
            parameters = {
                **nominal_parameters,
                "density_center_location_m": simulation_half_length,
                "density_fwhm_left_m": fwhm_left_m,
                "density_beta_left": beta_left,
            }
            simulation = template_class(
                parameters, campaign.config.execution["hyperparameters"]
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
                    beta_left=float(beta_left),
                    density=density,
                )
            )
    return tuple(cases)


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
            f"FWHM left = {case.fwhm_left_m * 1.0e3:g} mm, "
            f"beta left = {case.beta_left:g}"
        )
        axis.set_xlim(0.0, simulation_half_length * 1.0e3)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.3)

    for axis in axes[-1, :]:
        axis.set_xlabel("z (mm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized neutral density")
    fig.suptitle("Left density-ramp extrema")
    fig.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
