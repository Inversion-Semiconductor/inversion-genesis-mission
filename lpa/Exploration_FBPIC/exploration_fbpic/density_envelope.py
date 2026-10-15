"""Plot nominal and randomly sampled density profiles from an FBPIC campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import Campaign
from .density_ramps import _load_template_class


@dataclass(frozen=True)
class DensityEnvelopeCase:
    """One full campaign parameter vector and its constructed density profile."""

    parameters: dict[str, float | str]
    density: Any


def build_density_envelope_cases(
    campaign: Campaign,
    stage: str,
    random_samples: int,
    seed: int = 0,
) -> tuple[DensityEnvelopeCase, ...]:
    """Build a nominal profile followed by random samples of one campaign stage."""
    if random_samples < 0:
        raise ValueError("random_samples must be non-negative")

    template_class = _load_template_class(campaign)
    active_parameters = campaign.config.active_parameters(stage)
    random_generator = np.random.default_rng(seed)
    proposals: list[dict[str, float]] = [{}]
    for _ in range(random_samples):
        proposal = {}
        for parameter in active_parameters:
            assert parameter.lower is not None and parameter.upper is not None
            if parameter.log_scale:
                proposal[parameter.name] = float(
                    np.exp(
                        random_generator.uniform(
                            np.log(parameter.lower), np.log(parameter.upper)
                        )
                    )
                )
            else:
                proposal[parameter.name] = float(
                    random_generator.uniform(parameter.lower, parameter.upper)
                )
        proposals.append(proposal)

    cases = []
    for proposal in proposals:
        parameters = campaign.config.complete_parameters(stage, proposal)
        simulation = template_class(
            parameters, campaign.config.template_hyperparameters()
        ).build_simulation()
        try:
            density = next(
                density
                for density in simulation.densities
                if hasattr(density, "build_density_function")
            )
        except StopIteration as error:
            raise ValueError("template simulation has no density profile") from error
        cases.append(DensityEnvelopeCase(parameters=parameters, density=density))
    return tuple(cases)


def plot_density_envelope(
    campaign: Campaign,
    stage: str,
    output: str | Path,
    random_samples: int = 20,
    seed: int = 0,
    samples: int = 1000,
) -> Path:
    """Save nominal and random campaign density profiles on one set of axes."""
    if samples < 2:
        raise ValueError("samples must be at least 2")

    import matplotlib.pyplot as plt

    cases = build_density_envelope_cases(campaign, stage, random_samples, seed)
    z_min = min(case.density.get_z_extent()[0] for case in cases)
    z_max = max(case.density.get_z_extent()[1] for case in cases)
    z = np.linspace(z_min, z_max, samples)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for case in cases[1:]:
        density = case.density.build_density_function()(z, 0.0)
        axis.plot(z * 1.0e3, density, color="tab:orange", alpha=0.18, linewidth=0.8)
    nominal_density = cases[0].density.build_density_function()(z, 0.0)
    axis.plot(
        z * 1.0e3,
        nominal_density,
        color="tab:blue",
        linewidth=2.2,
        label="Nominal profile",
    )
    if random_samples:
        axis.plot(
            [],
            [],
            color="tab:orange",
            alpha=0.45,
            linewidth=1.2,
            label="Random campaign samples",
        )
    axis.set_xlabel("z (mm)")
    axis.set_ylabel("normalized neutral density")
    axis.set_title(f"Density-profile envelope ({stage})")
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
