"""Persistent Xopt exploration managed by libEnsemble resource sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from .campaign import Campaign
from .exploration import XoptFBPICExplorer
from .feasibility import BeamFeasibilityModel
from .libensemble_adapter import run_fbpic_simulation
from .store import CampaignStore


def next_run_id_offset(run_root: str | Path) -> int:
    """Return the first unused numeric ID among existing ``sim_`` directories."""
    numbers = [
        int(path.name.removeprefix("sim_"))
        for path in Path(run_root).glob("sim_*")
        if path.is_dir() and path.name.removeprefix("sim_").isdigit()
    ]
    return max(numbers, default=-1) + 1


def fallback_design(
    explorer: XoptFBPICExplorer, offset: int, count: int
) -> list[dict[str, float]]:
    """Return a non-overlapping deterministic design slice when GP training has no data."""
    return explorer.model_initial_design(offset + count, seed=0)[offset:]


def select_feasible_candidates(
    candidates: list[dict[str, float]],
    feasibility: BeamFeasibilityModel,
    count: int,
    minimum_probability: float,
) -> list[dict[str, float]]:
    """Keep feature-ranked feasible candidates, then backfill by feasibility.

    Xopt orders generated candidates by its feature-exploration acquisition.
    Candidates at or above the configured probability floor keep that order;
    only below-floor candidates are reordered for an unavoidable backfill.
    """
    return [
        candidates[index]
        for index in select_feasible_candidate_indices(
            candidates, feasibility, count, minimum_probability
        )
    ]


def select_feasible_candidate_indices(
    candidates: list[dict[str, float]],
    feasibility: BeamFeasibilityModel,
    count: int,
    minimum_probability: float,
) -> np.ndarray:
    """Return selected candidate positions, preserving Xopt order when feasible."""
    if len(candidates) <= count or not feasibility.is_ready:
        return np.arange(min(len(candidates), count))
    probabilities = feasibility.beam_probability(candidates)
    eligible = np.flatnonzero(probabilities >= minimum_probability)
    selected = list(eligible[:count])
    if len(selected) < count:
        ineligible = np.flatnonzero(probabilities < minimum_probability)
        backfill = ineligible[np.argsort(probabilities[ineligible])[::-1]]
        selected.extend(backfill[: count - len(selected)])
    return np.asarray(selected, dtype=int)


def preview_xopt_batch(
    explorer: XoptFBPICExplorer,
    data: pd.DataFrame,
    count: int,
    minimum_beam_feasibility: float | None = None,
    candidate_pool_factor: int = 8,
) -> dict[str, object]:
    """Return a no-submit preview of the Xopt/feasibility candidate decision."""
    if count < 1:
        raise ValueError("count must be positive")
    if candidate_pool_factor < 1:
        raise ValueError("candidate_pool_factor must be positive")
    if minimum_beam_feasibility is None:
        minimum_beam_feasibility = explorer.campaign.config.stage_settings(
            explorer.stage
        ).minimum_beam_feasibility
    if not 0.0 <= minimum_beam_feasibility <= 1.0:
        raise ValueError("minimum_beam_feasibility must be in [0, 1]")

    xopt = explorer.make_xopt(data)
    feasibility = BeamFeasibilityModel(explorer)
    feasibility.fit_campaign_data(data)
    pool_count = count * candidate_pool_factor
    if xopt.data is None or xopt.data.empty:
        proposed = fallback_design(explorer, 0, pool_count)
        source = "fallback_design"
    else:
        proposed = xopt.generator.generate(pool_count)
        source = "xopt_generator"
    probabilities = feasibility.beam_probability(proposed)
    uncertainty = _relative_combined_gp_uncertainty(explorer, xopt, data, proposed)
    selected_indices = select_feasible_candidate_indices(
        proposed, feasibility, count, minimum_beam_feasibility
    )
    selection_rank = {
        int(index): rank + 1 for rank, index in enumerate(selected_indices)
    }
    candidates = []
    for index, point in enumerate(proposed):
        candidates.append(
            {
                "xopt_rank": index + 1,
                "beam_feasibility": float(probabilities[index]),
                "relative_combined_gp_uncertainty": (
                    float(uncertainty[index])
                    if np.isfinite(uncertainty[index])
                    else None
                ),
                "meets_feasibility_floor": bool(
                    probabilities[index] >= minimum_beam_feasibility
                ),
                "selected_rank": selection_rank.get(index),
                "model_point": point,
                "physical_point": explorer.model_to_physical(point),
            }
        )
    return {
        "stage": explorer.stage,
        "source": source,
        "count": count,
        "candidate_pool_count": pool_count,
        "candidate_pool_factor": candidate_pool_factor,
        "minimum_beam_feasibility": minimum_beam_feasibility,
        "xopt_rank_note": (
            "Xopt uses joint qPosteriorVariance batch optimization; return order is "
            "not an independent uncertainty ranking."
        ),
        "relative_combined_gp_uncertainty_note": (
            "RMS posterior standard deviation across exploration features after each "
            "feature is divided by its observed training standard deviation; normalized "
            "to the maximum candidate-pool value."
        ),
        "candidates": candidates,
        "selected": [candidates[index] for index in selected_indices],
    }


def _relative_combined_gp_uncertainty(
    explorer: XoptFBPICExplorer,
    xopt: Any,
    raw_data: pd.DataFrame,
    candidates: list[dict[str, float]],
) -> np.ndarray:
    """Calculate marginal multi-feature GP uncertainty for preview diagnostics."""
    if not candidates:
        return np.empty(0)
    try:
        import torch

        generator = xopt.generator
        training = explorer.training_data(raw_data)
        model = generator.train_model(data=training, update_internal=False)
        inputs = torch.tensor(
            [
                [point[name] for name in explorer.variable_bounds]
                for point in candidates
            ],
            **generator.tkwargs,
        )
        standard_deviation = (
            model.posterior(inputs).variance.sqrt().detach().cpu().numpy()
        )
        scales = (
            training.loc[:, list(explorer.feature_names)]
            .std(axis=0)
            .to_numpy(dtype=float, copy=True)
        )
        scales[scales == 0] = 1.0
        combined = np.sqrt(np.mean((standard_deviation / scales) ** 2, axis=1))
        maximum = float(combined.max())
        return combined / maximum if maximum > 0 else np.zeros(len(combined))
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return np.full(len(candidates), np.nan)


def persistent_xopt(
    _: np.ndarray,
    persis_info: dict[str, Any],
    gen_specs: dict[str, Any],
    libE_info: dict[str, Any],
):
    """Seed Xopt with a fixed initial design, then asynchronously explore."""
    from libensemble.message_numbers import (
        EVAL_GEN_TAG,
        FINISHED_PERSISTENT_GEN_TAG,
        PERSIS_STOP,
        STOP_TAG,
    )
    from libensemble.tools.persistent_support import PersistentSupport

    user = gen_specs["user"]
    explorer = XoptFBPICExplorer(Campaign.from_file(user["config_path"]), user["stage"])
    xopt = explorer.make_xopt()
    feasibility = BeamFeasibilityModel(explorer)
    feasibility.fit_campaign_data(explorer.available_data())
    variable_names: tuple[str, ...] = tuple(user["variable_names"])
    feature_names: tuple[str, ...] = tuple(user["feature_names"])
    initial_points: list[dict[str, float]] = user["initial_points"]
    sim_max = int(user["sim_max"])
    fallback_offset = int(user.get("fallback_offset", 0))
    candidate_pool_factor = int(user.get("candidate_pool_factor", 8))
    minimum_beam_feasibility = float(user.get("minimum_beam_feasibility", 0.0))
    support = PersistentSupport(libE_info, EVAL_GEN_TAG)
    submitted = 0
    pending = list(initial_points)
    tag = None

    while tag not in {STOP_TAG, PERSIS_STOP}:
        if pending:
            generated = pending[: int(user["batch_size"])]
            pending = pending[len(generated) :]
        else:
            count = min(int(user["batch_size"]), sim_max - submitted)
            if count <= 0:
                break
            if xopt.data is None or xopt.data.empty:
                generated = fallback_design(
                    explorer, fallback_offset + submitted, count
                )
            else:
                proposed = xopt.generator.generate(count * candidate_pool_factor)
                generated = select_feasible_candidates(
                    proposed, feasibility, count, minimum_beam_feasibility
                )
        candidates = np.zeros(len(generated), dtype=gen_specs["out"])
        for index, point in enumerate(generated):
            for name in variable_names:
                candidates[name][index] = point[name]
        submitted += len(candidates)
        tag, _, completed = support.send_recv(candidates)

        if completed is not None and len(completed):
            rows = pd.DataFrame(
                {name: completed[name] for name in (*variable_names, *feature_names)}
            )
            feasibility.add_completed(
                rows.loc[:, variable_names], completed["status_code"]
            )
            successful = completed["status_code"] == 0
            successful &= np.isfinite(rows.loc[:, feature_names]).all(axis=1).to_numpy()
            if successful.any():
                xopt.add_data(rows.loc[successful].reset_index(drop=True))

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def make_ensemble(config_path: str | Path, stage: str):
    """Build a resource-managed libEnsemble workflow for one campaign stage."""
    from libensemble import Ensemble
    from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
    from libensemble.executors import MPIExecutor
    from libensemble.specs import (
        AllocSpecs,
        ExitCriteria,
        GenSpecs,
        LibeSpecs,
        SimSpecs,
    )

    explorer = XoptFBPICExplorer(Campaign.from_file(config_path), stage)
    explorer.require_fresh_run_root()
    explorer.require_compatible_run_root()
    explorer.training_data()
    settings = explorer.campaign.config.stage_settings(stage)
    completed_count = (
        explorer.completed_evaluation_count() if settings.continuation else 0
    )
    remaining_evaluations = settings.max_evaluations - completed_count
    if remaining_evaluations < 1:
        raise RuntimeError(
            f"stage {stage!r} already has {completed_count} compatible evaluations, "
            f"meeting its budget of {settings.max_evaluations}"
        )
    executor = MPIExecutor()
    executor.register_app(full_path=sys.executable, app_name="fbpic_python")
    ensemble = Ensemble(parse_args=True, executor=executor)

    variables = tuple(explorer.variable_bounds)
    features = explorer.feature_names
    # A continuation may expand the active parameter set. Its stage-local
    # initial design supplies coverage for those newly variable coordinates,
    # while all prior-stage rows remain available to train the conditional GPs.
    initial_count = min(
        max(0, settings.initial_design_size - completed_count),
        remaining_evaluations,
    )
    ensemble.libE_specs = LibeSpecs(
        comms="mpi",
        gen_on_manager=True,
        num_resource_sets=settings.concurrent_simulations,
        gpus_per_group=settings.gpus_per_simulation,
        platform="perlmutter_g",
        sim_dirs_make=False,
    )
    ensemble.sim_specs = SimSpecs(
        sim_f=run_fbpic_simulation,
        inputs=["sim_id", *variables],
        outputs=[("status_code", int), *((feature, float) for feature in features)],
        user={
            "executor": explorer.executor,
            "store": explorer.store,
            "stage": stage,
            "active_parameter_names": variables,
            "log_scaled_names": tuple(
                parameter.name
                for parameter in explorer.parameters
                if parameter.log_scale
            ),
            "linear_scales": {
                parameter.name: parameter.linear_scale
                for parameter in explorer.parameters
            },
            "run_id_offset": next_run_id_offset(explorer.campaign.config.run_root),
        },
    )
    ensemble.gen_specs = GenSpecs(
        gen_f=persistent_xopt,
        persis_in=["sim_id", "status_code", *variables, *features],
        out=[(name, float) for name in variables],
        user={
            "config_path": str(Path(config_path).resolve()),
            "stage": stage,
            "variable_names": variables,
            "feature_names": features,
            "initial_points": explorer.model_initial_design(
                completed_count + initial_count
            )[completed_count:],
            "sim_max": remaining_evaluations,
            "batch_size": settings.concurrent_simulations,
            "fallback_offset": completed_count,
            "candidate_pool_factor": 8,
            "minimum_beam_feasibility": settings.minimum_beam_feasibility,
        },
    )
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=only_persistent_gens,
        # FBPIC candidates are generated in worker-sized batches. Returning
        # partial batches can exhaust sim_max while candidates remain queued.
        user={"async_return": False, "init_sample_size": initial_count},
    )
    ensemble.exit_criteria = ExitCriteria(sim_max=remaining_evaluations)
    ensemble.add_random_streams()
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Xopt/libEnsemble FBPIC campaign"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", default="stage_1")
    args = parser.parse_args()
    campaign = Campaign.from_file(args.config).for_stage(args.stage)
    ensemble = make_ensemble(args.config, args.stage)
    if ensemble.nworkers < ensemble.gen_specs.user["batch_size"]:
        raise RuntimeError(
            f"need {ensemble.gen_specs.user['batch_size']} libEnsemble workers, "
            f"received {ensemble.nworkers}; launch with --nworkers"
        )
    try:
        ensemble.run()
    finally:
        # Workers persist result.json individually; aggregate once to avoid
        # filesystem locks unsupported by the shared scratch filesystem.
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            CampaignStore(campaign.config.run_root).rebuild()


if __name__ == "__main__":
    main()
