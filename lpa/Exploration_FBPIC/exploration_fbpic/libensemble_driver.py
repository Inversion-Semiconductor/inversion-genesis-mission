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
from .libensemble_adapter import run_fbpic_simulation


def next_run_id_offset(run_root: str | Path) -> int:
    """Return the first unused numeric ID among existing ``sim_`` directories."""
    numbers = [
        int(path.name.removeprefix("sim_"))
        for path in Path(run_root).glob("sim_*")
        if path.is_dir() and path.name.removeprefix("sim_").isdigit()
    ]
    return max(numbers, default=-1) + 1


def persistent_xopt(
    _: np.ndarray,
    persis_info: dict[str, Any],
    gen_specs: dict[str, Any],
    libE_info: dict[str, Any],
):
    """Seed Xopt with a fixed initial design, then asynchronously explore."""
    from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
    from libensemble.tools.persistent_support import PersistentSupport

    user = gen_specs["user"]
    xopt = user["xopt"]
    variable_names: tuple[str, ...] = tuple(user["variable_names"])
    feature_names: tuple[str, ...] = tuple(user["feature_names"])
    initial_points: list[dict[str, float]] = user["initial_points"]
    sim_max = int(user["sim_max"])
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
            generated = xopt.generator.generate(count)
        candidates = np.zeros(len(generated), dtype=gen_specs["out"])
        for index, point in enumerate(generated):
            for name in variable_names:
                candidates[name][index] = point[name]
        submitted += len(candidates)
        tag, _, completed = support.send_recv(candidates)

        if len(completed):
            rows = pd.DataFrame({name: completed[name] for name in (*variable_names, *feature_names)})
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
    from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

    explorer = XoptFBPICExplorer(Campaign.from_file(config_path), stage)
    settings = explorer.campaign.config.stage_settings(stage)
    executor = MPIExecutor()
    executor.register_app(full_path=sys.executable, app_name="fbpic_python")
    ensemble = Ensemble(parse_args=True, executor=executor)

    variables = tuple(explorer.variable_bounds)
    features = explorer.feature_names
    initial_count = min(settings.initial_design_size, settings.max_evaluations)
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
            "run_id_offset": next_run_id_offset(explorer.campaign.config.run_root),
        },
    )
    ensemble.gen_specs = GenSpecs(
        gen_f=persistent_xopt,
        persis_in=["sim_id", "status_code", *variables, *features],
        out=[(name, float) for name in variables],
        user={
            "xopt": explorer.make_xopt(),
            "explorer": explorer,
            "variable_names": variables,
            "feature_names": features,
            "initial_points": explorer.initial_design(initial_count),
            "sim_max": settings.max_evaluations,
            "batch_size": settings.concurrent_simulations,
        },
    )
    ensemble.alloc_specs = AllocSpecs(
        alloc_f=only_persistent_gens,
        user={"async_return": True},
    )
    ensemble.exit_criteria = ExitCriteria(sim_max=settings.max_evaluations)
    ensemble.add_random_streams()
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an Xopt/libEnsemble FBPIC campaign")
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", default="stage_1")
    args = parser.parse_args()
    ensemble = make_ensemble(args.config, args.stage)
    if ensemble.nworkers < ensemble.gen_specs.user["batch_size"]:
        raise RuntimeError(
            f"need {ensemble.gen_specs.user['batch_size']} libEnsemble workers, "
            f"received {ensemble.nworkers}; launch with --nworkers"
        )
    ensemble.run()


if __name__ == "__main__":
    main()