"""libEnsemble simulation function for resource-managed FBPIC execution."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path
from typing import Any

import numpy as np
from libensemble.executors.executor import TimeoutExpired

from .executor import ExecutionFailure, FBPICRunExecutor


def run_fbpic_simulation(
    H: np.ndarray,
    persis_info: dict[str, Any],
    sim_specs: dict[str, Any],
    libE_info: dict[str, Any],
):
    """Evaluate libEnsemble-generated candidates and return scalar outputs.

    Required ``sim_specs['user']`` entries are ``executor``, ``store``,
    ``stage``, and ``active_parameter_names``. Each worker uses libEnsemble's
    MPI executor, which assigns the worker's resource-set GPUs to the FBPIC
    ranks. The ``out`` dtype must include ``status_code`` and selected features.
    """
    user = sim_specs["user"]
    executor: FBPICRunExecutor = user["executor"]
    stage: str = user["stage"]
    active_names: tuple[str, ...] = tuple(user["active_parameter_names"])
    log_scaled_names: set[str] = set(user.get("log_scaled_names", ()))
    linear_scales: dict[str, float] = dict(user.get("linear_scales", {}))
    run_id_offset = int(user.get("run_id_offset", 0))
    simulation_timeout_s = executor.execution_timeout_s()
    out = np.zeros(len(H), dtype=sim_specs["out"])
    for name in out.dtype.names or ():
        if name != "status_code" and np.issubdtype(out.dtype[name], np.floating):
            out[name].fill(np.nan)
    for index, row in enumerate(H):
        model_point = {name: float(row[name]) for name in active_names}
        physical_point = {
            name: (float(np.exp(value)) if name in log_scaled_names else value)
            * linear_scales.get(name, 1.0)
            for name, value in model_point.items()
        }
        run_number = run_id_offset + (
            int(row["sim_id"]) if "sim_id" in H.dtype.names else index
        )
        run_id = f"sim_{run_number:06d}"
        run_directory, manifest = executor.prepare_run(run_id, stage, physical_point)

        def launch(directory: Path) -> None:
            with chdir(directory):
                task = libE_info["executor"].submit(
                    app_name="fbpic_python",
                    app_args="run_fbpic.py",
                    auto_assign_gpus=True,
                    match_procs_to_gpus=True,
                    extra_args="--cpu-bind=none --overlap",
                    stdout="stdout.log",
                    stderr="stderr.log",
                )
                try:
                    task.wait(timeout=simulation_timeout_s)
                except TimeoutExpired as error:
                    task.kill()
                    raise ExecutionFailure("timeout", str(error)) from error
            if task.state == "FINISHED":
                return
            raise ExecutionFailure(
                "fbpic_error",
                f"libEnsemble task ended as {task.state} (exit code {task.errcode})",
            )

        result = executor.complete_prepared_run(run_id, run_directory, manifest, launch)
        out["status_code"][index] = {
            "ok": 0,
            "no_beam": 1,
        }.get(result.status, 2)
        for feature in result.features:
            if feature in out.dtype.names:
                out[feature][index] = result.features[feature]
    return out, persis_info
