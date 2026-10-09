"""libEnsemble simulation function for resource-managed FBPIC execution."""

from __future__ import annotations

from contextlib import chdir
import time
from typing import Any

import numpy as np
from libensemble.executors.executor import TimeoutExpired

from .executor import EvaluationResult, FBPICRunExecutor


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
    store = user["store"]
    stage: str = user["stage"]
    active_names: tuple[str, ...] = tuple(user["active_parameter_names"])
    run_id_offset = int(user.get("run_id_offset", 0))
    out = np.zeros(len(H), dtype=sim_specs["out"])
    for index, row in enumerate(H):
        proposal = {name: float(row[name]) for name in active_names}
        run_number = run_id_offset + (int(row["sim_id"]) if "sim_id" in H.dtype.names else index)
        run_id = f"sim_{run_number:06d}"
        run_directory, manifest = executor.prepare_run(run_id, stage, proposal)
        started = time.perf_counter()
        with chdir(run_directory):
            task = libE_info["executor"].submit(
                app_name="fbpic_python",
                app_args="run_fbpic.py",
                auto_assign_gpus=True,
                match_procs_to_gpus=True,
                extra_args="--cpu-bind=none",
                stdout="stdout.log",
                stderr="stderr.log",
            )
            try:
                task.wait(timeout=float(executor.config.execution.get("timeout_s", 21600)))
            except TimeoutExpired as error:
                task.kill()
                result = EvaluationResult(
                    run_id, "timeout", {}, run_directory, str(error), time.perf_counter() - started
                )
            else:
                result = None
        if result is not None:
            pass
        elif task.state == "FINISHED":
            try:
                features = executor.extract_features(run_directory)
            except (FileNotFoundError, ValueError, OSError) as error:
                result = EvaluationResult(
                    run_id, "analysis_error", {}, run_directory, str(error), time.perf_counter() - started
                )
            else:
                result = EvaluationResult(run_id, "ok", features, run_directory, wall_s=time.perf_counter() - started)
        else:
            result = EvaluationResult(
                run_id,
                "fbpic_error",
                {},
                run_directory,
                f"libEnsemble task ended as {task.state} (exit code {task.errcode})",
                time.perf_counter() - started,
            )
        executor._write_result(run_directory, manifest, result)
        store.append(result)
        out["status_code"][index] = 0 if result.status == "ok" else 1
        for feature in result.features:
            if feature in out.dtype.names:
                out[feature][index] = result.features[feature]
    return out, persis_info
