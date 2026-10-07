"""libEnsemble simulation function for slot-aware FBPIC execution at NERSC.

Pass this function to libEnsemble with ``sim_specs['user']`` containing an
``FBPICRunExecutor``, the stage name, and the active parameter names. Resource
allocation and GPU binding remain libEnsemble's responsibility inside the Slurm
allocation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .executor import FBPICRunExecutor


def run_fbpic_simulation(
    H: np.ndarray,
    persis_info: dict[str, Any],
    sim_specs: dict[str, Any],
    libE_info: dict[str, Any],
):
    """Evaluate libEnsemble-generated candidates and return scalar outputs.

    Required ``sim_specs['user']`` entries are ``executor``, ``store``,
    ``stage``, and ``active_parameter_names``. The ``out`` dtype must include
    ``status_code`` and any requested scalar feature names.
    """
    user = sim_specs["user"]
    executor: FBPICRunExecutor = user["executor"]
    store = user["store"]
    stage: str = user["stage"]
    active_names: tuple[str, ...] = tuple(user["active_parameter_names"])
    out = np.zeros(len(H), dtype=sim_specs["out"])
    for index, row in enumerate(H):
        proposal = {name: float(row[name]) for name in active_names}
        run_number = int(row["sim_id"]) if "sim_id" in H.dtype.names else index
        result = executor.evaluate(f"sim_{run_number:06d}", stage, proposal)
        store.append(result)
        out["status_code"][index] = 0 if result.status == "ok" else 1
        for feature in result.features:
            if feature in out.dtype.names:
                out[feature][index] = result.features[feature]
    return out, persis_info
