"""Reusable Optimas optimization template for FBPIC simulations.

Provides :func:`run_optimas_exploration` to configure and launch an Optimas
Bayesian optimization with automatic GPU detection, logging, and optional
run resumption.  Wrapper class `VaryingParameter` allows for parameters in
the varying parameters list to be fixed when passing into the template

Example caller script (``optimas_runscript.py``)::

    import logging
    try:
        from inversion_fbpic.utils.optimas_template import run_optimas_exploration, VaryingParameter
        from inversion_fbpic.utils.optimas_analysis import analyze_hofi_simulation as analysis_func
    except ImportError:
        from optimas_analysis import analyze_hofi_simulation as analysis_func  # type: ignore
        from optimas_template import run_optimas_exploration, VaryingParameter  # type: ignore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("optimization.log"),
            logging.StreamHandler(),
        ],
    )

    if __name__ == "__main__":
        run_optimas_exploration(
            analysis_func=analysis_func,
            sim_template="hofi_template.py",
            varying_parameters=[
                VaryingParameter("laser_focal_position", -1e-3, 1.5e-3),
                VaryingParameter("dope_level", 0.01, 0.20),
                VaryingParameter("injection_height", 0.4, 0.7),
                VaryingParameter("laser_temporal_width_fs", 30, 40),
                VaryingParameter("target_a0", 1.7, 2.0).fix_value(1.8),
                VaryingParameter("plateau_end_position_mm", 10.0, 15.0),
            ],
            num_workers=8,
            max_evals=360,
        )

Run the caller with ``python optimas_runscript.py &``.
Do **not** use ``mpirun``; this script handles GPU allocation on its own.
"""

from __future__ import annotations

import logging
import traceback
import os
from typing import Any, Callable, List, Optional

from optimas.core import Parameter, Objective
from optimas.core import VaryingParameter as _VaryingParameter
from optimas.generators import AxSingleFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration
from optimas.diagnostics import ExplorationDiagnostics

try:
    from inversion_fbpic.utils.optimas_analysis import find_optimum_predicted_result
except ImportError:
    from optimas_analysis import find_optimum_predicted_result  # type: ignore

logger: logging.Logger = logging.getLogger(__name__)

# Better suited for more dimensionality.  More time to generate trials, but can lead to fewer
FULLY_BAYESIAN = False


class VaryingParameter(_VaryingParameter):
    """Wrapper class to handle fixing values"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def fix_value(self, value: float) -> VaryingParameter:
        super().fix_value(value)
        return self


def run_optimas_exploration(
    analysis_func: Callable,
    sim_template: str,
    varying_parameters: List[VaryingParameter],
    max_evals: int,
    analyzed_parameters: Optional[List[Parameter]] = None,
    num_workers: Optional[int] = None,
    n_gpus_per_sim: int = 1,
    resume: bool = False,
    sim_files: Optional[List[str]] = None,
) -> None:
    """Configure and run an Optimas Bayesian optimization exploration.

    This function encapsulates the full Optimas workflow: generator
    creation, GPU detection, evaluator setup, and the exploration loop
    with result logging.

    Must be called inside an ``if __name__ == "__main__":`` guard (required
    by multiprocessing on spawn/forkserver platforms).

    Args:
        analysis_func: Analysis function to evaluate each simulation.
            Must follow the Optimas analysis-function signature
            ``(simulation_directory, output_params, ...) -> dict``.
            If it has an ``analyzed_param_names`` attribute (set by the
            ``@analyzed_params`` decorator in ``optimas_analysis``), that
            list is used to auto-create :class:`Parameter` objects when
            *analyzed_parameters* is ``None``.
        sim_template: Path to the FBPIC simulation template script.
            Resolved relative to CWD if not absolute.
        varying_parameters: :class:`VaryingParameter` instances that define
            the optimization search space.
        max_evals: Maximum number of evaluations.  Ex: 360
        analyzed_parameters: Optional list of :class:`Parameter` objects for
            additional tracked quantities.  If ``None``, auto-detected from
            ``analysis_func.analyzed_param_names``.
        num_workers: Number of concurrent simulation workers (typically
            equal to the number of GPUs).  Defaults to number of GPUs.
        n_gpus_per_sim: Number of GPUs to allocate per simulation.  Defaults to 1.
        resume: If ``True``, resume from a previous optimization run
            stored in the current directory.  Default ``False``.
        sim_files: Extra files to copy over to the evaluation directory. Defaults to ``None``.
    """
    # --- GPU detection ------------------------------------------------------
    n_gpus_available: int = 0
    try:
        import cupy as cp  # type: ignore

        n_gpus_available = cp.cuda.runtime.getDeviceCount()
        logger.info("Detected %d GPUs available", n_gpus_available)
    except ImportError:
        logger.warning("CuPy not available, using CPU mode")

    if num_workers is None:
        num_workers = n_gpus_available

    if n_gpus_available > 0 and n_gpus_available >= num_workers:
        sim_workers: int = num_workers
    else:
        n_gpus_per_sim = 0
        sim_workers = 1
        logger.warning(
            "Only %d GPUs available but %d workers requested; falling back to CPU mode",
            n_gpus_available,
            num_workers,
        )

    # --- Analyzed parameters ------------------------------------------------
    if analyzed_parameters is None:
        analyzed_parameters = []
    p_names = [p.name for p in analyzed_parameters]

    if hasattr(analysis_func, "analyzed_param_names"):
        logger.info(
            "Auto-detected analyzed parameters: %s",
            analysis_func.analyzed_param_names,
        )
        for name in analysis_func.analyzed_param_names:
            if name not in p_names:
                analyzed_parameters.append(Parameter(name))
    # check if still no params
    if not analyzed_parameters:
        logger.warning(
            "No analyzed_parameters provided and analysis function has no "
            "analyzed_param_names attribute; proceeding with none."
        )

    # --- Objective & generator ----------------------------------------------
    objective: Objective = Objective("f", minimize=False)

    varying_names: List[str] = [p.name for p in varying_parameters]
    logger.info("Varying parameters: %s", varying_names)
    logger.info("Objective: %s, minimize=%s", objective.name, objective.minimize)

    generator: AxSingleFidelityGenerator = AxSingleFidelityGenerator(
        varying_parameters=varying_parameters,
        objectives=[objective],
        analyzed_parameters=analyzed_parameters,
        n_init=sim_workers,
        fully_bayesian=FULLY_BAYESIAN,
    )

    # --- Template path ------------------------------------------------------
    template_path: str = os.path.abspath(sim_template)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not os.access(template_path, os.R_OK):
        raise PermissionError(f"Template file not readable: {template_path}")
    logger.info(
        "Using template: %s (%d bytes)",
        template_path,
        os.path.getsize(template_path),
    )

    # --- Evaluator ----------------------------------------------------------
    evaluator_kwargs: dict[str, Any] = {
        "sim_template": template_path,
        "analysis_func": analysis_func,
        "sim_files": (
            [os.path.abspath(sf) for sf in sim_files] if sim_files is not None else None
        ),
    }
    if sim_files is not None:
        for sf in evaluator_kwargs["sim_files"]:
            if not os.path.exists(sf):
                raise FileNotFoundError(f"Simulation file not found: {sf}")
            if not os.access(sf, os.R_OK):
                raise PermissionError(f"Simulation file not readable: {sf}")

    if n_gpus_per_sim > 0:
        evaluator_kwargs["n_gpus"] = n_gpus_per_sim

    evaluator: TemplateEvaluator = TemplateEvaluator(**evaluator_kwargs)

    # --- Exploration --------------------------------------------------------
    exploration_kwargs: dict[str, Any] = {
        "generator": generator,
        "evaluator": evaluator,
        "max_evals": max_evals,
    }

    if n_gpus_per_sim > 0:
        exploration_kwargs["sim_workers"] = sim_workers
        exploration_kwargs["run_async"] = True
    else:
        exploration_kwargs["sim_workers"] = 1
        exploration_kwargs["run_async"] = False

    exploration_kwargs["resume"] = resume

    exploration: Exploration = Exploration(**exploration_kwargs)

    if n_gpus_per_sim > 0:
        logger.info(
            "Starting optimization with %d parallel workers, " "%d GPUs per simulation",
            sim_workers,
            n_gpus_per_sim,
        )
    else:
        logger.info("Starting optimization in CPU mode with synchronous execution")

    os.environ["OPTIMAS_N_GPUS_PER_SIM"] = str(n_gpus_per_sim)

    # --- Run ----------------------------------------------------------------
    logger.info(
        "Starting optimization exploration%s...",
        " (resuming)" if resume else "",
    )
    logger.info(
        "Note: The RandomModelBridge warning is expected during initial "
        "random trials"
    )

    try:
        exploration.run()
        logger.info("Optimization completed successfully!")

        exp_diags: ExplorationDiagnostics = ExplorationDiagnostics(exploration)
        best_ev, best_gp = find_optimum_predicted_result(
            exp_diags, "f", minimize_objective=False
        )

        logger.info("Best evaluated result:\n\t%s\n\t%s", best_ev[0], best_ev[1])
        logger.info(
            "Best predicted result:\n\t%s\n\t%s +/- %s",
            best_gp[0],
            best_gp[1],
            best_gp[2],
        )

    except Exception as e:
        logger.error("Optimization failed with error: %s", e)

        logger.error("Full traceback: %s", traceback.format_exc())
        raise
