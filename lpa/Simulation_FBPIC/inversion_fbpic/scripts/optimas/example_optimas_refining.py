"""Run a small Optimas refinement demo on an analytic objective.

This script creates a two-parameter optimization problem with a known optimum
and runs a short Optimas exploration. It then compares:

- the best completed evaluation found during exploration, and
- the surrogate-model predicted optimum returned by
  `find_optimum_predicted_result`.

Typical usage:
  python example_optimas_refining.py
"""

import tempfile

from optimas.core import Objective, VaryingParameter
from optimas.diagnostics import ExplorationDiagnostics
from optimas.evaluators import FunctionEvaluator
from optimas.explorations import Exploration
from optimas.generators import AxSingleFidelityGenerator

from inversion_fbpic.utils.optimas_analysis import find_optimum_predicted_result


def _evaluate_analytic_function(
    input_params: dict[str, float], output_params: dict[str, float]
) -> None:
    """Evaluate a smooth objective with a known global maximum.

    The function is:
      f(x, y) = -((x - 1)^2 + (y + 2)^2)

    Args:
        input_params: Mapping that contains numeric `"x"` and `"y"` entries.
        output_params: Mutable mapping where this function stores `"f"`.
    """
    x = float(input_params["x"])
    y = float(input_params["y"])
    output_params["f"] = -((x - 1.0) ** 2 + (y + 2.0) ** 2)


def main() -> None:
    """Run the demo exploration and print a comparison report.

    The report includes:
    - the best completed trial found by the optimizer,
    - the surrogate-model predicted optimum, and
    - the known analytic optimum for reference.
    """
    with tempfile.TemporaryDirectory(prefix="optimas_find_optimum_demo_") as tmp_dir:
        varying_parameters = [
            VaryingParameter("x", -4.0, 4.0),
            VaryingParameter("y", -4.0, 4.0),
        ]
        objectives = [Objective("f", minimize=False)]

        generator = AxSingleFidelityGenerator(
            varying_parameters=varying_parameters,
            objectives=objectives,
        )
        evaluator = FunctionEvaluator(function=_evaluate_analytic_function)

        exploration = Exploration(
            generator=generator,
            evaluator=evaluator,
            max_evals=20,
            sim_workers=1,
            exploration_dir_path=tmp_dir,
        )
        exploration.run()

        diagnostics = ExplorationDiagnostics(tmp_dir)
        predicted_best_params, predicted_best_value = find_optimum_predicted_result(
            exploration=diagnostics,
            objective="f",
            minimize_objective=False,
            verbose=False,
        )

        out: dict[str, float] = {}
        _evaluate_analytic_function(predicted_best_params, out)

        f_gp = diagnostics.build_gp_model("f", minimize=False)
        best_eval_params = f_gp.get_best_evaluation()[1]
        best_eval_out: dict[str, float] = {}
        _evaluate_analytic_function(best_eval_params, best_eval_out)

        _name_w = max(
            *(len(k) for k in best_eval_params),
            *(len(k) for k in predicted_best_params),
            len("true f"),
            len("model f"),
            len("f @ predicted"),
        )
        _val_w = 22
        _banner = "=" * (4 + _name_w + 2 + _val_w)

        def _kv_row(name: str, value: float) -> None:
            """Print one aligned key-value row in the demo summary table."""
            print(f"  {name:<{_name_w}}  {value:>{_val_w}.6g}")

        print(f"\n{_banner}\nDemo exploration — objective summary\n{_banner}")
        print("\nBest evaluated parameters (best completed trial)")
        for _k in sorted(best_eval_params):
            _kv_row(_k, float(best_eval_params[_k]))
        _kv_row("true f", float(best_eval_out["f"]))
        print("\nOptimas predicted optimum (surrogate model argmax)")
        for _k in sorted(predicted_best_params):
            _kv_row(_k, float(predicted_best_params[_k]))
        _kv_row("model f", float(predicted_best_value))
        print("\nTrue objective f(x, y) at predicted optimum")
        _kv_row("f @ predicted", float(out["f"]))
        print("\nKnown global optimum")
        _kv_row("x", 1.0)
        _kv_row("y", -2.0)
        _kv_row("f", 0.0)
        print(f"{_banner}\n")


if __name__ == "__main__":
    main()
