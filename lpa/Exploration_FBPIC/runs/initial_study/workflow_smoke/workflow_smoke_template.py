"""Short infrastructure-only subclass of the ionization-injection template."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import attrs

from inversion_fbpic.lib import diagnostics as dg


_BASE_TEMPLATE = (
    Path(__file__).parents[4]
    / "Simulation_FBPIC/runs/initial_sample/ionization_injection_template.py"
)
_SPECIFICATION = importlib.util.spec_from_file_location(
    "workflow_smoke_base_template", _BASE_TEMPLATE
)
if _SPECIFICATION is None or _SPECIFICATION.loader is None:
    raise ImportError(f"could not import {_BASE_TEMPLATE}")
_MODULE = importlib.util.module_from_spec(_SPECIFICATION)
_SPECIFICATION.loader.exec_module(_MODULE)


@attrs.define(kw_only=True, slots=False)
class _SmokeScalarDiagnostic(dg._Diagnostic):
    """Emit deterministic scalar output for the workflow-smoke Xopt batch."""

    SUBCLASS: ClassVar[str] = "workflow_smoke_scalar"

    values: dict[str, float] = attrs.field(factory=dict)

    def _analyze(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.values.items()}


class WorkflowSmokeSimulation(_MODULE.IonizationInjectionSimulation):
    """Run a fixed number of steps and emit one synthetic Xopt feature."""

    _SMOKE_STEPS = 10
    _HYPERPARAMETER_DEFAULTS = {
        **_MODULE.IonizationInjectionSimulation._HYPERPARAMETER_DEFAULTS,
        "smoke_feature_name": "smoke_laser_energy_J",
    }

    def build_simulation(self):
        simulation = super().build_simulation()
        datapoint = next(
            element
            for element in simulation.elements
            if isinstance(element, _MODULE.cc.ConfigContainer)
        )
        datapoint.configs = [
            config
            for config in datapoint.configs
            if not isinstance(config, dg._Diagnostic)
        ]
        simulation.diagnostics = []
        simulation._sort_component(
            _SmokeScalarDiagnostic(
                values={
                    self.hyperparameters[
                        "smoke_feature_name"
                    ]: self.physical_parameters["laser_energy_J"]
                }
            )
        )
        datapoint.configs.extend(simulation.diagnostics)
        return simulation

    def run(self) -> None:
        """Run the short smoke workload and persist the synthetic result."""
        hyperparameters = self.hyperparameters
        simulation = self.build_simulation()
        datapoint = _MODULE.cc.ConfigContainer(
            configs=[*simulation.parameters, *simulation.diagnostics]
        )
        simulation.setup_simulation()
        simulation.simulation.step(self._SMOKE_STEPS, show_progress=True)
        if simulation.do_write_to_disk:
            simulation._save_hash()
            simulation._analyze_diagnostics(before_simulation=False)
            simulation.to_file(
                Path(hyperparameters["results_directory"])
                / f"simulation.{_MODULE._SERIALIZATION_FORMAT}",
                _MODULE._SERIALIZATION_FORMAT,
                overwrite=True,
            )
            datapoint.to_file(
                Path(hyperparameters["results_directory"])
                / f"datapoint.{_MODULE._DATAPOINT_FORMAT}",
                _MODULE._DATAPOINT_FORMAT,
                group_path="/results",
                overwrite=True,
            )
