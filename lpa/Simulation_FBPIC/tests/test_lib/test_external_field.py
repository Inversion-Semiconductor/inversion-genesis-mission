"""Tests for serializable FBPIC external fields."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import attrs
import pytest

from inversion_fbpic.lib.external_field import (
    ExpressionExternalField,
    UniformExternalField,
)
from inversion_fbpic.lib.math_expression import MathExpression
from inversion_fbpic.lib.serializable_config import SerializableConfig


def test_uniform_external_field_builds_one_field_per_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inversion_fbpic.lib import external_field as module

    created: list[dict[str, Any]] = []

    def external_field(field_function, fieldtype, amplitude, lengthscale, **kwargs):
        created.append(
            {
                "field_function": field_function,
                "fieldtype": fieldtype,
                "amplitude": amplitude,
                "lengthscale": lengthscale,
                **kwargs,
            }
        )
        return MagicMock()

    monkeypatch.setattr(module, "FBPICExternalField", external_field)
    field = UniformExternalField(field_functions={"Ez": 2.0, "By": -0.5})
    species = object()

    assert len(field.build_external_fields(species=species, gamma_boost=2.0)) == 2
    assert {entry["fieldtype"] for entry in created} == {"Ez", "By"}
    assert all(
        entry["amplitude"] == 1.0
        and entry["lengthscale"] == 1.0
        and entry["species"] is species
        and entry["gamma_boost"] == 2.0
        for entry in created
    )
    assert created[0]["field_function"](1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0) in {
        3.0,
        0.5,
    }


def test_external_fields_fix_fbpics_amplitude_and_length_scale() -> None:
    field = UniformExternalField(field_functions={"Ez": 1.0})

    assert "amplitude" not in field.to_dict()["parameters"]
    assert "lengthscale" not in field.to_dict()["parameters"]
    with pytest.raises(ValueError, match="Unknown parameter"):
        UniformExternalField.from_dict(
            {
                **field.to_dict(),
                "parameters": {"field_functions": {"Ez": 1.0}, "amplitude": 2.0},
            }
        )


def test_expression_external_field_adds_lab_frame_contribution() -> None:
    field = ExpressionExternalField(
        field_functions={"Ez": MathExpression(expression="z + t", variables=("z", "t"))}
    )

    lab_expression, namespace = field.build_lab_field_expressions()["Ez"]
    assert not field.override
    field_function = field._build_field_function(
        lab_expression, namespace, override=field.override
    )

    assert field_function(1.0, 0.0, 0.0, 4.0, 5.0, 1.0, 2.0) == 10.0


def test_external_field_override_replaces_gathered_field() -> None:
    field = UniformExternalField(field_functions={"Ez": 2.0}, override=True)
    lab_expression, namespace = field.build_lab_field_expressions()["Ez"]
    field_function = field._build_field_function(
        lab_expression, namespace, override=field.override
    )

    assert field_function(5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0) == 2.0


@pytest.mark.parametrize("fieldtype", ["E", "Jz", "rho"])
def test_external_field_rejects_unsupported_field_types(fieldtype: str) -> None:
    with pytest.raises(ValueError, match="unsupported field"):
        UniformExternalField(field_functions={fieldtype: 1.0})


def test_expression_external_field_rejects_non_coordinate_variables() -> None:
    with pytest.raises(ValueError, match="variables must be drawn"):
        ExpressionExternalField(
            field_functions={"Ez": MathExpression(expression="r", variables=("r",))}
        )


@pytest.mark.parametrize(
    "field",
    [
        UniformExternalField(field_functions={"Ez": 1.0}),
        ExpressionExternalField(
            field_functions={"Ez": MathExpression(expression="z", variables=("z",))}
        ),
    ],
)
def test_external_field_round_trips(
    field: UniformExternalField | ExpressionExternalField,
) -> None:
    reloaded = SerializableConfig.from_json(field.to_json())

    assert type(reloaded) is type(field)
    assert reloaded.to_dict() == field.to_dict()


def test_simulation_resolves_external_field_species_and_boost(
    minimal_hyperparameters,
    minimal_density,
    minimal_laser,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.particle_beam import ExpressionParticleBeam
    from inversion_fbpic.lib.simulation import Simulation

    particle = object()
    captured: dict[str, Any] = {}
    beam = ExpressionParticleBeam(
        gamma0=1_000.0,
        n=1.0e22,
        p_zmin=-2.0e-6,
        p_zmax=2.0e-6,
        p_rmax=5.0e-6,
        particle_name="driver",
    )
    external = UniformExternalField(field_functions={"Ez": 1.0}, species="driver")
    external_fields = []

    def add_to_simulation(self, simulation, *, boost=None):
        return particle

    def build_external_fields(self, *, species, gamma_boost):
        captured["species"] = species
        captured["gamma_boost"] = gamma_boost
        return ["external"]

    monkeypatch.setattr(ExpressionParticleBeam, "add_to_simulation", add_to_simulation)
    monkeypatch.setattr(
        UniformExternalField, "build_external_fields", build_external_fields
    )
    minimal_hyperparameters = attrs.evolve(
        minimal_hyperparameters, gamma_boost=2.0, number_dumps=0
    )
    simulation = Simulation(
        elements=[
            minimal_hyperparameters,
            minimal_density,
            minimal_laser,
            beam,
            external,
        ]
    )
    backend = MagicMock(
        dt=minimal_hyperparameters.dt, external_fields=external_fields, diags=[]
    )
    monkeypatch.setattr(
        simulation_module, "FBPICSimulation", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        type(minimal_density),
        "add_to_simulation",
        MagicMock(return_value=(MagicMock(), None)),
    )
    monkeypatch.setattr(simulation_module, "add_laser_pulse", MagicMock())

    simulation.setup_simulation(working_directory=tmp_path)

    assert simulation.particle_species == {"driver": particle}
    assert captured == {"species": particle, "gamma_boost": 2.0}
    assert backend.external_fields == ["external"]


def test_external_field_targets_named_beam_with_impossible_diagnostic_selection(
    minimal_hyperparameters,
    minimal_density,
    minimal_laser,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.particle_beam import ExpressionParticleBeam
    from inversion_fbpic.lib.simulation import Simulation

    particle = MagicMock()
    captured: dict[str, Any] = {}
    driver = ExpressionParticleBeam(
        gamma0=1_000.0,
        n=1.0e22,
        p_zmin=-2.0e-6,
        p_zmax=2.0e-6,
        p_rmax=5.0e-6,
        particle_name="driver",
        particle_select={"x": [1.0e12, None]},
    )
    external = UniformExternalField(field_functions={"Ez": 1.0}, species="driver")
    simulation = Simulation(
        elements=[
            minimal_hyperparameters,
            minimal_density,
            minimal_laser,
            driver,
            external,
        ]
    )
    backend = MagicMock(dt=minimal_hyperparameters.dt, external_fields=[], diags=[])

    def build_external_fields(self, *, species, gamma_boost):
        captured["species"] = species
        return []

    particle_diagnostic = MagicMock()
    monkeypatch.setattr(
        simulation_module, "FBPICSimulation", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        type(minimal_density),
        "add_to_simulation",
        MagicMock(return_value=(MagicMock(), None)),
    )
    monkeypatch.setattr(
        ExpressionParticleBeam, "add_to_simulation", MagicMock(return_value=particle)
    )
    monkeypatch.setattr(
        UniformExternalField, "build_external_fields", build_external_fields
    )
    monkeypatch.setattr(simulation_module, "ParticleDiagnostic", particle_diagnostic)
    monkeypatch.setattr(simulation_module, "add_laser_pulse", MagicMock())

    simulation.setup_simulation(working_directory=tmp_path)

    assert captured["species"] is particle
    assert particle_diagnostic.call_count == 2
    assert all(
        call.kwargs["species"] == {"driver": particle}
        and call.kwargs["select"] == {"x": [1.0e12, None]}
        for call in particle_diagnostic.call_args_list
    )


def test_simulation_rejects_unknown_external_field_species(
    minimal_hyperparameters, minimal_density, minimal_laser, tmp_path, monkeypatch
) -> None:
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.simulation import Simulation

    external = UniformExternalField(field_functions={"Ez": 1.0}, species="unknown")
    simulation = Simulation(
        elements=[minimal_hyperparameters, minimal_density, minimal_laser, external]
    )
    backend = MagicMock(dt=minimal_hyperparameters.dt, external_fields=[], diags=[])
    monkeypatch.setattr(
        simulation_module, "FBPICSimulation", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        type(minimal_density),
        "add_to_simulation",
        MagicMock(return_value=(MagicMock(), None)),
    )
    monkeypatch.setattr(simulation_module, "add_laser_pulse", MagicMock())

    with pytest.raises(ValueError, match="not a recorded particle-species name"):
        simulation.setup_simulation(working_directory=tmp_path)
