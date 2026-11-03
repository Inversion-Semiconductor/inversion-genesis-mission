"""Tests for serializable FBPIC particle-beam configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pytest

from inversion_fbpic.lib.math_expression import MathExpression
from inversion_fbpic.lib.particle_beam import (
    FileParticleBeam,
    GaussianParticleBeam,
    OpenPMDParticleBeam,
    ExpressionParticleBeam,
)
from inversion_fbpic.lib.serializable_config import SerializableConfig


@pytest.fixture
def direct_beam() -> ExpressionParticleBeam:
    return ExpressionParticleBeam(
        gamma0=100.0,
        n=1.0e22,
        p_zmin=-2.0e-6,
        p_zmax=2.0e-6,
        p_rmax=5.0e-6,
        dens_func=MathExpression(expression="1 + z", variables=("z",)),
    )


def test_particle_beam_forwards_grid_initializer(
    monkeypatch: pytest.MonkeyPatch, direct_beam: ExpressionParticleBeam
) -> None:
    captured: dict[str, Any] = {}
    expected = object()

    def fake_add(simulation, **kwargs):
        captured["simulation"] = simulation
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "inversion_fbpic.lib.particle_beam.add_particle_bunch", fake_add
    )

    assert direct_beam.add_to_simulation("simulation", boost="boost") is expected
    assert captured["simulation"] == "simulation"
    assert captured["boost"] == "boost"
    assert captured["gamma0"] == 100.0
    np.testing.assert_array_equal(
        captured["dens_func"](np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        np.array([2.0, 3.0]),
    )


@pytest.mark.parametrize(
    ("beam", "initializer_name", "expected"),
    [
        (
            GaussianParticleBeam(
                sig_r=2.0e-6,
                sig_z=3.0e-6,
                n_emit=1.0e-6,
                gamma0=100.0,
                sig_gamma=0.1,
                n_physical_particles=1.0e8,
                n_macroparticles=1000,
                symmetrize=True,
            ),
            "add_particle_bunch_gaussian",
            {"sig_r": 2.0e-6, "symmetrize": True},
        ),
        (
            OpenPMDParticleBeam(
                ts_path=Path("beam/openpmd"),
                species="electrons",
                select={"z": [-1.0e-6, 1.0e-6]},
                iteration=42,
            ),
            "add_particle_bunch_openPMD",
            {"ts_path": "beam/openpmd", "iteration": 42},
        ),
        (
            FileParticleBeam(
                filename=Path("beam/distribution.npz"),
                n_physical_particles=1.0e8,
                direction="backward",
            ),
            "add_particle_bunch_file",
            {"filename": "beam/distribution.npz", "direction": "backward"},
        ),
    ],
)
def test_particle_beam_forwards_other_fbpci_initializers(
    monkeypatch: pytest.MonkeyPatch,
    beam: GaussianParticleBeam | OpenPMDParticleBeam | FileParticleBeam,
    initializer_name: str,
    expected: dict[str, Any],
) -> None:
    captured: dict[str, Any] = {}

    def fake_add(simulation, **kwargs):
        captured["simulation"] = simulation
        captured.update(kwargs)
        return "particles"

    monkeypatch.setattr(
        f"inversion_fbpic.lib.particle_beam.{initializer_name}", fake_add
    )

    assert beam.add_to_simulation("simulation", boost="boost") == "particles"
    assert captured["simulation"] == "simulation"
    assert captured["boost"] == "boost"
    assert expected.items() <= captured.items()


@pytest.mark.parametrize(
    "beam",
    [
        ExpressionParticleBeam(
            gamma0=100.0,
            n=1.0e22,
            p_zmin=-2.0e-6,
            p_zmax=2.0e-6,
            p_rmax=5.0e-6,
        ),
        GaussianParticleBeam(
            sig_r=2.0e-6,
            sig_z=3.0e-6,
            n_emit=1.0e-6,
            gamma0=100.0,
            sig_gamma=0.1,
            n_physical_particles=1.0e8,
            n_macroparticles=1000,
        ),
        OpenPMDParticleBeam(ts_path=Path("beam/openpmd")),
        FileParticleBeam(
            filename=Path("beam/distribution.npz"), n_physical_particles=1.0e8
        ),
    ],
)
def test_particle_beam_round_trips(
    beam: ExpressionParticleBeam, tmp_path: Path
) -> None:
    if isinstance(beam, OpenPMDParticleBeam):
        path = tmp_path / "openpmd"
        path.mkdir()
        beam = attrs.evolve(beam, ts_path=path)
    elif isinstance(beam, FileParticleBeam):
        path = tmp_path / "distribution.npz"
        path.touch()
        beam = attrs.evolve(beam, filename=path)

    reloaded = SerializableConfig.from_json(beam.to_json())

    assert type(reloaded) is type(beam)
    assert reloaded.to_dict() == beam.to_dict()


def test_simulation_sorts_particle_beams(
    minimal_hyperparameters,
    minimal_density,
    minimal_laser,
    direct_beam: ExpressionParticleBeam,
) -> None:
    from inversion_fbpic.lib.simulation import Simulation

    simulation = Simulation(
        elements=[minimal_hyperparameters, minimal_density, minimal_laser, direct_beam]
    )

    assert simulation.beams == [direct_beam]


def test_simulation_adds_particle_beams(
    minimal_hyperparameters,
    minimal_density,
    minimal_laser,
    direct_beam: ExpressionParticleBeam,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.simulation import Simulation

    added: list[tuple[object, object]] = []
    particle = object()
    diagnostics: list[dict[str, Any]] = []

    def add_to_simulation(self, fbpic_simulation, *, boost=None):
        added.append((fbpic_simulation, boost))
        return particle

    def particle_diagnostic(**kwargs):
        diagnostics.append(kwargs)
        return kwargs

    monkeypatch.setattr(ExpressionParticleBeam, "add_to_simulation", add_to_simulation)
    monkeypatch.setattr(simulation_module, "ParticleDiagnostic", particle_diagnostic)
    direct_beam = attrs.evolve(
        direct_beam,
        particle_name="injected_electrons",
        particle_select={"uz": [10.0, None]},
    )
    simulation = Simulation(
        elements=[minimal_hyperparameters, minimal_density, minimal_laser, direct_beam]
    )

    simulation.setup_simulation(working_directory=tmp_path)

    assert added == [(simulation.simulation, None)]
    assert len(diagnostics) == 2
    assert all(
        diagnostic["species"] == {"injected_electrons": particle}
        and diagnostic["select"] == {"uz": [10.0, None]}
        for diagnostic in diagnostics
    )


@pytest.mark.parametrize("variables", [("x",), ("z", "x")])
def test_particle_beam_rejects_unrelated_density_variables(
    variables: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="drawn"):
        ExpressionParticleBeam(
            gamma0=100.0,
            n=1.0e22,
            p_zmin=-2.0e-6,
            p_zmax=2.0e-6,
            p_rmax=5.0e-6,
            dens_func=MathExpression(expression=variables[0], variables=variables),
        )
