"""
Tests for SimulationHyperparameters, Simulation assembly, error paths,
directory loading, and a minimal FBPIC integration test.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_simulation.py -v
    pytest tests/test_lib/test_simulation.py -v -m integration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import attrs
import numpy as np
import pytest
import yaml
from scipy.constants import c, e, epsilon_0, m_e, pi

HYPERPARAMETERS_KWARGS: dict = {
    "zmin": -1.0e-5,
    "zmax": 0.0,
    "rmax": 1.0e-5,
    "nz": 8,
    "nr": 8,
    "nm": 1,
    "use_mpi": False,
    "number_dumps": 2,
}

EXAMPLE_DENSITY_KWARGS: dict = {
    "nominal_density": 1.0e24,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "length": 1.0e-6,
    "start_position": 0.0,
}

LASER_BASE_KWARGS: dict = {
    "z0": -3.0e-5,
    "wavelength": 8.0e-7,
    "tau_fwhm": 3.8e-14,
    "cep": 0.0,
    "waist": 2.8e-5,
    "focal_position": 3.0e-3,
    "polarization": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hyparams(**overrides):
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    kw = {**HYPERPARAMETERS_KWARGS, **overrides}
    return SimulationHyperparameters(**kw)


def _make_simulation_elements():
    """Return (hyparams, density, laser) as in-memory objects."""
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    hyp = SimulationHyperparameters(**HYPERPARAMETERS_KWARGS)
    den = ExampleDensityProfile(**EXAMPLE_DENSITY_KWARGS)
    las = GaussianLaserPulse(energy=5.0, **LASER_BASE_KWARGS)
    return hyp, den, las


def _make_simulation():
    from inversion_fbpic.lib.simulation import Simulation

    hyp, den, las = _make_simulation_elements()
    return Simulation(elements=[hyp, den, las])


@pytest.fixture()
def analysis_classes(monkeypatch: pytest.MonkeyPatch):
    from inversion_fbpic.lib.datapoint import _Datapoint
    from inversion_fbpic.lib.diagnostics import _Diagnostic

    monkeypatch.setattr(
        _Datapoint, "_CONCRETE_REGISTRY", dict(_Datapoint._CONCRETE_REGISTRY)
    )

    @attrs.define(kw_only=True, slots=False)
    class BeforeDiagnostics(_Diagnostic):
        SUBCLASS: ClassVar[str] = "test_simulation_before"
        RUN_BEFORE_SIMULATION: ClassVar[bool] = True

        calls: int = attrs.field(default=0, init=False)
        events: list[str] = attrs.field(factory=list, init=False)

        def _analyze(self) -> dict[str, Any]:
            self.calls += 1
            self.events.append("before" if self.RUN_BEFORE_SIMULATION else "after")
            return {"calls": self.calls}

    @attrs.define(kw_only=True, slots=False)
    class AfterDiagnostics(BeforeDiagnostics):
        SUBCLASS: ClassVar[str] = "test_simulation_after"
        RUN_BEFORE_SIMULATION: ClassVar[bool] = False

    return BeforeDiagnostics, AfterDiagnostics


@pytest.fixture()
def analysis_simulation(analysis_classes, monkeypatch: pytest.MonkeyPatch):
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.config_container import ConfigContainer
    from inversion_fbpic.lib.simulation import Simulation

    before_class, after_class = analysis_classes
    events: list[str] = []
    before, after = before_class(), after_class()
    before.events = after.events = events
    hyperparameters, density, laser = _make_simulation_elements()
    simulation = Simulation(
        elements=ConfigContainer(
            configs=[
                hyperparameters,
                ConfigContainer(configs=[after, density, before, laser]),
            ]
        )
    )
    backend = MagicMock(dt=hyperparameters.dt, diags=[])
    backend.step.side_effect = lambda *args, **kwargs: events.append("step")
    backend.set_moving_window.side_effect = lambda *args, **kwargs: events.append(
        "setup"
    )
    monkeypatch.setattr(
        simulation_module, "FBPICSimulation", MagicMock(return_value=backend)
    )
    monkeypatch.setattr(
        type(density), "add_to_simulation", MagicMock(return_value=(MagicMock(), None))
    )
    monkeypatch.setattr(simulation_module, "add_laser_pulse", MagicMock())
    return simulation, before, after, events


# ===================================================================
# SimulationHyperparameters — computed properties
# ===================================================================


class TestSimulationHyperparameters:
    def test_warns_for_nonoptimal_nz(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="inversion_fbpic.lib.simulation"):
            _make_hyparams(nz=15)

        assert "largest prime factor of nz=15 is 5" in caplog.text

    def test_dz(self) -> None:
        hp = _make_hyparams()
        expected = (hp.zmax - hp.zmin) / hp.nz
        assert hp.dz == pytest.approx(expected)

    def test_dr(self) -> None:
        hp = _make_hyparams()
        expected = hp.rmax / hp.nr
        assert hp.dr == pytest.approx(expected)

    def test_dt_lab_frame(self) -> None:
        hp = _make_hyparams()
        expected = (hp.zmax - hp.zmin) / hp.nz / c
        assert hp.dt == pytest.approx(expected)

    def test_is_boosted_false(self) -> None:
        hp = _make_hyparams()
        assert not hp.is_boosted

    def test_is_boosted_true(self) -> None:
        hp = _make_hyparams(gamma_boost=4.0)
        assert hp.is_boosted

    def test_is_boosted_gamma_one(self) -> None:
        hp = _make_hyparams(gamma_boost=1.0)
        assert not hp.is_boosted

    def test_dt_boosted(self) -> None:
        hp = _make_hyparams(gamma_boost=4.0)
        dt_z = (hp.zmax - hp.zmin) / hp.nz / c
        dt_r = hp.rmax / (2 * hp.gamma_boost * hp.nr) / c
        assert hp.dt == pytest.approx(min(dt_z, dt_r))

    def test_v_comoving_lab_frame(self) -> None:
        hp = _make_hyparams()
        assert hp.v_comoving == pytest.approx(0.0)

    def test_v_comoving_boosted(self) -> None:
        hp = _make_hyparams(gamma_boost=4.0)
        expected = -c * np.sqrt(1.0 - 1.0 / 4.0**2)
        assert hp.v_comoving == pytest.approx(expected)

    def test_n_order_without_mpi(self) -> None:
        hp = _make_hyparams(use_mpi=False)
        assert hp.n_order == -1

    def test_n_order_with_mpi(self) -> None:
        hp = _make_hyparams(use_mpi=True)
        assert hp.n_order == 32

    def test_grid_parameters_yaml(self) -> None:
        hp = _make_hyparams()
        yaml_str = hp.grid_parameters_yaml()
        parsed = yaml.safe_load(yaml_str)
        assert parsed["parameters"]["nz"] == hp.nz
        assert parsed["parameters"]["nr"] == hp.nr
        assert parsed["parameters"]["dt"] == pytest.approx(hp.dt)
        assert parsed["parameters"]["dz"] == pytest.approx(hp.dz)
        assert parsed["parameters"]["dr"] == pytest.approx(hp.dr)

    def test_yaml_round_trip(self) -> None:
        hp = _make_hyparams()
        yaml_str = hp.to_yaml(comments=False)
        from inversion_fbpic.lib.simulation import SimulationHyperparameters

        reloaded = SimulationHyperparameters.from_yaml(yaml_str)
        assert reloaded.nz == hp.nz
        assert reloaded.nr == hp.nr
        assert reloaded.zmin == pytest.approx(hp.zmin)
        assert reloaded.rmax == pytest.approx(hp.rmax)


# ===================================================================
# Simulation.calculate_group_beta
# ===================================================================


class TestCalculateGroupBeta:
    def test_known_value(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        wavelength = 8e-7
        density = 1e24
        expected = (
            1.0
            - wavelength**2 * e**2 / (8.0 * pi**2 * c**2 * m_e * epsilon_0) * density
        )
        result = Simulation.calculate_group_beta(wavelength, density)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_vacuum_returns_one(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        result = Simulation.calculate_group_beta(8e-7, 0.0)
        assert result == pytest.approx(1.0)

    def test_mean_group_beta(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation, SimulationHyperparameters
        from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        hyp = SimulationHyperparameters(**HYPERPARAMETERS_KWARGS)
        den = ExampleDensityProfile(**EXAMPLE_DENSITY_KWARGS)
        las = GaussianLaserPulse(energy=5.0, **LASER_BASE_KWARGS)
        sim = Simulation(elements=[hyp, den, las])
        result = sim.calculate_mean_group_beta(8e-7)
        assert result == pytest.approx(0.999857)


# ===================================================================
# Simulation._normalize_working_directory
# ===================================================================


class TestNormalizeWorkingDirectory:
    def test_none_returns_cwd(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        result = Simulation._normalize_working_directory(None)
        assert result == Path.cwd()

    def test_relative_path_resolved(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        result = Simulation._normalize_working_directory(tmp_path / "sub")
        assert result.is_absolute()

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        abs_path = tmp_path / "abs_dir"
        result = Simulation._normalize_working_directory(abs_path)
        assert result == abs_path.resolve()


# ===================================================================
# Simulation diagnostic periods
# ===================================================================


class TestDiagnosticPeriods:
    @pytest.mark.parametrize(
        ("num_steps", "number_dumps", "expected_period"),
        [
            (100, 4, 33),
            (100, 11, 10),
            (100, 21, 5),
            (100, 26, 4),
            (100, 1, 100),
        ],
    )
    def test_particle_diagnostic_period_does_not_exceed_dump_limit(
        self, num_steps: int, number_dumps: int, expected_period: int
    ) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        period = Simulation._particle_diagnostic_period_steps(num_steps, number_dumps)

        assert period == expected_period
        assert (num_steps - 1) // period + 1 <= number_dumps


# ===================================================================
# Simulation assembly and error paths
# ===================================================================


class TestSimulationAssembly:
    def test_from_objects(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim = Simulation(elements=[hyp, den, las])
        assert sim.hyparams is hyp
        assert len(sim.densities) == 1
        assert len(sim.lasers) == 1

    def test_from_dicts(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim = Simulation(elements=[hyp.to_dict(), den.to_dict(), las.to_dict()])
        assert sim.hyparams is not None
        assert len(sim.densities) == 1
        assert len(sim.lasers) == 1

    def test_missing_hyperparameters_raises(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        _, den, las = _make_simulation_elements()
        with pytest.raises(ValueError, match="No SimulationHyperparameters"):
            Simulation(elements=[den, las])

    def test_missing_density_raises(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, _, las = _make_simulation_elements()
        with pytest.raises(ValueError, match="No DensityProfile"):
            Simulation(elements=[hyp, las])

    def test_missing_laser_raises(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, _ = _make_simulation_elements()
        with pytest.raises(ValueError, match="No LaserPulse"):
            Simulation(elements=[hyp, den])

    def test_duplicate_hyperparameters_raises(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        hyp2 = _make_hyparams()
        with pytest.raises(ValueError, match="Multiple SimulationHyperparameters"):
            Simulation(elements=[hyp, hyp2, den, las])

    def test_unsupported_element_type_raises(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        with pytest.raises(ValueError, match="Unsupported input type"):
            Simulation(elements=[hyp, den, las, 42])

    def test_to_dict_round_trip(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim = Simulation(elements=[hyp, den, las])
        d = sim.to_dict()
        assert d["config_type"] == "simulation"
        assert d["subclass"] == "simulation"

    def test_run_simulation_before_setup_raises(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim = Simulation(elements=[hyp, den, las])
        with pytest.raises(ValueError, match="Simulation not setup"):
            sim.run_simulation()


class TestSupportingElements:
    def test_nested_containers_and_parameters_preserve_references(self) -> None:
        from inversion_fbpic.lib.config_container import ConfigContainer
        from inversion_fbpic.lib.datapoint import Parameters
        from inversion_fbpic.lib.simulation import Simulation

        hyperparameters, density, laser = _make_simulation_elements()
        data = {"config_type": "not_a_config", "label": "scan"}
        parameters = Parameters(data=data)
        container = ConfigContainer(
            configs=[
                hyperparameters,
                ConfigContainer(configs=[parameters, density, laser]),
            ]
        )
        simulation = Simulation(elements=container)
        assert simulation.elements is container
        assert simulation.hyparams is hyperparameters
        assert simulation.densities == [density]
        assert simulation.lasers == [laser]
        assert simulation.parameters[0] is parameters
        data["label"] = "changed"
        assert simulation.parameters[0].data["label"] == "changed"
        assert simulation.config_hash() == _make_simulation().config_hash()

    @pytest.mark.parametrize("method", ["from_dict", "from_json", "from_yaml"])
    def test_nested_elements_round_trip(self, analysis_classes, method: str) -> None:
        from inversion_fbpic.lib.config_container import ConfigContainer
        from inversion_fbpic.lib.datapoint import Parameters
        from inversion_fbpic.lib.serializable_config import SerializableConfig
        from inversion_fbpic.lib.simulation import Simulation

        before_class, after_class = analysis_classes
        before, after = before_class(), after_class()
        before.analyze()
        after.analyze()
        hyperparameters, density, laser = _make_simulation_elements()
        original = Simulation(
            elements=ConfigContainer(
                configs=[
                    hyperparameters,
                    ConfigContainer(
                        configs=[density, Parameters(data={"energy": 5.0}), before]
                    ),
                    laser,
                    after,
                ]
            )
        )
        payload = getattr(original, method.replace("from_", "to_"))()
        loaded = getattr(SerializableConfig, method)(payload)
        assert isinstance(loaded, Simulation)
        assert isinstance(loaded.elements, ConfigContainer)
        assert loaded.parameters[0].data == {"energy": 5.0}
        assert [type(item) for item in loaded.diagnostics] == [
            before_class,
            after_class,
        ]
        assert [item.RUN_BEFORE_SIMULATION for item in loaded.diagnostics] == [
            True,
            False,
        ]
        assert all(not item.analysis_complete for item in loaded.diagnostics)
        assert all(item.data == {"calls": 1} for item in loaded.diagnostics)
        assert loaded.config_hash() == original.config_hash()
        assert set(loaded.to_dict()["parameters"]) == {"elements", "verbosity"}
        assert "RUN_BEFORE_SIMULATION" not in before.to_dict()["parameters"]
        loaded.parameters[0].data["energy"] = 7.0
        loaded.diagnostics[0].analyze()
        loaded.diagnostics[0].analyze()
        saved = SerializableConfig.from_dict(loaded.to_dict())
        assert isinstance(saved, Simulation)
        assert saved.parameters[0].data == {"energy": 7.0}
        assert saved.diagnostics[0].data == {"calls": 2}

    def test_directory_accepts_parameter_and_container_files(
        self, tmp_path: Path
    ) -> None:
        from inversion_fbpic.lib.config_container import ConfigContainer
        from inversion_fbpic.lib.datapoint import Parameters
        from inversion_fbpic.lib.simulation import Simulation

        hyperparameters, density, laser = _make_simulation_elements()
        ConfigContainer(
            configs=[hyperparameters, ConfigContainer(configs=[density, laser])]
        ).to_yaml_file(tmp_path / "components.yaml")
        Parameters(data={"label": "scan"}).to_json_file(tmp_path / "parameters.json")
        simulation = Simulation(elements=tmp_path)
        assert simulation.hyparams is not None
        assert len(simulation.densities) == len(simulation.lasers) == 1
        assert simulation.parameters[0].data == {"label": "scan"}

    def test_nested_duplicate_hyperparameters_still_raise(self) -> None:
        from inversion_fbpic.lib.config_container import ConfigContainer
        from inversion_fbpic.lib.simulation import Simulation

        hyperparameters, density, laser = _make_simulation_elements()
        container = ConfigContainer(
            configs=[
                hyperparameters,
                density,
                laser,
                ConfigContainer(configs=[hyperparameters]),
            ]
        )
        with pytest.raises(ValueError, match="Multiple SimulationHyperparameters"):
            Simulation(elements=container)

    def test_parameters_do_not_replace_required_components(self) -> None:
        from inversion_fbpic.lib.datapoint import Parameters
        from inversion_fbpic.lib.simulation import Simulation

        with pytest.raises(ValueError, match="No SimulationHyperparameters"):
            Simulation(elements=Parameters(data={"label": "scan"}))

    def test_nested_relative_references_from_another_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyperparameters, density, laser = _make_simulation_elements()
        cfg = tmp_path / "cfg"
        hyperparameters.to_json_file(cfg / "hyperparameters.json")
        density.to_yaml_file(cfg / "nested" / "density.yaml")
        laser.to_json_file(cfg / "nested" / "laser.json")
        container = {
            "config_type": "config_container",
            "subclass": "config_container",
            "parameters": {"configs": ["density.yaml", "laser.json"]},
        }
        (cfg / "nested" / "container.yaml").write_text(
            yaml.safe_dump(container), encoding="utf-8"
        )
        outer = {
            "config_type": "simulation",
            "subclass": "simulation",
            "parameters": {
                "elements": ["hyperparameters.json", "nested/container.yaml"]
            },
        }
        simulation_file = cfg / "simulation.yaml"
        simulation_file.write_text(yaml.safe_dump(outer), encoding="utf-8")
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        loaded = Simulation.from_file(simulation_file)
        assert isinstance(loaded, Simulation)
        assert len(loaded.densities) == len(loaded.lasers) == 1
        assert (
            loaded.densities[0].source_file
            == (cfg / "nested" / "density.yaml").resolve()
        )
        assert loaded.lasers[0].source_file == (cfg / "nested" / "laser.json").resolve()

    def test_collecting_diagnostics_does_not_serialize_or_analyze(
        self, analysis_classes, caplog
    ) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        before_class, after_class = analysis_classes
        before, after = before_class(), after_class()
        with caplog.at_level(logging.DEBUG):
            simulation = Simulation(
                elements=[*_make_simulation_elements(), before, after],
                verbosity=logging.DEBUG,
            )
        assert simulation.diagnostics[0] is before
        assert simulation.diagnostics[1] is after
        assert before.calls == after.calls == 0
        assert not any(
            record.name == "inversion_fbpic.lib.diagnostics"
            for record in caplog.records
        )

    def test_invalid_analysis_phase_is_rejected(
        self, analysis_classes, monkeypatch
    ) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        before_class, _ = analysis_classes
        monkeypatch.setattr(before_class, "RUN_BEFORE_SIMULATION", "before")
        with pytest.raises(ValueError, match="RUN_BEFORE_SIMULATION must be a bool"):
            Simulation(elements=[*_make_simulation_elements(), before_class()])


class TestAttachedAnalysis:
    def test_moment_descriptor_runs_after_stepping(
        self, analysis_simulation, tmp_path, monkeypatch
    ) -> None:
        from inversion_fbpic.lib import diagnostics as module
        from inversion_fbpic.lib import simulation as simulation_module
        from inversion_fbpic.lib.diagnostics import MomentDescriptorDiagnostic
        from inversion_fbpic.utils.distributions import compute_moment_descriptor

        simulation, _, _, events = analysis_simulation
        diagnostic = MomentDescriptorDiagnostic(selection=("elec_name", "electrons"))
        simulation.densities = [
            attrs.evolve(simulation.densities[0], elec_name="electrons")
        ]
        monkeypatch.setattr(simulation_module, "ParticleDiagnostic", MagicMock())
        simulation._sort_component(diagnostic)
        simulation.setup_simulation(working_directory=tmp_path)
        assert not diagnostic.analysis_complete
        particles = np.random.default_rng(9).normal(size=(128, 6))
        weights = np.ones(len(particles))
        directory = simulation._save_directory / "hdf5"
        directory.mkdir(parents=True)
        latest = directory / "data00000010.h5"

        def finish_step(*args, **kwargs):
            events.append("step")
            latest.touch()

        loader = MagicMock(return_value=(particles, weights))
        monkeypatch.setattr(module, "load_openpmd_particles", loader)
        simulation.simulation.step.side_effect = finish_step
        simulation.run_simulation(record_hash=False)
        loader.assert_called_once_with(latest, "electrons")
        assert diagnostic.data == compute_moment_descriptor(particles, weights)
        assert diagnostic.analysis_complete

    @pytest.mark.parametrize("logger_friendly", [False, True])
    def test_phase_order_and_hash_recording(
        self, analysis_simulation, tmp_path: Path, monkeypatch, logger_friendly: bool
    ) -> None:
        simulation, before, after, events = analysis_simulation
        monkeypatch.setattr(simulation, "_save_hash", lambda: events.append("hash"))
        simulation.setup_simulation(working_directory=tmp_path)
        assert events == ["setup", "before"]
        assert before.analysis_complete and not after.analysis_complete
        simulation.run_simulation(
            show_progress=False, logger_friendly_progress=logger_friendly
        )
        assert events == ["setup", "before", "step", "hash", "after"]
        assert before.calls == after.calls == 1
        assert after.analysis_complete

    def test_post_analysis_runs_once_after_all_progress_chunks(
        self, analysis_simulation, tmp_path
    ) -> None:
        simulation, before, after, events = analysis_simulation
        simulation.setup_simulation(working_directory=tmp_path)
        simulation.T_interact = simulation.simulation.dt * 600
        simulation.run_simulation(logger_friendly_progress=True, record_hash=False)
        assert events[0:2] == ["setup", "before"]
        assert events[-1] == "after"
        assert events.count("step") == 100
        assert events.count("after") == 1
        assert before.calls == after.calls == 1
        assert (
            sum(call.args[0] for call in simulation.simulation.step.call_args_list)
            == simulation.num_steps
        )

    def test_setup_is_idempotent_and_explicit_runs_rerun_analysis(
        self, analysis_simulation, tmp_path
    ) -> None:
        simulation, before, after, _ = analysis_simulation
        simulation.setup_simulation(working_directory=tmp_path)
        simulation.setup_simulation(working_directory=tmp_path)
        assert before.calls == 1
        simulation.run_simulation(record_hash=False)
        simulation.run_simulation(record_hash=False)
        assert after.calls == 2

    @pytest.mark.parametrize("phase", ["setup", "run"])
    def test_hashed_phase_reruns_analysis_without_simulation_work(
        self, analysis_simulation, tmp_path, monkeypatch, phase
    ) -> None:
        simulation, before, after, events = analysis_simulation
        monkeypatch.setattr(simulation, "_is_hashed", lambda: True)
        if phase == "setup":
            simulation.setup_simulation(working_directory=tmp_path)
        else:
            simulation.run_simulation()
        assert events == ["before" if phase == "setup" else "after"]
        assert before.calls == (1 if phase == "setup" else 0)
        assert after.calls == (1 if phase == "run" else 0)
        assert not simulation.is_setup

    def test_disabled_skip_runs_analysis(
        self, analysis_simulation, tmp_path, monkeypatch
    ) -> None:
        simulation, before, after, _ = analysis_simulation
        monkeypatch.setattr(simulation, "_is_hashed", lambda: True)
        simulation.setup_simulation(working_directory=tmp_path, skip_if_hashed=False)
        simulation.run_simulation(skip_if_hashed=False, record_hash=False)
        assert before.calls == after.calls == 1

    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize("cached", [False, True])
    def test_only_mpi_write_rank_analyzes(
        self, analysis_simulation, tmp_path, monkeypatch, rank, cached
    ) -> None:
        from inversion_fbpic.lib import simulation as simulation_module

        simulation, before, after, _ = analysis_simulation
        simulation.hyparams = _make_hyparams(use_mpi=True)
        monkeypatch.setattr(simulation_module, "MPI_RANK", rank)
        if cached:
            monkeypatch.setattr(simulation, "_is_hashed", lambda: True)
        simulation.setup_simulation(working_directory=tmp_path)
        simulation.run_simulation(record_hash=False)
        assert before.calls == after.calls == (1 if rank == 0 else 0)

    @pytest.mark.parametrize("phase", ["before", "after"])
    def test_analysis_failure_preserves_successful_stepping_hash(
        self, analysis_simulation, tmp_path, monkeypatch, phase
    ) -> None:
        simulation, before, after, _ = analysis_simulation
        diagnostic = before if phase == "before" else after
        save_hash = MagicMock()
        monkeypatch.setattr(simulation, "_save_hash", save_hash)

        def fail():
            raise RuntimeError("analysis failed")

        monkeypatch.setattr(diagnostic, "_analyze", fail)
        if phase == "before":
            with pytest.raises(RuntimeError, match="analysis failed"):
                simulation.setup_simulation(working_directory=tmp_path)
            assert not simulation.is_setup
        else:
            simulation.setup_simulation(working_directory=tmp_path)
            with pytest.raises(RuntimeError, match="analysis failed"):
                simulation.run_simulation()
        assert not diagnostic.analysis_complete
        if phase == "before":
            save_hash.assert_not_called()
        else:
            save_hash.assert_called_once_with()

    def test_failed_post_analysis_retry_does_not_repeat_simulation(
        self, analysis_simulation, analysis_classes, tmp_path, monkeypatch
    ) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        simulation, _, after, events = analysis_simulation
        simulation.setup_simulation(working_directory=tmp_path)

        def fail():
            assert simulation._is_hashed()
            raise RuntimeError("post-analysis failed")

        monkeypatch.setattr(after, "_analyze", fail)
        with pytest.raises(RuntimeError, match="post-analysis failed"):
            simulation.run_simulation()
        assert simulation._hash_path.is_file()
        assert simulation._is_hashed()
        assert events.count("step") == 1
        backend = simulation.simulation
        backend.step.reset_mock()
        before_class, after_class = analysis_classes
        before_retry, after_retry = before_class(), after_class()
        retry = Simulation(
            elements=[
                simulation.hyparams,
                *simulation.densities,
                *simulation.lasers,
                before_retry,
                after_retry,
            ]
        )
        retry.setup_simulation(working_directory=tmp_path)
        assert not retry.is_setup
        assert before_retry.analysis_complete
        assert not hasattr(retry, "simulation")
        retry.run_simulation()
        backend.step.assert_not_called()
        assert after_retry.analysis_complete
        assert after_retry.calls == 1

    def test_disabling_hash_recording_does_not_create_hash_on_analysis_failure(
        self, analysis_simulation, tmp_path, monkeypatch
    ) -> None:
        simulation, _, after, _ = analysis_simulation
        simulation.setup_simulation(working_directory=tmp_path)
        monkeypatch.setattr(
            after, "_analyze", MagicMock(side_effect=RuntimeError("analysis failed"))
        )
        with pytest.raises(RuntimeError, match="analysis failed"):
            simulation.run_simulation(record_hash=False)
        assert not simulation._hash_path.exists()

    def test_hash_save_failure_does_not_start_analysis(
        self, analysis_simulation, tmp_path, monkeypatch
    ) -> None:
        simulation, _, after, _ = analysis_simulation
        simulation.setup_simulation(working_directory=tmp_path)
        monkeypatch.setattr(
            simulation, "_save_hash", MagicMock(side_effect=OSError("write failed"))
        )
        with pytest.raises(OSError, match="write failed"):
            simulation.run_simulation()
        assert after.calls == 0

    def test_failed_stepping_does_not_analyze(
        self, analysis_simulation, tmp_path
    ) -> None:
        simulation, _, after, _ = analysis_simulation
        simulation.setup_simulation(working_directory=tmp_path)
        simulation.simulation.step.side_effect = RuntimeError("step failed")
        with pytest.raises(RuntimeError, match="step failed"):
            simulation.run_simulation(record_hash=False)
        assert after.calls == 0

    def test_depth_first_order_within_phase(
        self, analysis_simulation, analysis_classes, tmp_path
    ) -> None:
        simulation, before, after, events = analysis_simulation
        before_class, _ = analysis_classes
        second = before_class()
        second.events = events
        simulation._sort_component(second)
        simulation.setup_simulation(working_directory=tmp_path)
        assert events == ["setup", "before", "before"]
        assert simulation.diagnostics == [after, before, second]


# ===================================================================
# Directory-based loading
# ===================================================================


class TestDirectoryLoading:
    def test_load_from_yaml_directory(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        hyp.to_yaml_file(cfg_dir / "hyperparameters.yaml", comments=False)
        den.to_yaml_file(cfg_dir / "density.yaml", comments=False)
        las.to_yaml_file(cfg_dir / "laser.yaml", comments=False)

        sim = Simulation(elements=str(cfg_dir))
        assert sim.hyparams is not None
        assert len(sim.densities) == 1
        assert len(sim.lasers) == 1

    def test_load_from_single_yaml(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        hyp.to_yaml_file(cfg_dir / "hyperparameters.yaml", comments=False)
        den.to_yaml_file(cfg_dir / "density.yaml", comments=False)
        las.to_yaml_file(cfg_dir / "laser.yaml", comments=False)

        sim = Simulation(
            elements=[
                str(cfg_dir / "hyperparameters.yaml"),
                str(cfg_dir / "density.yaml"),
                str(cfg_dir / "laser.yaml"),
            ]
        )
        assert sim.hyparams is not None
        assert len(sim.densities) == 1
        assert len(sim.lasers) == 1


# ===================================================================
# Minimal integration test
# ===================================================================


# ===================================================================
# config_hash
# ===================================================================


class TestConfigHash:
    def test_analysis_results_do_not_change_simulation_hash(
        self, analysis_simulation
    ) -> None:
        simulation, before, after, _ = analysis_simulation
        original = simulation.config_hash()
        before.analyze()
        after.analyze()
        assert simulation.config_hash() == original
        before.data["arbitrary_result"] = 123.0
        after.data = {"different_result": -1.0}
        assert simulation.config_hash() == original

    def test_same_config_same_hash(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim1 = Simulation(elements=[hyp, den, las])
        sim2 = Simulation(elements=[hyp, den, las])
        assert sim1.config_hash() == sim2.config_hash()

    def test_element_order_independent(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim1 = Simulation(elements=[hyp, den, las])
        sim2 = Simulation(elements=[las, den, hyp])
        assert sim1.config_hash() == sim2.config_hash()

    def test_dict_vs_object_input(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim_obj = Simulation(elements=[hyp, den, las])
        sim_dict = Simulation(elements=[hyp.to_dict(), den.to_dict(), las.to_dict()])
        assert sim_obj.config_hash() == sim_dict.config_hash()

    def test_different_config_different_hash(self) -> None:
        from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim1 = Simulation(elements=[hyp, den, las])
        den2 = ExampleDensityProfile(
            **{**EXAMPLE_DENSITY_KWARGS, "nominal_density": 2.0e24}
        )
        sim2 = Simulation(elements=[hyp, den2, las])
        assert sim1.config_hash() != sim2.config_hash()

    def test_verbosity_not_in_hash(self) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim1 = Simulation(elements=[hyp, den, las], verbosity=10)
        sim2 = Simulation(elements=[hyp, den, las], verbosity=40)
        assert sim1.config_hash() == sim2.config_hash()

    def test_multiple_densities_order_independent(self) -> None:
        from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den1, las = _make_simulation_elements()
        den2 = ExampleDensityProfile(
            **{**EXAMPLE_DENSITY_KWARGS, "nominal_density": 2.0e24}
        )
        sim1 = Simulation(elements=[hyp, den1, den2, las])
        sim2 = Simulation(elements=[hyp, den2, den1, las])
        assert sim1.config_hash() == sim2.config_hash()

    def test_yaml_vs_object_hash(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim_obj = Simulation(elements=[hyp, den, las])

        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        hyp.to_yaml_file(cfg_dir / "hyperparameters.yaml", comments=False)
        den.to_yaml_file(cfg_dir / "density.yaml", comments=False)
        las.to_yaml_file(cfg_dir / "laser.yaml", comments=False)
        sim_yaml = Simulation(elements=str(cfg_dir))
        assert sim_obj.config_hash() == sim_yaml.config_hash()


# ===================================================================
# Hash persistence and skip_if_hashed
# ===================================================================


class TestHashPersistence:
    def test_is_hashed_false_when_no_file(self, tmp_path: Path) -> None:

        sim = _make_simulation()
        sim.working_directory = tmp_path
        assert not sim._is_hashed()

    def test_save_hash_writes_config_hash(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import HASH_FILE

        sim = _make_simulation()
        sim.working_directory = tmp_path
        sim._save_hash()
        assert (
            tmp_path / sim.hyparams.save_directory / HASH_FILE
        ).read_text() == sim.config_hash()
        assert sim._is_hashed()

    def test_is_hashed_false_when_hash_differs(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import HASH_FILE

        sim = _make_simulation()
        sim.working_directory = tmp_path
        hash_path = tmp_path / sim.hyparams.save_directory / HASH_FILE
        hash_path.parent.mkdir()
        hash_path.write_text("not-a-matching-hash")
        assert not sim._is_hashed()

    def test_hash_uses_absolute_save_directory(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import HASH_FILE
        from inversion_fbpic.lib.simulation import Simulation

        _, density, laser = _make_simulation_elements()
        hyperparameters = _make_hyparams(save_directory=tmp_path / "external-diags")
        sim = Simulation(elements=[hyperparameters, density, laser])
        sim.working_directory = tmp_path / "run"
        sim._save_hash()
        assert (
            tmp_path / "external-diags" / HASH_FILE
        ).read_text() == sim.config_hash()


@pytest.mark.integration
class TestSkipIfHashed:
    def test_run_records_hash_on_completion(self, tmp_path: Path) -> None:
        from inversion_fbpic.lib.simulation import HASH_FILE

        sim = _make_simulation()
        sim.setup_simulation(working_directory=tmp_path)
        sim.run_simulation(show_progress=False)
        assert (
            tmp_path / sim.hyparams.save_directory / HASH_FILE
        ).read_text() == sim.config_hash()

    def test_run_skips_when_hashed(self, tmp_path: Path) -> None:
        sim = _make_simulation()
        sim.setup_simulation(working_directory=tmp_path)
        sim.run_simulation(show_progress=False)

        mock_step = MagicMock()
        sim.simulation.step = mock_step
        sim.run_simulation(show_progress=False, skip_if_hashed=True)
        mock_step.assert_not_called()

    def test_run_executes_when_skip_disabled(self, tmp_path: Path) -> None:
        sim = _make_simulation()
        sim.setup_simulation(working_directory=tmp_path)
        sim.run_simulation(show_progress=False)

        mock_step = MagicMock()
        sim.simulation.step = mock_step
        sim.run_simulation(show_progress=False, skip_if_hashed=False, record_hash=False)
        mock_step.assert_called_once()

    def test_setup_skips_when_hashed(self, tmp_path: Path) -> None:
        sim1 = _make_simulation()
        sim1.setup_simulation(working_directory=tmp_path)
        sim1.run_simulation(show_progress=False)

        sim2 = _make_simulation()
        sim2.setup_simulation(working_directory=tmp_path, skip_if_hashed=True)
        assert not sim2.is_setup

    def test_setup_executes_when_skip_disabled(self, tmp_path: Path) -> None:
        sim1 = _make_simulation()
        sim1.setup_simulation(working_directory=tmp_path)
        sim1.run_simulation(show_progress=False)

        sim2 = _make_simulation()
        sim2.setup_simulation(working_directory=tmp_path, skip_if_hashed=False)
        assert sim2.is_setup

    def test_run_skip_precedes_setup_prerequisite(self, tmp_path: Path) -> None:
        """A matching hash skips ``run_simulation`` before setup is required."""
        sim1 = _make_simulation()
        sim1.setup_simulation(working_directory=tmp_path)
        sim1.run_simulation(show_progress=False)

        sim2 = _make_simulation()
        sim2.setup_simulation(working_directory=tmp_path, skip_if_hashed=True)
        assert not sim2.is_setup
        sim2.run_simulation(skip_if_hashed=True)

    def test_run_skip_independent_of_setup_skip(self, tmp_path: Path) -> None:
        """After forcing setup, run can still skip independently via ``skip_if_hashed``."""
        sim1 = _make_simulation()
        sim1.setup_simulation(working_directory=tmp_path)
        sim1.run_simulation(show_progress=False)

        sim2 = _make_simulation()
        sim2.setup_simulation(working_directory=tmp_path, skip_if_hashed=False)
        assert sim2.is_setup

        mock_step = MagicMock()
        sim2.simulation.step = mock_step
        sim2.run_simulation(show_progress=False, skip_if_hashed=True)
        mock_step.assert_not_called()


@pytest.mark.integration
class TestIntegration:
    def test_setup_and_step(self, tmp_path: Path) -> None:
        """Set up a tiny simulation (8x8, nm=1) and step 3 times."""
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim = Simulation(elements=[hyp, den, las])
        sim.setup_simulation(working_directory=tmp_path)
        assert sim.is_setup
        sim.simulation.step(3, show_progress=False)

    def test_setup_is_idempotent(self, tmp_path: Path) -> None:
        """Calling setup_simulation twice should not raise."""
        from inversion_fbpic.lib.simulation import Simulation

        hyp, den, las = _make_simulation_elements()
        sim = Simulation(elements=[hyp, den, las])
        sim.setup_simulation(working_directory=tmp_path)
        sim.setup_simulation(working_directory=tmp_path)
        assert sim.is_setup
