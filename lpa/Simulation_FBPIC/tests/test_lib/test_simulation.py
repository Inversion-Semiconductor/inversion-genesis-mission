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
from unittest.mock import MagicMock

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
