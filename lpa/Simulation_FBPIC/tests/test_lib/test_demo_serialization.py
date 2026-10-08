"""Density and laser demos support YAML, JSON, and HDF5 generation and plotting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from unittest.mock import Mock

matplotlib.use("Agg")

from inversion_fbpic.lib.serializable_config import SerializableConfig

DEMOS = Path(__file__).resolve().parents[2] / "demos"


def _load_demo(directory: str, script: str):
    spec = importlib.util.spec_from_file_location(
        script, DEMOS / directory / f"{script}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("domain,count", [("density", 6), ("laser", 5)])
@pytest.mark.parametrize("serialization_format", ["yaml", "json", "hdf5"])
def test_demo_generation_and_plotting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: str,
    count: int,
    serialization_format: str,
) -> None:
    directory = "demo_densities" if domain == "density" else "demo_lasers"
    creator = _load_demo(directory, f"create_{domain}_configs")
    plotter = _load_demo(directory, f"plot_{domain}_configs")
    output = tmp_path / "cfg"
    monkeypatch.setattr(creator, "CFG_DIR", output)
    monkeypatch.setattr(plotter, "CFG_DIR", output)
    monkeypatch.setattr(plotter, "PLOTS_DIR", tmp_path / "plots")
    monkeypatch.setattr(sys, "argv", ["demo", "--format", serialization_format])
    creator.main()
    creator.main()  # HDF5 configs can safely be regenerated.
    create = getattr(creator, f"create_{domain}_config_files")
    paths = create(output, serialization_format=serialization_format)
    assert len(paths) == count
    suffix = {"yaml": ".yaml", "json": ".json", "hdf5": ".h5"}[serialization_format]
    assert all(path.suffix == suffix for path in paths.values())
    for path in paths.values():
        assert SerializableConfig.from_file(path).CONFIG_TYPE in (
            "density_profile",
            "laser_pulse",
        )
    # Plot after changing cwd to also verify linked density-file resolution.
    monkeypatch.chdir(tmp_path)
    plotter.main()
    plots = tmp_path / "plots"
    if serialization_format != "yaml":
        plots /= serialization_format
    assert (plots / "combined.png").is_file()
    assert len(list(plots.glob("*.png"))) == count + 1


@pytest.mark.parametrize("skip_run", [False, True])
def test_ionization_demo_exports_physical_inputs_and_beam_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skip_run: bool
) -> None:
    import matplotlib.pyplot as plt
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.config_container import ConfigContainer
    from inversion_fbpic.lib.datapoint import Parameters
    from inversion_fbpic.lib.diagnostics import MomentDescriptorDiagnostic
    from inversion_fbpic.utils.distributions import compute_moment_descriptor

    demo = _load_demo("demo_ionization_simulation", "run_simulation")
    output = tmp_path / "demo"
    output.mkdir()
    monkeypatch.setattr(demo, "__file__", str(output / "run_simulation.py"))
    monkeypatch.setattr(demo, "USE_MPI", False)
    monkeypatch.setattr(demo, "MPI_RANK", 0)
    monkeypatch.chdir(tmp_path)
    captured = []
    skip_current = False

    def setup(simulation, working_directory=None, **kwargs):
        simulation.working_directory = Path(working_directory)
        simulation.is_setup = not skip_current
        captured.append(simulation)

    def run(simulation, **kwargs):
        if not skip_current:
            simulation._analyze_diagnostics(before_simulation=False)

    monkeypatch.setattr(simulation_module.Simulation, "setup_simulation", setup)
    monkeypatch.setattr(simulation_module.Simulation, "run_simulation", run)
    particles = np.random.default_rng(21).normal(size=(128, 6))
    expected = compute_moment_descriptor(particles, np.ones(len(particles)))
    calculate = Mock(return_value=expected)
    monkeypatch.setattr(MomentDescriptorDiagnostic, "_analyze", calculate)
    monkeypatch.setattr(
        demo, "plot_from_hdf5_series", Mock(return_value=(output, "frame"))
    )
    monkeypatch.setattr(demo, "make_movie", Mock())
    monkeypatch.setattr(
        demo.analysis,
        "load_beam_data",
        Mock(return_value=(*[np.ones(4) for index in range(7)], 0.0)),
    )
    monkeypatch.setattr(demo.analysis, "analyze_beam", Mock(return_value={}))
    monkeypatch.setattr(demo.analysis, "print_beam_summary", Mock())
    monkeypatch.setattr(demo.analysis, "plot_beam_analysis", Mock())

    try:
        demo.main()
        skip_current = skip_run
        demo.main()
        assert calculate.call_count == (1 if skip_run else 2)
        path = output / "results.h5"
        loaded = SerializableConfig.from_file(path)
        assert isinstance(loaded, ConfigContainer)
        parameters, moments = loaded.configs
        assert isinstance(parameters, Parameters)
        assert isinstance(moments, MomentDescriptorDiagnostic)
        assert parameters.data["target_energy_ev"] == 430e6
        assert parameters.data["laser_energy_j"] == 4.5
        assert parameters.data["flattop_plasma_density_m_minus3"] == 1e24
        assert parameters.data["neutral_dopant_fraction"] == 0.03
        assert parameters.data["electron_diagnostic_selection"] == {"uz": [10.0, None]}
        assert parameters.data["laser_a0"] > 0.0
        assert parameters.data["flattop_length_m"] > 0.0
        assert moments.selection == ("all_of_species", "e")
        assert moments.data == expected
        assert len(moments.data) == 33
        assert captured[-1].parameters[0].data == parameters.data
        assert captured[-1].diagnostics[0].analysis_complete is (not skip_run)
        assert captured[-1].diagnostics[0].attached_simulation is captured[-1]
        assert (output / "cfgs" / "simulation.h5").is_file()
        assert (output / "plots" / "density_profiles.png").is_file()
        assert not (tmp_path / "results.h5").exists()
    finally:
        plt.close("all")


def test_ionization_demo_does_not_export_on_nonwriting_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inversion_fbpic.lib import simulation as simulation_module
    from inversion_fbpic.lib.diagnostics import MomentDescriptorDiagnostic

    demo = _load_demo("demo_ionization_simulation", "run_simulation")
    monkeypatch.setattr(demo, "__file__", str(tmp_path / "run_simulation.py"))
    monkeypatch.setattr(demo, "USE_MPI", True)
    monkeypatch.setattr(demo, "MPI_RANK", 1)
    monkeypatch.setattr(simulation_module, "MPI_RANK", 1)
    monkeypatch.setattr(simulation_module.Simulation, "setup_simulation", Mock())
    monkeypatch.setattr(simulation_module.Simulation, "run_simulation", Mock())
    calculate = Mock(side_effect=AssertionError("Nonwriting ranks must not analyze"))
    monkeypatch.setattr(MomentDescriptorDiagnostic, "_analyze", calculate)
    demo.main()
    calculate.assert_not_called()
    assert not (tmp_path / "results.h5").exists()
    assert not (tmp_path / "cfgs").exists()
