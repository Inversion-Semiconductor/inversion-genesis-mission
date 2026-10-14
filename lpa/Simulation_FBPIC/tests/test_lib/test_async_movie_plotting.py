"""Background movie jobs must not retain the active solver or use its MPI state."""

import gc
import importlib.util
import logging
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, call
import weakref

import attrs
import pytest


@pytest.fixture(scope="module")
def template_module():
    path = (
        Path(__file__).parents[2]
        / "runs/initial_sample/ionization_injection_template.py"
    )
    spec = importlib.util.spec_from_file_location("movie_template", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def movie_case(template_module):
    return template_module.IonizationInjectionSimulation.__new__(
        template_module.IonizationInjectionSimulation
    )


def test_non_write_rank_does_not_queue_movies(movie_case, template_module, monkeypatch):
    from_dict = Mock()
    monkeypatch.setattr(template_module.sm.Simulation, "from_dict", from_dict)
    simulation = SimpleNamespace(do_write_to_disk=False)
    assert movie_case._queue_movies(simulation, Path("results")) is None
    from_dict.assert_not_called()


def test_movie_job_returns_before_rendering_and_drops_solver_state(
    movie_case, template_module, minimal_simulation_elements, tmp_path, monkeypatch
):
    from inversion_fbpic.lib.laser import LasyLaserPulse
    from inversion_fbpic.lib.simulation import Simulation

    laser = LasyLaserPulse(
        energy=1.0,
        z0=0.0,
        z0_antenna=0.0,
        wavelength=800e-9,
        tau_fwhm=30e-15,
        waist=20e-6,
        focal_position=1e-3,
        super_gaussian_order=2.0,
    )
    simulation = Simulation(
        elements=[
            minimal_simulation_elements.hyparams,
            minimal_simulation_elements.densities[0],
            laser,
        ]
    )
    simulation.hyparams = attrs.evolve(simulation.hyparams, use_mpi=True)
    simulation.working_directory = tmp_path
    simulation.is_setup = True
    monkeypatch.setattr(template_module.sm, "MPI_RANK", 0)

    class SolverState:
        pass

    state = SolverState()
    simulation.simulation = state
    object.__setattr__(laser, "_high_order_laser", state)
    state_ref = weakref.ref(state)
    started, release = Event(), Event()
    main_thread = get_ident()
    calls, snapshots = [], []

    def render(snapshot, **kwargs):
        assert get_ident() != main_thread
        assert snapshot.hyparams.use_mpi is False
        assert not snapshot.is_setup
        assert not hasattr(snapshot, "simulation")
        assert snapshot.lasers[0]._high_order_laser is None
        snapshots.append(snapshot)
        calls.append(call(**kwargs))
        started.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(Simulation, "plot_movie", render)
    future = movie_case._queue_movies(simulation, Path("results"))
    try:
        assert future is not None
        assert started.wait(timeout=5)
        assert not future.done()
        assert simulation.hyparams.use_mpi is True
        simulation.working_directory = tmp_path / "next-run"
        del simulation, laser, state
        gc.collect()
        assert state_ref() is None
    finally:
        release.set()
        if future is not None:
            future.result(timeout=5)

    output = tmp_path / "results"
    assert len(calls) == 8
    assert calls[0] == call(
        field_name="rho",
        vminmax=(1e15, 1e18),
        cmap="magma",
        scale="log",
        save_path=output,
    )
    assert calls[-1] == call(field_name=None, component="eme", save_path=output)
    assert [entry.kwargs["component"] for entry in calls[1:7]] == [
        "x",
        "y",
        "z",
        "x",
        "y",
        "z",
    ]
    assert all(snapshot.working_directory == tmp_path for snapshot in snapshots)


def test_movie_batches_are_serialized(
    movie_case, template_module, minimal_simulation_elements, tmp_path, monkeypatch
):
    first_started, second_started, release = Event(), Event(), Event()

    def render(snapshot, **kwargs):
        if kwargs["save_path"] == tmp_path / "first":
            first_started.set()
            assert release.wait(timeout=5)
        else:
            second_started.set()

    monkeypatch.setattr(template_module.sm.Simulation, "plot_movie", render)
    first = movie_case._queue_movies(minimal_simulation_elements, tmp_path / "first")
    second = None
    try:
        assert first_started.wait(timeout=5)
        second = movie_case._queue_movies(
            minimal_simulation_elements, tmp_path / "second"
        )
        assert not second_started.wait(timeout=0.05)
        assert second is not None and not second.done()
    finally:
        release.set()
        if first is not None:
            first.result(timeout=5)
        if second is not None:
            second.result(timeout=5)
    assert second_started.is_set()


def test_movie_failures_are_logged_and_reported_by_future(
    movie_case,
    template_module,
    minimal_simulation_elements,
    tmp_path,
    monkeypatch,
    caplog,
):
    def render(snapshot, **kwargs):
        raise ValueError("invalid movie field")

    monkeypatch.setattr(template_module.sm.Simulation, "plot_movie", render)
    with caplog.at_level(logging.ERROR):
        future = movie_case._queue_movies(minimal_simulation_elements, tmp_path)
        with pytest.raises(ValueError, match="invalid movie field"):
            future.result(timeout=5)
    assert "Movie generation failed" in caplog.text


@pytest.mark.parametrize("plot_movies", [False, True])
def test_run_can_advance_while_movies_are_pending(
    template_module, tmp_path, monkeypatch, plot_movies
):
    cls = template_module.IonizationInjectionSimulation
    physical = {key: 0.0 for key in cls._PHYSICAL_PARAMETER_KEYS}
    physical.update(
        peak_neutral_density_cm3=1e18,
        nitrogen_dopant_fraction=0.03,
        density_fwhm_left_m=1e-3,
        density_fwhm_right_m=2e-3,
        density_beta_left=2.0,
        density_beta_right=4.0,
        density_center_location_m=2e-3,
        laser_energy_J=1.0,
        laser_wavelength_m=800e-9,
        laser_pulse_duration_fwhm_s=30e-15,
        laser_spot_size_m=20e-6,
        laser_focal_position_m=1e-3,
        laser_super_gaussian_order=2.0,
        laser_spectral_bandwidth_rad_s="auto",
        laser_cep_phase_rad=0.0,
    )
    monkeypatch.setattr(template_module.sm, "MPI_RANK", 0)
    monkeypatch.setattr(template_module.sm.Simulation, "setup_simulation", Mock())
    run = Mock()

    def record_run(simulation):
        run(simulation)

    monkeypatch.setattr(template_module.sm.Simulation, "run_simulation", record_run)
    monkeypatch.setattr(template_module.sm.Simulation, "to_hdf5_file", Mock())
    monkeypatch.setattr(template_module.cc.ConfigContainer, "to_hdf5_file", Mock())
    started, release = Event(), Event()

    def render(snapshot, **kwargs):
        started.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(template_module.sm.Simulation, "plot_movie", render)
    cases = [
        cls(
            physical,
            {
                "dump_fields": True,
                "plot_movies": plot_movies,
                "lab_diagnostic_directory": tmp_path / str(index) / "diags",
                "results_directory": tmp_path / str(index) / "results",
            },
        )
        for index in range(2)
    ]
    try:
        cases[0].run()
        if plot_movies:
            assert started.wait(timeout=5)
            assert not cases[0].movie_future.done()
        cases[1].run()
        assert run.call_count == 2
        for recorded in run.call_args_list:
            simulation = recorded.args[0]
            assert [density.species for density in simulation.densities] == ["He", "N"]
            for density in simulation.densities:
                assert density.fwhm_left == physical["density_fwhm_left_m"]
                assert density.fwhm_right == physical["density_fwhm_right_m"]
                assert density.beta_left == physical["density_beta_left"]
                assert density.beta_right == physical["density_beta_right"]
        if plot_movies:
            assert not cases[1].movie_future.done()
        else:
            assert all(case.movie_future is None for case in cases)
            assert not started.is_set()
    finally:
        release.set()
        for case in cases:
            if case.movie_future is not None:
                case.movie_future.result(timeout=5)


@pytest.mark.parametrize("dump_fields", [False, True])
def test_movie_flag_validation_uses_plot_movies(movie_case, dump_fields):
    movie_case.physical_parameters = {
        key: 0.0 for key in movie_case._PHYSICAL_PARAMETER_KEYS
    }
    movie_case.physical_parameters["nitrogen_dopant_fraction"] = 0.03
    movie_case.hyperparameters = {
        **movie_case._HYPERPARAMETER_DEFAULTS,
        "plot_movies": True,
        "dump_fields": dump_fields,
    }
    if dump_fields:
        movie_case._validate_parameters()
    else:
        with pytest.raises(ValueError, match="dump_fields must be True"):
            movie_case._validate_parameters()
