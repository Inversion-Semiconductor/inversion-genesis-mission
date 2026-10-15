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


@pytest.fixture(autouse=True)
def lasy_plot(monkeypatch):
    """Keep concurrency tests independent of expensive LASY pulse preparation."""
    import matplotlib.pyplot as plt

    from inversion_fbpic.lib.laser import LasyLaserPulse

    plot = Mock(side_effect=lambda: plt.figure())
    monkeypatch.setattr(LasyLaserPulse, "plot", plot)
    return plot


def test_non_write_rank_does_not_queue_movies(movie_case, template_module, monkeypatch):
    from_dict = Mock()
    monkeypatch.setattr(template_module.sm.Simulation, "from_dict", from_dict)
    simulation = SimpleNamespace(do_write_to_disk=False)
    assert movie_case._queue_movies(simulation, Path("results")) is None
    from_dict.assert_not_called()


def test_movie_job_returns_before_rendering_and_drops_solver_state(
    movie_case,
    template_module,
    minimal_simulation_elements,
    tmp_path,
    monkeypatch,
    lasy_plot,
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
    assert (output / "densities.png").is_file()
    assert (output / "laser_0.png").is_file()
    lasy_plot.assert_called_once_with()
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


@pytest.mark.parametrize("failing_profile", [None, "densities.png", "laser_0.png"])
def test_profile_figures_are_closed_even_when_saving_fails(
    movie_case,
    minimal_simulation_elements,
    tmp_path,
    monkeypatch,
    failing_profile,
    caplog,
):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    simulation = minimal_simulation_elements
    figures = []

    def profile_plot(*args, **kwargs):
        fig = plt.figure()
        figures.append(fig)
        return fig

    savefig = Figure.savefig

    def save_profile(fig, path, *args, **kwargs):
        if Path(path).name == failing_profile:
            raise OSError("profile save failed")
        return savefig(fig, path, *args, **kwargs)

    monkeypatch.setattr(type(simulation), "plot_densities", profile_plot)
    monkeypatch.setattr(type(simulation.lasers[0]), "plot", profile_plot)
    monkeypatch.setattr(Figure, "savefig", save_profile)
    movies = Mock()
    monkeypatch.setattr(type(simulation), "plot_movie", movies)
    output = tmp_path / "nested" / "results"
    try:
        with caplog.at_level(logging.ERROR):
            movie_case._plot_movies(simulation, output)
        assert len(figures) == 2
        assert movies.call_count == 8
        for filename in ["densities.png", "laser_0.png"]:
            assert (output / filename).is_file() == (filename != failing_profile)
        if failing_profile is not None:
            assert "profile save failed" in caplog.text
        assert all(not plt.fignum_exists(fig.number) for fig in figures)
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("failing_profile", ["density", "laser"])
def test_profile_plot_failures_do_not_cancel_movies(
    movie_case,
    minimal_simulation_elements,
    tmp_path,
    monkeypatch,
    failing_profile,
    caplog,
):
    import matplotlib.pyplot as plt

    simulation = minimal_simulation_elements
    simulation = type(simulation)(elements=[*simulation.elements, simulation.lasers[0]])
    figures = []

    def profile_plot(*args, **kwargs):
        fig = plt.figure()
        figures.append(fig)
        return fig

    density_plot = Mock(side_effect=profile_plot)
    laser_plot = Mock(side_effect=profile_plot)
    if failing_profile == "density":
        density_plot.side_effect = ValueError("density plot failed")
    else:
        laser_plot.side_effect = [ValueError("laser plot failed"), profile_plot()]
    monkeypatch.setattr(type(simulation), "plot_densities", density_plot)
    monkeypatch.setattr(type(simulation.lasers[0]), "plot", laser_plot)
    movies = Mock()
    monkeypatch.setattr(type(simulation), "plot_movie", movies)
    try:
        with caplog.at_level(logging.ERROR):
            future = movie_case._queue_movies(simulation, tmp_path)
            assert future is not None
            assert future.result(timeout=5) is None
        density_plot.assert_called_once_with()
        assert laser_plot.call_count == 2
        assert movies.call_count == 8
        assert (tmp_path / "laser_1.png").is_file()
        assert f"{failing_profile} plot failed" in caplog.text
        assert all(not plt.fignum_exists(fig.number) for fig in figures)
    finally:
        for fig in figures:
            plt.close(fig)


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
        density_left_ramp_edge_percent=0.5,
        density_right_ramp_edge_percent=0.5,
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
                one_percent_location = density.center - (
                    density.fwhm_left / 2.0
                    + physical["density_left_ramp_edge_percent"]
                    * (density.center - density.fwhm_left / 2.0)
                )
                assert density.build_density_function()(
                    one_percent_location, 0.0
                ) == pytest.approx(0.01)
                right_one_percent_location = density.center + (
                    density.fwhm_right / 2.0
                    + physical["density_right_ramp_edge_percent"]
                    * (density.center - density.fwhm_right / 2.0)
                )
                assert density.build_density_function()(
                    right_one_percent_location, 0.0
                ) == pytest.approx(0.01)
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
