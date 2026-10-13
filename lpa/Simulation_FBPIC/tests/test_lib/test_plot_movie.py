"""Movie plotting and MPI write-rank behavior for Simulation."""

from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import attrs
import numpy as np
import pytest


@pytest.fixture()
def movie_helpers(monkeypatch):
    plot = MagicMock()
    movie = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "inversion_fbpic.utils.plotting",
        SimpleNamespace(plot_from_hdf5_series=plot),
    )
    monkeypatch.setitem(
        sys.modules,
        "inversion_fbpic.utils.make_movie",
        SimpleNamespace(make_movie=movie),
    )
    return plot, movie


@pytest.mark.parametrize(("use_mpi", "rank"), [(False, 1), (True, 0)])
@pytest.mark.parametrize(
    ("field_name", "component", "prefix"),
    [("rho", None, "rho"), ("E", "z", "E_z"), (None, "eme", "eme")],
)
def test_plot_movie_defaults(
    minimal_simulation_elements,
    movie_helpers,
    tmp_path,
    monkeypatch,
    use_mpi,
    rank,
    field_name,
    component,
    prefix,
):
    from inversion_fbpic.lib import simulation as simulation_module

    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path
    simulation.hyparams = attrs.evolve(simulation.hyparams, use_mpi=use_mpi)
    monkeypatch.setattr(simulation_module, "MPI_RANK", rank)
    plot, movie = movie_helpers
    stills_dir = tmp_path / "plots" / "stills"
    plot.return_value = stills_dir, prefix
    movie.return_value = tmp_path / "plots" / f"{prefix}.mp4"

    assert simulation.plot_movie(field_name, component) == movie.return_value

    plot.assert_called_once_with(
        tmp_path / "diags" / "hdf5",
        stills_dir,
        field_name=field_name,
        component=component,
        vminmax=None,
        scale="linear",
        field_offset=0.0,
        cmap="bwr",
        rmax=None,
        font_size=16,
    )
    movie.assert_called_once_with(
        stills_dir,
        prefix,
        filename=prefix,
        framerate=10,
        movie_dir=tmp_path / "plots",
    )


@pytest.mark.parametrize("absolute_paths", [False, True])
def test_plot_movie_custom_options(
    minimal_simulation_elements,
    movie_helpers,
    tmp_path,
    absolute_paths,
):
    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path / "run"
    diagnostics = tmp_path / "diagnostics" if absolute_paths else Path("diagnostics")
    output = tmp_path / "movies" if absolute_paths else Path("movies")
    simulation.hyparams = attrs.evolve(simulation.hyparams, save_directory=diagnostics)
    resolved_diagnostics = (
        diagnostics if absolute_paths else simulation.working_directory / diagnostics
    )
    resolved_output = (
        output if absolute_paths else simulation.working_directory / output
    )
    plot, movie = movie_helpers
    stills_dir = resolved_output / "stills"
    plot.return_value = stills_dir, "E_x"
    movie.return_value = None

    assert (
        simulation.plot_movie(
            "E",
            "x",
            vminmax="even",
            scale="log",
            field_offset=2.0,
            cmap="magma",
            rmax=5e-6,
            font_size=12,
            save_path=str(output),
            filename="electric-field",
            framerate=24,
        )
        is None
    )

    plot.assert_called_once_with(
        resolved_diagnostics / "hdf5",
        stills_dir,
        field_name="E",
        component="x",
        vminmax="even",
        scale="log",
        field_offset=2.0,
        cmap="magma",
        rmax=5e-6,
        font_size=12,
    )
    movie.assert_called_once_with(
        stills_dir,
        "E_x",
        filename="electric-field",
        framerate=24,
        movie_dir=resolved_output,
    )


def test_plot_movie_non_write_rank_does_not_import_or_write(
    minimal_simulation_elements,
    tmp_path,
    monkeypatch,
):
    from inversion_fbpic.lib import simulation as simulation_module

    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path
    simulation.hyparams = attrs.evolve(simulation.hyparams, use_mpi=True)
    monkeypatch.setattr(simulation_module, "MPI_RANK", 1)
    monkeypatch.setitem(sys.modules, "inversion_fbpic.utils.plotting", None)
    monkeypatch.setitem(sys.modules, "inversion_fbpic.utils.make_movie", None)

    assert simulation.plot_movie("rho") is None
    assert not list(tmp_path.iterdir())


def test_plot_movie_plot_error_does_not_encode(
    minimal_simulation_elements,
    movie_helpers,
):
    plot, movie = movie_helpers
    plot.side_effect = ValueError("invalid field")

    with pytest.raises(ValueError, match="invalid field"):
        minimal_simulation_elements.plot_movie()

    movie.assert_not_called()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_plot_movie_creates_stills_and_movie(
    minimal_simulation_elements,
    tmp_path,
    monkeypatch,
):
    from inversion_fbpic.utils import plotting

    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path
    info = SimpleNamespace(zmin=0.0, zmax=1e-5, rmin=-1e-5, rmax=1e-5, time=0.0)
    series = MagicMock(iterations=[0, 1])
    series.get_field.return_value = np.zeros((8, 8)), info
    open_series = MagicMock(return_value=series)
    monkeypatch.setattr(plotting, "OpenPMDTimeSeries", open_series)

    movie_path = simulation.plot_movie("rho", vminmax=(0, 1e18), cmap="magma")

    open_series.assert_called_once_with(tmp_path / "diags" / "hdf5")
    assert movie_path == tmp_path / "plots" / "rho.mp4"
    assert movie_path.is_file()
    assert movie_path.stat().st_size > 0
    for iteration in [0, 1]:
        still = tmp_path / "plots" / "stills" / f"rho_{iteration:06d}.png"
        assert still.is_file()
        assert still.stat().st_size > 0
