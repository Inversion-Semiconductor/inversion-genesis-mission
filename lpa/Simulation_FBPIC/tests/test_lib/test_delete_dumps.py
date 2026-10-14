"""Diagnostic dump retention and MPI write-rank behavior for Simulation."""

from pathlib import Path

import attrs
import pytest


@pytest.fixture()
def simulation_with_dumps(minimal_simulation_elements, tmp_path):
    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path
    dump_directory = tmp_path / "diags" / "hdf5"
    dump_directory.mkdir(parents=True)
    for iteration in (0, 2, 10):
        (dump_directory / f"data{iteration:08d}.h5").write_bytes(b"dump")
    return simulation, dump_directory


@pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
def test_delete_dumps_keeps_latest_n(simulation_with_dumps, n):
    simulation, dump_directory = simulation_with_dumps
    expected = [f"data{iteration:08d}.h5" for iteration in (0, 2, 10)]
    expected = expected[-n:] if n else []

    assert simulation.delete_dumps(n) is None
    assert sorted(path.name for path in dump_directory.iterdir()) == expected
    assert not simulation.is_setup

    simulation.delete_dumps(n)
    assert sorted(path.name for path in dump_directory.iterdir()) == expected


def test_delete_dumps_orders_by_iteration(simulation_with_dumps):
    simulation, dump_directory = simulation_with_dumps
    for iteration in (9, 100):
        (dump_directory / f"data{iteration}.h5").write_bytes(b"dump")

    simulation.delete_dumps(2)

    assert sorted(path.name for path in dump_directory.iterdir()) == [
        "data00000010.h5",
        "data100.h5",
    ]


def test_delete_dumps_preserves_other_output(simulation_with_dumps):
    simulation, dump_directory = simulation_with_dumps
    preserved = [
        dump_directory / "notes.txt",
        dump_directory / "other.h5",
        dump_directory / "data_summary.h5",
        dump_directory / "data.h5",
        dump_directory / "data00000000.h5.bak",
        dump_directory.parent / "sim.sha256",
        simulation.working_directory / "checkpoints" / "data00000000.h5",
        dump_directory / "nested" / "data00000000.h5",
    ]
    for path in preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve")
    matching_directory = dump_directory / "data00000001.h5"
    matching_directory.mkdir()

    simulation.delete_dumps(0)

    assert matching_directory.is_dir()
    assert all(path.read_bytes() == b"preserve" for path in preserved)
    assert not any(path.is_file() for path in dump_directory.glob("data[0-9]*.h5"))


@pytest.mark.parametrize("absolute_path", [False, True])
def test_delete_dumps_custom_directory(
    minimal_simulation_elements, tmp_path, absolute_path
):
    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path / "run"
    save_directory = tmp_path / "output" if absolute_path else Path("output")
    simulation.hyparams = attrs.evolve(
        simulation.hyparams, save_directory=save_directory
    )
    resolved = (
        save_directory
        if absolute_path
        else simulation.working_directory / save_directory
    )
    dump_directory = resolved / "hdf5"
    dump_directory.mkdir(parents=True)
    for iteration in (0, 1):
        (dump_directory / f"data{iteration:08d}.h5").write_bytes(b"dump")

    simulation.delete_dumps(1)

    assert sorted(path.name for path in dump_directory.iterdir()) == ["data00000001.h5"]


@pytest.mark.parametrize("create_directory", [False, True])
def test_delete_dumps_missing_or_empty_output(
    minimal_simulation_elements, tmp_path, create_directory
):
    simulation = minimal_simulation_elements
    simulation.working_directory = tmp_path
    dump_directory = tmp_path / "diags" / "hdf5"
    if create_directory:
        dump_directory.mkdir(parents=True)

    simulation.delete_dumps(0)

    assert dump_directory.exists() == create_directory
    assert not list(tmp_path.rglob("*.h5"))


@pytest.mark.parametrize(
    ("use_mpi", "rank", "writes"),
    [(False, 1, True), (True, 0, True), (True, 1, False)],
)
def test_delete_dumps_respects_write_rank(
    simulation_with_dumps, monkeypatch, use_mpi, rank, writes
):
    from inversion_fbpic.lib import simulation as simulation_module

    simulation, dump_directory = simulation_with_dumps
    simulation.hyparams = attrs.evolve(simulation.hyparams, use_mpi=use_mpi)
    monkeypatch.setattr(simulation_module, "MPI_RANK", rank)
    if not writes:

        def unexpected_scan(*args, **kwargs):
            pytest.fail("Non-write ranks must not scan diagnostics")

        monkeypatch.setattr(Path, "glob", unexpected_scan)

    simulation.delete_dumps(1)

    assert len(list(dump_directory.iterdir())) == (1 if writes else 3)


@pytest.mark.parametrize(
    ("n", "error"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (True, TypeError),
    ],
)
def test_delete_dumps_rejects_invalid_n(simulation_with_dumps, n, error):
    simulation, dump_directory = simulation_with_dumps

    with pytest.raises(error, match="n must be"):
        simulation.delete_dumps(n)

    assert len(list(dump_directory.iterdir())) == 3
