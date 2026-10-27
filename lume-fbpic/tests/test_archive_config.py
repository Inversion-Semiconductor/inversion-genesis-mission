"""Tests for how an archive stores the config: native HDF5, no paths, input files by basename."""

from __future__ import annotations

import gc
import re
import shutil
import typing
from pathlib import Path

import attrs
import h5py
import numpy
import pytest

from archive_helpers import load_simulator, simulator_model
import lume_fbpic.density_profiles  # noqa: F401  (registers its classes)
import lume_fbpic.pwfa_config  # noqa: F401
from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile
from inversion_fbpic.lib.serializable_config import SerializableConfig
from lume_fbpic import archive, archive_config
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.lwfa_simulator import LWFASimulator

# Registered config classes with a path-like field the archive does not handle as a path, and why.
_UNHANDLED_PATH_FIELDS = {
    ("ConfigContainer", "configs"): "not a config of an archive",
    ("ModifiedDensityProfile", "base_density_profile"): "rejected at archive time",
    ("ModifiedDensityProfile", "modifiers"): "rejected at archive time",
    ("Simulation", "elements"): "not a config of an archive",
}


def test_a_changed_input_file_in_input_dirs_warns_when_loading(tmp_path):
    simulator, library = _referenced(tmp_path)
    archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)
    _write_table(library / "table.h5", scale=2.0)

    with pytest.warns(UserWarning, match="not the one that was archived"):
        loaded = load_simulator(tmp_path / "a.h5", input_dirs=library)

    assert loaded.densities[0].filename == (library / "table.h5").resolve()


def test_a_different_input_file_in_input_dirs_warns_and_is_not_embedded(tmp_path):
    simulator, library = _referenced(tmp_path)
    _write_table(library / "table.h5", scale=2.0)

    with pytest.warns(UserWarning, match="differs from the one"):
        archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)

    with h5py.File(tmp_path / "a.h5") as f:
        assert "data" not in f["inputs/0"]


def test_a_path_to_a_config_file_cannot_be_archived():
    @attrs.define
    class Modified:
        base_density_profile: str

    with pytest.raises(ValueError, match="cannot hold"):
        archive_config.collect_inputs({"plasma": Modified("base.yaml")}, {})


def test_a_save_directory_survives_the_archive_and_not_the_fingerprint(
    simulator, tmp_path
):
    fingerprint = archive.fingerprint(simulator)
    simulator.hyparams = attrs.evolve(simulator.hyparams, save_directory="scan_07")

    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    loaded = load_simulator(tmp_path / "a.h5")
    assert loaded.hyparams.save_directory == Path("scan_07")
    assert archive.fingerprint(loaded) == archive.fingerprint(simulator) == fingerprint


def test_an_absolute_save_directory_is_archived_by_its_name_only(simulator, tmp_path):
    simulator.hyparams = attrs.evolve(
        simulator.hyparams, save_directory=tmp_path / "somewhere" / "out"
    )

    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert f["hyparams/parameters/save_directory"][()] == b"out"
    loaded = load_simulator(tmp_path / "a.h5")
    assert loaded.hyparams.save_directory == Path("out")


def test_an_archive_holds_no_path_string(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert not _paths_in(f)


def test_an_archive_holds_the_config_as_native_hdf5_not_yaml(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert int(f["hyparams/parameters/nz"][()]) == simulator.hyparams.nz
        assert f["densities"].attrs["list"]
        assert len(f["densities"]) == len(simulator.densities)
        names = []
        f.visititems(lambda name, node: names.extend(node.attrs))
        assert "yaml" not in names


def test_an_archive_reads_each_input_file_once(tmp_path, mocker, monkeypatch):
    simulator, _ = _referenced(tmp_path)
    monkeypatch.chdir(
        tmp_path
    )  # the default archive name, from the fingerprint, goes here
    model = LUMEFBPICModel(simulator, [], dummy_run=True)
    md5 = mocker.spy(archive_config.hashlib, "md5")

    archive.save(model, input_dirs=tmp_path / "empty")

    assert md5.call_count == 1


def test_an_archive_with_a_lasy_laser_holds_no_path_string(tmp_path):
    import ionization_injection

    model = ionization_injection.build_model(working_directory=str(tmp_path / "wd"))
    laser = model.simulator.laser
    model.simulator.laser = attrs.evolve(laser, lasy_file=tmp_path / "out" / "lasy")
    model.dummy_run = True

    archive.save(model, tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert not _paths_in(f)
        assert f["laser/parameters/lasy_file"][()] == b"lasy"
    loaded = archive.load(tmp_path / "a.h5")
    assert loaded.simulator.laser.lasy_file == Path("diags/lasy")


def test_an_archive_with_input_files_holds_no_path_string(tmp_path):
    simulator, library = _referenced(tmp_path)
    archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)

    with h5py.File(tmp_path / "a.h5") as f:
        assert not _paths_in(f)
        assert f["inputs/0"].attrs["basename"] == "table.h5"
        assert f["densities/0/parameters/filename"][()] == b"table.h5"


def test_an_archive_without_embedded_input_files_keeps_no_temporary_directory(
    simulator, tmp_path
):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    loaded = load_simulator(tmp_path / "a.h5")

    assert loaded._inputs_directory is None


def test_an_embedded_input_file_does_not_replace_a_path_changed_on_purpose(tmp_path):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    loaded = load_simulator(tmp_path / "a.h5", working_directory=tmp_path / "wd")
    other = tmp_path / "other" / "table.h5"
    other.parent.mkdir()
    _write_table(other, scale=2.0)
    loaded.densities = [attrs.evolve(loaded.densities[0], filename=other)]

    loaded.configure()

    assert loaded.densities[0].filename == other  # not the embedded copy


def test_an_embedded_input_file_is_loaded_into_a_temporary_directory_without_a_working_directory(
    tmp_path, monkeypatch
):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    loaded = load_simulator(tmp_path / "a.h5")
    used = loaded.densities[0].filename

    assert used.is_file() and not used.is_relative_to(cwd)
    assert list(cwd.iterdir()) == []
    del loaded
    gc.collect()
    assert not used.exists()


def test_an_embedded_input_file_is_used_instead_of_one_in_input_dirs(tmp_path):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    other = tmp_path / "other"
    other.mkdir()
    _write_table(other / "table.h5", scale=3.0)

    loaded = load_simulator(
        tmp_path / "a.h5", input_dirs=other, working_directory=tmp_path / "wd"
    )

    used = loaded.densities[0].filename
    md5 = archive_config.checksum(tmp_path / "library" / "table.h5")
    assert used == (tmp_path / "wd" / "inputs" / md5 / "table.h5").resolve()
    assert archive_config.checksum(used) == md5


def test_an_embedded_input_file_is_written_again_when_the_working_directory_changes(
    tmp_path,
):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    loaded = load_simulator(tmp_path / "a.h5", working_directory=tmp_path / "one")

    loaded.working_directory = tmp_path / "two"
    loaded.configure()

    md5 = archive_config.checksum(tmp_path / "library" / "table.h5")
    expected = (tmp_path / "two" / "inputs" / md5 / "table.h5").resolve()
    assert loaded.densities[0].filename == expected and expected.is_file()


def test_an_embedded_input_file_loaded_without_a_working_directory_is_written_out_to_run(
    tmp_path, monkeypatch
):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    monkeypatch.chdir(tmp_path / "empty")
    loaded = load_simulator(tmp_path / "a.h5")

    loaded.configure()

    md5 = archive_config.checksum(tmp_path / "library" / "table.h5")
    expected = (tmp_path / "empty" / "inputs" / md5 / "table.h5").resolve()
    assert loaded.densities[0].filename == expected and expected.is_file()


def test_an_embedded_input_file_is_not_read_again_on_every_configure(tmp_path, mocker):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    loaded = load_simulator(tmp_path / "a.h5", working_directory=tmp_path / "one")
    loaded.configure()
    checksum = mocker.spy(archive_config, "checksum")

    loaded.configure()
    loaded.configure()

    checksum.assert_not_called()


def test_an_embedded_input_file_changed_on_disk_is_written_again(tmp_path):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    loaded = load_simulator(tmp_path / "a.h5", working_directory=tmp_path / "one")
    loaded.configure()
    extracted = Path(loaded.densities[0].filename)
    original = extracted.read_bytes()

    extracted.write_bytes(b"changed")
    loaded.configure()

    assert extracted.read_bytes() == original


def test_an_embedded_input_file_loads_in_an_empty_directory(tmp_path, monkeypatch):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    shutil.rmtree(tmp_path / "library")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loaded = load_simulator(tmp_path / "a.h5", working_directory=elsewhere / "wd")

    assert loaded.densities[0].filename.is_file()


def test_an_embedded_input_file_over_the_size_limit_warns(tmp_path, monkeypatch):
    simulator, _ = _referenced(tmp_path)
    monkeypatch.setattr(archive_config, "EMBED_WARNING_BYTES", 10)

    with pytest.warns(UserWarning, match="Embedding the input file"):
        archive.save(
            simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
        )


def test_an_embedded_input_file_survives_archiving_a_loaded_model(tmp_path):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    loaded = load_simulator(tmp_path / "a.h5", working_directory=tmp_path / "wd")

    archive.save(
        simulator_model(loaded), tmp_path / "b.h5", input_dirs=tmp_path / "empty"
    )

    again = load_simulator(tmp_path / "b.h5", working_directory=tmp_path / "wd2")
    assert again.densities[0].to_dict()["parameters"]["lineout_axis"] == (
        simulator.densities[0].to_dict()["parameters"]["lineout_axis"]
    )


def test_an_input_file_in_input_dirs_is_referred_to_and_not_embedded(tmp_path):
    simulator, library = _referenced(tmp_path)

    archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)

    with h5py.File(tmp_path / "a.h5") as f:
        assert "data" not in f["inputs/0"]
        assert f["inputs/0"].attrs["md5"] == archive_config.checksum(
            library / "table.h5"
        )


def test_an_input_file_loads_from_input_dirs_anywhere(tmp_path):
    simulator, library = _referenced(tmp_path)
    archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)
    moved = tmp_path / "moved"
    shutil.copytree(library, moved)
    shutil.copy(tmp_path / "a.h5", moved / "a.h5")
    shutil.rmtree(library)
    shutil.rmtree(tmp_path / "source")

    loaded = load_simulator(moved / "a.h5", input_dirs=moved)

    assert loaded.densities[0].filename == (moved / "table.h5").resolve()


def test_an_input_file_that_input_dirs_lacks_stops_loading(tmp_path):
    simulator, library = _referenced(tmp_path)
    archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)
    (library / "table.h5").unlink()

    with pytest.raises(FileNotFoundError, match="table.h5"):
        load_simulator(tmp_path / "a.h5", input_dirs=library)


def test_an_input_file_that_the_config_does_not_find_stops_archiving(tmp_path):
    simulator, library = _referenced(tmp_path)
    (tmp_path / "source" / "table.h5").unlink()

    with pytest.raises(FileNotFoundError, match="could not run"):
        archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)
    assert not (tmp_path / "a.h5").exists()


def test_embedded_input_files_of_one_archive_with_the_same_basename_stay_apart(
    tmp_path,
):
    simulator = _two_tables(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )

    loaded = load_simulator(
        tmp_path / "a.h5",
        input_dirs=tmp_path / "empty",
        working_directory=tmp_path / "wd",
    )

    assert [archive_config.checksum(d.filename) for d in loaded.densities] == [
        archive_config.checksum(d.filename) for d in simulator.densities
    ]
    assert archive.fingerprint(loaded) == archive.fingerprint(simulator)


def test_embedded_input_files_with_the_same_basename_from_two_archives_stay_apart(
    tmp_path,
):
    simulator, _ = _referenced(tmp_path)
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", input_dirs=tmp_path / "empty"
    )
    _write_table(tmp_path / "source" / "table.h5", scale=3.0)
    archive.save(
        simulator_model(simulator), tmp_path / "b.h5", input_dirs=tmp_path / "empty"
    )

    first = load_simulator(tmp_path / "a.h5", working_directory=tmp_path / "wd")
    fingerprint = archive.fingerprint(first)
    second = load_simulator(tmp_path / "b.h5", working_directory=tmp_path / "wd")

    assert first.densities[0].filename != second.densities[0].filename
    assert archive.fingerprint(first) == fingerprint


def test_every_config_path_field_is_handled_or_listed():
    handled = {
        (cls.__name__, name)
        for cls, names in archive_config.OUTPUT_PATH_FIELDS.items()
        for name in names
    }
    unexpected = []
    for domain in SerializableConfig._DOMAIN_REGISTRY.values():
        for cls in domain._CONCRETE_REGISTRY.values():
            for field in attrs.fields(cls):
                key = (cls.__name__, field.name)
                numeric = re.match(r"(float|int|bool)\b", str(field.type))  # not a path
                pathlike = re.search(r"\bPath\b", str(field.type)) or (
                    not numeric
                    and re.search(
                        r"(^|_)(directory|file|filename|path)(_|$)", field.name
                    )
                )
                if (
                    not field.init
                    or field.name == "source_file"
                    or not pathlike
                    or field.metadata.get("input_path")
                    or key in handled
                    or key in _UNHANDLED_PATH_FIELDS
                ):
                    continue
                unexpected.append(key)
    assert not unexpected, (
        f"path fields the archive does not handle: {unexpected}; add them to "
        "archive_config.OUTPUT_PATH_FIELDS, or mark them input_path"
    )


def test_the_fingerprint_does_not_depend_on_paths_but_on_the_input_data(tmp_path):
    simulator, library = _referenced(tmp_path)
    first = archive.fingerprint(simulator)
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(tmp_path / "source" / "table.h5", moved / "table.h5")
    simulator.densities = [
        attrs.evolve(simulator.densities[0], filename=moved / "table.h5")
    ]
    simulator.hyparams = attrs.evolve(simulator.hyparams, save_directory="elsewhere")

    assert archive.fingerprint(simulator) == first

    _write_table(moved / "table.h5", scale=2.0)
    simulator.densities = [
        attrs.evolve(simulator.densities[0], filename=moved / "table.h5")
    ]
    assert archive.fingerprint(simulator) != first


def test_the_model_passes_input_dirs_to_the_simulator(tmp_path):
    simulator, library = _referenced(tmp_path)
    model = LUMEFBPICModel(simulator, [], dummy_run=True)

    archive.save(model, tmp_path / "a.h5", input_dirs=library)
    loaded = archive.load(tmp_path / "a.h5", input_dirs=library)

    assert loaded.simulator.densities[0].filename == (library / "table.h5").resolve()


def test_the_order_of_a_config_dict_survives_the_archive(tmp_path):
    simulator, library = _referenced(tmp_path)
    archive.save(simulator_model(simulator), tmp_path / "a.h5", input_dirs=library)

    loaded = load_simulator(tmp_path / "a.h5", input_dirs=library)

    axes = loaded.densities[0].to_dict()["parameters"]["lineout_axis"]
    assert list(axes) == ["z_m", "x_mm"]  # not alphabetical


def _paths_in(group: h5py.Group) -> list[str]:
    """The string values in an archive that hold a directory separator: a path, not a basename."""
    found = []

    def look(name: str, node: typing.Any) -> None:
        values = list(node.attrs.values())
        if isinstance(node, h5py.Dataset) and node.dtype.kind in "OS":
            values.append(node[()])
        for value in values:
            text = value.decode() if isinstance(value, bytes) else value
            if isinstance(text, str) and re.search(r"[/\\]", text):
                found.append(f"{name}: {text}")

    group.visititems(look)
    return found


def _referenced(tmp_path: Path) -> tuple[LWFASimulator, Path]:
    """A simulator whose density profile reads `source/table.h5`, and a `library` directory with
    the same file. `empty` is a directory without it."""
    from inversion_fbpic.lib.laser import GaussianLaserPulse
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    (tmp_path / "source").mkdir()
    (tmp_path / "library").mkdir()
    (tmp_path / "empty").mkdir()
    _write_table(tmp_path / "source" / "table.h5")
    shutil.copy(tmp_path / "source" / "table.h5", tmp_path / "library" / "table.h5")
    profile = InterpolateFromH5Profile(
        filename=tmp_path / "source" / "table.h5",
        density_name="density",
        lineout_axis={
            "z_m": {"origin": 0.0, "coefficient": 1.0},
            "x_mm": {"origin": 0.0, "coefficient": 0.0},
        },
        interpolation_points={"pressure_bar": 1.5},
        p_nz=1,
        p_nr=1,
        p_nt=1,
        elec_name="electrons",
    )
    simulator = LWFASimulator(
        SimulationHyperparameters(
            zmin=-1.0e-5,
            zmax=0.0,
            rmax=1.0e-5,
            nz=8,
            nr=8,
            nm=1,
            use_mpi=False,
            number_dumps=2,
        ),
        GaussianLaserPulse(
            energy=5.0,
            z0=-3.0e-5,
            wavelength=8.0e-7,
            tau_fwhm=3.8e-14,
            cep=0.0,
            waist=2.8e-5,
            focal_position=3.0e-3,
            polarization=0.0,
        ),
        [profile],
        target_species="electrons",
        working_directory=tmp_path / "wd",
    )
    return simulator, tmp_path / "library"


def _two_tables(tmp_path: Path) -> LWFASimulator:
    """`_referenced()`'s simulator with a second density profile that reads a different file of
    the same basename, `other/table.h5`."""
    simulator, _ = _referenced(tmp_path)
    (tmp_path / "other").mkdir()
    _write_table(tmp_path / "other" / "table.h5", scale=3.0)
    profile = simulator.densities[0]
    simulator.densities = [
        profile,
        attrs.evolve(profile, filename=tmp_path / "other" / "table.h5", elec_name="e2"),
    ]
    return simulator


def _write_table(path: Path, scale: float = 1.0) -> None:
    """A small density table in the layout `InterpolateFromH5Profile` reads."""
    z = numpy.linspace(0.0, 4.0e-3, 17)
    x = numpy.linspace(-1.0e-3, 1.0e-3, 5)
    pressure = numpy.array([1.0, 2.0])
    profile = numpy.exp(-(((z - 2.0e-3) / 3.0e-4) ** 2))
    density = (
        scale
        * 1.0e24
        * profile[:, None, None]
        * numpy.ones((1, x.size, 1))
        * pressure[None, None, :]
    )
    with h5py.File(path, "w") as f:
        for name, values in (("z_m", z), ("x_mm", x), ("pressure_bar", pressure)):
            f[name] = values
        dataset = f.create_dataset("density", data=density)
        dataset.attrs["DIMENSION_LABELS"] = ["z_m", "x_mm", "pressure_bar"]
