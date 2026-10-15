"""Archives of a `LUMEFBPICModel`: `save()` writes one, `load()` rebuilds the model from it.

An archive is one HDF5 file (or group). It holds:

- attributes `dataType`, `software`, `config_kind` (the simulator's `CONFIG_KIND`, which `load()`
  uses to pick the simulator class) and `target_species`;
- one group per config object of `BaseSimulator.config()`, named as in its `CONFIG_NAMES` (a
  list is a group with the attribute `list` and one numbered group per object), written by
  `archive_config.write_config()`: native HDF5 and no paths. An output path (`save_directory`, `lasy_file`) is stored as its basename,
  and a loaded config has that name in the directory of the class default;
- `inputs`: the input files the config names, by basename and md5. A file is looked for in the
  `input_dirs` given to `save()` and `load()`; one that is not in `input_dirs` at `save()` is
  embedded instead (with a warning over 10 MB), and a loaded archive uses the embedded copy (see
  `archive_config`);
- `parameters`: the simulator's `parameters`, if it has one;
- with `save_final_particles`, `final_particles` (openPMD-beamphysics, so the file opens with
  `ParticleGroup(h5=...)`) and `stats`;
- `actions`: one numbered group per action, its class and parameters and, for a read-only scalar
  action, its value when it was saved (NaN where there was no result). The inputs are not stored
  as values; they are in the config.

The time-series diagnostics fbpic writes under `diags/` are never archived (they stay on disk).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
import typing

import h5py
import numpy

from lume.actions import Action, ReadOnlyActionMixin
from lume.variables import ScalarVariable

from beamphysics import ParticleGroup
from beamphysics.writers import pmd_init

import lume_fbpic.actions

# Config classes register themselves in `SerializableConfig` when their module is imported, which
# is how `from_hdf5()` finds the class of a stored config. The `inversion_fbpic.lib` ones are all
# imported by the `inversion_fbpic.lib` package itself; lume_fbpic's own density profiles are
# imported here, so that a fresh process can load an archive that uses one.
import lume_fbpic.density_profiles  # noqa: F401
from inversion_fbpic.lib.serializable_config import GIT_HASH_STR
from lume_fbpic import archive_config
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import BaseSimulator

# The groups of an archive that are not config: the model's actions, the final particles, the
# input files, the `parameters` metadata, and the statistics.
_ACTIONS = "actions"
_FINAL_PARTICLES = "final_particles"
_INPUTS = "inputs"
_PARAMETERS = "parameters"
_STATS = "stats"
_RESERVED_GROUPS = {_ACTIONS, _FINAL_PARTICLES, _INPUTS, _PARAMETERS, _STATS}


def fingerprint(simulator: BaseSimulator) -> str:
    """Stable hash of the simulator's current config: the same on any machine, and for any
    `save_directory` or `lasy_file`. `save()` names an archive by it by default.

    It is of the kind of simulator, the target species, the config without paths or output
    names and the md5 of each input file.
    """
    config = simulator.config()
    configs = {
        name: [
            {
                key: value
                for key, value in archive_config.normalized(item, output_names=False)
                .to_dict()
                .items()
                if key != GIT_HASH_STR
            }
            for item in (value if isinstance(value, (list, tuple)) else [value])
        ]
        for name, value in config.items()
    }
    inputs = [
        (ref.key, ref.file.md5)
        for ref in archive_config.collect_inputs(config, simulator.embedded_inputs)
    ]
    text = json.dumps(
        [simulator.CONFIG_KIND, simulator.target_species, configs, inputs],
        sort_keys=True,
        default=_json_value,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load(
    h5,
    *,
    dummy_run: bool = False,
    working_directory: str | Path | None = None,
    input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
) -> LUMEFBPICModel:
    """Rebuild a model, simulator and actions, from an archive written by `save()`.

    The simulator class is the one registered for the archive's `config_kind`
    (`BaseSimulator.of_kind()`), and its `CONFIG_NAMES` are the groups read as config. The final
    particles and `stats` are back if they were saved; otherwise the recorded output values are
    the simulator's `recorded_outputs`, where the output actions find them. Each action is rebuilt
    from its stored class and parameters; the class must be one of `lume_fbpic.actions`. What is
    loaded is the state `reset()` restores.

    An input file the archive refers to is looked for by basename in `input_dirs` (default the
    current directory); one with another md5 than at `save()` is used with a warning. An input
    file embedded in the archive is used as it is. A config reads its input file when it is
    built, so the file is written out at once: to `working_directory/inputs/<md5>/`, or, without
    a `working_directory`, to a temporary directory that is removed with the simulator (as
    lume-impact runs in a temporary directory), so that loading writes nothing to the current
    directory. `configure()` writes it again under the working directory for a run.

    Parameters
    ----------
    h5 : str, Path, or h5py.File
        The archive.
    dummy_run : bool, optional
        The model's `dummy_run`. Defaults to False.
    working_directory : str or Path, optional
        The simulator's working directory. Defaults to the current directory.
    input_dirs : str, Path or list of them, optional
        Directories to look for referenced input files in. Defaults to the current directory.

    Raises:
        FileNotFoundError: If a referenced input file is not in `input_dirs`.
        ValueError: If the archive has no actions, or is of an unknown kind, or an action class
            is unknown.
    """
    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as g:
            return load(
                g,
                dummy_run=dummy_run,
                working_directory=working_directory,
                input_dirs=input_dirs,
            )
    if _ACTIONS not in h5:
        raise ValueError("The archive has no actions: it was not written by save()")
    outputs = {
        name: value
        for name, value in _read_output_values(h5).items()
        if not math.isnan(value)
    }
    return LUMEFBPICModel(
        _load_simulator(h5, working_directory, input_dirs, outputs),
        _read_actions(h5[_ACTIONS]),
        dummy_run=dummy_run,
    )


def save(
    model: LUMEFBPICModel,
    h5=None,
    *,
    save_final_particles: bool = False,
    input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
):
    """Write `model` to an archive: its simulator's config and the actions, and the values of the
    read-only scalar actions (the outputs).

    An archive written to a path is written to a temporary file first and moved into place when
    it is complete, so an error leaves no partial file and a file that was there stays as it was.
    (A file passed in as a handle is written to directly.)

    Parameters
    ----------
    model : LUMEFBPICModel
        The model to archive.
    h5 : str, Path, or h5py.File, optional
        Destination. If None, `lume_fbpic_<fingerprint>.h5` in the current directory.
    save_final_particles : bool, optional
        Also store the simulator's `final_particles` (group `final_particles`) and `stats` (group
        `stats`). Defaults to False.
    input_dirs : str, Path or list of them, optional
        Directories that hold the input files, by basename. Defaults to the current directory.
        An input file in one of them with another md5 than the one the config names gets a
        warning, and is referred to, not embedded.

    Returns
    -------
    The h5 argument, or the generated filename if it was None.

    Raises:
        FileNotFoundError: If an input file named by the config does not exist: the simulation
            could not run.
        TypeError: If an action has a parameter an archive cannot store; raised before anything
            is written.
        ValueError: If `save_final_particles` is True and there are no final particles (nothing
            has run, or `load_results()` has not been called).
    """
    simulator = model.simulator
    if save_final_particles and simulator.final_particles is None:
        raise ValueError(
            "save_final_particles=True but there are no final particles; run the "
            "simulation or call load_results() first."
        )
    for action in model.supported_variables.values():
        _storable_parameters(action)  # raises before anything is written
    with archive_config.cached_checksums():  # each input file is read once
        config = simulator.config()
        refs = archive_config.collect_inputs(config, simulator.embedded_inputs)
        if h5 is None:
            h5 = f"lume_fbpic_{fingerprint(simulator)}.h5"
        if isinstance(h5, (str, Path)):
            with archive_config.atomic_h5(h5) as g:
                _write_model(g, model, config, refs, save_final_particles, input_dirs)
        else:
            _write_model(h5, model, config, refs, save_final_particles, input_dirs)
    return h5


def _action_from_config(config: dict[str, typing.Any]) -> Action:
    """Rebuild an action from `_action_to_config()`'s output; its class must be one of the
    classes of `lume_fbpic.actions`.

    Raises:
        ValueError: If the class name is not known.
    """
    classes = {
        name: cls
        for name, cls in inspect.getmembers(lume_fbpic.actions, inspect.isclass)
        if issubclass(cls, Action) and cls.__module__ == lume_fbpic.actions.__name__
    }
    name = config["class"]
    if name not in classes:
        raise ValueError(f"Unknown action class {name!r}; known: {sorted(classes)}")
    return classes[name](**config["parameters"])


def _action_to_config(action: Action) -> dict[str, typing.Any]:
    """JSON-compatible description of an action: `{"class": <class name>, "parameters": {...}}`."""
    return {
        "class": type(action).__name__,
        "parameters": action.model_dump(mode="json"),
    }


def _explicit_nones(entry) -> list[str]:
    """The names of the parameters of the archived action `entry` that were given as `None`."""
    return [
        name.decode() if isinstance(name, bytes) else str(name)
        for name in entry.attrs.get("explicit_nones", [])
    ]


def _json_value(value: typing.Any) -> typing.Any:
    """`value` for `json.dumps()` in full: an array as a list (its `str()` elides the middle of a
    large one) and a numpy scalar as a number.

    Raises:
        TypeError: For any other type. Its `str()` could vary from run to run (an object's
            address) and with it the fingerprint.
    """
    if isinstance(value, (numpy.ndarray, numpy.generic)):
        return value.tolist()
    raise TypeError(f"no fingerprint for {type(value).__name__}: {value!r}")


def _load_simulator(
    h5,
    working_directory: str | Path | None,
    input_dirs: typing.Any,
    outputs: dict[str, float],
) -> BaseSimulator:
    """The simulator of the open archive `h5`; see `load()`. `outputs` (the recorded output
    values) are its `recorded_outputs` if the archive holds no final particles."""
    cls = BaseSimulator.of_kind(str(h5.attrs["config_kind"]))
    temporary = (
        None
        if working_directory
        else tempfile.TemporaryDirectory(prefix="lume_fbpic_inputs_")
    )
    try:
        contents = _read_simulator(
            h5,
            cls.CONFIG_NAMES,
            input_dirs,
            Path(working_directory) if working_directory else Path(temporary.name),
        )
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        raise
    if temporary is not None and not contents["embedded"]:
        temporary.cleanup()
        temporary = None
    simulator = cls.from_config(
        contents["config"],
        target_species=contents["target_species"],
        working_directory=working_directory,
    )
    simulator.set_archived_state(
        final_particles=contents["final_particles"],
        stats=contents["stats"],
        recorded_outputs=outputs if contents["final_particles"] is None else {},
        parameters=contents["parameters"],
        embedded_inputs=contents["embedded"],
        inputs_directory=temporary,
    )
    return simulator


def _read_actions(group: h5py.Group) -> list[Action]:
    """The actions of an archive's `actions` group, in the order they were archived.

    Raises:
        ValueError: If an action class is not one of `lume_fbpic.actions`.
    """
    return [
        _action_from_config(
            {
                "class": str(group[key].attrs["class"]),
                "parameters": {
                    **{name: None for name in _explicit_nones(group[key])},
                    **{
                        name: value.item() if hasattr(value, "item") else value
                        for name, value in group[key]["parameters"].attrs.items()
                    },
                },
            }
        )
        for key in sorted(group)
    ]


def _read_output_values(h5) -> dict[str, float]:
    """The recorded output values, by action name, of an open archive; read-only actions without
    a scalar value (the final-particles action) are left out."""
    group = h5[_ACTIONS]
    return {
        str(group[key].attrs["name"]): float(group[key].attrs["value"])
        for key in sorted(group)
        if "value" in group[key].attrs
    }


def _read_simulator(
    h5, config_names: tuple[str, ...], input_dirs: typing.Any, directory: Path
) -> dict[str, typing.Any]:
    """Return the simulator part of an open archive: `config` (the objects or lists of objects
    of `config_names` that it holds), `target_species`, `embedded` (its embedded input files,
    written out under `directory/inputs/`), `parameters` (None if it has none), and
    `final_particles` (None) / `stats` ({}) unless the archive was written with
    `save_final_particles=True`."""
    paths, embedded = archive_config.load_inputs(h5.get(_INPUTS), input_dirs, directory)

    def overrides(name: str, index: int | None) -> dict[str, str]:
        return {
            field: path
            for (config, i, field), path in paths.items()
            if config == name and i == index
        }

    config: dict[str, typing.Any] = {}
    for name in config_names:
        if name not in h5:
            continue  # an optional config the simulator did not have
        group = h5[name]
        if group.attrs.get("list"):
            config[name] = [
                archive_config.read_config(group[str(i)], overrides(name, i))
                for i in range(len(group))
            ]
        else:
            config[name] = archive_config.read_config(group, overrides(name, None))
    final_particles = None
    stats: dict[str, typing.Any] = {}
    if _FINAL_PARTICLES in h5:
        final_particles = ParticleGroup(h5=h5)
        stats = {name: float(value) for name, value in h5[_STATS].attrs.items()}
    return {
        "config": config,
        "embedded": embedded,
        "target_species": str(h5.attrs["target_species"]),
        "final_particles": final_particles,
        "parameters": (
            archive_config.read_config(h5[_PARAMETERS]) if _PARAMETERS in h5 else None
        ),
        "stats": stats,
    }


def _storable_parameters(action: Action) -> dict[str, typing.Any]:
    """The parameters of `action` that an archive stores: those that are set (not `None`), apart
    from `variable_class`, the class name again.

    Raises:
        TypeError: If one is not a bool, int, float or string.
    """
    stored = {}
    for key, value in _action_to_config(action)["parameters"].items():
        if value is None or key == "variable_class":
            continue
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"parameter {key!r} of action {action.name!r} is a "
                f"{type(value).__name__}, which an archive cannot store"
            )
        stored[key] = value
    return stored


def _write_action(entry, action: Action, model: LUMEFBPICModel) -> None:
    """Write one action of `model` to `entry`, a group of an open archive: its name and class,
    its parameters as the attributes of a `parameters` group (those that are set), and, for a
    read-only scalar action, its current value as `value` (NaN where there is no result).
    A parameter that was given as `None`, and not left unset, is listed by name in the
    `explicit_nones` attribute, so that it is `None` again when the archive is loaded (an
    unset `unit` would get its default).

    Raises:
        TypeError: If a parameter is not a bool, int, float or string.
    """
    config = _action_to_config(action)
    entry.attrs["name"] = action.name
    entry.attrs["class"] = config["class"]
    parameters = entry.create_group("parameters")
    for key, value in _storable_parameters(action).items():
        parameters.attrs[key] = value
    explicit_nones = sorted(
        key for key in action.model_fields_set if config["parameters"].get(key) is None
    )
    if explicit_nones:
        entry.attrs["explicit_nones"] = explicit_nones
    if isinstance(action, ReadOnlyActionMixin) and isinstance(action, ScalarVariable):
        entry.attrs["value"] = float(model.get([action.name])[action.name])


def _write_actions(g, model: LUMEFBPICModel) -> None:
    """Write the `actions` group of `model` to the open archive `g`: one numbered entry per
    action, in registration order."""
    actions_group = g.create_group(_ACTIONS)
    for index, action in enumerate(model.supported_variables.values()):
        _write_action(actions_group.create_group(f"{index:04d}"), action, model)


def _write_model(
    g,
    model: LUMEFBPICModel,
    config: dict[str, typing.Any],
    refs: list[archive_config.InputRef],
    save_final_particles: bool,
    input_dirs: typing.Any,
) -> None:
    """Write the archive of `model`, its simulator with config `config` and input files `refs`
    and its actions, into the open file or group `g`; see `save()`."""
    _write_simulator(g, model.simulator, config, refs, save_final_particles, input_dirs)
    _write_actions(g, model)


def _write_simulator(
    g,
    simulator: BaseSimulator,
    config: dict[str, typing.Any],
    refs: list[archive_config.InputRef],
    save_final_particles: bool,
    input_dirs: typing.Any,
) -> None:
    """Write the simulator part of an archive into the open file or group `g`; see `save()`."""
    g.attrs["dataType"] = "lume-fbpic"
    g.attrs["software"] = "lume-fbpic"
    g.attrs["config_kind"] = simulator.CONFIG_KIND
    g.attrs["target_species"] = simulator.target_species

    for name, value in config.items():
        if name in _RESERVED_GROUPS:
            raise ValueError(f"the config {name!r} has the name of an archive group")
        if isinstance(value, (list, tuple)):
            group = g.create_group(name)
            group.attrs["list"] = True
            for i, item in enumerate(value):
                archive_config.write_config(group, str(i), item)
        else:
            archive_config.write_config(g, name, value)
    if refs:
        archive_config.store_inputs(g.create_group(_INPUTS), refs, input_dirs)
    if simulator.parameters is not None:
        archive_config.write_config(g, _PARAMETERS, simulator.parameters)

    if save_final_particles:
        # openPMD root attributes pointing at the `final_particles` group make the archive
        # itself readable as a particle file: `ParticleGroup(h5=<archive path>)`.
        pmd_init(g, basePath="/", particlesPath=_FINAL_PARTICLES)
        simulator.final_particles.write(g.create_group(_FINAL_PARTICLES))
        stats_group = g.create_group(_STATS)
        for name, value in simulator.stats.items():
            stats_group.attrs[name] = value
