"""`BaseSimulator`, the abstract simulator of `LUMEFBPICModel`.

It owns what does not depend on the kind of simulation: the lifecycle (`configure()`, `run()`,
`reset()`), the results (`final_particles` and `stats`, read back from fbpic's openPMD output with
`lume_fbpic.utils`, and their `moment_descriptor()`), and the state of a simulator loaded from an
archive (its recorded outputs and input files). A subclass
names its config objects (`CONFIG_NAMES`) and supplies the run:
`LWFASimulator` (`lume_fbpic.lwfa_simulator`) for LWFA, and `PWFASimulator`
(`lume_fbpic.pwfa_simulator`) for beam-driven PWFA. Archives are written and read by
`lume_fbpic.archive`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import attrs
import tempfile
import typing

from beamphysics import ParticleGroup

from inversion_fbpic.lib.datapoint import Parameters
from lume_fbpic import archive_config, descriptor, utils


class BaseSimulator(ABC):
    """What every lume-fbpic simulator shares, whatever it simulates.

    A subclass holds its config as SerializableConfig objects, one attribute each (a list of
    them for some), names them in `CONFIG_NAMES` (with `OPTIONAL_CONFIG_NAMES` and
    `SAVE_DIRECTORY_CONFIG`), takes them as keyword arguments of its constructor and provides
    `run_simulation()`. It also sets `CONFIG_KIND`, the name written to its archives: it is
    registered under that name, which is how `lume_fbpic.archive.load()` finds the right class.

    `parameters` is an optional `inversion_fbpic` `Parameters`: a dict of values that describe
    where the config came from (for example the dataset record it was rebuilt from). It is
    archived and loaded with the config, but it is not config: nothing reads it to run, it is
    not in the fingerprint, and `reset()` leaves it as it is.

    `recorded_outputs` are output values recorded without the particles they came from (an
    archive of a reconstructed run), by action name; the output actions read them while there
    are no `final_particles`. Results read back from a run replace them.

    `embedded_inputs` are the input files embedded in the archive the simulator was loaded from,
    by (config, index, field); see `lume_fbpic.archive_config`.

    Parameters
    ----------
    target_species : str
        Name of the electron species to read back as `final_particles` after a run.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    CONFIG_KIND: str = ""
    # The config objects, by attribute name, in the order of `config()`.
    CONFIG_NAMES: tuple[str, ...] = ()
    # Those of `CONFIG_NAMES` that may be None; `config()` leaves them out then.
    OPTIONAL_CONFIG_NAMES: tuple[str, ...] = ()
    # The one of `CONFIG_NAMES` whose `save_directory` a run writes its diagnostics under.
    SAVE_DIRECTORY_CONFIG: str = ""
    _registry: dict[str, type["BaseSimulator"]] = {}

    def __init__(
        self,
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> None:
        self.target_species = target_species
        self.working_directory = (
            Path(working_directory) if working_directory else Path.cwd()
        )
        self.final_particles: ParticleGroup | None = None
        self.stats: dict[str, typing.Any] = {}
        self.parameters: Parameters | None = None
        self.recorded_outputs: dict[str, float] = {}
        self.embedded_inputs: dict[
            tuple[str, int | None, str], archive_config.InputFile
        ] = {}
        # (final_particles, {selection: descriptor}); see `moment_descriptor()`
        self._descriptor_cache: (
            tuple[typing.Any, dict[tuple, dict[str, float]]] | None
        ) = None
        # The config, results and recorded outputs `reset()` restores; see `_remember_state()`.
        self._initial_state: tuple | None = None
        # Where `archive.load()` wrote the embedded input files when it was given no working
        # directory; held so that it lasts as long as the simulator.
        self._inputs_directory: tempfile.TemporaryDirectory | None = None

    def __init_subclass__(cls, **kwargs: typing.Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.CONFIG_KIND:
            BaseSimulator._registry[cls.CONFIG_KIND] = cls

    def config(self) -> dict[str, typing.Any]:
        """The config, by name: each value a SerializableConfig or a list of them. A new dict
        (and new lists) on every call, so it can be kept and later given to `set_config()`.
        """
        config = {}
        for name in self.CONFIG_NAMES:
            value = getattr(self, name)
            if value is not None or name not in self.OPTIONAL_CONFIG_NAMES:
                config[name] = list(value) if isinstance(value, list) else value
        return config

    def configure(self) -> None:
        """Validate that the current config is complete, and write out the input files embedded
        in the archive it was loaded from (`_prepare_inputs()`). `run()` calls it before every
        run.

        Raises:
            ValueError: If a config object that is not optional is missing (or an empty list).
        """
        for name in self.CONFIG_NAMES:
            value = getattr(self, name)
            if name not in self.OPTIONAL_CONFIG_NAMES and (
                value is None or value == []
            ):
                raise ValueError(f"{name} is required.")
        self._prepare_inputs()

    def diagnostics_directory(self, working_directory: Path) -> Path:
        """The openPMD directory a run executed under `working_directory` writes."""
        # POSIT: fbpic writes its openPMD files in the subdirectory `hdf5` of a diagnostic's
        # `write_dir`, which is `save_directory`.
        return (
            working_directory
            / getattr(self, self.SAVE_DIRECTORY_CONFIG).save_directory
            / "hdf5"
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, typing.Any],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> "BaseSimulator":
        """Build a simulator from a `config()`-shaped dict."""
        return cls(
            **config,
            target_species=target_species,
            working_directory=working_directory,
        )

    def load_results(self, working_directory: str | Path | None = None) -> None:
        """Read `final_particles` and `stats` from a finished run's diagnostics.

        Reads the directory `diagnostics_directory()` names for `working_directory` (default:
        this simulator's own), for runs executed elsewhere -- for example a batch job -- or by
        `run()` itself.
        """
        directory = (
            Path(working_directory) if working_directory else self.working_directory
        )
        self._update_output(self.diagnostics_directory(directory))

    def moment_descriptor(
        self,
        uz_min: float | None,
        central_fraction: float | None,
        longitudinal_bins: int,
    ) -> dict[str, float]:
        """`lume_fbpic.descriptor.compute()` of `final_particles` for one selection; empty if
        there are no particles or fewer than two are selected.

        Kept for the current `final_particles`, one descriptor per selection, so that the
        actions of all its features share one computation.
        """
        particles = self.final_particles
        if particles is None:
            return {}
        if self._descriptor_cache is None or self._descriptor_cache[0] is not particles:
            self._descriptor_cache = (particles, {})
        descriptors = self._descriptor_cache[1]
        key = (uz_min, central_fraction, longitudinal_bins)
        if key not in descriptors:
            try:
                descriptors[key] = descriptor.compute(
                    particles,
                    uz_min=uz_min,
                    central_fraction=central_fraction,
                    longitudinal_bins=longitudinal_bins,
                )
            except ValueError:  # fewer than two usable particles after the selection
                descriptors[key] = {}
        return descriptors[key]

    @classmethod
    def of_kind(cls, kind: str) -> type["BaseSimulator"]:
        """The simulator class whose `CONFIG_KIND` is `kind`.

        Raises:
            ValueError: If no class has it.
        """
        if kind not in BaseSimulator._registry:
            raise ValueError(f"unknown kind {kind!r} of simulator")
        return BaseSimulator._registry[kind]

    def reset(self) -> None:
        """Restore the config, results and recorded outputs the simulator started with.

        That is the state when it was constructed or, for one loaded from an archive, the
        archived state. Nothing is run -- a full fbpic run is far too expensive to trigger
        implicitly from `reset()` (unlike Cheetah's cheap eager re-`track()`).

        Raises:
            RuntimeError: If the subclass never recorded a starting state (`_remember_state()`).
        """
        if self._initial_state is None:
            raise RuntimeError(
                "the simulator has no starting state; call _remember_state()"
            )
        config, self.final_particles, stats, recorded = self._initial_state
        self.set_config(config)
        self.stats = dict(stats)
        self.recorded_outputs = dict(recorded)

    def run(self) -> None:
        """Run the simulation from the current config, then read the results back.

        The config is checked with `configure()` first, on every run (it raises `ValueError` if the
        config is incomplete): it can have changed since the last one, through `set_config()`, an
        action or a new `working_directory`, and `configure()` is cheap next to a run.

        A `run()` call means "run this now", not "skip if a previous run in this working
        directory already matches this config" (that skip-if-hashed default is meant for
        repeated script invocations, not a live model's `set()`).
        """
        self.configure()
        self.run_simulation()
        self.load_results()

    @abstractmethod
    def run_simulation(self) -> typing.Any:
        """Run the simulation WITHOUT reading any output back.

        This is the part of `run()` every MPI rank executes; `load_results()` reads the
        diagnostics afterward, once, wherever they are visible.
        """

    def set_archived_state(
        self,
        *,
        final_particles: ParticleGroup | None,
        stats: dict[str, typing.Any],
        recorded_outputs: dict[str, float],
        parameters: Parameters | None,
        embedded_inputs: dict,
        inputs_directory: tempfile.TemporaryDirectory | None,
    ) -> None:
        """Take the results and metadata of an archive, which `lume_fbpic.archive.load()` read,
        as the state `reset()` restores. `inputs_directory` holds the embedded input files when
        the archive was loaded without a working directory; it lasts as long as the simulator.
        """
        self.final_particles = final_particles
        self.stats = dict(stats)
        self.recorded_outputs = dict(recorded_outputs)
        self.parameters = parameters
        self.embedded_inputs = embedded_inputs
        self._inputs_directory = inputs_directory
        self._remember_state()

    def set_config(self, config: dict[str, typing.Any]) -> None:
        """Replace the config with a `config()`-shaped dict."""
        for name in self.CONFIG_NAMES:
            value = (
                config.get(name) if name in self.OPTIONAL_CONFIG_NAMES else config[name]
            )
            setattr(self, name, list(value) if isinstance(value, list) else value)

    def _prepare_inputs(self) -> None:
        """Write the embedded input files under `working_directory/inputs/<md5>/` and point the
        config at them, for those not already there. `configure()` calls it, so that a run in
        another working directory than the one the archive was loaded in finds them. A field that
        was pointed at another existing file since is left as it is."""
        if not self.embedded_inputs:
            return
        config = self.config()
        changed = False
        for (name, index, field), file in self.embedded_inputs.items():
            item = config[name] if index is None else config[name][index]
            current = Path(getattr(item, field))
            if (
                file.extracted is not None
                and current != file.extracted
                and current.is_file()
            ):
                continue  # the config was pointed at another file on purpose
            target = file.write_to(self.working_directory / "inputs")
            if Path(getattr(item, field)) == target:
                continue
            item = attrs.evolve(item, **{field: target})
            if index is None:
                config[name] = item
            else:
                config[name][index] = item
            changed = True
        if changed:
            self.set_config(config)

    def _remember_state(self) -> None:
        """Record the current config, results and recorded outputs as the state `reset()`
        restores. A subclass calls it at the end of its `__init__`."""
        self._initial_state = (
            self.config(),
            self.final_particles,
            dict(self.stats),
            dict(self.recorded_outputs),
        )

    def _update_output(self, diags_path: Path) -> None:
        """Read `final_particles` and `stats` from the diagnostics in `diags_path`; they replace
        any `recorded_outputs`."""
        self.final_particles, self.stats = utils.load_final_particles(
            diags_path, self.target_species
        )
        self.recorded_outputs = {}
