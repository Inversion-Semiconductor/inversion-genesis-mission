"""`FBPICSimulator`, the simulator of `LUMEFBPICModel`, on `inversion_fbpic`'s `Simulation`.

It holds the config (the hyperparameters, and lists of lasers, density profiles, particle beams and
external fields), runs it, and owns the lifecycle (`configure()`, `run()`, `reset()`), the results
(`final_particles` and `stats`, read back from fbpic's openPMD output with `lume_fbpic.utils`, and
their `moment_descriptor()`), and the state of a simulator loaded from an archive (its recorded
outputs and input files). Archives are written and read by `lume_fbpic.archive`.
"""

from __future__ import annotations

from pathlib import Path
import attrs
import tempfile
import typing

from beamphysics import ParticleGroup

from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.external_field import _ExternalField
from inversion_fbpic.lib.laser import _LaserPulse
from inversion_fbpic.lib.particle_beam import _ParticleBeam
from inversion_fbpic.lib.simulation import (
    Simulation as _InversionSimulation,
    SimulationHyperparameters,
)
from lume_fbpic import archive_config, descriptor, utils


class FBPICSimulator:
    """Wraps `inversion_fbpic.lib.simulation.Simulation` for use as a LUMEModel simulator.

    Holds the hyperparameters and lists of lasers, density profiles, particle beams and external
    fields, all frozen config objects, and builds a fresh `Simulation` from them on every `run()`:
    a parameter change needs a full PIC rebuild, not an incremental update (unlike Cheetah's cheap
    `track()`). It covers laser-driven (LWFA) and beam-driven (PWFA) runs and both together, and a
    beam with no plasma, for which `hyparams.simulation_length` sets how far the window moves.

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
    hyparams : SimulationHyperparameters
    lasers : sequence of _LaserPulse, optional
    densities : sequence of _DensityProfile, optional
    beams : sequence of _ParticleBeam, optional
    external_fields : sequence of _ExternalField, optional
    target_species : str
        Name of the electron species read back as `final_particles` after a run: a density
        profile's `elec_name` or a beam's `particle_name`.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    # The config objects that are lists, by attribute name
    CONFIG_LISTS: tuple[str, ...] = (
        "lasers",
        "densities",
        "beams",
        "external_fields",
    )
    # All the config objects, in the order of `config()`: `hyparams`, the only one that is not
    # a list, then the lists
    CONFIG_NAMES: tuple[str, ...] = ("hyparams", *CONFIG_LISTS)

    def __init__(
        self,
        hyparams: SimulationHyperparameters,
        *,
        lasers: typing.Sequence[_LaserPulse] = (),
        densities: typing.Sequence[_DensityProfile] = (),
        beams: typing.Sequence[_ParticleBeam] = (),
        external_fields: typing.Sequence[_ExternalField] = (),
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> None:
        self.hyparams = hyparams
        self.lasers = list(lasers)
        self.densities = list(densities)
        self.beams = list(beams)
        self.external_fields = list(external_fields)
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
        self._initial_state: tuple = ()
        # Where `archive.load()` wrote the embedded input files when it was given no working
        # directory; held so that it lasts as long as the simulator.
        self._inputs_directory: tempfile.TemporaryDirectory | None = None
        self._remember_state()

    def config(self) -> dict[str, typing.Any]:
        """The config, by name: each value a SerializableConfig or a list of them. A new dict
        (and new lists) on every call, so it can be kept and later given to `set_config()`.
        """
        return {
            "hyparams": self.hyparams,
            **{n: list(getattr(self, n)) for n in self.CONFIG_LISTS},
        }

    def configure(self) -> None:
        """Check that the config is complete, and write out the input files embedded in the
        archive it was loaded from (`_prepare_inputs()`). `run()` calls it before every run.

        Complete is what `Simulation` needs of the config together: `hyparams`, a laser or a beam,
        a density profile or `hyparams.simulation_length`, and a `target_species` that some
        profile or beam writes.

        Raises:
            ValueError: If one of these is missing.
        """
        if self.hyparams is None:
            raise ValueError("hyparams is required.")
        if not (self.lasers or self.beams):
            raise ValueError("a laser or a particle beam is required.")
        if not self.densities and self.hyparams.simulation_length is None:
            raise ValueError(
                "a density profile, or hyparams.simulation_length, is required."
            )
        if self.target_species not in self.species_names():
            raise ValueError(
                f"target_species {self.target_species!r} is not one of "
                + f"{self.species_names()}, the electron species the config writes."
            )
        self._prepare_inputs()

    def diagnostics_directory(self, working_directory: Path) -> Path:
        """The openPMD directory a run executed under `working_directory` writes."""
        # POSIT: fbpic writes its openPMD files in the subdirectory `hdf5` of a diagnostic's
        # `write_dir`, which is `save_directory`.
        return working_directory / self.hyparams.save_directory / "hdf5"

    @classmethod
    def from_config(
        cls,
        config: dict[str, typing.Any],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> "FBPICSimulator":
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

    def reset(self) -> None:
        """Restore the config, results and recorded outputs the simulator started with.

        That is the state when it was constructed or, for one loaded from an archive, the
        archived state. Nothing is run -- a full fbpic run is far too expensive to trigger
        implicitly from `reset()` (unlike Cheetah's cheap eager re-`track()`).
        """
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

        A run replaces the dumps of the previous run in the same working directory: the
        `data*.h5` files in `diagnostics_directory()` are removed first. The results are read
        from the last dump there, so a dump left by a longer earlier run would otherwise be read
        as this run's. Nothing else is removed. A script that calls `run_simulation()` itself,
        on every MPI rank, does not get this, and should run in a fresh working directory.
        """
        self.configure()
        self._remove_dumps()
        self.run_simulation()
        self.load_results()

    def run_simulation(self) -> _InversionSimulation:
        """Build a fresh `Simulation` and run it, WITHOUT reading any output back.

        This is the part of `run()` every MPI rank executes; `load_results()` reads the
        diagnostics afterward, once, wherever they are visible. Passes `skip_if_hashed=False` to
        both `setup_simulation()` and `run_simulation()`. An antenna laser with no `v_antenna` is
        run with `v_antenna=0.0`, fbpic's default (`_antenna_at_rest()`).
        """
        simulation = _InversionSimulation(
            elements=[
                self.hyparams,
                *(_antenna_at_rest(x) for x in self.lasers),
                *self.densities,
                *self.beams,
                *self.external_fields,
            ]
        )
        simulation.setup_simulation(
            working_directory=self.working_directory, skip_if_hashed=False
        )
        simulation.run_simulation(skip_if_hashed=False)
        return simulation

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
        """Replace the config with a `config()`-shaped dict; a list left out is an empty list."""
        self.hyparams = config["hyparams"]
        for name in self.CONFIG_LISTS:
            setattr(self, name, list(config.get(name, ())))

    def species_names(self) -> list[str]:
        """The names of the electron species the config writes: each density profile's
        `elec_name` and each beam's `particle_name` that is set."""
        return [
            n
            for n in (
                *(d.elec_name for d in self.densities),
                *(b.particle_name for b in self.beams),
            )
            if n is not None
        ]

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
        restores. `__init__` calls it last."""
        self._initial_state = (
            self.config(),
            self.final_particles,
            dict(self.stats),
            dict(self.recorded_outputs),
        )

    def _remove_dumps(self) -> None:
        """Remove the openPMD dumps (`data*.h5`) in `diagnostics_directory()` of the working
        directory, which a previous run left there."""
        for dump in self.diagnostics_directory(self.working_directory).glob("data*.h5"):
            dump.unlink()

    def _update_output(self, diags_path: Path) -> None:
        """Read `final_particles` and `stats` from the diagnostics in `diags_path`; they replace
        any `recorded_outputs`."""
        self.final_particles, self.stats = utils.load_final_particles(
            diags_path, self.target_species
        )
        self.recorded_outputs = {}


def _antenna_at_rest(laser: _LaserPulse) -> _LaserPulse:
    """`laser`, or for an antenna-injected laser with no `v_antenna` a copy with `v_antenna=0.0`.

    `_LaserPulse.v_antenna` defaults to None, which `Simulation` passes to fbpic as it is, but
    fbpic's `add_laser_pulse` takes a number (its default is 0) and fails on None in every frame:
    a boosted antenna refuses any velocity but 0, a lab-frame one adds it to its particles'
    velocity, and a LASY profile refuses any but 0. The copy is given only to the run, so the
    config, its archive and its fingerprint keep the None.
    """
    if laser.method != "antenna" or laser.v_antenna is not None:
        return laser
    return utils.evolve_laser(laser, v_antenna=0.0)
