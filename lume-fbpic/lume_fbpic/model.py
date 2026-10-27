"""LUMEFBPICModel(ActionModel) wrapping a lume-fbpic simulator."""

from __future__ import annotations

from beamphysics import ParticleGroup
from lume.actions import Action, ActionModel
from lume.staged_model import FinalParticlesMixIn
from lume_fbpic.simulator import BaseSimulator
from pathlib import Path
import typing


class LUMEFBPICModel(FinalParticlesMixIn, ActionModel[BaseSimulator]):
    """LUMEModel using the actions framework, wrapping a `BaseSimulator`
    (`LWFASimulator`, or another subclass).

    No `InitialParticlesMixIn` -- the plasma-accelerator stage makes its own bunch (by injection
    off a density downramp or dopant, or from a driver bunch's wake), it doesn't consume an
    externally-supplied one. It is the first stage of a `StagedModel` chain.

    Like `LUMECheetahModel` and `LUMEImpactModel`, `set()` applies the new parameter values and
    then runs the simulation (checking the config with `configure()` first), so `get()`
    afterwards returns outputs for those inputs; `set({})` runs it without changing anything. A
    real fbpic run costs minutes to hours, so pass `dummy_run=True` (as `LUMEImpactModel`
    allows) to make `set()` only update parameters without running -- useful for building up a
    configuration, or in tests. A `set()` that raises, in an action or in the run, leaves the
    config as it was, so the outputs keep matching it.

    `archive()` writes the model to an archive and `from_archive()` rebuilds one, as lume-base's
    `archive()` and `from_archive()` do; `lume_fbpic.archive` holds the format. The outputs are
    the last results, recorded or computed: with `dummy_run=True`, setting an input changes the
    config only and they stay as they are, and a run replaces them.

    `reset()` restores the simulator's starting state (for a model loaded from an archive, the
    archived config and results); it never runs the simulation.

    An action made without a `unit` gets its default one: see `lume_fbpic.actions`.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        actions: list[Action],
        dummy_run: bool = False,
    ) -> None:
        _check_simulator(simulator, actions)
        super().__init__(
            simulator=simulator,
            action_variables=_with_default_units(simulator, actions),
        )
        self.dummy_run = dummy_run

    def archive(
        self,
        h5=None,
        *,
        save_final_particles: bool = False,
        input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
    ):
        """Write the model to an archive: `lume_fbpic.archive.save()`, which describes the
        arguments. Returns `h5`, or the name of the file written if it was None."""
        from lume_fbpic import archive  # which imports this module

        return archive.save(
            self, h5, save_final_particles=save_final_particles, input_dirs=input_dirs
        )

    @property
    def final_particles(self) -> ParticleGroup | None:
        """The last run's final electron bunch (`BaseSimulator.final_particles`), or None."""
        return self.simulator.final_particles

    @classmethod
    def from_archive(
        cls,
        h5,
        *,
        dummy_run: bool = False,
        working_directory: str | Path | None = None,
        input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
        strict: bool = False,
    ) -> "LUMEFBPICModel":
        """The model an archive holds: `lume_fbpic.archive.load()`, which describes the
        arguments."""
        from lume_fbpic import archive  # which imports this module

        return archive.load(
            h5,
            dummy_run=dummy_run,
            working_directory=working_directory,
            input_dirs=input_dirs,
            strict=strict,
        )

    def reset(self) -> None:
        """Restore the simulator's starting state: its config and results when it was built or,
        for a model loaded from an archive, the archived ones (see `BaseSimulator.reset()`).

        Replaces `ActionModel.reset()`, which sets every writable action to its
        `default_value` -- `None` for all of ours, which fails. Nothing is run.
        """
        self.simulator.reset()

    def _set(self, values: dict[str, typing.Any]) -> None:
        # A set that fails -- an action refusing its value after earlier ones were applied, or the
        # run itself -- restores the config, so that it still matches the outputs of the last run
        config = self.simulator.config()
        try:
            super()._set(values)
            if not self.dummy_run:
                self.simulator.run()
        except BaseException:
            self.simulator.set_config(config)
            raise


def _check_simulator(simulator: BaseSimulator, actions: list[Action]) -> None:
    """Check that each action works on `simulator`, an instance of its `SIMULATOR`.

    Raises:
        TypeError: For an action of another kind of simulator.
    """
    for action in actions:
        required = getattr(action, "SIMULATOR", None)
        if required is not None and not isinstance(simulator, required):
            raise TypeError(
                f"action {action.name!r} needs a {required.__name__}, "
                f"not a {type(simulator).__name__}"
            )


def _with_default_units(
    simulator: BaseSimulator, actions: list[Action]
) -> list[Action]:
    """`actions`, each made without a `unit` replaced by a copy with the one its
    `default_unit()` finds; the actions given are left as they are, so a list can serve several
    models. An action with a `unit` passed to it, `None` included, is kept as it is.
    """
    result = []
    for action in actions:
        default_unit = getattr(action, "default_unit", None)
        unit = (
            default_unit(simulator)
            if default_unit is not None and "unit" not in action.model_fields_set
            else None
        )
        result.append(
            action if unit is None else action.model_copy(update={"unit": unit})
        )
    return result
