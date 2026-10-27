"""`PWFASimulator`: the beam-driven (PWFA) simulator of `LUMEFBPICModel`, built on fbpic directly.

It holds a `PWFAGrid`, a plasma density profile and the driver and witness bunches, and builds a
fresh fbpic `Simulation` from them for every run. Everything that does not depend on the kind of
simulation is `BaseSimulator`'s.
"""

from __future__ import annotations

from pathlib import Path
import typing

import numpy
from scipy.constants import c

from inversion_fbpic.lib.density_core import _DensityProfile
from lume_fbpic.pwfa_config import PWFAGrid, _ParticleBunch
from lume_fbpic.simulator import BaseSimulator


class PWFASimulator(BaseSimulator):
    """A lab-frame, beam-driven plasma wakefield simulation, built directly on fbpic.

    Holds a `PWFAGrid`, a plasma density profile (`species=None` for bare electrons), a driver
    bunch and optionally a witness bunch, all frozen config objects, and rebuilds a fresh fbpic
    `Simulation` on every run. The plasma, the driver and the witness are written to the
    diagnostics as the species `plasma`, `driver` and `witness`; `target_species` is the one
    read back as `final_particles`.

    It runs in the lab frame only: there is no boosted-frame option (`gamma_boost`), as
    `LWFASimulator` has through `SimulationHyperparameters`. A boost is what would make long
    plasmas practical, such as the 20 cm cell of `docs/examples/pwfa_clara_febe.py`. It is left
    for when beams are added to `inversion_fbpic`'s `Simulation`, which already handles boosted
    frames and back-transformed diagnostics, and this class merges with `LWFASimulator`.

    Parameters
    ----------
    grid : PWFAGrid
    plasma : _DensityProfile
    driver : _ParticleBunch
    witness : _ParticleBunch, optional
    target_species : str
        `driver` or `witness`.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    CONFIG_KIND = "pwfa"
    CONFIG_NAMES = ("grid", "plasma", "driver", "witness")
    OPTIONAL_CONFIG_NAMES = ("witness",)
    SAVE_DIRECTORY_CONFIG = "grid"

    def __init__(
        self,
        grid: PWFAGrid,
        plasma: _DensityProfile,
        driver: _ParticleBunch,
        witness: _ParticleBunch | None = None,
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__(
            target_species=target_species, working_directory=working_directory
        )
        self.grid = grid
        self.plasma = plasma
        self.driver = driver
        self.witness = witness
        self._remember_state()

    def configure(self) -> None:
        """`BaseSimulator.configure()`, and check that `target_species` is one of the bunches."""
        if self.target_species not in self._species_names():
            raise ValueError(
                f"target_species {self.target_species!r} is not one of {self._species_names()}"
            )
        super().configure()

    def run_simulation(self) -> typing.Any:
        """Build a fresh fbpic `Simulation` and run it, WITHOUT reading any output back.

        A `random_seed` seeds numpy's global generator, which draws the bunches; the state of the
        generator from before the run is restored when it ends.
        """
        state = numpy.random.get_state() if self.grid.random_seed is not None else None
        try:
            return self._run_seeded()
        finally:
            if state is not None:
                numpy.random.set_state(state)

    def _run_seeded(self) -> typing.Any:
        """Seed the generator, then build a fresh fbpic `Simulation` and run it."""
        from fbpic.fields.smoothing import BinomialSmoother
        from fbpic.main import Simulation

        grid = self.grid
        if grid.random_seed is not None:
            numpy.random.seed(grid.random_seed)
        # The smoother is always passed: fbpic replaces `smoother=None` by its one-pass default,
        # so zero passes, no smoothing, has to be a filter of its own
        smoother = BinomialSmoother(
            n_passes={"z": grid.smoother_passes, "r": grid.smoother_passes},
            compensator={
                "z": grid.smoother_compensator,
                "r": grid.smoother_compensator,
            },
        )
        simulation = Simulation(
            grid.nz,
            grid.zmax,
            grid.nr,
            grid.rmax,
            grid.nm,
            grid.time_step,
            zmin=grid.zmin,
            boundaries=grid.boundaries,
            use_cuda=grid.use_cuda,
            smoother=smoother,
        )
        species = {}
        species["plasma"], _ = self.plasma.add_to_simulation(
            simulation, is_boosted=False
        )
        species["driver"] = self.driver.add_to_simulation(simulation)
        if self.witness is not None:
            species["witness"] = self.witness.add_to_simulation(simulation)
        simulation.set_moving_window(v=c)
        write_dir = str(self.working_directory / grid.save_directory)

        written = {
            k: v for k, v in species.items() if k != "plasma" or grid.write_plasma
        }
        # fbpic writes a dump at the start of a step, so `step(n)` never writes iteration n. The
        # periodic dumps stop before `n_steps`, and one more step writes the dump of iteration
        # `n_steps`, the end of the run, whatever `write_period` is; its push is discarded.
        simulation.diags = [
            *_diagnostics(
                simulation,
                written,
                write_dir,
                grid.write_period,
                iteration_max=grid.n_steps,
            ),
            *_diagnostics(
                simulation,
                written,
                write_dir,
                grid.n_steps,
                iteration_min=grid.n_steps,
            ),
        ]
        simulation.step(grid.n_steps + 1)
        return simulation

    def _species_names(self) -> list[str]:
        return ["driver", "witness"] if self.witness is not None else ["driver"]


def _diagnostics(
    simulation: typing.Any,
    species: dict[str, typing.Any],
    write_dir: str,
    period: int,
    **iterations: int,
) -> list[typing.Any]:
    """The field and particle diagnostics of a run, every `period` steps within `iterations`
    (fbpic's `iteration_min` and `iteration_max`)."""
    from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

    return [
        FieldDiagnostic(
            period,
            simulation.fld,
            comm=simulation.comm,
            write_dir=write_dir,
            **iterations,
        ),
        ParticleDiagnostic(
            period,
            species,
            comm=simulation.comm,
            write_dir=write_dir,
            **iterations,
        ),
    ]
