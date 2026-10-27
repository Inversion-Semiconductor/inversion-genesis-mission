"""`LWFASimulator`: the LWFA simulator of `LUMEFBPICModel`, on `inversion_fbpic`'s `Simulation`.

It holds the hyperparameters, laser and density-profile config objects, and builds a fresh
`inversion_fbpic.lib.simulation.Simulation` from them for every run. Everything that does not depend
on the kind of simulation is `BaseSimulator`'s.
"""

from __future__ import annotations

from pathlib import Path

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.laser import _LaserPulse
from inversion_fbpic.lib.simulation import (
    Simulation as FBPICSimulation,
    SimulationHyperparameters,
)
from lume_fbpic.simulator import BaseSimulator


class LWFASimulator(BaseSimulator):
    """Wraps `inversion_fbpic.lib.simulation.Simulation` for use as a LUMEModel simulator.

    Holds the current hyperparameters/laser/density-profile config objects and rebuilds a
    fresh `inversion_fbpic.lib.simulation.Simulation` on every `run()` call -- matches how
    fbpic simulations actually work: a parameter change requires a full PIC rebuild, not an
    incremental update (unlike e.g. Cheetah's cheap `track()`).

    Parameters
    ----------
    hyparams : SimulationHyperparameters
    laser : _LaserPulse
    densities : list[_DensityProfile]
    target_species : str
        Name of the electron species (matching a density profile's `elec_name`) to read back
        as `final_particles` after a run.
    working_directory : str | Path, optional
        Directory fbpic writes diagnostics under. Defaults to the current working directory.
    """

    CONFIG_KIND = "lwfa"
    CONFIG_NAMES = ("hyparams", "laser", "densities")
    SAVE_DIRECTORY_CONFIG = "hyparams"

    def __init__(
        self,
        hyparams: SimulationHyperparameters,
        laser: _LaserPulse,
        densities: list[_DensityProfile],
        *,
        target_species: str,
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__(
            target_species=target_species, working_directory=working_directory
        )
        self.hyparams = hyparams
        self.laser = laser
        self.densities = list(densities)
        self._remember_state()

    def run_simulation(self) -> FBPICSimulation:
        """Build a fresh `Simulation` and run it, WITHOUT reading any output back.

        Passes `skip_if_hashed=False` to both `setup_simulation()` and `run_simulation()`.
        """
        simulation = FBPICSimulation(
            elements=[self.hyparams, self.laser, *self.densities]
        )
        simulation.setup_simulation(
            working_directory=self.working_directory, skip_if_hashed=False
        )
        simulation.run_simulation(skip_if_hashed=False)
        return simulation
