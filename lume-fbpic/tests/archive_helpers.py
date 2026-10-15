"""Helpers of the archive tests: archive a simulator alone, load one back, read the recorded
outputs of an archive."""

from __future__ import annotations

import h5py

from lume_fbpic import archive
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import BaseSimulator


def load_simulator(*args, **kwargs) -> BaseSimulator:
    """The simulator of the model `archive.load()` rebuilds."""
    return archive.load(*args, **kwargs).simulator


def recorded_values(path) -> dict[str, float]:
    """The recorded output values of the archive at `path`."""
    with h5py.File(path, "r") as f:
        return archive._read_output_values(f)


def simulator_model(simulator: BaseSimulator) -> LUMEFBPICModel:
    """`simulator` in a model without actions, to archive the simulator alone."""
    return LUMEFBPICModel(simulator, [], dummy_run=True)
