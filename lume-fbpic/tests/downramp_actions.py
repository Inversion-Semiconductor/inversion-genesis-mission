"""The HTU downramp-injection action set: `make_actions(simulator)`.

An action list for an `LWFASimulator` shaped like `demo_downramp_simulation`: one
`GaussianLaserPulse` and two `SmoothSineFlattop` density profiles (index 0 the flattop, index 1 the
downramp). Hardcoded to that shape; `docs/examples/lwfa.py` and `ionization_injection.py` show lists
built for other shapes from the same building-block action classes (`lume_fbpic.actions`). A test
helper: the fixtures in `conftest.py` use this set.
"""

from __future__ import annotations

from lume.actions import Action

from lume_fbpic.actions import (
    DensityFieldAction,
    FinalParticlesAction,
    LaserFieldAction,
    make_stat_actions,
)
from lume_fbpic.lwfa_simulator import LWFASimulator


def make_actions(simulator: LWFASimulator) -> list[Action]:
    """Build the HTU downramp-injection control/output variable set.

    Matches `demo_downramp_simulation`'s config shape: one `GaussianLaserPulse` and two
    `SmoothSineFlattop` density profiles (index 0 = flattop, index 1 = downramp).

    Control variables (writable): laser energy, focal position, temporal width; flattop and
    downramp nominal densities; downramp length.
    Outputs (read-only): charge, mean/std energy, and `final_particles`.
    """
    return [
        LaserFieldAction(name="laser_energy", field_name="energy"),
        LaserFieldAction(name="laser_focal_position", field_name="focal_position"),
        LaserFieldAction(name="laser_temporal_width", field_name="tau_fwhm"),
        DensityFieldAction(
            name="flattop_density",
            density_index=0,
            field_name="nominal_density",
        ),
        DensityFieldAction(
            name="downramp_density",
            density_index=1,
            field_name="nominal_density",
        ),
        DensityFieldAction(
            name="downramp_length",
            density_index=1,
            field_name="downramp_length",
        ),
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]
