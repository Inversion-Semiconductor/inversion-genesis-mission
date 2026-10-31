# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""State containers of the functional stepper (inversion plan Section 4.2).

`StepState` holds the complete primal state of a lab-frame, single-GPU
simulation. The production `Simulation` object owns the field arrays, the
particle arrays, the window, injection, and tracking scalars, and the clock;
`StepState` refers to it and adds what production does not hold: the
expected-value (EV) ionization arrays of the nitrogen ions, the exchange
bookkeeping, and the cumulative emission ledger. `step` updates the state in
place and returns the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EV_ARRAY_NAMES = ("pop5", "pop6", "pop7", "w_pending")


@dataclass
class EVArrays:
    """Per-ion EV arrays, aligned with the arrays of the ion species.

    pop5, pop6, pop7 are the population fractions of N5+, N6+, and N7+.
    w_pending is the birth weight W that the ion has not emitted yet.
    Every sort and exchange keeps them aligned with the ion tracker IDs.
    """

    pop5: object
    pop6: object
    pop7: object
    w_pending: object

    def arrays(self):
        return [getattr(self, name) for name in EV_ARRAY_NAMES]


@dataclass
class StepDecisions:
    """Discrete decisions of one step that the FD harness needs (Section 4.5).

    Attributes:
        iteration: Iteration at the start of the step.
        exchanged: Whether the step began with a particle exchange.
        injected: Injected macroparticles per species name at that exchange.
        n_move: Cells by which the moving window moved in this step.
        emitted_parent_ids: Device array of the tracker IDs of the ions that
            emitted an EV electron in this step, or None.
    """

    iteration: int
    exchanged: bool = False
    injected: dict = field(default_factory=dict)
    n_move: int = 0
    emitted_parent_ids: object = None


@dataclass
class StepState:
    """Complete primal state of the stepper.

    Attributes:
        sim: Production `Simulation`; owns fields, particles, window, clock.
        species: Species by name; `ev_species` and `ev_target` name the EV
            ions and the species that receives their electrons.
        ionization: "none" or "ev".
        ev: EV arrays of the ion species, or None.
        ev_params: Host ADK coefficients and the emission settings, or None.
        last_exchange: Iteration of the latest particle exchange.
        ledger: Cumulative host scalars of the EV invariants (Phase P3).
    """

    sim: object
    species: dict
    ionization: str = "none"
    ev_species: str | None = None
    ev_target: str | None = None
    ev: EVArrays | None = None
    ev_params: dict | None = None
    last_exchange: int = -1
    ledger: dict = field(default_factory=dict)

    @property
    def iteration(self) -> int:
        return int(self.sim.iteration)

    @property
    def time(self) -> float:
        return float(self.sim.time)

    def species_by_object(self) -> list:
        """(name, species) pairs in the order of `sim.ptcl`."""
        names = {id(sp): name for name, sp in self.species.items()}
        return [
            (names.get(id(sp), f"species{i}"), sp) for i, sp in enumerate(self.sim.ptcl)
        ]


@dataclass
class TangentState:
    """Tangents of the differentiable arrays, leading dimension K (Phase P5)."""

    K: int
    fields: dict = field(default_factory=dict)
    particles: dict = field(default_factory=dict)
    ev: dict = field(default_factory=dict)


def host(array) -> np.ndarray:
    """Host copy of a NumPy or CuPy array."""
    import cupy

    return cupy.asnumpy(array)
