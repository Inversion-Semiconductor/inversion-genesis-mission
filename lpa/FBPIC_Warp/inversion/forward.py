"""Forward runs of the twin and their beam statistics.

`run_production` uses production `Simulation.step` (stochastic or no
ionization). The EV model runs through the autodiff stepper (Phase P3).
"""

from __future__ import annotations

import time

import numpy as np
from twin_config import build_twin, species_arrays

PARTICLE_NAMES = ("x", "y", "z", "ux", "uy", "uz", "w")


def device_memory() -> dict:
    """Used device memory now [bytes] and the CUDA mempool high-water marks."""
    import cupy
    from fbpic.utils.warp import _mempool_attribute

    free, total = cupy.cuda.runtime.memGetInfo()
    out = {"device_used": total - free}
    if cupy.cuda.runtime.deviceGetAttribute(
        cupy.cuda.runtime.cudaDevAttrMemoryPoolsSupported, 0
    ):
        out["pool_used_high"] = _mempool_attribute("UsedMemHigh")
        out["pool_reserved_high"] = _mempool_attribute("ReservedMemHigh")
    return out


def run_production(
    theta,
    config,
    ionization="stochastic",
    seed=None,
    rng_seed=None,
    p_nz=1,
    num_steps=None,
):
    """Run production `Simulation.step` to the observation step.

    Args:
        theta: (E_L [J], z_f [m]).
        config: Loaded configuration dict.
        ionization: "stochastic" or "none".
        seed: NumPy injection seed (default: config).
        rng_seed: Philox seed of the ionizer (default: config).
        p_nz: Particles per cell along z.
        num_steps: Steps to run (default: config `num_steps`).

    Returns:
        A dict with the host arrays of `elec_from_N`, `z_win`, species
        counts, the wall time, and the device memory (the mempool high-water
        marks are per process).
    """
    import cupy

    twin = build_twin(
        theta, config, ionization=ionization, seed=seed, rng_seed=rng_seed, p_nz=p_nz
    )
    sim = twin.sim
    n_total = config["num_steps"] if num_steps is None else num_steps
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    # One call: each `Simulation.step` call forces a particle exchange at
    # its first iteration, so splitting the run would change the physics.
    sim.step(n_total, show_progress=False)
    cupy.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0
    mem = device_memory()
    return {
        "elec": species_arrays(twin.elec_from_N, PARTICLE_NAMES),
        "z_win": float(sim.comm._zmin_global_domain),
        "counts": {k: int(s.Ntot) for k, s in twin.species.items()},
        "wall_s": wall,
        "memory": mem,
        "theta": np.asarray(theta, float),
    }


def build_ev_state(
    theta,
    config,
    seed=None,
    p_nz=1,
    epsilon=None,
    immobile=False,
    record_emissions=False,
):
    """Build a twin with fixed-charge ions and attach the EV model.

    Returns:
        (twin, state): the `Twin` and the stepper `StepState` after `begin`.
    """
    from fbpic.autodiff.stepper import begin

    twin = build_twin(theta, config, ionization="ev", seed=seed, p_nz=p_nz)
    ev = config["ev"]
    options = {
        "epsilon": ev["epsilon"] if epsilon is None else epsilon,
        "K_e": ev["K_e"],
        "immobile": immobile,
        "record_emissions": record_emissions,
    }
    state = begin(
        twin.sim,
        twin.species,
        ionization="ev",
        ev_species="nitrogen",
        ev_target="elec_from_N",
        ev_options=options,
    )
    return twin, state


def run_ev(
    theta, config, seed=None, p_nz=1, epsilon=None, immobile=False, num_steps=None
):
    """Run the EV model with the autodiff stepper to the observation step.

    Args:
        theta: (E_L [J], z_f [m]).
        config: Loaded configuration dict.
        seed: NumPy injection seed (default: config).
        p_nz: Particles per cell along z.
        epsilon: Emission threshold (default: config `ev.epsilon`).
        immobile: Freeze the N ions (Phase P3 step 8).
        num_steps: Steps to run (default: config `num_steps`).

    Returns:
        The dict of `run_production`, plus the EV ledger totals.
    """
    import cupy
    from fbpic.autodiff.stepper import step

    twin, state = build_ev_state(theta, config, seed, p_nz, epsilon, immobile)
    n_total = config["num_steps"] if num_steps is None else num_steps
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    births = 0
    for _ in range(n_total):
        decisions = step(state)[2]
        births += int(decisions.emitted_parent_ids.size)
    cupy.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0
    return {
        "elec": species_arrays(twin.elec_from_N, PARTICLE_NAMES),
        "z_win": float(twin.sim.comm._zmin_global_domain),
        "counts": {k: int(s.Ntot) for k, s in twin.species.items()},
        "births": births,
        "emitted_weight": float(state.ledger["emitted"]),
        "wall_s": wall,
        "memory": device_memory(),
        "theta": np.asarray(theta, float),
    }
