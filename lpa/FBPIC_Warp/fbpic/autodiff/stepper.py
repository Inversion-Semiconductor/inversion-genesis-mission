# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Functional stepper: `begin` and `step` (inversion plan Section 4.3).

The stepper mirrors the loop body of `Simulation.step` (`fbpic/main.py`) for
the twin configuration: one GPU, lab frame, moving window, open z and
reflective r boundaries, curl-free current correction, no PML, no antenna, no
Galilean shift, and no CUDA graphs. It calls the production kernels and
methods for every operation, with three differences:

1. It sorts every species itself, before each deposit, with the production
   cell key and a stable sort (`cupy.argsort`), and applies the permutation
   to every particle array, including the EV arrays (Section 4.5).
2. It replaces production ionization of the EV species with the expected-value
   model of `fbpic.autodiff.ionization` (Section 5).
3. It does not reproduce the step-0 diagnostic J deposit (Section 4.3, rule 4).

Production `Simulation.step` stays unchanged (D17). The stepper requires
`sim.use_cuda_graphs` to be False and reads `sim.iteration` and `sim.time`.
"""

from __future__ import annotations

import cupy
import numpy as np

from fbpic.autodiff.state import EV_ARRAY_NAMES, StepDecisions, StepState
from fbpic.particles.utilities.cuda_sorting import (
    get_cell_idx_per_particle,
    incl_prefix_sum,
    prefill_prefix_sum,
)
from fbpic.utils.cuda import cuda_tpb_bpg_1d, send_data_to_gpu
from fbpic.utils.warp import get_runtime, release_unused_device_memory

PARTICLE_FLOAT_NAMES = ("x", "y", "z", "ux", "uy", "uz", "w", "inv_gamma")
GATHERED_NAMES = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


def _check_supported(sim):
    if sim.use_cuda_graphs:
        raise ValueError("The stepper runs eagerly: build with use_cuda_graphs=False.")
    if not sim.use_cuda:
        raise ValueError("The stepper needs use_cuda=True.")
    if sim.comm.size > 1:
        raise ValueError("The stepper supports one rank only (D1).")
    if sim.use_pml or sim.laser_antennas or sim.external_fields:
        raise ValueError("PML, antennas, and external fields are not supported.")
    if sim.v_comoving is not None:
        raise ValueError("The Galilean frame is not supported.")
    if sim.comm.moving_win is None:
        raise ValueError("The stepper expects a moving window.")
    for species in sim.ptcl:
        if species.ionizer is not None or species.spin_tracker is not None:
            raise ValueError("Production ionizers and spin are not supported.")
        if species.tracker is None:
            raise ValueError("Every species needs a tracker (Section 4.2, rule 4).")


# Sorting
# -------


def stable_sort(species, fld, extra=()):
    """Sort a species by the production cell key with a stable sort.

    Computes the key `ir_upper + iz_upper * (Nr + 1)` with the production
    kernel, sorts it with `cupy.argsort` (stable), rebuilds the prefix sum
    with the production kernels, and permutes every data array of the
    species and the arrays of `extra` in place.

    Args:
        species: GPU `Particles` object. Its fields on the particles (Ex to
            Bz) are permuted only while `keep_fields_sorted` is True, as in
            production: they are dead outside gather to ionization.
        fld: Production `Fields` object.
        extra: Further (object, attribute name) pairs of Ntot elements.

    Returns:
        The permutation (CuPy int64 array): new[i] = old[perm[i]].
    """
    n = int(species.Ntot)
    grid = fld.interp[0]
    perm = cupy.arange(n, dtype=np.int64)
    if n > 0:
        cell_idx = cupy.empty(n, dtype=np.int32)
        sorted_idx = cupy.empty(n, dtype=np.int32)
        grid_1d, block_1d = cuda_tpb_bpg_1d(n)
        get_cell_idx_per_particle[grid_1d, block_1d](
            cell_idx,
            sorted_idx,
            species.x,
            species.y,
            species.z,
            grid.invdz,
            grid.zmin,
            grid.Nz,
            grid.invdr,
            grid.rmin,
            grid.Nr,
        )
        perm = cupy.argsort(cell_idx, kind="stable")
        sorted_key = cell_idx[perm]
        grid_2d, block_2d = cuda_tpb_bpg_1d(species.prefix_sum.shape[0])
        prefill_prefix_sum[grid_2d, block_2d](sorted_key, species.prefix_sum, n)
        incl_prefix_sum[grid_1d, block_1d](sorted_key, species.prefix_sum)
        for obj, attr in list(_data_pairs(species)) + list(extra):
            array = getattr(obj, attr)
            array[:n] = array[:n][perm]
    else:
        species.prefix_sum[...] = 0
    species.prefix_sum_shift = 0
    species.sorted = True
    return perm


def _data_pairs(species):
    yield from ((species, n) for n in PARTICLE_FLOAT_NAMES)
    yield (species.tracker, "id")
    if species.keep_fields_sorted:
        yield from ((species, n) for n in GATHERED_NAMES)


def sort_all(state: StepState):
    """Stable-sort every species; the EV arrays follow their ions."""
    for name, species in state.species_by_object():
        extra = []
        if state.ev is not None and name == state.ev_species:
            extra = [(state.ev, n) for n in EV_ARRAY_NAMES]
        stable_sort(species, state.sim.fld, extra)


# Prologue
# --------


def begin(
    sim,
    species: dict,
    ionization: str = "none",
    ev_species=None,
    ev_target=None,
    ev_options: dict | None = None,
):
    """Replicate the prologue of `Simulation.step` once (`main.py:475-518`).

    Sets up the injection positions, sends the data to the GPU, damps the
    interp E and B at the open z boundaries, transforms them to spectral
    space, and performs the first particle exchange and `rho_prev` deposit.

    Args:
        sim: Production `Simulation`, built for the twin and not stepped.
        species: Species by name. Every species of `sim.ptcl` must appear.
        ionization: "none", or "ev" for the expected-value model.
        ev_species, ev_target: Names of the EV ions and of the species that
            receives their electrons (with ionization="ev").
        ev_options: Keyword arguments of `fbpic.autodiff.ionization.attach`
            (epsilon, K_e, immobile).

    Returns:
        The new `StepState`. The GPU owns the data until the caller copies it.
    """
    _check_supported(sim)
    if len(species) != len(sim.ptcl):
        raise ValueError("Name every species of the simulation.")
    state = StepState(sim, dict(species), ionization, ev_species, ev_target)
    for sp in sim.ptcl:
        sp.fuse_deposition = sim.fuse_particle_kernels
        sp.fuse_gather = sim.fuse_particle_kernels
    for sp in sim.ptcl:
        if sp.continuous_injection:
            sp.injector.initialize_injection_positions(
                sim.comm, sim.comm.moving_win.v, sp.z, sim.dt
            )
    get_runtime().use()
    sim._cuda_graphs.clear()
    send_data_to_gpu(sim)
    fld = sim.fld
    sim.comm.exchange_fields(fld.interp, "E", "replace")
    sim.comm.exchange_fields(fld.interp, "B", "replace")
    sim.comm.damp_EB_open_boundary(fld.interp)
    fld.interp2spect("E")
    fld.interp2spect("B")
    if ionization == "ev":
        from fbpic.autodiff import ionization as ev

        ev.attach(state, **(ev_options or {}))
    elif ionization != "none":
        raise ValueError(f"Unknown ionization model: {ionization}")
    _exchange(state, StepDecisions(state.iteration))
    return state


# One step
# --------


def _exchange(state: StepState, decisions: StepDecisions):
    """Particle exchange, `rho_prev` deposit, and memory housekeeping.

    Mirrors `main.py:540-563`. Removal and injection happen in production
    code, which does not permute the EV arrays; they are realigned by
    tracker ID afterwards.
    """
    sim = state.sim
    before = {
        name: int(sp.tracker.next_attributed_id) for name, sp in state.species.items()
    }
    ev_snapshot = None
    if state.ev is not None:
        from fbpic.autodiff import ionization as ev

        ev_snapshot = ev.snapshot(state)
    sim.comm.exchange_all_particles(sim.ptcl, sim.fld, sim.time)
    if ev_snapshot is not None:
        ev.realign(state, ev_snapshot)
    sort_all(state)
    if state.ev is not None:
        ev.set_deposit_charge(state)
    sim.deposit("rho_prev", exchange=False)
    if state.ev is not None:
        ev.restore_charge(state)
    release_unused_device_memory()
    for sp in sim.ptcl:
        sp.reserve_step_workspaces()
    decisions.exchanged = True
    decisions.injected = {
        name: int(sp.tracker.next_attributed_id) - before[name]
        for name, sp in state.species.items()
    }
    state.last_exchange = state.iteration


def step(state: StepState, tangents=None):
    """Advance the state by one PIC cycle (`main.py:533-687`).

    Args:
        state: State from `begin` or a previous `step`; updated in place.
        tangents: Must be None. Tangents arrive in Phase P5.

    Returns:
        (state, tangents, decisions): the same state object, None, and the
        `StepDecisions` of this step.
    """
    if tangents is not None:
        raise NotImplementedError("Tangents are not implemented (Phase P5).")
    sim = state.sim
    fld, comm, dt = sim.fld, sim.comm, sim.dt
    decisions = StepDecisions(state.iteration)
    ev = None
    if state.ev is not None:
        from fbpic.autodiff import ionization as ev

    # 1. Exchange iteration: removal, injection, rho_prev deposit
    if (
        state.iteration % comm.exchange_period == 0
        and state.last_exchange != state.iteration
    ):
        _exchange(state, decisions)

    # 2. Gather at t = n dt (interp E and B were updated by the last step)
    for sp in sim.ptcl:
        sp.keep_fields_sorted = True
    for sp in sim.ptcl:
        sp.gather(fld.interp, comm)

    # 3. Momentum push, then the first half position push
    for name, sp in state.species_by_object():
        if ev is not None and name == state.ev_species:
            ev.push_ion_momenta(state)
        else:
            sp.push_p(sim.time + 0.5 * dt)
    _push_positions(state)
    _hook(state, "before_ionization")

    # 4. Ionization and emission at t = (n + 1/2) dt
    if ev is not None:
        decisions.emitted_parent_ids = ev.ionize_and_emit(state)
    _hook(state, "after_ionization")
    for sp in sim.ptcl:
        sp.keep_fields_sorted = False

    # 5. Sort, J deposit, second half position push, sort, rho_next deposit
    sort_all(state)
    if ev is not None:
        ev.set_deposit_charge(state)
    sim.deposit("J", exchange=False)
    if ev is not None:
        ev.restore_charge(state)
    _push_positions(state)
    sort_all(state)
    if ev is not None:
        ev.set_deposit_charge(state)
    sim.deposit("rho_next", exchange=False)
    if ev is not None:
        ev.restore_charge(state)

    # 6. Curl-free current correction, PSATD push (with push_rho)
    fld.correct_currents(check_exchanges=False)
    fld.exchanged_source["J"] = True
    fld.push(False, check_exchanges=False)

    # 7. Moving window and open-z damping, then interp E and B for step n+1
    zmin_before = comm._zmin_global_domain
    comm.move_grids(fld, sim.ptcl, dt, sim.time)
    decisions.n_move = round((comm._zmin_global_domain - zmin_before) / comm.dz)
    sim.exchange_and_damp_EB()

    sim.time += dt
    sim.iteration += 1
    return state, None, decisions


def _push_positions(state: StepState):
    """Half position push of every species; immobile EV ions stay in place."""
    immobile = state.ev_params is not None and state.ev_params.get("immobile", False)
    for name, sp in state.species_by_object():
        if immobile and name == state.ev_species:
            continue
        sp.push_x(0.5 * state.sim.dt)


def _hook(state: StepState, name: str):
    """Call a test hook `state.hooks[name](state)` if one is registered."""
    hook = getattr(state, "hooks", {}).get(name)
    if hook is not None:
        hook(state)


def run(state: StepState, n_steps: int):
    """Call `step` n_steps times and return the list of decisions."""
    return [step(state)[2] for _ in range(n_steps)]
