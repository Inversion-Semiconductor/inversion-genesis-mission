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

With a `TangentState`, `step` applies the tangent of each primal operation
right after it (forward mode, Sections 4.3 to 4.9). The particle tangents use
the kernels of `fbpic.autodiff.tangent`; the field tangents run the production
linear field operators on swapped tangent arrays (`FieldTangents.apply`).
Tangents follow every primal decision and never change it. EV tangents arrive
in Phase P6.

Production `Simulation.step` stays unchanged (D17). The stepper requires
`sim.use_cuda_graphs` to be False and reads `sim.iteration` and `sim.time`.
"""

from __future__ import annotations

import cupy
import numpy as np

from fbpic.autodiff import tangent as tg
from fbpic.autodiff.state import (
    EV_ARRAY_NAMES,
    TANGENT_PARTICLE_NAMES,
    StepDecisions,
    StepState,
    TangentState,
)
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


def sort_all(state: StepState, tangents: TangentState | None = None) -> dict:
    """Stable-sort every species; the EV arrays and the tangents follow.

    Returns:
        The permutation of each species by name (Section 4.5, rule 1).
    """
    perms = {}
    for name, species in state.species_by_object():
        extra = []
        if state.ev is not None and name == state.ev_species:
            extra = [(state.ev, n) for n in EV_ARRAY_NAMES]
        perms[name] = stable_sort(species, state.sim.fld, extra)
    if tangents is not None:
        _hook(state, "tangent_sort:before")
        for name, perm in perms.items():
            for array in tangents.particles[name].values():
                array[...] = array[:, perm]
        _hook(state, "tangent_sort:after")
    return perms


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
    _exchange(state, StepDecisions(state.iteration), check_sorted=False)
    return state


# One step
# --------


def _exchange(
    state: StepState,
    decisions: StepDecisions,
    tangents: TangentState | None = None,
    check_sorted: bool = True,
):
    """Particle exchange, `rho_prev` deposit, and memory housekeeping.

    Mirrors `main.py:540-563`. Removal and injection happen in production
    code, which does not expose a permutation; the EV arrays and the
    particle tangents are realigned by tracker ID afterwards (Section 4.9).
    Injected particles get zero tangents.

    Args:
        check_sorted: Assert `sp.sorted` for every species first. Otherwise
            `get_stay_range_gpu` runs the production radix sort with a
            permutation that the tangents never see (Section 4.5, rule 1).
            `begin` exchanges unsorted particles with zero tangents.
    """
    sim = state.sim
    if check_sorted:
        unsorted = [name for name, sp in state.species.items() if not sp.sorted]
        assert not unsorted, f"Species not sorted before the exchange: {unsorted}"
    before = {
        name: (int(sp.Ntot), int(sp.tracker.next_attributed_id))
        for name, sp in state.species.items()
    }
    ev_snapshot = None
    if state.ev is not None:
        from fbpic.autodiff import ionization as ev

        ev_snapshot = ev.snapshot(state)
    tangent_snapshot = None
    if tangents is not None:
        tangent_snapshot = {
            name: sp.tracker.id[: int(sp.Ntot)].copy()
            for name, sp in state.species.items()
        }
    sim.comm.exchange_all_particles(sim.ptcl, sim.fld, sim.time)
    if ev_snapshot is not None:
        ev.realign(state, ev_snapshot)
    decisions.injected = {}
    decisions.removed = {}
    for name, sp in state.species.items():
        n_before, next_id = before[name]
        injected = int(sp.tracker.next_attributed_id) - next_id
        decisions.injected[name] = injected
        decisions.removed[name] = n_before - (int(sp.Ntot) - injected)
    if tangents is not None:
        _hook(state, "tangent_realign:before")
        _realign_tangents(state, tangents, tangent_snapshot, before)
        _hook(state, "tangent_realign:after")
    sort_all(state, tangents)
    if state.ev is not None:
        ev.set_deposit_charge(state)
    _hook(state, "stage:rho_prev")
    sim.deposit("rho_prev", exchange=False)
    if state.ev is not None:
        ev.restore_charge(state)
    if tangents is not None:
        _deposit_tangents(state, tangents, "rho_prev")
    release_unused_device_memory()
    for sp in sim.ptcl:
        sp.reserve_step_workspaces()
    decisions.exchanged = True
    state.last_exchange = state.iteration


def _realign_tangents(state, tangents, old_ids, before):
    """Carry each kept particle's tangents to its new slot by tracker ID."""
    from fbpic.autodiff.ionization import _match

    for name, sp in state.species.items():
        n = int(sp.Ntot)
        new_ids = sp.tracker.id[:n]
        idx, found = _match(old_ids[name], new_ids)
        if not bool(cupy.all(found | (new_ids >= before[name][1]))):
            raise RuntimeError(f"An exchange created {name} with old tracker IDs.")
        arrays = tangents.particles[name]
        for comp in TANGENT_PARTICLE_NAMES:
            old = arrays[comp]
            if old.shape[1] == 0:
                arrays[comp] = cupy.zeros((tangents.K, n))
            else:
                arrays[comp] = cupy.where(found[None, :], old[:, idx], 0.0)


def step(state: StepState, tangents: TangentState | None = None):
    """Advance the state by one PIC cycle (`main.py:533-687`).

    Args:
        state: State from `begin` or a previous `step`; updated in place.
        tangents: K tangents of the state, updated in place, or None for the
            plain forward step. The primal result does not depend on them
            (L2t).

    Returns:
        (state, tangents, decisions): the same objects, and the
        `StepDecisions` of this step.
    """
    sim = state.sim
    fld, comm, dt = sim.fld, sim.comm, sim.dt
    decisions = StepDecisions(state.iteration)
    ev = None
    if state.ev is not None:
        from fbpic.autodiff import ionization as ev

        if tangents is not None:
            raise NotImplementedError("EV tangents arrive in Phase P6.")

    # 1. Exchange iteration: removal, injection, rho_prev deposit
    if (
        state.iteration % comm.exchange_period == 0
        and state.last_exchange != state.iteration
    ):
        _exchange(state, decisions, tangents)

    # 2. Gather at t = n dt (interp E and B were updated by the last step)
    for sp in sim.ptcl:
        sp.keep_fields_sorted = True
    _hook(state, "stage:gather")
    for sp in sim.ptcl:
        sp.gather(fld.interp, comm)
    gathered = None
    if tangents is not None:
        gathered = _gather_tangents(state, tangents)

    # 3. Momentum push, then the first half position push. The Vay tangent
    # needs the momenta before the push (Section 4.3, rule 2).
    stash = None
    if tangents is not None:
        stash = {
            name: tuple(getattr(sp, n)[: int(sp.Ntot)].copy() for n in MOMENTUM_NAMES)
            for name, sp in state.species_by_object()
        }
    for name, sp in state.species_by_object():
        if ev is not None and name == state.ev_species:
            ev.push_ion_momenta(state)
        else:
            sp.push_p(sim.time + 0.5 * dt)
    if tangents is not None:
        _vay_tangents(state, tangents, stash, gathered)
        del stash, gathered
    _push_positions(state, tangents)
    _hook(state, "before_ionization")

    # 4. Ionization and emission at t = (n + 1/2) dt
    if ev is not None:
        decisions.emitted_parent_ids = ev.ionize_and_emit(state)
    _hook(state, "after_ionization")
    for sp in sim.ptcl:
        sp.keep_fields_sorted = False

    # 5. Sort, J deposit, second half position push, sort, rho_next deposit
    sort_all(state, tangents)
    if ev is not None:
        ev.set_deposit_charge(state)
    _hook(state, "stage:J")
    sim.deposit("J", exchange=False)
    if ev is not None:
        ev.restore_charge(state)
    if tangents is not None:
        _deposit_tangents(state, tangents, "J")
    _push_positions(state, tangents)
    sort_all(state, tangents)
    if ev is not None:
        ev.set_deposit_charge(state)
    _hook(state, "stage:rho_next")
    sim.deposit("rho_next", exchange=False)
    if ev is not None:
        ev.restore_charge(state)
    if tangents is not None:
        _deposit_tangents(state, tangents, "rho_next")

    # 6. Curl-free current correction, PSATD push (with push_rho)
    fld.correct_currents(check_exchanges=False)
    fld.exchanged_source["J"] = True
    fld.push(False, check_exchanges=False)
    if tangents is not None:

        def push_fields():
            fld.correct_currents(check_exchanges=False)
            fld.push(False, check_exchanges=False)

        _field_section(state, tangents, "push", push_fields)

    # 7. Moving window and open-z damping, then interp E and B for step n+1
    zmin_before = comm._zmin_global_domain
    comm.move_grids(fld, sim.ptcl, dt, sim.time)
    decisions.n_move = round((comm._zmin_global_domain - zmin_before) / comm.dz)
    sim.exchange_and_damp_EB()
    if tangents is not None:
        n_move = decisions.n_move

        def shift_and_damp():
            if n_move != 0:
                for grid in fld.spect:
                    comm.moving_win.shift_spect_grid(grid, n_move)
            sim.exchange_and_damp_EB()

        _field_section(state, tangents, "window", shift_and_damp)

    sim.time += dt
    sim.iteration += 1
    return state, tangents, decisions


MOMENTUM_NAMES = ("ux", "uy", "uz", "inv_gamma")
GATHERED_TANGENT_NAMES = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
DEPOSIT_TANGENT_NAMES = ("x", "y", "z", "w", "ux", "uy", "uz", "inv_gamma")


def _particle_tangents(tangents, name, names):
    return [tangents.particles[name][n] for n in names]


def _field_section(state, tangents, label, operation):
    """Run a linear field operation on every tangent (Section 4.6)."""
    _hook(state, f"tangent_{label}:before")
    tangents.fields.apply(state.sim.fld, operation)
    _hook(state, f"tangent_{label}:after")


def _gather_tangents(state, tangents):
    """Gathered E and B tangents per species, (K, N) each (a work buffer)."""
    _hook(state, "tangent_gather:before")
    sim = state.sim
    rmax_gather = sim.comm.get_rmax(with_damp=False)
    field_view = tangents.fields.gather_view()
    gathered = {}
    for name, sp in state.species_by_object():
        n = int(sp.Ntot)
        outputs = [cupy.zeros((tangents.K, n)) for _ in GATHERED_TANGENT_NAMES]
        if sp.q != 0 and n > 0:
            tg.gather_tangent(
                sp.x[:n],
                sp.y[:n],
                sp.z[:n],
                _particle_tangents(tangents, name, ("x", "y", "z")),
                sim.fld.interp,
                field_view,
                rmax_gather,
                outputs,
            )
        gathered[name] = outputs
    _hook(state, "tangent_gather:after")
    return gathered


def _vay_tangents(state, tangents, stash, gathered):
    """Vay push tangent of every pushed species (Section 4.7)."""
    _hook(state, "tangent_vay:before")
    sim = state.sim
    for name, sp in state.species_by_object():
        n = int(sp.Ntot)
        if sp.q == 0 or n == 0:
            continue
        tg.vay_tangent(
            stash[name],
            [getattr(sp, f)[:n] for f in GATHERED_TANGENT_NAMES],
            sp.q,
            sp.m,
            sim.dt,
            _particle_tangents(tangents, name, MOMENTUM_NAMES),
            gathered[name],
        )
    _hook(state, "tangent_vay:after")


def _deposit_tangents(state, tangents, fieldtype):
    """Tangent of `sim.deposit(fieldtype)`: kernels, then the linear post-ops.

    Mirrors `main.py::deposit`: erase, deposit every charged species, divide
    by the cell volume, interp2spect, and the spectral filter.
    """
    _hook(state, f"tangent_deposit_{fieldtype}:before")
    sim = state.sim
    fld = sim.fld
    kind = "J" if fieldtype == "J" else "rho"
    outputs = tangents.fields.deposit_view(kind)
    for array in outputs:
        array[...] = 0
    for name, sp in state.species_by_object():
        n = int(sp.Ntot)
        if sp.q == 0 or sp.is_tracer or n == 0:
            continue
        if sp.ionizer is not None:
            raise NotImplementedError("Production ionizers have no tangent.")
        tg.deposit_tangent(
            [getattr(sp, c)[:n] for c in DEPOSIT_TANGENT_NAMES],
            _particle_tangents(tangents, name, DEPOSIT_TANGENT_NAMES),
            sp.q,
            fld.interp,
            sp.prefix_sum,
            outputs,
            current=kind == "J",
        )

    def post():
        fld.divide_by_volume(kind)
        fld.interp2spect(fieldtype)
        if sim.filter_currents:
            fld.filter_spect(fieldtype)

    tangents.fields.apply(fld, post)
    _hook(state, f"tangent_deposit_{fieldtype}:after")


def _push_positions(state: StepState, tangents: TangentState | None = None):
    """Half position push of every species; immobile EV ions stay in place."""
    immobile = state.ev_params is not None and state.ev_params.get("immobile", False)
    for name, sp in state.species_by_object():
        if immobile and name == state.ev_species:
            continue
        sp.push_x(0.5 * state.sim.dt)
    if tangents is None:
        return
    _hook(state, "tangent_position:before")
    for name, sp in state.species_by_object():
        n = int(sp.Ntot)
        if (immobile and name == state.ev_species) or n == 0:
            continue
        tg.position_tangent(
            [getattr(sp, c)[:n] for c in MOMENTUM_NAMES],
            _particle_tangents(tangents, name, MOMENTUM_NAMES),
            0.5 * state.sim.dt,
            _particle_tangents(tangents, name, ("x", "y", "z")),
        )
    _hook(state, "tangent_position:after")


def _hook(state: StepState, name: str):
    """Call the test hooks `state.hooks[name](state)` and `state.hooks["*"](state, name)`."""
    hooks = getattr(state, "hooks", {})
    hook = hooks.get(name)
    if hook is not None:
        hook(state)
    every = hooks.get("*")
    if every is not None:
        every(state, name)


def run(state: StepState, n_steps: int, tangents: TangentState | None = None):
    """Call `step` n_steps times and return the list of decisions."""
    return [step(state, tangents)[2] for _ in range(n_steps)]
