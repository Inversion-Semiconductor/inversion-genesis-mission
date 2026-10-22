# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
Level 1 check of the NCCL domain exchanges (NCCL_MIGRATION.md, Section 8.2).

Each rank fills its grids with exact, position-coded values. Then the same
exchange runs with MPI and with NCCL, and both results must be bitwise equal
to each other and to an exact oracle.

Usage (from lpa/FBPIC_Warp):
    mpiexec -n 2 python -m pytest -q tests/test_nccl_exchange.py
"""
import numpy as np
import pytest

from fbpic.main import Simulation
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.mpi import comm, mpi_installed
from fbpic.utils.nccl import nccl_installed

pytestmark = pytest.mark.skipif(
    not (cuda_installed and mpi_installed and nccl_installed and comm.size == 2),
    reason="needs 2 MPI ranks, CUDA and nccl4py",
)

if cuda_installed:
    import cupy
    from fbpic.utils.cuda import send_data_to_gpu
    from fbpic.utils.nccl import get_nccl_comm
    from fbpic.utils.warp import get_runtime

# Global grid. Each local grid must have at least 4 n_guard cells along z.
Nz, Nr, Nm = 160, 16, 2
zmin, zmax, rmax = 0.0, 40.0e-6, 20.0e-6
n_order = 8

# Exchanged quantities: field type -> (grid attributes, method)
VECTOR = ["r", "t", "z"]
FIELDS = {
    "E": ["E" + c for c in VECTOR],
    "B": ["B" + c for c in VECTOR],
    "J": ["J" + c for c in VECTOR],
    "rho": ["rho"],
}
PML_FIELDS = {"E": ["Er_pml", "Et_pml"], "B": ["Br_pml", "Bt_pml"]}


def make_sim(z_boundary, r_boundary):
    return Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt=(zmax - zmin) / Nz / 3.0e8,
        zmin=zmin,
        n_order=n_order,
        boundaries={"z": z_boundary, "r": r_boundary},
        use_cuda=True,
        verbose_level=0,
    )


def grid_names(sim, fldtype):
    names = list(FIELDS[fldtype])
    if sim.comm.use_pml and fldtype in PML_FIELDS:
        names += PML_FIELDS[fldtype]
    return names


def code(rank, i_name, m, Nz_loc, Nr_loc, offset):
    """Exact, distinct value for each (rank, grid, mode, cell); real != imag."""
    iz = np.arange(Nz_loc)[:, None]
    ir = np.arange(Nr_loc)[None, :]
    q = (((rank * 16 + i_name) * Nm + m) * 1024 + iz) * 1024 + ir + offset
    return (q - 1j * (q + 0.5)).astype(np.complex128)


def fill(sim, names, offset):
    Nz_loc, Nr_loc = sim.fld.interp[0].Er.shape
    for i_name, name in enumerate(names):
        for m in range(Nm):
            values = code(sim.comm.rank, i_name, m, Nz_loc, Nr_loc, offset)
            getattr(sim.fld.interp[m], name)[...] = cupy.asarray(values)


def collect(sim, names):
    get_runtime().synchronize()
    return {
        (name, m): getattr(sim.fld.interp[m], name).get()
        for name in names
        for m in range(Nm)
    }


def oracle(sim, names, method, offset, Nz_locs):
    """Exact result of the exchange (Section 8.2), from pre-exchange codes."""
    ng = sim.comm.n_guard
    rank = sim.comm.rank
    left, right = sim.comm.left_proc, sim.comm.right_proc
    Nz_loc, Nr_loc = sim.fld.interp[0].Er.shape
    result = {}
    for i_name, name in enumerate(names):
        for m in range(Nm):
            mine = code(rank, i_name, m, Nz_loc, Nr_loc, offset)
            out = mine.copy()
            if method == "replace":
                if left is not None:
                    peer = code(left, i_name, m, Nz_locs[left], Nr_loc, offset)
                    out[:ng] = peer[Nz_locs[left] - 2 * ng : Nz_locs[left] - ng]
                if right is not None:
                    peer = code(right, i_name, m, Nz_locs[right], Nr_loc, offset)
                    out[Nz_loc - ng :] = peer[ng : 2 * ng]
            else:
                if left is not None:
                    peer = code(left, i_name, m, Nz_locs[left], Nr_loc, offset)
                    out[: 2 * ng] = mine[: 2 * ng] + peer[Nz_locs[left] - 2 * ng :]
                if right is not None:
                    peer = code(right, i_name, m, Nz_locs[right], Nr_loc, offset)
                    out[Nz_loc - 2 * ng :] = mine[Nz_loc - 2 * ng :] + peer[: 2 * ng]
            result[(name, m)] = out
    return result


def bitwise_equal(a, b):
    return all(np.array_equal(a[k].view(np.uint64), b[k].view(np.uint64)) for k in a)


def poison_buffers(sim):
    # The MPI run leaves correct data in the shared device buffers:
    # overwrite them, so that an NCCL transfer that does nothing is detected
    buffers = sim.comm.mpi_buffers
    for d_buffers in (
        buffers.d_send_l,
        buffers.d_send_r,
        buffers.d_recv_l,
        buffers.d_recv_r,
    ):
        for value in d_buffers.values():
            value.fill(np.nan)


def run_exchange(sim, fldtype, method, names, offset, nccl_comm):
    sim.comm.nccl_comm = nccl_comm
    if nccl_comm is not None:
        poison_buffers(sim)
    fill(sim, names, offset)
    sim.comm.exchange_fields(sim.fld.interp, fldtype, method)
    return collect(sim, names)


@pytest.mark.parametrize("r_boundary", ["reflective", "open"], ids=["no_pml", "pml"])
@pytest.mark.parametrize("z_boundary", ["periodic", "open"], ids=["F1", "F2"])
def test_field_exchange_matches_mpi_and_oracle(z_boundary, r_boundary):
    sim = make_sim(z_boundary, r_boundary)
    send_data_to_gpu(sim)
    nccl_comm = get_nccl_comm(sim.comm.mpi_comm)
    Nz_locs = comm.allgather(sim.fld.interp[0].Er.shape[0])
    assert min(Nz_locs) >= 4 * sim.comm.n_guard
    if z_boundary == "periodic":
        # P = 2 periodic: both faces go to the same peer (swap detection)
        assert sim.comm.left_proc == sim.comm.right_proc

    failures = []
    cases = [
        ("EB", "replace", grid_names(sim, "E") + grid_names(sim, "B")),
        ("E", "replace", grid_names(sim, "E")),
        ("B", "replace", grid_names(sim, "B")),
        ("J", "add", grid_names(sim, "J")),
        ("rho", "add", grid_names(sim, "rho")),
    ]
    for i_case, (fldtype, method, names) in enumerate(cases):
        offset = 7 * i_case
        expected = oracle(sim, names, method, offset, Nz_locs)
        mpi_result = run_exchange(sim, fldtype, method, names, offset, None)
        nccl_result = run_exchange(sim, fldtype, method, names, offset, nccl_comm)
        if not bitwise_equal(mpi_result, expected):
            failures.append("%s: MPI != oracle" % fldtype)
        if not bitwise_equal(nccl_result, expected):
            failures.append("%s: NCCL != oracle" % fldtype)
        if not bitwise_equal(nccl_result, mpi_result):
            failures.append("%s: NCCL != MPI" % fldtype)
    sim.comm.nccl_comm = None

    all_failures = comm.allgather(failures)
    assert not any(all_failures), all_failures


# Particle exchange (cases P1 to P4)
# ----------------------------------
# Particle groups, chosen from the particle ID: stay, cross the left face,
# cross the right face. Each case selects which groups exist.
PARTICLE_CASES = {
    "P1": ("periodic", ("stay", "left", "right")),
    "P2": ("periodic", ("stay", "right")),
    "P3": ("periodic", ("stay",)),
    "P4": ("open", ("stay", "left", "right")),
}
FLOAT_CODES = {"ux": 3, "uy": 4, "uz": 5, "inv_gamma": 6, "w": 7}


def make_particle_sim(z_boundary, groups):
    from scipy.constants import e, m_e, m_p

    np.random.seed(1234 + comm.rank)
    sim = make_sim(z_boundary, "reflective")
    common = dict(p_zmin=zmin, p_zmax=zmax, p_rmax=0.8 * rmax, p_nr=2, p_nt=4)
    electrons = sim.add_new_species(q=-e, m=m_e, n=1.0e24, p_nz=2, **common)
    ions = sim.add_new_species(q=e, m=14 * m_p, n=2.0e23, p_nz=1, **common)
    ions.make_ionizable("N", target_species=electrons, level_start=1)
    for species in sim.ptcl:
        species.track(sim.comm)
        code_particles(sim, species, groups)
    return sim


def group_of(ids, groups):
    return np.array(groups)[(ids % len(groups)).astype(np.int64)]


def source_z(sim, ids, groups, zmin_loc, zmax_loc):
    """Position of each particle before the exchange (exact in float64)."""
    dz = sim.fld.interp[0].dz
    ng = sim.comm.n_guard
    g = group_of(ids, groups)
    z = zmin_loc + (ng + 2 + (ids % 5).astype(np.float64) + 0.3) * dz
    z = np.where(g == "left", zmin_loc + (ng - 2.3) * dz, z)
    z = np.where(g == "right", zmax_loc - (ng - 2.3) * dz, z)
    return z


def code_particles(sim, species, groups):
    ids = species.tracker.id
    grid = sim.fld.interp[0]
    species.z[:] = source_z(sim, ids, groups, grid.zmin, grid.zmax)
    for name, a in FLOAT_CODES.items():
        getattr(species, name)[:] = ids.astype(np.float64) * 16 + a
    if species.ionizer is not None:
        species.ionizer.w_times_level[:] = ids.astype(np.float64) * 16 + 8
        species.ionizer.ionization_level[:] = ids % 5 + 1


def particle_arrays(species):
    get_runtime().synchronize()
    out = {}
    for name in ["x", "y", "z", "ux", "uy", "uz", "inv_gamma", "w"]:
        out[name] = getattr(species, name).get()
    out["id"] = species.tracker.id.get()
    if species.ionizer is not None:
        out["w_times_level"] = species.ionizer.w_times_level.get()
        out["ionization_level"] = species.ionizer.ionization_level.get()
    return out


def check_particle_oracle(sim, before, after, groups):
    """Compare one species with the exact result of the exchange."""
    rank, left, right = sim.comm.rank, sim.comm.left_proc, sim.comm.right_proc
    grid = sim.fld.interp[0]
    edges = comm.allgather((grid.zmin, grid.zmax))
    all_ids = comm.allgather(before["id"])
    # Expected IDs: kept particles, and particles that cross toward this rank
    g = group_of(before["id"], groups)
    expected = [before["id"][g == "stay"]]
    if left is not None:
        g_left = group_of(all_ids[left], groups)
        expected.append(all_ids[left][g_left == "right"])
    if right is not None:
        g_right = group_of(all_ids[right], groups)
        expected.append(all_ids[right][g_right == "left"])
    expected = np.sort(np.concatenate(expected))
    errors = []
    if not np.array_equal(np.sort(after["id"]), expected):
        errors.append("IDs differ from the oracle")
        return errors
    ids = after["id"]
    for name, a in FLOAT_CODES.items():
        if not np.array_equal(after[name], ids.astype(np.float64) * 16 + a):
            errors.append("%s differs from its code" % name)
    if "ionization_level" in after:
        if not np.array_equal(after["ionization_level"], ids % 5 + 1):
            errors.append("ionization_level differs from its code")
        if not np.array_equal(after["w_times_level"], ids.astype(np.float64) * 16 + 8):
            errors.append("w_times_level differs from its code")
    # Positions: the source rank of each particle follows from its ID
    Ltot = sim.comm._Nz_global_domain * sim.comm.dz
    src = (ids % comm.size).astype(np.int64)
    z_expected = np.empty_like(after["z"])
    for r in range(comm.size):
        sel = src == r
        z_expected[sel] = source_z(sim, ids[sel], groups, *edges[r])
    from_right = (src == right) & (group_of(ids, groups) == "left") & (src != rank)
    from_left = (src == left) & (group_of(ids, groups) == "right") & (src != rank)
    if right == 0:
        z_expected[from_right] = z_expected[from_right] + Ltot
    if left == comm.size - 1:
        z_expected[from_left] = z_expected[from_left] - Ltot
    if not np.array_equal(after["z"], z_expected):
        errors.append("z differs from the oracle")
    return errors


@pytest.mark.parametrize("case", sorted(PARTICLE_CASES))
def test_particle_exchange_matches_mpi_and_oracle(case):
    z_boundary, groups = PARTICLE_CASES[case]
    nccl_comm = None
    results = {}
    for backend in ("mpi", "nccl"):
        sim = make_particle_sim(z_boundary, groups)
        send_data_to_gpu(sim)
        if nccl_comm is None:
            nccl_comm = get_nccl_comm(sim.comm.mpi_comm)
        sim.comm.nccl_comm = nccl_comm if backend == "nccl" else None
        before = [particle_arrays(species) for species in sim.ptcl]
        sim.comm.exchange_all_particles(sim.ptcl, sim.fld, 0.0)
        after = [particle_arrays(species) for species in sim.ptcl]
        sim.comm.nccl_comm = None
        results[backend] = (before, after)

    failures = []
    for i_species in range(2):
        before, after = results["mpi"][0][i_species], results["mpi"][1][i_species]
        failures += [
            "species %d MPI: %s" % (i_species, err)
            for err in check_particle_oracle(sim, before, after, groups)
        ]
        nccl_after = results["nccl"][1][i_species]
        for name, values in after.items():
            if not np.array_equal(
                values.view(np.uint64), nccl_after[name].view(np.uint64)
            ):
                failures.append("species %d: NCCL != MPI for %s" % (i_species, name))
        if case == "P3" and len(after["id"]) != len(before["id"]):
            failures.append("species %d: P3 changed the particle number" % i_species)

    all_failures = comm.allgather(failures)
    assert not any(all_failures), all_failures


def test_particle_layout_mismatch_raises():
    """Different attributes on two neighbors give an error on both, no hang."""
    from scipy.constants import e, m_e

    sim = make_sim("periodic", "reflective")
    electrons = sim.add_new_species(
        q=-e, m=m_e, n=1.0e24, p_nz=1, p_nr=1, p_nt=4, p_zmin=zmin, p_zmax=zmax
    )
    if comm.rank == 0:
        electrons.track(sim.comm)
    send_data_to_gpu(sim)
    sim.comm.nccl_comm = get_nccl_comm(sim.comm.mpi_comm)
    with pytest.raises(RuntimeError, match="attributes differ"):
        sim.comm.exchange_all_particles(sim.ptcl, sim.fld, 0.0)
    sim.comm.nccl_comm = None
