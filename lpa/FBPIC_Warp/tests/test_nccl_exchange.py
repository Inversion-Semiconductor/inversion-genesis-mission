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
