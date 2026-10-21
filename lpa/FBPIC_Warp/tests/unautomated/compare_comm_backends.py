# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
Compare the MPI and NCCL exchange backends on complete 2-rank runs
(NCCL_MIGRATION.md, Sections 8.3 and 8.4).

The backend comes from FBPIC_COMM_BACKEND. Run from lpa/FBPIC_Warp:

    mpiexec -n 2 -genv FBPIC_COMM_BACKEND nccl python \
        tests/unautomated/compare_comm_backends.py run --case periodic_plasma_wave \
        --out OUT/periodic_plasma_wave/nccl.1
    python tests/unautomated/compare_comm_backends.py compare \
        --root OUT/periodic_plasma_wave --k 10
    mpiexec -n 2 -genv FBPIC_COMM_BACKEND nccl python \
        tests/unautomated/compare_comm_backends.py bench --case periodic_plasma_wave

`run` writes, per rank, the interpolation grids (guard cells included) and
the particles sorted by ID. `compare` applies the Level 2 gate: for each field,
max e_F(NCCL, MPI) <= k * max(noise floor of the MPI runs, 1e-14), and equal
ID sets on each rank. `bench` prints the time per step.
"""
import argparse
import glob
import itertools
import os
import sys
import time

import numpy as np
from scipy.constants import c

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FIELDS = ["Er", "Et", "Ez", "Br", "Bt", "Bz", "Jr", "Jt", "Jz", "rho"]
PARTICLE_ATTRS = ["x", "y", "z", "ux", "uy", "uz", "w"]


def build_periodic_plasma_wave(use_cuda_graphs):
    import test_periodic_plasma_wave as t
    from fbpic.main import Simulation

    sim = Simulation(
        t.Nz, t.zmax, t.Nr, t.rmax, t.Nm, t.dt,
        t.p_zmin, t.p_zmax, t.p_rmin, t.p_rmax, t.p_nz, t.p_nr, t.p_nt, t.n_e,
        n_order=t.n_order,
        use_cuda=True,
        particle_shape="linear",
        use_cuda_graphs=use_cuda_graphs,
    )  # fmt: skip
    t.impart_momenta(sim.ptcl[0], t.epsilons, t.k0, t.w0, t.wp)
    return sim, t.N_step, dict(correct_currents=True)


def build_linear_wakefield(use_cuda_graphs):
    import test_linear_wakefield as t
    from fbpic.main import Simulation
    from fbpic.lpa_utils.laser import add_laser_pulse, LaguerreGaussLaser

    sim = Simulation(
        t.Nz, t.zmax, t.Nr, t.rmax, 1, t.dt,
        t.p_zmin, t.p_zmax, t.p_rmin, t.p_rmax, t.p_nz, t.p_nr, 2, t.n_e,
        n_order=32,
        use_cuda=True,
        boundaries={"z": "open", "r": "reflective"},
        use_cuda_graphs=use_cuda_graphs,
    )  # fmt: skip
    # Azimuthally-polarized pulse, as in the test with Nm=1
    profile = LaguerreGaussLaser(
        0, 1, a0=t.a0, waist=t.w0, tau=t.tau, z0=t.z0, theta_pol=np.pi / 2, theta0=0.0
    ) + LaguerreGaussLaser(
        0, 1, a0=t.a0, waist=t.w0, tau=t.tau, z0=t.z0, theta_pol=0.0, theta0=-np.pi / 2
    )
    add_laser_pulse(sim, profile)
    sim.set_moving_window(v=c)
    return sim, t.N_step, dict(correct_currents=sim.comm.size == 1)


CASES = {
    "periodic_plasma_wave": build_periodic_plasma_wave,
    "linear_wakefield": build_linear_wakefield,
}


def check_backend(sim, comm_backend):
    """Stop if the active exchange backend is not the requested one."""
    active = "nccl" if sim.comm.nccl_comm is not None else "mpi"
    if active != comm_backend:
        sys.exit("FBPIC_COMM_BACKEND=%s, but %s is active" % (comm_backend, active))


def run(args):
    from fbpic.utils.mpi import comm

    comm_backend = os.environ.get("FBPIC_COMM_BACKEND", "mpi")

    # Fixed seed: the plasma loading draws random positions
    np.random.seed(args.seed + comm.rank)
    sim, N_step, step_kwargs = CASES[args.case](args.graphs)
    check_backend(sim, comm_backend)
    for species in sim.ptcl:
        species.track(sim.comm)
    if args.steps is not None:
        N_step = args.steps
    comm.Barrier()
    t0 = time.perf_counter()
    sim.step(N_step, show_progress=False, **step_kwargs)
    comm.Barrier()
    wall = time.perf_counter() - t0

    out = {"N_step": N_step, "wall": wall}
    for m, grid in enumerate(sim.fld.interp):
        for name in FIELDS:
            out["%s_m%d" % (name, m)] = np.asarray(getattr(grid, name))
    for i, species in enumerate(sim.ptcl):
        order = np.argsort(species.tracker.id)
        out["p%d_id" % i] = species.tracker.id[order]
        for attr in PARTICLE_ATTRS:
            out["p%d_%s" % (i, attr)] = getattr(species, attr)[order]
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "rank%d.npz" % comm.rank), **out)
    if comm.rank == 0:
        print(
            "%s %s graphs=%s: %d steps, %.3f s"
            % (args.case, comm_backend, args.graphs, N_step, wall)
        )


def rel_err(a, b):
    norm = np.linalg.norm(b)
    return np.linalg.norm(a - b) / norm if norm > 0 else np.linalg.norm(a)


def field_errors(run_1, run_2, rank):
    a = np.load(os.path.join(run_1, "rank%d.npz" % rank))
    b = np.load(os.path.join(run_2, "rank%d.npz" % rank))
    errors = {}
    for name in FIELDS:
        keys = sorted(k for k in a.files if k.startswith(name + "_m"))
        fa = np.concatenate([a[k].ravel() for k in keys])
        fb = np.concatenate([b[k].ravel() for k in keys])
        errors[name] = rel_err(fa, fb)
    species = sorted({k.split("_")[0] for k in a.files if k.startswith("p")})
    ids_equal = all(np.array_equal(a[s + "_id"], b[s + "_id"]) for s in species)
    for s, attr in itertools.product(species, PARTICLE_ATTRS):
        if ids_equal:
            errors["%s_%s" % (s, attr)] = rel_err(a[s + "_" + attr], b[s + "_" + attr])
    return errors, ids_equal


def compare(args):
    mpi_runs = sorted(glob.glob(os.path.join(args.root, args.ref + ".*")))
    nccl_runs = sorted(glob.glob(os.path.join(args.root, args.test + ".*")))
    if len(mpi_runs) < 2 or not nccl_runs:
        sys.exit(
            "compare needs >= 2 %s.* runs and >= 1 %s.* run in %s"
            % (args.ref, args.test, args.root)
        )
    n_ranks = len(glob.glob(os.path.join(mpi_runs[0], "rank*.npz")))
    ok = True
    for rank in range(n_ranks):
        noise, cross, ids_ok = {}, {}, True
        for r1, r2 in itertools.combinations(mpi_runs, 2):
            errors, same_ids = field_errors(r1, r2, rank)
            ids_ok &= same_ids
            for k, v in errors.items():
                noise[k] = max(noise.get(k, 0.0), v)
        for r1, r2 in itertools.product(nccl_runs, mpi_runs):
            errors, same_ids = field_errors(r1, r2, rank)
            ids_ok &= same_ids
            for k, v in errors.items():
                cross[k] = max(cross.get(k, 0.0), v)
        print("rank %d: particle IDs %s" % (rank, "equal" if ids_ok else "DIFFER"))
        ok &= ids_ok
        print("  %-10s %12s %12s %8s" % ("quantity", "noise", "test-ref", "gate"))
        for k in cross:
            limit = args.k * max(noise.get(k, 0.0), args.e_min)
            passed = cross[k] <= limit
            ok &= passed
            print(
                "  %-10s %12.3e %12.3e %8s"
                % (k, noise.get(k, 0.0), cross[k], "ok" if passed else "FAIL")
            )
    print("LEVEL2 %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def bench(args):
    from fbpic.utils.mpi import comm
    from fbpic.utils.warp import get_runtime

    comm_backend = os.environ.get("FBPIC_COMM_BACKEND", "mpi")

    np.random.seed(args.seed + comm.rank)
    sim, _, step_kwargs = CASES[args.case](args.graphs)
    check_backend(sim, comm_backend)
    sim.step(args.warmup, show_progress=False, **step_kwargs)
    get_runtime().synchronize()
    comm.Barrier()
    t0 = time.perf_counter()
    sim.step(args.steps, show_progress=False, **step_kwargs)
    get_runtime().synchronize()
    comm.Barrier()
    t_step = (time.perf_counter() - t0) / args.steps
    if comm.rank == 0:
        print(
            "BENCH %s %s graphs=%s: %.4f ms/step"
            % (args.case, comm_backend, args.graphs, 1e3 * t_step)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="mode", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--case", choices=CASES, required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--steps", type=int, default=None)
    p_run.add_argument("--graphs", action="store_true")
    p_run.add_argument("--seed", type=int, default=0)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--root", required=True)
    p_cmp.add_argument("--k", type=float, default=10.0)
    p_cmp.add_argument("--e-min", type=float, default=1e-14)
    p_cmp.add_argument("--ref", default="mpi", help="prefix of the reference runs")
    p_cmp.add_argument("--test", default="nccl", help="prefix of the tested runs")
    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--case", choices=CASES, required=True)
    p_bench.add_argument("--steps", type=int, default=300)
    p_bench.add_argument("--warmup", type=int, default=20)
    p_bench.add_argument("--graphs", action="store_true")
    p_bench.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"run": run, "compare": compare, "bench": bench}[args.mode](args)


if __name__ == "__main__":
    main()
