"""L4w: linear-wakefield derivative check (inversion plan Section 6.5).

Suggested by Remi Lehe. The upstream test `tests/test_linear_wakefield.py`
compares Ez and Er after a linear wakefield run with linear theory, whose
fields scale as a0^2. So dE/da0 = 2 E / a0 exactly.

Items (Section 6.5):

0. `forward`: run the upstream Nm=2 forward test, `test_linear_wakefield(
   Nm=2, show=False)`, in its own process.
1. Own builder: the upstream configuration with Nm=2 (GaussianLaser,
   theta_pol = pi/2, a0 = 0.01, w0 = 20 um, c tau = 6 um, n_e = 8e24 m^-3,
   800 x 120 cells, 1500 steps, electrons only, direct laser), with
   use_cuda_graphs=False and a tracked species, as the stepper requires.
2. Parameter a0, seed F0 / a0.
3. Output: Ez and Er at the last step in the theta = 0 plane (mode 0 plus
   twice the real part of mode 1), as in `compare_fields`.
4. Analytic check: dE/da0 against 2 E_an / a0, upstream tolerances
   (atol 0.08 max for Ez, 0.11 max for Er).
5. Self-consistency: dE/da0 against 2 E_sim / a0 within 1e-3 of the max,
   at a0 = 0.01 and 0.005; the relative residual must drop by a factor
   between 3 and 5 (it scales as a0^2).
6. Central FD in a0 (relative steps h of a0) with the Section 6.4 rule:
   sigma from 5 base runs, K(h) on a sample of tracker IDs.

Usage: python inversion/verify_linear_wakefield.py forward | check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tests"))
RESULTS = HERE / "results"
H_VALUES = (1e-2, 1e-3, 1e-4)
KEY_SAMPLE = 256  # record cell keys for tracker IDs divisible by this
STAGES = ("stage:gather", "stage:J", "stage:rho_next", "stage:rho_prev")


def run_forward_test() -> dict:
    """Item 0: the upstream Nm=2 test in its own process."""
    code = (
        "import sys; sys.path.insert(0, 'tests');"
        "from test_linear_wakefield import test_linear_wakefield;"
        "test_linear_wakefield(Nm=2, show=False); print('UPSTREAM_NM2_PASSED')"
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    passed = "UPSTREAM_NM2_PASSED" in proc.stdout
    out = {
        "passed": passed,
        "returncode": proc.returncode,
        "wall_time_s": time.perf_counter() - t0,
        "stderr_tail": proc.stderr[-2000:],
    }
    print(f"upstream test_linear_wakefield(Nm=2): passed={passed}")
    return out


def builder():
    """build(theta) -> (sim, species, profile) with theta = (a0,)."""
    import test_linear_wakefield as lw
    from scipy.constants import c

    from fbpic.lpa_utils.laser import GaussianLaser, add_laser_pulse
    from fbpic.main import Simulation

    def build(theta):
        a0 = float(theta[0])
        Nm = 2
        sim = Simulation(
            lw.Nz,
            lw.zmax,
            lw.Nr,
            lw.rmax,
            Nm,
            lw.dt,
            lw.p_zmin,
            lw.p_zmax,
            lw.p_rmin,
            lw.p_rmax,
            lw.p_nz,
            lw.p_nr,
            2 * Nm,
            lw.n_e,
            use_cuda=True,
            boundaries={"z": "open", "r": "reflective"},
            use_cuda_graphs=False,
            verbose_level=0,
        )
        electrons = sim.ptcl[0]
        electrons.track(sim.comm)
        profile = GaussianLaser(
            a0=a0, waist=lw.w0, tau=lw.tau, z0=lw.z0, theta_pol=np.pi / 2
        )
        profile.a0 = a0
        add_laser_pulse(sim, profile)
        sim.set_moving_window(v=c)
        return sim, {"electrons": electrons}, profile

    return build


def _theta_plane(grids, name):
    """Mode 0 plus twice the real part of mode 1 (`compare_fields`)."""
    value = grids[0][name].real.copy()
    for grid in grids[1:]:
        value += 2 * grid[name].real
    return value


def _host_grid(grid):
    """Shallow copy of an interpolation grid with host field arrays."""
    import copy

    import cupy

    host = copy.copy(grid)
    for name in ("Er", "Et", "Ez", "Br", "Bt", "Bz", "Jr", "Jt", "Jz", "rho"):
        setattr(host, name, cupy.asnumpy(getattr(grid, name)))
    host.use_cuda = False
    return host


def _gathered(sim):
    grids = [sim.comm.gather_grid(_host_grid(g)) for g in sim.fld.interp]
    z, r = grids[0].z, grids[0].r
    arrays = [{n: np.asarray(getattr(g, n)) for n in ("Ez", "Er")} for g in grids]
    return z, r, {n: _theta_plane(arrays, n) for n in ("Ez", "Er")}


def _gathered_tangent(sim, tangents):
    with tangents.fields.swapped(sim.fld, 0):
        _, _, fields = _gathered(sim)
    return fields


class SampledKeys:
    """Hook: per-stage cell keys of a sample of tracker IDs."""

    def __init__(self, reference=None):
        self.reference = reference
        self.records = {} if reference is None else None
        self.count = 0

    def __call__(self, state, name):
        if name not in STAGES:
            return
        import cupy

        sp = state.species["electrons"]
        grid = state.sim.fld.interp[0]
        n = int(sp.Ntot)
        ids = sp.tracker.id[:n]
        pick = ids % KEY_SAMPLE == 0
        ids = ids[pick]
        r = cupy.sqrt(sp.x[:n][pick] ** 2 + sp.y[:n][pick] ** 2)
        ir = cupy.minimum(cupy.ceil(grid.invdr * (r - grid.rmin) - 0.5), grid.Nr)
        iz = cupy.ceil(grid.invdz * (sp.z[:n][pick] - grid.zmin) - 0.5)
        key = iz.astype(np.int64) * (grid.Nr + 1) + ir.astype(np.int64)
        order = cupy.argsort(ids)
        ids, key = ids[order], key[order]
        tag = (state.iteration, name)
        if self.records is not None:
            self.records[tag] = (ids, key)
            return
        ref_ids, ref_key = self.reference.pop(tag)
        common, i, j = cupy.intersect1d(ref_ids, ids, return_indices=True)
        self.count += int(cupy.count_nonzero(ref_key[i] != key[j]))
        self.count += int(ref_ids.size + ids.size - 2 * common.size)


def primal(build, a0, n_steps, hook=None):
    from fbpic.autodiff.stepper import begin, step

    sim, species, _ = build((a0,))
    state = begin(sim, species)
    if hook is not None:
        state.hooks = {"*": hook}
    records = []
    for _ in range(n_steps):
        d = step(state)[2]
        if d.exchanged or d.n_move:
            records.append(
                (
                    d.iteration,
                    d.n_move,
                    sorted(d.injected.items()),
                    sorted(d.removed.items()),
                )
            )
    _, _, fields = _gathered(sim)
    return fields, records, int(species["electrons"].Ntot)


def tangent_run(build, a0, n_steps):
    from fbpic.autodiff.driver import jvp

    result = jvp((a0,), build, ("a0",), n_steps)
    sim = result.state.sim
    z, r, fields = _gathered(sim)
    derivative = _gathered_tangent(sim, result.tangents)
    records = [
        (d.iteration, d.n_move, sorted(d.injected.items()), sorted(d.removed.items()))
        for d in result.decisions
        if d.exchanged or d.n_move
    ]
    return z, r, sim.time, fields, derivative, records, result.info


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))


def check(n_steps: int, n_base: int):
    import test_linear_wakefield as lw

    build = builder()
    record = {"n_steps": n_steps, "a0_values": [0.01, 0.005]}
    runs = {}
    for a0 in (0.01, 0.005):
        z, r, t, fields, deriv, records, info = tangent_run(build, a0, n_steps)
        scale = (a0 / lw.a0) ** 2
        analytic = {
            "Ez": lw.Ez(z, r, t, 2) * scale,
            "Er": lw.Er(z, r, t, 2) * scale,
        }
        entry = {"time_per_step_s": info["time_per_step_s"], "decisions": records}
        for name, tol in (("Ez", 0.08), ("Er", 0.11)):
            target_an = 2 * analytic[name] / a0
            target_sim = 2 * fields[name] / a0
            entry[name] = {
                "analytic_residual_over_max": float(
                    np.max(np.abs(deriv[name] - target_an)) / np.max(np.abs(target_an))
                ),
                "analytic_tolerance": tol,
                "analytic_pass": bool(
                    np.allclose(
                        deriv[name], target_an, atol=tol * np.abs(target_an).max()
                    )
                ),
                "forward_vs_analytic_over_max": float(
                    np.max(np.abs(fields[name] - analytic[name]))
                    / np.max(np.abs(analytic[name]))
                ),
                "self_residual_over_max": _rel(deriv[name], target_sim),
            }
        runs[a0] = {
            "entry": entry,
            "fields": fields,
            "deriv": deriv,
            "records": records,
        }
        record[f"a0={a0}"] = entry
        print(
            f"a0={a0}: "
            + json.dumps({k: v for k, v in entry.items() if k != "decisions"})
        )
    for name in ("Ez", "Er"):
        r1 = record["a0=0.01"][name]["self_residual_over_max"]
        r2 = record["a0=0.005"][name]["self_residual_over_max"]
        record[f"self_ratio_{name}"] = r1 / r2
        record[f"self_pass_{name}"] = bool(r1 <= 1e-3 and 3.0 <= r1 / r2 <= 5.0)
    # Item 6: central FD in a0 (relative steps), Section 6.4 rule
    a0 = 0.01
    deriv = runs[a0]["deriv"]
    base = [primal(build, a0, n_steps)[0] for _ in range(n_base)]
    sigma = {
        name: max(
            float(np.max(np.abs(a[name] - b[name])))
            for a, b in pairwise(base + base[:1])
        )
        for name in ("Ez", "Er")
    }
    fd_rows = {"Ez": [], "Er": []}
    for h in H_VALUES:
        recorder = SampledKeys()
        plus, rec_p, _ = primal(build, a0 * (1 + h), n_steps, recorder)
        comparer = SampledKeys(recorder.records)
        minus, rec_m, _ = primal(build, a0 * (1 - h), n_steps, comparer)
        counts_ok = rec_p == runs[a0]["records"] and rec_m == runs[a0]["records"]
        for name in ("Ez", "Er"):
            fd = (plus[name] - minus[name]) / (2 * a0 * h)
            g_inf = max(float(np.max(np.abs(deriv[name]))), float(np.max(np.abs(fd))))
            e = float(np.max(np.abs(deriv[name] - fd))) / g_inf
            noise = 10 * np.sqrt(2) * sigma[name] / (2 * a0 * h * g_inf)
            fd_rows[name].append(
                {
                    "h": h,
                    "e": e,
                    "K_sampled": comparer.count,
                    "counts_valid": counts_ok,
                    "noise_term": noise,
                    "tau": max(1e-4, noise),
                    "usable": noise <= 1e-3 and counts_ok,
                }
            )
        print(f"FD h={h}: K_sampled={comparer.count} counts_ok={counts_ok} "
              + " ".join(f"{n}: e={fd_rows[n][-1]['e']:.2e} tau={fd_rows[n][-1]['tau']:.1e}"
                         for n in fd_rows))  # fmt: skip
    for name, rows in fd_rows.items():
        ok = any(
            a["usable"] and b["usable"] and a["K_sampled"] == 0 and b["K_sampled"] == 0
            and a["e"] <= a["tau"] and b["e"] <= b["tau"]
            for a, b in pairwise(rows)
        )  # fmt: skip
        record[f"fd_{name}"] = {"rows": rows, "sigma": sigma[name], "pass": bool(ok)}
    record["key_sample_modulus"] = KEY_SAMPLE
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("forward", "check"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--base", type=int, default=5)
    args = parser.parse_args()
    path = RESULTS / "p5_l4w.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    if args.stage == "forward":
        record["item0_upstream_nm2"] = run_forward_test()
    else:
        t0 = time.perf_counter()
        record["check"] = check(args.steps, args.base)
        record["check"]["wall_time_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
