"""L4: forward-mode parameter gradient against frozen-decision central FD.

Inversion plan Sections 6.2 to 6.4, Phase P5 step 6 and step 10, Phase P6
steps 3 and 4. Stage A: one tangent (E_L). One AD run (`jvp` with the E_L
seed) records the decisions and evaluates J_chk and its gradient at every
horizon. The FD runs are plain `step` runs at theta_hat_E +- h. With
`--ionization ev`, every FD run and base run reuses the emitting parent IDs
of the AD run at every step as `emit_mask` (frozen-decision FD, rule 3).

FD rules (Section 6.3):

- Rule 2: every FD run asserts the `n_move` sequence and the injected and
  removed counts of every species at every exchange against the AD run.
- Rule 7: sigma_J is the largest pairwise spread of 5 base runs; the
  tolerance is tau(h) = max(1e-4, 10 sqrt(2) sigma_J / (2 h |g|)), and an h
  whose second term exceeds 1e-3 is rejected.
- Rule 8: every FD run records the production cell key of every particle
  (by tracker ID) at every stage (gather, J deposit, rho_next deposit, and
  the exchange-step rho_prev deposit). K(h) is the number of (particle,
  stage) keys that differ between the +h and -h runs, up to the horizon.

Pass rule (Section 6.4): two adjacent h with K(h) = 0 give e(h) <= tau(h);
or both |g_AD| and |g_FD| are below 1e-6 |g|_inf. Without two adjacent
usable h with K(h) = 0, the FD reference is invalid; the component then
passes if g_AD lies within the spread of g_FD(h), or within the smoothed
slope of a +-0.05 slice of 41 points ("FD reference invalid (K > 0)").

Usage (from lpa/FBPIC_Warp, CUDA_VISIBLE_DEVICES set):

    python inversion/verify_param_gradient.py nc twin_xs
    python inversion/verify_param_gradient.py l4 twin_xs
    python inversion/verify_param_gradient.py l4 twin_xs --ionization ev \
        --steps 28 --horizons 20 28 --h 1e-6 1e-7 1e-8 --slice-points 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from builders import twin_builder  # noqa: E402
from twin_config import config_sha256, density_shape, load_config  # noqa: E402

RESULTS = HERE / "results"
H_VALUES = (1e-2, 1e-3, 1e-4)
STAGES = ("stage:gather", "stage:J", "stage:rho_next", "stage:rho_prev")
BACKGROUND = ("electrons", "helium", "nitrogen")
TERMS = ("field", "background", "elec_from_N", "total")


# n_c
# ---


def compute_n_c(size: str) -> dict:
    """First step at which the laser peak reaches S(z) = 0.1 at nominal theta."""
    from scipy.constants import c
    from scipy.optimize import brentq

    config = load_config(size)
    plasma, laser, grid = config["plasma"], config["laser"], config["grid"]
    z_lo, z_c = plasma["z_lo"], plasma["z_c"]
    if density_shape(z_lo, plasma) >= 0.1:
        z_01 = z_lo
    else:
        z_01 = brentq(lambda z: density_shape(z, plasma) - 0.1, z_lo, z_c)
    dz = (grid["zmax"] - grid["zmin"]) / grid["Nz"]
    dt = grid["dt_cells"] * dz / c
    n_c = int(np.ceil((z_01 - laser["z0"]) / (c * dt)))
    out = {
        "size": size,
        "n_c": n_c,
        "z_S01_m": z_01,
        "laser_z0_m": laser["z0"],
        "dt_s": dt,
        "config_sha256": config_sha256(config),
    }
    (RESULTS / f"n_c_{size}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def load_n_c(size: str) -> int:
    config = load_config(size)
    data = json.loads((RESULTS / f"n_c_{size}.json").read_text())
    if data["config_sha256"] != config_sha256(config):
        raise RuntimeError(f"n_c_{size}.json belongs to another configuration.")
    return int(data["n_c"])


# Runs
# ----


def _terms(values: dict) -> dict:
    """Group the J_chk terms into field, background, elec_from_N, total."""
    out = {
        "field": values["field"],
        "background": sum(values[name] for name in BACKGROUND),
        "elec_from_N": values["elec_from_N"],
    }
    out["total"] = sum(values[name] for name in values)
    return out


def _decision_record(decisions) -> list:
    return [
        (d.iteration, d.n_move, sorted(d.injected.items()), sorted(d.removed.items()))
        for d in decisions
        if d.exchanged or d.n_move
    ]


class KeyRecorder:
    """Hook that records or compares per-stage cell keys by tracker ID.

    Section 6.3 rule 8. The records live on the host. With `stride > 1`,
    only the tracker IDs divisible by `stride` are recorded (Phase P6 step 0:
    256 on Twin-S, where full records do not fit in device memory); K(h)
    then counts the sampled keys, and `sampled`/`total` give the fraction.
    """

    def __init__(self, reference=None, stride: int = 1):
        self.reference = reference
        self.stride = int(stride)
        self.records = {} if reference is None else None
        self.differences = []  # (step, species, count) per stage event
        self.sampled = 0
        self.total = 0

    def __call__(self, state, name):
        if name not in STAGES:
            return
        import cupy

        from fbpic.particles.utilities.cuda_sorting import get_cell_idx_per_particle
        from fbpic.utils.cuda import cuda_tpb_bpg_1d

        grid = state.sim.fld.interp[0]
        for species, sp in state.species.items():
            n = int(sp.Ntot)
            key = (state.iteration, name, species)
            if n == 0:
                ids = cupy.zeros(0, np.int64)
                cells = cupy.zeros(0, np.int32)
            else:
                cells = cupy.empty(n, np.int32)
                scratch = cupy.empty(n, np.int32)
                blocks, threads = cuda_tpb_bpg_1d(n)
                get_cell_idx_per_particle[blocks, threads](
                    cells,
                    scratch,
                    sp.x,
                    sp.y,
                    sp.z,
                    grid.invdz,
                    grid.zmin,
                    grid.Nz,
                    grid.invdr,
                    grid.rmin,
                    grid.Nr,
                )
                ids = sp.tracker.id[:n]
                if self.stride > 1:
                    keep = ids % self.stride == 0
                    ids, cells = ids[keep], cells[keep]
                order = cupy.argsort(ids)
                ids, cells = ids[order], cells[order]
            ids, cells = cupy.asnumpy(ids), cupy.asnumpy(cells)
            self.sampled += ids.size
            self.total += n
            if self.records is not None:
                self.records[key] = (ids, cells)
                continue
            ref_ids, ref_cells = self.reference.pop(key)
            if ref_ids.size == ids.size and np.array_equal(ref_ids, ids):
                count = int(np.count_nonzero(ref_cells != cells))
            else:
                common, i, j = np.intersect1d(ref_ids, ids, return_indices=True)
                count = int(np.count_nonzero(ref_cells[i] != cells[j]))
                count += int(ref_ids.size + ids.size - 2 * common.size)
            self.differences.append((state.iteration, species, count))


def first_mismatch(record, reference) -> int | None:
    """Iteration of the first decision that differs from the reference."""
    for a, b in zip(record, reference):
        if a != b:
            return min(a[0], b[0])
    if len(record) != len(reference):
        return min(len(record), len(reference))
    return None


def count_valid(mismatch, horizon) -> bool:
    """Rule 2: the decisions up to `horizon` steps equal the AD run's."""
    return mismatch is None or mismatch >= horizon


def primal_run(
    build,
    theta,
    horizons,
    n_steps,
    functional,
    hooks=None,
    begin_kwargs=None,
    emit_masks=None,
    loss=None,
):
    """Plain `step` run; J_chk terms at each horizon and the decisions.

    Args:
        emit_masks: {iteration: device array of parent IDs} for every step,
            from the AD run (frozen emissions), or None (thresholds active).
        loss: (objective, horizon): also store the residuals r_w of the
            residual objective at that horizon under the key "r_w".
    """
    from fbpic.autodiff.stepper import begin, step

    sim, species, _ = build(theta)
    state = begin(sim, species, **(begin_kwargs or {}))
    if emit_masks is not None:
        state.emit_masks = emit_masks
    if hooks is not None:
        state.hooks = {"*": hooks}
    check = functional(state)
    values, decisions = {}, []
    for n in range(1, n_steps + 1):
        decisions.append(step(state)[2])
        if n in horizons:
            values[n] = _terms({k: v[0] for k, v in check(state).items()})
        if loss is not None and n == loss[1]:
            values.setdefault(n, {})["r_w"] = loss[0](state).residuals
    return values, _decision_record(decisions)


def emission_masks(decisions) -> dict | None:
    """Emitting parent IDs of every step of a run, keyed by iteration."""
    import cupy

    if all(d.emitted_parent_ids is None for d in decisions):
        return None
    return {
        d.iteration: (
            cupy.zeros(0, np.uint64)
            if d.emitted_parent_ids is None
            else d.emitted_parent_ids
        )
        for d in decisions
    }


def ad_run(
    build, theta, horizons, n_steps, functional, bounds, begin_kwargs=None, loss=None
):
    """`jvp` run with the E_L tangent; J and dJ/dtheta_hat per term and horizon."""
    import cupy

    from fbpic.autodiff.driver import contract, jvp, tangent_norms

    lo, hi = bounds
    out = {"values": {}, "grad_hat": {}, "norms": []}
    holder = {}

    def callback(state, tangents, decision):
        n = decision.iteration + 1
        if "check" not in holder:
            holder["check"] = functional(state)
        if n in horizons:
            terms = holder["check"](state)
            values, grads = {}, {}
            for name, (J, g) in terms.items():
                values[name] = J
                grads[name] = float(contract(g, tangents, state.sim)[0]) * (
                    hi[0] - lo[0]
                )
            out["values"][n] = _terms(values)
            out["grad_hat"][n] = _terms(grads)
        if loss is not None and n == loss[1]:
            value = loss[0](state)
            out["loss"] = {
                "r_w": value.residuals,
                "jac_hat": value.contract(tangents)[:, 0] * (hi[0] - lo[0]),
            }
        if n % 5 == 0 or n <= 10:
            norms = tangent_norms(tangents)[0]
            out["norms"].append({"step": n, **norms})

    result = jvp(
        theta, build, ("E_L",), n_steps, callback=callback, begin_kwargs=begin_kwargs
    )
    cupy.cuda.runtime.deviceSynchronize()
    out["decisions"] = _decision_record(result.decisions)
    out["emit_masks"] = emission_masks(result.decisions)
    out["births"] = [
        0 if d.emitted_parent_ids is None else int(d.emitted_parent_ids.size)
        for d in result.decisions
    ]
    out["info"] = {k: v for k, v in result.info.items() if k != "tangent_norms"}
    return out


# Evaluation
# ----------


def evaluate(g_ad, fd, sigma, slope, horizon):
    """Section 6.4 errors, tolerances, and the pass decision for one term."""
    rows = []
    g_inf = max([abs(g_ad)] + [abs(r["g_fd"]) for r in fd.values()])
    for h, r in fd.items():
        g_fd = r["g_fd"]
        denominator = max(abs(g_ad), abs(g_fd), 1e-6 * abs(g_fd))
        e = abs(g_ad - g_fd) / denominator if denominator > 0 else 0.0
        noise = 10 * np.sqrt(2) * sigma / (2 * h * abs(g_fd)) if g_fd != 0 else np.inf
        rows.append(
            {
                "h": h,
                "g_fd": g_fd,
                "e": e,
                "K": r["K"][horizon],
                "noise_term": noise,
                "tau": max(1e-4, noise),
                "usable": noise <= 1e-3 and r["counts_valid"],
                "counts_valid": r["counts_valid"],
            }
        )
    tiny = all(abs(v) <= 1e-6 * g_inf for v in [g_ad] + [r["g_fd"] for r in rows])
    valid_pairs = [
        (a, b)
        for a, b in pairwise(rows)
        if a["usable"] and b["usable"] and a["K"] == 0 and b["K"] == 0
    ]
    if tiny:
        verdict = "pass (both below 1e-6 |g|_inf)"
    elif valid_pairs:
        ok = any(a["e"] <= a["tau"] and b["e"] <= b["tau"] for a, b in valid_pairs)
        verdict = "pass" if ok else "FAIL"
    else:
        values = [r["g_fd"] for r in rows if r["usable"]] or [r["g_fd"] for r in rows]
        within_spread = min(values) <= g_ad <= max(values)
        within_slope = slope is not None and (
            min(slope["linear"], slope["quadratic"])
            <= g_ad
            <= max(slope["linear"], slope["quadratic"])
        )
        verdict = (
            "FD reference invalid (K > 0): pass"
            if within_spread or within_slope
            else "FD reference invalid (K > 0): FAIL"
        )
    return {"g_ad": g_ad, "rows": rows, "sigma_J": sigma, "verdict": verdict}


def _loss_results(ad_loss, base, fd, h_values, horizon):
    """Section 6.4 errors of each residual column and of J = |r_w|^2."""
    r0 = np.asarray(ad_loss["r_w"])
    jac = np.asarray(ad_loss["jac_hat"])
    reps = [np.asarray(b[horizon]["r_w"]) for b in base]
    sigma_r = np.max([np.abs(a - b) for a, b in pairwise(reps + reps[:1])], axis=0)
    J_reps = [float(r @ r) for r in reps]
    sigma_J = max(abs(a - b) for a, b in pairwise(J_reps + J_reps[:1]))
    out = {"r_w": r0.tolist(), "J": float(r0 @ r0), "residuals": []}
    for j in range(r0.size):
        terms_fd = {
            h: {
                "g_fd": (
                    fd[h]["plus"][horizon]["r_w"][j] - fd[h]["minus"][horizon]["r_w"][j]
                )
                / (2 * h),
                "K": fd[h]["K"],
                "counts_valid": all(count_valid(m, horizon) for m in fd[h]["mismatch"]),
            }
            for h in h_values
        }
        out["residuals"].append(
            evaluate(float(jac[j]), terms_fd, float(sigma_r[j]), None, horizon)
        )
    terms_fd = {
        h: {
            "g_fd": (
                float(fd[h]["plus"][horizon]["r_w"] @ fd[h]["plus"][horizon]["r_w"])
                - float(fd[h]["minus"][horizon]["r_w"] @ fd[h]["minus"][horizon]["r_w"])
            )
            / (2 * h),
            "K": fd[h]["K"],
            "counts_valid": all(count_valid(m, horizon) for m in fd[h]["mismatch"]),
        }
        for h in h_values
    }
    out["J_gradient"] = evaluate(float(2 * jac @ r0), terms_fd, sigma_J, None, horizon)
    for j, row in enumerate(out["residuals"]):
        es = " ".join(f"{r['e']:.1e}" for r in row["rows"])
        print(f"loss r_{j:2d}: g_AD={row['g_ad']:+.4e} e(h)={es} -> {row['verdict']}")
    row = out["J_gradient"]
    es = " ".join(f"{r['e']:.1e}" for r in row["rows"])
    print(f"loss J: g_AD={row['g_ad']:+.4e} e(h)={es} -> {row['verdict']}")
    return out


def slice_slopes(points, values):
    """Linear and quadratic least-squares slopes at the center of a slice."""
    x = np.asarray(points) - np.mean(points)
    y = np.asarray(values)
    linear = np.polyfit(x, y, 1)[0]
    quadratic = np.polyfit(x, y, 2)[1]
    return {"linear": float(linear), "quadratic": float(quadratic)}


def run_l4(
    size: str,
    n_steps: int | None,
    n_base: int,
    slice_points: int,
    h_values=H_VALUES,
    tag: str = "",
    key_stride: int = 1,
    ionization: str = "none",
    horizons_override=None,
    loss_target: str | None = None,
):
    from fbpic.autodiff.loss import CheckFunctional

    config = load_config(size)
    build = twin_builder(config, ionization=ionization)
    begin_kwargs = None
    if ionization == "ev":
        begin_kwargs = {
            "ionization": "ev",
            "ev_species": "nitrogen",
            "ev_target": "elec_from_N",
            "ev_options": {
                "epsilon": config["ev"]["epsilon"],
                "K_e": config["ev"]["K_e"],
            },
        }
    lo, hi = (
        np.asarray(v, float) for v in (config["theta"]["lo"], config["theta"]["hi"])
    )
    theta0 = np.asarray(config["theta"]["nominal"], float)
    hat0 = (theta0[0] - lo[0]) / (hi[0] - lo[0])
    n_c = load_n_c(size)
    pass_horizons = sorted({1, 10, 50, n_c + 10, n_c + 50, n_c + 200})
    n_steps = n_steps or config["num_steps"]
    report_horizons = [n for n in (1000, n_steps) if n not in pass_horizons]
    horizons = sorted(
        {n for n in pass_horizons + report_horizons if n <= n_steps} | {5, 20, 30, 40}
    )
    if horizons_override:
        horizons = sorted(set(horizons_override))
        pass_horizons = horizons
    horizons = [n for n in horizons if n <= n_steps]
    lambda0 = config["laser"]["lambda0"]

    def functional(state):
        return CheckFunctional(state, lambda0, seed=0)

    def theta_at(hat):
        theta = theta0.copy()
        theta[0] = lo[0] + hat * (hi[0] - lo[0])
        return theta

    record = {
        "size": size,
        "ionization": ionization,
        "config_sha256": config_sha256(config),
        "theta_nominal": theta0.tolist(),
        "theta_hat_E": hat0,
        "n_c": n_c,
        "pass_horizons": pass_horizons,
        "horizons": horizons,
        "h_values": list(h_values),
    }
    t0 = time.perf_counter()
    loss = None
    if loss_target is not None:
        from residual_objective import residual_objective

        loss = (residual_objective(config, loss_target), n_steps)
        record["loss"] = {"target": loss_target, "horizon": n_steps}
    ad = ad_run(
        build,
        theta0,
        set(horizons),
        n_steps,
        functional,
        (lo, hi),
        begin_kwargs,
        loss,
    )
    masks = ad.pop("emit_masks")
    record["births_per_step"] = ad.pop("births")
    run_kwargs = {"begin_kwargs": begin_kwargs, "emit_masks": masks, "loss": loss}
    record["check_functional"] = functional(_FakeState(build, theta0)).record()
    record["ad_info"] = ad["info"]
    record["tangent_norms"] = ad["norms"]
    print(f"AD run: {ad['info']['time_per_step_s'] * 1e3:.1f} ms per step")

    base, base_mismatch = [], []
    for i in range(n_base):
        values, decisions = primal_run(
            build, theta0, set(horizons), n_steps, functional, **run_kwargs
        )
        base_mismatch.append(first_mismatch(decisions, ad["decisions"]))
        base.append(values)
    record["base_first_mismatch"] = base_mismatch
    print(f"base runs: first decision mismatch {base_mismatch}")
    sigma = {
        n: {
            t: max(abs(a[n][t] - b[n][t]) for a, b in pairwise(base + base[:1]))
            for t in TERMS
        }
        for n in horizons
    }
    fd = {}
    for h in h_values:
        recorder = KeyRecorder(stride=key_stride)
        plus, d_plus = primal_run(
            build,
            theta_at(hat0 + h),
            set(horizons),
            n_steps,
            functional,
            recorder,
            **run_kwargs,
        )
        comparer = KeyRecorder(recorder.records, key_stride)
        minus, d_minus = primal_run(
            build,
            theta_at(hat0 - h),
            set(horizons),
            n_steps,
            functional,
            comparer,
            **run_kwargs,
        )
        del recorder
        mismatch = [
            first_mismatch(d_plus, ad["decisions"]),
            first_mismatch(d_minus, ad["decisions"]),
        ]
        k_cum, k_species = {}, {}
        for n in horizons:
            k_cum[n] = sum(c for it, _, c in comparer.differences if it < n)
            k_species[n] = {
                name: sum(
                    c for it, sp, c in comparer.differences if it < n and sp == name
                )
                for name in ("electrons", "helium", "nitrogen", "elec_from_N")
            }
        fd[h] = {
            "plus": plus,
            "minus": minus,
            "K": k_cum,
            "K_species": k_species,
            "mismatch": mismatch,
            "key_sample": {
                "stride": key_stride,
                "sampled": comparer.sampled,
                "total": comparer.total,
            },
        }
        print(f"h={h}: K(h) at horizons {k_cum}; first decision mismatch {mismatch}")
    slice_hats = hat0 + np.linspace(-0.05, 0.05, slice_points)
    slices = []
    for hat in slice_hats if slice_points >= 3 else []:
        values, _ = primal_run(
            build, theta_at(hat), set(horizons), n_steps, functional, **run_kwargs
        )
        slices.append(values)
    results = {}
    for n in horizons:
        results[n] = {}
        for t in TERMS:
            terms_fd = {
                h: {
                    "g_fd": (fd[h]["plus"][n][t] - fd[h]["minus"][n][t]) / (2 * h),
                    "K": fd[h]["K"],
                    "counts_valid": all(count_valid(m, n) for m in fd[h]["mismatch"]),
                }
                for h in h_values
            }
            if t == "elec_from_N" and ad["grad_hat"][n][t] == 0.0:
                results[n][t] = {"verdict": "skipped (identically zero)"}
                continue
            slope = (
                slice_slopes(slice_hats - hat0, [s[n][t] for s in slices])
                if slices
                else None
            )
            row = evaluate(ad["grad_hat"][n][t], terms_fd, sigma[n][t], slope, n)
            row["J"] = ad["values"][n][t]
            row["slice_slopes"] = slope
            row["slice_values"] = [s[n][t] for s in slices]
            results[n][t] = row
    record["results"] = results
    if loss is not None:
        record["loss"]["results"] = _loss_results(
            ad["loss"], base, fd, h_values, n_steps
        )
    record["fd"] = {
        h: {
            "K": fd[h]["K"],
            "K_species": fd[h]["K_species"],
            "mismatch": fd[h]["mismatch"],
            "key_sample": fd[h]["key_sample"],
        }
        for h in h_values
    }
    record["slice_hats"] = slice_hats.tolist()
    record["wall_time_s"] = time.perf_counter() - t0
    phase = "p6" if ionization == "ev" else "p5"
    out = RESULTS / f"{phase}_l4_{size}{tag}.json"
    out.write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    _print_summary(record)
    return record


class _FakeState:
    """A built, not stepped, state for `CheckFunctional.record`."""

    def __init__(self, build, theta):
        sim, species, _ = build(theta)
        self.sim, self.species = sim, species


def _json_default(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value))


def _print_summary(record):
    print(f"\nL4 {record['size']}: n_c = {record['n_c']}")
    for n, terms in record["results"].items():
        tag = "pass horizon" if n in record["pass_horizons"] else "report"
        for t, row in terms.items():
            if "rows" not in row:
                print(f"N={n:4d} {t:12s} {row['verdict']}")
                continue
            es = " ".join(
                f"h={r['h']:.0e}: e={r['e']:.1e} tau={r['tau']:.1e} K={r['K']}"
                + ("" if r["counts_valid"] else " (counts differ)")
                for r in row["rows"]
            )
            print(
                f"N={n:4d} ({tag}) {t:12s} g_AD={row['g_ad']:+.4e} {es} -> {row['verdict']}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("nc", "l4"))
    parser.add_argument("size")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--base", type=int, default=5)
    parser.add_argument("--slice-points", type=int, default=41)
    parser.add_argument("--h", type=float, nargs="+", default=list(H_VALUES))
    parser.add_argument("--tag", default="")
    parser.add_argument("--ionization", default="none", choices=("none", "ev"))
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument(
        "--loss-target",
        default=None,
        help="also check the residual Jacobian against this target at --steps",
    )
    parser.add_argument(
        "--key-stride",
        type=int,
        default=1,
        help="record the cell keys of tracker IDs divisible by this (rule 8)",
    )
    args = parser.parse_args()
    if args.stage == "nc":
        print(compute_n_c(args.size))
    else:
        run_l4(
            args.size,
            args.steps,
            args.base,
            args.slice_points,
            args.h,
            args.tag,
            args.key_stride,
            args.ionization,
            args.horizons,
            args.loss_target,
        )


if __name__ == "__main__":
    main()
