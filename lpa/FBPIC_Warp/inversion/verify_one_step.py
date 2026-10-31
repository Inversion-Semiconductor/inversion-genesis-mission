"""L3: the tangent of one full `step` against central FD of `step`.

Inversion plan Sections 6.1 (L3) and 6.3 (rule 6). Twin-XS with `p_nz = 2`
(so that resting particles do not sit on shape boundaries), nominal theta.
Phase P5: `ionization="none"`, one ordinary step and the exchange steps 14
and 154 (154 is the first exchange with removal). Phase P6 (`--ionization
ev`): steps with emissions. The base step records the emitting parent IDs,
and every tangent and FD step reuses them as `emit_mask` (frozen decisions,
Section 6.3 rule 3). The particle block then also perturbs the EV arrays
(pop5, pop6, pop7, w_pending) of the ions. The ion weight w stays fixed:
its tangent is zero under D21, and the EV tangents treat it as a constant.

For each step n:

1. Run the plain stepper to iteration n and snapshot the state.
2. Base step from the snapshot with hooks: record the per-stage cell keys
   and the axis and `rmax_gather` branches of every particle (the branch
   signature), and mask every particle whose fractional cell coordinate in
   z or r, at gather, J deposit, rho_next deposit, or the exchange-step
   rho_prev deposit, lies within 1e-3 of an integer.
3. Two blocks, each with 2 random directions: the particle inputs (x, y, z,
   ux, uy, uz, inv_gamma, w of every species, masked) and the field inputs
   (spectral Ep, Em, Ez, Bp, Bm, Bz, rho_prev of every mode; the interp E
   and B follow as spect2interp of the direction, Section 4.2 rule 3).
4. Tangent step from the snapshot with that direction as the tangent.
5. Central FD from the snapshot for each epsilon; an epsilon whose branch
   signature or decisions differ from the base is rejected.
6. Error (vector form of Section 6.1) over the whole output vector: the
   spectral state fields per mode and each particle component per species,
   matched by tracker ID. Each output array is divided by a fixed scale,
   its largest primal value in the base step, so that one infinity norm
   over mixed units is meaningful:
       |J d - FD|_inf / max(|J d|_inf, |FD|_inf, 1e-12)  (scaled outputs).
   Per-output errors are recorded as diagnostics. Pass: two adjacent
   accepted epsilons with error <= 1e-6.

Usage: python inversion/verify_one_step.py [--steps 30 14 154]
       python inversion/verify_one_step.py --ionization ev --steps 24 28
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
from state_snapshot import restore, snapshot  # noqa: E402
from twin_config import config_sha256, load_config  # noqa: E402

SPECT_STATE = ("Ep", "Em", "Ez", "Bp", "Bm", "Bz", "rho_prev")
PARTICLE = ("x", "y", "z", "ux", "uy", "uz", "inv_gamma", "w")
EV_NAMES = ("pop5", "pop6", "pop7", "w_pending")
STAGES = ("stage:gather", "stage:J", "stage:rho_next", "stage:rho_prev")
EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
TOLERANCE = 1e-6
MASK_DISTANCE = 1e-3


class StageRecorder:
    """Hook: per-stage cell keys, branches, and near-boundary IDs by tracker ID."""

    def __init__(self):
        self.signature = {}
        self.near = {}

    def __call__(self, state, name):
        if name not in STAGES:
            return
        import cupy

        sim = state.sim
        grid = sim.fld.interp[0]
        rmax_gather = sim.comm.get_rmax(with_damp=False)
        for species, sp in state.species.items():
            n = int(sp.Ntot)
            if n == 0:
                continue
            ids = sp.tracker.id[:n]
            r = cupy.sqrt(sp.x[:n] ** 2 + sp.y[:n] ** 2)
            r_cell = grid.invdr * (r - grid.rmin) - 0.5
            z_cell = grid.invdz * (sp.z[:n] - grid.zmin) - 0.5
            ir = cupy.minimum(cupy.ceil(r_cell), grid.Nr).astype(np.int64)
            iz = cupy.ceil(z_cell).astype(np.int64)
            branch = (r == 0).astype(np.int64) + 2 * (r >= rmax_gather)
            key = (iz * (grid.Nr + 1) + ir) * 4 + branch
            order = cupy.argsort(ids)
            self.signature[(name, species)] = (ids[order], key[order])
            distance = cupy.minimum(
                cupy.abs(z_cell - cupy.rint(z_cell)),
                cupy.abs(r_cell - cupy.rint(r_cell)),
            )
            flagged = ids[distance < MASK_DISTANCE]
            previous = self.near.get(species)
            self.near[species] = (
                flagged if previous is None else cupy.union1d(previous, flagged)
            )


def _same_signature(a, b):
    import cupy

    if a.keys() != b.keys():
        return False
    for key in a:
        (ia, ka), (ib, kb) = a[key], b[key]
        if ia.shape != ib.shape or not bool(cupy.all(ia == ib)):
            return False
        if not bool(cupy.all(ka == kb)):
            return False
    return True


def _decisions(d):
    import cupy

    emitted = d.emitted_parent_ids
    emitted = () if emitted is None else tuple(cupy.asnumpy(emitted).tolist())
    return (d.n_move, sorted(d.injected.items()), sorted(d.removed.items()), emitted)


def _components(state, species):
    if state.ev is not None and species == state.ev_species:
        return PARTICLE + EV_NAMES
    return PARTICLE


def _array_of(state, species, comp):
    if comp in EV_NAMES:
        return getattr(state.ev, comp)
    return getattr(state.species[species], comp)


def _output(state):
    """Spectral state fields and particle arrays sorted by tracker ID (host)."""
    import cupy

    out = {}
    for m, grid in enumerate(state.sim.fld.spect):
        for name in SPECT_STATE:
            out[f"spect{m}.{name}"] = cupy.asnumpy(getattr(grid, name))
    orders = {}
    for species, sp in state.species.items():
        n = int(sp.Ntot)
        order = cupy.argsort(sp.tracker.id[:n])
        orders[species] = order
        for comp in _components(state, species):
            array = _array_of(state, species, comp)[:n]
            out[f"{species}.{comp}"] = cupy.asnumpy(array[order])
    return out, orders


def _tangent_output(state, tangents, orders):
    import cupy

    out = {}
    for m in range(len(state.sim.fld.spect)):
        for name in SPECT_STATE:
            out[f"spect{m}.{name}"] = cupy.asnumpy(tangents.fields.spect[m][name][0])
    for species in state.species:
        for comp in _components(state, species):
            out[f"{species}.{comp}"] = cupy.asnumpy(
                tangents.particles[species][comp][0][orders[species]]
            )
    return out


def _direction(state, block, rng, masks):
    """Random direction on one block, as host arrays in the current order."""
    import cupy

    sim = state.sim
    d = {"fields": {}, "particles": {}}
    if block == "fields":
        for m, grid in enumerate(sim.fld.spect):
            for name in SPECT_STATE:
                kind = "rho" if name.startswith("rho") else name[0]
                scale = max(
                    float(cupy.abs(getattr(g, n)).max())
                    for g in sim.fld.spect
                    for n in SPECT_STATE
                    if (n.startswith("rho") if kind == "rho" else n[0] == kind)
                )
                shape = getattr(grid, name).shape
                value = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                d["fields"][(m, name)] = 1e-3 * scale * value
        return d
    cell = min(sim.comm.dz, sim.comm.dr)
    for species, sp in state.species.items():
        n = int(sp.Ntot)
        mask = masks[species]
        for comp in _components(state, species):
            if comp == "w" and state.ev is not None and species == state.ev_species:
                continue
            primal = cupy.asnumpy(_array_of(state, species, comp)[:n])
            if comp in ("x", "y", "z"):
                scale = 0.05 * cell
            else:
                scale = 1e-2 * float(np.max(np.abs(primal), initial=0)) or 1e-3
            d["particles"][(species, comp)] = scale * rng.normal(size=n) * mask
    return d


def _apply_direction(state, d, eps):
    """Add eps * d to the restored state; recompute the interp E and B."""
    import cupy

    sim = state.sim
    for (m, name), value in d["fields"].items():
        array = getattr(sim.fld.spect[m], name)
        array += eps * cupy.asarray(value)
    for (species, comp), value in d["particles"].items():
        n = int(state.species[species].Ntot)
        _array_of(state, species, comp)[:n] += eps * cupy.asarray(value)
    sim.fld.spect2interp("E")
    sim.fld.spect2interp("B")


def _seed(state, d):
    import cupy

    from fbpic.autodiff.state import TangentState

    sim = state.sim
    tangents = TangentState.zeros(state, 1)
    for (m, name), value in d["fields"].items():
        tangents.fields.spect[m][name][0] = cupy.asarray(value)
    for (species, comp), value in d["particles"].items():
        tangents.particles[species][comp][0] = cupy.asarray(value)

    def derive():
        sim.fld.spect2interp("E")
        sim.fld.spect2interp("B")

    tangents.fields.apply(sim.fld, derive)
    return tangents


def check_step(build, theta, n_step, rng, n_directions=2, begin_kwargs=None):
    from fbpic.autodiff.stepper import begin, step

    sim, species, _ = build(theta)
    state = begin(sim, species, **(begin_kwargs or {}))
    for _ in range(n_step):
        step(state)
    snap = snapshot(state)
    ev = state.ev is not None
    # Base step, twice: the second checks that restore is complete.
    bases = []
    for _ in range(2):
        restore(state, snap)
        sim.fld.spect2interp("E")
        sim.fld.spect2interp("B")
        recorder = StageRecorder()
        state.hooks = {"*": recorder}
        decision = step(state)[2]
        out, orders = _output(state)
        bases.append((recorder, decision, out, orders))
    (base_rec, base_dec, base_out, orders0), (_, dec2, out2, _) = bases
    if ev:
        # Freeze the emissions of this step for every later step from snap.
        state.emit_masks = {n_step: base_dec.emitted_parent_ids.copy()}
    scales = {
        key: float(np.max(np.abs(value), initial=0)) or 1.0
        for key, value in base_out.items()
    }
    reproducibility = max(
        float(np.max(np.abs(out2[k] - v), initial=0))
        / (float(np.max(np.abs(v), initial=0)) or 1.0)
        for k, v in base_out.items()
    )
    assert _decisions(dec2) == _decisions(base_dec), "Restore is incomplete."
    restore(state, snap)
    import cupy

    masks = {}
    masked = {}
    for name, sp in state.species.items():
        n = int(sp.Ntot)
        ids = sp.tracker.id[:n]
        near = base_rec.near.get(name)
        if near is None or near.size == 0:
            keep = cupy.ones(n, bool)
        else:
            keep = ~cupy.isin(ids, near)
        masks[name] = cupy.asnumpy(keep).astype(float)
        masked[name] = int(n - int(keep.sum()))
    result = {
        "step": n_step,
        "exchange": bool(base_dec.exchanged),
        "decisions": _decisions(base_dec)[:3],
        "emissions": len(_decisions(base_dec)[3]),
        "restore_reproducibility": reproducibility,
        "masked_particles": masked,
        "blocks": {},
    }
    for block in ("particles", "fields"):
        directions = []
        for _ in range(n_directions):
            restore(state, snap)
            d = _direction(state, block, rng, masks)
            restore(state, snap)
            sim.fld.spect2interp("E")
            sim.fld.spect2interp("B")
            tangents = _seed(state, d)
            state.hooks = {}
            step(state, tangents)
            _, orders = _output(state)
            jd = _tangent_output(state, tangents, orders)
            samples = []
            for eps in EPSILONS:
                values, accepted = [], True
                for sign in (1, -1):
                    restore(state, snap)
                    _apply_direction(state, d, sign * eps)
                    recorder = StageRecorder()
                    state.hooks = {"*": recorder}
                    decision = step(state)[2]
                    out, _ = _output(state)
                    values.append(out)
                    if _decisions(decision) != _decisions(base_dec) or not (
                        _same_signature(recorder.signature, base_rec.signature)
                    ):
                        accepted = False
                if not accepted:
                    samples.append({"epsilon": eps, "accepted": False})
                    continue
                per_output = {}
                residual = tangent_norm = fd_norm = 0.0
                for key, analytical in jd.items():
                    upper, lower = values[0][key], values[1][key]
                    if upper.shape != analytical.shape:
                        accepted = False
                        break
                    numerical = (upper - lower) / (2 * eps)
                    scale = scales[key]
                    diff = float(np.max(np.abs(analytical - numerical), initial=0))
                    a_norm = float(np.max(np.abs(analytical), initial=0))
                    f_norm = float(np.max(np.abs(numerical), initial=0))
                    residual = max(residual, diff / scale)
                    tangent_norm = max(tangent_norm, a_norm / scale)
                    fd_norm = max(fd_norm, f_norm / scale)
                    per_output[key] = diff / max(a_norm, f_norm, 1e-300)
                if not accepted:
                    samples.append({"epsilon": eps, "accepted": False})
                    continue
                error = residual / max(tangent_norm, fd_norm, 1e-12)
                worst = max(
                    per_output,
                    key=lambda k: np.max(
                        np.abs(jd[k] - (values[0][k] - values[1][k]) / (2 * eps)),
                        initial=0,
                    )
                    / scales[k],
                )
                samples.append(
                    {
                        "epsilon": eps,
                        "accepted": True,
                        "error": error,
                        "worst_output": worst,
                        "per_output": per_output,
                    }
                )
            passed = any(
                a.get("accepted")
                and b.get("accepted")
                and a["error"] <= TOLERANCE
                and b["error"] <= TOLERANCE
                for a, b in pairwise(samples)
            )
            best = min((s["error"] for s in samples if s.get("accepted")), default=None)
            directions.append(
                {"passed": passed, "best_error": best, "samples": samples}
            )
            print(
                f"step {n_step} {block}: passed={passed} best={best} "
                + " ".join(
                    f"{s['epsilon']:.0e}:"
                    + (
                        f"{s['error']:.1e}({s['worst_output']})"
                        if s["accepted"]
                        else "rej"
                    )
                    for s in samples
                )
            )
        result["blocks"][block] = directions
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", default="twin_xs")
    parser.add_argument("--steps", type=int, nargs="+", default=[30, 14, 154])
    parser.add_argument("--p-nz", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--ionization", default="none", choices=("none", "ev"))
    args = parser.parse_args()
    config = load_config(args.size)
    build = twin_builder(config, ionization=args.ionization, p_nz=args.p_nz)
    begin_kwargs = None
    if args.ionization == "ev":
        begin_kwargs = {
            "ionization": "ev",
            "ev_species": "nitrogen",
            "ev_target": "elec_from_N",
            "ev_options": {
                "epsilon": config["ev"]["epsilon"],
                "K_e": config["ev"]["K_e"],
            },
        }
    theta = np.asarray(config["theta"]["nominal"], float)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    results = [
        check_step(build, theta, n, rng, begin_kwargs=begin_kwargs) for n in args.steps
    ]
    record = {
        "size": args.size,
        "ionization": args.ionization,
        "config_sha256": config_sha256(config),
        "p_nz": args.p_nz,
        "theta": theta.tolist(),
        "epsilons": EPSILONS,
        "tolerance": TOLERANCE,
        "mask_distance": MASK_DISTANCE,
        "results": results,
        "wall_time_s": time.perf_counter() - t0,
    }
    phase = "p6" if args.ionization == "ev" else "p5"
    out = HERE / "results" / f"{phase}_l3_{args.size}.json"
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    for r in results:
        verdicts = {b: [d["passed"] for d in v] for b, v in r["blocks"].items()}
        print(
            f"step {r['step']} (exchange={r['exchange']}): {verdicts}, "
            f"emissions {r['emissions']}, "
            f"restore reproducibility {r['restore_reproducibility']:.1e}, "
            f"masked {r['masked_particles']}"
        )


if __name__ == "__main__":
    main()
