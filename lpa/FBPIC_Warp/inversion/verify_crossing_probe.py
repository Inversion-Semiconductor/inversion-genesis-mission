"""Crossing probe: free FD minus frozen FD at one emission threshold.

Inversion plan Section 6.3 rule 5, Phase P6 step 5. Twin-XS with EV. A hook
at `before_ionization` (after the gather and the push) overwrites the
gathered fields of every N ion: zero for all ions except one probe ion,
which sees the prescribed ramp

    E_x(n) = a E_0 min(1, (n + 1) / n_ramp),   B = 0.

Only the probe ion ionizes, so the parameter a enters the run only through
its EV update and its emissions. The objective is the `elec_from_N` term of
J_chk at N_t.

1. Scan a and bisect the first a_c at which the emission steps of the probe
   ion change: the threshold crossing.
2. Jump: Delta J = J_free(a_c + delta) - J_free(a_c - delta), delta small.
3. At a_0 = a_c + d, compare free central FD (thresholds active) with frozen
   central FD (emissions of the a_0 run reused as `emit_mask`) for h on
   both sides of |d|. Expected: free - frozen = Delta J / (2 h) when a_c is
   inside [a_0 - h, a_0 + h], and 0 otherwise. Inside, the two branches
   also differ in slope, g_above - g_below, over the part of the stencil
   past a_c. The full prediction is

       free - frozen = (Delta J + sign(d) (g_above - g_below) (|d| - h)) / (2 h),

   with the slopes from the frozen FD at h = 1e-5 on each side.
   Pass: the prediction holds within 1e-3 relative to Delta J / (2 h), and
   free - frozen is 0 outside within 1e-9 relative to the slope.

Usage (from lpa/FBPIC_Warp, CUDA_VISIBLE_DEVICES set):

    python inversion/verify_crossing_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from builders import twin_builder  # noqa: E402
from twin_config import config_sha256, load_config  # noqa: E402

N_STEPS = 20
N_RAMP = 20
E0 = 5e12
SIZE = "twin_xs"


class Probe:
    """Builds and runs the one-ion probe; caches nothing between runs."""

    def __init__(self):
        self.config = load_config(SIZE)
        self.build = twin_builder(self.config, ionization="ev")
        self.theta = np.asarray(self.config["theta"]["nominal"], float)
        self.begin_kwargs = {
            "ionization": "ev",
            "ev_species": "nitrogen",
            "ev_target": "elec_from_N",
            "ev_options": {
                "epsilon": self.config["ev"]["epsilon"],
                "K_e": self.config["ev"]["K_e"],
            },
        }
        self.probe_id = None

    def run(self, a, emit_masks=None):
        """J and the emission steps of the probe ion at amplitude a."""
        import cupy

        from fbpic.autodiff.loss import CheckFunctional
        from fbpic.autodiff.stepper import begin, step

        sim, species, _ = self.build(self.theta)
        state = begin(sim, species, **self.begin_kwargs)
        ion = state.species["nitrogen"]
        if self.probe_id is None:
            self.probe_id = int(ion.tracker.id[: int(ion.Ntot)].min())
        probe_id = self.probe_id

        def prescribe(state):
            n = int(ion.Ntot)
            for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                getattr(ion, name)[:n] = 0.0
            ramp = min(1.0, (state.iteration + 1) / N_RAMP)
            here = ion.tracker.id[:n] == probe_id
            ion.Ex[:n] = cupy.where(here, a * E0 * ramp, 0.0)

        state.hooks = {"before_ionization": prescribe}
        if emit_masks is not None:
            state.emit_masks = emit_masks
        decisions = [step(state)[2] for _ in range(N_STEPS)]
        check = CheckFunctional(state, self.config["laser"]["lambda0"], seed=0)
        J = float(check(state)["elec_from_N"][0])
        emitted = [
            d.iteration
            for d in decisions
            if bool(cupy.any(d.emitted_parent_ids == probe_id))
        ]
        masks = {d.iteration: d.emitted_parent_ids for d in decisions}
        return J, emitted, masks


def bisect_crossing(probe, lo, hi, tol=1e-13):
    """a_c between lo and hi, where the emission steps change."""
    _, e_lo, _ = probe.run(lo)
    _, e_hi, _ = probe.run(hi)
    assert e_lo != e_hi, "No crossing in the bracket."
    while (hi - lo) > tol * hi:
        mid = 0.5 * (lo + hi)
        _, e_mid, _ = probe.run(mid)
        if e_mid == e_lo:
            lo = mid
        else:
            hi = mid
    return lo, hi, e_lo, e_hi


def main():
    probe = Probe()
    scan = np.linspace(0.90, 1.10, 11)
    patterns = []
    for a in scan:
        J, emitted, _ = probe.run(a)
        patterns.append(emitted)
        print(f"a={a:.3f} J={J:.6e} emissions at {emitted}")
    k = next(i for i in range(len(scan) - 1) if patterns[i] != patterns[i + 1])
    lo, hi, e_lo, e_hi = bisect_crossing(probe, scan[k], scan[k + 1])
    a_c = 0.5 * (lo + hi)
    print(f"a_c = {a_c:.15f} (bracket {hi - lo:.1e}); {e_lo} -> {e_hi}")
    delta = 1e-9
    J_minus, _, _ = probe.run(a_c - delta)
    J_plus, _, _ = probe.run(a_c + delta)
    jump = J_plus - J_minus
    print(f"Delta J = {jump:.6e}")
    rows = []
    H_VALUES = (1e-3, 5e-4, 2e-4, 1e-4, 1e-5)
    for d in (+3e-4, -3e-4):
        a0 = a_c + d
        _, _, masks = probe.run(a0)
        for h in H_VALUES:
            Jfp, _, _ = probe.run(a0 + h)
            Jfm, _, _ = probe.run(a0 - h)
            Jzp, ep, _ = probe.run(a0 + h, masks)
            Jzm, em, _ = probe.run(a0 - h, masks)
            free = (Jfp - Jfm) / (2 * h)
            frozen = (Jzp - Jzm) / (2 * h)
            inside = abs(d) < h
            rows.append(
                {
                    "d": d,
                    "h": h,
                    "inside": bool(inside),
                    "free": free,
                    "frozen": frozen,
                    "free_minus_frozen": free - frozen,
                }
            )
    slope = {np.sign(r["d"]): r["frozen"] for r in rows if r["h"] == min(H_VALUES)}
    g_above, g_below = slope[1.0], slope[-1.0]
    for row in rows:
        h, d = row["h"], row["d"]
        leading = jump / (2 * h)
        if row["inside"]:
            branch = np.sign(d) * (g_above - g_below) * (abs(d) - h)
            expected = (jump + branch) / (2 * h)
            row["error_leading"] = abs(row["free_minus_frozen"] - leading) / abs(
                leading
            )
            row["relative_error"] = abs(row["free_minus_frozen"] - expected) / abs(
                leading
            )
            row["pass"] = bool(row["relative_error"] <= 1e-3)
        else:
            expected = 0.0
            row["relative_error"] = abs(row["free_minus_frozen"]) / abs(row["frozen"])
            row["pass"] = bool(row["relative_error"] <= 1e-9)
        row["expected"] = expected
        print(
            f"d={d:+.0e} h={h:.0e} inside={row['inside']}: free={row['free']:+.6e} "
            f"frozen={row['frozen']:+.6e} free-frozen={row['free_minus_frozen']:+.6e} "
            f"DJ/2h={leading:+.6e} predicted={expected:+.6e} "
            f"rel.err={row['relative_error']:.2e}"
        )
    passed = all(r["pass"] for r in rows)
    record = {
        "size": SIZE,
        "config_sha256": config_sha256(probe.config),
        "n_steps": N_STEPS,
        "E0": E0,
        "n_ramp": N_RAMP,
        "probe_id": probe.probe_id,
        "scan": {"a": scan.tolist(), "emission_steps": patterns},
        "a_c": a_c,
        "emissions_below": e_lo,
        "emissions_above": e_hi,
        "jump": jump,
        "slope_above": g_above,
        "slope_below": g_below,
        "rows": rows,
        "pass": bool(passed),
    }
    out = HERE / "results" / "p6_crossing_probe.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
