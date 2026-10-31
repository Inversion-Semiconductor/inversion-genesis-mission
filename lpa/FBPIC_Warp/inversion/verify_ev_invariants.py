"""Phase P3 steps 3 and 4: EV charge test and invariants on Twin-S.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python verify_ev_invariants.py

Step 3: deposit rho from the N ions and `elec_from_N` before and after the EV
call; the difference must be below 1e-12 relative, on steps with and without
emission.
Step 4.1: closed cohort (ions of the first injection event and their
electrons), before any of them reaches the left window edge:
P5 + P6 + P7 = 1 to 1e-14, and sum w_e + sum W = sum w (Zbar - 5) to 1e-12.
Step 4.2: open window, with the removal ledgers: B = live + removed electron
weight, and B + W(live + removed) = ionized weight (live + removed), each to
1e-12 relative.
Step 4.3: the populations follow the tracker IDs through sorts and exchanges.
"""

from __future__ import annotations

import json

import cupy
import numpy as np

from fbpic.autodiff import ionization as ev
from fbpic.autodiff.state import StepDecisions
from fbpic.autodiff.stepper import _exchange, sort_all, step
from forward import build_ev_state
from twin_config import load_config, theta_from_hat

CHARGE_STEPS = (100, 300, 450, 500, 550, 650, 800, 1200)
COHORT_STEPS = (450, 550, 650, 700, 750)
LEDGER_STEPS = (450, 800, 1200, 1600, 2000)
SORT_STEPS = (460, 900)
EXCHANGE_STEPS = (476, 1008)


def _rho(state):
    """Deposit rho of the N ions and elec_from_N; return the interp arrays."""
    sim = state.sim
    sort_all(state)
    ev.set_deposit_charge(state)
    sim.deposit(
        "rho_next",
        species_list=[state.species["nitrogen"], state.species["elec_from_N"]],
    )
    ev.restore_charge(state)
    return [cupy.asnumpy(grid.rho).copy() for grid in sim.fld.interp]


def _by_id(state):
    ion = state.species["nitrogen"]
    n = int(ion.Ntot)
    ids = cupy.asnumpy(ion.tracker.id[:n])
    order = np.argsort(ids)
    arrays = [cupy.asnumpy(a)[order] for a in state.ev.arrays()]
    return ids[order], arrays


def main():
    config = load_config("twin_s")
    theta = theta_from_hat(config, config["theta_star_hat"])
    twin, state = build_ev_state(theta, config)
    report = {"charge": [], "cohort": [], "ledger": [], "follow_ids": []}
    charge = {}

    def before(st):
        if st.iteration in CHARGE_STEPS:
            charge["before"] = _rho(st)
        if st.iteration in SORT_STEPS:
            ids0, arrays0 = _by_id(st)
            sort_all(st)
            ids1, arrays1 = _by_id(st)
            same = np.array_equal(ids0, ids1) and all(
                np.array_equal(a, b) for a, b in zip(arrays0, arrays1)
            )
            report["follow_ids"].append(
                {"iteration": st.iteration, "event": "sort", "equal": bool(same)}
            )

    def after(st):
        if st.iteration in CHARGE_STEPS:
            rho_after = _rho(st)
            scale = max(np.abs(r).max() for r in charge["before"])
            diff = max(np.abs(a - b).max() for a, b in zip(rho_after, charge["before"]))
            report["charge"].append(
                {
                    "iteration": st.iteration,
                    "emitted": int(charge["emitted"]),
                    "relative_difference": float(diff / scale),
                }
            )

    state.hooks = {"before_ionization": before, "after_ionization": after}

    cohort_ids = None
    parents = {}
    n_total = config["num_steps"]
    while state.iteration < n_total:
        it = state.iteration
        if it in EXCHANGE_STEPS:
            ids0, arrays0 = _by_id(state)
            decisions = StepDecisions(it)
            _exchange(state, decisions)
            ids1, arrays1 = _by_id(state)
            common, k0, k1 = np.intersect1d(ids0, ids1, return_indices=True)
            same = all(np.array_equal(a[k0], b[k1]) for a, b in zip(arrays0, arrays1))
            new = ids1[~np.isin(ids1, ids0)]
            fresh = np.isin(ids1, new)
            fresh_ok = bool(
                np.all(arrays1[0][fresh] == 1.0)
                and all(np.all(a[fresh] == 0.0) for a in arrays1[1:])
            )
            report["follow_ids"].append(
                {
                    "iteration": it,
                    "event": "exchange",
                    "equal": bool(same),
                    "kept": len(common),
                    "removed": len(ids0) - len(common),
                    "injected": int(new.size),
                    "injected_initialized": fresh_ok,
                }
            )
        charge["emitted"] = 0
        _, _, decisions = step(state)
        if decisions.emitted_parent_ids is not None and report["charge"]:
            last = report["charge"][-1]
            if last["iteration"] == it:
                last["emitted"] = int(decisions.emitted_parent_ids.size)
        if cohort_ids is None and decisions.injected.get("nitrogen", 0) > 0:
            ion = state.species["nitrogen"]
            stop = int(ion.tracker.next_attributed_id)
            cohort_ids = (stop - decisions.injected["nitrogen"], stop)
            report["cohort_definition"] = {
                "iteration": it,
                "id_range": list(cohort_ids),
            }
        if state.iteration in COHORT_STEPS and cohort_ids is not None:
            report["cohort"].append(_cohort(state, cohort_ids, parents))
        if state.iteration in LEDGER_STEPS:
            report["ledger"].append(_ledger(state))

    print(json.dumps(report, indent=2))
    checks = {
        "charge": all(c["relative_difference"] < 1e-12 for c in report["charge"]),
        "charge_with_and_without_emission": any(
            c["emitted"] > 0 for c in report["charge"]
        )
        and any(c["emitted"] == 0 for c in report["charge"]),
        "cohort_complete": all(c["electrons_missing"] == 0 for c in report["cohort"]),
        "cohort_populations": all(
            c["max_population_error"] <= 1e-14 for c in report["cohort"]
        ),
        "cohort_weights": all(
            c["relative_weight_error"] <= 1e-12 for c in report["cohort"]
        ),
        "ledger": all(
            row["relative_emitted_error"] <= 1e-12
            and row["relative_balance_error"] <= 1e-12
            for row in report["ledger"]
        ),
        "follow_ids": all(f["equal"] for f in report["follow_ids"])
        and all(f.get("injected_initialized", True) for f in report["follow_ids"]),
    }
    print(json.dumps(checks, indent=2))
    assert all(checks.values()), checks
    print("PASS")


def _parent_map(state, parents):
    """Electron tracker ID -> parent ion ID, from the emission records."""
    for iteration, first_id, parent_ids in state.ledger["emitted_first_id"]:
        if iteration in parents:
            continue
        p = cupy.asnumpy(parent_ids)
        parents[iteration] = (first_id + np.arange(p.size, dtype=np.uint64), p)
    if not parents:
        return np.zeros(0, np.uint64), np.zeros(0, np.uint64)
    e_ids = np.concatenate([v[0] for v in parents.values()])
    p_ids = np.concatenate([v[1] for v in parents.values()])
    return e_ids, p_ids


def _cohort(state, id_range, parents):
    lo, hi = id_range
    ion = state.species["nitrogen"]
    elec = state.species["elec_from_N"]
    n, ne = int(ion.Ntot), int(elec.Ntot)
    ion_ids = cupy.asnumpy(ion.tracker.id[:n])
    in_cohort = (ion_ids >= lo) & (ion_ids < hi)
    w = cupy.asnumpy(ion.w[:n])[in_cohort]
    p5, p6, p7, pending = (cupy.asnumpy(a)[in_cohort] for a in state.ev.arrays())
    e_ids, p_ids = _parent_map(state, parents)
    cohort_electrons = e_ids[(p_ids >= lo) & (p_ids < hi)]
    live_e_ids = cupy.asnumpy(elec.tracker.id[:ne])
    live_mask = np.isin(live_e_ids, cohort_electrons)
    elec_w = cupy.asnumpy(elec.w[:ne])[live_mask].sum()
    ionized = (w * (5 * p5 + 6 * p6 + 7 * p7 - 5)).sum()
    return {
        "iteration": state.iteration,
        "ions": int(in_cohort.sum()),
        "ions_expected": int(hi - lo),
        "electrons": int(live_mask.sum()),
        "electrons_missing": int(cohort_electrons.size - live_mask.sum())
        + int(hi - lo - in_cohort.sum()),
        "max_population_error": float(np.abs(p5 + p6 + p7 - 1).max()),
        "ionized_weight": float(ionized),
        "relative_weight_error": float(
            abs(elec_w + pending.sum() - ionized) / max(ionized, 1e-300)
        ),
    }


def _ledger(state):
    ion = state.species["nitrogen"]
    elec = state.species["elec_from_N"]
    n, ne = int(ion.Ntot), int(elec.Ntot)
    led = state.ledger
    w = ion.w[:n]
    p5, p6, p7, pending = state.ev.arrays()
    live_ionized = float((w * (5 * p5 + 6 * p6 + 7 * p7 - 5)).sum())
    live_pending = float(pending.sum())
    live_elec = float(elec.w[:ne].sum())
    emitted = float(led["emitted"])
    lhs2 = emitted + live_pending + led["removed_ion_pending"]
    rhs2 = live_ionized + led["removed_ion_ionized"]
    return {
        "iteration": state.iteration,
        "emitted": emitted,
        "removed_elec_w": led["removed_elec_w"],
        "removed_ion_ionized": led["removed_ion_ionized"],
        "relative_emitted_error": abs(emitted - live_elec - led["removed_elec_w"])
        / max(emitted, 1e-300),
        "relative_balance_error": abs(lhs2 - rhs2) / max(rhs2, 1e-300),
    }


if __name__ == "__main__":
    main()
