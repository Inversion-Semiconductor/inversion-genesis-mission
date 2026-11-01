"""Phase P7 step 2: K = 2 tangents in one pass against two K = 1 passes.

Twin-XS with EV, nominal theta, N_t = 10. Seeds: the E_L seed and a
synthetic seed (a fixed random direction on the initial interp fields, seed
0, scaled to the laser fields). Runs:

1. one `jvp` with K = 2 (E_L, synthetic);
2. one `jvp` with K = 1 for each seed;
3. a repeat of each K = 1 run, for the run-to-run difference d_rep of the
   deposit atomics (`CLAUDE.md` rule 4).

For every tangent array (spectral fields of every mode, particle and EV
arrays of every species, by tracker ID), the difference between row k of
the K = 2 run and the K = 1 run of seed k is taken relative to the largest
entry of that array. Pass: the largest difference is at most
max(10 d_rep, 1e-13), where d_rep is the largest repeat difference of the
same array set. This shows that the driver is consistent for any K; it does
not show that the derivative is correct (Section 6.4).

Usage (from lpa/FBPIC_Warp, CUDA_VISIBLE_DEVICES set):

    python inversion/verify_batched_tangents.py
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

SIZE = "twin_xs"
N_STEPS = 10


def synthetic_seed(sim, profile, theta):
    """Fixed random direction on the initial interp fields (seed 0)."""
    from fbpic.autodiff.laser import laser_initial_fields

    initial = laser_initial_fields(sim, profile)
    rng = np.random.default_rng(0)
    scale = max(float(np.abs(v).max()) for mode in initial for v in mode.values())
    out = []
    for mode in initial:
        out.append(
            {
                name: scale
                * 1e-3
                * (rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape))
                for name, v in mode.items()
            }
        )
    return out


def tangent_arrays(result) -> dict:
    """Host tangent arrays by name and row: fields per mode, particles by ID."""
    import cupy

    state, tangents = result.state, result.tangents
    out = {}
    for m, mode in enumerate(tangents.fields.spect):
        for name, array in mode.items():
            out[f"spect{m}.{name}"] = cupy.asnumpy(array)
    for species, arrays in tangents.particles.items():
        sp = state.species[species]
        order = cupy.argsort(sp.tracker.id[: int(sp.Ntot)])
        for comp, array in arrays.items():
            out[f"{species}.{comp}"] = cupy.asnumpy(array[:, order])
    return out


def max_difference(a: dict, row_a: int, b: dict, row_b: int) -> tuple[float, str]:
    worst, where = 0.0, ""
    for key in a:
        x, y = a[key][row_a], b[key][row_b]
        scale = max(float(np.abs(x).max(initial=0)), float(np.abs(y).max(initial=0)))
        if scale == 0:
            continue
        d = float(np.abs(x - y).max()) / scale
        if d > worst:
            worst, where = d, key
    return worst, where


def main():
    from fbpic.autodiff.driver import jvp

    config = load_config(SIZE)
    build = twin_builder(config, ionization="ev")
    theta = np.asarray(config["theta"]["nominal"], float)
    begin_kwargs = {
        "ionization": "ev",
        "ev_species": "nitrogen",
        "ev_target": "elec_from_N",
        "ev_options": {
            "epsilon": config["ev"]["epsilon"],
            "K_e": config["ev"]["K_e"],
        },
    }

    def run(seeds):
        result = jvp(theta, build, seeds, N_STEPS, begin_kwargs=begin_kwargs)
        births = sum(int(d.emitted_parent_ids.size) for d in result.decisions)
        return tangent_arrays(result), births

    both, births = run(("E_L", synthetic_seed))
    single = [run(("E_L",))[0], run((synthetic_seed,))[0]]
    repeat = [run(("E_L",))[0], run((synthetic_seed,))[0]]
    record = {
        "size": SIZE,
        "config_sha256": config_sha256(config),
        "n_steps": N_STEPS,
        "births": births,
        "seeds": ["E_L", "synthetic (random interp fields, seed 0, 1e-3 x max)"],
        "rows": [],
    }
    passed = True
    for k, name in enumerate(("E_L", "synthetic")):
        d_batch, where_batch = max_difference(both, k, single[k], 0)
        d_rep, where_rep = max_difference(single[k], 0, repeat[k], 0)
        tolerance = max(10 * d_rep, 1e-13)
        ok = d_batch <= tolerance
        passed &= ok
        record["rows"].append(
            {
                "seed": name,
                "K2_vs_K1": d_batch,
                "K2_vs_K1_worst": where_batch,
                "repeat_K1": d_rep,
                "repeat_K1_worst": where_rep,
                "tolerance": tolerance,
                "pass": bool(ok),
            }
        )
        print(
            f"{name}: K=2 vs K=1 {d_batch:.2e} ({where_batch}); repeat K=1 "
            f"{d_rep:.2e} ({where_rep}); tolerance {tolerance:.1e} -> "
            f"{'pass' if ok else 'FAIL'}"
        )
    record["pass"] = bool(passed)
    out = HERE / "results" / "p7_batched_tangents.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
