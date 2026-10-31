"""Phase P3 step 8: mean-charge closure (mobile against immobile EV ions).

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python verify_ev_closure.py twin_xs
    python verify_ev_closure.py twin_s

Runs EV twice, with the N ions mobile and immobile, and production stochastic
ionization with 16 Philox seeds. Reports the differences of Q, every m_k, and
the minimal loss in units of the SE of the 16-seed mean. Twin-XS runs at the
nominal theta, against a target made from its mobile EV beam. Twin-S runs at
theta* against the T0 target, and reuses the seeds 101 to 104 of Phase P1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from forward import run_ev, run_production
from identifiability import _run as p1_run
from make_target import load_target, target_from_beam
from observable import MOMENT_NAMES, beam_statistics, minimal_loss
from twin_config import load_config, theta_from_hat

RESULT_FILE = Path(__file__).resolve().parent / "results" / "p3_closure.json"
SEEDS = tuple(range(101, 117))


def main(size):
    config = load_config(size)
    sel = config["selection"]
    if size == "twin_s":
        hat = np.asarray(config["theta_star_hat"])
        target = load_target(size, "T0", config)
    else:
        hat = np.array([0.5, 0.5])
        target = None
    theta = theta_from_hat(config, hat)
    mobile = run_ev(theta, config)
    immobile = run_ev(theta, config, immobile=True)
    if target is None:
        target = target_from_beam(
            mobile["elec"], mobile["z_win"], config, theta, "XS", 0
        )

    def quantities(run):
        st = beam_statistics(
            run["elec"], run["z_win"], sel["u_c"], sel["delta_u"], target["Q_f"]
        )
        return np.concatenate(
            [[st["Q"] * 1e12], st["moments"], [minimal_loss(st, target)]]
        )

    seeds = []
    for s in SEEDS:
        if size == "twin_s":
            run = p1_run(f"star_s{s}", hat, config, rng_seed=s)
        else:
            run = run_production(theta, config, rng_seed=s)
        seeds.append(quantities(run))
    seeds = np.array(seeds)
    se = seeds.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    qm, qi = quantities(mobile), quantities(immobile)
    names = ["Q_pC", *MOMENT_NAMES, "loss"]
    rows = {
        n: {
            "mobile": float(a),
            "immobile": float(b),
            "seed_mean": float(m),
            "se": float(s),
            "diff_over_se": float((a - b) / s) if s > 0 else float("inf"),
        }
        for n, a, b, m, s in zip(names, qm, qi, seeds.mean(axis=0), se)
    }
    worst = max(abs(r["diff_over_se"]) for r in rows.values())
    out = {"theta_hat": hat.tolist(), "rows": rows, "max_abs_diff_over_se": worst}
    data = json.loads(RESULT_FILE.read_text()) if RESULT_FILE.exists() else {}
    data[size] = out
    RESULT_FILE.parent.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(data, indent=2) + "\n")
    for n, r in rows.items():
        print(
            f"{n:9s} mobile={r['mobile']:.5e} immobile={r['immobile']:.5e} "
            f"se={r['se']:.3e} diff/se={r['diff_over_se']:+.3f}"
        )
    print("max |diff| / SE =", worst, "->", "within 2 SE" if worst <= 2 else "ASK USER")


if __name__ == "__main__":
    main(sys.argv[1])
