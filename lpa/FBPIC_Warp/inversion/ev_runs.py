"""Phase P3 steps 5 and 6: EV forward runs, birth counts, and memory.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU, one stage per process so
that the mempool high-water marks belong to one run):

    python ev_runs.py twin_s nominal
    python ev_runs.py twin_s corner00
    python ev_runs.py twin_s corner11
    python ev_runs.py twin_xs immediate
    python ev_runs.py twin_s_reduced immediate

Step 5 runs Twin-S with EV at the nominal theta and at two opposite corners.
Step 6 runs the immediate-emission reference (epsilon = 0) on Twin-XS and on
a reduced Twin-S in which the nitrogen occupies a short region. Results are
appended to `results/p3_ev_runs.json`.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

from forward import run_ev
from observable import beam_statistics
from twin_config import load_config, theta_from_hat

RESULT_FILE = Path(__file__).resolve().parent / "results" / "p3_ev_runs.json"
POINTS = {"nominal": (0.5, 0.5), "corner00": (0.0, 0.0), "corner11": (1.0, 1.0)}
# Reduced Twin-S: the nitrogen occupies [z_c - 2 um, z_c + 2 um] only.
REDUCED_HALF_WIDTH = 2e-6


def reduced_config():
    config = copy.deepcopy(load_config("twin_s"))
    zc = config["plasma"]["z_c"]
    config["plasma"]["nitrogen_z_range"] = [
        zc - REDUCED_HALF_WIDTH,
        zc + REDUCED_HALF_WIDTH,
    ]
    config["name"] = "twin_s_reduced"
    return config


def main(size, point):
    config = reduced_config() if size == "twin_s_reduced" else load_config(size)
    epsilon = 0.0 if point == "immediate" else None
    hat = POINTS.get(point, (0.5, 0.5))
    result = run_ev(theta_from_hat(config, hat), config, epsilon=epsilon)
    sel = config["selection"]
    stats = beam_statistics(
        result["elec"], result["z_win"], sel["u_c"], sel["delta_u"], 0.0
    )
    entry = {
        "size": size,
        "point": point,
        "theta_hat": list(hat),
        "epsilon": config["ev"]["epsilon"] if epsilon is None else epsilon,
        "births": result["births"],
        "counts": result["counts"],
        "emitted_weight": result["emitted_weight"],
        "Q_pC": stats["Q"] * 1e12,
        "n_selected": stats["n_selected"],
        "wall_s": result["wall_s"],
        "memory": result["memory"],
    }
    print(json.dumps(entry, indent=2))
    data = json.loads(RESULT_FILE.read_text()) if RESULT_FILE.exists() else {}
    data[f"{size}:{point}"] = entry
    RESULT_FILE.parent.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(data, indent=2) + "\n")
    np.savez(
        Path(__file__).resolve().parent / "runs" / f"p3_{size}_{point}.npz",
        **result["elec"],
        z_win=result["z_win"],
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
