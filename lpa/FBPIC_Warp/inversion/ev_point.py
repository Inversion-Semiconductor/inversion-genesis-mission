"""One loss evaluation: an EV run of Twin-S against the T0 target.

    python ev_point.py <theta_hat_0> <theta_hat_1> <out.json>

Writes J, Q, the moments, the selected count, the birth count, and the wall
time to out.json. `fd_recovery.py` launches this script once per point.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from forward import run_ev
from make_target import load_target, target_path
from observable import beam_statistics, minimal_loss
from twin_config import config_sha256, load_config, theta_from_hat

SIZE, VARIANT = "twin_s", "T0"
HERE = Path(__file__).resolve().parent
# The simulation code only: the loss is recomputed from the stored Q and
# moments, so loss changes do not invalidate cached runs. `beam_statistics`
# (raw moments) must stay unchanged.
CODE_FILES = (
    HERE / "forward.py",
    HERE / "twin_config.py",
    HERE.parent / "fbpic" / "autodiff" / "ionization.py",
    HERE.parent / "fbpic" / "autodiff" / "stepper.py",
    HERE.parent / "fbpic" / "autodiff" / "particles.py",
)


def provenance() -> dict:
    """Hashes of the config, the T0 target file, and the EV code."""
    config = load_config(SIZE)
    target = target_path(SIZE, VARIANT, config["num_steps"])
    code = hashlib.sha256(b"".join(f.read_bytes() for f in CODE_FILES)).hexdigest()
    return {
        "config_sha256": config_sha256(config),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "code_sha256": code,
        "code_scope": "simulation",
    }


def main(t0, t1, out):
    config = load_config(SIZE)
    target = load_target(SIZE, VARIANT, config)
    theta_hat = np.array([t0, t1])
    run = run_ev(theta_from_hat(config, theta_hat), config)
    sel = config["selection"]
    stats = beam_statistics(
        run["elec"], run["z_win"], sel["u_c"], sel["delta_u"], target["Q_f"]
    )
    J = minimal_loss(stats, target)
    result = {
        "theta_hat": theta_hat.tolist(),
        "J": J,
        "Q_pC": stats["Q"] * 1e12,
        "moments": stats["moments"].tolist(),
        "n_selected": stats["n_selected"],
        "births": run["births"],
        "wall_s": run["wall_s"],
        "provenance": provenance(),
    }
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
