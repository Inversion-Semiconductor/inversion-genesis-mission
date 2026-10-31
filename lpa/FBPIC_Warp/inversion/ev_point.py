"""One loss evaluation: an EV run of a twin against a target.

    python ev_point.py <size> <target> <theta_hat_0> <theta_hat_1> <out.json>

Writes J, Q, the moments (formed with the charge floor Q_f of the target),
Q_f, the selected count, the birth count, and the wall time to out.json.
`fd_recovery.py` launches this script once per point. Caches of these runs
are keyed by `code_sha256()` (D30).
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


def code_sha256() -> str:
    """SHA-256 of the simulation code files."""
    return hashlib.sha256(b"".join(f.read_bytes() for f in CODE_FILES)).hexdigest()


def provenance(size, variant) -> dict:
    """Hashes of the config, the target file, and the EV code."""
    config = load_config(size)
    target = target_path(size, variant, config["num_steps"])
    return {
        "size": size,
        "target": variant,
        "config_sha256": config_sha256(config),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "code_sha256": code_sha256(),
        "code_scope": "simulation",
    }


def main(size, variant, t0, t1, out):
    config = load_config(size)
    target = load_target(size, variant, config)
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
        "Q_f": float(target["Q_f"]),
        "n_selected": stats["n_selected"],
        "births": run["births"],
        "wall_s": run["wall_s"],
        "provenance": provenance(size, variant),
    }
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
