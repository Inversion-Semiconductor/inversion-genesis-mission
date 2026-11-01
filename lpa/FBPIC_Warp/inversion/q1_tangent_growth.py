"""Q1 companion: per-species growth of the E_L tangent at 1x and 4x particles.

Usage (from lpa/FBPIC_Warp, CUDA_VISIBLE_DEVICES set):

    python inversion/q1_tangent_growth.py 1
    python inversion/q1_tangent_growth.py 4

Runs `driver.jvp` with EV at theta_hat_E = 0.588 (z_f at z_f*) and the
particle multiplier m of `q1_roughness.py`. Every EVERY steps, a callback
records per species the median, 90th percentile, and maximum of the
momentum tangent |du/dtheta_hat_E| and of the position tangent
|dx/dtheta_hat_E|. If the growth rate after trapping falls with m, the
chaos is of N-body type and more particles help AD; if it does not, the
growth is collective (mean-field) or physical.
Writes inversion/results/q1_tangent_growth_m<m>.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from q1_roughness import X0, _config  # noqa: E402
from twin_config import theta_from_hat  # noqa: E402

EVERY = 50


def main(m):
    import cupy

    from builders import twin_builder

    from fbpic.autodiff.driver import jvp

    config = _config(m)
    lo, hi = np.asarray(config["theta"]["lo"]), np.asarray(config["theta"]["hi"])
    scale = hi[0] - lo[0]
    star = np.asarray(config["theta_star_hat"], float)
    theta = theta_from_hat(config, np.array([X0, star[1]]))
    rows = []

    def stats(a):
        a = cupy.sort(a)
        n = a.size
        return {
            "median": float(a[n // 2]) * scale,
            "p90": float(a[int(0.9 * (n - 1))]) * scale,
            "max": float(a[-1]) * scale,
        }

    def callback(state, tangents, decisions):
        it = int(state.sim.iteration)
        if it % EVERY:
            return
        row = {"step": it}
        for name, sp in state.species.items():
            n = int(sp.Ntot)
            t = tangents.particles.get(name)
            if n == 0 or t is None or "ux" not in t:
                continue
            du = cupy.sqrt(sum(t[c][0][:n] ** 2 for c in ("ux", "uy", "uz")))
            dx = cupy.sqrt(sum(t[c][0][:n] ** 2 for c in ("x", "y", "z")))
            row[name] = {"n": n, "du": stats(du), "dx": stats(dx)}
        rows.append(row)

    t0 = time.perf_counter()
    jvp(
        theta,
        twin_builder(config, ionization="ev"),
        ("E_L",),
        config["num_steps"],
        begin_kwargs={
            "ionization": "ev",
            "ev_species": "nitrogen",
            "ev_target": "elec_from_N",
            "ev_options": {
                "epsilon": config["ev"]["epsilon"],
                "K_e": config["ev"]["K_e"],
            },
        },
        callback=callback,
    )
    out = {"m": m, "x": X0, "every": EVERY, "wall_s": time.perf_counter() - t0}
    out["rows"] = rows
    path = HERE / "results" / f"q1_tangent_growth_m{m}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path} ({out['wall_s']:.0f} s)")


if __name__ == "__main__":
    main(int(sys.argv[1]))
