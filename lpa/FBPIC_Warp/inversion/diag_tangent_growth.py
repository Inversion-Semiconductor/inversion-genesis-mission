"""Diagnostic: growth of the E_L tangent norms along a twin run.

Usage: python inversion/diag_tangent_growth.py <size> <n_steps> [every]
Writes inversion/results/p5_tangent_growth_<size>.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from builders import twin_builder  # noqa: E402

from fbpic.autodiff.driver import jvp  # noqa: E402

size, n_steps = sys.argv[1], int(sys.argv[2])
every = int(sys.argv[3]) if len(sys.argv) > 3 else 25
build = twin_builder(size)
theta = np.asarray(build.config["theta"]["nominal"], float)
result = jvp(theta, build, ("E_L",), n_steps, norm_every=every)
info = {k: v for k, v in result.info.items() if k != "tangent_norms"}
rows = [{"step": e["step"], **e["norms"][0]} for e in result.info["tangent_norms"]]
out = {"size": size, "theta": theta.tolist(), "info": info, "norms": rows}
(HERE / "results" / f"p5_tangent_growth_{size}.json").write_text(
    json.dumps(out, indent=1) + "\n"
)
print(json.dumps(info))
for r in rows:
    print(
        r["step"],
        f"ux {r['particles.ux']:.3e} x {r['particles.x']:.3e} Ez {r['spect.Ez']:.3e}",
    )
