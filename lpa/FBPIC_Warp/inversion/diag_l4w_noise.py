"""Diagnostic: run-to-run noise of the L4w fields at a0 = 0.01 and 0.005.

Two primal runs per a0; the spread relative to max |E| is the floor of the
L4w self-consistency residual, which compares a tangent with 2 E_sim / a0
from one noisy trajectory. Writes inversion/results/p5_l4w_noise.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verify_linear_wakefield import builder, primal  # noqa: E402

build = builder()
out = {}
for a0 in (0.01, 0.005):
    runs = [primal(build, a0, 1500)[0] for _ in range(2)]
    out[str(a0)] = {
        name: float(np.max(np.abs(runs[0][name] - runs[1][name])))
        / float(np.max(np.abs(runs[0][name])))
        for name in ("Ez", "Er")
    }
    print(a0, out[str(a0)])
(HERE / "results" / "p5_l4w_noise.json").write_text(json.dumps(out, indent=1) + "\n")
