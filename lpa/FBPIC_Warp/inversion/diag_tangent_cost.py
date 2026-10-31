"""Diagnostic: time per step of the primal stepper and of K = 1 tangents.

For each configuration, two primal runs and two tangent runs alternate;
each time excludes the first 20 steps (kernel compilation and warm-up).
Writes inversion/results/p5_tangent_cost.json.
"""

import json
import sys
import time
from pathlib import Path

import cupy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from builders import twin_builder  # noqa: E402
from verify_linear_wakefield import builder as l4w_builder  # noqa: E402

from fbpic.autodiff.driver import resolve_seeds, seed_tangents  # noqa: E402
from fbpic.autodiff.stepper import begin, step  # noqa: E402

WARM = 20


def timed(build, theta, seeds, n_steps, with_tangents):
    sim, species, profile = build(theta)
    resolved = resolve_seeds(sim, profile, theta, seeds) if with_tangents else None
    state = begin(sim, species)
    tangents = seed_tangents(state, resolved) if with_tangents else None
    for _ in range(WARM):
        step(state, tangents)
    cupy.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps - WARM):
        step(state, tangents)
    cupy.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - t0) / (n_steps - WARM)


cases = {
    "twin_xs": (twin_builder("twin_xs"), None, ("E_L",), 300),
    "twin_s": (twin_builder("twin_s"), None, ("E_L",), 1000),
    "l4w": (l4w_builder(), (0.01,), ("a0",), 1500),
}
out = {}
for name, (build, theta, seeds, n_steps) in cases.items():
    if theta is None:
        theta = np.asarray(build.config["theta"]["nominal"], float)
    times = {"primal": [], "tangent": []}
    for _ in range(2):
        times["primal"].append(timed(build, theta, seeds, n_steps, False))
        times["tangent"].append(timed(build, theta, seeds, n_steps, True))
    primal, tangent = min(times["primal"]), min(times["tangent"])
    out[name] = {
        "n_steps": n_steps,
        "primal_ms": 1e3 * primal,
        "tangent_K1_ms": 1e3 * tangent,
        "ratio": tangent / primal,
        "runs_ms": {k: [1e3 * v for v in vals] for k, vals in times.items()},
    }
    print(name, json.dumps(out[name]))
(HERE / "results" / "p5_tangent_cost.json").write_text(json.dumps(out, indent=1) + "\n")
