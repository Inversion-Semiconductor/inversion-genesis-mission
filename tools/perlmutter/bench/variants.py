"""Benchmark variants: the solver engine and the FBPIC constructor options of each."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Variant:
    """One solver configuration to time.

    Attributes:
        name: Variant label used in run directories and reports.
        engine: "numba" for upstream FBPIC or "warp" for FBPIC-Warp.
        simulation_kwargs: Extra ``fbpic.main.Simulation`` arguments. Upstream
            FBPIC rejects the Warp-only arguments, so numba keeps this empty.
        hankel_dtype: Expected dtype of the device Hankel buffers.
        deposit_tpb: Threads per block for linear deposition, set after setup.
            None keeps the solver default (16 on A100 for both solvers).
    """

    name: str
    engine: str
    simulation_kwargs: dict[str, Any] = field(default_factory=dict)
    hankel_dtype: str = "float64"
    deposit_tpb: int | None = None


# ID-based ionization draws: Philox keyed by (seed, stream, ion ID, iteration, level).
RNG_SEED = 0
RNG_STREAMS = {"He": 0, "N": 1}

VARIANTS = {
    variant.name: variant
    for variant in (
        Variant("numba", "numba"),
        Variant(
            "warp-fp64",
            "warp",
            {
                "use_cuda_graphs": False,
                "fuse_particle_kernels": True,
                "hankel_precision": "float64",
            },
            deposit_tpb=32,
        ),
        Variant(
            "warp-fp32",
            "warp",
            {
                "use_cuda_graphs": False,
                "fuse_particle_kernels": True,
                "hankel_precision": "float32",
            },
            hankel_dtype="float32",
            deposit_tpb=32,
        ),
    )
}

BASELINE = "numba"
