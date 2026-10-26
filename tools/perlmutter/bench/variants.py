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
        deposit_lanes: Lanes per cell in fused deposition, per field type, for
            example ``{"J": 4}``; set after setup. None keeps one lane per cell.
            Needs an FBPIC-Warp with ``Particles.deposit_lanes``.
        deposit_tpb_single_mode: Threads per block for a fused deposition group
            of one mode (the last current group for five modes); set after
            setup. None uses ``deposit_tpb``.
    """

    name: str
    engine: str
    simulation_kwargs: dict[str, Any] = field(default_factory=dict)
    hankel_dtype: str = "float64"
    deposit_tpb: int | None = None
    deposit_lanes: dict[str, int] | None = None
    deposit_tpb_single_mode: int | None = None


# Graphs off, fused particle kernels, FP64 Hankel: the warp-fp64 options.
WARP_FP64 = {
    "use_cuda_graphs": False,
    "fuse_particle_kernels": True,
    "hankel_precision": "float64",
}

# ID-based ionization draws: Philox keyed by (seed, stream, ion ID, iteration, level).
RNG_SEED = 0
RNG_STREAMS = {"He": 0, "N": 1}

VARIANTS = {
    variant.name: variant
    for variant in (
        Variant("numba", "numba"),
        Variant("warp-fp64", "warp", dict(WARP_FP64), deposit_tpb=32),
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
        # Control for the deposition launch configuration: warp-fp64 at the
        # upstream A100 block size, to compare 16 and 32 threads per block.
        Variant("warp-fp64-tpb16", "warp", dict(WARP_FP64), deposit_tpb=16),
        # Deposition experiments on warp-fp64. Each changes one setting:
        # several lanes per cell for current (jg*) or charge (rhog*), or 16
        # threads per block for the one-mode current group only (m1tpb16).
        Variant(
            "warp-fp64-jg2",
            "warp",
            dict(WARP_FP64),
            deposit_tpb=32,
            deposit_lanes={"J": 2},
        ),
        Variant(
            "warp-fp64-jg4",
            "warp",
            dict(WARP_FP64),
            deposit_tpb=32,
            deposit_lanes={"J": 4},
        ),
        Variant(
            "warp-fp64-jg8",
            "warp",
            dict(WARP_FP64),
            deposit_tpb=32,
            deposit_lanes={"J": 8},
        ),
        Variant(
            "warp-fp64-rhog4",
            "warp",
            dict(WARP_FP64),
            deposit_tpb=32,
            deposit_lanes={"rho": 4},
        ),
        # Best measured deposition settings on A100 (2026-10-23 kernel runs).
        Variant(
            "warp-fp64-jg4-rhog4",
            "warp",
            dict(WARP_FP64),
            deposit_tpb=32,
            deposit_lanes={"J": 4, "rho": 4},
        ),
        Variant(
            "warp-fp64-m1tpb16",
            "warp",
            dict(WARP_FP64),
            deposit_tpb=32,
            deposit_tpb_single_mode=16,
        ),
    )
}

BASELINE = "numba"
