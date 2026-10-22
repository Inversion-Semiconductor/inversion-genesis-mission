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
    """

    name: str
    engine: str
    simulation_kwargs: dict[str, Any] = field(default_factory=dict)
    hankel_dtype: str = "float64"


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
        ),
    )
}

BASELINE = "numba"
