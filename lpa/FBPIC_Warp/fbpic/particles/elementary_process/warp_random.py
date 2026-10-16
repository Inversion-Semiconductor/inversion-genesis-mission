# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Generate double precision random samples from per-batch Warp states."""

import warp as wp
from typing import TYPE_CHECKING

from fbpic.utils.warp import warp_kernel

if TYPE_CHECKING:
    import cupy


# Threads per block for the one-time state initialization (not performance-critical).
RNG_INIT_BLOCK_DIM = 128


@warp_kernel
def _initialize_states(states: wp.array(dtype=wp.uint32), seed: int):
    index = wp.tid()
    if index < states.shape[0]:
        states[index] = wp.rand_init(seed, index)


def create_random_states(count: int, seed: int) -> "cupy.ndarray":
    """Allocate and initialize independent CuPy-owned states on the current GPU.

    Args:
        count: Number of batches that need independent random samples.
        seed: Seed used to initialize the batch states.

    Returns:
        A mutable uint32 CuPy array with one state per batch.
    """
    import cupy

    states = cupy.empty(count, dtype=cupy.uint32)
    blocks = (count + RNG_INIT_BLOCK_DIM - 1) // RNG_INIT_BLOCK_DIM
    _initialize_states[blocks, RNG_INIT_BLOCK_DIM](states, int(seed))
    return states


@wp.func
def random_uniform(states: wp.array(dtype=wp.uint32), index: int):
    state = states[index]
    # Combine 27 and 26 random bits to retain double precision sampling.
    upper = wp.randu(state) >> wp.uint32(5)
    lower = wp.randu(state) >> wp.uint32(6)
    states[index] = state
    return (
        wp.float64(upper) * wp.float64(67108864.0) + wp.float64(lower)
    ) * wp.float64(1.0 / 9007199254740992.0)


@wp.func
def random_normal(states: wp.array(dtype=wp.uint32), index: int):
    radial = wp.float64(1.0) - random_uniform(states, index)
    angular = random_uniform(states, index)
    return wp.sqrt(-wp.float64(2.0) * wp.log(radial)) * wp.cos(
        wp.float64(6.283185307179586) * angular
    )
