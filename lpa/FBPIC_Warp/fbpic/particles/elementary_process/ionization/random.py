# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Generate float32 ionization draws from persistent particle IDs.

Philox4x32-10 uses the Random123 constants and round schedule:
https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h
The counter contains ID low/high words, the timestep, and stream/level bits.
The final word packs a 24-bit species stream above an 8-bit ionization level.
The key contains the low/high words of the 64-bit seed.
"""

import numba
import numpy as np

from fbpic.utils.cuda import cuda_installed
from fbpic.utils.threading import njit_parallel, prange


def _philox4x32(c0, c1, c2, c3, k0, k1):
    c0, c1, c2, c3 = np.uint32(c0), np.uint32(c1), np.uint32(c2), np.uint32(c3)
    k0, k1 = np.uint32(k0), np.uint32(k1)
    for round_index in range(10):
        product0 = np.uint64(c0) * np.uint64(0xD2511F53)
        product1 = np.uint64(c2) * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            np.uint32(product1 >> np.uint64(32)) ^ c1 ^ k0,
            np.uint32(product1),
            np.uint32(product0 >> np.uint64(32)) ^ c3 ^ k1,
            np.uint32(product0),
        )
        if round_index < 9:
            k0 = np.uint32(np.uint64(k0) + np.uint64(0x9E3779B9))
            k1 = np.uint32(np.uint64(k1) + np.uint64(0xBB67AE85))
    return c0, c1, c2, c3


philox4x32 = numba.njit(inline="always")(_philox4x32)


@njit_parallel
def generate_ionization_draws_cpu(
    ids: np.ndarray,
    levels: np.ndarray,
    iteration: np.uint32,
    seed: np.uint64,
    stream: np.uint32,
    draws: np.ndarray,
) -> None:
    """Fill a caller-owned float32 array on the CPU.

    Args:
        ids: Persistent uint64 particle IDs, with shape ``(N,)``.
        levels: Current uint64 ionization levels, with shape ``(N,)``.
        iteration: Simulation timestep below ``2**32``.
        seed: Unsigned 64-bit seed.
        stream: Species stream below ``2**24``.
        draws: Mutable float32 output with shape ``(N,)``.
    """
    for index in prange(ids.size):
        word, _, _, _ = philox4x32(
            np.uint32(ids[index]),
            np.uint32(ids[index] >> np.uint64(32)),
            np.uint32(iteration),
            (np.uint32(stream) << np.uint32(8)) | np.uint32(levels[index]),
            np.uint32(seed),
            np.uint32(seed >> np.uint64(32)),
        )
        # The top 24 bits give exactly representable float32 values in [0, 1).
        draws[index] = np.float32(word >> np.uint32(8)) * np.float32(2.0**-24)


if cuda_installed:
    # Re-export the GPU generators through the same RNG module as CPU callers.
    from .warp_random import generate_ionization_draws_gpu, philox4x32_gpu  # noqa: F401
