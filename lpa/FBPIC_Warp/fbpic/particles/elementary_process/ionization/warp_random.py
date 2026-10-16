# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Implement the shared Philox4x32-10 protocol with Warp integer arithmetic."""

import warp as wp

from fbpic.utils.warp import warp_kernel

_UNIFORM_SCALE = wp.constant(wp.float32(2.0**-24))


@wp.func
def philox4x32_gpu(
    c0: wp.uint32,
    c1: wp.uint32,
    c2: wp.uint32,
    c3: wp.uint32,
    k0: wp.uint32,
    k1: wp.uint32,
):
    for round_index in range(10):
        product0 = wp.uint64(c0) * wp.uint64(0xD2511F53)
        product1 = wp.uint64(c2) * wp.uint64(0xCD9E8D57)
        next0 = wp.uint32(product1 >> wp.uint64(32)) ^ c1 ^ k0
        next1 = wp.uint32(product1)
        next2 = wp.uint32(product0 >> wp.uint64(32)) ^ c3 ^ k1
        next3 = wp.uint32(product0)
        c0, c1, c2, c3 = next0, next1, next2, next3
        if round_index < 9:
            k0 = wp.uint32(wp.uint64(k0) + wp.uint64(0x9E3779B9))
            k1 = wp.uint32(wp.uint64(k1) + wp.uint64(0xBB67AE85))
    return c0, c1, c2, c3


@warp_kernel
def generate_ionization_draws_gpu(
    ids: wp.array(dtype=wp.uint64),
    levels: wp.array(dtype=wp.uint64),
    iteration: wp.uint32,
    seed: wp.uint64,
    stream: wp.uint32,
    draws: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    if index < ids.shape[0]:
        word, _unused1, _unused2, _unused3 = philox4x32_gpu(
            wp.uint32(ids[index]),
            wp.uint32(ids[index] >> wp.uint64(32)),
            iteration,
            (stream << wp.uint32(8)) | wp.uint32(levels[index]),
            wp.uint32(seed),
            wp.uint32(seed >> wp.uint64(32)),
        )
        draws[index] = wp.float32(word >> wp.uint32(8)) * _UNIFORM_SCALE
