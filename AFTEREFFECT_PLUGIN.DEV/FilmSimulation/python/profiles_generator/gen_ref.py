#!/usr/bin/env python3
"""Dump Algo 02 reference vectors from the Python model for the C++ diff test.

Format (little endian):
    uint32 count
    float64 stops
    float64 src[count]
    float64 expected[count]

Reused verbatim for every later stage -- only the transform changes.
"""
import struct, sys, numpy as np

rng = np.random.default_rng(20260729)
N = 4096
# Deliberately nasty inputs: the pipeline runs UNCLAMPED, so the kernel must
# survive negatives, zero, denormals, huge highlights and non-finite guards.
src = np.concatenate([
    np.array([0.0, -0.0, 1.0, 0.18, 65504.0, 1e-300, -1e-300, -1.0, -12.5],
             dtype=np.float64),
    (0.18 * 2.0 ** rng.uniform(-16.0, 16.0, N - 9)).astype(np.float64),
])
assert src.size == N

for stops in (0.0, +1.0, -1.0, +3.5, -7.25, +0.3333333333333333):
    exp = src * (2.0 ** stops)          # the reference transform
    with open(f"ref_algo_02_{stops:+.6f}.bin", "wb") as f:
        f.write(struct.pack("<I", N))
        f.write(struct.pack("<d", stops))
        f.write(src.astype("<f8").tobytes())
        f.write(exp.astype("<f8").tobytes())
    print(f"wrote ref_algo_02_{stops:+.6f}.bin  "
          f"range {exp.min():.6e} .. {exp.max():.6e}")
