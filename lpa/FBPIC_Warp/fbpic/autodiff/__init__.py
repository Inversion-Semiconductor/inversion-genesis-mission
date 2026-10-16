# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Opt-in FP64 operator derivatives with explicit state versions.

These operators do not make Simulation.step differentiable. Production forward
execution retains its existing allocation and kernel dispatch behavior.
"""
