# Changes from the FBPIC Baseline

The reference implementation is FBPIC commit `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.

This document records implemented changes and their reasons.

## Documentation

The port includes this change document and a validation report. These files record the agreed scope and completed work.

The [solver porting guide](../warp-porting-guide/README.md) connects the original decisions, implementation, performance findings, and future research directions.

## Environment

`environment.yml` records the Conda packages for the baseline and port.

The original port used Warp 1.17.0, CuPy 14.2.0, and CUDA 12.9 libraries.
The current environment uses Warp 1.18.0 with the CUDA 12 build.

The local environment selects NVIDIA CUDA Python bindings with `NUMBA_CUDA_USE_NVIDIA_BINDING=1`.

This setting avoids an incompatible driver function lookup in Numba 0.61.2. The baseline source remains unchanged.

## Shared GPU runtime

`fbpic/utils/warp.py` provides typed Warp kernels and views of CuPy arrays.

CuPy retains array ownership. Complex128 elements appear as two float64 values in Warp.

The views retain each current array allocation. This preserves correct references when particle sorting swaps buffers.

Existing kernel names and `[blocks, threads]` launch calls remain available.

CuPy, Warp, cuFFT, and cuBLAS use one explicit CUDA stream. CUDA graph checks keep cuBLAS outside capture because CuPy rejects captured calls.

Backward kernel generation is disabled for milestone 1. Gradient implementation and validation remain deferred.

## Field and particle kernels

Warp replaces Numba CUDA kernels in field updates, transform helpers, particle pushing, gathering, deposition, and sorting.

The port retains the baseline equations, precision, mode conventions, and deposition stencils.

Core particle kernels use multiplication for squares and cubes instead of floating-exponent `wp.pow` calls.
These changes cover deposition, gathering, the Vay momentum push, and particle cell indexing.
Typed `square` and `cube` helpers retain float64 arithmetic.
Power expressions in optional physics remain unchanged.

Seven focused GPU tests passed under Warp 1.18 before publication of the arithmetic changes.
These checks cover linear and cubic deposition, eager and graph plasma waves, and stationary laser-antenna deposition.
The full test suite was not rerun for this publication.

Comoving GPU coefficients use complex128 storage when the selected velocity is zero. This provides consistent Warp argument types without changing coefficient values.

Direct GPU checks passed against baseline formulas. The checks included axis behavior, periodic wrapping, irregular transform dimensions, and integer particle identifiers.

The coupled solver advanced three GPU steps with 12,288 particles. This setup check does not establish full solver parity.

## Boundaries and diagnostics

Warp replaces 20 GPU kernels in boundary buffers, damping, moving windows, particle buffers, and boosted diagnostic extraction.

Host functions and CPU functions remain unchanged in these modules.

All 170 direct comparison cases passed in eager execution and repeated graph replay.

## CUDA graph execution

`Simulation` accepts `use_cuda_graphs=True` by default. Set this option to `False` for eager execution.

Five sections capture gathering, particle pushes, current correction, and spectral field updates.

Kernel preparation does not execute physical updates. Diagnostics, MPI, sorting, external fields, and dynamic particle allocation retain their original execution positions.

Graph keys include scalar values, array addresses, shapes, strides, dtypes, and launch dimensions.

Changed arguments trigger capture again. This includes moving geometry and ballistic injection parameters.

Frequent scalar changes can reduce graph reuse. Persistent parameter buffers remain an optimization option.

Application validation is complete. The first 16 existing and extended core tests passed with eager GPU execution.

The application group passed 41 tests with default graph execution. Coverage includes lasers, moving windows, boosted diagnostics, and checkpoint/restart.

The consolidated suite passed 65 tests. The final external-field rerun passed 24 tests after localized callback fixes.

## Optional particle physics

Warp replaces GPU kernels for spin tracking, ionization, Compton scattering, synchrotron radiation, particle creation, and identifier generation.

The CPU physics helpers retain the baseline implementation.

Warp uses explicitly updated random states for Compton and synchrotron sampling. Double precision sampling preserves the required distributions, but individual events can differ.

The seven existing optional physics tests passed with eager execution. The same cases passed with graph replay across the recorded runs.

The spin-ionization test selects new electrons by identifier. This preserves the existing assertions when GPU sorting changes particle order.

## External fields and PICMI

External field callbacks use typed Warp source translation for GPU execution. Their CPU implementation continues to use Numba.

GPU callbacks compile on first GPU use. CPU application does not compile GPU kernels.

Missing scalar math operations use native double precision calls. These include accurate `expm1`, `log1p`, and `hypot` implementations.

Constant lookup tables preserve their numeric types and bit patterns.

The callback adapter preserves baseline numeric promotion and signed division behavior. Regression checks cover mixed types, powers, conditional expressions, and lookup tables.

PICMI-generated callbacks retain their source as metadata. This lets Warp compile the existing callback expressions.

Direct tests passed for scalar callbacks, closures, boosted fields, and generated expressions.

## Validation changes

Existing tests now check `Jy`, GPU charge deposition, graph execution, and transform roundoff.

The new transform test derives from `tests/unautomated/test_cuda_transform.py`. The original manual script remains available.

Serial PML and space-charge cases reuse existing physical checks. These cases defer execution with multiple MPI ranks.

The environment sets `OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0`, as the baseline CI does. This preserves compatibility with existing diagnostic records.

Read the [validation report](milestone-1-validation.md) for commands, results, and remaining validation limits.

## Profiling branch

The persistent `snidhan/dev-profiling` branch contains performance scripts, benchmark plans, and measurement reports.
The branch inherits solver changes from `main`. Generated measurements remain outside version control.
The [performance chapter](../warp-porting-guide/02-performance.md) retains the measured findings and their interpretation limits.
