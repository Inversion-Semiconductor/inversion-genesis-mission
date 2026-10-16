# FBPIC-Warp: Agent Context

FBPIC-Warp is a fork of [FBPIC](https://github.com/fbpic/fbpic) with its GPU kernels ported to NVIDIA Warp.
[README.md](README.md) has the overview and environment setup.
[CONTRIBUTING.md](CONTRIBUTING.md) has the workflow and test commands.
This file records the rules that an agent cannot derive from the code.

## Rules

1. **Pin the GPU stack.** Use Warp 1.18.0 (CUDA 12 wheel) and CuPy 14.2 from conda-forge, as in `environment.yml`.
   Do not install CuPy with pip; a second CuPy build breaks the shared stream.
2. **Keep the import name `fbpic`.** The distribution name is `fbpic` with a `+warp.N` local version.
   An editable install satisfies `fbpic` requirements of sibling packages, such as `lpa/Simulation_FBPIC`.
3. **Give evidence for every GPU kernel change.** Report three results in the commit message:
   - the fused-versus-per-mode tests (`tests/test_fused_particle_operations.py`, `tests/test_gpu_end_to_end.py`);
   - a fixed-seed simulation compared with the previous commit, within run-to-run noise
     (about 2e-11 relative for fp64, about 1e-6 for the float32 Hankel transform, after 300 steps);
   - the time per step before and after, from interleaved runs.
4. **Do not expect bitwise reproducibility.** Deposition uses GPU atomics, so repeated runs differ by rounding.
   Particle IDs and ionization levels must match exactly.
5. **Run the GPU tests in this directory.** The repository-root `pytest.ini` does not collect them.
   Run each test file in its own process (see CONTRIBUTING.md).
6. **Use one CUDA stream.** Warp kernels, CuPy operations, and cuBLAS handles all use the runtime stream from `fbpic.utils.warp.get_runtime()`.
   Launch Warp kernels through `warp_kernel` or `launch_kernel`, so that CUDA graph sections can record them.
7. **Keep host copies out of the step.** Device-to-host transfers belong in diagnostics and tests only.
8. **Format with Black 25.1.0 and lint with Ruff 0.11.11.** These are the versions that the repository pre-commit hooks pin.

## Conventions that cause errors

- Warp 1.18 floors integer `//` but truncates integer `%`. Do not assume either convention in kernels.
- Warp has no arrays of arrays. Mode groups pass their grids as structs with fixed fields (`GroupFields`, `OutputBuffers`).
- `wp.set_module_options({"enable_backward": False})` also disables adjoints for `@wp.func` helpers.
  Keep it in forward-only modules. The autodiff branch controls it on purpose.
- Upstream tests patch the example scripts in `docs/source/example_input/` with regular expressions.
  Keep those patterns tolerant of whitespace.
