# Milestone 1 Validation

## Scope

The reference is FBPIC commit `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.
The baseline working tree remains unchanged.

Validation covers forward execution on GPU 0 and the retained CPU path.
Gradient checks and execution with multiple GPU ranks remain deferred.

The consolidated single-GPU run passed 65 tests. Ten tests requiring MPI execution were excluded.

After localized callback fixes, all 24 external-field tests passed. Eleven independent callback arithmetic cases matched baseline Numba exactly.

Milestone 1 is complete. The baseline remains unchanged.
The port was uncommitted when these checks completed.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX 6000 Ada Generation. |
| Driver | 595.91.07. |
| Conda environment | `fbpic-warp`. |
| Python | 3.11. |
| Warp | 1.17.0. |
| CuPy and CUDA libraries | CuPy 14.2.0 and CUDA 12.9. |
| CPU and baseline compiler | Numba 0.61.2. |
| PICMI standard | 0.34.0. |

`environment.yml` records the package configuration.
Baseline and port checks use separate processes with explicit checkout paths.

The environment sets `NUMBA_CUDA_USE_NVIDIA_BINDING=1` for baseline compatibility with the installed driver.
It also sets `OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0`, matching baseline CI.

## Completed checks

| Check | Result |
| --- | --- |
| Consolidated single-GPU suite | Sixty-five tests passed in 238.39 seconds. Exit status was zero. |
| Final external-field rerun | Twenty-four tests passed in 11.74 seconds. Exit status was zero. |
| GPU setup | Warp, CuPy views, cuFFT, cuBLAS, and five shared-stream graph replays passed. |
| Baseline setup | Three GPU steps passed with 12,288 particles. |
| Initial eager core suite | Sixteen tests passed. |
| Application and core suite | Forty-one tests passed in 129.19 seconds. Six MPI tests were deselected. |
| Optional physics | Seven cases passed with eager execution. The cases with graph replay also passed. |
| Serial PML and space charge | Eight cases passed, including both graph settings and PML restart. |
| CPU execution | Three tests passed with CUDA devices hidden and four CPU threads. |
| PICMI | Four GPU cases passed with lab/boosted frames and both graph settings. |
| Additional solver combinations | Six configurations matched baseline results with graphs enabled and disabled. |

Existing physical assertions and tolerances remain unchanged.
New transform checks compare CPU and GPU results with a `1e-12` relative-to-maximum bound.

The solver comparison covered cross-deposition, both particle shapes, finite order, and zero/nonzero comoving velocities.
Initial arrays, particle identifiers, times, and domain bounds matched exactly.

Symmetry-zero field components contain cancellation noise.
The comparison therefore normalized errors with the corresponding physical vector magnitude.
The maximum normalized difference was `6.03e-12`, below the temporary probe's `1e-10` bound.

Direct kernel checks also covered modal phases, axis behavior, irregular grid sizes, particle sorting, stochastic physics, and exact uint64 identifiers.

SciPy emitted one integration warning during unchanged synchrotron table construction. The synchrotron physical assertions passed.

## Test adaptations

The deposition comparison now includes `Jy`.
Uniform-density and charged-cylinder tests now exercise the GPU path.
Plasma-wave tests check graph capture and reuse with both particle shapes.

The spin-ionization test selects new particles by identifier because GPU sorting changes array order.
The ionization test fixes random seeds to prevent dependence on earlier tests.
These changes preserve the existing physical checks.

Serial PML and space-charge tests reuse the existing analytical, symmetry, field, and restart checks.
The original MPI tests remain available.

## Evidence

Test logs are stored under `/tmp` on the development machine.
The main logs are `/tmp/fbpic-warp-applications.log`, `/tmp/fbpic-optional-eager.log`, and `/tmp/fbpic-picmi-offline-gpu.log`.

The consolidated log is `/tmp/fbpic-warp-final-suite.log`.

Final callback logs are `/tmp/fbpic-warp-external-final.log` and `/tmp/fbpic-callback-arithmetic-final.log`.

Solver comparison results are in `/tmp/fbpic-warp-solver-comparison-normalized.log`.
The comparison snapshots are in `/tmp/fbpic-warp-solver-combinations/`.

Temporary probes do not replace the repository's test suite.
No gradient or performance claim follows from these forward checks.

## Reproduce the consolidated run

1. Activate the environment described in the repository README.
2. Run the following command from the port repository.

```bash
CUDA_VISIBLE_DEVICES=0 NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH="$PWD" MPLBACKEND=Agg python -m pytest -q tests \
    --ignore=tests/unautomated -k 'not twoproc' \
    --deselect=tests/test_pml.py::test_laser_periodic \
    --deselect=tests/test_pml.py::test_laser_galilean \
    --deselect=tests/test_space_charge.py::test_bunch_from_file \
    --deselect=tests/test_space_charge.py::test_bunch_gaussian
```

The current test collection includes additional callback cases added after the recorded consolidated run.
The targeted external-field rerun covers those additions.
