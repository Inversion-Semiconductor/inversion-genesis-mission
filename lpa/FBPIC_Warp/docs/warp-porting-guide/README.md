# Port a Scientific Solver to Warp

This guide turns the FBPIC port into a reusable engineering workflow.
It connects decisions, source changes, validation, performance measurements, and future research.
The intended readers are solver developers and agents working under their direction.

The original `main` reference contains the forward port from commit `681c892` and the user's multiplication replacements in core particle kernels.
The environment specification pins the CUDA 12 build of Warp 1.18.
The original performance results describe development measurements recorded on 2026-10-06.
The FP32 Hankel and ionization chapter records subsequent work on October 7 and 8, 2026.
The persistent `snidhan/dev-profiling` branch contains performance scripts, benchmark plans, and detailed measurement reports.
The original benchmark paths refer to that branch.
The FP32 Hankel experiment and its simulation comparisons reside on the FP32 Hankel experiment branch.
The fusion experiments followed on October 9 and 10, 2026.
The complete Inversion comparison finished on October 12, 2026, in the Inversion workload experiment branch.
Production integration is committed as `edc52c4`, and local `main` now includes the selected fusion changes and optional FP32 Hankel.
The expanded guide records the source integration and the preceding experiments.
Generated results remain local artifacts excluded from Git.

The current implementation provides a validated forward GPU solver.
Using Warp does not automatically make the complete simulation differentiable.
Selective mixed precision includes optional FP32 Hankel GEMM and one complete, 17,720-step Inversion comparison.
Other precision changes and additional physical configurations remain separate research work.
Selective precision aims to use FP32 for suitable operations while retaining FP64 elsewhere and preserving complete simulation accuracy.

## Read the guide

| Chapter | What it provides |
| --- | --- |
| [Forward port](01-forward-port.md) | Establish the reference, preserve structure, integrate libraries, manage memory, and validate the coupled solver. |
| [Performance](02-performance.md) | Compare complete simulations, isolate kernels, inspect arithmetic, and verify improvements. |
| [Autodiff and mixed precision](03-autodiff-and-precision.md) | Define future derivative contracts and test selected FP32 operations against FP64 simulation accuracy. |
| [Evidence index](04-evidence.md) | Distinguish retained results from historical reports and proposals. |
| [FP32 Hankel and ionization RNG](05-hankel-fp32-and-ionization-rng.md) | Follow the GEMM experiment, sorting-dependent draw diagnosis, ID-based fix, and production rerun. |
| [Threads, shared memory, and fusion](06-threads-shared-memory-and-fusion.md) | Compare selected optimizations with rejected shared-memory, two-pass, and additional fusion experiments. |
| [Complete Inversion comparison](07-inversion-full-workload.md) | Reproduce kernel timings, complete-run speedups, field errors, and the precise measured configuration. |
| [Individual kernel gradients](08-individual-kernel-gradients.md) | Define the FP64 derivative contract and verification sequence before composing a differentiable simulation. |

## Apply the workflow to another solver

1. Freeze an executable baseline and define physical acceptance criteria.
2. Port the forward GPU path without changing precision or numerical algorithms.
3. Measure representative solver states and identify costly kernels through comparable captures.
4. Test arithmetic, launch configuration, fusion, and selective precision against accuracy and complete-workload timing.
5. Validate individual operator derivatives before composing a differentiable simulation.

Each stage needs its own evidence.
A passing forward test does not validate an adjoint.
A faster kernel does not establish a faster simulation.
A small local rounding error does not establish acceptable accumulated error.

Keep precision experiments and arithmetic optimizations separate when measuring their effects.
This makes failures and improvements easier to attribute.
Preserve the baseline, test inputs, source hashes, commands, and measurements throughout the work.
Inspect PTX or SASS when timing and source inspection do not explain a costly kernel.
Create separate worktrees for independent experiments, and preserve unsuccessful experiments with their rejection evidence.

The historical development order differs from this reusable sequence.
The FP32 Hankel and RNG experiments preceded the final deposition and gather fusion experiments.
Each chapter preserves its measured source, duration, and reference instead of combining results from different stages.

## What FBPIC demonstrated

| Area | Evidence and current limit |
| --- | --- |
| Forward correctness | The original milestone passed 65 consolidated tests and a later 24-test callback rerun under Warp 1.17. |
| Library integration | CuPy-owned storage, Warp kernels, cuFFT, and cuBLAS share one CUDA stream. |
| Original performance workload | The Warp 1.18 workload measured 12.486 ms/step eager and 12.372 ms/step with graphs, versus Numba's 14.658. |
| Autodiff | Backward kernel generation is disabled. Simulation gradients remain unimplemented and unvalidated. |
| Selective precision | Optional FP32 Hankel GEMM achieved `2.561x` complete-run speedup, with maximum field relative L2 error `0.2701%`. |
| Optimized FP64 configuration | The selected Warp configuration achieved `2.208x` complete-run speedup while retaining FP64 arithmetic. |
| Integration verification | 101 GPU tests and 39 CPU tests passed; 14 CPU cases were skipped. |
| MPI | MPI interfaces remain present. Parallel execution and adjoints remain unvalidated. |

The original performance result uses a 1600 by 100 grid, two modes, and five independent processes per variant.
Each process advances 4000 warmup steps and 4000 measured steps.
The measured speedup applies to this workload, device, implementation, and environment.

The user replaced floating-exponent powers in core kernels with explicit multiplication.
The isolated before/after comparisons both used Warp 1.17 and retained float64 arithmetic.
The original complete solver rerun also includes the later Warp 1.18 upgrade.
These experiments support different levels of attribution, as the performance chapter explains.

The later producer comparison used a `1500 x 300` grid and five modes on one RTX 6000 Ada GPU.
Numba FP64, Warp FP64, and Warp FP32 Hankel used the same float32 ID-based ionization draws.
Warp compute speedups were `1.758x` for FP64 and `1.956x` for FP32 Hankel.
The [experiment chapter](05-hankel-fp32-and-ionization-rng.md) records all field errors, input differences, timing scope, and unresolved observations.

The [complete study](07-inversion-full-workload.md) uses a later producer revision and covers all 17,720 requested steps.
Its original Numba reference took `12297.922 s`; fused Warp FP64 took `5570.933 s`; fusion with FP32 Hankel took `4801.397 s`.
Maximum field relative L2 errors were `3.933373e-10` for FP64 and `2.701057e-3` for FP32 Hankel.
The accuracy reference uses shared ionization draws, and the analysis includes 19 complete snapshots.
The complete-run result does not establish FP32 acceptance for other physical observables or configurations.
The [kernel-gradient milestone](08-individual-kernel-gradients.md) proceeds separately on `snidhan/dev-autodiff`, based on integration commit `edc52c4`.

## How to use the evidence

Source code defines current behavior.
Saved commands, hashes, outputs, and timing samples support measured claims.
Historical reports describe their recorded version and test selection.
Future research sections describe work that has not been implemented.

The [baseline change record](../baseline-changes/README.md) remains the implementation inventory and the scope record.
Its upstream reference commit is `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.
This guide connects that record, the original forward-port snapshot, and the later local measurements.
