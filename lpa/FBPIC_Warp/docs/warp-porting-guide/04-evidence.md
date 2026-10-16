# Evidence Index

This index records the original guide sources and subsequent experiments through the complete Inversion comparison on 2026-10-12.
Design decisions, implemented behavior, measured results, and future proposals have different evidence requirements.

The original `main` reference contains the forward port at commit `681c892` and the user's core arithmetic replacements.
The `snidhan/dev-profiling` branch contains performance scripts, benchmark plans, and detailed measurement reports.
The environment specification pins the CUDA 12 build of Warp 1.18.
Benchmark script and report paths identify files on the profiling branch. Generated result paths identify excluded local artifacts.
The [FP32 Hankel chapter](05-hankel-fp32-and-ionization-rng.md) records the later experiment on the FP32 Hankel experiment branch.
Its Numba counterpart uses the Numba ionization-RNG branch, branched from the original `dev` reference.

Before publication, seven existing GPU tests passed on the exact arithmetic changes under Warp 1.18.
These tests cover deposition, plasma waves with eager and graph execution, and stationary laser-antenna deposition.
This focused run does not replace a complete test-suite rerun or the recorded performance measurements.

## Retained measurements

The following generated results were present and checked when this guide was written.
They remain local artifacts excluded from Git.
The profiling branch reports preserve their main findings.
The [performance chapter](02-performance.md) includes the principal timing tables for readers without those local records.

| Measurement | Retained artifact | Evidence status |
| --- | --- | --- |
| Original seven-kernel timings | `benchmarks/results/20261006T004434Z_kernel_benchmark_before_pow_removal_20261006T013746Z/comparison.json`. | The earlier timings and numerical comparisons remain available. |
| Updated seven-kernel timings | `benchmarks/results/20261006T004434Z_kernel_benchmark/comparison.json`. | All 468 analysis checks passed across 24 complete output arrays. |
| Current kernel locations | `benchmarks/results/20261006T004434Z_kernel_benchmark/source_mapping.json`. | The artifact records definitions, caller locations, and source hashes. |
| Warp 1.18 complete solver | `benchmarks/results/20261006T020411Z_lwfa_1600x100_warp118/analysis_repeats/summary.json`. | Fifteen runs completed with zero reproducibility issues. |
| Current resource measurements | `benchmarks/results/20261006T020411Z_lwfa_1600x100_warp118/resource_summary.json`. | GPU telemetry and process host-memory summaries remain available. |

The original whole-step result directory, original Nsight directory, and cell-index sweep directory are absent from the current results tree.
The profiling branch reports `benchmark-lwfa-1600x100.md`, `fbpic-profiling.md`, and `cell-index-sweep.md` remain available.
Numbers from those reports are historical observations, rather than newly rechecked raw measurements.

The old and updated seven-kernel experiments both used Warp 1.17.0.
The later complete solver experiment used Warp 1.18.0 with the CUDA 12 build.
Kernel improvements and version-upgrade effects therefore require different comparisons.

## Code anchors for future work

| Topic | Code evidence |
| --- | --- |
| Forward-only Warp integration | [WarpKernel and CudaGraphSections](../../fbpic/utils/warp.py) disable backward generation and implement current graph ownership. |
| External transforms | [Fourier](../../fbpic/fields/spectral_transform/fourier.py) and [Hankel](../../fbpic/fields/spectral_transform/hankel.py) define layout and normalization. |
| State mutation and reordering | [Simulation](../../fbpic/main.py) and [Particles](../../fbpic/particles/particles.py) define execution order and buffer replacement. |
| Precision-sensitive setup | [PSATD coefficients](../../fbpic/fields/psatd_coefs.py) and [ionization coefficients](../../fbpic/particles/elementary_process/ionization/ionizer.py) contain cancellation and range hazards. |
| Random state | [Warp random helpers](../../fbpic/particles/elementary_process/warp_random.py) retain general sampling. [Ionization draws](../../fbpic/particles/elementary_process/ionization/random.py) define the ID-based float32 contract. |

These code anchors support future experiment design.
They do not establish gradient correctness or general mixed-precision accuracy.

## Subsequent Hankel and ionization evidence

The following records support the [development sequence](05-hankel-fp32-and-ionization-rng.md) from October 7 and 8, 2026.
The simulation artifacts reside in the FP32 Hankel experiment branch worktree. They remain excluded from Git.
The chapter retains their main measurements and links to the implementation and plotting scripts.

| Record | Location | Finding and limit |
| --- | --- | --- |
| Isolated FP32 Hankel study | Profiling worktree: `roofline/results/hankel_fp32_gpu1_20261007/report.md`. | At `Nr=300`, GEMM speedup was `21.500x`; conversion-inclusive forward speedup was `10.051x`. Simulation accuracy was untested. |
| Reconstructed simulation | `benchmarks/results/hankel_simulation_2048_gpu1_20261007/plots/comparison.json`. | Final FP64 charge error was `0.9885396`. Loading and initial species differed from the producer. |
| Original producer FP64 comparison | `benchmarks/results/producer_fp64_2048_gpu1_2223c2bf/comparison/summary.json`. | Original index-based draws gave final charge error `9.19335e-11`. Runs overlapped on two GPUs. |
| Intermediate RNG comparison | `benchmarks/results/ionization_rng_fp64_512_gpu1_20261008/comparison/summary.json`. | Recorded 29 targeted passing tests per backend. The final `0.8088%` reconstructed charge residual remained unexplained. |
| Final producer comparison | `benchmarks/results/producer_rng_2048_gpu1_2223c2bf_20261008/comparison/summary.json`. | Three variants completed serially on GPU 1. The comparison covers 2048 steps and one run per variant. |

Step-128 sorting snapshots retain identical random arrays but different draw assignments for 159,718 matched ions.
The chapter records the coordinate-matching procedure and the 13 different cell keys.
The intermediate CPU charge probe omitted a required forward transform. Its results are excluded from the evidence.

The final run manifests identify producer revision `2223c2bf447930befccf62ab2492197932bb964b`, the shared LASY hash, and actual numerical settings.
They also record source-file hashes, branch dirty state, precision buffers, ionization streams, and the physical GPU identity.
The benchmark instructions in `hankel-fp32/benchmarks/README.md#producer-rng-comparison` (local artifact, not in the repository) describe reproduction and offline plotting.

## Subsequent threads and fusion evidence

The [fusion chapter](06-threads-shared-memory-and-fusion.md) records October 9 and 10 experiments.
FP32 Hankel and ID-based RNG work preceded these experiments.
The reusable workflow order does not rewrite the historical development order.

| Record | Worktree and local artifact | Finding and limit |
| --- | --- | --- |
| Artificial roofline and forward contributions | `profiling/roofline/results/representative_gpu1_20261007/report.md`. | 76 Warp definitions and 119 isolated cases. Artificial forward Hankel GPU time was 42.42% of solve time. |
| 16-by-16 shared-memory patches | `profiling/roofline/results/forward_smem_16x16_256_20261009b/report.md`. | 256-thread SMEM measured `0.987x` against 32-thread deposition. Independent numerical verification passed 356 checks. |
| Deposition candidates | `deposition-fusion`, `deposition-two-pass`; profiling base `2653b790`. | Process-local adapters isolated five-mode fusion and two-pass accumulation. Existing solver files remained intact. |
| Producer deposition matrix | `profiling/roofline/results/deposition_workload_summary_20261010/report.md`. | FP64 fusion gave `1.16241x`; two-pass gave `0.97684x`. Comparisons covered 2048 steps. |
| Additional FP64 fusion matrix | `profiling/roofline/results/fusion_fp64_summary_20261010/report.md`. | Gather gave `1.070x` over deposition-fused Warp. Other isolated candidates had small gains in one process. |

Separate reports preserve run ordering, source archives, interval timings, telemetry, field differences, and final ion-state comparisons.
The selected integration excludes SMEM, two-pass deposition, and marginal additional fusions.
The historical `2.31986x` optimized-to-original compute ratio is a 2048-step result, separate from the later complete-run timing.

## Complete Inversion evidence

The [complete study chapter](07-inversion-full-workload.md) records measured source, exact configuration, timing scopes, accuracy norms, and reproduction commands.
Its root is `inversion-warp-setup/benchmarks/results/inversion_20261012T033535Z`, relative to the worktree parent.
The source worktree branch is the Inversion workload experiment branch, based on `0a717d062994751388f08b88b34051778005285b`, with archived uncommitted changes.
The later producer revision is `923aba09e52b71325be02d29aaabd47269601043`.
The original Numba revision remains `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.

| Retained record | Path below the study root | Evidence status |
| --- | --- | --- |
| Source identity | `source_inventory.json`, `study_paths.json`. | File hashes identify measured working files and separate original and matched-RNG Numba. |
| Kernel matrix | `kernel_runs/`, `kernel_report_summary.json`. | 48 workers completed; all 30,888 result records passed validation. |
| Complete primary runs | `full_runs/gpu0/`. | Four cases completed 17,720 steps with isolated compilation caches. |
| Full numerical comparison | `plots/full_gpu0_20261012T173113Z_legend_fix/full_summary.json`. | Seven fields at 19 complete snapshots; 31 partial snapshots excluded. |
| Final report and artifact index | `FINAL_REPORT.md`, `final_artifacts.json`. | Timings, hashes, plots, executed commands, and primary completion remain recorded. |
| Simplified figures | `plots/three_case_comparison/`. | Original Numba is the timing baseline; shared-RNG Numba is the exact zero-error accuracy reference. |

The selected FP64 complete run achieved `2.2075x`, while FP32 Hankel achieved `2.5613x` against original Numba.
FP32 maximum relative L2 field error was `0.2701%`, with 53 changed final ionization levels and electron-count differences of `+5` and `-10`.
These results apply to this single-GPU setup and one complete measurement per configuration.
They do not establish the producer's eight-rank MPI performance or a general FP32 acceptance threshold.

The user released GPU 1 during the complete-run study.
The final execution policy requires four complete primary cases on GPU 0, while preserving supplementary GPU 1 cases and the canceled run's partial outputs.
The final artifact index records completion despite the legacy controller's intentional-cancellation failure status.

## Kernel-gradient evidence status

Approved integration commit `edc52c4` contains the selected fusion path and optional FP32 Hankel, and local `main` includes that commit.
Integration checks passed 101 GPU tests and 39 CPU tests, with 14 skipped CPU cases.
The GPU checks comprise 53 precision/RNG tests, 17 existing coupled-solver tests, and 31 new fusion tests.
The current openPMD reader required `OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0` for constant-record extents; dependencies remained unchanged.
Numerical modules match the measured source, apart from removing an extra fused-deposition EOF blank line.

The [individual-kernel chapter](08-individual-kernel-gradients.md) records validated opt-in FP64 eager adapters and their verification contract.
The separate `snidhan/dev-autodiff` branch starts from the cleaned baseline `snidhan/dev-main`.
Checks include primal parity, directional derivatives, adjoint identities, composition, and explicit gradient-policy rejection.
Its operator derivatives remain separate from production `Simulation.step`, external FFT/GEMM derivatives, stochastic events, and complete-trajectory gradients.
Set `FBPIC_AUTODIFF_REPORT_DIR` to retain per-case JSON residuals, fixture hashes, operator identity, and continuous-input selections.
Do not infer differentiability from Warp kernel generation, forward parity, or CUDA graph replay.
