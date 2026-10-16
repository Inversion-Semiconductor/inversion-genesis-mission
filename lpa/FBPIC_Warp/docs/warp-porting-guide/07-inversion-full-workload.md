# Complete Inversion workload comparison

The complete comparison finished on October 12, 2026, after the earlier partial-run experiments.
Fused Warp FP64 achieved `2.2075x` end-to-end speedup against original Numba.
Fusion with optional FP32 Hankel achieved `2.5613x`, with maximum field relative L2 error `0.2701%`.
That precision change also changed some ionization levels and electron counts.
Report those errors with the speedup; the experiment does not define an application-specific accuracy tolerance.

## Freeze the actual producer and reference

The physical setup comes from Inversion Semiconductor's `inversion-genesis-mission` repository.
The selected script is `lpa/Simulation_FBPIC/runs/initial_sample/ionization_injection_runscript_00.py`.
The measured local producer revision is `923aba09e52b71325be02d29aaabd47269601043`.
This later revision differs from the October 7 and 8 producer revision recorded in [chapter 05](05-hankel-fp32-and-ionization-rng.md).

The original solver reference is Numba FBPIC `0.27.1`, revision `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.
Its original RNG remains unchanged.
The numerical accuracy reference uses a separate Numba worktree with shared ID-based draws and unchanged numerical kernels.
Measured Warp source resides in the Inversion workload experiment branch, based on `0a717d062994751388f08b88b34051778005285b`, with recorded uncommitted changes.
Branch names and HEAD commits alone do not identify those measured working files; the source inventory records their SHA-256 hashes.

| Setting | Measured value |
| --- | --- |
| Physical grid | 1500 axial by 300 radial cells. |
| Allocated local grid | 1852 axial by 300 radial cells, including guards and other buffers. |
| Azimuthal modes | Five. |
| Boost factor | 1.5. |
| Spectral order | 32. |
| Ion loading | 2 by 2 by 15 particles per cell for each species. |
| Boosted timestep | `4.075316647058382e-16 s`. |
| Complete interaction | 17,720 steps. |
| Device | NVIDIA RTX 6000 Ada. |
| Warp | `1.18.0+cu12`. |
| Persistent particles and fields | Float64 particles and complex128 fields. |

All four cases preserve physical settings, numeric laser datasets, diagnostic settings, and matching written grid coverage.
Output directory names differ.
The producer's eight-rank MPI launch was adapted to one GPU per process.
This study therefore measures the producer's physical setup on one GPU; it does not measure its eight-rank production launch.

## Match changing workload states for kernel measurements

1. Advance matched-RNG Numba for 2048 steps and capture genuine operation inputs.
2. Preserve species, PIC phase, mode coverage, guard cells, scalar arguments, and array layouts.
3. Replay every variant on the same captured input state and restore mutated inputs before timing.
4. Use ten warmup calls and 100 CUDA-event samples per captured operation.
5. Repeat each process three times and retain all timings, validations, and provenance.

The reference harvest contains 918 genuine operation captures.
Capture steps are `512`, `640`, `767`, `1024`, `1152`, `1279`, `1792`, `1920`, and `2047`.
These samples cover three workload windows rather than every iteration in each window.
Sampling several states preserves parity while exposing changing particle populations and input distributions.
It does not establish kernel behavior throughout the later 17,720-step trajectory.

The eight-variant matrix ran three repetitions on each of two independent GPU queues.
All 48 workers succeeded, and all `30,888` result records passed validation.
Kernel timing excludes compilation, allocation, restoration, validation, and capture transfers.
Replay uses prepared eager launches; the complete Warp simulations retain CUDA graphs.
No Nsight profiling was used for these measurements.

| FP64 operation | Median paired speedup against original Numba |
| --- | ---: |
| Charge deposition | 9.134x |
| Current deposition | 5.977x |
| Field gather | 2.750x |
| Momentum push | 1.005x |
| Position push | 0.986x |
| Cell index | 1.005x |
| Hankel transform | 1.003x |

These ratios use median process timings for each case, then the median paired speedup across all nine captured steps.
Complete deposition and gather operations include their required mode passes.
The selected deposition fusion adds `2.440x` for charge and `1.433x` for current against 32-thread unfused Warp.
The arithmetic and eight-to-32-thread ablations remain separate, as [chapter 06](06-threads-shared-memory-and-fusion.md) records.

FP32 Hankel achieved `10.484x` conversion-inclusive speedup and `19.821x` GEMM-only speedup against Warp FP64.
FP32 overview particle rows reuse measured FP64 timings because those operations are unchanged.
The FP32 Hankel rows contain actual FP32 measurements.
Kernel replay maximum normalized errors were `1.253753e-12` for FP64 and `1.855088e-6` for FP32 Hankel.
These replay errors are distinct from accumulated simulation field errors.

## Measure the complete application

All four primary cases completed 17,720 steps on GPU 0.
Each case used a fresh process and isolated cold compilation caches.
The primary timer includes heavy imports, laser preparation, setup, compilation, simulation, production diagnostics, and the final diagnostic flush.
It excludes source hashing, output analysis, and final ion-state archives.
Parent launch-to-exit times include those verification operations and remain available in the run manifests.

| Complete configuration | End-to-end s | End-to-end min | Simulation s | Speedup against original Numba |
| --- | ---: | ---: | ---: | ---: |
| Numba FP64, original RNG | 12297.922 | 204.965 | 12282.502 | 1.0000x |
| Numba FP64, shared RNG | 12315.132 | 205.252 | 12305.067 | 0.9986x |
| Warp FP64, deposition and gather fusion | 5570.933 | 92.849 | 5560.330 | 2.2075x |
| Warp fusion, FP32 Hankel | 4801.397 | 80.023 | 4789.874 | 2.5613x |

Against shared-RNG Numba, the Warp speedups are `2.2106x` and `2.5649x`.
FP32 Hankel adds approximately `1.160x` end-to-end speedup over fused Warp FP64.
Each configuration has one complete measurement, so no complete-run confidence interval is available.
Do not combine the old partial-run compute denominator with this end-to-end timer.

The selected Warp blocks contain 32 deposition threads, 128 gather threads, and 256 push threads.
Transform conversion blocks contain 2 by 16 threads.
cuBLAS and cuFFT manage their internal launches.
FP32 changes Hankel matrices, GEMM operands, and conversion scratch; particles, stored fields, and FFTs retain double precision.

## Report field error and stochastic changes

The accuracy reference is shared-RNG FP64 Numba.
Matched cases use persistent ion IDs, Philox seed zero, helium stream zero, and nitrogen stream one.
Identical draws avoid sorting-dependent RNG assignment differences.
They do not guarantee identical event decisions after numerical field trajectories change.
Original-RNG Numba remains a timing reference and separate stochastic control.

Relative L2 error divides the volume-weighted field-difference norm by the instantaneous reference norm.
Cylindrical cell volumes weight the spatial norm.
Packed angular-mode weights are one for mode zero and one-half for each higher-mode real or imaginary plane.
Peak-normalized L2 error divides by the largest complete-snapshot reference norm for that field.
Near-zero instantaneous norms can produce large relative errors despite small absolute differences.

| Field | Maximum relative L2, Warp FP64 | Maximum relative L2, FP32 Hankel |
| --- | ---: | ---: |
| Er | `9.491224e-13` | `3.971968e-5` |
| Eθ | `4.283856e-13` | `4.315712e-5` |
| Ez | `7.063970e-11` | `1.905418e-4` |
| Br | `4.282351e-13` | `4.268531e-5` |
| Bθ | `7.411785e-13` | `4.007146e-5` |
| Bz | `3.626645e-12` | `3.463779e-4` |
| rho | `3.933373e-10` | `2.701057e-3` |

Maximum peak-normalized L2 errors are `2.421820e-11` for FP64 and `8.930238e-4` for FP32 Hankel.
FP64 preserves final counts, ion ID sets, and every final ionization level.
FP32 changes 20 helium and 33 nitrogen levels among `13,761,900` common ion IDs per species.
Its final electron counts differ by `+5` helium electrons and `-10` nitrogen electrons.
Matching ID sets does not establish identical electron birth times or trajectories.

Each run produced 50 lab snapshots, but only 19 snapshots contain all 641 axial columns.
The other 31 snapshots contain 640 columns and are excluded from the field comparison.
The complete snapshots span `0.371217` to `17.818420 ps` in the lab frame.
The retained coverage table verifies identical missing-column coverage across all four cases.
This measurement covers the complete simulation duration through available complete snapshots, rather than every timestep or all 50 snapshots.

The requested three-case accuracy figure uses shared-RNG Numba as its exact zero-error curve.
The Warp FP64 and FP32 curves compare against that same reference.
Its symlog axis preserves exact zeros and uses a linear region below `1e-14`.
The corresponding timing figure uses original Numba as its baseline.
The figure README states this reference distinction explicitly.

## Locate and reproduce the evidence

The measured study root is `inversion-warp-setup/benchmarks/results/inversion_20261012T033535Z`, relative to the worktree parent (local artifact, not in the repository).
Generated arrays, diagnostics, traces, caches, and plots remain excluded from Git.
The final report `FINAL_REPORT.md` in the study root preserves detailed measurements.
The source inventory `source_inventory.json` in the study root records all measured file hashes.
The artifact index `final_artifacts.json` in the study root identifies manifests, plots, completion status, and executed commands.

| Evidence | Location below the study root |
| --- | --- |
| Input source paths | `study_paths.json`. |
| Kernel timings and checks | `kernel_runs/` and `kernel_report_summary.json`. |
| Four complete primary runs | `full_runs/gpu0/`. |
| Kernel plots and CSV files | `plots/kernels_before_full_20261012T155338Z/`. |
| Complete-run summary, errors, and coverage | `plots/full_gpu0_20261012T173113Z_legend_fix/full_summary.json`. |
| Simplified three-case figures | `plots/three_case_comparison/`. |

Representative measured SHA-256 hashes identify the exact implementations below.
Integration cleanup can change hashes without retroactively changing the measured source.
Approved integration commit `edc52c4` preserves the measured numerical modules and removes an extra fused-deposition EOF blank line.
Local `main` now contains that integration; complete-run timing still identifies the original measured source inventory.

| Measured file | SHA-256 |
| --- | --- |
| Fused deposition | `e4aec84cbfeaf4895708d327c04ca29bf31df89eaa8a788113650a1196b10511`. |
| Fused gather | `f49e8b36bd5adfa97b6e3f6fbd3b343394eb182e9684cb80204024ae050e8ed0`. |
| Hankel transform | `7045dfc6547256a12a9e43c87bd7458395597f0f374b5b43687f69a48e412258`. |
| Numeric laser envelope | `b5372a6e02231362b77315d5578c62f9474f5d5149746d90907d937377903150`. |

Recreate the simplified figures from the retained summary in the `inversion-warp-setup` worktree.

```bash
python benchmarks/inversion_plot_three_cases.py \
  --summary benchmarks/results/inversion_20261012T033535Z/plots/full_gpu0_20261012T173113Z_legend_fix/full_summary.json \
  --output benchmarks/results/inversion_20261012T033535Z/plots/new_three_case_comparison
```

Use the retained study manifest to run a phase through `benchmarks/inversion_study.py --phase capture`, `kernels`, `full`, or `plots`.
Pass `--results` with the study directory and `--gpus 0` for new work.
The launcher skips verified completed jobs, so a completed study directory does not request fresh measurements.
For a fresh complete Warp FP32 run, use the worker command below with a new output directory.

```bash
python benchmarks/inversion_full_run.py \
  --producer ../Simulation_FBPIC \
  --fbpic-root . --engine warp --rng shared --hankel-precision float32 \
  --gpu 0 --output benchmarks/results/new_full_warp_fp32
```

Omit `--steps` to preserve the producer's complete interaction duration.
Use `--hankel-precision float64` for the FP64 Warp case.
The original and matched Numba worker commands require their recorded `--fbpic-root` paths and respective `--rng` settings.
Do not edit measured source while GPU workers run.

The original plan assigned four full variants to each GPU.
The user later released GPU 1, and the completed four-run GPU 0 matrix became the required primary comparison.
The interrupted GPU 1 original-Numba run remains preserved with its partial outputs.
Completed supplementary GPU 1 cases also remain preserved.
The final artifact index distinguishes primary completion from the legacy controller's intentional-cancellation status.

Raw manifests preserve device metadata even when simplified figure labels omit device names.
The study does not establish MPI parity, statistical ensemble agreement, beam-quality acceptance, or derivative correctness.
