# Performance measurement and arithmetic

This chapter gives repeatable steps to measure a ported solver against its baseline.
The numbered steps and the **Common errors** paragraphs apply to any port.
Paragraphs marked **FBPIC example** record measurements from this repository.
FBPIC examples are not requirements or performance expectations for other ports.

**FBPIC example:** The measurements come from the local development workspace on 2026-10-06.
The multiplication replacements in core particle kernels are included on `main`.
The `snidhan/dev-profiling` branch contains the performance scripts, benchmark plans, and detailed reports.
Script and report paths below refer to that branch. Generated result paths identify local artifacts excluded from Git.
The [FP32 Hankel experiment](05-hankel-fp32-and-ionization-rng.md) records later operator measurements and simulation comparisons on the FP32 Hankel experiment branch.
The [fusion chapter](06-threads-shared-memory-and-fusion.md) extends this record through October 10, 2026.
The [complete Inversion study](07-inversion-full-workload.md) records the final kernel captures and 17,720-step measurements.

## Inspect arithmetic before changing precision

1. Find translated expressions whose form differs from the baseline, such as integer powers.
2. Measure matched kernels before and after each arithmetic change.
3. Check kernel outputs after each arithmetic change.
4. Rule out arithmetic translation costs before you attribute a slowdown to precision.

Equivalent mathematical expressions can produce different GPU execution costs.
Arithmetic changes can also change rounding.

**Common errors:**

- A slow double-precision kernel does not prove that double precision causes the slowdown.
- A combined set of edits does not attribute a speedup to one replacement.
- A timing difference does not establish the generated instruction counts.
- An isolated expression probe gives no evidence of solver accuracy.
- A lower-precision operation adds conversion and temporary storage costs. Include them in complete solver timing.

**FBPIC example:** The port initially used floating-exponent `wp.pow` calls where baseline kernels used integer squares.
The user replaced these expressions with explicit multiplication, including `x * x` for squares.
Matched kernel measurements then showed substantial improvements while retaining float64 arithmetic.
The original slowdown therefore did not establish that float64 precision caused the implementation gap.
The measurements cover the combined user edits and do not isolate each replacement.
The `benchmarks/repro_pow_vs_mul.py` is an expression probe, without evidence for mixed-precision solver accuracy.

The later Hankel experiment uses float32 GEMM while the remaining solver retains double precision.
Its [development record](05-hankel-fp32-and-ionization-rng.md) includes complete-step timings and a 2048-step producer comparison.
That partial result remains distinct from the later [complete-run validation](07-inversion-full-workload.md).

## Fix the workload before comparing implementations

1. Fix the grid, physical configuration, dtypes, and step counts for every compared variant.
2. Run warmup steps until the solver reaches a representative state.
3. Start every variant from the same saved state. Check initial hashes and random states.
4. Time complete steps, including every cost that the application pays during a step.
5. Run several independent processes per variant. Change the variant order between repetition sets.

Include diagnostics and checkpoints when the target workload includes them.
Keep snapshot creation and output comparison outside timing.

**Common errors:**

- Independent initialization roundoff can look like a backend difference.
- Compilation and start-up transients distort a measurement without warmup.
- Work per step can change during a run although the configuration is fixed. Record the state of the measured interval.
- A fixed variant order can mix clock or thermal drift into the comparison.
- One process per variant does not quantify variation between processes.

**FBPIC example:** The representative workload uses a `1600 x 100` physical grid and two azimuthal modes.
Boundary buffers increase allocated fields to `1920 x 100` cells per mode.
Particles use float64, and fields use complex128.
The initial physical domain contains no particles.
Four thousand warmup steps populate the moving window before measurement.

Each complete solver run measures another 4000 steps through one synchronized `Simulation.step` call.
Injection, removal, sorting, allocation, transfers, transforms, Python dispatch, and graph capture remain inside timing.
Diagnostics and checkpoints remain disabled.
The particle population changes during propagation, although the workload configuration stays fixed.

Each variant uses five independent processes, with variant order changed between repetition sets.
All processes restore the same saved baseline initial fields, including boundary buffers, before warmup.
Initial field hashes, particle hashes, random states, and physical configurations must match.
The profiling branch report `benchmark-lwfa-warp-1.18.md` records the workload and checks.

## Isolate kernels with captured inputs

1. Save kernel inputs from a representative baseline invocation after warmup.
2. Give both implementations identical array contents, shapes, scalar values, and launch dimensions.
3. Compile each kernel and prepare launch arguments outside timing.
4. Restore every modified input before each launch. Synchronize the restoration before the CUDA start event.
5. Record CUDA events on the stream of the measured launch.

Collect many samples after several warmup launches.
Report the median and keep the raw samples.

**Common errors:**

- Overlapping or noncontiguous saved inputs might not reproduce the original launch. Reject them or record them.
- Without input restoration, later samples of an in-place kernel measure different work.
- Events on another stream measure the wrong interval.
- Input restoration changes cache state, and short intervals can include host submission gaps.
- Kernel timings do not replace complete solver measurements.

**FBPIC example:** Kernel inputs come from the baseline's first relevant invocation after 4000 warmup steps.
The saved particle arrays contain 2,648,667 particles.
The `benchmarks/capture_kernel_inputs.py` rejects overlapping or noncontiguous arguments.
The baseline uses its production default stream; Warp uses its shared runtime stream.
The `benchmarks/benchmark_kernels.py` collects 100 samples after ten warmup launches.
Input restoration, wrapper dispatch, compilation, and validation are excluded.

## Report before and after evidence

1. Record the device, clock settings, and library versions for each measurement.
2. Change one factor between compared states. If several factors change, state that the result does not separate them.
3. Rerun the baseline for each new comparison.
4. Compare the outputs of each timed kernel with the baseline outputs.
5. Define each ratio, including which time is the numerator.

Integer outputs require exact matches.
Floating outputs require a bound normalized by the output magnitude.

**Common errors:**

- A result on one GPU, workload, or launch size does not predict another configuration.
- A code edit combined with a library upgrade does not attribute the speedup to either change.
- Kernel output checks do not replace physical solver tests.
- Observed differences between runs do not define a new physical acceptance tolerance.

**FBPIC example:** All reported measurements use one NVIDIA RTX 6000 Ada GPU.
GPU clocks were not locked.
Both solvers retain cuFFT and cuBLAS through CuPy.

The following table contains median complete solver times in milliseconds per step.
Each entry uses five processes with 4000 warmup steps and 4000 measured steps.

| Implementation state | Numba baseline | Warp eager | Warp graphs |
| --- | ---: | ---: | ---: |
| Original port, Warp 1.17 | 14.675601 | 27.618265 | 27.516334 |
| User multiplication edits, Warp 1.18 | 14.657563 | 12.486426 | 12.371794 |

The original solver report, the profiling branch report `benchmark-lwfa-1600x100.md`, preserves the first measurements.
The latest eager solver is approximately 1.174 times faster than its fresh Numba baseline.
The latest graph solver is approximately 1.185 times faster.
This complete solver comparison combines the user edits with a Warp version upgrade.
It does not separate their contributions.

The isolated kernel rerun provides a narrower comparison: both kernel experiments used Warp 1.17.0.
The original momentum-push ratio was `8.822`, and the original cell-index ratio was `9.460`.
Each ratio divides Warp time by Numba time.
The `benchmarks/results/20261006T004434Z_kernel_benchmark_before_pow_removal_20261006T013746Z/comparison.json` records those results.

The following measurements use the user multiplication edits and a fresh Numba run.
Times are median microseconds per prepared launch.

| Kernel | Numba | Updated Warp | Warp / Numba |
| --- | ---: | ---: | ---: |
| `deposit_rho_gpu_linear` | 2674.208 | 2162.224 | 0.809 |
| `deposit_J_gpu_linear` | 4426.032 | 3461.024 | 0.782 |
| `gather_field_gpu_linear` | 1652.064 | 738.304 | 0.447 |
| `push_p_gpu` | 408.256 | 404.480 | 0.991 |
| `push_x_gpu` | 204.704 | 209.920 | 1.025 |
| `get_cell_idx_per_particle` | 83.968 | 83.264 | 0.992 |
| `write_sorting_buffer` | 15.168 | 14.176 | 0.935 |

The `benchmarks/results/20261006T004434Z_kernel_benchmark/comparison.json` records timing summaries, output comparisons, and source provenance.
Per-kernel metrics in the same result directory retain the raw samples.
All 24 saved output arrays passed comparison against the fresh baseline run.
Integer outputs and sorting copies matched exactly.
The maximum floating error, normalized by its corresponding output maximum, was `1.78e-15`.
The `benchmarks/analyze_kernels.py` applies a `1e-10` normalized maximum bound separately to each floating output.

The latest solver benchmark passed its selected preflight checks, but it did not rerun the complete test suite.
Its nonlinear field differences have the same order of magnitude as baseline repeat variation.

## Connect timings to source

1. Map each measured kernel to its baseline definition, its port definition, and its solver caller.
2. Derive definition and call-site lines from current source when you generate each report.
3. Label profiler symbols and register counts with the capture that produced them.

Line numbers written by hand become incorrect after source edits.
Profiler symbols from an earlier capture can name code that no longer exists.

**FBPIC example:** The port links below identify the modules on `main`, including the user's multiplication replacements.
Paths beginning with `baseline/` identify the original FBPIC checkout at commit `dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9`.
Function names identify the measured kernels within each module.

| Kernel | Baseline source | Warp source | Solver caller |
| --- | --- | --- | --- |
| `deposit_rho_gpu_linear` | `baseline/fbpic/particles/deposition/cuda_methods.py` | [Deposition](../../fbpic/particles/deposition/cuda_methods.py) | `Particles.deposit` |
| `deposit_J_gpu_linear` | `baseline/fbpic/particles/deposition/cuda_methods.py` | [Deposition](../../fbpic/particles/deposition/cuda_methods.py) | `Particles.deposit` |
| `gather_field_gpu_linear` | `baseline/fbpic/particles/gathering/cuda_methods.py` | [Gathering](../../fbpic/particles/gathering/cuda_methods.py) | `Particles.gather` |
| `push_p_gpu` | `baseline/fbpic/particles/push/cuda_methods.py` | [Pushing](../../fbpic/particles/push/cuda_methods.py) | `Particles.push_p` |
| `push_x_gpu` | `baseline/fbpic/particles/push/cuda_methods.py` | [Pushing](../../fbpic/particles/push/cuda_methods.py) | `Particles.push_x` |
| `get_cell_idx_per_particle` | `baseline/fbpic/particles/utilities/cuda_sorting.py` | [Sorting](../../fbpic/particles/utilities/cuda_sorting.py) | `Particles.sort_particles` |
| `write_sorting_buffer` | `baseline/fbpic/particles/utilities/cuda_sorting.py` | [Sorting](../../fbpic/particles/utilities/cuda_sorting.py) | `Particles.rearrange_particle_arrays` |

The caller methods are in `baseline/fbpic/particles/particles.py` and [Warp particles](../../fbpic/particles/particles.py).
The momentum algorithm uses `baseline/fbpic/particles/push/inline_functions.py` and the [Warp Vay helper](../../fbpic/particles/push/cuda_inline_functions.py).
The linked Warp helper includes the measured multiplication replacements.
The `benchmarks/results/20261006T004434Z_kernel_benchmark/source_mapping.json` records exact definition and call-site lines.
The analyzer derives these lines from current source using the `benchmarks/kernel_mapping.json`.
Historical profiler symbols and register counts in that manifest describe the earlier captures.

## Interpret profiles and CUDA graphs carefully

1. Check profiler warnings and the compatibility of the profiler with the driver before you use a capture.
2. Confirm that corresponding kernel counts match between variants.
3. Compute GPU busy time as the union of activity intervals across streams.
4. Repeat profiled runs before you report timing variability or profiler overhead.
5. Record the graph reuse fraction and include capture cost in solver timing.

**Common errors:**

- CPU API durations can overlap GPU activity and include waits. Do not add them to GPU durations or call them Python overhead.
- Register counts alone do not establish occupancy, spills, or the generated arithmetic cost.
- A capture with collection warnings can lack some event types, although kernel events are present.
- A graph with a low reuse fraction can give a small speedup over eager execution.

**FBPIC example:** The initial Nsight captures identified longer GPU execution in deposition, momentum pushing, and particle indexing.
Corresponding kernel counts matched, with 87,144 launches per variant.
The profiling branch report `fbpic-profiling.md` records the durations and collection limits.

Nsight Systems reported an unsupported driver interface and used CUDA 13.1 tracing libraries with a CUDA 13.2 driver interface.
The reports warned about potentially incomplete CUDA, NVTX, and cuBLAS collection.
NVTX marks the measured interval through NVIDIA Tools Extension annotations.
Expected kernel counts and API correlation identifiers were present, but cuBLAS CPU API events were absent.
The cuBLAS GPU kernels were present.
These observations support kernel comparisons without proving complete collection of every event type.

Graphs reduced the latest median Warp time by `0.92%` compared with eager execution.
Each measured graph run launched 20,000 graphs, including 12,002 new captures and 7,998 reused captures.
The reuse fraction was `39.99%`.
Moving-window geometry, scalar arguments, and particle-buffer changes can invalidate cached graphs.

## Keep provenance with each result

1. Keep the report, commands, source hashes, package versions, and numerical checks with each performance result.
2. Before you share a raw profiler report, check it for inherited environment metadata.

**FBPIC example:** The linked generated artifacts are local and excluded from Git.

## Extend the optimization sequence

1. Compare launch configurations while retaining the reference arithmetic and input state.
2. Test fusion opportunities that share inputs, geometry, or repeated launches.
3. Measure complete operations, including scratch, casts, reductions, and dispatch.
4. Repeat representative workload windows as particle populations and physical states change.
5. Rerun the complete application before reporting an end-to-end speedup.

The [fusion record](06-threads-shared-memory-and-fusion.md) includes rejected methods as well as selected changes.
Its two-pass result shows why fewer global atomics do not guarantee a faster workload.
Its small field-fusion gains show why one timing difference cannot establish a useful improvement.

The [Inversion record](07-inversion-full-workload.md) uses identical captured inputs for kernel comparisons.
Its full-run timer includes production diagnostics, setup, and cold compilation.
Kernel measurements and complete-run measurements therefore have different timing scopes.
Do not multiply isolated kernel speedups to predict the application speedup.
